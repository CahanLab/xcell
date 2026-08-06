"""PySingleCellNet cell-type classification, as an optional xcell backend.

`PySingleCellNet <https://github.com/CahanLab/PySingleCellNet>`_ trains a
random forest over *top-scoring gene pairs* — for each cell it asks, for a few
hundred pre-selected pairs, "is gene A above gene B in this cell?" and votes a
cell type from the answers. The per-class vote proportions come back as a
cells x classes score matrix; the argmax is the call.

Two consequences shape this adapter:

**The transform is per-cell rank-based.** Comparing two genes *within* a cell
is unaffected by any monotone per-cell rescaling, so raw counts, CPM and
log-normalized data all give identical pair features. Per-*gene* rescaling is
a different matter: z-scored input reorders genes within a cell and silently
produces garbage. :mod:`xcell.layer_scale` is what the UI uses to warn about
that before a run rather than after.

**Only the classifier's own genes matter.** PySCN reindexes the query onto
``clf['tpGeneArray']`` with ``fill_value=0``, so genes the query lacks become
zeros with no warning — a query that shares few genes with the training data
still returns confident-looking scores. Two guards follow from that:
:func:`assess_gene_overlap` reports coverage *before* a run, and
:func:`build_query_adata` narrows the query to exactly those genes up front,
which also keeps the dense intermediate PySCN builds proportional to the
classifier (a few hundred columns) rather than to the dataset (tens of
thousands).

The package is an optional dependency. Nothing here imports it at module
scope; import failures surface as a :class:`ValueError` carrying install
instructions, which routes turn into a 400.
"""
from __future__ import annotations

import pickle
from pathlib import Path
from typing import Any, Callable

import numpy as np
import pandas as pd

INSTALL_HINT = (
    "PySingleCellNet is not installed. Install it with "
    "`pip install pySingleCellNet` (or `pixi add --pypi pySingleCellNet`), "
    "then restart the xcell backend."
)

#: Keys a dict must have to be a PySCN classifier (see ``tl.train_classifier``).
REQUIRED_CLF_KEYS = ('tpGeneArray', 'topPairs', 'classifier')

#: The synthetic class PySCN trains on random profiles. Not a cell type.
RAND_CLASS = 'rand'

#: Below this many usable genes, gene-pair selection has nothing to work with.
MIN_TRAINING_GENES = 20

# Coverage of the classifier's gene set below which results stop being
# trustworthy. Missing genes are silently zero-filled by PySCN, and every
# absent gene corrupts every pair feature it participates in.
_OVERLAP_WARN = 0.95
_OVERLAP_ERROR = 0.60


def _import_module():
    """Return the pySingleCellNet module, or None if it cannot be imported.

    Split out so tests can simulate the package being absent.
    """
    try:
        import pySingleCellNet  # noqa: PLC0415

        return pySingleCellNet
    except Exception:
        return None


def import_pyscn():
    """Return the pySingleCellNet module or raise with install instructions."""
    mod = _import_module()
    if mod is None:
        raise ValueError(INSTALL_HINT)
    return mod


def underscore_gene_names(names) -> list[str]:
    """Gene symbols PySCN's pair encoding cannot represent.

    A top-scoring pair is stored as the string ``"geneA_geneB"`` and decoded
    with ``split("_")``. A symbol containing an underscore therefore decodes
    into fragments that are not genes, and the transform dies inside pandas
    with a ``KeyError`` listing nonsense. Callers exclude these genes up
    front, or refuse the run and say why.
    """
    return [str(n) for n in names if '_' in str(n)]


def _version(mod) -> str:
    """Installed PySCN version.

    The package declares ``__version__`` in ``__all__`` but does not bind it
    at import time, so the attribute is usually absent; the distribution
    metadata is the reliable source.
    """
    v = getattr(mod, '__version__', None)
    if v:
        return str(v)
    try:
        from importlib.metadata import version  # noqa: PLC0415

        return version('pySingleCellNet')
    except Exception:
        return 'unknown'


def availability() -> dict[str, Any]:
    """Whether PySCN can be used right now, and why not if it can't.

    Cheap enough to call on every modal open; the UI uses it to disable the
    feature with an explanation rather than failing at run time.
    """
    try:
        import pySingleCellNet  # noqa: PLC0415

        return {
            'available': True,
            'version': _version(pySingleCellNet),
            'error': None,
            'install_hint': INSTALL_HINT,
        }
    except Exception as e:
        return {
            'available': False,
            'version': None,
            'error': f'{type(e).__name__}: {e}',
            'install_hint': INSTALL_HINT,
        }


# --------------------------------------------------------------------------
# Classifier loading and introspection
# --------------------------------------------------------------------------

def describe_classifier(clf: dict) -> dict[str, Any]:
    """Summarize a trained classifier for display before it is run."""
    est = clf.get('classifier')
    classes = [str(c) for c in getattr(est, 'classes_', [])]
    return {
        'classes': classes,
        # 'rand' exists so the forest can say "this looks like noise"; it is
        # never a biological answer, so the UI lists the real types separately.
        'cell_type_classes': [c for c in classes if c != RAND_CLASS],
        'n_classes': len(classes),
        'n_genes': int(len(clf.get('tpGeneArray', []))),
        'n_gene_pairs': int(len(clf.get('topPairs', []))),
        'n_trees': int(getattr(est, 'n_estimators', 0) or 0),
        'train_params': {k: _jsonable(v) for k, v in (clf.get('argList') or {}).items()},
        'genes': [str(g) for g in clf.get('tpGeneArray', [])],
    }


def _jsonable(v: Any) -> Any:
    if isinstance(v, (np.integer,)):
        return int(v)
    if isinstance(v, (np.floating,)):
        return float(v)
    if isinstance(v, np.ndarray):
        return v.tolist()
    return v


def load_classifier(path: str) -> tuple[dict, dict[str, Any]]:
    """Unpickle a trained classifier and validate its shape.

    Args:
        path: Path to a pickle written from ``tl.train_classifier``'s return
            value (xcell's Train tab writes these, as does a notebook doing
            ``pickle.dump(clf, fh)``).

    Returns:
        ``(clf, metadata)`` where metadata is :func:`describe_classifier`'s
        output plus the resolved path.

    Raises:
        ValueError: The file is missing, unreadable, or is not a classifier.
    """
    p = Path(path).expanduser()
    if not p.exists():
        raise ValueError(f"Classifier file not found: {p}")
    if not p.is_file():
        raise ValueError(f"Not a file: {p}")

    try:
        with open(p, 'rb') as fh:
            clf = pickle.load(fh)
    except ModuleNotFoundError as e:
        # Pickles of sklearn estimators need sklearn (and sometimes PySCN)
        # importable at load time.
        raise ValueError(
            f"Could not unpickle {p.name}: a module it references is not "
            f"installed ({e}). {INSTALL_HINT}"
        )
    except Exception as e:
        raise ValueError(f"Could not read {p.name} as a pickle: {e}")

    if not isinstance(clf, dict):
        raise ValueError(
            f"{p.name} contains a {type(clf).__name__}, not a PySingleCellNet "
            "classifier dict."
        )
    missing = [k for k in REQUIRED_CLF_KEYS if k not in clf]
    if missing:
        raise ValueError(
            f"{p.name} is not a PySingleCellNet classifier — missing "
            f"{', '.join(missing)}. Expected the dict returned by "
            "pySingleCellNet.tl.train_classifier."
        )
    if not hasattr(clf.get('classifier'), 'predict_proba'):
        raise ValueError(
            f"{p.name}: the 'classifier' entry is not a fitted estimator."
        )

    # Catch this here rather than letting it explode inside pandas during the
    # first prediction — such a classifier can never be applied.
    bad = underscore_gene_names(clf.get('tpGeneArray', []))
    if bad:
        raise ValueError(
            f"{p.name} was trained on {len(bad)} gene symbol(s) containing an "
            f"underscore ({', '.join(bad[:5])}"
            f"{'…' if len(bad) > 5 else ''}). PySingleCellNet encodes gene "
            "pairs as 'geneA_geneB' and splits on '_', so these cannot be "
            "decoded and the classifier cannot be applied. It needs to be "
            "retrained on renamed genes."
        )

    meta = describe_classifier(clf)
    meta['path'] = str(p)
    meta['file_size_mb'] = round(p.stat().st_size / 1e6, 2)
    return clf, meta


def classifier_colors_hex(clf: dict) -> dict[str, str]:
    """Per-class colors as ``#rrggbb``.

    PySCN stores whatever ``get_unique_colors`` produced — usually RGB float
    triples. xcell's category coloring reads ``uns['<column>_colors']`` as hex
    strings, so training colors carry into the plot for free once converted.
    """
    out: dict[str, str] = {}
    for name, c in (clf.get('ctColors') or {}).items():
        out[str(name)] = _to_hex(c)
    return out


def _to_hex(c: Any) -> str:
    if isinstance(c, str):
        return c if c.startswith('#') else f'#{c}'
    try:
        rgb = np.asarray(c, dtype=float).ravel()[:3]
        if rgb.size < 3:
            return '#808080'
        if rgb.max() <= 1.0:
            rgb = rgb * 255.0
        r, g, b = (int(round(min(255, max(0, v)))) for v in rgb)
        return f'#{r:02x}{g:02x}{b:02x}'
    except Exception:
        return '#808080'


# --------------------------------------------------------------------------
# Gene overlap
# --------------------------------------------------------------------------

def assess_gene_overlap(var_names, clf: dict) -> dict[str, Any]:
    """How much of the classifier's gene set the query actually has.

    PySCN zero-fills genes it cannot find, so this is the difference between
    a real classification and a confident-looking artifact. Also reports how
    many more genes would match if case were ignored, because human/mouse
    symbol conventions (``ACTB`` vs ``Actb``) are the usual cause of a total
    miss.

    Returns:
        Dict with ``n_required``, ``n_found``, ``frac_found``, ``missing``
        (capped sample), ``n_found_case_insensitive``, ``case_mismatch_only``,
        and a ``severity`` of 'ok' | 'warn' | 'error'.
    """
    required = [str(g) for g in clf.get('tpGeneArray', [])]
    have = {str(g) for g in var_names}
    have_lower = {str(g).lower() for g in var_names}

    found = [g for g in required if g in have]
    missing = [g for g in required if g not in have]
    found_ci = [g for g in required if g.lower() in have_lower]

    n_req = len(required)
    frac = (len(found) / n_req) if n_req else 0.0
    frac_ci = (len(found_ci) / n_req) if n_req else 0.0

    if frac >= _OVERLAP_WARN:
        severity = 'ok'
    elif frac >= _OVERLAP_ERROR:
        severity = 'warn'
    else:
        severity = 'error'

    return {
        'n_required': n_req,
        'n_found': len(found),
        'frac_found': frac,
        # Capped: a classifier can require hundreds of genes and the UI only
        # needs enough to recognize the pattern.
        'missing': missing[:50],
        'n_missing': len(missing),
        'n_found_case_insensitive': len(found_ci),
        'frac_found_case_insensitive': frac_ci,
        # True when matching case-insensitively would recover genes that are
        # otherwise missing — i.e. the symbols are the same, the casing isn't.
        'case_mismatch_only': bool(len(found_ci) > len(found)),
        'severity': severity,
    }


# --------------------------------------------------------------------------
# Query construction
# --------------------------------------------------------------------------

def build_query_adata(
    adata,
    clf: dict,
    *,
    layer: str | None = None,
    case_insensitive: bool = False,
):
    """Build a minimal AnnData holding exactly the classifier's genes, in order.

    Doing the alignment here rather than leaving it to PySCN's internal
    ``reindex`` has two effects. The dense matrix PySCN materializes becomes
    ``n_cells x n_classifier_genes`` instead of ``n_cells x n_dataset_genes``,
    which is the difference between megabytes and tens of gigabytes on a large
    query. And genes the dataset lacks become explicit zero columns here,
    where :func:`assess_gene_overlap` has already reported them, instead of
    appearing silently inside the prediction.

    Args:
        adata: The query dataset.
        clf: A loaded classifier dict.
        layer: Read from ``adata.layers[layer]``; None reads ``adata.X``.
        case_insensitive: Match gene symbols ignoring case. Off by default —
            it can create false matches — but the fix for a pure
            ``ACTB``/``Actb`` mismatch.

    Returns:
        AnnData with sparse CSR ``.X``, ``var_names`` equal to
        ``clf['tpGeneArray']`` in order, and the query's ``obs_names``.

    Raises:
        ValueError: The layer is missing, or no classifier gene is present.
    """
    import anndata
    from scipy.sparse import csr_matrix, issparse

    required = [str(g) for g in clf.get('tpGeneArray', [])]
    if not required:
        raise ValueError("Classifier declares no genes (empty tpGeneArray).")

    if layer in (None, '', 'X'):
        source = adata.X
    else:
        if layer not in adata.layers:
            raise ValueError(
                f"Layer '{layer}' not found. Available: "
                f"['X'] + {sorted(adata.layers.keys())}"
            )
        source = adata.layers[layer]

    # First occurrence wins for duplicated symbols — matching pandas' own
    # behaviour on a non-unique index, without the exception.
    position: dict[str, int] = {}
    for i, name in enumerate(adata.var_names):
        position.setdefault(str(name), i)
    if case_insensitive:
        lowered: dict[str, int] = {}
        for i, name in enumerate(adata.var_names):
            lowered.setdefault(str(name).lower(), i)

    col_for: list[int | None] = []
    for g in required:
        idx = position.get(g)
        if idx is None and case_insensitive:
            idx = lowered.get(g.lower())
        col_for.append(idx)

    n_found = sum(1 for c in col_for if c is not None)
    if n_found == 0:
        raise ValueError(
            f"No overlap between this dataset's genes and the classifier's "
            f"{len(required)} genes. Check that both use the same symbol "
            "convention (e.g. mouse 'Actb' vs human 'ACTB')."
        )

    source = source.tocsc() if issparse(source) else np.asarray(source)
    n_obs = adata.n_obs
    columns = []
    for idx in col_for:
        if idx is None:
            columns.append(csr_matrix((n_obs, 1), dtype=np.float32))
        elif issparse(source):
            columns.append(csr_matrix(source[:, idx], dtype=np.float32))
        else:
            columns.append(csr_matrix(source[:, [idx]].astype(np.float32)))

    from scipy.sparse import hstack

    X = csr_matrix(hstack(columns, format='csr'), dtype=np.float32)

    q = anndata.AnnData(
        X=X,
        obs=pd.DataFrame(index=pd.Index([str(o) for o in adata.obs_names])),
        var=pd.DataFrame(index=pd.Index(required)),
    )
    return q


# --------------------------------------------------------------------------
# Prediction
# --------------------------------------------------------------------------

def classify_scores(
    query,
    clf: dict,
    *,
    chunk_size: int = 20_000,
    progress: Callable[[float, str | None], None] | None = None,
) -> pd.DataFrame:
    """Run the classifier and return the cells x classes score matrix.

    Chunked over cells because PySCN densifies its input: with the query
    already narrowed to the classifier's genes, one chunk costs
    ``chunk_size x n_genes`` floats regardless of dataset size.

    ``nrand`` is fixed at 0. A non-zero value makes PySCN append synthetic
    random profiles to the result, which no longer aligns row-for-row with the
    cells — it is a diagnostic for training, not for annotating a query.

    Args:
        query: Output of :func:`build_query_adata`.
        clf: A loaded classifier dict.
        chunk_size: Cells per prediction call.
        progress: Optional ``(fraction, message)`` callback.

    Returns:
        DataFrame indexed by cell name, one column per class.
    """
    pyscn_mod = import_pyscn()

    n = query.n_obs
    if n == 0:
        raise ValueError("Query has no cells.")

    frames: list[pd.DataFrame] = []
    for start in range(0, n, max(1, chunk_size)):
        stop = min(n, start + chunk_size)
        block = query[start:stop].copy()
        pyscn_mod.tl.classify_anndata(block, clf, nrand=0)
        scores = block.obsm['SCN_score']
        if not isinstance(scores, pd.DataFrame):
            scores = pd.DataFrame(np.asarray(scores), index=block.obs_names)
        frames.append(scores)
        if progress is not None:
            progress(stop / n, f'Classified {stop:,} / {n:,} cells')

    out = pd.concat(frames, axis=0)
    out.index = pd.Index([str(o) for o in query.obs_names])
    return out


# --------------------------------------------------------------------------
# Post-processing
# --------------------------------------------------------------------------

def summarize_scores(scores: pd.DataFrame) -> dict[str, Any]:
    """Argmax class per cell plus the winning score (the call's confidence)."""
    arr = scores.to_numpy(dtype=np.float64, copy=False)
    cols = [str(c) for c in scores.columns]
    idx = arr.argmax(axis=1)
    return {
        'labels': [cols[i] for i in idx],
        'confidence': arr[np.arange(arr.shape[0]), idx].tolist(),
        'classes': cols,
    }


def compute_thresholds(
    scores: pd.DataFrame,
    labels: list[str],
    quantile: float = 0.05,
) -> dict[str, float]:
    """Per-class score thresholds, mirroring PySCN's ``tl.comp_ct_thresh``.

    For each class, the given quantile of the scores of the cells that class
    won. A class no cell was assigned gets 0.0 — it cannot then exclude
    anything, which is the conservative choice. ``rand`` is excluded, as in
    PySCN: it is not a cell type a query cell should be "above threshold" for.
    """
    arr = np.asarray(labels)
    out: dict[str, float] = {}
    for cls in scores.columns:
        name = str(cls)
        if name == RAND_CLASS:
            continue
        picked = scores.loc[arr == name, cls].to_numpy(dtype=np.float64)
        out[name] = float(np.quantile(picked, quantile)) if picked.size else 0.0
    return out


def derive_class_types(
    scores: pd.DataFrame,
    labels: list[str],
    thresholds: dict[str, float],
) -> list[str]:
    """Bucket each cell by how decisive its classification was.

    Follows the rules of PySCN's ``tl.categorize_classification`` for the
    cases that need no extra information:

    * exactly one cell type above threshold -> ``Singular``
    * none above threshold -> ``None``
    * argmax is the synthetic random class -> ``Rand``

    PySCN splits the "more than one above threshold" case into
    ``Intermediate`` and ``Hybrid`` using a graph of cell-type relatedness
    (from PAGA on the training data). xcell has no such graph, so those cells
    are reported as ``Ambiguous`` rather than guessed at — the distinction is
    biological, and inventing it here would be worse than declining to.
    """
    cols = [str(c) for c in scores.columns]
    keep = [c for c in cols if c != RAND_CLASS]
    sub = scores[keep].to_numpy(dtype=np.float64, copy=False)
    thr = np.array([thresholds.get(c, 0.0) for c in keep], dtype=np.float64)
    above = (sub > thr[None, :]).sum(axis=1)

    out: list[str] = []
    for i, label in enumerate(labels):
        if label == RAND_CLASS:
            out.append('Rand')
        elif above[i] == 1:
            out.append('Singular')
        elif above[i] == 0:
            out.append('None')
        else:
            out.append('Ambiguous')
    return out


#: Display order and colors for :func:`derive_class_types`' output. Mirrors
#: PySCN's SCN_CATEGORY_COLOR_DICT for the categories they share.
CLASS_TYPE_ORDER = ['Singular', 'Ambiguous', 'None', 'Rand']
CLASS_TYPE_COLORS = {
    'Singular': '#1f77b4',
    'Ambiguous': '#ff7f0e',
    'None': '#7f7f7f',
    'Rand': '#c7c7c7',
}


# --------------------------------------------------------------------------
# Training
# --------------------------------------------------------------------------

# What preprocessing each source scale still needs before `train_classifier`.
#
# PySCN wants log-normalized values with `var['highly_variable']` set. Plenty of
# public references are distributed *only* as log-normalized data, and running
# normalize_total + log1p over those again yields log1p(scale(log1p(x))) — which
# compresses the fold changes that drive marker-gene selection, and makes the
# seurat_v3 HVG flavor (which requires integer counts) operate on nonsense.
#
# Scope of the damage, measured rather than assumed: the top-scoring-pair
# features are *bit-identical* under a double log, because normalize_total and
# log1p are both per-cell monotone and the transform only ever compares two
# genes within one cell (see test_pair_features_are_invariant_...). What changes
# is gene *selection* — rank_genes_groups, the PCA/dendrogram used to find
# similar cell types, and HVG — plus a seurat_v3 call handed a 'counts' layer
# that actually holds log values, which is invalid input for that flavor.
# So this is a correctness problem in which genes get picked, not in how a cell
# is scored once they are; its practical effect depends on the dataset.
_TRAINING_PLANS: dict[str, dict[str, Any]] = {
    'raw_counts': {
        'normalize': True, 'log1p': True,
        'hvg_flavor': 'seurat_v3', 'snapshot_counts': True,
        'reason': 'Source is raw counts — normalizing library size and applying log1p.',
    },
    'normalized_linear': {
        'normalize': False, 'log1p': True,
        'hvg_flavor': 'seurat', 'snapshot_counts': False,
        'reason': 'Source is already library-size normalized — applying log1p only.',
    },
    'log_normalized': {
        'normalize': False, 'log1p': False,
        'hvg_flavor': 'seurat', 'snapshot_counts': False,
        'reason': 'Source is already log-normalized — using it as-is.',
    },
    'log_transformed': {
        'normalize': False, 'log1p': False,
        'hvg_flavor': 'seurat', 'snapshot_counts': False,
        'reason': (
            'Source is already on a log scale — using it as-is. Library sizes '
            'were never equalized, so marker selection may be depth-biased.'
        ),
    },
    'unknown': {
        'normalize': False, 'log1p': False,
        'hvg_flavor': 'seurat', 'snapshot_counts': False,
        'uncertain': True,
        'reason': (
            'Could not identify the scale of the source matrix. Leaving it '
            'untransformed, since transforming already-processed data is the '
            'worse error. Set the scale explicitly if you know it.'
        ),
    },
}

# Scales no amount of preprocessing makes trainable.
_UNTRAINABLE = {
    'z_scored': (
        'The source matrix looks scaled / z-scored. Per-gene centering reorders '
        'genes within a cell, which is exactly what the top-scoring-pair '
        'transform reads, so a classifier trained on it would be meaningless. '
        'Train from counts or log-normalized values instead.'
    ),
    'binary': (
        'The source matrix is binary (presence/absence). Gene-pair comparisons '
        'cannot rank two genes that are both 0 or both 1, so there is nothing '
        'for the classifier to learn.'
    ),
    'empty': 'The source matrix has no non-zero values.',
}


def training_plan(verdict: str, override: str | None = None) -> dict[str, Any]:
    """Decide what preprocessing a source matrix still needs before training.

    Args:
        verdict: A :mod:`xcell.layer_scale` verdict for the source matrix.
        override: A user-supplied scale that wins over ``verdict``. Detection is
            a heuristic run on sampled values; whoever produced the data knows
            better, so they get the last word.

    Returns:
        Dict with ``normalize``, ``log1p``, ``hvg_flavor``, ``snapshot_counts``,
        ``reason``, ``source_scale``, ``overridden``, and ``uncertain``.

    Raises:
        ValueError: The scale cannot be trained on at all, or the override is
            not a recognized scale.
    """
    scale = override or verdict
    if override and override not in _TRAINING_PLANS and override not in _UNTRAINABLE:
        raise ValueError(
            f"Unknown source scale '{override}'. Expected one of: "
            f"{', '.join(sorted(_TRAINING_PLANS))}."
        )
    if scale in _UNTRAINABLE:
        raise ValueError(_UNTRAINABLE[scale])

    plan = dict(_TRAINING_PLANS.get(scale, _TRAINING_PLANS['unknown']))
    plan.setdefault('uncertain', False)
    plan['source_scale'] = scale
    plan['detected_scale'] = verdict
    plan['overridden'] = bool(override and override != verdict)
    return plan

def train_and_save(adata, params: dict[str, Any], progress) -> dict[str, Any]:
    """Balance, normalize, find HVGs, train, and pickle the classifier.

    This is the PySCN quickstart sequence, run on a copy the caller has
    already made. Preprocessing is done here rather than expected of the user
    because ``train_classifier`` requires a log-normalized matrix with
    ``var['highly_variable']`` set, and silently produces a poor classifier if
    given something else — a hard precondition badly suited to a UI checkbox.

    Args:
        adata: A private copy of the training data, with raw-ish counts.
        params: Snapshot from ``prepare_pyscn_train`` (groupby, out, and the
            PySCN training parameters).
        progress: ``(fraction, message)`` callback.

    Returns:
        Dict with the written path, classifier metadata, per-class training
        counts, and the number of cells used.
    """
    import pickle as _pickle

    import scanpy as sc

    pyscn_mod = import_pyscn()

    groupby = params['groupby']
    layer = params.get('layer')

    progress(0.05, 'Selecting training cells…')

    # PySCN wants counts in .X to derive HVGs with the seurat_v3 flavor.
    if layer not in (None, '', 'X'):
        adata.X = adata.layers[layer].copy()

    labels = adata.obs[groupby].astype(str)
    adata.obs[groupby] = pd.Categorical(labels)

    cap = params.get('n_cells_per_type')
    if cap:
        rng = np.random.default_rng(0)
        keep: list[int] = []
        for value in adata.obs[groupby].cat.categories:
            idx = np.flatnonzero((adata.obs[groupby] == value).to_numpy())
            if idx.size > cap:
                idx = rng.choice(idx, size=int(cap), replace=False)
            keep.extend(idx.tolist())
        keep.sort()
        adata = adata[keep].copy()

    # Drop labels too rare to learn from; one cell in a class is noise the
    # forest will happily memorize.
    counts = adata.obs[groupby].value_counts()
    too_small = [str(k) for k, v in counts.items() if v < 3]
    if too_small:
        adata = adata[~adata.obs[groupby].astype(str).isin(too_small)].copy()
        adata.obs[groupby] = pd.Categorical(adata.obs[groupby].astype(str))
    if int(adata.obs[groupby].nunique()) < 2:
        raise ValueError(
            "Fewer than 2 cell types have at least 3 cells after balancing — "
            "not enough to train a classifier."
        )

    # Drop symbols PySCN's pair encoding cannot represent. Done after cell
    # selection so the count reported reflects the genes training actually saw.
    dropped_underscore = underscore_gene_names(adata.var_names)
    if dropped_underscore:
        adata = adata[:, [g for g in adata.var_names if '_' not in str(g)]].copy()
    if adata.n_vars < MIN_TRAINING_GENES:
        raise ValueError(
            f"Only {adata.n_vars} usable gene(s) remain after removing symbols "
            "containing underscores, which PySingleCellNet's gene-pair encoding "
            "cannot represent. Rename those genes and train again."
        )

    plan = params['plan']
    progress(0.15, plan['reason'])

    # Only call it 'counts' when it really is counts — a log-valued layer under
    # that name is a lie seurat_v3 would silently consume.
    if plan['snapshot_counts']:
        adata.layers['counts'] = adata.X.copy()
    if plan['normalize']:
        sc.pp.normalize_total(adata)
    if plan['log1p']:
        sc.pp.log1p(adata)

    n_hvg = min(2000, max(200, adata.n_vars - 1))
    hvg_flavor = plan['hvg_flavor']
    try:
        if hvg_flavor == 'seurat_v3':
            sc.pp.highly_variable_genes(
                adata, n_top_genes=n_hvg, flavor='seurat_v3', layer='counts',
            )
        else:
            sc.pp.highly_variable_genes(adata, n_top_genes=n_hvg, flavor='seurat')
    except Exception:
        # HVG is finicky about gene count and dispersion; falling back to
        # "everything is variable" still trains, just with less gene pre-filtering.
        adata.var['highly_variable'] = True
        hvg_flavor = 'none (fallback)'

    progress(0.3, 'Training the random forest…')
    n_comps = min(int(params['n_comps']), min(adata.shape) - 1)
    clf = pyscn_mod.tl.train_classifier(
        adata,
        groupby=groupby,
        n_rand=params.get('n_rand'),
        n_top_genes=int(params['n_top_genes']),
        n_top_gene_pairs=int(params['n_top_gene_pairs']),
        n_trees=int(params['n_trees']),
        n_comps=max(2, n_comps),
    )

    progress(0.92, 'Writing the classifier…')
    out = Path(params['out'])
    with open(out, 'wb') as fh:
        _pickle.dump(clf, fh)

    meta = describe_classifier(clf)
    meta['path'] = str(out)
    meta['file_size_mb'] = round(out.stat().st_size / 1e6, 2)

    train_counts = adata.obs[groupby].value_counts()
    progress(1.0, 'Done')
    return {
        'path': str(out),
        'classifier': meta,
        'colors': classifier_colors_hex(clf),
        'n_cells_used': int(adata.n_obs),
        'groupby': groupby,
        'dropped_labels': too_small,
        'n_genes_dropped_underscore': len(dropped_underscore),
        'preprocessing': {**plan, 'hvg_flavor': hvg_flavor},
        'training_counts': [
            {'name': str(k), 'n_cells': int(v)} for k, v in train_counts.items()
        ],
    }
