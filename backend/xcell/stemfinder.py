"""PyStemFinder differentiation scores, as an optional xcell backend.

`PyStemFinder <https://github.com/CahanLab/PyStemFinder>`_ estimates how far
each cell has progressed through differentiation: less differentiated cells
vary more in their expression of cell cycle genes than their neighbours do,
and stemFinder scores that heterogeneity within each cell's kNN neighbourhood
(Noller & Cahan, *Brief. Bioinform.* 2024).

PyStemFinder works on an AnnData whose ``.X`` is already in the right form and
whose kNN graph was built for it. xcell's live AnnData is neither, so this
module does the preparing and hands PyStemFinder a throwaway AnnData of just
the marker columns:

**Two expression scales.** The default ``gini`` method binarizes at a
threshold of 0 on *scaled* expression — each gene split at its own mean. The
``stdev`` / ``variance`` methods and the cell-cycle mean read
*log-normalized* expression, as the R package does. Both are derived from the
marker columns only: normalizing to 1e4 needs every gene's counts (the row
sums), but log1p and per-gene scaling do not, so a 97-gene slice of a
30,000-gene matrix gives the same numbers as scaling all of it.

**The graph is an intermediate.** The method prescribes k = √n, far larger
than the 15 a clustering graph uses, so by default a graph is built from a PC
embedding for the run and discarded. PyStemFinder reads only its sparsity.

The package is an optional dependency, the way :mod:`xcell.pyscn` handles
PySingleCellNet: nothing imports it at module scope, and its absence surfaces
as a :class:`ValueError` carrying install instructions.
"""
from __future__ import annotations

import warnings
from typing import Any, Sequence

import numpy as np
from scipy import sparse

INSTALL_HINT = (
    "PyStemFinder is not installed. Install it with `pip install pystemfinder` "
    "(a fresh `pixi install` includes it), then restart the xcell backend."
)

#: Metric key → the .obs columns it writes (before any suffix).
METRIC_COLUMNS: dict[str, tuple[str, ...]] = {
    'stemfinder': ('stemfinder', 'stemfinder_raw'),
    'diffometer': ('diffometer',),
    'cc_mean': ('stemfinder_cc_mean',),
    'n_tfs': ('stemfinder_n_TFs',),
}
METHODS = ('gini', 'stdev', 'variance')
BINARIZE_ON = ('scaled', 'log_normalized')
WEIGHT_BY = ('equal', 'expression', 'presence')
SPECIES = ('mouse', 'human', 'celegans')

# recipe_stemfinder's normalize_total target; Seurat's default too.
TARGET_SUM = 1e4
# sc.pp.scale's clip in recipe_stemfinder. Irrelevant at threshold 0.
SCALE_MAX = 10.0

_LINEAR_VERDICTS = ('raw_counts', 'normalized_linear')


def _import_module():
    """Return the pystemfinder module, or None if it cannot be imported.

    Split out so tests can simulate the package being absent.
    """
    try:
        import pystemfinder  # noqa: PLC0415

        return pystemfinder
    except Exception:
        return None


def import_psf():
    """Return the pystemfinder module or raise with install instructions."""
    mod = _import_module()
    if mod is None:
        raise ValueError(INSTALL_HINT)
    return mod


def availability() -> dict[str, Any]:
    mod = _import_module()
    if mod is None:
        return {'available': False, 'version': None, 'error': 'not installed', 'install_hint': INSTALL_HINT}
    return {'available': True, 'version': getattr(mod, '__version__', None), 'error': None, 'install_hint': None}


def cell_cycle_markers(species: str) -> list[str]:
    if species not in SPECIES:
        raise ValueError(f"Unknown species '{species}'. Expected one of {', '.join(SPECIES)}.")
    return list(import_psf().cell_cycle_genes(species))


def transcription_factors(species: str) -> list[str]:
    if species not in SPECIES:
        raise ValueError(f"Unknown species '{species}'. Expected one of {', '.join(SPECIES)}.")
    return list(import_psf().transcription_factors(species))


def match_genes(wanted: Sequence[str], var_names: Sequence[str]) -> tuple[list[str], list[str]]:
    """Find ``wanted`` among ``var_names``, returning the dataset's spelling.

    Case-insensitive, so the human cell cycle list finds ``Mcm2`` in mouse data;
    an exact match wins over a case-folded one. Each gene appears once.
    """
    names = [str(v) for v in var_names]
    exact = set(names)
    folded: dict[str, str] = {}
    for v in names:
        folded.setdefault(v.lower(), v)
    present: list[str] = []
    missing: list[str] = []
    seen: set[str] = set()
    for g in wanted:
        g = str(g)
        hit = g if g in exact else folded.get(g.lower())
        if hit is None:
            if g.lower() not in seen:
                missing.append(g)
                seen.add(g.lower())
            continue
        if hit not in present:
            present.append(hit)
        seen.add(g.lower())
    return present, missing


def log_normalized_columns(X, cols: Sequence[int], verdict: str) -> np.ndarray:
    """Dense log-normalized expression of columns ``cols`` (cells x len(cols)).

    Linear input is normalized to :data:`TARGET_SUM` using each cell's total
    over *every* gene, then log1p'd; log-scale input is returned as is.
    """
    cols = list(cols)
    sub = X[:, cols]
    M = sub.toarray() if sparse.issparse(sub) else np.asarray(sub)
    M = M.astype(np.float64, copy=True)
    if verdict in _LINEAR_VERDICTS:
        totals = np.asarray(X.sum(axis=1), dtype=np.float64).ravel()
        totals[totals == 0] = 1.0
        M = np.log1p(M / totals[:, None] * TARGET_SUM)
    return M


def scale_columns(M: np.ndarray) -> np.ndarray:
    """Per-gene z-scores, clipped at :data:`SCALE_MAX` — ``sc.pp.scale`` itself."""
    import anndata  # noqa: PLC0415
    import scanpy as sc  # noqa: PLC0415

    ad = anndata.AnnData(X=np.array(M, dtype=np.float64))
    sc.pp.scale(ad, max_value=SCALE_MAX)
    return np.asarray(ad.X, dtype=np.float64)


def count_expressed(X, cols: Sequence[int], threshold: float = 0.0) -> np.ndarray:
    """How many of columns ``cols`` each cell expresses above ``threshold``.

    Kept sparse: the transcription factor list is ~1,800 genes, and a dense
    cells x TFs block for 50k cells is most of a gigabyte.
    """
    sub = X[:, list(cols)]
    if sparse.issparse(sub):
        return np.asarray((sub > threshold).sum(axis=1)).ravel().astype(np.int64)
    return (np.asarray(sub) > threshold).sum(axis=1).astype(np.int64)


def knn_from_embedding(rep: np.ndarray, n_neighbors: int, n_pcs: int | None, seed: int = 0) -> sparse.csr_matrix:
    """The kNN distances graph ``sc.pp.neighbors`` builds on ``rep``.

    ``n_neighbors`` counts the cell itself, as in scanpy, so each cell gets
    ``n_neighbors - 1`` neighbours. Clamped below the number of cells.
    """
    import anndata  # noqa: PLC0415
    import scanpy as sc  # noqa: PLC0415

    rep = np.asarray(rep, dtype=np.float32)
    n = rep.shape[0]
    if n < 3:
        raise ValueError(f"Need at least 3 cells to build a neighbour graph, got {n}.")
    k = int(max(2, min(int(n_neighbors), n - 1)))
    use = None if n_pcs is None else int(max(1, min(int(n_pcs), rep.shape[1])))
    ad = anndata.AnnData(obs={'i': np.arange(n)})
    ad.obsm['X_rep'] = rep if use is None else rep[:, :use]
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        sc.pp.neighbors(ad, n_neighbors=k, use_rep='X_rep', random_state=seed)
    return sparse.csr_matrix(ad.obsp['distances'])


def subgraph(graph, idx: np.ndarray) -> sparse.csr_matrix:
    """Edges among ``idx`` only, re-indexed to ``idx``'s order."""
    g = sparse.csr_matrix(graph)
    return sparse.csr_matrix(g[idx][:, idx])


def validate_options(*, metrics: Sequence[str], method: str, binarize_on: str,
                     weight_by: str, scale_verdict: str) -> None:
    """Reject combinations that would run and return meaningless numbers."""
    if not metrics:
        raise ValueError("Pick at least one metric.")
    bad = [m for m in metrics if m not in METRIC_COLUMNS]
    if bad:
        raise ValueError(f"Unknown metric(s) {bad}. Expected any of {list(METRIC_COLUMNS)}.")
    if method not in METHODS:
        raise ValueError(f"Unknown stemFinder method '{method}'. Expected one of {list(METHODS)}.")
    if binarize_on not in BINARIZE_ON:
        raise ValueError(f"Unknown binarizing input '{binarize_on}'. Expected one of {list(BINARIZE_ON)}.")
    if weight_by not in WEIGHT_BY:
        raise ValueError(f"Unknown diffOmeter weighting '{weight_by}'. Expected one of {list(WEIGHT_BY)}.")
    if weight_by == 'expression' and 'diffometer' in metrics and binarize_on == 'scaled':
        raise ValueError(
            "diffOmeter's 'expression' weighting averages each gene's expression, which is ~0 "
            "for scaled expression. Binarize on log-normalized expression, or weight equally."
        )
    if scale_verdict == 'z_scored':
        needs_log = []
        if 'stemfinder' in metrics and method != 'gini':
            needs_log.append(f"stemFinder '{method}'")
        if 'cc_mean' in metrics:
            needs_log.append('cell-cycle expression')
        if 'n_tfs' in metrics:
            needs_log.append('expressed TFs')
        if binarize_on == 'log_normalized' and ({'stemfinder', 'diffometer'} & set(metrics)):
            needs_log.append('binarizing on log-normalized expression')
        if needs_log:
            raise ValueError(
                "The expression matrix is scaled (z-scored), so it cannot give log-normalized "
                f"expression for: {', '.join(needs_log)}. Pick a layer of counts or "
                "log-normalized data, or use stemFinder 'gini' and diffOmeter only."
            )


def compute_scores(
    log_expr: np.ndarray,
    markers: Sequence[str],
    graph,
    *,
    metrics: Sequence[str],
    method: str = 'gini',
    threshold: float = 0.0,
    binarize_on: str = 'scaled',
    weight_by: str = 'equal',
    include_self: bool = True,
    scaled_expr: np.ndarray | None = None,
    tf_counts: np.ndarray | None = None,
) -> dict[str, np.ndarray]:
    """Run the chosen PyStemFinder metrics; returns column name → per-cell values.

    ``log_expr`` is the markers' log-normalized expression (cells x markers),
    or None when only scaled input exists (z-scored data). ``scaled_expr``
    defaults to scaling ``log_expr``. ``graph`` is any cells x cells matrix
    whose sparsity is the kNN neighbourhood. A cell with no neighbours gets
    NaN — PyStemFinder would divide by zero there, and the NaN would then
    reach ``raw.max()`` and blank every cell's normalized score.
    """
    import anndata  # noqa: PLC0415

    psf = import_psf()
    markers = list(markers)
    if scaled_expr is None:
        scaled_expr = scale_columns(log_expr)
    ad = anndata.AnnData(X=np.asarray(scaled_expr, dtype=np.float64))
    ad.var_names = markers
    if log_expr is not None:
        ad.layers['log'] = np.asarray(log_expr, dtype=np.float64)
    if graph is not None:          # the cell-cycle mean needs no neighbourhood
        ad.obsp['distances'] = sparse.csr_matrix(graph)
    bin_layer = None if binarize_on == 'scaled' else 'log'

    out: dict[str, np.ndarray] = {}
    # Division by an empty neighbourhood is expected (see above); the NaNs it
    # leaves are the answer, not a warning.
    with np.errstate(divide='ignore', invalid='ignore'), warnings.catch_warnings():
        warnings.simplefilter('ignore', RuntimeWarning)
        if 'stemfinder' in metrics:
            layer = bin_layer if method == 'gini' else 'log'
            psf.stemfinder(ad, markers, threshold=threshold, method=method, layer=layer)
            raw = np.asarray(ad.obs['stemfinder_raw'], dtype=np.float64)
            raw[~np.isfinite(raw)] = np.nan
            top = np.nanmax(raw) if np.isfinite(raw).any() else np.nan
            out['stemfinder'] = 1.0 - raw / top if top and np.isfinite(top) else np.full_like(raw, np.nan)
            out['stemfinder_raw'] = raw
        if 'diffometer' in metrics:
            psf.diffometer(ad, markers, threshold=threshold, weight_by=weight_by,
                           include_self=include_self, layer=bin_layer)
            d = np.asarray(ad.obs['diffometer'], dtype=np.float64)
            d[~np.isfinite(d)] = np.nan
            out['diffometer'] = d
        if 'cc_mean' in metrics:
            psf.gene_set_score(ad, markers, layer='log', key_added='cc_mean')
            out['stemfinder_cc_mean'] = np.asarray(ad.obs['cc_mean'], dtype=np.float64)
    if 'n_tfs' in metrics:
        if tf_counts is None:
            raise ValueError("Expressed-TF counts were not computed.")
        out['stemfinder_n_TFs'] = np.asarray(tf_counts, dtype=np.float64)
    return out


def _finite_or_none(x: float) -> float | None:
    return float(x) if np.isfinite(x) else None


def column_stats(values: np.ndarray) -> dict[str, Any]:
    v = np.asarray(values, dtype=np.float64)
    v = v[np.isfinite(v)]
    if v.size == 0:
        return {'n': 0, 'min': None, 'median': None, 'max': None}
    return {'n': int(v.size), 'min': float(v.min()), 'median': float(np.median(v)), 'max': float(v.max())}


def group_summary(scores: dict[str, np.ndarray], labels: np.ndarray, sort_by: str,
                  descending: bool = False) -> list[dict[str, Any]]:
    """Per-group medians of each score, sorted by ``sort_by`` (ascending by default).

    Groups whose ``sort_by`` median is not computable sort last. Values are
    JSON-safe: None, never NaN.
    """
    labels = np.asarray(labels, dtype=object)
    rows = []
    for g in _unique_in_order(labels):
        if g is None or (isinstance(g, float) and np.isnan(g)):
            continue
        sel = labels == g
        medians = {}
        for col, vals in scores.items():
            v = np.asarray(vals, dtype=np.float64)[sel]
            v = v[np.isfinite(v)]
            medians[col] = float(np.median(v)) if v.size else None
        rows.append({'group': str(g), 'n': int(sel.sum()), 'medians': medians})
    sign = -1.0 if descending else 1.0
    rows.sort(key=lambda r: (r['medians'].get(sort_by) is None, sign * (r['medians'].get(sort_by) or 0.0)))
    return rows


def _unique_in_order(values: np.ndarray) -> list:
    """Unique values in first-seen order (pandas' order, without the import)."""
    seen: dict = {}
    for v in values:
        if v not in seen:
            seen[v] = None
    return list(seen)
