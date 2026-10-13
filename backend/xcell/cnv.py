"""Copy-number inference from expression, inferCNV-style, and clone calling.

The idea (Patel et al. 2014, Tirosh et al. 2016; inferCNV, infercnvpy): a
chromosomal gain or loss raises or lowers the expression of most genes in
the affected region a little. Any one gene is too noisy to show it, but the
running mean over a hundred neighbouring genes along the genome is not. So:
log-normalized expression, minus what a reference of normal cells expresses
for each gene, clipped, smoothed along each chromosome in windows, centered
per cell. Cells with the same gains and losses then cluster together on the
window matrix, which is how sub-clones are resolved.

Everything here is pure: arrays in, arrays and plain dicts out. The adaptor
pulls the matrix, picks the cells and writes the results back.

Gene positions come from two ``.var`` columns, or from the shipped GENCODE
tables (``data/gene_positions_{human,mouse}.tsv.gz``; see
``data/build_gene_positions.py``).
"""
from __future__ import annotations

import gzip
from pathlib import Path
from typing import Any, Callable, Sequence

import numpy as np

_DATA = Path(__file__).resolve().parent / 'data'

SPECIES = ('human', 'mouse')
AUTOSOMES = {'human': 22, 'mouse': 19}
CLUSTER_METHODS = ('leiden', 'hierarchical', 'none')
GROUP_ORDERS = ('cnv_score', 'category')
ROW_MODES = ('cells', 'groups')
#: Ward linkage holds an n×n distance matrix; above this many cells it is
#: gigabytes, and Leiden answers the same question in seconds.
HIERARCHICAL_MAX_CELLS = 20_000
#: Cells per chunk when densifying the gene-filtered matrix.
CHUNK = 2000
KNN = 15
#: Clone colors: tab20 without the pale pairs, so neighbours in a legend differ.
CLONE_PALETTE = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f',
                 '#bcbd22', '#17becf', '#aec7e8', '#ffbb78', '#98df8a', '#ff9896', '#c5b0d5', '#c49c94',
                 '#f7b6d2', '#c7c7c7', '#dbdb8d', '#9edae5']
#: Expression scales that are linear and need normalize + log1p first.
_LINEAR_VERDICTS = ('raw_counts', 'normalized_linear')
TARGET_SUM = 1e4


# ---------------------------------------------------------------------------
# Positions
# ---------------------------------------------------------------------------

def chromosome_list(species: str, include_xy: bool = False) -> list[str]:
    """Karyotype order: autosomes, then X and Y when asked. Never M."""
    if species not in AUTOSOMES:
        raise ValueError(f"No chromosome list for '{species}'; xcell knows {', '.join(SPECIES)}.")
    out = [f'chr{i}' for i in range(1, AUTOSOMES[species] + 1)]
    if include_xy:
        out += ['chrX', 'chrY']
    return out


def chromosomes_present(positions: dict[str, tuple[str, int, int]], include_xy: bool = False) -> list[str]:
    """Karyotype order for positions of unknown species: numbered
    chromosomes ascending, then X and Y when asked; anything else dropped."""
    names = {rec[0] for rec in positions.values()}
    numbered = sorted((int(n[3:]) for n in names if n[3:].isdigit()))
    out = [f'chr{i}' for i in numbered]
    if include_xy:
        out += [c for c in ('chrX', 'chrY') if c in names]
    return out


def log_normalize(X, verdict: str):
    """Log-normalized copy of ``X`` (sparse stays sparse).

    Linear input is normalized to :data:`TARGET_SUM` per cell and log1p'd;
    log-scale input is returned as a copy. Done once over the whole matrix
    rather than per chunk because the per-cell totals need every gene.
    """
    from scipy import sparse  # noqa: PLC0415
    if sparse.issparse(X):
        M = sparse.csr_matrix(X, dtype=np.float64, copy=True)
        if verdict in _LINEAR_VERDICTS:
            totals = np.asarray(M.sum(axis=1), dtype=np.float64).ravel()
            totals[totals == 0] = 1.0
            rows = np.repeat(np.arange(M.shape[0]), np.diff(M.indptr))
            M.data = np.log1p(M.data / totals[rows] * TARGET_SUM)
        return M
    M = np.array(X, dtype=np.float64, copy=True)
    if verdict in _LINEAR_VERDICTS:
        totals = M.sum(axis=1, keepdims=True)
        totals[totals == 0] = 1.0
        M = np.log1p(M / totals * TARGET_SUM)
    return M


def _norm_chrom(value: Any) -> str | None:
    """``1`` / ``chr1`` / ``Chr1`` → ``chr1``; ``MT`` → ``chrM``; blanks → None."""
    if value is None:
        return None
    s = str(value).strip()
    if not s or s.lower() in ('nan', 'none', 'na'):
        return None
    if s.lower().startswith('chr'):
        s = s[3:]
    if s.upper() == 'MT':
        s = 'M'
    return f'chr{s}'


def load_positions(species: str) -> dict[str, tuple[str, int, int]]:
    """The shipped table for a species, keyed by Ensembl id *and* by symbol.

    Symbols are keyed as written and upper-cased, so a lookup can try the
    dataset's spelling first and the case-folded form second.
    """
    if species not in SPECIES:
        raise ValueError(f"No gene position table for '{species}'; xcell ships {' and '.join(SPECIES)}.")
    path = _DATA / f'gene_positions_{species}.tsv.gz'
    if not path.exists():
        raise ValueError(f'The {species} gene position table is missing from this install ({path.name}). '
                         'Rebuild it with `python xcell/data/build_gene_positions.py`.')
    table: dict[str, tuple[str, int, int]] = {}
    with gzip.open(path, 'rt') as fh:
        header = fh.readline()
        if not header.startswith('gene_id'):
            raise ValueError(f'{path.name} has an unexpected header')
        for line in fh:
            gid, sym, chrom, start, end = line.rstrip('\n').split('\t')
            rec = (chrom, int(start), int(end))
            table[gid] = rec
            # First symbol wins: a readthrough or antisense gene sharing a
            # name would otherwise displace the gene itself.
            table.setdefault(sym, rec)
            table.setdefault(sym.upper(), rec)
    return table


def match_positions(var_names: Sequence[str], table: dict[str, tuple[str, int, int]],
                    ) -> tuple[dict[str, tuple[str, int, int]], dict[str, int]]:
    """Positions for the dataset's own gene names, with how each matched.

    Order of attempts: the name as an Ensembl id with any version stripped,
    the name as a symbol, the upper-cased symbol (a human table against
    Title-case mouse spellings, or the reverse).
    """
    out: dict[str, tuple[str, int, int]] = {}
    how = {'ensembl': 0, 'symbol': 0, 'symbol_upper': 0, 'unmatched': 0}
    for name in var_names:
        name = str(name)
        base = name.split('.', 1)[0] if name.startswith('ENS') else None
        if base is not None and base in table:
            out[name] = table[base]
            how['ensembl'] += 1
        elif name in table:
            out[name] = table[name]
            how['symbol'] += 1
        elif name.upper() in table:
            out[name] = table[name.upper()]
            how['symbol_upper'] += 1
        else:
            how['unmatched'] += 1
    return out, how


def positions_from_columns(var_names: Sequence[str], chrom_values: Sequence[Any],
                           start_values: Sequence[Any]) -> dict[str, tuple[str, int, int]]:
    """Positions from two ``.var`` columns. Genes lacking either are left out."""
    out: dict[str, tuple[str, int, int]] = {}
    for name, chrom, start in zip(var_names, chrom_values, start_values):
        c = _norm_chrom(chrom)
        if c is None:
            continue
        try:
            s = float(start)
        except (TypeError, ValueError):
            continue
        if not np.isfinite(s):
            continue
        out[str(name)] = (c, int(s), int(s))
    return out


def gene_order(positions: dict[str, tuple[str, int, int]], var_names: Sequence[str],
               include: Sequence[str]) -> list[dict[str, Any]]:
    """Genes grouped by chromosome in ``include``'s order, sorted by start.

    Each entry: ``chromosome``, ``indices`` (positions in ``var_names``),
    ``starts``. Chromosomes with no placed gene are dropped.
    """
    by_chrom: dict[str, list[tuple[int, int]]] = {c: [] for c in include}
    for i, name in enumerate(var_names):
        rec = positions.get(str(name))
        if rec is None or rec[0] not in by_chrom:
            continue
        by_chrom[rec[0]].append((rec[1], i))
    out = []
    for chrom in include:
        rows = sorted(by_chrom[chrom])
        if not rows:
            continue
        out.append({'chromosome': chrom,
                    'indices': np.array([i for _, i in rows], dtype=int),
                    'starts': np.array([s for s, _ in rows], dtype=np.int64)})
    return out


# ---------------------------------------------------------------------------
# Windows
# ---------------------------------------------------------------------------

def plan_windows(order: list[dict[str, Any]], window: int, step: int,
                 ) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Lay running-mean windows along each chromosome.

    ``gene_idx`` in each window indexes columns of the *filtered* matrix,
    whose columns are ``order``'s genes concatenated. A chromosome with fewer
    genes than ``window`` gets a single window over all of them: dropping it
    would leave a gap in the karyogram that reads as "no data", which is
    worse than a coarser estimate.
    """
    window, step = int(window), int(step)
    if window < 2:
        raise ValueError('window must be at least 2 genes')
    if step < 1 or step > window:
        raise ValueError(f'step ({step}) must be between 1 and the window ({window})')
    windows: list[dict[str, Any]] = []
    chromosomes: list[dict[str, Any]] = []
    offset = 0
    for entry in order:
        n = len(entry['indices'])
        first = len(windows)
        if n <= window:
            spans = [(0, n)]
        else:
            spans = [(s, s + window) for s in range(0, n - window + 1, step)]
            # The regular grid stops short of the end whenever (n − window) is
            # not a multiple of step; one last window keeps the tail genes in.
            if spans[-1][1] < n:
                spans.append((n - window, n))
        for lo, hi in spans:
            windows.append({
                'chromosome': entry['chromosome'],
                'start': int(entry['starts'][lo]),
                'end': int(entry['starts'][hi - 1]),
                'n_genes': hi - lo,
                'gene_idx': np.arange(offset + lo, offset + hi, dtype=int),
            })
        chromosomes.append({'name': entry['chromosome'], 'start': first, 'size': len(windows) - first})
        offset += n
    return windows, chromosomes


def _window_operator(windows: list[dict[str, Any]], n_genes: int):
    """Sparse (n_genes × n_windows) averaging matrix: ``X @ W`` smooths."""
    from scipy import sparse  # noqa: PLC0415
    rows = np.concatenate([w['gene_idx'] for w in windows])
    cols = np.concatenate([np.full(len(w['gene_idx']), j) for j, w in enumerate(windows)])
    vals = np.concatenate([np.full(len(w['gene_idx']), 1.0 / len(w['gene_idx'])) for w in windows])
    return sparse.csc_matrix((vals, (rows, cols)), shape=(n_genes, len(windows)), dtype=np.float64)


# ---------------------------------------------------------------------------
# Inference
# ---------------------------------------------------------------------------

def reference_subtract(X: np.ndarray, ref_means: np.ndarray) -> np.ndarray:
    """Expression relative to the reference.

    One reference group: subtract its per-gene mean. Several: a value between
    the lowest and highest group mean is 0 (normal cells of any of those
    types would show it), otherwise the distance to the nearest bound — the
    rule infercnvpy uses, so a mixed normal reference does not paint every
    tumor cell with the differences *between* normal types.
    """
    ref_means = np.asarray(ref_means, dtype=np.float64)
    if ref_means.ndim == 1:
        ref_means = ref_means[None, :]
    X = np.asarray(X, dtype=np.float64)
    if ref_means.shape[0] == 1:
        return X - ref_means[0]
    lo = ref_means.min(axis=0)
    hi = ref_means.max(axis=0)
    out = np.zeros_like(X)
    above = X > hi
    below = X < lo
    out[above] = (X - hi)[above]
    out[below] = (X - lo)[below]
    return out


def infer(get_rows: Callable[[np.ndarray], np.ndarray], cells: np.ndarray, windows: list[dict[str, Any]],
          ref_means: np.ndarray, *, lfc_clip: float, center_cells: bool, chunk: int = CHUNK,
          report: Callable[[float, str], None] | None = None) -> np.ndarray:
    """The cells × windows CNV matrix.

    ``get_rows(idx)`` returns dense log-normalized expression for those cells
    over the filtered genes, in window-plan order. Cells are processed in
    chunks so a large matrix is never dense all at once.
    """
    cells = np.asarray(cells, dtype=int)
    n_genes = int(np.asarray(ref_means).shape[-1])
    W = _window_operator(windows, n_genes)
    out = np.empty((cells.size, len(windows)), dtype=np.float32)
    n_chunks = max(1, int(np.ceil(cells.size / max(1, int(chunk)))))
    for k, lo in enumerate(range(0, cells.size, max(1, int(chunk)))):
        hi = min(cells.size, lo + int(chunk))
        X = get_rows(cells[lo:hi])
        rel = reference_subtract(X, ref_means)
        np.clip(rel, -float(lfc_clip), float(lfc_clip), out=rel)
        sm = np.asarray(rel @ W)
        if center_cells:
            sm -= np.median(sm, axis=1, keepdims=True)
        out[lo:hi] = sm.astype(np.float32)
        if report is not None:
            report(0.1 + 0.6 * (k + 1) / n_chunks, f'Smoothing windows: {hi:,} / {cells.size:,} cells')
    return out


def apply_noise_threshold(M: np.ndarray, k: float) -> np.ndarray:
    """Zero values below ``k`` standard deviations of the whole matrix.

    inferCNV's dynamic threshold: most windows in most cells carry no event,
    so the matrix-wide spread is the noise floor.
    """
    if not k or k <= 0:
        return M
    sd = float(np.nanstd(M))
    out = M.copy()
    out[np.abs(out) < k * sd] = 0
    return out


def cnv_score(M: np.ndarray) -> np.ndarray:
    """Per-cell CNV burden: mean |value| over windows (NaN rows stay NaN)."""
    with np.errstate(invalid='ignore'):
        return np.nanmean(np.abs(M), axis=1) if M.shape[1] else np.full(M.shape[0], np.nan)


def chromosome_means(M: np.ndarray, chromosomes: list[dict[str, Any]]) -> np.ndarray:
    out = np.empty((M.shape[0], len(chromosomes)), dtype=np.float64)
    for j, c in enumerate(chromosomes):
        out[:, j] = M[:, c['start']:c['start'] + c['size']].mean(axis=1)
    return out


# ---------------------------------------------------------------------------
# Clones
# ---------------------------------------------------------------------------

def _pca(M: np.ndarray, n_pcs: int, seed: int) -> np.ndarray:
    from sklearn.decomposition import PCA  # noqa: PLC0415
    M = np.asarray(M, dtype=np.float64)
    # arpack needs n_components strictly below both dimensions; with one or
    # two windows there is nothing to reduce, so cluster the matrix itself.
    n = int(min(int(n_pcs), M.shape[0] - 1, M.shape[1] - 1))
    if n < 1:
        return M
    return PCA(n_components=n, svd_solver='arpack', random_state=int(seed)).fit_transform(M)


def _rank_by_score(labels: np.ndarray, score: np.ndarray) -> np.ndarray:
    """Relabel so 0 is the clone with the lowest mean CNV burden."""
    ids = np.unique(labels)
    means = np.array([np.nanmean(score[labels == k]) for k in ids])
    rank = {int(k): r for r, k in enumerate(ids[np.argsort(means, kind='stable')])}
    return np.array([rank[int(k)] for k in labels], dtype=int)


def cluster_clones(M: np.ndarray, *, method: str, resolution: float, n_clusters: int, n_pcs: int,
                   seed: int = 0) -> np.ndarray:
    """Clone labels (0..k−1, ranked by burden) from the window matrix.

    ``leiden``: PCA, exact kNN (k = 15), Leiden on the symmetrized graph.
    ``hierarchical``: Ward linkage on the PCA scores cut into ``n_clusters``
    — inferCNV's own grouping, for when a fixed number of clones is wanted.
    """
    if method not in ('leiden', 'hierarchical'):
        raise ValueError(f"Unknown clone method '{method}'; expected leiden or hierarchical.")
    M = np.asarray(M)
    n = M.shape[0]
    if n < 3:
        raise ValueError(f'Need at least 3 cells to call clones, got {n}.')
    if method == 'hierarchical' and n > HIERARCHICAL_MAX_CELLS:
        raise ValueError(f'Hierarchical clustering of {n:,} cells needs an n×n distance matrix; '
                         f'above {HIERARCHICAL_MAX_CELLS:,} cells use leiden.')
    Z = _pca(M, n_pcs, seed)
    if method == 'hierarchical':
        from scipy.cluster.hierarchy import fcluster, linkage  # noqa: PLC0415
        k = int(max(1, min(int(n_clusters), n)))
        labels = fcluster(linkage(Z, method='ward'), t=k, criterion='maxclust') - 1
    else:
        import anndata  # noqa: PLC0415
        import scanpy as sc  # noqa: PLC0415
        from scipy import sparse  # noqa: PLC0415
        from sklearn.neighbors import NearestNeighbors  # noqa: PLC0415
        k = int(max(1, min(KNN, n - 1)))
        D = NearestNeighbors(n_neighbors=k).fit(Z).kneighbors_graph(mode='connectivity')
        D = sparse.csr_matrix(D)
        A = ((D + D.T) > 0).astype(np.float32)
        ad = anndata.AnnData(obs={'i': np.arange(n)})
        ad.obsp['connectivities'] = sparse.csr_matrix(A)
        ad.uns['neighbors'] = {'connectivities_key': 'connectivities', 'distances_key': 'connectivities',
                               'params': {'method': 'knn', 'n_neighbors': k}}
        sc.tl.leiden(ad, resolution=float(resolution), key_added='c', random_state=int(seed),
                     flavor='igraph', n_iterations=2, directed=False)
        labels = ad.obs['c'].astype(int).to_numpy()
    return _rank_by_score(np.asarray(labels, dtype=int), cnv_score(M))


def _finite(v: float) -> float | None:
    return float(v) if np.isfinite(v) else None


def clone_summary(clones: np.ndarray, score: np.ndarray, chrom_means: np.ndarray, chrom_names: Sequence[str],
                  *, ref_mask: np.ndarray | None = None, group_labels: np.ndarray | None = None,
                  ) -> list[dict[str, Any]]:
    """One row per clone: size, burden, how much of the reference it holds,
    which input group dominates it, and its mean CNV per chromosome."""
    rows = []
    for k in np.unique(clones):
        m = clones == k
        row: dict[str, Any] = {
            'clone': int(k),
            'n_cells': int(m.sum()),
            'mean_score': _finite(np.nanmean(score[m])),
            'reference_fraction': _finite(ref_mask[m].mean()) if ref_mask is not None else None,
            'top_group': None,
            'top_group_fraction': None,
            'chromosome_means': {str(c): _finite(chrom_means[m, j].mean()) for j, c in enumerate(chrom_names)},
        }
        if group_labels is not None:
            vals, counts = np.unique(np.asarray(group_labels)[m].astype(str), return_counts=True)
            j = int(np.argmax(counts))
            row['top_group'] = str(vals[j])
            row['top_group_fraction'] = float(counts[j] / counts.sum())
        rows.append(row)
    return rows


# ---------------------------------------------------------------------------
# Heatmap
# ---------------------------------------------------------------------------

def heatmap_data(M: np.ndarray, *, groups: np.ndarray | None, group_order: str, row_mode: str, max_rows: int,
                 vmax: float | None, chromosomes: list[dict[str, Any]]) -> dict[str, Any]:
    """Rows × windows for the CNV heatmap.

    Rows are cells binned within their group (``cells``: each group gets rows
    in proportion to its size, at least one; cells inside a group are
    ordered by burden so a bin averages like with like) or one mean row per
    group (``groups``). NaN rows (cells outside the run's mask) are left out.
    """
    if group_order not in GROUP_ORDERS:
        raise ValueError(f"group_order must be one of {GROUP_ORDERS}")
    if row_mode not in ROW_MODES:
        raise ValueError(f"row_mode must be one of {ROW_MODES}")
    M = np.asarray(M, dtype=np.float64)
    keep = ~np.isnan(M).any(axis=1)
    M = M[keep]
    score = cnv_score(M) if M.size else np.zeros(0)
    if groups is None:
        labels = np.array([None] * M.shape[0], dtype=object)
        names: list[Any] = [None]
    else:
        labels = np.asarray(groups, dtype=object)[keep]
        names = list(dict.fromkeys(str(g) for g in labels))
        labels = labels.astype(str)
        if group_order == 'category':
            names = sorted(names)
        else:
            names.sort(key=lambda g: float(np.mean(score[labels == g])))
    max_rows = int(max(1, max_rows))
    matrix: list[list[float]] = []
    row_labels: list[str] = []
    row_groups: list[str | None] = []
    sizes = {g: int((labels == g).sum()) if g is not None else M.shape[0] for g in names}
    total = max(1, sum(sizes.values()))
    for g in names:
        m = np.ones(M.shape[0], dtype=bool) if g is None else labels == g
        sub = M[m]
        if sub.shape[0] == 0:
            continue
        if row_mode == 'groups':
            matrix.append(sub.mean(axis=0).tolist())
            row_labels.append(str(g) if g is not None else 'all cells')
            row_groups.append(g)
            continue
        n_rows = int(max(1, min(sub.shape[0], round(max_rows * sizes[g] / total))))
        order = np.argsort(score[m], kind='stable')
        for part in np.array_split(order, n_rows):
            matrix.append(sub[part].mean(axis=0).tolist())
            row_labels.append(f'{g} ({len(part)})' if g is not None else f'{len(part)} cells')
            row_groups.append(g)
    if vmax is None or not (vmax > 0):
        flat = np.abs(np.asarray(matrix)) if matrix else np.zeros(1)
        vmax = float(np.percentile(flat, 99)) if flat.size else 0.0
        if not vmax > 0:
            vmax = float(flat.max()) if flat.size else 1.0
        if not vmax > 0:
            vmax = 1.0
    return {
        'matrix': matrix,
        'row_labels': row_labels,
        'row_groups': row_groups,
        'column_groups': [dict(c) for c in chromosomes],
        'n_cells': int(M.shape[0]),
        'vmax': float(vmax),
    }
