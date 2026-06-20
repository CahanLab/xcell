"""Robust gene co-expression module detection.

Pure NumPy / SciPy / sklearn. Operates on a (n_genes, n_cells) expression
matrix (one row per gene). All internal helpers work on integer gene
row-indices; gene names are mapped back only at the top-level boundary.
"""
from __future__ import annotations

import numpy as np
from scipy.stats import rankdata
from scipy.cluster.hierarchy import linkage, fcluster
from scipy.spatial.distance import squareform
from sklearn.metrics import silhouette_score

_METRICS = ("bicor", "pearson", "spearman")


def _center_unit(X: np.ndarray) -> np.ndarray:
    """Center each row and scale to unit L2 norm (rows of all-equal -> zeros)."""
    Xc = X - X.mean(axis=1, keepdims=True)
    norm = np.linalg.norm(Xc, axis=1, keepdims=True)
    norm[norm == 0] = 1.0
    return Xc / norm


def _bicor_standardize(X: np.ndarray, c: float = 9.0) -> np.ndarray:
    """Biweight-midcorrelation standardization (WGCNA bicor).

    Rows whose median absolute deviation is 0 fall back to Pearson
    standardization (WGCNA's pearsonFallback).
    """
    med = np.median(X, axis=1, keepdims=True)
    d = X - med
    mad = np.median(np.abs(d), axis=1, keepdims=True)
    zero_mad = mad[:, 0] == 0
    mad_safe = np.where(mad == 0, 1.0, mad)
    u = d / (c * mad_safe)
    w = (1.0 - u ** 2) ** 2
    w[np.abs(u) >= 1.0] = 0.0
    xw = d * w
    norm = np.linalg.norm(xw, axis=1, keepdims=True)
    norm[norm == 0] = 1.0
    Z = xw / norm
    if zero_mad.any():
        Z[zero_mad] = _center_unit(X[zero_mad])
    return Z


def _standardize_profiles(X: np.ndarray, metric: str) -> np.ndarray:
    """Transform rows so that ``Z @ Z.T`` is the requested correlation matrix."""
    if metric not in _METRICS:
        raise ValueError(f"Unknown metric: {metric!r}; expected one of {_METRICS}")
    X = np.asarray(X, dtype=float)
    if metric == "pearson":
        return _center_unit(X)
    if metric == "spearman":
        ranks = np.vstack([rankdata(row) for row in X])
        return _center_unit(ranks)
    return _bicor_standardize(X)


def corr_matrix(X: np.ndarray, metric: str = "bicor") -> np.ndarray:
    """Gene-by-gene correlation matrix under the chosen robust metric."""
    Z = _standardize_profiles(X, metric)
    C = Z @ Z.T
    np.clip(C, -1.0, 1.0, out=C)
    C = 0.5 * (C + C.T)  # enforce exact symmetry
    np.fill_diagonal(C, 1.0)
    return C


def distance_matrix(X: np.ndarray, metric: str = "bicor") -> np.ndarray:
    """Signed correlation distance ``1 - corr`` in ``[0, 2]`` (zero diagonal)."""
    D = 1.0 - corr_matrix(X, metric)
    np.clip(D, 0.0, 2.0, out=D)
    np.fill_diagonal(D, 0.0)
    return D


def _module_coherence(profiles: np.ndarray) -> float:
    """Fraction of variance the top PC (eigengene) explains for a module.

    ``profiles`` are standardized rows (``profiles @ profiles.T`` is the
    correlation matrix with unit diagonal, so its trace is the gene count).
    Returns 1.0 for a single-gene module.
    """
    g = profiles.shape[0]
    if g <= 1:
        return 1.0
    C = profiles @ profiles.T
    w = np.linalg.eigvalsh(C)  # ascending
    return float(w[-1] / g)


def _module_eigengene(profiles: np.ndarray) -> np.ndarray:
    """Module eigengene: top PC over cells, sign-aligned, centered, unit norm."""
    g = profiles.shape[0]
    if g == 1:
        eg = profiles[0].astype(float).copy()
    else:
        C = profiles @ profiles.T
        _, V = np.linalg.eigh(C)
        eg = profiles.T @ V[:, -1]  # length n_cells
    mean_prof = profiles.mean(axis=0)
    if float(eg @ mean_prof) < 0:
        eg = -eg
    eg = eg - eg.mean()
    norm = np.linalg.norm(eg)
    if norm > 0:
        eg = eg / norm
    return eg


def _auto_cut_hierarchical(D: np.ndarray, k_max: int = 20) -> np.ndarray:
    """Average-linkage clustering, K chosen by max silhouette over the distance.

    Returns an integer label per gene. Always yields a partition (every gene
    assigned); the refinement pass corrects under-/over-segmentation.
    """
    g = D.shape[0]
    if g <= 2:
        return np.zeros(g, dtype=int)
    condensed = squareform(D, checks=False)
    Z = linkage(condensed, method="average")
    k_hi = min(k_max, g - 1)
    best_labels = np.zeros(g, dtype=int)
    best_score = -np.inf
    for k in range(2, k_hi + 1):
        labels = fcluster(Z, t=k, criterion="maxclust")
        if len(set(labels)) < 2:
            continue
        score = silhouette_score(D, labels, metric="precomputed")
        if score > best_score:
            best_score, best_labels = score, labels
    return best_labels


def _two_way_split(idx: np.ndarray, Z: np.ndarray, min_genes: int):
    """Average-linkage 2-cut of one module; None if either side < min_genes."""
    sub = Z[idx]
    C = sub @ sub.T
    np.clip(C, -1.0, 1.0, out=C)
    D = 1.0 - C
    np.clip(D, 0.0, 2.0, out=D)
    np.fill_diagonal(D, 0.0)
    link = linkage(squareform(D, checks=False), method="average")
    labels = fcluster(link, t=2, criterion="maxclust")
    a = idx[labels == 1]
    b = idx[labels == 2]
    if len(a) < min_genes or len(b) < min_genes:
        return None
    return a, b


def _split_recursive(idx, Z, purity_threshold, min_genes, depth):
    profiles = Z[idx]
    parent_pve = _module_coherence(profiles)
    if depth <= 0 or len(idx) < 2 * min_genes or parent_pve >= purity_threshold:
        return [idx]
    pair = _two_way_split(idx, Z, min_genes)
    if pair is None:
        return [idx]
    a, b = pair
    if _module_coherence(Z[a]) > parent_pve and _module_coherence(Z[b]) > parent_pve:
        return (
            _split_recursive(a, Z, purity_threshold, min_genes, depth - 1)
            + _split_recursive(b, Z, purity_threshold, min_genes, depth - 1)
        )
    return [idx]


def split_impure_modules(modules, Z, *, purity_threshold, min_genes, max_split_depth):
    """Recursively split modules whose eigengene PVE < purity_threshold."""
    out = []
    for m in modules:
        out.extend(
            _split_recursive(
                np.asarray(m), Z, purity_threshold, min_genes, max_split_depth
            )
        )
    return out


def merge_similar_modules(modules, Z, *, merge_threshold):
    """Iteratively merge the closest module pair whose eigengenes correlate
    at or above merge_threshold, recomputing eigengenes after each merge."""
    modules = [np.asarray(m) for m in modules]
    while len(modules) >= 2:
        egs = [_module_eigengene(Z[m]) for m in modules]
        best_c, bi, bj = -np.inf, -1, -1
        for i in range(len(modules)):
            for j in range(i + 1, len(modules)):
                c = float(egs[i] @ egs[j])
                if c > best_c:
                    best_c, bi, bj = c, i, j
        if best_c < merge_threshold:
            break
        merged = np.concatenate([modules[bi], modules[bj]])
        modules = [m for k, m in enumerate(modules) if k not in (bi, bj)]
        modules.append(merged)
    return modules


def _unit(vec: np.ndarray) -> np.ndarray:
    v = vec - vec.mean()
    n = np.linalg.norm(v)
    return v / n if n > 0 else v


def prune_small_modules(modules, Z, *, min_genes, reassign_floor, extra_orphans=()):
    """Drop modules smaller than min_genes; reassign their genes (and any
    extra_orphans, e.g. base-clustering noise / zero-variance genes) to the
    nearest surviving module whose eigengene correlation >= reassign_floor,
    else collect them into the returned unassigned list.

    Returns (kept_modules, unassigned_gene_indices).
    """
    modules = [np.asarray(m) for m in modules]
    keep = [m for m in modules if len(m) >= min_genes]
    orphans = [int(i) for m in modules if len(m) < min_genes for i in m]
    orphans += [int(i) for i in extra_orphans]
    if not keep:
        return [], sorted(set(orphans))
    egs = [_module_eigengene(Z[m]) for m in keep]
    unassigned = []
    for gi in orphans:
        z = _unit(Z[gi])
        cors = [float(z @ eg) for eg in egs]
        best = int(np.argmax(cors))
        if cors[best] >= reassign_floor:
            keep[best] = np.append(keep[best], gi)
        else:
            unassigned.append(gi)
    return keep, sorted(set(unassigned))
