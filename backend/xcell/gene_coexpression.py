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
