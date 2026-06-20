"""Robust gene co-expression module detection.

Pure NumPy / SciPy / sklearn. Operates on a (n_genes, n_cells) expression
matrix (one row per gene). All internal helpers work on integer gene
row-indices; gene names are mapped back only at the top-level boundary.
"""
from __future__ import annotations

import numpy as np
from scipy.stats import rankdata

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
