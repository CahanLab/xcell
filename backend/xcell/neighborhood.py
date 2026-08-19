"""Cell-type neighborhood composition and co-location enrichment.

Quantifies, for every cell, which cell types surround it, and tests for every
pair of types whether they neighbor each other more or less often than expected
from their abundances alone. The method is the standard permutation approach of
histoCAT (Schapiro et al., Nat Methods 2017) and squidpy's nhood_enrichment
(Palla et al., Nat Methods 2022), with per-cell composition windows as in
Schürch et al. (Cell 2020):

  * a spatial neighbor graph (kNN, default k=10, or an epsilon-ball radius);
  * per-cell neighborhood composition = fraction of each type among a cell's
    neighbors (cells x types), aggregated to a directed types x types matrix
    (row = the type whose neighborhoods we look at);
  * observed directed neighbor-pair counts vs a null built by shuffling the
    type labels over fixed graph positions -- this conditions on both the
    tissue architecture and the type abundances, so enrichment cannot be
    driven by a type merely being common;
  * z-scores against that null (continuous scores, preferred over categorical
    significance calls), empirical two-sided p-values, and BH-adjusted q-values.

With ``sections``, edges never cross sections and labels shuffle only within
their own section, so serial sections neither share neighbors nor mix nulls.

Pure: arrays in, dict out. Never imports the adaptor, never mutates state.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from scipy.spatial import cKDTree

from xcell.ligrec import benjamini_hochberg, section_permutation


def knn_edges(
    coords: np.ndarray,
    n_neighs: int,
    sections: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Directed kNN edges (src -> its k nearest neighbors), self excluded.

    k is clamped to section size - 1, so tiny sections degrade instead of
    erroring. Returns (src, dst) index arrays.
    """
    coords = np.asarray(coords, float)
    n = coords.shape[0]
    if sections is None:
        groups = [np.arange(n)]
    else:
        sec = np.asarray(sections)
        groups = [np.where(sec == s)[0] for s in np.unique(sec)]

    src_parts: list[np.ndarray] = []
    dst_parts: list[np.ndarray] = []
    for idx in groups:
        m = len(idx)
        k = min(n_neighs, m - 1)
        if k < 1:
            continue
        tree = cKDTree(coords[idx])
        _, nbr = tree.query(coords[idx], k=k + 1)
        nbr = np.atleast_2d(nbr)
        # Drop self from each row. With exact duplicate coordinates the self
        # index may land anywhere (or, off a tie, not at all) -- so mask it
        # explicitly and, where it never appeared, drop the farthest instead.
        keep = nbr != np.arange(m)[:, None]
        keep[keep.sum(axis=1) == k + 1, -1] = False
        src_parts.append(idx[np.repeat(np.arange(m), k)])
        dst_parts.append(idx[nbr[keep]])
    if not src_parts:
        return np.array([], dtype=int), np.array([], dtype=int)
    return np.concatenate(src_parts), np.concatenate(dst_parts)


def radius_edges(
    coords: np.ndarray,
    radius: float,
    sections: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Directed edges between all pairs within ``radius``, self excluded."""
    coords = np.asarray(coords, float)
    tree = cKDTree(coords)
    pairs = tree.query_pairs(r=float(radius), output_type="ndarray")
    if len(pairs) and sections is not None:
        sec = np.asarray(sections)
        pairs = pairs[sec[pairs[:, 0]] == sec[pairs[:, 1]]]
    if not len(pairs):
        return np.array([], dtype=int), np.array([], dtype=int)
    src = np.concatenate([pairs[:, 0], pairs[:, 1]])
    dst = np.concatenate([pairs[:, 1], pairs[:, 0]])
    return src, dst


def compute_neighborhood(
    coords: np.ndarray,
    labels: np.ndarray,
    *,
    categories: list[str] | None = None,
    mode: str = "knn",
    n_neighs: int = 10,
    radius: float | None = None,
    n_perms: int = 1000,
    sections: np.ndarray | None = None,
    seed: int = 0,
    progress_callback: Any = None,
) -> dict[str, Any]:
    """Neighborhood composition + permutation enrichment for one label column.

    Args:
        coords: (n_cells, 2) spatial coordinates.
        labels: per-cell type/cluster labels.
        categories: explicit axis order (defaults to sorted unique labels).
            Every label must appear in it.
        mode: 'knn' (default) or 'radius'.
        n_neighs: neighbors per cell for kNN mode.
        radius: neighborhood radius for radius mode (required there).
        n_perms: label permutations for the null.
        sections: optional per-cell section labels; edges and permutations
            never cross sections.
        seed: RNG seed.
        progress_callback: optional ``report(frac, message)``.

    Returns:
        dict with categories, composition / zscores / pvals / qvals / log2fc /
        expected / counts (all K x K nested lists, JSON-safe), and
        cell_composition, an (n_cells, K) float array of neighbor fractions
        (NaN rows for cells with no neighbors -- NOT JSON-safe, caller strips).
    """
    coords = np.asarray(coords, float)
    labels = np.asarray(labels).astype(str)
    n_cells = coords.shape[0]

    if categories is None:
        categories = [str(c) for c in np.unique(labels)]
    categories = [str(c) for c in categories]
    if len(categories) < 2:
        raise ValueError(
            f"Neighborhood enrichment needs at least 2 cell types; got {len(categories)}"
        )
    cat_index = {c: i for i, c in enumerate(categories)}
    unknown = sorted(set(labels) - set(categories))
    if unknown:
        raise ValueError(f"Labels not in categories: {unknown[:5]}")
    codes = np.array([cat_index[v] for v in labels], dtype=int)

    if mode == "knn":
        if n_neighs < 1:
            raise ValueError("n_neighs must be >= 1")
        src, dst = knn_edges(coords, n_neighs, sections=sections)
    elif mode == "radius":
        if radius is None or radius <= 0:
            raise ValueError("radius mode needs a positive radius")
        src, dst = radius_edges(coords, radius, sections=sections)
    else:
        raise ValueError(f"Unknown mode '{mode}' (expected 'knn' or 'radius')")
    if not len(src):
        raise ValueError("The spatial graph has no edges; increase radius or n_neighs")

    def _report(frac: float, message: str) -> None:
        if progress_callback is not None:
            progress_callback(frac, message)

    _report(0.0, "Counting neighbors")
    T = len(categories)

    # Per-cell neighbor-type counts -> composition fractions.
    per_cell = np.zeros((n_cells, T), dtype=float)
    np.add.at(per_cell, (src, codes[dst]), 1.0)
    deg = per_cell.sum(axis=1)
    has_nbrs = deg > 0
    cell_composition = np.full((n_cells, T), np.nan)
    cell_composition[has_nbrs] = per_cell[has_nbrs] / deg[has_nbrs, None]

    # Directed cluster x cluster mean composition (each cell weighted equally).
    composition = np.zeros((T, T), dtype=float)
    for a in range(T):
        rows = (codes == a) & has_nbrs
        if rows.any():
            composition[a] = cell_composition[rows].mean(axis=0)

    # Observed directed pair counts, then the label-shuffling null.
    def pair_counts(c: np.ndarray) -> np.ndarray:
        return np.bincount(c[src] * T + c[dst], minlength=T * T).astype(float)

    obs = pair_counts(codes)

    rng = np.random.default_rng(seed)
    if sections is not None:
        sec = np.asarray(sections)
        groups = [np.where(sec == s)[0] for s in np.unique(sec)]
    else:
        groups = None
    n_perms = max(1, int(n_perms))
    total = np.zeros(T * T)
    total_sq = np.zeros(T * T)
    n_ge = np.zeros(T * T)
    n_le = np.zeros(T * T)
    report_every = max(1, n_perms // 50)
    for t in range(n_perms):
        perm = section_permutation(n_cells, sections, rng, groups=groups)
        c = pair_counts(codes[perm])
        total += c
        total_sq += c * c
        n_ge += c >= obs
        n_le += c <= obs
        if (t + 1) % report_every == 0 or t == n_perms - 1:
            _report((t + 1) / n_perms, f"Permutation {t + 1}/{n_perms}")

    mean = total / n_perms
    var = np.maximum(total_sq / n_perms - mean * mean, 0.0)
    std = np.sqrt(var)
    safe_std = np.where(std > 0, std, 1.0)
    z = np.where(std > 0, (obs - mean) / safe_std, 0.0)
    p_hi = (n_ge + 1.0) / (n_perms + 1.0)
    p_lo = (n_le + 1.0) / (n_perms + 1.0)
    pvals = np.minimum(1.0, 2.0 * np.minimum(p_hi, p_lo))
    qvals = benjamini_hochberg(pvals)
    # +1 pseudocount keeps the ratio finite when either side is 0.
    log2fc = np.log2((obs + 1.0) / (mean + 1.0))
    mean_mat = mean.reshape(T, T)
    row_sum = mean_mat.sum(axis=1, keepdims=True)
    expected = np.where(row_sum > 0, mean_mat / np.where(row_sum > 0, row_sum, 1.0), 0.0)

    return {
        "categories": categories,
        "composition": composition.tolist(),
        "counts": obs.reshape(T, T).astype(int).tolist(),
        "zscores": z.reshape(T, T).tolist(),
        "pvals": pvals.reshape(T, T).tolist(),
        "qvals": qvals.reshape(T, T).tolist(),
        "log2fc": log2fc.reshape(T, T).tolist(),
        "expected": expected.tolist(),
        "cell_composition": cell_composition,
        "n_cells": int(n_cells),
        "n_edges": int(len(src)),
        "mean_neighbors": float(len(src) / n_cells) if n_cells else 0.0,
        "n_perms": int(n_perms),
    }
