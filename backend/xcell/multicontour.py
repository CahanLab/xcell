"""Multi-contour tissue annotation.

Score several gene-set modules spatially, binarize each to "high / not-high",
and fuse them into a single tissue label per spot. Spots that are high in more
than one module are resolved by looking at their spatial neighbors, ranking
those by Euclidean distance in ``X_pca``, and taking the majority unambiguous
label among the nearest ``profile_k``. Ties or empty neighbor pools stay
``unassigned``. Single pass: resolution uses the original unambiguous labels.

These are pure functions (no I/O); the adaptor orchestrates them.
"""

from __future__ import annotations

import math

import numpy as np

from xcell.adaptor import _contour_score_field


def _spatial_key(adata):
    for k in ("spatial", "X_spatial"):
        if k in adata.obsm:
            return k
    raise ValueError("No spatial coordinates found")


def _expr_dict(adata, genes):
    out = {}
    for g in genes:
        x = adata[:, g].X
        out[g] = x.toarray().ravel() if hasattr(x, "toarray") else np.asarray(x).ravel()
    return out


def auto_cutoff(thresholds):
    """Default high cutoff = top band (highest threshold)."""
    thresholds = np.asarray(thresholds)
    return float(thresholds[-1]) if thresholds.size else 0.0


def score_module(adata, genes, contour_levels=3, log_transform=True,
                 clip_percentiles=(1, 99), grid_res=200, smooth_sigma=2.0,
                 sections=None):
    """Score one gene-set module: continuous score, threshold bands, histogram.

    When ``sections`` (an (n,) array of section labels) is given, the spatial
    interpolation runs per section so expression does not bleed across the gap
    between sections.

    Returns a dict with: score (n,), bands (n,), thresholds, band_values,
    histogram (count per band value), and auto_cutoff (top band).
    """
    coords = np.asarray(adata.obsm[_spatial_key(adata)])
    score, vmax = _contour_score_field(
        coords, _expr_dict(adata, genes), log_transform,
        tuple(clip_percentiles), grid_res, smooth_sigma, sections=sections)

    thresholds = np.linspace(0, vmax, contour_levels + 2)[1:-1]
    band_values = np.unique(np.concatenate(([0.0], thresholds)))

    bands = np.zeros(adata.n_obs, dtype=float)
    for t in sorted(thresholds):
        bands[score >= t] = t

    histogram = [int(np.sum(bands == bv)) for bv in band_values]
    return {
        "score": score,
        "bands": bands,
        "thresholds": thresholds,
        "band_values": band_values,
        "histogram": histogram,
        "auto_cutoff": auto_cutoff(thresholds),
    }


def binarize(bands, cutoff):
    """Boolean 'high' mask: band >= cutoff."""
    return np.asarray(bands) >= cutoff


def _neighbor_lists(spatial_conn, coords, n, k_fallback=15):
    """Adjacency as a list-of-lists.

    Uses ``spatial_conn`` (scipy sparse or dense adjacency) if given, else builds
    a coordinate kNN graph from ``coords``.
    """
    if spatial_conn is not None:
        conn = spatial_conn
        if hasattr(conn, "tocsr"):
            conn = conn.tocsr()
            return [list(conn.indices[conn.indptr[i]:conn.indptr[i + 1]]) for i in range(n)]
        conn = np.asarray(conn)
        return [list(np.where(conn[i] != 0)[0]) for i in range(n)]

    from scipy.spatial import cKDTree
    coords = np.asarray(coords)
    tree = cKDTree(coords)
    k = min(k_fallback + 1, n)
    _, idx = tree.query(coords, k=k)
    idx = np.atleast_2d(idx)
    return [[int(j) for j in row if j != i] for i, row in enumerate(idx)]


def assign_tissue(highs, adata, profile_k, spatial_conn, pca, coords, sections=None):
    """Fuse per-module high masks into a single tissue label per spot.

    Args:
        highs: dict module_name -> bool mask (n,). Names define the label space.
        adata: unused (kept for signature symmetry / future use).
        profile_k: number of nearest unambiguous spatial neighbors to vote over.
        spatial_conn: spatial adjacency (sparse/dense) or None to build from coords.
        pca: (n, d) profile coordinates used to rank neighbors (e.g. X_pca).
        coords: (n, 2) spatial coordinates (used only for the kNN fallback).
        sections: optional (n,) section labels; conflict resolution then only
            votes with neighbors from the same section.

    Returns:
        (labels, status): object arrays of length n. status is one of
        'single', 'resolved', 'unassigned'.
    """
    names = list(highs.keys())
    H = np.column_stack([np.asarray(highs[m], dtype=bool) for m in names])  # (n, M)
    n = H.shape[0]
    counts = H.sum(axis=1)

    labels = np.array(["unassigned"] * n, dtype=object)
    status = np.array(["unassigned"] * n, dtype=object)

    single = counts == 1
    if single.any():
        single_idx_of = np.argmax(H, axis=1)
        names_arr = np.array(names, dtype=object)
        labels[single] = names_arr[single_idx_of[single]]
        status[single] = "single"

    conflict_ids = np.where(counts >= 2)[0]
    if conflict_ids.size:
        pca = np.asarray(pca)
        sections = np.asarray(sections) if sections is not None else None
        neigh = _neighbor_lists(spatial_conn, coords, n)
        for s in conflict_ids:
            cands = [
                j for j in neigh[s]
                if single[j] and (sections is None or sections[j] == sections[s])
            ]
            if not cands:
                continue
            d = np.linalg.norm(pca[cands] - pca[s], axis=1)
            nearest = [cands[i] for i in np.argsort(d)[:profile_k]]
            votes: dict[str, int] = {}
            for j in nearest:
                votes[labels[j]] = votes.get(labels[j], 0) + 1
            top = max(votes.values())
            winners = [lab for lab, c in votes.items() if c == top]
            if len(winners) == 1:
                labels[s] = winners[0]
                status[s] = "resolved"
            # tie -> leave unassigned
    return labels, status


def suggest_grid_res(n_spots):
    """Grid resolution ~ sqrt(n_spots), clamped to [50, 600]."""
    return int(min(600, max(50, round(math.sqrt(max(1, n_spots))))))


def suggest_smooth_sigma(coords, grid_res):
    """~2 grid-pixels at the median spot spacing; clamped to [1, 6]."""
    from scipy.spatial import cKDTree

    coords = np.asarray(coords)
    if coords.shape[0] < 2:
        return 2.0
    d, _ = cKDTree(coords).query(coords, k=2)
    median_spacing = float(np.median(d[:, 1]))
    extent = float(max(np.ptp(coords[:, 0]), np.ptp(coords[:, 1]))) or 1.0
    px = extent / grid_res
    sigma = 2.0 * (median_spacing / px) if px > 0 else 2.0
    return float(min(6.0, max(1.0, sigma)))
