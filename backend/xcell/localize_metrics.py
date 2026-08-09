"""Is a predicted spatial map any good?

Pure: arrays in, plain dicts out. No AnnData, no adaptor import — the adaptor
pulls the matrices out, calls in here, and hands the result to a route.

Every metric is **rank-based**, for a reason that is not stylistic. The
reference is spot-resolved and the query is dissociated cells, so their
expression scales are not comparable; and a method that shrinks the map changes
the coordinate scale without changing what it claims. Ranking both sides makes
each metric invariant to any monotone rescale, which is exactly the invariance
the comparison needs: an affine rescale of the predictions cannot change a rank
correlation, so a metric that moved under one would be measuring the wrong
thing.

The four questions, and what each catches:

``spatial_pattern_fidelity``
    Does a cell type land where it lives? The headline metric — it is what
    catches an epidermis predicted into the middle of the bud.
``axis_fidelity``
    Are the gradients real? Reported against the reference's own value, which
    is the ceiling, never as a bare number.
``dispersion``
    Does the map fill the tissue? Catches both a collapse to 0.15 of the area
    and an overshoot to 2.93.
``occupancy``
    Is it piling up? Catches a better-recovered spot absorbing many cells.
"""
from __future__ import annotations

from typing import Any

import numpy as np


def _f(value: Any) -> float | None:
    """JSON-safe float: None for anything not finite, per the house rule."""
    try:
        v = float(value)
    except (TypeError, ValueError):
        return None
    return v if np.isfinite(v) else None


def _inner_hull_area(pts: np.ndarray, inner: float) -> float | None:
    """Convex-hull area of the innermost ``inner`` fraction of points.

    The plain hull is set by its most extreme member, so a single stray
    prediction can multiply it. Trimming by distance from the median first
    makes the number describe the bulk of the map. The median rather than the
    mean, because the mean is itself dragged by the outlier being trimmed.
    """
    from scipy.spatial import ConvexHull, QhullError

    pts = np.asarray(pts, dtype=float)
    pts = pts[np.isfinite(pts).all(axis=1)]
    if len(pts) < 3:
        return None
    d = np.linalg.norm(pts - np.median(pts, axis=0), axis=1)
    keep = pts[d <= np.quantile(d, inner)]
    if len(keep) < 3:
        return None
    try:
        return float(ConvexHull(keep).volume)   # 'volume' is area in 2-D
    except QhullError:
        return None                             # collinear or degenerate


def dispersion(ref_coords, pred_coords, *, inner: float = 0.95) -> dict[str, Any]:
    """How much of the tissue the predictions actually occupy.

    ``area_ratio`` 1.0 is right; well under 1 is the shrinkage that averaging
    neighbours produces, well over 1 is an overshoot. ``frac_outside`` is
    reported separately because a rescale can reach the right area by pushing
    cells out of the tissue, which the ratio alone would hide.
    """
    ref = np.asarray(ref_coords, dtype=float)[:, :2]
    pred = np.asarray(pred_coords, dtype=float)[:, :2]
    pred = pred[np.isfinite(pred).all(axis=1)]

    ref_area = _inner_hull_area(ref, inner)
    pred_area = _inner_hull_area(pred, inner)
    ratio = (pred_area / ref_area) if (ref_area and pred_area and ref_area > 0) else None

    ref_sd = ref.std(axis=0)
    pred_sd = pred.std(axis=0) if len(pred) else np.array([np.nan, np.nan])

    lo, hi = ref.min(axis=0), ref.max(axis=0)
    outside = (
        float(np.mean(np.any((pred < lo) | (pred > hi), axis=1))) if len(pred) else 0.0
    )

    return {
        'area_ratio': _f(ratio),
        'std_ratio_x': _f(pred_sd[0] / ref_sd[0]) if ref_sd[0] > 0 else None,
        'std_ratio_y': _f(pred_sd[1] / ref_sd[1]) if ref_sd[1] > 0 else None,
        'frac_outside': _f(outside),
    }


def _rank_weights(scores: np.ndarray) -> np.ndarray:
    """Scores as ranks in [0, 1].

    Weighting a density by raw score would let one platform's dynamic range
    decide the answer, and a z-scored matrix can go negative, which a density
    cannot use. Ranks are non-negative, bounded, and identical under any
    monotone rescale of either side.
    """
    from scipy.stats import rankdata

    s = np.asarray(scores, dtype=float).ravel()
    if len(s) == 0:
        return s
    return (rankdata(s) - 1.0) / max(len(s) - 1, 1)


def _mean_score_field(coords, weights, edges_x, edges_y, sigma: float = 1.0):
    """Mean marker score per grid bin, plus how many cells backed each bin.

    The obvious formulation — a marker-*weighted* density — is confounded by
    the cell density underneath it. Both maps then contain the same "where are
    there cells at all" signal, which dominates: two unrelated markers score
    +0.33 and a fully inverted one barely reaches -0.07. Dividing the weighted
    sum by the count isolates "is this marker high here" from "are there many
    cells here", which is the question actually being asked.

    Smoothing numerator and denominator separately before dividing is a
    Nadaraya-Watson estimate; it keeps a bin holding one cell from swinging the
    field, without discarding it.
    """
    from scipy.ndimage import gaussian_filter

    num, _, _ = np.histogram2d(
        coords[:, 0], coords[:, 1], bins=[edges_x, edges_y], weights=weights,
    )
    den, _, _ = np.histogram2d(coords[:, 0], coords[:, 1], bins=[edges_x, edges_y])
    num = gaussian_filter(num, sigma=sigma, mode='nearest')
    den_s = gaussian_filter(den, sigma=sigma, mode='nearest')
    with np.errstate(invalid='ignore', divide='ignore'):
        field = np.where(den_s > 1e-12, num / np.maximum(den_s, 1e-12), np.nan)
    return field, den


def spatial_pattern_fidelity(
    ref_coords, ref_scores, pred_coords, pred_scores, *, bins: int = 32,
) -> dict[str, Any]:
    """Does this cell type land where it lives?

    Both sides are binned on the **reference's** grid and compared as density
    maps, so the question asked is "is the marker-weighted density in the same
    places", not "did any individual cell go to the right spot" — which is not
    answerable and not what a user is looking at.

    Correlation is Spearman over the bins the reference actually covers; empty
    tissue outside the bud would otherwise dominate the count and inflate every
    score toward agreement-on-nothing.
    """
    from scipy.stats import spearmanr

    ref = np.asarray(ref_coords, dtype=float)[:, :2]
    pred = np.asarray(pred_coords, dtype=float)[:, :2]
    rs = np.asarray(ref_scores, dtype=float).ravel()
    ps = np.asarray(pred_scores, dtype=float).ravel()

    ok = np.isfinite(pred).all(axis=1)
    pred, ps = pred[ok], ps[ok]
    if len(pred) < 3 or len(ref) < 3:
        return {'correlation': None, 'n_bins_compared': 0}

    edges_x = np.linspace(ref[:, 0].min(), ref[:, 0].max(), bins + 1)
    edges_y = np.linspace(ref[:, 1].min(), ref[:, 1].max(), bins + 1)

    ref_field, ref_n = _mean_score_field(ref, _rank_weights(rs), edges_x, edges_y)
    pred_field, pred_n = _mean_score_field(pred, _rank_weights(ps), edges_x, edges_y)

    # Compare only where both sides put cells. A bin the reference never
    # covered carries no claim about where the cell type belongs, and one the
    # prediction never reached has no estimate to offer.
    usable = (ref_n > 0) & (pred_n > 0) & np.isfinite(ref_field) & np.isfinite(pred_field)
    if usable.sum() < 8:
        return {'correlation': None, 'n_bins_compared': int(usable.sum())}

    a, b = ref_field[usable], pred_field[usable]
    if np.allclose(a, a[0]) or np.allclose(b, b[0]):
        return {'correlation': None, 'n_bins_compared': int(usable.sum())}

    rho, _ = spearmanr(a, b)
    return {'correlation': _f(rho), 'n_bins_compared': int(usable.sum())}


def occupancy(ref_coords, pred_coords) -> dict[str, Any]:
    """How many distinct reference locations the predictions actually use.

    Computed from coordinates rather than from a method's own assignment, so it
    applies to every aggregation rather than only to the matching ones.
    """
    from scipy.spatial import cKDTree

    ref = np.asarray(ref_coords, dtype=float)[:, :2]
    pred = np.asarray(pred_coords, dtype=float)[:, :2]
    pred = pred[np.isfinite(pred).all(axis=1)]
    if len(pred) == 0 or len(ref) == 0:
        return {'n_distinct': 0, 'max_per_spot': 0, 'effective_n': 0.0, 'n_pred': 0}

    _, idx = cKDTree(ref).query(pred, k=1)
    counts = np.bincount(np.asarray(idx).ravel(), minlength=len(ref))
    nz = counts[counts > 0]
    p = nz / nz.sum()
    # Exponential of the entropy: the number of *equally used* spots that would
    # give this much spread. Falls when a few spots absorb most cells, which a
    # distinct count cannot see.
    effective = float(np.exp(-(p * np.log(p)).sum()))

    return {
        'n_distinct': int(len(nz)),
        'max_per_spot': int(nz.max()),
        'effective_n': _f(effective),
        'n_pred': int(len(pred)),
    }
