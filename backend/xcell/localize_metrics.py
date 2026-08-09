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


def axis_direction(coords, scores) -> np.ndarray:
    """Unit direction along which ``scores`` varies most, by rank correlation.

    Maximizing ``corr(s, coords @ u)`` over unit ``u`` is a least-squares
    problem: the maximizer is proportional to the regression of the score on
    the coordinates. Doing it on ranks makes the answer a Spearman direction
    and immune to a few extreme values.

    Taking this from the **reference** is what makes the whole metric
    orientation-free — the user never has to declare which way is distal, and
    the sign convention comes from the data. (On the limb pair, +x turned out
    to be *proximal* in the stored coordinates even though the rendered view
    shows distal to the right.)
    """
    from scipy.stats import rankdata

    C = np.asarray(coords, dtype=float)[:, :2]
    s = np.asarray(scores, dtype=float).ravel()
    if len(C) < 3 or np.allclose(s, s[0]):
        return np.array([1.0, 0.0])

    Cr = np.column_stack([rankdata(C[:, 0]), rankdata(C[:, 1])]).astype(float)
    sr = rankdata(s).astype(float)
    Cr -= Cr.mean(axis=0)
    sr -= sr.mean()

    u, *_ = np.linalg.lstsq(Cr, sr, rcond=None)
    n = float(np.linalg.norm(u))
    return (u / n) if n > 0 else np.array([1.0, 0.0])


def axis_fidelity(ref_coords, ref_scores, pred_coords, pred_scores) -> dict[str, Any]:
    """Is the gradient real, and how does it compare with the best possible?

    Both sides are projected onto the direction found in the reference, so the
    two numbers are directly comparable. The reference value is the ceiling and
    is always returned with the prediction: an attenuated correlation is only
    interpretable next to what the reference itself achieves.
    """
    from scipy.stats import spearmanr

    ref = np.asarray(ref_coords, dtype=float)[:, :2]
    pred = np.asarray(pred_coords, dtype=float)[:, :2]
    rs = np.asarray(ref_scores, dtype=float).ravel()
    ps = np.asarray(pred_scores, dtype=float).ravel()

    ok = np.isfinite(pred).all(axis=1)
    pred, ps = pred[ok], ps[ok]

    u = axis_direction(ref, rs)

    def _rho(scores, coords):
        if len(coords) < 3 or np.allclose(scores, scores[0]):
            return np.nan
        return spearmanr(scores, coords @ u)[0]

    return {
        'reference': _f(_rho(rs, ref)),
        'prediction': _f(_rho(ps, pred)),
        'direction': [float(u[0]), float(u[1])],
    }


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


def evaluate_map(ref_coords, pred_coords, marker_sets: list[dict]) -> dict[str, Any]:
    """The whole panel for one predicted map.

    ``marker_sets`` entries are ``{'name', 'ref_scores', 'pred_scores'}``. The
    caller supplies scores rather than genes because only it can read a matrix.
    """
    return {
        'dispersion': dispersion(ref_coords, pred_coords),
        'occupancy': occupancy(ref_coords, pred_coords),
        'markers': [
            {
                'name': str(m['name']),
                'pattern': spatial_pattern_fidelity(
                    ref_coords, m['ref_scores'], pred_coords, m['pred_scores'],
                ),
                'axis': axis_fidelity(
                    ref_coords, m['ref_scores'], pred_coords, m['pred_scores'],
                ),
            }
            for m in marker_sets
        ],
    }


def _collapse_at(
    C: np.ndarray, s: np.ndarray, top: float, neighbors: int,
) -> dict[str, Any]:
    """One reading of "the population": the top ``top`` fraction by score."""
    from scipy.spatial import cKDTree

    n = len(C)
    n_pop = max(3, int(round(float(top) * n)))
    if n_pop > n:
        n_pop = n
    pop = C[np.argsort(-s, kind='stable')[:n_pop]]
    m = max(1, min(int(neighbors), len(pop) - 1))
    tree = cKDTree(pop)
    centroid = pop.mean(axis=0)

    d_centroid = float(np.atleast_1d(tree.query(centroid, k=m)[0]).mean())
    # Column 0 of each member's own query is itself, at distance 0.
    d_typical = float(np.median(tree.query(pop, k=m + 1)[0][:, 1:].mean(axis=1)))

    if d_centroid <= 0:
        risk, ratio = 0.0, 0.0          # the mean is sitting on the population
    else:
        risk = float(np.clip(1.0 - d_typical / d_centroid, 0.0, 1.0))
        ratio = d_centroid / d_typical if d_typical > 0 else None

    return {
        'risk': _f(risk),
        'distance_ratio': _f(ratio),
        'population_fraction': _f(n_pop / n),
        'centroid': [_f(centroid[0]), _f(centroid[1])],
        'n_population': int(n_pop),
    }


def mean_collapse_risk(
    coords, scores, *,
    fractions: tuple[float, ...] = (0.02, 0.05, 0.10, 0.20),
    neighbors: int = 5,
) -> dict[str, Any]:
    """Would averaging this population's positions land outside the population?

    Reference-only, and that is the point: ``weighted_mean`` predicts, for every
    query cell of a type, roughly the centroid of that type's reference members.
    Whether that centroid is a place the type actually occupies depends on the
    reference's geometry and on nothing else — no query, no parameters, no run.
    So the warning can be raised while the parameters are still being chosen
    rather than after a map has been made and believed.

    **The mean of a ring is its hole.** The population is the top fraction of
    cells by score, and its centroid is where the estimator would send every one
    of them. The question is how far that centroid sits from the population,
    measured **in the population's own units**: against how far its members sit
    from each other. A ratio of 1 means the mean is an ordinary place within the
    population; 8 means it is eight times farther out than the population's own
    spacing, which is a hole.

    **"The population" is swept, not assumed**, and that is not a refinement —
    a single fraction misses the case this exists for. Real markers differ
    enormously in prevalence: on the E11.5 limb the skin genes are detected in
    7% of spots, so the top 20% is mostly interior tissue with trace expression,
    the "population" fills the bud, and its mean is comfortably inside it —
    risk 0.14, silent, on a map Phase 1 had already measured as inverted
    (pattern fidelity -0.11). At the top 2%, which is the epidermis proper, the
    same reference scores 0.88. So each fraction in ``fractions`` is scored and
    the worst is reported, with the fraction that produced it: for *some*
    reasonable reading of this population, its mean lands outside it. Measured
    over 118 draws of pure noise the swept maximum never reached the warning
    threshold.

    Two other formulations were tried and are worth naming, because both look
    right and neither is:

    - *How many of the cells around the centroid are population members?* Right
      in every case and far too noisy — about 8 members expected among 40
      neighbours, so it strays past the warning threshold on an ordinary
      scattered population roughly 1 time in 60. A user with a dozen gene sets
      would see a spurious warning most sessions.
    - *What is the mean rank-score of the tissue around the centroid, against
      chance?* Low-variance, but it silently depends on how much of the tissue
      the population covers. For a realistic marker — high in the population,
      flat noise everywhere else — the background's mean rank is ``(1-f)/2``,
      not 0, so the risk saturates near 0.25 for a population covering a quarter
      of the tissue and the warning never fires. Measured on a 20x20 grid with a
      rim population: 0.34, below the threshold, on geometry that plainly
      collapses.

    A distance ratio has neither problem: it is scale-free, it averages over
    ``neighbors`` distances rather than counting rare events, and it never
    consults the background at all. Measured across 59 seeds a scattered
    population peaks at 0.20, against 0.96 for a ring, 0.94 for two patches and
    0.79 for a one-spot-wide perimeter.

    ``neighbors`` is 5 because it decides how far "the population's own
    spacing" reaches, and a *thin* population — an epidermis is one or two spots
    wide — has only its two neighbours along the band. Averaging over 25 instead
    walks a quarter of the way around the ring, inflates that spacing, and hides
    the hole: on a 20x20 grid with a rim population the risk falls from 0.85 to
    0.47, i.e. from firing to silent. Larger values are quieter on noise (0.06
    against 0.20) but buy it by missing exactly the shape this exists for; 5
    keeps every measured case at least 1.6x clear of the warning threshold in
    both directions.

    What it does **not** catch: a population scattered through a *ring-shaped
    tissue*, where the mean is off-tissue but not off-population. That one needs
    an actual run, and :func:`dispersion`'s ``frac_outside`` is what reports it.

    Args:
        coords: ``(n, 2)`` reference coordinates.
        scores: ``(n,)`` marker score per reference cell.
        fractions: the readings of "the population" to score, worst reported.
        neighbors: how many population members each distance averages over.

    Returns:
        risk: ``1 - typical/centroid`` distance, clipped to ``[0, 1]``, at the
            worst fraction. 0 means the mean lands among the population; 0.5
            means twice as far out as the population's own spacing; 1 means far
            outside it. ``None`` when there is nothing to measure.
        distance_ratio: the raw ratio, for the UI to quote — "the mean sits 8
            times farther from these cells than they sit from each other".
        population_fraction: the fraction that produced that worst case, which
            is what makes the number self-explaining — 0.02 says it is the
            population's core that forms a ring, not its diffuse tail.
        centroid: ``[x, y]`` — where the estimator would put every one of them.
        n_population: how many reference cells that population has.
    """
    C = np.asarray(coords, dtype=float)[:, :2]
    s = np.asarray(scores, dtype=float).ravel()
    n = len(C)

    # Under ~20 cells a neighbourhood is the whole tissue and none of this
    # means anything; a constant score has no population to speak of.
    if n < 20 or len(s) != n or not np.isfinite(s).any() or np.allclose(s, s[0]):
        return {
            'risk': None, 'distance_ratio': None,
            'population_fraction': None, 'centroid': None, 'n_population': 0,
        }

    scored = [_collapse_at(C, s, t, neighbors) for t in fractions]
    return max(scored, key=lambda d: d['risk'] or 0.0)
