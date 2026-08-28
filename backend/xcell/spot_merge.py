"""Merging neighbouring ST spots into single large cells.

A spot is a patch of tissue, not a cell. Where the cell is much larger than the
spot — a hypertrophic chondrocyte against a Visium HD bin — one cell is spread
across several spots, each holding a fraction of its transcripts, so every spot
in the region looks shallow and none is a usable profile of the cell.

This module puts the pieces back: it groups neighbouring spots up to the size of
one cell so their counts can be summed. Geometry drives the grouping, because
the low-count profiles that motivate merging are exactly the ones too noisy to
decide it — a correlation between two 40-count profiles is largely sampling
noise, and ranking by it gives a different answer on every run. Similarity
enters only as a veto.

Pure: arrays in, arrays out. Knows nothing of AnnData.
"""
from __future__ import annotations

import heapq
import math
from dataclasses import dataclass, field

import numpy as np


def estimate_pitch_um(coords: np.ndarray) -> float:
    """Spot spacing, as the median nearest-neighbour distance.

    Measured rather than assumed: it costs one KD-tree query, and it turns a
    wrong µm-per-unit into an obviously absurd number the user can see instead
    of a silently wrong size cap on every merge.
    """
    if len(coords) < 2:
        raise ValueError("Estimating spot pitch needs at least two spots.")
    from scipy.spatial import cKDTree

    # k=2 because a point's own nearest neighbour is itself, at distance 0.
    distances, _ = cKDTree(coords).query(coords, k=2)
    return float(np.median(distances[:, 1]))


def max_spots_for(diameter_um: float, pitch_um: float) -> int:
    """How many lattice points a disc of this diameter covers."""
    area_ratio = math.pi * (diameter_um / 2.0) ** 2 / (pitch_um ** 2)
    return max(1, int(math.ceil(area_ratio)))


@dataclass
class MergeParams:
    """What the caller controls. Diameter is the one with biological meaning."""
    max_diameter_um: float = 40.0
    max_spots: int | None = None          # None -> derived from the pitch
    min_correlation: float = 0.15
    eligibility: str = "all"              # 'all' | 'min_counts' | 'quantile'
    min_counts: float | None = None
    quantile: float | None = None


@dataclass
class MergeResult:
    labels: np.ndarray                    # group id per input spot
    n_groups: int
    pitch_um: float
    max_spots: int
    size_histogram: dict[int, int] = field(default_factory=dict)
    n_vetoed: int = 0
    correlation_quantiles: dict[str, float] = field(default_factory=dict)


def _neighbour_pairs(coords: np.ndarray) -> set[tuple[int, int]]:
    """Adjacent spot pairs, by Delaunay triangulation.

    Delaunay rather than a fixed radius: it adapts to irregular spacing without
    being told the pitch. It needs a non-degenerate 2-D simplex, so collinear
    or tiny inputs — a single row of spots along a slide edge is neither exotic
    nor an error — fall back to all pairs, which the distance cap then trims.
    """
    from scipy.spatial import Delaunay, QhullError

    if len(coords) >= 4:
        try:
            simplices = Delaunay(coords).simplices
        except (QhullError, ValueError):
            simplices = None
        if simplices is not None:
            pairs = set()
            for simplex in simplices:
                for i in range(len(simplex)):
                    for j in range(i + 1, len(simplex)):
                        a, b = int(simplex[i]), int(simplex[j])
                        pairs.add((min(a, b), max(a, b)))
            if pairs:
                return pairs
    return {(i, j) for i in range(len(coords)) for j in range(i + 1, len(coords))}


def _correlation(a: np.ndarray, b: np.ndarray) -> float:
    """Pearson r between two profiles; 0 when either is flat."""
    a_c, b_c = a - a.mean(), b - b.mean()
    denom = float(np.linalg.norm(a_c) * np.linalg.norm(b_c))
    if denom < 1e-12:
        return 0.0
    return float(np.dot(a_c, b_c) / denom)


def eligible_mask(counts: np.ndarray, params: MergeParams) -> np.ndarray:
    """Which spots in the region may be merged.

    The region is already the user's statement of intent, so 'all' is the
    default and gives one comparable population. The other modes exist for
    leaving well-covered spots alone — at the price of an output whose count
    distributions differ by construction.
    """
    depth = np.asarray(counts, dtype=float).sum(axis=1)
    mode = params.eligibility
    if mode == "all":
        return np.ones(len(depth), dtype=bool)
    if mode == "min_counts":
        if params.min_counts is None:
            raise ValueError("eligibility 'min_counts' needs a min_counts value.")
        return depth < float(params.min_counts)
    if mode == "quantile":
        if params.quantile is None:
            raise ValueError("eligibility 'quantile' needs a quantile value.")
        if not 0.0 < params.quantile <= 1.0:
            raise ValueError("quantile must be in (0, 1].")
        return depth <= np.quantile(depth, params.quantile)
    raise ValueError(
        f"Unknown eligibility '{mode}'. Use 'all', 'min_counts' or 'quantile'."
    )


def merge_spots(
    coords: np.ndarray,
    counts: np.ndarray,
    normalized: np.ndarray,
    params: MergeParams,
) -> MergeResult:
    """Group neighbouring spots into cell-sized clusters.

    Greedy agglomeration over the neighbour graph: repeatedly join the closest
    adjacent pair whose merged footprint still fits inside one cell and whose
    profiles are not clearly different. Constrained hierarchical clustering with
    a diameter cap — deterministic, and it never lets a noisy low-count
    correlation decide *which* spots join, only whether a join is refused.

    Args:
        coords: (n, 2) spot positions in microns.
        counts: (n, g) raw counts — summed, so integer-like.
        normalized: (n, g) CPM+log1p profiles — correlated, never summed.
        params: see MergeParams.

    Returns:
        MergeResult; ``labels`` gives each input spot its group id.
    """
    n = len(coords)
    pitch = estimate_pitch_um(coords) if n >= 2 else params.max_diameter_um
    max_spots = params.max_spots or max_spots_for(params.max_diameter_um, pitch)

    parent = list(range(n))
    members: dict[int, list[int]] = {i: [i] for i in range(n)}
    group_counts = np.asarray(counts, dtype=float).copy()
    group_norm = np.asarray(normalized, dtype=float).copy()

    def find(i: int) -> int:
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    # Tie-break on the pair's own indices so equal distances — the common case
    # on a regular lattice — resolve the same way on every run.
    eligible = eligible_mask(counts, params)
    heap = []
    for a, b in _neighbour_pairs(coords):
        if not (eligible[a] and eligible[b]):
            continue
        d = float(np.linalg.norm(coords[a] - coords[b]))
        if d <= params.max_diameter_um:
            heap.append((d, a, b))
    heapq.heapify(heap)

    n_vetoed = 0
    correlations: list[float] = []

    while heap:
        _, ra, rb = heapq.heappop(heap)
        ga, gb = find(ra), find(rb)
        if ga == gb:
            continue

        joined = members[ga] + members[gb]
        if len(joined) > max_spots:
            continue
        pts = coords[joined]
        spread = float(np.linalg.norm(pts[:, None, :] - pts[None, :, :], axis=-1).max())
        if spread > params.max_diameter_um:
            continue

        r = _correlation(group_norm[ga], group_norm[gb])
        correlations.append(r)
        if r < params.min_correlation:
            n_vetoed += 1
            continue

        # Keep the lower root so labels stay tied to source order.
        keep, drop = (ga, gb) if ga < gb else (gb, ga)
        parent[drop] = keep
        members[keep] = sorted(joined)
        del members[drop]
        group_counts[keep] = group_counts[ga] + group_counts[gb]
        # The merged profile is the summed counts renormalized, not the mean of
        # two profiles: the veto must judge the group as it will actually exist.
        depth = float(group_counts[keep].sum()) or 1.0
        group_norm[keep] = np.log1p(group_counts[keep] / depth * 1e4)

    roots = sorted({find(i) for i in range(n)})
    remap = {root: gid for gid, root in enumerate(roots)}
    labels = np.array([remap[find(i)] for i in range(n)], dtype=int)

    sizes = np.bincount(labels)
    histogram = {int(size): int(count) for size, count
                 in zip(*np.unique(sizes, return_counts=True))}

    quantiles: dict[str, float] = {}
    if correlations:
        p10, p50, p90 = np.percentile(correlations, [10, 50, 90])
        quantiles = {"p10": float(p10), "p50": float(p50), "p90": float(p90)}

    return MergeResult(
        labels=labels,
        n_groups=len(roots),
        pitch_um=pitch,
        max_spots=max_spots,
        size_histogram=histogram,
        n_vetoed=n_vetoed,
        correlation_quantiles=quantiles,
    )
