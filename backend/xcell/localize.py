"""Predicting where a dissociated cell came from, using a spatial reference.

Pure: arrays in, arrays and plain dicts out. No AnnData, no adaptor import — the
adaptor pulls the matrices out, calls in here, and writes the result back.

The method is kNN projection: find each query cell's nearest neighbours among
the spatially-resolved cells (in expression space) and aggregate their
coordinates. It is transparent and parameter-interpretable, and — the reason it
is the right first method — it fails in ways that can be measured rather than
ways that hide inside a fitted model.

**The failure mode everything here is shaped around.** Averaging the coordinates
of k transcriptionally-similar cells is only meaningful if those cells are
spatially co-located. A cell type dispersed through the tissue has neighbours
scattered everywhere, and their centroid lands in the middle of the tissue —
possibly somewhere that cell type never occurs. A type present in two distant
patches is worse: the mean lands squarely in the gap. Both produce a smooth,
convincing, wrong map, and nothing about the coordinate itself gives it away.

So every projection carries two independent scores:

``similarity``
    Does this cell resemble anything in the reference at all? A cell type absent
    from the spatial data is still placed *somewhere*; this is what says so.
``confidence``
    Do its neighbours agree on a location? The weighted spatial spread of the k
    neighbours, normalized by the reference tissue's own radius, so 1 means they
    sat on one spot and 0 means they were as scattered as the whole tissue.

They are orthogonal, and a prediction needs both to be trusted.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

import numpy as np

TRANSFORMS = ('none', 'log1p', 'zscore', 'rank')
METRICS = ('correlation', 'cosine', 'euclidean')
AGGREGATIONS = (
    'weighted_mean', 'median', 'best_match', 'densest', 'injective', 'transport',
)

# Query cells per chunk. Keeps the similarity block bounded regardless of how
# many cells are being mapped, and gives progress something to report.
_CHUNK = 2048

# ...and bounded regardless of how many reference spots there are, which the row
# count alone does not do: 2,048 rows against a 100,000-spot reference is a
# 820 MB transient block, for every aggregation rather than only the assigning
# one. 2e7 float32 entries is 80 MB.
_MAX_BLOCK_ENTRIES = 20_000_000

# Below this, the dense solver is affordable and exactly optimal, so it stays.
# 4e7 entries is ~480 MB peak — scipy converts its input to float64 internally,
# so a float32 matrix costs 12 bytes per entry, not 4 (measured: +210 MB on top
# of a 105 MB array).
#
# Set by what already works rather than by a round number: the E11.5 limb pair
# is 2.6e7 entries and runs dense today, so a lower ceiling would push a
# currently-exact case onto the approximate path — a regression dressed as a
# memory saving. The candidate path takes over precisely where the alternative
# used to be a refusal.
_DENSE_MAX_ENTRIES = 40_000_000

# Candidates per cell for the sparse path. Measured on the E11.5 limb pair
# (2,683 x 9,773): 128 candidates put the objective 0.18% below optimal with 87%
# of cells on the identical spot, using 4.1 MB against 315 MB — and the
# map-quality metrics cannot tell the two apart at all.
_CANDIDATES = 128

# The candidate graph's own ceiling: n_query x candidates edges, each an index
# plus a float64 cost. 5e7 edges is ~600 MB, so the refusal now fires around
# 400,000 cells rather than around 10,000.
_MAX_CANDIDATE_EDGES = 50_000_000

# Sinkhorn holds the cost matrix, one workspace of the same shape and the
# coupling, all float64 for numerical safety — about three arrays, so 4e7
# entries is ~960 MB at the ceiling and the E11.5 limb pair (2.6e7) costs
# ~630 MB. Set by what already works, as the dense assignment ceiling was: that
# pair must keep running against the full reference, because a subsampled
# answer is a different answer rather than a slower one.
_TRANSPORT_MAX_ENTRIES = 40_000_000


@dataclass
class Projection:
    """Predicted coordinates plus everything needed to judge them."""

    coords: np.ndarray               # (n_query, 2), NaN where below min_confidence
    confidence: np.ndarray           # (n_query,) in [0, 1]
    similarity: np.ndarray           # (n_query,) mean similarity to the k neighbours
    neighbor_indices: np.ndarray     # (n_query, k) into the reference
    sections: np.ndarray | None      # (n_query,) assigned section label, or None
    n_unplaced: int
    # How an injective assignment was solved, and over how many candidates.
    # 'exact' is provably optimal; 'candidate' is near-optimal, and the
    # difference is reported rather than assumed away.
    assignment: str | None = None
    n_candidates: int | None = None
    # Which reference the coupling was solved against, and how well it
    # converged. A subsampled run is a different answer from a full one and
    # must never be readable as the same; a run that spent its whole iteration
    # budget has not satisfied the constraint that justifies the method.
    transport: str | None = None
    n_ref_used: int | None = None
    sinkhorn_iterations: int | None = None
    marginal_error: float | None = None


# --- transforms -----------------------------------------------------------

def transform_expression(X: np.ndarray, method: str) -> np.ndarray:
    """Prepare one dataset's matrix for cross-dataset comparison.

    Applied to each dataset **independently** — that is the whole point.
    ``zscore`` standardizes each gene within its own dataset, which is what
    removes platform-level per-gene capture efficiency; ``rank`` goes further and
    survives any monotone platform difference.
    """
    if method not in TRANSFORMS:
        raise ValueError(f"transform must be one of {list(TRANSFORMS)}, got '{method}'")
    X = np.asarray(X, dtype=np.float32)
    if method == 'none':
        return X
    if method == 'log1p':
        return np.log1p(np.maximum(X, 0.0))
    if method == 'zscore':
        mean = X.mean(axis=0)
        std = X.std(axis=0)
        # Two failure modes, one guard. A gene absent from one platform is
        # constant there, and dividing by 0 would make the column NaN and poison
        # every distance. A gene that varies only by measurement noise is worse
        # than useless: standardizing rescales that noise to unit variance, so
        # it counts as much as a real spatial gradient. Flooring the divisor at
        # a fraction of the matrix's overall scale leaves genuine variation
        # untouched and collapses negligible variation towards zero.
        floor = 1e-3 * float(X.std() or 1.0)
        std = np.maximum(std, max(floor, 1e-8))
        return (X - mean) / std
    # rank: within each cell, so a cell's profile is its gene ordering.
    order = np.argsort(X, axis=1)
    ranks = np.empty_like(order, dtype=np.float32)
    rows = np.arange(X.shape[0])[:, None]
    ranks[rows, order] = np.arange(X.shape[1], dtype=np.float32)
    return ranks


# --- similarity -----------------------------------------------------------

def _prepare_for_metric(X: np.ndarray, metric: str) -> np.ndarray:
    """Row-transform so that a plain dot product gives the wanted similarity."""
    if metric == 'euclidean':
        return X
    if metric == 'correlation':
        # Pearson between two cells == cosine after centering each cell.
        X = X - X.mean(axis=1, keepdims=True)
    norms = np.linalg.norm(X, axis=1, keepdims=True)
    return X / np.where(norms > 1e-12, norms, 1.0)


def _similarity_block(q: np.ndarray, ref: np.ndarray, metric: str) -> np.ndarray:
    """(n_q, n_ref) similarity, higher = more alike, for any supported metric."""
    if metric == 'euclidean':
        # -distance, so "larger is better" holds for every metric.
        d2 = (
            (q ** 2).sum(axis=1)[:, None]
            - 2.0 * (q @ ref.T)
            + (ref ** 2).sum(axis=1)[None, :]
        )
        return -np.sqrt(np.maximum(d2, 0.0))
    return q @ ref.T


# --- aggregation ----------------------------------------------------------

def _weights(sims: np.ndarray) -> np.ndarray:
    """Similarities → non-negative weights summing to 1, per row.

    Shifted so the worst of the k neighbours contributes nothing rather than
    negatively; a row where every neighbour ties falls back to uniform.
    """
    w = sims - sims.min(axis=1, keepdims=True)
    total = w.sum(axis=1, keepdims=True)
    uniform = np.full_like(w, 1.0 / w.shape[1])
    return np.where(total > 1e-12, w / np.where(total > 1e-12, total, 1.0), uniform)


def _weighted_median(values: np.ndarray, weights: np.ndarray) -> np.ndarray:
    """Per-row weighted median of a (n, k) array."""
    order = np.argsort(values, axis=1)
    rows = np.arange(values.shape[0])[:, None]
    v = values[rows, order]
    w = weights[rows, order]
    cum = np.cumsum(w, axis=1)
    idx = (cum >= 0.5 * cum[:, -1:]).argmax(axis=1)
    return v[np.arange(values.shape[0]), idx]


def _densest_subset(coords: np.ndarray, weights: np.ndarray) -> np.ndarray:
    """Average only the tightest cluster among each row's neighbour coordinates.

    For bimodality: when a population sits in two distant patches, the mean of
    its neighbours lands in the empty gap between them, which is a location the
    tissue does not have. This lands the cell in one of the patches instead.

    **It cannot say which patch**, and does not pretend to — two patches with
    the same expression program are indistinguishable to any method reading
    expression, so on the bundled benchmark this picks the right one about half
    the time. What it buys is a prediction on real tissue rather than in a gap;
    the confidence score (near zero for such cells either way) is what says not
    to trust the choice. Trading a never-right-never-catastrophic estimate for a
    sometimes-exact-sometimes-wrong one is a real decision, which is why both
    remain available and the default stays the mean.

    The radius comes from the neighbourhood's own pairwise distances, so it needs
    no tissue-scale constant and degrades to the plain mean when the cloud is
    unimodal.
    """
    n, k, _ = coords.shape
    out = np.empty((n, 2), dtype=float)
    for i in range(n):
        c = coords[i]
        d = np.linalg.norm(c[:, None, :] - c[None, :, :], axis=2)
        radius = np.percentile(d, 25)
        if radius <= 0:
            out[i] = c[0]
            continue
        within = d <= radius
        counts = within.sum(axis=1)
        best = np.flatnonzero(counts == counts.max())
        # Ties broken by weight, so the most similar of the equally dense wins.
        seed = best[np.argmax(weights[i][best])]
        members = within[seed]
        w = weights[i][members]
        total = w.sum()
        out[i] = (c[members] * w[:, None]).sum(axis=0) / total if total > 1e-12 \
            else c[members].mean(axis=0)
    return out


def _aggregate(coords: np.ndarray, weights: np.ndarray, how: str) -> np.ndarray:
    if how == 'weighted_mean':
        return (coords * weights[:, :, None]).sum(axis=1)
    if how == 'median':
        return np.column_stack([
            _weighted_median(coords[:, :, 0], weights),
            _weighted_median(coords[:, :, 1], weights),
        ])
    if how == 'best_match':
        return coords[:, 0, :].astype(float)
    return _densest_subset(coords, weights)


def injective_feasibility(n_query: int, n_ref: int) -> str | None:
    """Why injective assignment cannot run at these sizes, or None if it can.

    One message, three callers: the estimator itself, ``project_knn`` — which
    checks before doing a minute of matching work it would then throw away — and
    the adaptor, which turns it into a 400 rather than a background task that
    dies later.
    """
    if n_ref < n_query:
        return (
            f'Injective assignment needs at least one reference spot per query '
            f'cell: {n_query:,} cells against {n_ref:,} spots. Use best_match, '
            f'or subsample the query.'
        )
    if n_query * min(_CANDIDATES, n_ref) > _MAX_CANDIDATE_EDGES:
        return (
            f'Injective assignment would need a candidate graph of about '
            f'{n_query * min(_CANDIDATES, n_ref):,} edges for {n_query:,} cells, '
            f'which is beyond what this refuses to allocate in memory. '
            f'Subsample the query, or use best_match.'
        )
    return None


def injective_assignment(similarity: np.ndarray) -> np.ndarray:
    """One distinct reference spot per query cell, maximizing total similarity.

    ``best_match`` takes each cell's single best spot independently, so a spot
    that correlates well with everything — better recovered, less noisy —
    absorbs many cells. Solving for the best *set* of assignments subject to
    distinct spots removes the pile-up entirely and costs very little: measured
    on the limb pair, 2,683 cells onto 2,683 distinct spots (from 1,749 on all
    shared genes, or 865 on HVG), with mean similarity falling 0.163 to 0.151.

    What it does **not** buy is placement. Like every single-location estimator
    it keeps the tissue's full extent and retains no axis signal — measured
    axis fidelity -0.056 / +0.017 / +0.033 / -0.027 against a reference ceiling
    of -0.327 / +0.316 / +0.306 / -0.178, which is greedy ``best_match`` to
    within noise. It is a different point on the same frontier.

    Args:
        similarity: ``(n_query, n_ref)``, larger meaning more alike — which is
            the convention every metric in this module already produces.

    Returns:
        ``(n_query,)`` column index per row, all distinct.

    Raises:
        ValueError: when the sizes make a distinct assignment impossible, or
            when the matrix is larger than this will allocate.
    """
    from scipy.optimize import linear_sum_assignment

    S = np.asarray(similarity)
    if S.ndim != 2:
        raise ValueError('similarity must be a 2-D (n_query, n_ref) matrix')
    n_query, n_ref = S.shape
    problem = injective_feasibility(n_query, n_ref)
    if problem:
        raise ValueError(problem)

    # maximize=True on a rectangular matrix assigns every row exactly once and
    # leaves the surplus columns unused, which is the shape we want.
    rows, cols = linear_sum_assignment(S, maximize=True)
    out = np.empty(n_query, dtype=np.int64)
    out[rows] = cols
    return out


def _feasible_seed(cand_idx: np.ndarray, best_sim: np.ndarray, n_ref: int) -> np.ndarray:
    """Some injective assignment — any one — over the candidate lists.

    Its only job is to exist. Adding these edges to the candidate graph makes a
    full matching a theorem rather than something to hope for, and the optimizer
    can only improve on it. Without it, a top-c graph on real data had no full
    matching at any c up to 64: many query cells rank the same well-recovered
    spots highest, so their candidate sets overlap and Hall's condition fails —
    which is the very pile-up injective exists to remove.

    Cells choose in descending order of their best similarity, so the most
    confident cells are the ones that get their first choice.
    """
    n_query = cand_idx.shape[0]
    used = np.zeros(n_ref, dtype=bool)
    out = np.full(n_query, -1, dtype=np.int64)

    for i in np.argsort(-np.asarray(best_sim)):
        for j in cand_idx[i]:
            if not used[j]:
                out[i] = j
                used[j] = True
                break

    stranded = np.flatnonzero(out < 0)
    if len(stranded):
        # Every candidate of these cells was taken. There is always a spot left
        # over — n_ref >= n_query is checked before any of this runs.
        out[stranded] = np.flatnonzero(~used)[:len(stranded)]
    return out


def candidate_assignment(cand_idx, cand_sim, n_ref: int) -> np.ndarray:
    """Injective assignment over each cell's best candidates, not every spot.

    The optimal assignment overwhelmingly uses each cell's few best spots, so
    restricting the problem to them costs almost nothing and removes the memory
    wall entirely: on the E11.5 limb pair, 4.1 MB against 315 MB, an objective
    0.18% below optimal, 87% of cells on the identical spot, and map-quality
    metrics that cannot separate the two.

    It is **near-optimal, not optimal**. Callers report which path produced an
    assignment for exactly that reason.

    Args:
        cand_idx: ``(n_query, c)`` reference indices, best first.
        cand_sim: ``(n_query, c)`` their similarities, larger being more alike.
        n_ref: how many reference spots exist in total.

    Returns:
        ``(n_query,)`` column index per row, all distinct.
    """
    from scipy.sparse import csr_matrix
    from scipy.sparse.csgraph import min_weight_full_bipartite_matching

    cand_idx = np.asarray(cand_idx, dtype=np.int64)
    cand_sim = np.asarray(cand_sim, dtype=float)
    n_query, c = cand_idx.shape

    problem = injective_feasibility(n_query, n_ref)
    if problem:
        raise ValueError(problem)

    seed = _feasible_seed(cand_idx, cand_sim.max(axis=1), n_ref)

    rows = np.concatenate([np.repeat(np.arange(n_query), c), np.arange(n_query)])
    cols = np.concatenate([cand_idx.ravel(), seed])
    # A stranded seed edge is not in the candidate list, so its similarity was
    # never computed. Scoring it below every candidate keeps the solver away
    # from it unless a cell genuinely has nothing else left.
    sims = np.concatenate([
        cand_sim.ravel(),
        np.full(n_query, float(cand_sim.min()) - 1.0),
    ])

    # An edge is an edge. Building the CSR with repeats *sums* them, and most
    # seed edges are also candidates, so their costs would double and the solver
    # would avoid exactly the edges it should prefer — measured at 12% of the
    # objective. np.unique keeps the first occurrence, and candidates come first
    # in the concatenation above, so the real similarity is the one kept.
    _, keep = np.unique(rows.astype(np.int64) * n_ref + cols, return_index=True)
    rows, cols, sims = rows[keep], cols[keep], sims[keep]

    # Maximize similarity by minimizing its complement. The +1e-9 matters: a
    # stored zero reads as "no edge" to the matcher, so the best edge in the
    # graph would silently disappear.
    cost = (float(sims.max()) - sims) + 1e-9
    matrix = csr_matrix((cost, (rows, cols)), shape=(n_query, n_ref))

    row_ind, col_ind = min_weight_full_bipartite_matching(matrix)
    out = np.empty(n_query, dtype=np.int64)
    out[row_ind] = col_ind
    return out


# --- projection -----------------------------------------------------------

def _chunk_rows(n_ref: int) -> int:
    """How many query cells to score at once against this many spots."""
    return max(1, min(_CHUNK, _MAX_BLOCK_ENTRIES // max(int(n_ref), 1)))


def _tissue_radius(coords: np.ndarray) -> float:
    """RMS distance of the reference cells from their own centroid.

    The natural scale for "how scattered is scattered": neighbours spread this
    far carry no more positional information than picking a cell at random.
    """
    centred = coords - coords.mean(axis=0)
    return float(np.sqrt((centred ** 2).sum(axis=1).mean())) or 1.0


def project_knn(
    query_expr: np.ndarray,
    ref_expr: np.ndarray,
    ref_coords: np.ndarray,
    *,
    k: int = 15,
    metric: str = 'correlation',
    transform: str = 'zscore',
    aggregation: str = 'weighted_mean',
    ref_sections: np.ndarray | None = None,
    min_confidence: float = 0.0,
    epsilon: float = 0.05,
    max_iterations: int = 300,
    progress: Callable[[float, str], None] | None = None,
) -> Projection:
    """Predict a coordinate for every query cell from a spatial reference.

    ``query_expr`` and ``ref_expr`` must already be restricted to the same genes,
    in the same order — matching them by name is the caller's job, because only
    the caller has the names.
    """
    if metric not in METRICS:
        raise ValueError(f"metric must be one of {list(METRICS)}, got '{metric}'")
    if aggregation not in AGGREGATIONS:
        raise ValueError(
            f"aggregation must be one of {list(AGGREGATIONS)}, got '{aggregation}'"
        )

    query_expr = np.asarray(query_expr, dtype=np.float32)
    ref_expr = np.asarray(ref_expr, dtype=np.float32)
    ref_coords = np.asarray(ref_coords, dtype=float)

    if ref_expr.shape[0] == 0:
        raise ValueError('The spatial reference has no cells')
    if query_expr.shape[1] != ref_expr.shape[1]:
        raise ValueError(
            f'Query and reference must share the same genes: '
            f'{query_expr.shape[1]} vs {ref_expr.shape[1]}'
        )
    if ref_coords.shape[0] != ref_expr.shape[0]:
        raise ValueError(
            f'Reference coordinates ({ref_coords.shape[0]}) do not match '
            f'reference cells ({ref_expr.shape[0]})'
        )
    if ref_coords.ndim != 2 or ref_coords.shape[1] < 2:
        raise ValueError('Reference coordinates must be (n_cells, >=2)')
    ref_coords = ref_coords[:, :2]
    if ref_sections is not None:
        ref_sections = np.asarray(ref_sections).astype(str)
        if len(ref_sections) != ref_expr.shape[0]:
            raise ValueError('Section labels do not match the reference cells')

    k = max(1, min(int(k), ref_expr.shape[0]))

    # Checked before the matching work rather than after it: on a real pair the
    # similarity blocks take tens of seconds, and failing then would waste them.
    needs_full = aggregation == 'injective'
    if needs_full:
        problem = injective_feasibility(query_expr.shape[0], ref_expr.shape[0])
        if problem:
            raise ValueError(problem)
    if aggregation == 'transport' and epsilon <= 0:
        raise ValueError(f'epsilon must be positive, got {epsilon}')

    q = _prepare_for_metric(transform_expression(query_expr, transform), metric)
    r = _prepare_for_metric(transform_expression(ref_expr, transform), metric)

    n_query = q.shape[0]
    n_ref = r.shape[0]
    idx = np.empty((n_query, k), dtype=np.int64)
    sims = np.empty((n_query, k), dtype=float)

    # Assignment is a global decision, so it needs more than the k neighbours
    # the confidence score is measured over. Dense keeps the whole matrix and is
    # exactly optimal; candidates keep each row's best few, which is all the
    # optimum turns out to use, at a fraction of the memory.
    use_dense = needs_full and n_query * n_ref <= _DENSE_MAX_ENTRIES
    n_cand = min(_CANDIDATES, n_ref) if (needs_full and not use_dense) else 0
    take = max(k, n_cand)

    full = np.empty((n_query, n_ref), dtype=np.float32) if use_dense else None
    cand_idx = np.empty((n_query, n_cand), dtype=np.int64) if n_cand else None
    cand_sim = np.empty((n_query, n_cand), dtype=float) if n_cand else None

    chunk = _chunk_rows(n_ref)
    for start in range(0, n_query, chunk):
        stop = min(start + chunk, n_query)
        block = _similarity_block(q[start:stop], r, metric)
        if full is not None:
            full[start:stop] = block
        # One argpartition serves both readers: the k neighbours the confidence
        # score needs, and the wider candidate set the assignment chooses among.
        part = np.argpartition(-block, take - 1, axis=1)[:, :take]
        rows = np.arange(stop - start)[:, None]
        part_sims = block[rows, part]
        order = np.argsort(-part_sims, axis=1)
        part = part[rows, order]
        part_sims = part_sims[rows, order]
        idx[start:stop] = part[:, :k]
        sims[start:stop] = part_sims[:, :k]
        if cand_idx is not None:
            cand_idx[start:stop] = part[:, :n_cand]
            cand_sim[start:stop] = part_sims[:, :n_cand]
        if progress is not None:
            progress(stop / n_query, f'Matched {stop:,} of {n_query:,} cells')

    weights = _weights(sims)
    neighbor_coords = ref_coords[idx]

    sections_out: np.ndarray | None = None
    if ref_sections is not None:
        # Averaging coordinates across sections lands in the gap between them.
        # Assign the weighted-majority section, then keep only its neighbours.
        neighbor_sections = ref_sections[idx]
        sections_out = np.empty(n_query, dtype=object)
        for i in range(n_query):
            labels, inverse = np.unique(neighbor_sections[i], return_inverse=True)
            totals = np.bincount(inverse, weights=weights[i], minlength=len(labels))
            chosen = labels[int(np.argmax(totals))]
            sections_out[i] = chosen
            keep = neighbor_sections[i] == chosen
            # Zero the others' weight rather than reshaping — keeps the arrays
            # rectangular, and a zero-weight neighbour contributes nothing to the
            # coordinate or to the spread.
            weights[i] = np.where(keep, weights[i], 0.0)
            total = weights[i].sum()
            weights[i] = weights[i] / total if total > 1e-12 else keep / keep.sum()
        sections_out = sections_out.astype(str)

    assignment_kind: str | None = None
    n_candidates_used: int | None = None
    transport_out: tuple[str, int, int, float] | None = None
    transport_spread: np.ndarray | None = None
    if needs_full:
        if progress is not None:
            progress(1.0, f'Assigning {n_query:,} cells to distinct spots')
        if full is not None:
            assigned = injective_assignment(full)
            assignment_kind = 'exact'
        else:
            assigned = candidate_assignment(cand_idx, cand_sim, n_ref)
            assignment_kind = 'candidate'
            n_candidates_used = n_cand
        coords = ref_coords[assigned].astype(float)
        if sections_out is not None:
            # The assignment ranges over the whole reference, so the honest
            # label is the assigned spot's own section rather than the majority
            # of the k neighbours — that majority exists to stop an *average*
            # landing in the gap between cuts, and nothing is being averaged.
            sections_out = ref_sections[assigned]
    elif aggregation == 'transport':
        from xcell import transport as tr

        ref_idx = np.arange(n_ref)
        transport_kind = 'full'
        if n_query * n_ref > _TRANSPORT_MAX_ENTRIES:
            budget = max(1, _TRANSPORT_MAX_ENTRIES // max(n_query, 1))
            ref_idx = tr.stratified_subsample(ref_coords, budget)
            transport_kind = 'subsampled'

        if progress is not None:
            progress(1.0, f'Transporting {n_query:,} cells onto '
                          f'{len(ref_idx):,} spots')

        # Recomputed rather than reused: the matching pass above keeps only
        # each row's best k, and the coupling is solved over every entry.
        sim = _similarity_block(q, r[ref_idx], metric)
        cost, _ = tr.normalize_cost(-sim)
        result = tr.sinkhorn_log(
            cost, epsilon, max_iterations=max_iterations, progress=progress,
        )
        sub_sections = ref_sections[ref_idx] if ref_sections is not None else None
        coords, chosen = tr.barycentric_projection(
            result.coupling, ref_coords[ref_idx], sub_sections,
        )
        if chosen is not None:
            sections_out = chosen
        transport_out = (transport_kind, len(ref_idx), result.iterations,
                         result.marginal_error)
        # Each row of the coupling is a spatial posterior, so the spread of
        # that posterior is the honest confidence — and it is the same
        # quantity the k-NN path measures, so the two stay comparable.
        transport_spread = tr.posterior_spread(
            result.coupling, ref_coords[ref_idx], coords,
        )
    else:
        coords = _aggregate(neighbor_coords, weights, aggregation)

    # Confidence: weighted RMS spread of the neighbours about the prediction,
    # against the tissue's own radius. 1 = one spot, 0 = as scattered as the
    # whole reference, i.e. no positional information.
    if transport_spread is not None:
        spread = transport_spread
    else:
        offsets = neighbor_coords - coords[:, None, :]
        spread = np.sqrt((weights * (offsets ** 2).sum(axis=2)).sum(axis=1))
    confidence = np.clip(1.0 - spread / _tissue_radius(ref_coords), 0.0, 1.0)

    similarity = (weights * sims).sum(axis=1)

    n_unplaced = 0
    if min_confidence > 0:
        below = confidence < min_confidence
        n_unplaced = int(below.sum())
        coords = coords.astype(float)
        coords[below] = np.nan

    if progress is not None:
        progress(1.0, f'Placed {n_query - n_unplaced:,} of {n_query:,} cells')

    return Projection(
        coords=coords,
        confidence=confidence,
        similarity=similarity,
        neighbor_indices=idx,
        sections=sections_out,
        n_unplaced=n_unplaced,
        assignment=assignment_kind,
        n_candidates=n_candidates_used,
        transport=transport_out[0] if transport_out else None,
        n_ref_used=transport_out[1] if transport_out else None,
        sinkhorn_iterations=transport_out[2] if transport_out else None,
        marginal_error=transport_out[3] if transport_out else None,
    )


# --- evaluation -----------------------------------------------------------

def _f(value: Any) -> float | None:
    """JSON-safe float: None for anything not finite, per the house rule."""
    if value is None:
        return None
    v = float(value)
    return v if np.isfinite(v) else None


def _spearman(a: np.ndarray, b: np.ndarray) -> float | None:
    """Rank correlation without pulling in scipy.stats for two arrays."""
    if len(a) < 3:
        return None
    ra = np.argsort(np.argsort(a)).astype(float)
    rb = np.argsort(np.argsort(b)).astype(float)
    ra -= ra.mean()
    rb -= rb.mean()
    denom = np.sqrt((ra ** 2).sum() * (rb ** 2).sum())
    return _f(float((ra * rb).sum() / denom)) if denom > 0 else None


def evaluate_localization(
    pred_coords: np.ndarray,
    true_coords: np.ndarray,
    *,
    ref_coords: np.ndarray | None = None,
    confidence: np.ndarray | None = None,
    groups: np.ndarray | None = None,
    seed: int = 0,
    max_pairs: int = 2000,
) -> dict[str, Any]:
    """Score predicted coordinates against known truth.

    The baselines are not decoration. On a compact tissue, predicting the
    centroid for every cell scores deceptively well, so an error reported on its
    own is unfalsifiable — a method that cannot beat *predict-the-middle* is
    doing nothing, and only the comparison shows it.
    """
    pred = np.asarray(pred_coords, dtype=float)[:, :2]
    truth = np.asarray(true_coords, dtype=float)[:, :2]
    if pred.shape[0] != truth.shape[0]:
        raise ValueError('Predicted and true coordinates must describe the same cells')

    placed = np.isfinite(pred).all(axis=1)
    n_unplaced = int((~placed).sum())
    locations = np.asarray(ref_coords, dtype=float)[:, :2] if ref_coords is not None \
        else truth

    out: dict[str, Any] = {
        'n_cells': int(len(pred)),
        'n_evaluated': int(placed.sum()),
        'n_unplaced': n_unplaced,
    }
    # Diameter of the true tissue — the yardstick that makes an error in
    # unknown units interpretable.
    spans = truth.max(axis=0) - truth.min(axis=0)
    diameter = float(np.linalg.norm(spans)) or 1.0
    out['tissue_diameter'] = _f(diameter)

    if not placed.any():
        out.update({
            'median_error': None, 'mean_error': None,
            'median_error_normalized': None,
            'baseline_centroid': None, 'baseline_random': None,
            'fraction_within': {'5%': None, '10%': None, '25%': None},
            'structure_correlation': None,
            'confidence_error_correlation': None,
            'per_group': {},
        })
        return out

    err = np.linalg.norm(pred[placed] - truth[placed], axis=1)
    out['median_error'] = _f(np.median(err))
    out['mean_error'] = _f(err.mean())
    out['median_error_normalized'] = _f(np.median(err) / diameter)

    centroid = locations.mean(axis=0)
    out['baseline_centroid'] = _f(
        np.median(np.linalg.norm(truth[placed] - centroid, axis=1))
    )
    rng = np.random.default_rng(seed)
    drawn = locations[rng.integers(0, len(locations), placed.sum())]
    out['baseline_random'] = _f(
        np.median(np.linalg.norm(truth[placed] - drawn, axis=1))
    )

    out['fraction_within'] = {
        label: _f(float((err <= frac * diameter).mean()))
        for label, frac in (('5%', 0.05), ('10%', 0.10), ('25%', 0.25))
    }

    # Structure: do cells predicted near each other actually sit near each
    # other? A method can score well per-cell and badly here, or the reverse.
    n = int(placed.sum())
    if n >= 3:
        m = min(max_pairs, n * (n - 1) // 2)
        i = rng.integers(0, n, m)
        j = rng.integers(0, n, m)
        keep = i != j
        pd_ = np.linalg.norm(pred[placed][i[keep]] - pred[placed][j[keep]], axis=1)
        td = np.linalg.norm(truth[placed][i[keep]] - truth[placed][j[keep]], axis=1)
        out['structure_correlation'] = _spearman(pd_, td)
    else:
        out['structure_correlation'] = None

    # The design's own falsification test: a confidence score that does not
    # predict error is decoration, and the report has to be able to say so.
    if confidence is not None:
        conf = np.asarray(confidence, dtype=float)[placed]
        out['confidence_error_correlation'] = _spearman(conf, err)
    else:
        out['confidence_error_correlation'] = None

    per_group: dict[str, Any] = {}
    if groups is not None:
        g = np.asarray(groups).astype(str)[placed]
        for label in sorted(set(g.tolist())):
            sel = g == label
            per_group[label] = {
                'n': int(sel.sum()),
                'median_error': _f(np.median(err[sel])),
                'median_error_normalized': _f(np.median(err[sel]) / diameter),
            }
    out['per_group'] = per_group
    return out


def cross_validate_localization(
    ref_expr: np.ndarray,
    ref_coords: np.ndarray,
    *,
    holdout_fraction: float = 0.2,
    ref_sections: np.ndarray | None = None,
    groups: np.ndarray | None = None,
    seed: int = 0,
    progress: Callable[[float, str], None] | None = None,
    **project_kwargs: Any,
) -> dict[str, Any]:
    """Hold out part of the reference and predict it from the rest.

    Answers "will these parameters work on this tissue?" before anything is
    trusted, with exact ground truth. It is an **upper bound**: within one
    dataset there is no platform gap to cross, so real scRNA-seq → spatial error
    will be worse. The returned ``same_platform`` flag exists so a caller cannot
    quietly present this as the real number.
    """
    ref_expr = np.asarray(ref_expr, dtype=np.float32)
    ref_coords = np.asarray(ref_coords, dtype=float)
    n = ref_expr.shape[0]
    if not 0 < holdout_fraction < 1:
        raise ValueError('holdout_fraction must be between 0 and 1')
    n_hold = max(1, int(round(n * holdout_fraction)))
    if n - n_hold < 2:
        raise ValueError('Not enough reference cells to hold any out')

    rng = np.random.default_rng(seed)
    held = rng.permutation(n)[:n_hold]
    keep = np.ones(n, dtype=bool)
    keep[held] = False

    projection = project_knn(
        ref_expr[held],
        ref_expr[keep],
        ref_coords[keep],
        ref_sections=None if ref_sections is None else np.asarray(ref_sections)[keep],
        progress=progress,
        **project_kwargs,
    )
    metrics = evaluate_localization(
        projection.coords,
        ref_coords[held],
        ref_coords=ref_coords[keep],
        confidence=projection.confidence,
        groups=None if groups is None else np.asarray(groups)[held],
        seed=seed,
    )
    metrics['same_platform'] = True
    metrics['n_holdout'] = int(n_hold)
    metrics['n_reference'] = int(keep.sum())
    return metrics
