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
AGGREGATIONS = ('weighted_mean', 'median', 'best_match', 'densest')

# Query cells per chunk. Keeps the similarity block bounded regardless of how
# many cells are being mapped, and gives progress something to report.
_CHUNK = 2048


@dataclass
class Projection:
    """Predicted coordinates plus everything needed to judge them."""

    coords: np.ndarray               # (n_query, 2), NaN where below min_confidence
    confidence: np.ndarray           # (n_query,) in [0, 1]
    similarity: np.ndarray           # (n_query,) mean similarity to the k neighbours
    neighbor_indices: np.ndarray     # (n_query, k) into the reference
    sections: np.ndarray | None      # (n_query,) assigned section label, or None
    n_unplaced: int


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

    Answers bimodality: when a population sits in two distant patches, the mean
    of its neighbours lands in the empty gap between them. Here the radius comes
    from the neighbourhood's own pairwise distances, so it needs no tissue-scale
    constant and degrades to the plain mean when the cloud is unimodal.
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


# --- projection -----------------------------------------------------------

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

    q = _prepare_for_metric(transform_expression(query_expr, transform), metric)
    r = _prepare_for_metric(transform_expression(ref_expr, transform), metric)

    n_query = q.shape[0]
    idx = np.empty((n_query, k), dtype=np.int64)
    sims = np.empty((n_query, k), dtype=float)

    for start in range(0, n_query, _CHUNK):
        stop = min(start + _CHUNK, n_query)
        block = _similarity_block(q[start:stop], r, metric)
        # argpartition for the top-k, then order those k properly.
        part = np.argpartition(-block, k - 1, axis=1)[:, :k]
        rows = np.arange(stop - start)[:, None]
        part_sims = block[rows, part]
        order = np.argsort(-part_sims, axis=1)
        idx[start:stop] = part[rows, order]
        sims[start:stop] = part_sims[rows, order]
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

    coords = _aggregate(neighbor_coords, weights, aggregation)

    # Confidence: weighted RMS spread of the neighbours about the prediction,
    # against the tissue's own radius. 1 = one spot, 0 = as scattered as the
    # whole reference, i.e. no positional information.
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
