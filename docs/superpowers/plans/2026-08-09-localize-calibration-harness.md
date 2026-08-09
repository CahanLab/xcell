# Localize Calibration Harness — Implementation Plan (Phase 1)

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make the quality of a Localize map measurable, so a run can be judged against the reference and two runs compared — instead of "the map looks wrong."

**Architecture:** A pure metrics module takes arrays and returns a panel of four numbers per marker set. The adaptor pulls the matrices out and calls it; one route scores *several* predicted embeddings at once so the user can rank the variants they already have; the Localize modal renders the table.

**Tech Stack:** Python 3 / numpy / scipy (`ndimage`, `stats`, `spatial`); FastAPI; React 18 + TypeScript; pytest; vitest.

**Spec:** `docs/superpowers/specs/2026-08-09-localize-calibration-and-transport-design.md`

## Global Constraints

- Branch `feat/localize-metrics` off `main`; merge back with `--no-ff`.
- TDD: write the failing test, run it, watch it fail, then implement.
- Backend tests from `backend/`: `pixi run -e dev pytest`. Frontend: `cd frontend && npm test`, typecheck `npx tsc --noEmit`.
- `backend/tests/` has **no `conftest.py`** — each file is self-contained, with module-level `_adata()` / `_adaptor()` factories, seeded `np.random.default_rng(0)`, `csr_matrix` + `float32`.
- `backend/xcell/localize_metrics.py` is **pure**: arrays in, plain dicts out. No AnnData, no adaptor import — same contract as `localize.py`, which its docstring states explicitly.
- Everything crossing the API must be JSON-serializable: **no `inf`, no `NaN`**. Use `None` for "not computable" (see `layer_scale._cv` and `localize._f`).
- Heavy imports (`scipy.ndimage`, `scipy.spatial`) stay function-local.
- Error mapping in routes: `ValueError → 400`, `KeyError → 404`, `except HTTPException: raise` before any catch-all.
- Styling is inline. Palette: panel `#16213e`, border `#0f3460`, inset `#0f1625`, accent `#4ecdc4`, alert `#e94560`, warn `#e9a23b`, muted `#aaa`/`#888`.
- Comments explain *why*, especially where a threshold was chosen.

**The one design decision that governs the whole module: every metric is
rank-based.** The reference is 55 µm ST spots and the query is dissociated
cells; their absolute expression scales are not comparable, and neither are
their coordinate scales once a method shrinks the map. Ranking both sides
before comparing removes that entirely, and makes each metric invariant to any
monotone rescale — which matters because an affine rescale of the predictions
provably cannot change a rank correlation, so a metric that moved under one
would be measuring the wrong thing.

## File Structure

| file | responsibility |
|---|---|
| `backend/xcell/localize_metrics.py` (new) | the four pure metrics + `evaluate_map` orchestrator |
| `backend/tests/test_localize_metrics.py` (new) | unit tests on synthetic geometry with known answers |
| `backend/xcell/adaptor.py` (modify) | `evaluate_localization_maps()` — pulls arrays, calls the module |
| `backend/xcell/api/routes.py` (modify) | `POST /api/localize/evaluate_map` |
| `backend/tests/test_localize_metrics_routes.py` (new) | route contract |
| `frontend/src/components/LocalizeModal.tsx` (modify) | "Map quality" table |

**Module API**, fixed here so every task agrees:

```python
def axis_direction(coords: np.ndarray, scores: np.ndarray) -> np.ndarray          # (2,) unit vector
def axis_fidelity(ref_coords, ref_scores, pred_coords, pred_scores) -> dict
def spatial_pattern_fidelity(ref_coords, ref_scores, pred_coords, pred_scores, *, bins: int = 32) -> dict
def dispersion(ref_coords, pred_coords, *, inner: float = 0.95) -> dict
def occupancy(ref_coords, pred_coords) -> dict
def evaluate_map(ref_coords, pred_coords, marker_sets: list[dict]) -> dict
```

`marker_sets` entries are `{'name': str, 'ref_scores': np.ndarray, 'pred_scores': np.ndarray}`.

---

### Task 1: Dispersion and occupancy

The two coordinate-only metrics. No markers involved, so they are the simplest
place to establish the module and its conventions.

**Files:**
- Create: `backend/xcell/localize_metrics.py`
- Create: `backend/tests/test_localize_metrics.py`

**Interfaces:**
- Consumes: nothing.
- Produces:
  - `dispersion(ref_coords, pred_coords, *, inner=0.95) -> {'area_ratio': float|None, 'std_ratio_x': float|None, 'std_ratio_y': float|None, 'frac_outside': float}`
  - `occupancy(ref_coords, pred_coords) -> {'n_distinct': int, 'max_per_spot': int, 'effective_n': float, 'n_pred': int}`

- [ ] **Step 1: Write the failing tests**

Create `backend/tests/test_localize_metrics.py`:

```python
"""Measuring whether a predicted spatial map is any good.

The metrics exist because 'the map looks wrong' was not actionable: eight saved
parameter variants existed on one dataset with no way to rank them. Each test
here builds geometry whose right answer is known by construction.
"""

import numpy as np
import pytest

from xcell.localize_metrics import dispersion, occupancy


def _disc(n=2000, r=1.0, seed=0):
    """A filled disc of radius r, centred at the origin."""
    rng = np.random.default_rng(seed)
    rad = r * np.sqrt(rng.random(n))
    th = rng.random(n) * 2 * np.pi
    return np.column_stack([rad * np.cos(th), rad * np.sin(th)])


def test_dispersion_is_one_when_the_prediction_matches_the_reference():
    ref = _disc(seed=0)
    out = dispersion(ref, _disc(seed=1))
    assert out['area_ratio'] == pytest.approx(1.0, abs=0.15)
    assert out['std_ratio_x'] == pytest.approx(1.0, abs=0.15)
    assert out['frac_outside'] == pytest.approx(0.0, abs=0.05)


def test_dispersion_detects_a_collapsed_map():
    """The failure this exists to catch: weighted_mean shrank the limb map to
    0.15 of the tissue area while every axis correlation stayed correct."""
    ref = _disc(seed=0)
    out = dispersion(ref, _disc(seed=1) * 0.5)      # half the radius
    assert out['area_ratio'] == pytest.approx(0.25, abs=0.06)   # area goes as r^2
    assert out['std_ratio_x'] == pytest.approx(0.5, abs=0.06)


def test_dispersion_detects_an_overshoot_and_counts_cells_outside():
    ref = _disc(seed=0)
    out = dispersion(ref, _disc(seed=1) * 2.0)
    assert out['area_ratio'] > 3.0
    # Rescaling to match variance pushed 14.8% of real cells out of the tissue;
    # that has to be visible rather than hidden inside an area ratio.
    assert out['frac_outside'] > 0.5


def test_dispersion_ignores_a_handful_of_outliers():
    """inner=0.95 trims the hull, so one stray point cannot dominate."""
    ref = _disc(seed=0)
    pred = np.vstack([_disc(seed=1), np.array([[50.0, 50.0]])])
    assert dispersion(ref, pred)['area_ratio'] == pytest.approx(1.0, abs=0.2)


def test_occupancy_is_clean_when_predictions_spread():
    ref = _disc(n=500, seed=0)
    out = occupancy(ref, _disc(n=400, seed=1))
    assert out['n_pred'] == 400
    assert out['n_distinct'] > 250          # most cells find their own spot
    assert out['max_per_spot'] <= 6


def test_occupancy_detects_pile_up():
    """greedy best_match put 2,683 cells on 865 spots, 15 on a single one."""
    ref = _disc(n=500, seed=0)
    pred = np.repeat(ref[:10], 40, axis=0)   # 400 cells crammed onto 10 spots
    out = occupancy(ref, pred)
    assert out['n_distinct'] == 10
    assert out['max_per_spot'] == 40
    assert out['effective_n'] == pytest.approx(10.0, abs=0.5)


def test_occupancy_effective_n_penalises_unevenness():
    """Two maps can use the same spots and still differ: effective_n is the
    exponential of the assignment entropy, so it drops when a few spots absorb
    most cells even though n_distinct does not."""
    ref = _disc(n=500, seed=0)
    even = np.repeat(ref[:20], 10, axis=0)                       # 10 each
    skewed = np.vstack([np.repeat(ref[:1], 181, axis=0),
                        np.repeat(ref[1:20], 1, axis=0)])        # 181 then 1s
    assert occupancy(ref, even)['n_distinct'] == 20
    assert occupancy(ref, skewed)['n_distinct'] == 20
    assert occupancy(ref, skewed)['effective_n'] < occupancy(ref, even)['effective_n']


def test_metrics_are_json_safe_on_degenerate_input():
    import json
    ref = _disc(n=10, seed=0)
    single = np.tile(ref[:1], (5, 1))        # no spread at all -> no hull
    json.dumps(dispersion(ref, single))      # raises on NaN/inf
    json.dumps(occupancy(ref, single))
```

- [ ] **Step 2: Run them and watch them fail**

```bash
cd backend && pixi run -e dev pytest tests/test_localize_metrics.py -q
```

Expected: `ModuleNotFoundError: No module named 'xcell.localize_metrics'`.

- [ ] **Step 3: Implement**

Create `backend/xcell/localize_metrics.py`:

```python
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
    """Convex-hull area of the innermost `inner` fraction of points.

    The plain hull is set by its most extreme member, so a single stray
    prediction can multiply it. Trimming by distance from the median first
    makes the number describe the bulk of the map.
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
    ref = np.asarray(ref_coords, dtype=float)
    pred = np.asarray(pred_coords, dtype=float)
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


def occupancy(ref_coords, pred_coords) -> dict[str, Any]:
    """How many distinct reference locations the predictions actually use.

    Computed from coordinates rather than from a method's own assignment, so it
    applies to every aggregation rather than only to the matching ones.
    """
    from scipy.spatial import cKDTree

    ref = np.asarray(ref_coords, dtype=float)
    pred = np.asarray(pred_coords, dtype=float)
    pred = pred[np.isfinite(pred).all(axis=1)]
    if len(pred) == 0:
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
```

- [ ] **Step 4: Run the tests**

```bash
cd backend && pixi run -e dev pytest tests/test_localize_metrics.py -q
```

Expected: 8 passed. If `test_dispersion_ignores_a_handful_of_outliers` is
borderline, check the trim is on distance from the **median** rather than the
mean — the mean is itself dragged by the outlier.

- [ ] **Step 5: Commit**

```bash
git add backend/xcell/localize_metrics.py backend/tests/test_localize_metrics.py
git commit -m "feat(localize): dispersion and occupancy metrics for a predicted map"
```

---

### Task 2: Spatial pattern fidelity

The headline metric: does a cell type land where it lives?

**Files:**
- Modify: `backend/xcell/localize_metrics.py`
- Modify: `backend/tests/test_localize_metrics.py`

**Interfaces:**
- Consumes: `_f` (Task 1).
- Produces: `spatial_pattern_fidelity(ref_coords, ref_scores, pred_coords, pred_scores, *, bins=32) -> {'correlation': float|None, 'n_bins_compared': int}`

- [ ] **Step 1: Write the failing tests**

Append to `backend/tests/test_localize_metrics.py`:

```python
from xcell.localize_metrics import spatial_pattern_fidelity


def _annulus_scores(coords, inner=0.75):
    """High on the rim, low inside — the epidermis pattern."""
    r = np.linalg.norm(coords, axis=1)
    return (r >= inner).astype(float)


def test_pattern_fidelity_is_high_when_the_pattern_is_reproduced():
    ref, pred = _disc(seed=0), _disc(seed=1)
    out = spatial_pattern_fidelity(ref, _annulus_scores(ref), pred, _annulus_scores(pred))
    assert out['correlation'] > 0.7


def test_pattern_fidelity_is_negative_when_the_pattern_is_inverted():
    """The limb failure exactly: epidermis belongs on the rim and was predicted
    into the centre. This must come out clearly negative, not merely low."""
    ref, pred = _disc(seed=0), _disc(seed=1)
    out = spatial_pattern_fidelity(
        ref, _annulus_scores(ref), pred, 1.0 - _annulus_scores(pred),
    )
    assert out['correlation'] < -0.3


def test_pattern_fidelity_is_near_zero_for_an_unrelated_pattern():
    ref, pred = _disc(seed=0), _disc(seed=1)
    rng = np.random.default_rng(3)
    out = spatial_pattern_fidelity(ref, _annulus_scores(ref), pred, rng.random(len(pred)))
    assert abs(out['correlation']) < 0.3


def test_pattern_fidelity_ignores_the_scale_of_the_scores():
    """ST spots and dissociated cells are on different expression scales, so a
    metric sensitive to that would compare the platforms, not the biology."""
    ref, pred = _disc(seed=0), _disc(seed=1)
    a = spatial_pattern_fidelity(ref, _annulus_scores(ref), pred, _annulus_scores(pred))
    b = spatial_pattern_fidelity(
        ref, _annulus_scores(ref), pred, 100.0 * _annulus_scores(pred) + 7.0,
    )
    assert a['correlation'] == pytest.approx(b['correlation'], abs=1e-9)


def test_pattern_fidelity_handles_an_empty_overlap():
    import json
    ref = _disc(seed=0)
    far = _disc(seed=1) + 1000.0            # nowhere near the reference
    out = spatial_pattern_fidelity(ref, _annulus_scores(ref), far, _annulus_scores(far))
    json.dumps(out)
    assert out['correlation'] is None or np.isfinite(out['correlation'])
```

- [ ] **Step 2: Run them and watch them fail**

```bash
cd backend && pixi run -e dev pytest tests/test_localize_metrics.py -q -k pattern
```

Expected: `ImportError: cannot import name 'spatial_pattern_fidelity'`.

- [ ] **Step 3: Implement**

Append to `backend/xcell/localize_metrics.py`:

```python
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


def _density(coords, weights, edges_x, edges_y, sigma: float = 1.0) -> np.ndarray:
    """Marker-weighted cell density on a fixed grid, lightly smoothed."""
    from scipy.ndimage import gaussian_filter

    H, _, _ = np.histogram2d(
        coords[:, 0], coords[:, 1], bins=[edges_x, edges_y], weights=weights,
    )
    H = gaussian_filter(H, sigma=sigma, mode='nearest')
    total = H.sum()
    return H / total if total > 0 else H


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

    ref = np.asarray(ref_coords, dtype=float)
    pred = np.asarray(pred_coords, dtype=float)
    rs = np.asarray(ref_scores, dtype=float).ravel()
    ps = np.asarray(pred_scores, dtype=float).ravel()

    ok = np.isfinite(pred).all(axis=1)
    pred, ps = pred[ok], ps[ok]
    if len(pred) < 3 or len(ref) < 3:
        return {'correlation': None, 'n_bins_compared': 0}

    edges_x = np.linspace(ref[:, 0].min(), ref[:, 0].max(), bins + 1)
    edges_y = np.linspace(ref[:, 1].min(), ref[:, 1].max(), bins + 1)

    ref_d = _density(ref, _rank_weights(rs), edges_x, edges_y)
    pred_d = _density(pred, _rank_weights(ps), edges_x, edges_y)

    # Only where the reference has tissue. An all-zero reference bin carries no
    # claim about where the cell type belongs.
    occupied = _density(ref, np.ones(len(ref)), edges_x, edges_y) > 0
    if occupied.sum() < 8:
        return {'correlation': None, 'n_bins_compared': int(occupied.sum())}

    rho, _ = spearmanr(ref_d[occupied], pred_d[occupied])
    return {'correlation': _f(rho), 'n_bins_compared': int(occupied.sum())}
```

- [ ] **Step 4: Run the tests**

```bash
cd backend && pixi run -e dev pytest tests/test_localize_metrics.py -q
```

Expected: all pass. If `test_pattern_fidelity_is_negative_when_the_pattern_is_inverted`
lands between −0.3 and 0, raise `bins` to 40 — too coarse a grid blurs a thin
rim into its interior and washes the sign out. Do **not** loosen the assertion:
detecting that inversion is the metric's entire purpose.

- [ ] **Step 5: Commit**

```bash
git add backend/xcell/localize_metrics.py backend/tests/test_localize_metrics.py
git commit -m "feat(localize): spatial pattern fidelity — does a cell type land where it lives"
```

---

### Task 3: Axis fidelity

**Files:**
- Modify: `backend/xcell/localize_metrics.py`
- Modify: `backend/tests/test_localize_metrics.py`

**Interfaces:**
- Consumes: `_f`, `_rank_weights` (Tasks 1–2).
- Produces:
  - `axis_direction(coords, scores) -> np.ndarray` shape `(2,)`, unit length
  - `axis_fidelity(...) -> {'reference': float|None, 'prediction': float|None, 'direction': [float, float]}`

- [ ] **Step 1: Write the failing tests**

Append to `backend/tests/test_localize_metrics.py`:

```python
from xcell.localize_metrics import axis_direction, axis_fidelity


def test_axis_direction_finds_the_gradient():
    """A score that increases with x must yield a direction along +x, whatever
    the coordinate units are."""
    coords = _disc(seed=0)
    u = axis_direction(coords, coords[:, 0])
    assert abs(u[0]) == pytest.approx(1.0, abs=0.05)
    assert u[0] > 0
    assert abs(u[1]) < 0.2


def test_axis_direction_finds_a_diagonal_gradient():
    coords = _disc(seed=0)
    u = axis_direction(coords, coords[:, 0] + coords[:, 1])
    assert u[0] == pytest.approx(0.707, abs=0.1)
    assert u[1] == pytest.approx(0.707, abs=0.1)


def test_axis_fidelity_reports_the_reference_as_the_ceiling():
    """The prediction is never reported alone: -0.17 means nothing until you
    know the reference itself only reaches -0.33."""
    ref = _disc(seed=0)
    pred = _disc(seed=1)
    out = axis_fidelity(ref, ref[:, 0], pred, pred[:, 0])
    assert out['reference'] > 0.9
    assert out['prediction'] > 0.9


def test_axis_fidelity_falls_when_the_gradient_is_lost():
    """best_match kept full tissue area and drove every axis correlation to
    roughly zero; that has to be visible."""
    ref = _disc(seed=0)
    pred = _disc(seed=1)
    rng = np.random.default_rng(5)
    out = axis_fidelity(ref, ref[:, 0], pred, rng.random(len(pred)))
    assert out['reference'] > 0.9
    assert abs(out['prediction']) < 0.2


def test_axis_fidelity_uses_the_references_direction_for_both():
    """Deriving the direction from the reference is what makes this
    orientation-free — the user never declares which way is distal."""
    ref = _disc(seed=0)
    pred = _disc(seed=1)
    out = axis_fidelity(ref, ref[:, 1], pred, pred[:, 1])
    assert abs(out['direction'][1]) > 0.9      # the y axis, discovered
    assert out['prediction'] > 0.9


def test_axis_fidelity_survives_a_constant_score():
    import json
    ref, pred = _disc(seed=0), _disc(seed=1)
    out = axis_fidelity(ref, np.ones(len(ref)), pred, np.ones(len(pred)))
    json.dumps(out)
```

- [ ] **Step 2: Run them and watch them fail**

```bash
cd backend && pixi run -e dev pytest tests/test_localize_metrics.py -q -k axis
```

Expected: `ImportError: cannot import name 'axis_direction'`.

- [ ] **Step 3: Implement**

Append to `backend/xcell/localize_metrics.py`:

```python
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
    ref_rho = spearmanr(rs, ref @ u)[0] if len(ref) > 2 else np.nan
    pred_rho = spearmanr(ps, pred @ u)[0] if len(pred) > 2 else np.nan

    return {
        'reference': _f(ref_rho),
        'prediction': _f(pred_rho),
        'direction': [float(u[0]), float(u[1])],
    }
```

- [ ] **Step 4: Run the tests**

```bash
cd backend && pixi run -e dev pytest tests/test_localize_metrics.py -q
```

Expected: all pass.

- [ ] **Step 5: Commit**

```bash
git add backend/xcell/localize_metrics.py backend/tests/test_localize_metrics.py
git commit -m "feat(localize): axis fidelity against the reference's own ceiling"
```

---

### Task 4: `evaluate_map`, the adaptor method, and the route

**Files:**
- Modify: `backend/xcell/localize_metrics.py`
- Modify: `backend/tests/test_localize_metrics.py`
- Modify: `backend/xcell/adaptor.py` — add `evaluate_localization_maps` next to `evaluate_localization` (~line 8153, where `spatial_reference_bundle` is consumed)
- Modify: `backend/xcell/api/routes.py` — add the route after `/localize/evaluate`
- Create: `backend/tests/test_localize_metrics_routes.py`

**Interfaces:**
- Consumes: all four metrics (Tasks 1–3).
- Produces:
  - `evaluate_map(ref_coords, pred_coords, marker_sets) -> {'dispersion': dict, 'occupancy': dict, 'markers': [{'name', 'pattern', 'axis'}]}`
  - `DataAdaptor.evaluate_localization_maps(bundle, embeddings: list[str], gene_sets: dict[str, list[str]]) -> dict`
  - `POST /api/localize/evaluate_map`

- [ ] **Step 1: Write the failing test for the orchestrator**

Append to `backend/tests/test_localize_metrics.py`:

```python
from xcell.localize_metrics import evaluate_map


def test_evaluate_map_bundles_every_metric_per_marker_set():
    ref, pred = _disc(seed=0), _disc(seed=1)
    out = evaluate_map(ref, pred, [
        {'name': 'skin', 'ref_scores': _annulus_scores(ref),
         'pred_scores': _annulus_scores(pred)},
        {'name': 'distal', 'ref_scores': ref[:, 0], 'pred_scores': pred[:, 0]},
    ])
    assert set(out) == {'dispersion', 'occupancy', 'markers'}
    assert [m['name'] for m in out['markers']] == ['skin', 'distal']
    assert out['markers'][0]['pattern']['correlation'] > 0.7
    assert out['markers'][1]['axis']['prediction'] > 0.9
    assert out['dispersion']['area_ratio'] == pytest.approx(1.0, abs=0.15)


def test_evaluate_map_is_json_safe_with_no_marker_sets():
    import json
    ref, pred = _disc(seed=0), _disc(seed=1)
    json.dumps(evaluate_map(ref, pred, []))
```

- [ ] **Step 2: Run it and watch it fail**

```bash
cd backend && pixi run -e dev pytest tests/test_localize_metrics.py -q -k evaluate_map
```

Expected: `ImportError: cannot import name 'evaluate_map'`.

- [ ] **Step 3: Implement the orchestrator**

Append to `backend/xcell/localize_metrics.py`:

```python
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
```

- [ ] **Step 4: Add the adaptor method**

In `backend/xcell/adaptor.py`, beside the other Localize methods:

```python
    def evaluate_localization_maps(
        self,
        bundle: dict[str, Any],
        embeddings: list[str],
        gene_sets: dict[str, list[str]],
    ) -> dict[str, Any]:
        """Score one or more predicted maps against the spatial reference.

        Scoring several at once is the point: the value of these numbers is
        comparative, and a user arrives here with a handful of saved variants
        and no way to rank them.

        Args:
            bundle: from the reference's ``spatial_reference_bundle``.
            embeddings: obsm keys on *this* (query) dataset holding predictions.
            gene_sets: marker name -> gene list. Genes absent from either
                dataset are dropped; a set left with none is skipped and named
                in ``skipped``.
        """
        from xcell import localize_metrics as lm

        ref_coords = np.asarray(bundle['coords'], dtype=float)
        ref_expr = np.asarray(bundle['expr'], dtype=np.float32)
        ref_genes = {g: i for i, g in enumerate(bundle['genes'])}

        query_matrix = self._resolve_source_matrix(None)

        marker_sets: list[dict[str, Any]] = []
        skipped: list[str] = []
        for name, genes in gene_sets.items():
            present, _ = self._split_present_genes(list(genes))
            cols = [g for g in present if g in ref_genes]
            if not cols:
                skipped.append(str(name))
                continue
            q = query_matrix[:, [self.adata.var_names.get_loc(g) for g in cols]]
            q = q.toarray() if hasattr(q, 'toarray') else np.asarray(q)
            marker_sets.append({
                'name': name,
                'ref_scores': ref_expr[:, [ref_genes[g] for g in cols]].mean(axis=1),
                'pred_scores': np.asarray(q, dtype=float).mean(axis=1),
            })

        results = []
        for key in embeddings:
            if key not in self.adata.obsm:
                raise ValueError(f"Embedding '{key}' not found in obsm.")
            pred = np.asarray(self.adata.obsm[key], dtype=float)[:, :2]
            results.append({'embedding': key,
                            **lm.evaluate_map(ref_coords, pred, marker_sets)})

        return {'maps': results, 'skipped_gene_sets': skipped,
                'n_reference_cells': int(ref_coords.shape[0])}
```

Check `_resolve_source_matrix` and `_split_present_genes` exist with those
names before relying on them: `rg -n "def _resolve_source_matrix|def _split_present_genes" backend/xcell/adaptor.py`.

- [ ] **Step 5: Write the failing route test**

Create `backend/tests/test_localize_metrics_routes.py`:

```python
"""Route contract for scoring predicted Localize maps."""

import numpy as np
import anndata
from fastapi.testclient import TestClient
from scipy.sparse import csr_matrix

from xcell.main import app
from xcell.api import routes
from xcell.adaptor import DataAdaptor

client = TestClient(app)


def _spatial_ref(n_side=20):
    coords = np.array([[float(i), float(j)]
                       for i in range(n_side) for j in range(n_side)])
    n = len(coords)
    rng = np.random.default_rng(0)
    X = rng.poisson(1.0, (n, 4)).astype(np.float32)
    r = np.linalg.norm(coords - coords.mean(0), axis=1)
    X[r > np.quantile(r, 0.75), 0] += 20.0          # 'rim' gene on the outside
    ad = anndata.AnnData(X=csr_matrix(X))
    ad.var_names = ['Rim', 'B', 'C', 'D']
    ad.obsm['X_spatial'] = coords
    return ad


def _query(n=200):
    rng = np.random.default_rng(1)
    X = rng.poisson(1.0, (n, 4)).astype(np.float32)
    X[: n // 4, 0] += 20.0                          # a quarter are 'rim' cells
    ad = anndata.AnnData(X=csr_matrix(X))
    ad.var_names = ['Rim', 'B', 'C', 'D']
    ad.obsm['X_good'] = np.column_stack([
        rng.random(n) * 19, rng.random(n) * 19,     # spread over the tissue
    ])
    ad.obsm['X_collapsed'] = np.full((n, 2), 9.5) + rng.normal(0, 0.3, (n, 2))
    return ad


def _install():
    routes.set_adaptor(DataAdaptor('ref.h5ad', adata=_spatial_ref()), slot='primary')
    routes.set_adaptor(DataAdaptor('qry.h5ad', adata=_query()), slot='secondary')


def _body(embeddings):
    return {'reference': 'primary', 'dataset': 'secondary',
            'embeddings': embeddings, 'gene_sets': {'rim': ['Rim']}}


def test_evaluate_map_scores_each_embedding():
    _install()
    r = client.post('/api/localize/evaluate_map?dataset=secondary',
                    json=_body(['X_good', 'X_collapsed']))
    assert r.status_code == 200, r.text
    body = r.json()
    assert [m['embedding'] for m in body['maps']] == ['X_good', 'X_collapsed']
    for m in body['maps']:
        assert set(m) >= {'embedding', 'dispersion', 'occupancy', 'markers'}


def test_a_collapsed_map_scores_a_lower_area_ratio():
    _install()
    r = client.post('/api/localize/evaluate_map?dataset=secondary',
                    json=_body(['X_good', 'X_collapsed'])).json()
    good, bad = r['maps'][0]['dispersion'], r['maps'][1]['dispersion']
    assert bad['area_ratio'] < good['area_ratio']


def test_unknown_embedding_is_a_400():
    _install()
    r = client.post('/api/localize/evaluate_map?dataset=secondary',
                    json=_body(['X_nope']))
    assert r.status_code == 400
    assert 'X_nope' in r.json()['detail']


def test_a_gene_set_with_no_usable_genes_is_reported_not_fatal():
    _install()
    r = client.post('/api/localize/evaluate_map?dataset=secondary', json={
        'reference': 'primary', 'dataset': 'secondary', 'embeddings': ['X_good'],
        'gene_sets': {'rim': ['Rim'], 'nonsense': ['NotAGene']},
    })
    assert r.status_code == 200
    assert r.json()['skipped_gene_sets'] == ['nonsense']
    assert [m['name'] for m in r.json()['maps'][0]['markers']] == ['rim']


def test_response_is_json_safe():
    import json
    _install()
    r = client.post('/api/localize/evaluate_map?dataset=secondary',
                    json=_body(['X_good']))
    json.dumps(r.json())
```

Confirm the installer idiom against an existing route test first:
`rg -n "set_adaptor" backend/tests/*.py | head`.

- [ ] **Step 6: Run it and watch it fail**

```bash
cd backend && pixi run -e dev pytest tests/test_localize_metrics_routes.py -q
```

Expected: 404 — the route does not exist.

- [ ] **Step 7: Add the route**

In `backend/xcell/api/routes.py`, after the `/localize/evaluate` route, following
`LocalizeRequest`'s shape:

```python
class EvaluateMapRequest(BaseModel):
    reference: str = "primary"
    embeddings: list[str]
    gene_sets: dict[str, list[str]] = {}
    dataset: str | None = None


@router.post("/localize/evaluate_map")
def localize_evaluate_map(
    request: EvaluateMapRequest, dataset: str | None = Query(None),
):
    """Score predicted maps against the spatial reference.

    Several embeddings at once, because these numbers are only meaningful
    comparatively — a user arrives with a handful of saved variants and no way
    to rank them.
    """
    query_slot = request.dataset or dataset
    query = get_adaptor(query_slot)
    reference = _resolve_reference(request.reference, query_slot)
    try:
        bundle = reference.spatial_reference_bundle()
        return query.evaluate_localization_maps(
            bundle, request.embeddings, request.gene_sets,
        )
    except HTTPException:
        raise
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
```

- [ ] **Step 8: Run both test files, then the whole suite**

```bash
cd backend && pixi run -e dev pytest tests/test_localize_metrics.py tests/test_localize_metrics_routes.py -q
cd backend && pixi run -e dev pytest -q
```

Expected: all pass, no regressions.

- [ ] **Step 9: Commit**

```bash
git add backend/xcell/localize_metrics.py backend/xcell/adaptor.py \
        backend/xcell/api/routes.py backend/tests/test_localize_metrics.py \
        backend/tests/test_localize_metrics_routes.py
git commit -m "feat(localize): score predicted maps against the reference"
```

---

### Task 5: The Map quality table

**Files:**
- Modify: `frontend/src/components/LocalizeModal.tsx` — a section after "Accuracy"

**Interfaces:**
- Consumes: `POST /api/localize/evaluate_map` (Task 4); `roles.referenceSlot` / `roles.querySlot` and `assignRoles` already in the modal.
- Produces: no exported names.

- [ ] **Step 1: Add the types and the fetch**

```tsx
interface MapMetrics {
  embedding: string
  dispersion: { area_ratio: number | null; std_ratio_x: number | null
                std_ratio_y: number | null; frac_outside: number | null }
  occupancy: { n_distinct: number; max_per_spot: number
               effective_n: number | null; n_pred: number }
  markers: { name: string
             pattern: { correlation: number | null; n_bins_compared: number }
             axis: { reference: number | null; prediction: number | null } }[]
}

const [mapMetrics, setMapMetrics] = useState<MapMetrics[] | null>(null)
const [scoring, setScoring] = useState(false)

const scoreMaps = async () => {
  if (!reference || !querySlot) return
  setScoring(true); setError(null)
  try {
    // Every gene set curated in the Gene panel, as marker sets.
    const sets: Record<string, string[]> = {}
    geneSets.forEach((g) => { if (g.genes.length) sets[g.name] = g.genes })
    const embeddings = (useStore.getState().datasets[querySlot as DatasetSlot]
      ?.schema?.embeddings || []).filter((e) => e !== 'spatial' && e !== 'X_spatial')
    const r = await fetch(
      appendDataset(`${API}/localize/evaluate_map`, querySlot as DatasetSlot),
      { method: 'POST', headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ reference, dataset: querySlot, embeddings,
                               gene_sets: sets }) },
    )
    const payload = await r.json()
    if (!r.ok) throw new Error(payload.detail || `HTTP ${r.status}`)
    setMapMetrics(payload.maps)
  } catch (e) {
    setError(e instanceof Error ? e.message : String(e))
  } finally { setScoring(false) }
}
```

Check how `geneSets` is already derived in this file (`rg -n "geneSets" frontend/src/components/LocalizeModal.tsx`) and reuse it rather than re-deriving.

- [ ] **Step 2: Render the table**

After the Accuracy block:

```tsx
<div style={{ ...sectionLabel, marginTop: 16 }}>Map quality</div>
<div style={{ fontSize: 11, color: dark.sub, marginBottom: 6 }}>
  Scores every predicted embedding in the query against the reference. The
  numbers are comparative — a pattern score means little alone, and an axis
  correlation means nothing until you see what the reference itself reaches.
</div>
<button onClick={scoreMaps} disabled={scoring || !ready} style={ghost}>
  {scoring ? 'Scoring…' : 'Score predicted maps'}
</button>

{mapMetrics && mapMetrics.length > 0 && (
  <table style={{ borderCollapse: 'collapse', width: '100%', fontSize: 11,
                  marginTop: 8 }}>
    <thead>
      <tr style={{ color: dark.faint, textAlign: 'left' }}>
        <th style={th}>embedding</th>
        <th style={th}>area</th>
        <th style={th}>outside</th>
        <th style={th}>spots used</th>
        {mapMetrics[0].markers.map((m) => <th key={m.name} style={th}>{m.name}</th>)}
      </tr>
    </thead>
    <tbody>
      {mapMetrics.map((m) => (
        <tr key={m.embedding}>
          <td style={td}><code>{m.embedding}</code></td>
          <td style={{ ...td, color: areaColor(m.dispersion.area_ratio) }}>
            {fmt(m.dispersion.area_ratio)}
          </td>
          <td style={td}>{pct(m.dispersion.frac_outside)}</td>
          <td style={td}>
            {m.occupancy.n_distinct.toLocaleString()}
            <span style={{ color: dark.faint }}> / {m.occupancy.n_pred.toLocaleString()}</span>
          </td>
          {m.markers.map((k) => (
            <td key={k.name} style={{ ...td, color: patternColor(k.pattern.correlation) }}
                title={`axis: ${fmt(k.axis.prediction)} of ${fmt(k.axis.reference)} possible`}>
              {fmt(k.pattern.correlation)}
            </td>
          ))}
        </tr>
      ))}
    </tbody>
  </table>
)}
```

With these helpers beside the component:

```tsx
const fmt = (v: number | null) => (v == null ? '—' : v.toFixed(2))
const pct = (v: number | null) => (v == null ? '—' : `${Math.round(v * 100)}%`)
// 1.0 is right; the limb failure was 0.15 one way and 2.93 the other.
const areaColor = (v: number | null) =>
  v == null ? dark.faint : (v < 0.5 || v > 1.8) ? '#f0c987' : dark.text
// Negative means the cell type was placed where it is not — the epidermis
// inversion. That is a different kind of wrong from merely weak.
const patternColor = (v: number | null) =>
  v == null ? dark.faint : v < 0 ? '#ff8fa3' : v < 0.2 ? '#f0c987' : dark.text
```

`th` and `td` already exist in this file (used by `ParameterGuide`); reuse them.

- [ ] **Step 3: Typecheck and test**

```bash
cd frontend && npx tsc --noEmit && npm test
```

- [ ] **Step 4: Commit**

```bash
git add frontend/src/components/LocalizeModal.tsx
git commit -m "feat(localize): Map quality table for comparing predicted maps"
```

---

### Task 6: Verify on the limb pair, and docs

The harness only earns its place if it reproduces, as numbers, the failure that
was diagnosed by eye. This task is the proof.

**Files:**
- Modify: `README.md`, `CHANGELOG.md`

- [ ] **Step 1: Start both servers**

```bash
pixi run backend      # :8000
pixi run dev          # :5173
```

Stop later with `pkill -f "uvicorn xcell.main:app"; pkill -f vite` — killing by
port leaves a reloader that re-binds and serves stale data. Editing a backend
file restarts the server and drops the loaded datasets, so batch any fixes.

- [ ] **Step 2: Load the limb pair**

Reference `E11.5_best_102424.h5ad` and query `GSM4227224_E11_112125.h5ad`
(paths in the spec). The query already carries eight predicted embeddings from
earlier runs.

- [ ] **Step 3: Score them and check the numbers reproduce the diagnosis**

Open Localize → **Score predicted maps**. The panel must show, at minimum:

- `X_spatial_pred` (weighted_mean) with an **area ratio well under 0.5**, and a
  **negative** pattern score for the `skin` set — the epidermis inversion.
- `X_spatial_pred_best` with an area ratio near 1 and a **higher** skin pattern
  score, since best_match recovered the rim.

If the skin pattern score for `X_spatial_pred` is not negative, the metric is
not detecting what the picture plainly shows — investigate the grid resolution
before accepting it. That is the whole point of this phase.

- [ ] **Step 4: Cross-check one number outside the UI**

```bash
curl -s -X POST "localhost:8000/api/localize/evaluate_map?dataset=secondary" \
  -H 'Content-Type: application/json' \
  -d '{"reference":"primary","dataset":"secondary",
       "embeddings":["X_spatial_pred","X_spatial_pred_best"],
       "gene_sets":{"skin":["Perp","Krt18","Krt14","Krt5","Epcam","Cdh1"]}}' \
  | python3 -m json.tool | head -40
```

Expected: two `maps` entries, `X_spatial_pred` with the lower area ratio and the
lower skin pattern correlation.

- [ ] **Step 5: Stop the servers and write the docs**

`README.md` — in the Localize section, describe Map quality: what the four
numbers mean, and that the axis figure is reported against the reference's own
ceiling rather than as an absolute.

`CHANGELOG.md` — an entry under **Added** covering the harness, the four
metrics, why they are rank-based, and the limb measurements that motivated
them.

- [ ] **Step 6: Commit and merge**

```bash
git add README.md CHANGELOG.md
git commit -m "docs: Localize map-quality metrics"
git checkout main
git merge --no-ff feat/localize-metrics
```

---

## Self-Review

**Spec coverage.** Spatial pattern fidelity → Task 2. Axis fidelity, orientation-free via the reference's direction → Task 3. Dispersion including `frac_outside` for the overshoot case → Task 1. Occupancy including effective-n → Task 1. Pure module, arrays in/dicts out → Task 1's docstring and the module's imports. Orchestrator, adaptor, route → Task 4. Comparison across several embeddings, so the eight existing variants can be ranked → Task 4 route shape and Task 5 table. Verification against the limb diagnosis → Task 6.

**Deliberately deferred to later phases**, per the spec's phasing: injective assignment, the OT mode, and changing the `weighted_mean` default. This plan builds only what makes those decidable.

**Names used consistently:** `axis_direction`, `axis_fidelity`, `spatial_pattern_fidelity`, `dispersion`, `occupancy`, `evaluate_map`, `_f`, `_rank_weights`, `_density`, `_inner_hull_area`, `evaluate_localization_maps`, `EvaluateMapRequest`. The dict keys returned in Tasks 1–3 are the same ones consumed in Task 4's orchestrator and Task 5's `MapMetrics` type.

**Known soft spots**, each with an in-plan instruction rather than a silent assumption: the exact names of `_resolve_source_matrix` / `_split_present_genes` (Task 4 Step 4), the route-test installer idiom (Task 4 Step 5), how `geneSets` is already derived in the modal (Task 5 Step 1), and the grid resolution needed to keep a thin rim from blurring away (Task 2 Step 4 and Task 6 Step 3).
