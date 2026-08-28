# ST Spot Merging Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Merge neighbouring ST spots inside a user-drawn region into single spots that approximate one large cell, producing a new sibling dataset.

**Architecture:** A pure module (`spot_merge.py`) does greedy agglomeration over a Delaunay neighbour graph, capped by cell diameter and vetoed by profile correlation. `DataAdaptor.merge_spots()` extracts arrays, calls it, and assembles a merged AnnData without touching the source. `POST /api/spots/merge` registers the result under a new slot, following `/combine`.

**Tech Stack:** Python 3.11, numpy, scipy (`Delaunay`, `cKDTree`), anndata, FastAPI, pytest; React 18 + Zustand + vitest.

**Spec:** `docs/superpowers/specs/2026-08-28-st-spot-merging-design.md`

## Global Constraints

- The source dataset is **never** mutated. `merge_spots` is read-only on `self`.
- The merge sums **raw counts**; the veto correlates **CPM+log1p** profiles.
- Merged datasets carry **no** numeric or QC `.obs`, and **no** `.obsm` except the spatial key.
- Defaults: `max_diameter_um=40`, `min_correlation=0.15`, `eligibility='all'`, `dry_run=True`.
- `max_spots` derives as `ceil(pi * (diameter/2)**2 / pitch**2)` unless given.
- Determinism: identical input yields identical labels. Distance ties break on the two groups' lowest member source indices as a sorted pair.
- Pure modules never import the adaptor (CLAUDE.md). Routes stay thin; `ValueError -> 400`.
- Tests: no `conftest.py`; module-level `_adata()` / `_adaptor()` factories; seed `np.random.default_rng(0)`; `csr_matrix` + `float32`.
- Run backend tests from `backend/`: `pixi run -e dev pytest tests/<file> -q`.
- Frontend: `npx vitest run <file>` and `npx tsc --noEmit` from `frontend/`.

---

### Task 1: Pitch estimation and max_spots

**Files:**
- Create: `backend/xcell/spot_merge.py`
- Test: `backend/tests/test_spot_merge.py`

**Interfaces:**
- Consumes: nothing.
- Produces: `estimate_pitch_um(coords: np.ndarray) -> float`, `max_spots_for(diameter_um: float, pitch_um: float) -> int`. `coords` is `(n, 2)` float in **microns**.

- [ ] **Step 1: Write the failing test**

```python
"""Merging ST spots into single large cells: the pure algorithm."""

import numpy as np
import pytest

from xcell.spot_merge import estimate_pitch_um, max_spots_for


def _grid(nx: int, ny: int, pitch: float = 15.0) -> np.ndarray:
    """A regular lattice, the shape a real ST slide has."""
    xs, ys = np.meshgrid(np.arange(nx) * pitch, np.arange(ny) * pitch)
    return np.column_stack([xs.ravel(), ys.ravel()]).astype(float)


def test_pitch_is_the_lattice_spacing():
    assert estimate_pitch_um(_grid(6, 6, pitch=15.0)) == pytest.approx(15.0)


def test_pitch_survives_jitter_and_a_few_gaps():
    rng = np.random.default_rng(0)
    coords = _grid(8, 8, pitch=20.0)
    coords += rng.normal(0, 0.5, coords.shape)      # imaging jitter
    coords = np.delete(coords, [3, 17, 40], axis=0)  # missing spots
    assert estimate_pitch_um(coords) == pytest.approx(20.0, abs=1.5)


def test_max_spots_counts_lattice_points_in_the_disc():
    # A 40 um disc over a 15 um lattice: pi*20^2 / 15^2 = 5.58 -> 6
    assert max_spots_for(40.0, 15.0) == 6
    # Diameter equal to pitch admits exactly one spot
    assert max_spots_for(15.0, 15.0) == 1


def test_max_spots_is_never_below_one():
    assert max_spots_for(1.0, 15.0) == 1


def test_pitch_needs_two_spots():
    with pytest.raises(ValueError, match="at least two"):
        estimate_pitch_um(np.array([[0.0, 0.0]]))
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd backend && pixi run -e dev pytest tests/test_spot_merge.py -q`
Expected: FAIL — `ModuleNotFoundError: No module named 'xcell.spot_merge'`

- [ ] **Step 3: Write minimal implementation**

```python
"""Merging neighbouring ST spots into single large cells.

A spot is a patch of tissue, not a cell. Where the cell is much larger than
the spot — a hypertrophic chondrocyte against a Visium HD bin — one cell is
spread across several spots, each holding a fraction of its transcripts.

This module puts the pieces back: it groups neighbouring spots up to the size
of one cell so their counts can be summed. Geometry drives the grouping,
because the low-count profiles that motivate merging are exactly the ones too
noisy to decide it; similarity enters only as a veto.

Pure: arrays in, arrays out. Knows nothing of AnnData.
"""
from __future__ import annotations

import math

import numpy as np


def estimate_pitch_um(coords: np.ndarray) -> float:
    """Spot spacing, as the median nearest-neighbour distance.

    Measured rather than assumed: it costs one KD-tree query and turns a wrong
    um_per_unit into an obviously absurd number instead of a silently wrong cap.
    """
    if len(coords) < 2:
        raise ValueError("Estimating spot pitch needs at least two spots.")
    from scipy.spatial import cKDTree

    # k=2 because the nearest neighbour of a point is itself, at distance 0.
    distances, _ = cKDTree(coords).query(coords, k=2)
    return float(np.median(distances[:, 1]))


def max_spots_for(diameter_um: float, pitch_um: float) -> int:
    """How many lattice points a disc of this diameter covers."""
    area_ratio = math.pi * (diameter_um / 2.0) ** 2 / (pitch_um ** 2)
    return max(1, int(math.ceil(area_ratio)))
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd backend && pixi run -e dev pytest tests/test_spot_merge.py -q`
Expected: PASS (5 tests)

- [ ] **Step 5: Commit**

```bash
git add backend/xcell/spot_merge.py backend/tests/test_spot_merge.py
git commit -m "feat: estimate ST spot pitch and derive the merge size cap"
```

---

### Task 2: Greedy agglomeration with diameter cap and correlation veto

**Files:**
- Modify: `backend/xcell/spot_merge.py`
- Test: `backend/tests/test_spot_merge.py`

**Interfaces:**
- Consumes: `estimate_pitch_um`, `max_spots_for` from Task 1.
- Produces: `MergeParams` dataclass; `merge_spots(coords, counts, normalized, params) -> MergeResult` where `MergeResult` has fields `labels: np.ndarray` (int, `-1` = not merged/ineligible), `n_groups: int`, `pitch_um: float`, `max_spots: int`, `size_histogram: dict[int, int]`, `n_vetoed: int`, `correlation_quantiles: dict[str, float]`. `counts` and `normalized` are dense `(n, g)` float arrays over the region's spots only.

- [ ] **Step 1: Write the failing test**

Append to `backend/tests/test_spot_merge.py`:

```python
from xcell.spot_merge import MergeParams, merge_spots


def _profiles(n: int, g: int = 8, kind: int = 0, rng=None) -> np.ndarray:
    """Counts whose gene pattern depends on `kind`, so types differ."""
    rng = rng or np.random.default_rng(0)
    base = np.full((n, g), 1.0)
    base[:, kind * 2:(kind + 1) * 2] += 20.0     # a type-specific block
    return base + rng.poisson(0.5, (n, g))


def _norm(counts: np.ndarray) -> np.ndarray:
    """CPM + log1p, the matrix the veto sees."""
    depth = counts.sum(axis=1, keepdims=True)
    depth[depth == 0] = 1.0
    return np.log1p(counts / depth * 1e4)


def _run(coords, counts, **kw):
    params = MergeParams(**kw)
    return merge_spots(coords, counts, _norm(counts), params)


def test_groups_respect_the_diameter_cap():
    coords = _grid(6, 6, pitch=15.0)
    counts = _profiles(36)
    r = _run(coords, counts, max_diameter_um=40.0)
    for gid in range(r.n_groups):
        members = coords[r.labels == gid]
        if len(members) < 2:
            continue
        pairwise = np.linalg.norm(members[:, None, :] - members[None, :, :], axis=-1)
        assert pairwise.max() <= 40.0 + 1e-9


def test_every_spot_lands_in_exactly_one_group():
    coords = _grid(5, 5, pitch=15.0)
    r = _run(coords, _profiles(25), max_diameter_um=40.0)
    assert (r.labels >= 0).all()
    assert sorted(set(r.labels.tolist())) == list(range(r.n_groups))


def test_merging_actually_reduces_the_spot_count():
    coords = _grid(6, 6, pitch=15.0)
    r = _run(coords, _profiles(36), max_diameter_um=40.0)
    assert r.n_groups < 36


def test_the_veto_blocks_merging_across_a_planted_boundary():
    # Left half is type 0, right half type 1; they must never share a group.
    coords = _grid(6, 4, pitch=15.0)
    left = coords[:, 0] < 45.0
    counts = np.zeros((len(coords), 8))
    counts[left] = _profiles(int(left.sum()), kind=0)
    counts[~left] = _profiles(int((~left).sum()), kind=1)

    r = _run(coords, counts, max_diameter_um=40.0, min_correlation=0.5)
    for gid in range(r.n_groups):
        members = r.labels == gid
        assert len(set(left[members].tolist())) == 1, "group spans the boundary"
    assert r.n_vetoed > 0


def test_a_permissive_veto_does_not_block_within_one_type():
    coords = _grid(4, 4, pitch=15.0)
    r = _run(coords, _profiles(16), max_diameter_um=40.0, min_correlation=0.15)
    assert r.n_vetoed == 0
    assert r.n_groups < 16


def test_max_spots_caps_group_size():
    coords = _grid(6, 6, pitch=15.0)
    r = _run(coords, _profiles(36), max_diameter_um=1000.0, max_spots=3)
    sizes = np.bincount(r.labels)
    assert sizes.max() <= 3


def test_the_same_input_gives_the_same_labels_every_time():
    coords = _grid(6, 6, pitch=15.0)
    counts = _profiles(36)
    a = _run(coords, counts, max_diameter_um=40.0)
    b = _run(coords, counts, max_diameter_um=40.0)
    np.testing.assert_array_equal(a.labels, b.labels)


def test_reports_what_the_veto_did():
    coords = _grid(6, 4, pitch=15.0)
    left = coords[:, 0] < 45.0
    counts = np.zeros((len(coords), 8))
    counts[left] = _profiles(int(left.sum()), kind=0)
    counts[~left] = _profiles(int((~left).sum()), kind=1)
    r = _run(coords, counts, max_diameter_um=40.0, min_correlation=0.5)
    q = r.correlation_quantiles
    assert set(q) == {"p10", "p50", "p90"}
    assert -1.0 <= q["p10"] <= q["p50"] <= q["p90"] <= 1.0


def test_a_single_spot_is_left_alone():
    r = _run(np.array([[0.0, 0.0]]), _profiles(1), max_diameter_um=40.0)
    assert r.n_groups == 1
    assert r.labels.tolist() == [0]


def test_collinear_spots_do_not_defeat_the_graph():
    # Delaunay raises on degenerate input; a row of spots is a real slide edge.
    coords = np.column_stack([np.arange(5) * 15.0, np.zeros(5)])
    r = _run(coords, _profiles(5), max_diameter_um=40.0)
    assert r.n_groups < 5


def test_size_histogram_accounts_for_every_spot():
    coords = _grid(5, 5, pitch=15.0)
    r = _run(coords, _profiles(25), max_diameter_um=40.0)
    assert sum(size * n for size, n in r.size_histogram.items()) == 25
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd backend && pixi run -e dev pytest tests/test_spot_merge.py -q`
Expected: FAIL — `ImportError: cannot import name 'MergeParams'`

- [ ] **Step 3: Write minimal implementation**

Append to `backend/xcell/spot_merge.py`:

```python
import heapq
from dataclasses import dataclass, field


@dataclass
class MergeParams:
    max_diameter_um: float = 40.0
    max_spots: int | None = None          # None -> derived from pitch
    min_correlation: float = 0.15
    eligibility: str = "all"              # 'all' | 'min_counts' | 'quantile'
    min_counts: float | None = None
    quantile: float | None = None


@dataclass
class MergeResult:
    labels: np.ndarray
    n_groups: int
    pitch_um: float
    max_spots: int
    size_histogram: dict[int, int] = field(default_factory=dict)
    n_vetoed: int = 0
    correlation_quantiles: dict[str, float] = field(default_factory=dict)


def _neighbour_edges(coords: np.ndarray, max_distance: float) -> list[tuple[float, int, int]]:
    """Delaunay edges no longer than max_distance, shortest first.

    Delaunay rather than a fixed radius: it adapts to irregular spacing without
    being told the pitch. It needs a 2-D simplex to exist, so collinear or
    tiny inputs — a single row of spots at a slide edge is neither exotic nor
    an error — fall back to all pairs within range.
    """
    from scipy.spatial import Delaunay, QhullError

    pairs: set[tuple[int, int]] = set()
    if len(coords) >= 4:
        try:
            tri = Delaunay(coords)
            for simplex in tri.simplices:
                for i in range(len(simplex)):
                    for j in range(i + 1, len(simplex)):
                        a, b = int(simplex[i]), int(simplex[j])
                        pairs.add((min(a, b), max(a, b)))
        except QhullError:
            pairs = set()
    if not pairs:
        pairs = {(i, j) for i in range(len(coords)) for j in range(i + 1, len(coords))}

    edges = []
    for a, b in pairs:
        d = float(np.linalg.norm(coords[a] - coords[b]))
        if d <= max_distance:
            edges.append((d, a, b))
    return edges


def _correlation(a: np.ndarray, b: np.ndarray) -> float:
    """Pearson r between two profiles; 0 when either is flat."""
    a_c, b_c = a - a.mean(), b - b.mean()
    denom = float(np.linalg.norm(a_c) * np.linalg.norm(b_c))
    if denom < 1e-12:
        return 0.0
    return float(np.dot(a_c, b_c) / denom)


def merge_spots(
    coords: np.ndarray,
    counts: np.ndarray,
    normalized: np.ndarray,
    params: MergeParams,
) -> MergeResult:
    """Group neighbouring spots into cell-sized clusters.

    Greedy agglomeration over a neighbour graph: repeatedly join the closest
    adjacent pair whose merged footprint still fits inside one cell and whose
    profiles are not clearly different. Constrained hierarchical clustering
    with a diameter cap, which is deterministic and does not let noisy
    low-count correlations decide the grouping.

    Args:
        coords: (n, 2) spot positions in microns.
        counts: (n, g) raw counts — summed, so integer-like.
        normalized: (n, g) CPM+log1p profiles — correlated, never summed.
        params: see MergeParams.

    Returns:
        MergeResult. `labels` gives each input spot its group id.
    """
    n = len(coords)
    pitch = estimate_pitch_um(coords) if n >= 2 else params.max_diameter_um
    max_spots = params.max_spots or max_spots_for(params.max_diameter_um, pitch)

    parent = list(range(n))
    members: dict[int, list[int]] = {i: [i] for i in range(n)}
    group_counts = counts.astype(float).copy()
    group_norm = normalized.astype(float).copy()

    def find(i: int) -> int:
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    heap = [
        (d, min(a, b), max(a, b), a, b)
        for d, a, b in _neighbour_edges(coords, params.max_diameter_um)
    ]
    heapq.heapify(heap)

    n_vetoed = 0
    correlations: list[float] = []

    while heap:
        _, _, _, ra, rb = heapq.heappop(heap)
        ga, gb = find(ra), find(rb)
        if ga == gb:
            continue

        joined = members[ga] + members[gb]
        if len(joined) > max_spots:
            continue
        pts = coords[joined]
        spread = np.linalg.norm(pts[:, None, :] - pts[None, :, :], axis=-1).max()
        if spread > params.max_diameter_um:
            continue

        r = _correlation(group_norm[ga], group_norm[gb])
        correlations.append(r)
        if r < params.min_correlation:
            n_vetoed += 1
            continue

        # Union by keeping the lower id, so labels stay tied to source order.
        keep, drop = (ga, gb) if ga < gb else (gb, ga)
        parent[drop] = keep
        members[keep] = joined
        del members[drop]
        group_counts[keep] = group_counts[ga] + group_counts[gb]
        # The merged profile is the summed counts renormalized, not a mean of
        # profiles: the veto must judge the group as it will actually exist.
        depth = group_counts[keep].sum()
        group_norm[keep] = np.log1p(group_counts[keep] / (depth or 1.0) * 1e4)

        for member in joined:
            centroid = coords[joined].mean(axis=0)
            del centroid  # centroid is recomputed by the caller; see merge_spots docs
            break
        for d, a, b in _neighbour_edges(coords[joined], params.max_diameter_um):
            del d, a, b
            break

    roots = sorted({find(i) for i in range(n)})
    remap = {root: gid for gid, root in enumerate(roots)}
    labels = np.array([remap[find(i)] for i in range(n)], dtype=int)

    sizes = np.bincount(labels)
    histogram = {int(size): int(count) for size, count in
                 zip(*np.unique(sizes, return_counts=True))}

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
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd backend && pixi run -e dev pytest tests/test_spot_merge.py -q`
Expected: PASS (16 tests). Remove the two dead `for ... break` blocks flagged during implementation — they are scaffolding, not logic; the loop needs no per-merge edge push because every Delaunay edge is already in the heap and stale ones are skipped by the `ga == gb` guard.

- [ ] **Step 5: Commit**

```bash
git add backend/xcell/spot_merge.py backend/tests/test_spot_merge.py
git commit -m "feat: greedy spot agglomeration with a diameter cap and correlation veto"
```

---

### Task 3: Eligibility modes

**Files:**
- Modify: `backend/xcell/spot_merge.py`
- Test: `backend/tests/test_spot_merge.py`

**Interfaces:**
- Consumes: `MergeParams`, `merge_spots` from Task 2.
- Produces: `eligible_mask(counts: np.ndarray, params: MergeParams) -> np.ndarray` (bool, length n). `merge_spots` gives ineligible spots their own singleton group.

- [ ] **Step 1: Write the failing test**

```python
from xcell.spot_merge import eligible_mask


def test_all_is_the_default_and_takes_everything():
    counts = _profiles(9)
    assert eligible_mask(counts, MergeParams()).all()


def test_min_counts_takes_only_the_shallow_spots():
    counts = np.ones((4, 3))
    counts[0] *= 10.0   # depth 30
    counts[1] *= 1.0    # depth 3
    counts[2] *= 100.0  # depth 300
    counts[3] *= 2.0    # depth 6
    mask = eligible_mask(counts, MergeParams(eligibility="min_counts", min_counts=10))
    assert mask.tolist() == [False, True, False, True]


def test_quantile_takes_the_shallow_fraction():
    counts = np.ones((10, 2)) * np.arange(1, 11)[:, None]
    mask = eligible_mask(counts, MergeParams(eligibility="quantile", quantile=0.3))
    assert mask.sum() == 3
    assert mask[:3].all()


def test_an_unknown_eligibility_mode_is_refused():
    with pytest.raises(ValueError, match="eligibility"):
        eligible_mask(_profiles(4), MergeParams(eligibility="vibes"))


def test_min_counts_mode_needs_a_threshold():
    with pytest.raises(ValueError, match="min_counts"):
        eligible_mask(_profiles(4), MergeParams(eligibility="min_counts"))


def test_ineligible_spots_pass_through_unmerged():
    coords = _grid(4, 4, pitch=15.0)
    counts = _profiles(16)
    counts[5] *= 1000.0        # far too deep to be eligible
    r = _run(coords, counts, max_diameter_um=40.0,
             eligibility="min_counts", min_counts=1000.0)
    assert (r.labels == r.labels[5]).sum() == 1, "a deep spot was absorbed"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd backend && pixi run -e dev pytest tests/test_spot_merge.py -q`
Expected: FAIL — `ImportError: cannot import name 'eligible_mask'`

- [ ] **Step 3: Write minimal implementation**

Add to `backend/xcell/spot_merge.py`:

```python
def eligible_mask(counts: np.ndarray, params: MergeParams) -> np.ndarray:
    """Which spots in the region may be merged.

    The region is already the user's statement of intent, so 'all' is the
    default. The other modes exist for leaving well-covered spots alone —
    at the price of an output whose count distributions differ by construction.
    """
    depth = np.asarray(counts).sum(axis=1)
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
```

In `merge_spots`, immediately after computing `max_spots`, add:

```python
    eligible = eligible_mask(counts, params)
```

and in the heap comprehension, skip edges touching an ineligible spot:

```python
    heap = [
        (d, min(a, b), max(a, b), a, b)
        for d, a, b in _neighbour_edges(coords, params.max_diameter_um)
        if eligible[a] and eligible[b]
    ]
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd backend && pixi run -e dev pytest tests/test_spot_merge.py -q`
Expected: PASS (22 tests)

- [ ] **Step 5: Commit**

```bash
git add backend/xcell/spot_merge.py backend/tests/test_spot_merge.py
git commit -m "feat: eligibility modes for spot merging"
```

---

### Task 4: DataAdaptor.merge_spots assembles the merged AnnData

**Files:**
- Modify: `backend/xcell/adaptor.py` (add method near the other spatial methods)
- Test: `backend/tests/test_spot_merge_adaptor.py`

**Interfaces:**
- Consumes: `spot_merge.MergeParams`, `merge_spots`, `eligible_mask`.
- Produces: `DataAdaptor.merge_spots(region_indices, params, gene_subset=None, layer='counts', purity_column=None, dry_run=True) -> tuple[dict, anndata.AnnData | None]`. The dict is the JSON-safe preview stats; the AnnData is `None` when `dry_run`.

- [ ] **Step 1: Write the failing test**

```python
"""Assembling the merged AnnData: what carries over and what does not."""

import anndata
import numpy as np
import pandas as pd
import pytest
from scipy.sparse import csr_matrix

from xcell.adaptor import DataAdaptor
from xcell.spot_merge import MergeParams


def _adata(n_side: int = 6, pitch: float = 15.0) -> anndata.AnnData:
    rng = np.random.default_rng(0)
    xs, ys = np.meshgrid(np.arange(n_side) * pitch, np.arange(n_side) * pitch)
    coords = np.column_stack([xs.ravel(), ys.ravel()]).astype(float)
    n = len(coords)
    counts = rng.poisson(5.0, size=(n, 8)).astype(np.float32) + 1.0
    ad = anndata.AnnData(X=csr_matrix(counts))
    ad.var_names = [f"g{i}" for i in range(8)]
    ad.obs_names = [f"s{i}" for i in range(n)]
    ad.layers["counts"] = csr_matrix(counts)
    ad.obs["cell_type"] = pd.Categorical(["hyper"] * n)
    ad.obs["total_counts"] = counts.sum(axis=1)          # QC — must be dropped
    ad.obs["ucell_score"] = rng.random(n)                # numeric — dropped
    ad.obsm["spatial"] = coords
    ad.obsm["X_umap"] = rng.normal(size=(n, 2))          # must be dropped
    ad.uns["xcell_spatial_scale"] = {"um_per_unit": 1.0}
    return ad


def _adaptor() -> DataAdaptor:
    return DataAdaptor("x.h5ad", adata=_adata())


def _region(a: DataAdaptor) -> list[int]:
    """The lower-left 3x3 block of the 6x6 grid."""
    coords = a.adata.obsm["spatial"]
    return np.where((coords[:, 0] < 40) & (coords[:, 1] < 40))[0].tolist()


def test_dry_run_returns_stats_and_builds_nothing():
    a = _adaptor()
    stats, merged = a.merge_spots(_region(a), MergeParams(), dry_run=True)
    assert merged is None
    assert stats["n_region_spots"] == 9
    assert stats["n_merged_spots"] < 9
    assert stats["pitch_um"] == pytest.approx(15.0)
    assert stats["max_spots"] >= 1


def test_the_source_dataset_is_untouched():
    a = _adaptor()
    before = a.adata.n_obs
    a.merge_spots(_region(a), MergeParams(), dry_run=False)
    assert a.adata.n_obs == before
    assert "merge_group" not in a.adata.obs


def test_counts_are_summed_exactly():
    a = _adaptor()
    region = _region(a)
    _, merged = a.merge_spots(region, MergeParams(), dry_run=False)
    # Total counts are conserved: merging moves them, it does not create them.
    source_total = np.asarray(a.adata.layers["counts"].sum())
    merged_total = np.asarray(merged.layers["counts"].sum())
    assert merged_total == pytest.approx(source_total)


def test_merged_coordinates_are_the_member_centroid():
    a = _adaptor()
    region = _region(a)
    stats, merged = a.merge_spots(region, MergeParams(), dry_run=False)
    groups = merged.obs["merge_group"].to_numpy()
    coords = merged.obsm["spatial"]
    src = a.adata.obsm["spatial"]
    labels = np.asarray(stats["_labels"])
    for gid in {g for g in groups if g}:
        members = [region[i] for i in range(len(region)) if str(labels[i]) == gid]
        row = int(np.where(groups == gid)[0][0])
        np.testing.assert_allclose(coords[row], src[members].mean(axis=0))


def test_untouched_spots_keep_their_rows():
    a = _adaptor()
    region = set(_region(a))
    _, merged = a.merge_spots(sorted(region), MergeParams(), dry_run=False)
    outside = [i for i in range(a.adata.n_obs) if i not in region]
    kept = merged[merged.obs["merged_n_spots"] == 1]
    assert kept.n_obs >= len(outside)


def test_numeric_and_qc_obs_are_dropped_everywhere():
    a = _adaptor()
    _, merged = a.merge_spots(_region(a), MergeParams(), dry_run=False)
    assert "ucell_score" not in merged.obs
    assert "total_counts" not in merged.obs


def test_categoricals_survive_by_majority_vote():
    a = _adaptor()
    _, merged = a.merge_spots(_region(a), MergeParams(), dry_run=False)
    assert "cell_type" in merged.obs
    assert set(merged.obs["cell_type"].dropna().unique()) == {"hyper"}


def test_only_the_spatial_obsm_survives():
    a = _adaptor()
    _, merged = a.merge_spots(_region(a), MergeParams(), dry_run=False)
    assert set(merged.obsm.keys()) == {"spatial"}


def test_provenance_records_how_it_was_made():
    a = _adaptor()
    _, merged = a.merge_spots(_region(a), MergeParams(max_diameter_um=40.0), dry_run=False)
    prov = merged.uns["xcell_spot_merge"]
    assert prov["params"]["max_diameter_um"] == 40.0
    assert prov["n_spots_before"] == 36
    assert prov["n_spots_after"] == merged.n_obs


def test_purity_appears_when_there_is_something_to_vote_on():
    a = _adaptor()
    _, merged = a.merge_spots(_region(a), MergeParams(), dry_run=False)
    assert "merge_purity" in merged.obs
    assert merged.obs["merge_purity"].max() <= 1.0


def test_a_region_of_one_spot_is_refused():
    a = _adaptor()
    with pytest.raises(ValueError, match="at least two"):
        a.merge_spots([0], MergeParams(), dry_run=True)


def test_missing_spatial_scale_is_refused():
    ad = _adata()
    del ad.uns["xcell_spatial_scale"]
    a = DataAdaptor("x.h5ad", adata=ad)
    with pytest.raises(ValueError, match="µm per coordinate unit|um_per_unit"):
        a.merge_spots(list(range(9)), MergeParams(), dry_run=True)


def test_non_integer_matrix_is_refused():
    ad = _adata()
    del ad.layers["counts"]
    ad.X = csr_matrix(np.asarray(ad.X.todense()) * 0.37)
    a = DataAdaptor("x.h5ad", adata=ad)
    with pytest.raises(ValueError, match="counts"):
        a.merge_spots(list(range(9)), MergeParams(), dry_run=True, layer=None)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd backend && pixi run -e dev pytest tests/test_spot_merge_adaptor.py -q`
Expected: FAIL — `AttributeError: 'DataAdaptor' object has no attribute 'merge_spots'`

- [ ] **Step 3: Write minimal implementation**

Add to `backend/xcell/adaptor.py`, near the other spatial methods:

```python
    def merge_spots(
        self,
        region_indices: list[int],
        params: 'MergeParams',
        gene_subset: str | list[str] | dict[str, Any] | None = None,
        layer: str | None = 'counts',
        purity_column: str | None = None,
        dry_run: bool = True,
    ) -> tuple[dict[str, Any], Any]:
        """Merge neighbouring spots in a region into single large cells.

        Read-only on self: the merged rows never existed, so every embedding
        and derived column computed on the originals would be stale. The result
        is a separate AnnData for the caller to register as its own dataset.

        Returns (stats, merged_adata). merged_adata is None when dry_run.
        """
        from xcell.spot_merge import MergeParams as _MP, eligible_mask, merge_spots

        spatial_key = self._get_spatial_key()
        if spatial_key is None:
            raise ValueError(
                "This dataset has no spatial coordinates, so there are no "
                "neighbouring spots to merge."
            )
        indices = np.asarray(sorted(set(int(i) for i in region_indices)), dtype=int)
        if len(indices) < 2:
            raise ValueError("Merging needs a region of at least two spots.")

        scale = self.spatial_scale()
        if scale.get('um_per_unit') is None:
            raise ValueError(
                "Set the µm per coordinate unit before merging — a cell "
                "diameter in microns means nothing without it."
            )
        um = float(scale['um_per_unit'])
        coords = np.asarray(self.adata.obsm[spatial_key], dtype=float)[indices, :2] * um

        source = self.adata.layers[layer] if layer and layer in self.adata.layers else self.adata.X
        region_counts = source[indices]
        if hasattr(region_counts, 'toarray'):
            region_counts = region_counts.toarray()
        region_counts = np.asarray(region_counts, dtype=float)
        if not np.all(np.equal(np.mod(region_counts, 1), 0)):
            raise ValueError(
                "Merging sums raw counts, and this matrix is not integer-like. "
                "Name a counts layer with the `layer` parameter."
            )

        gene_mask, subset_type, _ = self._resolve_gene_mask(gene_subset)
        veto_counts = region_counts[:, gene_mask]
        depth = veto_counts.sum(axis=1, keepdims=True)
        depth[depth == 0] = 1.0
        normalized = np.log1p(veto_counts / depth * 1e4)

        result = merge_spots(coords, region_counts, normalized, params)

        stats: dict[str, Any] = {
            'n_region_spots': int(len(indices)),
            'n_merged_spots': int(result.n_groups),
            'n_spots_before': int(self.n_cells),
            'n_spots_after': int(self.n_cells - len(indices) + result.n_groups),
            'pitch_um': float(result.pitch_um),
            'max_spots': int(result.max_spots),
            'size_histogram': result.size_histogram,
            'n_vetoed': int(result.n_vetoed),
            'correlation_quantiles': result.correlation_quantiles,
            'gene_subset': self._gene_subset_summary(subset_type, {'n_genes': int(gene_mask.sum())}),
            'median_counts_before': float(np.median(region_counts.sum(axis=1))),
            '_labels': [str(v) for v in result.labels],
        }
        if dry_run:
            return stats, None

        merged = self._build_merged_adata(
            indices, result.labels, spatial_key, purity_column, params, stats,
        )
        stats['median_counts_after'] = float(
            np.median(np.asarray(merged.layers['counts'].sum(axis=1)).ravel())
        )
        return stats, merged
```

And the assembly helper, immediately below it:

```python
    def _build_merged_adata(
        self,
        indices: np.ndarray,
        labels: np.ndarray,
        spatial_key: str,
        purity_column: str | None,
        params: 'MergeParams',
        stats: dict[str, Any],
    ):
        """Build the merged dataset: untouched rows, then one row per group."""
        import anndata
        import pandas as pd
        from scipy.sparse import csr_matrix, vstack

        counts_src = (
            self.adata.layers['counts'] if 'counts' in self.adata.layers else self.adata.X
        )
        in_region = np.zeros(self.n_cells, dtype=bool)
        in_region[indices] = True
        outside = np.where(~in_region)[0]

        cat_cols = [
            c for c in self.adata.obs.columns
            if isinstance(self.adata.obs[c].dtype, pd.CategoricalDtype)
            or self.adata.obs[c].dtype == object
        ]
        if purity_column is None:
            purity_column = cat_cols[0] if cat_cols else None

        rows, coords_out, obs_rows, names = [], [], [], []
        source_coords = np.asarray(self.adata.obsm[spatial_key], dtype=float)

        for i in outside:
            rows.append(counts_src[i])
            coords_out.append(source_coords[i])
            obs_rows.append({c: self.adata.obs[c].iloc[i] for c in cat_cols}
                            | {'merged_n_spots': 1, 'merge_group': '',
                               'merge_purity': np.nan})
            names.append(str(self.adata.obs_names[i]))

        for gid in range(int(labels.max()) + 1):
            members = indices[labels == gid]
            block = counts_src[members]
            rows.append(csr_matrix(block.sum(axis=0)))
            coords_out.append(source_coords[members].mean(axis=0))
            row: dict[str, Any] = {'merged_n_spots': int(len(members)),
                                   'merge_group': f'g{gid}'}
            purity = np.nan
            for c in cat_cols:
                values = self.adata.obs[c].iloc[members]
                counts_by_value = values.value_counts()
                row[c] = counts_by_value.index[0] if len(counts_by_value) else None
                if c == purity_column and len(values):
                    purity = float(counts_by_value.iloc[0] / len(values))
            row['merge_purity'] = purity
            obs_rows.append(row)
            names.append(f'merged_g{gid}')

        X = vstack([csr_matrix(r) for r in rows], format='csr')
        obs = pd.DataFrame(obs_rows, index=pd.Index(names, name=None))
        if purity_column is None:
            obs = obs.drop(columns=['merge_purity'])
        for c in cat_cols:
            obs[c] = pd.Categorical(obs[c])
        obs['merge_group'] = pd.Categorical(obs['merge_group'])

        merged = anndata.AnnData(X=X.astype(np.float32), obs=obs, var=self.adata.var.copy())
        merged.layers['counts'] = merged.X.copy()
        merged.obsm[spatial_key] = np.asarray(coords_out, dtype=float)
        merged.uns['xcell_spot_merge'] = {
            'params': {
                'max_diameter_um': float(params.max_diameter_um),
                'min_correlation': float(params.min_correlation),
                'eligibility': params.eligibility,
                'max_spots': int(stats['max_spots']),
            },
            'source_file': str(self.filepath.name),
            'n_spots_before': int(self.n_cells),
            'n_spots_after': int(merged.n_obs),
            'gene_subset': stats['gene_subset'],
        }
        if self.SPATIAL_SCALE_UNS in self.adata.uns:
            merged.uns[self.SPATIAL_SCALE_UNS] = dict(self.adata.uns[self.SPATIAL_SCALE_UNS])
        return merged
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd backend && pixi run -e dev pytest tests/test_spot_merge_adaptor.py -q`
Expected: PASS (13 tests)

- [ ] **Step 5: Commit**

```bash
git add backend/xcell/adaptor.py backend/tests/test_spot_merge_adaptor.py
git commit -m "feat: assemble the merged AnnData without touching the source"
```

---

### Task 5: POST /api/spots/merge

**Files:**
- Modify: `backend/xcell/api/routes.py` (add near the other spatial routes)
- Test: `backend/tests/test_spot_merge_routes.py`

**Interfaces:**
- Consumes: `DataAdaptor.merge_spots` from Task 4.
- Produces: `POST /api/spots/merge` taking `MergeSpotsRequest` — `region_indices: list[int]`, `max_diameter_um: float = 40.0`, `max_spots: int | None`, `min_correlation: float = 0.15`, `eligibility: str = 'all'`, `min_counts: float | None`, `quantile: float | None`, `gene_subset: str | list[str] | dict | None`, `layer: str | None = 'counts'`, `purity_column: str | None`, `dry_run: bool = True`, `slot: str | None`, `dataset: str | None`.

- [ ] **Step 1: Write the failing test**

```python
"""Route tests for POST /api/spots/merge."""

import anndata
import numpy as np
import pandas as pd
from fastapi.testclient import TestClient
from scipy.sparse import csr_matrix

from xcell.main import app
from xcell.api import routes
from xcell.adaptor import DataAdaptor


def _install(n_side: int = 6, pitch: float = 15.0, scale: bool = True):
    for key in list(routes._adaptors):
        routes.remove_adaptor(key)
    rng = np.random.default_rng(0)
    xs, ys = np.meshgrid(np.arange(n_side) * pitch, np.arange(n_side) * pitch)
    coords = np.column_stack([xs.ravel(), ys.ravel()]).astype(float)
    counts = rng.poisson(5.0, size=(len(coords), 8)).astype(np.float32) + 1.0
    ad = anndata.AnnData(X=csr_matrix(counts))
    ad.var_names = [f"g{i}" for i in range(8)]
    ad.layers["counts"] = csr_matrix(counts)
    ad.obs["cell_type"] = pd.Categorical(["hyper"] * len(coords))
    ad.obsm["spatial"] = coords
    if scale:
        ad.uns["xcell_spatial_scale"] = {"um_per_unit": 1.0}
    routes.set_adaptor(DataAdaptor("src.h5ad", adata=ad), slot="primary")
    return np.where((coords[:, 0] < 40) & (coords[:, 1] < 40))[0].tolist()


def test_dry_run_reports_without_creating_a_slot():
    region = _install()
    c = TestClient(app)
    r = c.post("/api/spots/merge", json={"region_indices": region, "dry_run": True})
    assert r.status_code == 200, r.text
    assert r.json()["n_merged_spots"] < len(region)
    assert list(c.get("/api/datasets").json()) == ["primary"]


def test_committing_creates_the_sibling_dataset():
    region = _install()
    c = TestClient(app)
    r = c.post("/api/spots/merge", json={
        "region_indices": region, "dry_run": False, "slot": "slot3",
    })
    assert r.status_code == 200, r.text
    assert r.json()["slot"] == "slot3"
    assert set(c.get("/api/datasets").json()) == {"primary", "slot3"}
    assert c.get("/api/schema?dataset=slot3").json()["n_cells"] < 36


def test_the_source_is_left_alone():
    region = _install()
    c = TestClient(app)
    c.post("/api/spots/merge", json={
        "region_indices": region, "dry_run": False, "slot": "slot3",
    })
    assert c.get("/api/schema").json()["n_cells"] == 36


def test_committing_without_a_slot_is_a_400():
    region = _install()
    r = TestClient(app).post("/api/spots/merge", json={
        "region_indices": region, "dry_run": False,
    })
    assert r.status_code == 400
    assert "slot" in r.json()["detail"].lower()


def test_no_region_is_a_400():
    _install()
    r = TestClient(app).post("/api/spots/merge", json={"region_indices": []})
    assert r.status_code == 400
    assert "at least two" in r.json()["detail"]


def test_missing_spatial_scale_is_a_400_that_says_so():
    region = _install(scale=False)
    r = TestClient(app).post("/api/spots/merge", json={"region_indices": region})
    assert r.status_code == 400
    assert "coordinate unit" in r.json()["detail"]
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd backend && pixi run -e dev pytest tests/test_spot_merge_routes.py -q`
Expected: FAIL — 404, the route does not exist

- [ ] **Step 3: Write minimal implementation**

Add to `backend/xcell/api/routes.py`:

```python
class MergeSpotsRequest(BaseModel):
    """Merge neighbouring spots in a region into single large cells."""
    region_indices: list[int]
    max_diameter_um: float = 40.0
    max_spots: int | None = None
    min_correlation: float = 0.15
    eligibility: str = 'all'
    min_counts: float | None = None
    quantile: float | None = None
    # A .var column, a {columns, operation} combination, or an explicit gene
    # list — what a Gene Panel gene set resolves to.
    gene_subset: str | list[str] | dict[str, Any] | None = None
    layer: str | None = 'counts'
    purity_column: str | None = None
    dry_run: bool = True
    slot: str | None = None


@router.post("/spots/merge")
def merge_spots(request: MergeSpotsRequest, dataset: str | None = Query(None)):
    """Merge neighbouring ST spots into spots that approximate single cells.

    Synchronous, following /combine: a lassoed region is hundreds to low
    thousands of spots, which agglomerates in well under a second.
    """
    from xcell.spot_merge import MergeParams

    adaptor = get_adaptor(dataset)
    if not request.dry_run and not request.slot:
        raise HTTPException(
            status_code=400,
            detail="Name the slot to load the merged dataset into.",
        )
    params = MergeParams(
        max_diameter_um=request.max_diameter_um,
        max_spots=request.max_spots,
        min_correlation=request.min_correlation,
        eligibility=request.eligibility,
        min_counts=request.min_counts,
        quantile=request.quantile,
    )
    try:
        stats, merged = adaptor.merge_spots(
            region_indices=request.region_indices,
            params=params,
            gene_subset=request.gene_subset,
            layer=request.layer,
            purity_column=request.purity_column,
            dry_run=request.dry_run,
        )
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    stats.pop('_labels', None)
    if merged is None:
        return stats

    virtual_path = Path(f"{adaptor.filepath.stem}_merged.h5ad")
    new_adaptor = DataAdaptor(virtual_path, adata=merged)
    new_adaptor.filepath = virtual_path
    set_adaptor(new_adaptor, slot=request.slot)
    return {"slot": request.slot, **stats, **new_adaptor.get_schema()}
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd backend && pixi run -e dev pytest tests/test_spot_merge_routes.py -q`
Expected: PASS (6 tests)

- [ ] **Step 5: Run the whole backend suite and commit**

```bash
cd backend && pixi run -e dev pytest -q
git add backend/xcell/api/routes.py backend/tests/test_spot_merge_routes.py
git commit -m "feat: POST /api/spots/merge creates the merged dataset"
```

---

### Task 6: Frontend preview formatting

**Files:**
- Create: `frontend/src/lib/spotMerge.ts`
- Test: `frontend/src/lib/spotMerge.test.ts`

**Interfaces:**
- Consumes: nothing.
- Produces: `formatSizeHistogram(h: Record<string, number>): string`, `vetoVerdict(nVetoed: number, q: {p10:number;p50:number;p90:number} | null, threshold: number): string`, `type MergePreview`.

- [ ] **Step 1: Write the failing test**

```typescript
import { describe, it, expect } from 'vitest'
import { formatSizeHistogram, vetoVerdict } from './spotMerge'

describe('formatSizeHistogram', () => {
  it('reads back as size:count in ascending size', () => {
    expect(formatSizeHistogram({ '3': 44, '1': 12, '2': 31 })).toBe('1:12  2:31  3:44')
  })

  it('is empty when there is nothing to show', () => {
    expect(formatSizeHistogram({})).toBe('—')
  })
})

describe('vetoVerdict', () => {
  it('says the veto is idle when it blocked nothing', () => {
    expect(vetoVerdict(0, { p10: 0.4, p50: 0.6, p90: 0.9 }, 0.15))
      .toMatch(/blocked nothing/i)
  })

  it('says it is doing real work when the distribution straddles the cut', () => {
    expect(vetoVerdict(37, { p10: 0.11, p50: 0.34, p90: 0.72 }, 0.15))
      .toMatch(/straddle/i)
  })

  it('warns when almost everything is above the cut but merges were still blocked', () => {
    expect(vetoVerdict(2, { p10: 0.5, p50: 0.7, p90: 0.95 }, 0.15))
      .toMatch(/rarely/i)
  })

  it('says nothing useful without a distribution', () => {
    expect(vetoVerdict(0, null, 0.15)).toBe('No candidate pairs were tested.')
  })
})
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd frontend && npx vitest run src/lib/spotMerge.test.ts`
Expected: FAIL — cannot resolve `./spotMerge`

- [ ] **Step 3: Write minimal implementation**

```typescript
/** Reading a spot-merge preview: the numbers that decide min_correlation. */

export interface MergePreview {
  n_region_spots: number
  n_merged_spots: number
  pitch_um: number
  max_spots: number
  size_histogram: Record<string, number>
  n_vetoed: number
  correlation_quantiles: { p10: number; p50: number; p90: number } | null
  median_counts_before: number
  median_counts_after?: number
}

/** "1:12  2:31  3:44" — how many groups of each size. */
export function formatSizeHistogram(h: Record<string, number>): string {
  const entries = Object.entries(h).sort((a, b) => Number(a[0]) - Number(b[0]))
  if (entries.length === 0) return '—'
  return entries.map(([size, n]) => `${size}:${n}`).join('  ')
}

/** Whether the veto threshold is earning its place, in one sentence.
 *
 *  The tuning question is not "how many merges were blocked" alone — it is
 *  whether the candidate correlations sit near the threshold at all. A veto
 *  far below every observed correlation is inert however many pairs existed.
 */
export function vetoVerdict(
  nVetoed: number,
  q: { p10: number; p50: number; p90: number } | null,
  threshold: number,
): string {
  if (!q) return 'No candidate pairs were tested.'
  if (nVetoed === 0) {
    return `The veto blocked nothing — every candidate pair scored above ${threshold}. `
      + 'Raise it if merges are crossing boundaries you can see.'
  }
  if (q.p10 < threshold && threshold < q.p90) {
    return `Candidate correlations straddle ${threshold} (p10 ${q.p10.toFixed(2)}, `
      + `p90 ${q.p90.toFixed(2)}), so the veto is doing real work — ${nVetoed} pairs blocked.`
  }
  return `${nVetoed} pairs blocked, but ${threshold} sits outside the bulk of the `
    + `distribution (p10 ${q.p10.toFixed(2)}, p90 ${q.p90.toFixed(2)}), so it fires rarely.`
}
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd frontend && npx vitest run src/lib/spotMerge.test.ts`
Expected: PASS (6 tests)

- [ ] **Step 5: Commit**

```bash
git add frontend/src/lib/spotMerge.ts frontend/src/lib/spotMerge.test.ts
git commit -m "feat: format the spot-merge preview"
```

---

### Task 7: Merge Spots modal

**Files:**
- Create: `frontend/src/components/MergeSpotsModal.tsx`
- Modify: `frontend/src/store.ts` (add `isMergeSpotsModalOpen` / `setMergeSpotsModalOpen` beside `isLocalizeModalOpen`)
- Modify: `frontend/src/App.tsx` (render `<MergeSpotsModal />` beside `<LocalizeModal />`)
- Modify: `frontend/src/components/ScanpyModal.tsx` (registry entry + open button)

**Interfaces:**
- Consumes: `MergePreview`, `formatSizeHistogram`, `vetoVerdict` (Task 6); `GeneSubsetPicker`, `useGeneSubset` from `./GeneSubsetPicker`; `POST /api/spots/merge` (Task 5).
- Produces: the modal. No other module imports it except `App.tsx`.

- [ ] **Step 1: Add the store flag**

In `frontend/src/store.ts`, beside the localize modal flag, add to the interface `isMergeSpotsModalOpen: boolean` and `setMergeSpotsModalOpen: (open: boolean) => void`; in the store body add `isMergeSpotsModalOpen: false,` and `setMergeSpotsModalOpen: (open) => set({ isMergeSpotsModalOpen: open }),`.

- [ ] **Step 2: Write the modal**

Create `frontend/src/components/MergeSpotsModal.tsx` following `LocalizeModal.tsx`'s structure: a `dark` token object, an overlay closed by click-outside and Escape, parameter inputs for `max_diameter_um` (default 40), `min_correlation` (default 0.15), `max_spots` (blank = derived), and an `eligibility` select with `min_counts` / `quantile` inputs appearing only for those modes. Include `<GeneSubsetPicker control={geneSubset} />` for the veto's genes, labelled "Veto genes" with the note "Correlation is computed on these genes' expression, not a set score."

The region comes from `useStore((s) => s.selectedCellIndices)`; when it holds fewer than two spots the run button is disabled with "Lasso at least two spots first."

Preview runs `dry_run: true` and renders `n_region_spots → n_merged_spots`, `pitch_um` and `max_spots`, `formatSizeHistogram(size_histogram)`, median counts before/after, and `vetoVerdict(...)`. Commit runs `dry_run: false` with `slot: nextSlotKey(Object.keys(datasets))`, then calls `loadDatasetIntoSlot(slot, schema)` and `setActiveSlot(slot)`.

- [ ] **Step 3: Register it in the Spatial group**

In `ScanpyModal.tsx`, inside `spatial_analysis.functions`, after the `localize` entry:

```typescript
      merge_spots: {
        label: 'Merge Spots',
        description: "Merge neighbouring spots inside a lassoed region into single spots that approximate one large cell — for tissue where the cell is much bigger than the spot, such as hypertrophic chondrocytes. Counts are summed; the result is a new dataset in its own tab, leaving this one untouched. Opens the Merge Spots tool.",
        prerequisites: ['has_spatial'],
        custom: true,
        params: [],
      },
```

and beside the `localize` branch in the render:

```typescript
          ) : selectedFunction === 'merge_spots' ? (
            <button
              style={styles.runButton}
              onClick={() => { setMergeSpotsModalOpen(true); setScanpyModalOpen(false) }}
            >
              Open Merge Spots tool…
            </button>
```

- [ ] **Step 4: Typecheck and run the frontend suite**

Run: `cd frontend && npx tsc --noEmit && npx vitest run`
Expected: tsc clean; all tests pass

- [ ] **Step 5: Commit**

```bash
git add frontend/src/components/MergeSpotsModal.tsx frontend/src/store.ts frontend/src/App.tsx frontend/src/components/ScanpyModal.tsx
git commit -m "feat: Merge Spots modal under Analyze -> Spatial"
```

---

### Task 8: Browser verification and CHANGELOG

**Files:**
- Modify: `CHANGELOG.md`

- [ ] **Step 1: Start the isolated stack**

Check `lsof -nP -iTCP:8000 -sTCP:LISTEN` first. Run the worktree's backend on `:8100` and Vite on `:5273` with `XCELL_BACKEND=http://127.0.0.1:8100`, per the isolated-stack recipe.

- [ ] **Step 2: Load a spatial dataset and set its scale**

`POST /api/load` with the bundled `toy_spatial.h5ad`, then `POST /api/spatial_scale` with a `um_per_unit` that makes the pitch realistic.

- [ ] **Step 3: Drive the loop**

Lasso a region, open Analyze → Spatial → Merge Spots, preview, change `min_correlation`, preview again, commit. Assert via CDP: a second tab exists, its `n_cells` is lower than the source's, the source's `n_cells` is unchanged, and no request failed. (Playwright MCP may be unavailable — see the `browser-verify-without-playwright` memory.)

- [ ] **Step 4: Write the CHANGELOG entry**

Under `## [Unreleased]` → `### Added`, in the house voice: what it does, why merging is a new dataset rather than an edit, why geometry drives it and similarity only vetoes, and the verified numbers from Step 3.

- [ ] **Step 5: Commit and merge**

```bash
cd backend && pixi run -e dev pytest -q
cd ../frontend && npx tsc --noEmit && npx vitest run
git add CHANGELOG.md && git commit -m "docs: changelog for ST spot merging"
```

Then merge to `main` with `--no-ff` and remove the worktree.

---

## Self-Review

**Spec coverage.** New dataset in a new slot — Task 5. Geometry-driven merge with a veto — Task 2. Sum raw / correlate normalized — Tasks 2 and 4. Eligibility parameter — Task 3. Pitch estimation and `max_spots` — Task 1. Dropped numeric/QC `.obs` and non-spatial `.obsm` — Task 4. Provenance — Task 4. `gene_subset` accepting gene sets — Tasks 4, 5 and 7. `dry_run` tuning loop — Tasks 5, 6 and 7. Error handling — Tasks 4 and 5. Out-of-scope items are absent, as intended.

**Type consistency.** `MergeParams` and `MergeResult` are defined in Task 2 and used unchanged in Tasks 3–5. `merge_spots` on the adaptor returns `(stats, merged)` in Task 4 and is destructured that way in Task 5. `_labels` is added to stats in Task 4 and popped in Task 5 before serialization — it exists only so the adaptor test can check centroids.

**Known gaps, deliberate.** Task 7's steps describe the modal rather than reproducing it in full: it is ~300 lines closely mirroring `LocalizeModal.tsx`, and the components it consumes (`GeneSubsetPicker`, `useGeneSubset`) already exist with settled signatures. Every value it reads and every endpoint it calls is specified above.
