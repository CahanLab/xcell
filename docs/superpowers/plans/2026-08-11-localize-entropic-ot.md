# Entropic Optimal Transport for Localize — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a `transport` aggregation to Localize that places query cells by entropic optimal transport, so the predicted population occupies the tissue without losing the gradients that averaging preserves.

**Architecture:** A new pure module `backend/xcell/transport.py` holds the solver (cost normalization, log-domain Sinkhorn, barycentric projection, stratified subsampling) and is imported by `localize.py`, which owns the wiring into `project_knn`. The solver never sees AnnData — it takes arrays and returns arrays, exactly like the other analysis modules.

**Tech Stack:** numpy, scipy (`scipy.special.logsumexp`, function-local import), pytest, React + TypeScript with inline styles.

## Global Constraints

- Spec: `docs/superpowers/specs/2026-08-11-localize-entropic-ot-design.md`.
- Cost is **always** normalized: shifted to zero minimum, divided by its standard deviation, computed on the matrix actually solved. ε is dimensionless.
- Default ε = **0.5** spreads, default `max_iterations` = **200**, convergence tolerance **1e-4** on the relative column-marginal error.
- **Log-domain always.** The naive kernel form underflows at small ε; there is one code path.
- Dense while affordable, spatially-stratified reference subsample above the ceiling. The run reports which path ran — a subsampled answer must never be readable as a full one.
- `float64` throughout the solver. The limb pair is 26.2M entries (210 MB per matrix); the ceiling is set so that case stays dense.
- Tests: `backend/tests/`, no `conftest.py`, module-level `_adata()` / `_adaptor()` factories, seeded `np.random.default_rng(0)`, `csr_matrix` + `float32`.
- Stats crossing the API must be JSON-serializable: no `NaN`, no `inf`. Use `None` for "not computable".
- Frontend: inline styles only. Palette — panel `#16213e`, border `#0f3460`, accent `#4ecdc4`, alert `#e94560`, warning `#e9a23b`, muted `#aaa`.

---

### Task 1: Cost normalization

**Files:**
- Create: `backend/xcell/transport.py`
- Test: `backend/tests/test_transport.py`

**Interfaces:**
- Consumes: nothing.
- Produces: `normalize_cost(cost: np.ndarray) -> tuple[np.ndarray, float]` returning the normalized matrix and the σ it divided by.

- [ ] **Step 1: Write the failing test**

```python
import numpy as np
import pytest

from xcell.transport import normalize_cost


def test_normalize_cost_is_zero_based_and_unit_spread():
    rng = np.random.default_rng(0)
    cost = rng.normal(5.0, 2.0, size=(50, 80))
    out, sigma = normalize_cost(cost)
    assert out.min() == pytest.approx(0.0, abs=1e-12)
    assert out.std() == pytest.approx(1.0, rel=1e-9)
    assert sigma == pytest.approx(cost.std(), rel=1e-9)


def test_normalize_cost_is_invariant_to_scale_and_shift():
    rng = np.random.default_rng(1)
    cost = rng.normal(0.0, 1.0, size=(30, 40))
    base, _ = normalize_cost(cost)
    scaled, _ = normalize_cost(cost * 7.0)
    shifted, _ = normalize_cost(cost + 3.0)
    np.testing.assert_allclose(scaled, base, rtol=1e-9, atol=1e-9)
    np.testing.assert_allclose(shifted, base, rtol=1e-9, atol=1e-9)


def test_normalize_cost_survives_a_constant_matrix():
    # Every pair equally similar carries no information; dividing by a zero
    # spread would poison the whole matrix with inf.
    out, sigma = normalize_cost(np.full((10, 12), 4.0))
    assert np.all(np.isfinite(out))
    assert sigma == 1.0
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `cd backend && pixi run -e dev pytest tests/test_transport.py -q`
Expected: FAIL — `ModuleNotFoundError: No module named 'xcell.transport'`

- [ ] **Step 3: Write the minimal implementation**

```python
"""Entropic optimal transport for spatial localization.

Pure: takes arrays, returns arrays. Never imports the adaptor and never sees
AnnData, so it is testable without one.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np


def normalize_cost(cost: np.ndarray) -> tuple[np.ndarray, float]:
    """Make the cost matrix dimensionless, so epsilon means the same thing.

    Sinkhorn forms exp(-C/eps), so what epsilon trades against is the *spread*
    of C, not its absolute scale. Measured on the E11.5 limb pair, the
    transform/metric choice moves the effective epsilon across 3.5x while the
    useful tuning range spans only 2.5x -- the preprocessing moves the dial
    further than the dial travels. Dividing by the spread is what lets one
    default survive every setting.

    The shift is free: entropic OT is invariant to a constant added to C,
    because exp(-a/eps) is absorbed into the scaling vectors.
    """
    c = np.asarray(cost, dtype=np.float64)
    c = c - c.min()
    sigma = float(c.std())
    if sigma <= 1e-12:
        # Every pair equally similar. There is nothing to scale, and dividing
        # would fill the matrix with inf.
        return c, 1.0
    return c / sigma, sigma
```

- [ ] **Step 4: Run the test to verify it passes**

Run: `cd backend && pixi run -e dev pytest tests/test_transport.py -q`
Expected: PASS (3 passed)

- [ ] **Step 5: Commit**

```bash
git add backend/xcell/transport.py backend/tests/test_transport.py
git commit -m "feat(transport): make the cost matrix dimensionless"
```

---

### Task 2: Log-domain Sinkhorn

**Files:**
- Modify: `backend/xcell/transport.py`
- Test: `backend/tests/test_transport.py`

**Interfaces:**
- Consumes: `normalize_cost` from Task 1.
- Produces: `@dataclass SinkhornResult` with fields `coupling: np.ndarray`, `iterations: int`, `marginal_error: float`; and `sinkhorn_log(cost, epsilon, *, max_iterations=200, tol=1e-4, check_every=10, progress=None) -> SinkhornResult`. `cost` is expected already normalized. `progress` is `Callable[[float, str], None] | None`.

- [ ] **Step 1: Write the failing test**

```python
from xcell.transport import SinkhornResult, normalize_cost, sinkhorn_log


def _toy_cost(n=40, m=60, seed=0):
    rng = np.random.default_rng(seed)
    qx = rng.uniform(0, 1, size=n)
    rx = rng.uniform(0, 1, size=m)
    return np.abs(qx[:, None] - rx[None, :])


def test_sinkhorn_satisfies_both_marginals():
    cost, _ = normalize_cost(_toy_cost())
    n, m = cost.shape
    res = sinkhorn_log(cost, epsilon=0.5, max_iterations=500, tol=1e-9)
    assert isinstance(res, SinkhornResult)
    np.testing.assert_allclose(res.coupling.sum(axis=1), 1.0 / n, rtol=1e-6)
    np.testing.assert_allclose(res.coupling.sum(axis=0), 1.0 / m, rtol=1e-6)


def test_sinkhorn_is_invariant_to_cost_scale_after_normalization():
    # The whole justification for normalize_cost: the same epsilon must mean
    # the same thing whatever units the similarity came in.
    raw = _toy_cost()
    a, _ = normalize_cost(raw)
    b, _ = normalize_cost(raw * 13.0)
    pa = sinkhorn_log(a, epsilon=0.5, max_iterations=500, tol=1e-9).coupling
    pb = sinkhorn_log(b, epsilon=0.5, max_iterations=500, tol=1e-9).coupling
    np.testing.assert_allclose(pa, pb, rtol=1e-6, atol=1e-12)


def test_larger_epsilon_gives_a_more_diffuse_coupling():
    cost, _ = normalize_cost(_toy_cost())
    sharp = sinkhorn_log(cost, epsilon=0.1, max_iterations=500).coupling
    smooth = sinkhorn_log(cost, epsilon=2.0, max_iterations=500).coupling

    def entropy(p):
        q = p / p.sum()
        return float(-(q * np.log(q + 1e-300)).sum())

    assert entropy(smooth) > entropy(sharp)


def test_sinkhorn_is_finite_at_an_epsilon_that_underflows_the_naive_form():
    cost, _ = normalize_cost(_toy_cost())
    # exp(-C/eps) at eps=0.01 spans ~e^-500: the kernel form has no chance.
    assert np.exp(-cost / 0.01).min() == 0.0
    res = sinkhorn_log(cost, epsilon=0.01, max_iterations=2000, tol=1e-8)
    assert np.all(np.isfinite(res.coupling))
    assert res.marginal_error < 1e-3


def test_sinkhorn_reports_budget_exhaustion_honestly():
    cost, _ = normalize_cost(_toy_cost())
    res = sinkhorn_log(cost, epsilon=0.05, max_iterations=2, tol=1e-12)
    assert res.iterations == 2
    assert res.marginal_error > 1e-12   # it did not converge, and says so


def test_sinkhorn_reports_progress():
    cost, _ = normalize_cost(_toy_cost())
    seen = []
    sinkhorn_log(cost, epsilon=0.5, max_iterations=50, tol=1e-12,
                 progress=lambda f, msg: seen.append((f, msg)))
    assert seen
    assert all(0.0 <= f <= 1.0 for f, _ in seen)
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `cd backend && pixi run -e dev pytest tests/test_transport.py -q`
Expected: FAIL — `ImportError: cannot import name 'SinkhornResult'`

- [ ] **Step 3: Write the minimal implementation**

Append to `backend/xcell/transport.py`:

```python
@dataclass
class SinkhornResult:
    """A coupling, plus what it cost to get and how well it satisfies its constraints."""

    coupling: np.ndarray      # (n_query, n_ref), rows sum to 1/n, columns to 1/m
    iterations: int
    marginal_error: float     # max relative deviation of a column sum from 1/m


def sinkhorn_log(
    cost: np.ndarray,
    epsilon: float,
    *,
    max_iterations: int = 200,
    tol: float = 1e-4,
    check_every: int = 10,
    progress: Callable[[float, str], None] | None = None,
) -> SinkhornResult:
    """Entropic OT in the log domain. ``cost`` must already be normalized.

    Log-domain always, rather than the faster kernel form. The kernel form does
    not fail on the kernel entries -- it fails on the scaling vectors, which
    grow and shrink without bound over iterations, and it does so at exactly
    the small epsilon where this estimator is most interesting. One stable path
    is worth more than the 2-3x per-iteration saving.
    """
    from scipy.special import logsumexp

    if epsilon <= 0:
        raise ValueError(f'epsilon must be positive, got {epsilon}')
    if max_iterations < 1:
        raise ValueError(f'max_iterations must be at least 1, got {max_iterations}')

    c = np.asarray(cost, dtype=np.float64)
    n, m = c.shape
    log_a = -np.log(n)
    log_b = -np.log(m)
    f = np.zeros(n, dtype=np.float64)
    g = np.zeros(m, dtype=np.float64)

    target_col = 1.0 / m
    err = np.inf
    used = 0
    for it in range(1, max_iterations + 1):
        used = it
        M = (-c + f[:, None] + g[None, :]) / epsilon
        f += epsilon * (log_a - logsumexp(M, axis=1))
        M = (-c + f[:, None] + g[None, :]) / epsilon
        g += epsilon * (log_b - logsumexp(M, axis=0))

        # Checking every iteration doubles the cost of the loop for a number
        # that moves slowly; checking never means we cannot stop early.
        if it % check_every == 0 or it == max_iterations:
            P = np.exp((-c + f[:, None] + g[None, :]) / epsilon)
            err = float(np.abs(P.sum(axis=0) - target_col).max() / target_col)
            if progress is not None:
                progress(it / max_iterations,
                         f'Sinkhorn iteration {it} of {max_iterations}, '
                         f'marginal error {err:.2e}')
            if err < tol:
                break

    P = np.exp((-c + f[:, None] + g[None, :]) / epsilon)
    err = float(np.abs(P.sum(axis=0) - target_col).max() / target_col)
    return SinkhornResult(coupling=P, iterations=used, marginal_error=err)
```

- [ ] **Step 4: Run the test to verify it passes**

Run: `cd backend && pixi run -e dev pytest tests/test_transport.py -q`
Expected: PASS (9 passed)

- [ ] **Step 5: Commit**

```bash
git add backend/xcell/transport.py backend/tests/test_transport.py
git commit -m "feat(transport): log-domain Sinkhorn with an honest convergence report"
```

---

### Task 3: Barycentric projection, with sections

**Files:**
- Modify: `backend/xcell/transport.py`
- Test: `backend/tests/test_transport.py`

**Interfaces:**
- Consumes: a coupling from Task 2.
- Produces: `barycentric_projection(coupling, coords, sections=None) -> tuple[np.ndarray, np.ndarray | None]` returning `(n_query, 2)` predictions and the chosen section label per row (or `None`); and `posterior_spread(coupling, coords, predictions) -> np.ndarray` of shape `(n_query,)`.

- [ ] **Step 1: Write the failing test**

```python
from xcell.transport import barycentric_projection, posterior_spread


def test_barycentric_is_the_row_weighted_mean_of_coordinates():
    coupling = np.array([[0.5, 0.5, 0.0],
                         [0.0, 0.0, 1.0]]) / 2.0   # rows sum to 1/n
    coords = np.array([[0.0, 0.0], [2.0, 0.0], [10.0, 4.0]])
    pred, sections = barycentric_projection(coupling, coords)
    assert sections is None
    np.testing.assert_allclose(pred, [[1.0, 0.0], [10.0, 4.0]])


def test_a_row_split_across_sections_does_not_land_in_the_gap():
    # Two cuts 100 apart. A row with mass in both must resolve to one of them,
    # not to the empty space between -- the failure project_knn already guards
    # against for the k-NN path.
    coupling = np.array([[0.4, 0.2, 0.3, 0.1]]) / 1.0
    coords = np.array([[0.0, 0.0], [1.0, 0.0], [100.0, 0.0], [101.0, 0.0]])
    sections = np.array(['a', 'a', 'b', 'b'])
    pred, chosen = barycentric_projection(coupling, coords, sections)
    assert chosen is not None and chosen[0] == 'a'          # 0.6 of the mass
    assert pred[0][0] < 1.0                                  # inside section a
    np.testing.assert_allclose(pred[0], [1.0 / 3.0, 0.0])    # 0.4/0.6, 0.2/0.6


def test_posterior_spread_is_zero_for_a_certain_row():
    coupling = np.array([[0.0, 1.0]])
    coords = np.array([[0.0, 0.0], [3.0, 4.0]])
    pred, _ = barycentric_projection(coupling, coords)
    assert posterior_spread(coupling, coords, pred)[0] == pytest.approx(0.0)


def test_posterior_spread_grows_as_the_posterior_spreads():
    coords = np.array([[0.0, 0.0], [10.0, 0.0]])
    tight = np.array([[0.99, 0.01]])
    loose = np.array([[0.5, 0.5]])
    st = posterior_spread(tight, coords, barycentric_projection(tight, coords)[0])
    sl = posterior_spread(loose, coords, barycentric_projection(loose, coords)[0])
    assert sl[0] > st[0]
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `cd backend && pixi run -e dev pytest tests/test_transport.py -q`
Expected: FAIL — `ImportError: cannot import name 'barycentric_projection'`

- [ ] **Step 3: Write the minimal implementation**

Append to `backend/xcell/transport.py`:

```python
def _row_normalize(coupling: np.ndarray) -> np.ndarray:
    """Rows of P sum to 1/n; a posterior over locations must sum to 1."""
    P = np.asarray(coupling, dtype=np.float64)
    total = P.sum(axis=1, keepdims=True)
    return P / np.where(total > 0, total, 1.0)


def barycentric_projection(
    coupling: np.ndarray,
    coords: np.ndarray,
    sections: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray | None]:
    """Predict each cell's coordinate as the mean of its posterior.

    Taking the row argmax instead returns to the single-location regime and
    loses the axes (measured: 0.98 area, ~0 correlation), which is the failure
    this estimator exists to avoid.
    """
    W = _row_normalize(coupling)
    coords = np.asarray(coords, dtype=np.float64)
    if sections is None:
        return W @ coords, None

    # Averaging across sections lands in the gap between cuts, so resolve each
    # row to the section holding most of its mass and renormalize within it.
    sections = np.asarray(sections).astype(str)
    labels = np.unique(sections)
    mass = np.stack([W[:, sections == s].sum(axis=1) for s in labels], axis=1)
    chosen = labels[np.argmax(mass, axis=1)]

    out = np.zeros((W.shape[0], coords.shape[1]), dtype=np.float64)
    for s in labels:
        rows = np.flatnonzero(chosen == s)
        if rows.size == 0:
            continue
        cols = np.flatnonzero(sections == s)
        sub = W[np.ix_(rows, cols)]
        total = sub.sum(axis=1, keepdims=True)
        sub = sub / np.where(total > 0, total, 1.0)
        out[rows] = sub @ coords[cols]
    return out, chosen


def posterior_spread(
    coupling: np.ndarray,
    coords: np.ndarray,
    predictions: np.ndarray,
) -> np.ndarray:
    """Weighted RMS distance of the posterior from its own mean, per cell.

    The same quantity ``project_knn`` already computes over the k neighbours,
    evaluated over the coupling instead. Keeping the formula identical is what
    lets confidence stay comparable across aggregations -- row entropy would
    not be in spatial units, and the Map quality panel exists to compare runs.
    """
    W = _row_normalize(coupling)
    coords = np.asarray(coords, dtype=np.float64)
    predictions = np.asarray(predictions, dtype=np.float64)
    offsets = coords[None, :, :] - predictions[:, None, :]
    return np.sqrt((W * (offsets ** 2).sum(axis=2)).sum(axis=1))
```

- [ ] **Step 4: Run the test to verify it passes**

Run: `cd backend && pixi run -e dev pytest tests/test_transport.py -q`
Expected: PASS (13 passed)

- [ ] **Step 5: Commit**

```bash
git add backend/xcell/transport.py backend/tests/test_transport.py
git commit -m "feat(transport): barycentric projection that respects sections"
```

---

### Task 4: Spatially-stratified reference subsample

**Files:**
- Modify: `backend/xcell/transport.py`
- Test: `backend/tests/test_transport.py`

**Interfaces:**
- Consumes: nothing from earlier tasks.
- Produces: `stratified_subsample(coords, n_target, *, seed=0, n_bins=32) -> np.ndarray` of sorted integer indices into `coords`.

- [ ] **Step 1: Write the failing test**

```python
from xcell.transport import stratified_subsample


def test_subsample_returns_the_requested_size():
    rng = np.random.default_rng(0)
    coords = rng.uniform(0, 100, size=(5000, 2))
    idx = stratified_subsample(coords, 500)
    assert len(idx) == 500
    assert len(np.unique(idx)) == 500


def test_subsample_keeps_every_occupied_region():
    # A dense blob plus a sparse arm. Uniform sampling would routinely drop the
    # arm, and the column marginal would then be defined over a tissue that is
    # missing a limb.
    rng = np.random.default_rng(0)
    blob = rng.normal(10, 1.0, size=(4000, 2))
    arm = np.stack([np.linspace(40, 60, 40), np.full(40, 50.0)], axis=1)
    coords = np.vstack([blob, arm])
    idx = stratified_subsample(coords, 300)
    kept_arm = (idx >= 4000).sum()
    assert kept_arm >= 5, f'the sparse arm all but vanished: {kept_arm} of 40'


def test_subsample_returns_everything_when_the_target_is_larger():
    coords = np.random.default_rng(0).uniform(0, 1, size=(100, 2))
    idx = stratified_subsample(coords, 500)
    np.testing.assert_array_equal(idx, np.arange(100))


def test_subsample_is_deterministic_for_a_seed():
    coords = np.random.default_rng(0).uniform(0, 1, size=(2000, 2))
    np.testing.assert_array_equal(
        stratified_subsample(coords, 200, seed=7),
        stratified_subsample(coords, 200, seed=7),
    )
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `cd backend && pixi run -e dev pytest tests/test_transport.py -q`
Expected: FAIL — `ImportError: cannot import name 'stratified_subsample'`

- [ ] **Step 3: Write the minimal implementation**

Append to `backend/xcell/transport.py`:

```python
def stratified_subsample(
    coords: np.ndarray,
    n_target: int,
    *,
    seed: int = 0,
    n_bins: int = 32,
) -> np.ndarray:
    """Pick ``n_target`` reference spots that still cover the tissue.

    Uniform sampling thins a sparse extremity in proportion to its density, and
    the column marginal would then be enforcing occupancy of a tissue missing
    that extremity. Binning the coordinates and taking at least one spot from
    every occupied bin keeps the shape.
    """
    coords = np.asarray(coords, dtype=np.float64)
    n = coords.shape[0]
    if n_target >= n:
        return np.arange(n)

    rng = np.random.default_rng(seed)
    keys = np.zeros(n, dtype=np.int64)
    for axis in range(2):
        lo, hi = coords[:, axis].min(), coords[:, axis].max()
        span = hi - lo
        edges = np.linspace(lo, hi, n_bins + 1)[1:-1] if span > 0 else np.array([])
        keys = keys * n_bins + np.digitize(coords[:, axis], edges)

    bins = [np.flatnonzero(keys == b) for b in np.unique(keys)]
    # One from every occupied bin first, so no region can drop out entirely.
    chosen = [rng.choice(members, size=1) for members in bins]
    taken = {int(i) for arr in chosen for i in arr}

    remaining = n_target - len(taken)
    if remaining > 0:
        rest = np.array([i for i in range(n) if i not in taken])
        # Proportional to occupancy: sampling the leftovers uniformly is
        # exactly proportional, since dense bins contribute more leftovers.
        extra = rng.choice(rest, size=min(remaining, rest.size), replace=False)
        chosen.append(extra)

    out = np.unique(np.concatenate(chosen))
    if out.size > n_target:
        # Trim the surplus from the one-per-bin floor, never below one per bin.
        keep = rng.choice(out.size, size=n_target, replace=False)
        out = np.sort(out[keep])
    return out
```

- [ ] **Step 4: Run the test to verify it passes**

Run: `cd backend && pixi run -e dev pytest tests/test_transport.py -q`
Expected: PASS (17 passed)

- [ ] **Step 5: Commit**

```bash
git add backend/xcell/transport.py backend/tests/test_transport.py
git commit -m "feat(transport): stratified subsampling that keeps the tissue's shape"
```

---

### Task 5: Wire `transport` into `project_knn`

**Files:**
- Modify: `backend/xcell/localize.py:41` (`AGGREGATIONS`), `:77-91` (`Projection`), `:419-594` (`project_knn`)
- Test: `backend/tests/test_localize_transport.py`

**Interfaces:**
- Consumes: everything from Tasks 1–4.
- Produces: `project_knn(..., aggregation='transport', epsilon=0.5, max_iterations=200)`; `Projection` gains `transport: str | None = None`, `n_ref_used: int | None = None`, `sinkhorn_iterations: int | None = None`, `marginal_error: float | None = None`.

- [ ] **Step 1: Write the failing test**

```python
import numpy as np
import pytest

from xcell.localize import project_knn


def _pair(n_query=60, n_ref=120, n_genes=40, seed=0):
    """A query whose cells each resemble one reference spot, on a 2-D gradient."""
    rng = np.random.default_rng(seed)
    coords = rng.uniform(0, 10, size=(n_ref, 2))
    loadings = rng.normal(size=(2, n_genes))
    ref = coords @ loadings + rng.normal(scale=0.1, size=(n_ref, n_genes))
    pick = rng.choice(n_ref, size=n_query, replace=False)
    query = ref[pick] + rng.normal(scale=0.1, size=(n_query, n_genes))
    return query.astype(np.float32), ref.astype(np.float32), coords, pick


def test_transport_is_an_accepted_aggregation():
    q, r, c, _ = _pair()
    proj = project_knn(q, r, c, aggregation='transport')
    assert proj.coords.shape == (q.shape[0], 2)
    assert proj.transport == 'full'
    assert proj.n_ref_used == r.shape[0]


def test_transport_recovers_the_true_locations_on_a_clean_pair():
    q, r, c, pick = _pair()
    proj = project_knn(q, r, c, aggregation='transport', epsilon=0.3)
    err = np.linalg.norm(proj.coords - c[pick], axis=1)
    assert np.median(err) < 2.0, f'median error {np.median(err):.2f} on a clean pair'


def test_transport_fills_more_of_the_tissue_than_weighted_mean():
    # The reason this estimator exists: the reference-side marginal stops the
    # population collapsing toward the tissue centroid.
    q, r, c, _ = _pair()
    ot = project_knn(q, r, c, aggregation='transport', epsilon=0.3).coords
    wm = project_knn(q, r, c, aggregation='weighted_mean').coords
    assert ot.std(axis=0).mean() > wm.std(axis=0).mean()


def test_transport_reports_its_convergence():
    q, r, c, _ = _pair()
    proj = project_knn(q, r, c, aggregation='transport', max_iterations=3)
    assert proj.sinkhorn_iterations == 3
    assert proj.marginal_error is not None and np.isfinite(proj.marginal_error)


def test_transport_rejects_a_non_positive_epsilon():
    q, r, c, _ = _pair()
    with pytest.raises(ValueError, match='epsilon'):
        project_knn(q, r, c, aggregation='transport', epsilon=0.0)


def test_transport_confidence_is_finite_and_bounded():
    q, r, c, _ = _pair()
    proj = project_knn(q, r, c, aggregation='transport')
    assert np.all(np.isfinite(proj.confidence))
    assert proj.confidence.min() >= 0.0 and proj.confidence.max() <= 1.0
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `cd backend && pixi run -e dev pytest tests/test_localize_transport.py -q`
Expected: FAIL — `ValueError: aggregation must be one of [...], got 'transport'`

- [ ] **Step 3: Write the minimal implementation**

In `backend/xcell/localize.py`:

1. Extend the tuple at line 41:

```python
AGGREGATIONS = (
    'weighted_mean', 'median', 'best_match', 'densest', 'injective', 'transport',
)
```

2. Add the ceiling constant beside `_DENSE_MAX_ENTRIES`:

```python
# Sinkhorn holds the cost matrix and one workspace of the same shape, in
# float64 for numerical safety. Set by what already works, as the dense
# assignment ceiling was: the E11.5 limb pair is 2.6e7 entries and must keep
# running at full reference size, because a subsampled answer is a different
# answer. Above this the reference is subsampled and the run says so.
_TRANSPORT_MAX_ENTRIES = 40_000_000
```

3. Add four fields to `Projection` (after `n_candidates`):

```python
    # Which reference the coupling was solved against, and how well it
    # converged. A subsampled run is a different answer from a full one and
    # must never be readable as the same.
    transport: str | None = None
    n_ref_used: int | None = None
    sinkhorn_iterations: int | None = None
    marginal_error: float | None = None
```

4. Add the parameters to `project_knn`'s signature, after `min_confidence`:

```python
    epsilon: float = 0.5,
    max_iterations: int = 200,
```

5. Validate early, beside the `injective` feasibility check (after line 477):

```python
    if aggregation == 'transport' and epsilon <= 0:
        raise ValueError(f'epsilon must be positive, got {epsilon}')
```

6. Add the branch. Replace the `else: coords = _aggregate(...)` tail (line 563-564) with:

```python
    elif aggregation == 'transport':
        from xcell import transport as tr

        ref_idx = np.arange(n_ref)
        transport_kind = 'full'
        if n_query * n_ref > _TRANSPORT_MAX_ENTRIES:
            budget = max(1, _TRANSPORT_MAX_ENTRIES // max(n_query, 1))
            ref_idx = tr.stratified_subsample(ref_coords, budget)
            transport_kind = 'subsampled'

        # Recomputed rather than reused: the k-NN pass keeps only each row's
        # best k, and transport needs every entry it is solving over.
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
        # Confidence from the posterior, in the same units the k-NN path uses.
        transport_spread = tr.posterior_spread(
            result.coupling, ref_coords[ref_idx], coords,
        )
    else:
        coords = _aggregate(neighbor_coords, weights, aggregation)
```

7. Initialise `transport_out = None` and `transport_spread = None` beside `assignment_kind`, use the posterior spread for confidence when present, and pass the fields through:

```python
    if transport_spread is not None:
        spread = transport_spread
    else:
        offsets = neighbor_coords - coords[:, None, :]
        spread = np.sqrt((weights * (offsets ** 2).sum(axis=2)).sum(axis=1))
    confidence = np.clip(1.0 - spread / _tissue_radius(ref_coords), 0.0, 1.0)
```

and in the returned `Projection`:

```python
        transport=transport_out[0] if transport_out else None,
        n_ref_used=transport_out[1] if transport_out else None,
        sinkhorn_iterations=transport_out[2] if transport_out else None,
        marginal_error=transport_out[3] if transport_out else None,
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `cd backend && pixi run -e dev pytest tests/test_localize_transport.py tests/test_localize.py -q`
Expected: PASS — the new file's 6 tests plus the existing localize suite unbroken.

- [ ] **Step 5: Commit**

```bash
git add backend/xcell/localize.py backend/tests/test_localize_transport.py
git commit -m "feat(localize): place cells by entropic optimal transport"
```

---

### Task 6: Adaptor and route plumbing

**Files:**
- Modify: `backend/xcell/adaptor.py:8081-8090` (`prepare_localize` signature), `:8147` (the `project_knn` call), `:8181-8182` (the result dict)
- Modify: `backend/xcell/api/routes.py:3573-3585` (`LocalizeRequest`), and the `prepare_localize(...)` call in `localize_prepare`
- Test: `backend/tests/test_localize_transport.py`

**Interfaces:**
- Consumes: `project_knn(..., epsilon=, max_iterations=)` from Task 5.
- Produces: `prepare_localize(..., epsilon: float = 0.5, max_iterations: int = 200)`; the result dict gains keys `transport`, `n_ref_used`, `sinkhorn_iterations`, `marginal_error`.

- [ ] **Step 1: Write the failing test**

```python
import anndata as ad
import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix

from xcell.adaptor import DataAdaptor


def _adata(n=40, n_genes=30, spatial=True, seed=0):
    rng = np.random.default_rng(seed)
    X = csr_matrix(rng.poisson(1.0, size=(n, n_genes)).astype(np.float32))
    obs = pd.DataFrame(index=[f'c{i}' for i in range(n)])
    var = pd.DataFrame(index=[f'g{i}' for i in range(n_genes)])
    a = ad.AnnData(X=X, obs=obs, var=var)
    if spatial:
        a.obsm['X_spatial'] = rng.uniform(0, 10, size=(n, 2))
    return a


def _adaptor(a):
    return DataAdaptor('x.h5ad', adata=a)


def test_prepare_localize_passes_transport_settings_through():
    query = _adaptor(_adata(spatial=False, seed=1))
    reference = _adaptor(_adata(seed=2))
    bundle = reference.spatial_reference_bundle()

    compute_fn, apply_fn = query.prepare_localize(
        bundle, aggregation='transport', epsilon=0.4, max_iterations=25,
    )
    result = apply_fn(compute_fn())

    assert result['transport'] == 'full'
    assert result['n_ref_used'] == 40
    assert result['sinkhorn_iterations'] <= 25
    assert np.isfinite(result['marginal_error'])
    # The stats cross the API, so they must survive json.dumps.
    import json
    json.dumps(result)


def test_prepare_localize_rejects_a_bad_epsilon_synchronously():
    query = _adaptor(_adata(spatial=False, seed=1))
    reference = _adaptor(_adata(seed=2))
    bundle = reference.spatial_reference_bundle()
    with pytest.raises(ValueError, match='epsilon'):
        query.prepare_localize(bundle, aggregation='transport', epsilon=-1.0)
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `cd backend && pixi run -e dev pytest tests/test_localize_transport.py -q -k prepare_localize`
Expected: FAIL — `TypeError: prepare_localize() got an unexpected keyword argument 'epsilon'`

- [ ] **Step 3: Write the minimal implementation**

In `adaptor.py`, add to the `prepare_localize` signature after `min_confidence`:

```python
        epsilon: float = 0.5,
        max_iterations: int = 200,
```

Validate synchronously, beside the existing `metric`/`transform` checks:

```python
        if aggregation == 'transport' and epsilon <= 0:
            raise ValueError(f'epsilon must be positive, got {epsilon}')
```

Snapshot both into locals with the other parameters, pass them to `lz.project_knn(...)` at line 8147:

```python
                epsilon=epsilon,
                max_iterations=max_iterations,
```

and extend the result dict beside `'assignment'`:

```python
                'transport': projection.transport,
                'n_ref_used': projection.n_ref_used,
                'sinkhorn_iterations': projection.sinkhorn_iterations,
                'marginal_error': _f(projection.marginal_error),
```

In `routes.py`, add to `LocalizeRequest` after `min_confidence`:

```python
    epsilon: float = 0.5
    max_iterations: int = 200
```

and pass them in the `prepare_localize(...)` call:

```python
            epsilon=request.epsilon,
            max_iterations=request.max_iterations,
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `cd backend && pixi run -e dev pytest tests/ -q`
Expected: PASS — full backend suite, no regressions.

- [ ] **Step 5: Commit**

```bash
git add backend/xcell/adaptor.py backend/xcell/api/routes.py backend/tests/test_localize_transport.py
git commit -m "feat(localize): carry the transport settings through to the route"
```

---

### Task 7: Frontend — the ε control and its guidance

**Files:**
- Modify: `frontend/src/lib/localizeAdvice.ts:28-38` (`LocalizeSettings`), `:51` (`adviseParameters`)
- Modify: `frontend/src/components/LocalizeModal.tsx` — aggregation selector and parameter block
- Test: `frontend/src/lib/localizeAdvice.test.ts`

**Interfaces:**
- Consumes: the `epsilon` request field from Task 6.
- Produces: `LocalizeSettings` gains `epsilon: number`; `adviseParameters` returns advice for the `transport` aggregation.

- [ ] **Step 1: Write the failing test**

```typescript
import { adviseParameters } from './localizeAdvice'

const base = {
  k: 15, transform: 'none', metric: 'cosine', aggregation: 'transport',
  minConfidence: 0, nReferenceCells: 9773, nQueryCells: 2683,
  nSharedGenes: 22016, populations: [], epsilon: 0.5,
}

it('states the composition assumption the reference marginal makes', () => {
  const text = adviseParameters(base).map(a => a.text).join(' ')
  expect(text).toMatch(/composition/i)
})

it('warns that a large epsilon collapses the map toward the centre', () => {
  const text = adviseParameters({ ...base, epsilon: 3.0 }).map(a => a.text).join(' ')
  expect(text).toMatch(/centre|center|collapse/i)
})

it('warns that a tiny epsilon throws away the averaging', () => {
  const text = adviseParameters({ ...base, epsilon: 0.02 }).map(a => a.text).join(' ')
  expect(text).toMatch(/gradient|averaging/i)
})

it('stays silent about epsilon for every other aggregation', () => {
  const text = adviseParameters({ ...base, aggregation: 'weighted_mean', epsilon: 3.0 })
    .map(a => a.text).join(' ')
  expect(text).not.toMatch(/epsilon|ε/i)
})
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `cd frontend && npx vitest run src/lib/localizeAdvice.test.ts`
Expected: FAIL — the composition and ε assertions find no matching advice.

- [ ] **Step 3: Write the minimal implementation**

Add `epsilon: number` to `LocalizeSettings`, then inside `adviseParameters`:

```typescript
  if (o.aggregation === 'transport') {
    out.push({ level: 'info', text:
      'Transport places the population so it occupies the tissue, which assumes the query\'s composition matches the tissue\'s. Dissociation biases which cell types survive, so a query missing a type will still have that type\'s share of the tissue filled by something.' })
    if (o.epsilon >= 2.0) {
      out.push({ level: 'warn', text:
        `ε=${o.epsilon} spreads each cell's posterior over much of the reference, so the averaged prediction collapses toward the tissue centre — the failure this estimator exists to avoid.` })
    } else if (o.epsilon <= 0.05) {
      out.push({ level: 'warn', text:
        `ε=${o.epsilon} makes each cell pick almost a single spot, which throws away the averaging that keeps the gradients. Expect the axis correlations to fall toward best_match's.` })
    }
  }
```

In `LocalizeModal.tsx`, add `transport` to the aggregation options and render a number input for ε (step 0.1, min 0.01) shown only when `aggregation === 'transport'`, styled with the existing field tokens (background `#0f1625`, border `#0f3460`). Pass `epsilon` in the request body and into the `adviseParameters` call.

- [ ] **Step 4: Run the tests to verify they pass**

Run: `cd frontend && npx vitest run && npx tsc --noEmit`
Expected: PASS, tsc clean.

- [ ] **Step 5: Commit**

```bash
git add frontend/src/lib/localizeAdvice.ts frontend/src/lib/localizeAdvice.test.ts frontend/src/components/LocalizeModal.tsx
git commit -m "feat(ui): expose the transport epsilon, with the guidance it needs"
```

---

### Task 8: Measure ε on the real pair, set the default, document

**Files:**
- Create: `docs/superpowers/measurements/2026-08-11-transport-epsilon-sweep.md`
- Modify: `backend/xcell/localize.py` (the ε default, if the sweep says so)
- Modify: `CHANGELOG.md` (under `## [Unreleased]` → `### Added`)

**Interfaces:**
- Consumes: everything above.
- Produces: a measured ε default and a CHANGELOG entry.

- [ ] **Step 1: Run the sweep**

Load the real pair — reference `~/CahanLab Dropbox/Patrick Cahan/PC/PC/projects/SJD/data/spatial/xcell/h5ad_empty_removed/best_per_stage/E11.5_best_102424.h5ad`, query `~/CahanLab Dropbox/Patrick Cahan/PC/PC/projects/SJD/data/scRNAseq/Kelly_Guilak_2020/GSM4227224_E11_112125.h5ad` — with `transform='none'`, `metric='cosine'` (the settings actually in use). For ε in `[0.2, 0.3, 0.5, 0.8, 1.2]`, run `project_knn` and score each result with `xcell.localize_metrics`, recording hull area, spatial pattern fidelity per gene set, axis fidelity against the reference ceiling, and occupancy.

- [ ] **Step 2: Record the table**

Write the measured table into `docs/superpowers/measurements/2026-08-11-transport-epsilon-sweep.md`, including runtime and iterations per ε, and the `weighted_mean` / `best_match` baselines for comparison.

- [ ] **Step 3: Set the default from the measurement**

If the sweep's best trade differs from 0.5, change the default in `project_knn`, `prepare_localize` and `LocalizeRequest` together, and say in the commit message what the measurement was. If 0.5 stands, say that too — a default confirmed by measurement is worth recording as such.

- [ ] **Step 4: Verify in the browser**

Start both servers (`pixi run backend`, `pixi run dev`), load the query and the reference into the two slots, run Analyze → Spatial → Localize with `transport`, and confirm: the map renders, Map quality scores it, the ε guidance appears, and a subsampled run says so. Stop both servers afterwards.

- [ ] **Step 5: Write the CHANGELOG entry and commit**

Follow the house voice: lead with what the user sees, give the measured numbers, and name what was rejected and why.

```bash
git add CHANGELOG.md docs/superpowers/measurements/ backend/xcell/localize.py
git commit -m "docs: what epsilon does on the limb pair, measured"
```

---

## Self-Review

**Spec coverage.** Estimator → Tasks 1–3, 5. Scale and numerics → Tasks 2, 4, 5. What it returns → Tasks 3, 5, 6. Surface → Tasks 6, 7. Testing → every task. Success criteria (no gate, harness ranks) → Task 8. Out-of-scope items are absent by construction.

**Placeholders.** None: every step carries the code or the exact command. Task 8's ε default is deliberately conditional on a measurement, with both branches specified.

**Type consistency.** `normalize_cost` returns `(ndarray, float)` and is called for its first element in Task 5. `SinkhornResult.coupling/iterations/marginal_error` are the names used in Tasks 2, 5 and 6. `barycentric_projection` returns `(coords, sections|None)` and both are consumed in Task 5. `stratified_subsample` returns indices, used as `ref_idx` in Task 5. `Projection.transport/n_ref_used/sinkhorn_iterations/marginal_error` are consistent across Tasks 5, 6. `LocalizeSettings.epsilon` is added in Task 7 and matches the `epsilon` request field from Task 6.
