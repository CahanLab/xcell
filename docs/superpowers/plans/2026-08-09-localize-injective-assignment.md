# Localize: injective assignment, and a warning the default has earned — Implementation Plan (Phase 2)

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add an aggregation that gives every query cell a *different* reference
spot, and make the modal say — before a run, from the reference alone — when the
default `weighted_mean` is about to collapse a population into a place it does
not exist.

**Architecture:** Two independent pieces sharing a branch. (1) `injective` joins
`AGGREGATIONS`: `project_knn` materializes the full query × reference similarity
matrix and hands it to `scipy.optimize.linear_sum_assignment`; it is off by
default and refused when the reference has fewer spots than the query has cells.
(2) A new pure metric, `mean_collapse_risk`, answers "would averaging this
population's positions land outside the population?" from the reference's
geometry alone; a route exposes it per gene set, and the Localize modal's live
parameter advice — extracted to a testable `src/lib/localizeAdvice.ts` — fires a
named warning when `weighted_mean` is selected against such a reference.

**Tech Stack:** Python 3 / numpy / scipy (`optimize.linear_sum_assignment`,
`spatial.cKDTree`); FastAPI; React 18 + TypeScript; pytest; vitest.

**Spec:** `docs/superpowers/specs/2026-08-09-localize-calibration-and-transport-design.md`
(§3 Injective assignment, §4 Change the default; phasing item 2).

**Phase 1, already merged and the reason this phase is decidable:**
`docs/superpowers/plans/2026-08-09-localize-calibration-harness.md` —
`backend/xcell/localize_metrics.py`, `POST /api/localize/evaluate_map`, and the
Map quality table.

## Global Constraints

- Branch `feat/localize-injective` off `main`; merge back with `--no-ff`.
- TDD: write the failing test, run it, watch it fail, then implement.
- Backend tests from `backend/`: `pixi run -e dev pytest`. Frontend:
  `cd frontend && npm test`, typecheck `npx tsc --noEmit`.
- `backend/tests/` has **no `conftest.py`** — each file is self-contained, with
  module-level `_x()` factories, seeded `np.random.default_rng(0)`, `csr_matrix`
  + `float32`.
- `backend/xcell/localize.py` and `backend/xcell/localize_metrics.py` are
  **pure**: arrays in, arrays and plain dicts out. No AnnData, no adaptor
  import. Both docstrings say so; keep it true.
- Heavy imports (`scipy.optimize`, `scipy.spatial`) stay **function-local**.
- Everything crossing the API must be JSON-serializable: **no `inf`, no `NaN`**.
  Use `None` for "not computable" — `localize._f` and `localize_metrics._f`
  already exist for exactly this.
- Route error mapping: `ValueError → 400`, `KeyError → 404`,
  `except HTTPException: raise` before any catch-all.
- **Validate synchronously in `prepare_*`**, so bad input is a 400 rather than a
  background task that dies a minute later.
- Styling is inline. Palette: panel `#16213e`, border `#0f3460`, inset
  `#0f1625`, accent `#4ecdc4`, alert `#e94560`, warn `#e9a23b`, muted
  `#aaa`/`#888`.
- Comments explain *why*, especially where a threshold was chosen.

**The claim this phase must not overstate.** Injective assignment fixes
**pile-up**, not **placement**. On the limb pair it takes distinct spots from
1,749 (all shared genes) or 865 (HVG) to 2,683 of 2,683, at a cost of mean
similarity 0.163 → 0.151 — and its axis fidelities (−0.056, +0.017, +0.033,
−0.027 against a reference ceiling of −0.327, +0.316, +0.306, −0.178) are just
as dead as greedy `best_match`'s. It is a *different point on the same
frontier*, not a step off it. Every string this plan writes into the UI says so.
Presenting it as strictly better would be the exact failure Phase 1 was built to
prevent.

**Why it is off by default anyway.** Injectivity asserts that the query's
composition matches the tissue's. Dissociation biases which cell types survive,
so forcing one-to-one pushes over-represented types into locations they do not
belong.

## File Structure

| file | responsibility |
|---|---|
| `backend/xcell/localize.py` (modify) | `injective_feasibility`, `injective_assignment`, `'injective'` in `AGGREGATIONS`, the full-matrix path in `project_knn` |
| `backend/tests/test_localize.py` (modify) | the estimator's behaviour on synthetic tissue |
| `backend/xcell/adaptor.py` (modify) | early feasibility check in `prepare_localize`; new `localize_reference_geometry` |
| `backend/tests/test_localize_adaptor.py` (modify) | the 400-before-the-task guard |
| `backend/xcell/localize_metrics.py` (modify) | `mean_collapse_risk` |
| `backend/tests/test_localize_metrics.py` (modify) | its answers on geometry with known shape |
| `backend/xcell/api/routes.py` (modify) | `POST /api/localize/reference_geometry` |
| `backend/tests/test_localize_metrics_routes.py` (modify) | the route contract |
| `frontend/src/lib/localizeAdvice.ts` (new) | `adviseParameters`, moved out of the modal and given real logic |
| `frontend/src/lib/localizeAdvice.test.ts` (new) | one case per rule, plus one per rule staying silent |
| `frontend/src/components/LocalizeModal.tsx` (modify) | the `injective` option, the geometry fetch, the guide |

**Module API**, fixed here so every task agrees:

```python
# localize.py
def injective_feasibility(n_query: int, n_ref: int) -> str | None
def injective_assignment(similarity: np.ndarray) -> np.ndarray      # (n_query,) int64

# localize_metrics.py
def mean_collapse_risk(coords, scores, *, top: float = 0.2) -> dict

# adaptor.py  (called on the *reference* adaptor, about itself)
def localize_reference_geometry(gene_sets: dict[str, list[str]],
                                layer: str | None = None) -> dict
```

```ts
// frontend/src/lib/localizeAdvice.ts
export interface Advice { level: 'warn' | 'info'; text: string }
export interface PopulationGeometry {
  name: string
  risk: number | null
  centroid_membership: number | null
  population_fraction: number
  n_population: number
}
export const COLLAPSE_RISK: number
export function adviseParameters(o: LocalizeSettings): Advice[]
```

---

### Task 1: `injective` as an aggregation

**Files:**
- Modify: `backend/xcell/localize.py`
- Modify: `backend/tests/test_localize.py`

**Interfaces:**
- Consumes: nothing new.
- Produces:
  - `injective_feasibility(n_query: int, n_ref: int) -> str | None` — a message
    explaining why it cannot run on these sizes, or `None` if it can.
  - `injective_assignment(similarity: np.ndarray) -> np.ndarray` — `(n_query,)`
    `int64` column indices, all distinct.
  - `AGGREGATIONS` gains `'injective'`.

- [ ] **Step 1: Write the failing tests**

Append to `backend/tests/test_localize.py`. First extend the import at the top of
the file (line 16) so it reads:

```python
from xcell.localize import (
    AGGREGATIONS,
    METRICS,
    TRANSFORMS,
    cross_validate_localization,
    evaluate_localization,
    injective_assignment,
    injective_feasibility,
    project_knn,
    transform_expression,
)
```

Then append this section at the end of the file:

```python
# --- injective assignment: no spot absorbs many cells ---------------------

def _clones(n_ref=400, n_query=60, seed=0):
    """A query whose cells are near-identical.

    Greedy matching sends every one of them to the same reference cell, which is
    the pile-up injective assignment exists to remove: on the limb pair it put
    2,683 cells onto 865 distinct spots, 15 on a single one.
    """
    ref_coords, ref_expr = _tissue(n_ref, seed=seed)
    rng = np.random.default_rng(seed + 7)
    base = _gradient_genes(np.array([[50.0, 50.0]]), seed=seed)[0]
    q_expr = base[None, :] + rng.normal(0, 0.02, (n_query, len(base)))
    return ref_expr, ref_coords, q_expr.astype(np.float32)


def test_injective_assignment_beats_taking_each_rows_best():
    """Both rows prefer column 0. Taking the best *set* gives it to the row that
    gains more by it — which is the entire difference from best_match."""
    S = np.array([[0.9, 0.8], [1.0, 0.1]])
    assert list(S.argmax(axis=1)) == [0, 0]          # greedy collides
    assert list(injective_assignment(S)) == [1, 0]


def test_injective_assignment_uses_each_spot_at_most_once():
    S = np.random.default_rng(0).random((40, 60))
    assert len(set(injective_assignment(S).tolist())) == 40


def test_injective_assignment_needs_a_spot_for_every_cell():
    assert injective_feasibility(5, 9) is None
    assert 'at least one reference spot' in (injective_feasibility(9, 5) or '')
    with pytest.raises(ValueError, match='at least one reference spot'):
        injective_assignment(np.zeros((9, 5)))


def test_a_similarity_matrix_too_large_to_hold_is_refused_by_size():
    """It materializes the whole query x reference matrix, which the chunked
    top-k path never does — so the size limit is a real one, not a formality."""
    assert injective_feasibility(200_000, 200_000) is not None
    assert 'memory' in injective_feasibility(200_000, 200_000)


def test_injective_spreads_what_best_match_piles_up():
    ref_expr, ref_coords, q_expr = _clones()
    greedy = project_knn(q_expr, ref_expr, ref_coords, k=15, aggregation='best_match')
    inj = project_knn(q_expr, ref_expr, ref_coords, k=15, aggregation='injective')
    assert len(np.unique(greedy.coords, axis=0)) < 10
    assert len(np.unique(inj.coords, axis=0)) == 60


def test_injective_lands_every_cell_on_a_real_reference_position():
    ref_expr, ref_coords, q_expr, _ = _matched()
    p = project_knn(q_expr, ref_expr, ref_coords, k=10, aggregation='injective')
    for c in p.coords:
        assert np.isclose(ref_coords, c).all(axis=1).any()


def test_injective_still_recovers_a_recoverable_tissue():
    """Spreading the cells must not be bought by placing them badly."""
    ref_expr, ref_coords, q_expr, q_coords = _matched()
    p = project_knn(q_expr, ref_expr, ref_coords, k=10, aggregation='injective')
    centroid_error = float(np.median(
        np.linalg.norm(q_coords - ref_coords.mean(axis=0), axis=1)
    ))
    assert _median_error(p.coords, q_coords) < centroid_error / 2


def test_injective_is_refused_when_the_reference_is_too_small():
    ref_expr, ref_coords, q_expr, _ = _matched(n_ref=40, n_query=120)
    with pytest.raises(ValueError, match='at least one reference spot'):
        project_knn(q_expr, ref_expr, ref_coords, k=10, aggregation='injective')


def test_injective_reports_the_section_of_the_spot_it_assigned():
    """The assignment is over the whole reference, so the honest section label is
    the assigned spot's own — not the weighted majority of the k neighbours,
    which is what the *averaging* estimators need and this one does not."""
    ref_expr, ref_coords, sections, query = _two_sections()
    p = project_knn(query, ref_expr, ref_coords, k=20,
                    ref_sections=sections, aggregation='injective')
    assert p.sections is not None
    for coord, label in zip(p.coords, p.sections):
        hit = np.isclose(ref_coords, coord).all(axis=1)
        assert label in set(sections[hit])
```

- [ ] **Step 2: Run them and watch them fail**

```bash
cd backend && pixi run -e dev pytest tests/test_localize.py -q
```

Expected: `ImportError: cannot import name 'injective_assignment' from 'xcell.localize'`
— which fails the whole file, including the tests that passed before. That is
the expected state; Step 4 restores them.

- [ ] **Step 3: Implement**

In `backend/xcell/localize.py`, change line 41:

```python
AGGREGATIONS = ('weighted_mean', 'median', 'best_match', 'densest', 'injective')
```

Add below the `_CHUNK` constant (line 45):

```python
# Injective assignment needs the whole query x reference similarity matrix at
# once; the chunked top-k path never materializes it. 1e8 float32 entries is
# 400 MB, which is the ceiling this refuses past. Time is not the binding
# constraint: 2,683 x 9,773 (26 M entries, 105 MB) solves in 0.2 s.
_MAX_INJECTIVE_ENTRIES = 100_000_000
```

Add these two functions immediately after `_aggregate` (i.e. at the end of the
`# --- aggregation ---` section, before `# --- projection ---`):

```python
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
    if n_query * n_ref > _MAX_INJECTIVE_ENTRIES:
        return (
            f'Injective assignment holds the whole {n_query:,} x {n_ref:,} '
            f'similarity matrix in memory, which is beyond what this refuses to '
            f'allocate. Subsample the query or the reference, or use best_match.'
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
    axis fidelity −0.056 / +0.017 / +0.033 / −0.027 against a reference ceiling
    of −0.327 / +0.316 / +0.306 / −0.178, which is greedy ``best_match`` to
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
```

Now wire it into `project_knn`. After the `k = max(1, min(int(k), ref_expr.shape[0]))`
line, insert the early check:

```python
    # Checked before the matching work rather than after it: on a real pair the
    # similarity blocks take tens of seconds, and failing then would waste them.
    needs_full = aggregation == 'injective'
    if needs_full:
        problem = injective_feasibility(query_expr.shape[0], ref_expr.shape[0])
        if problem:
            raise ValueError(problem)
```

Then, just before the chunk loop (after `sims = np.empty((n_query, k), dtype=float)`):

```python
    # The chunked path keeps only the top k, so injective assignment — which is
    # a global decision — needs the blocks kept as they are computed. Storing
    # them costs memory, not a second pass.
    full = np.empty((n_query, r.shape[0]), dtype=np.float32) if needs_full else None
```

Inside the loop, immediately after `block = _similarity_block(...)`:

```python
        if full is not None:
            full[start:stop] = block
```

Finally, replace the single line `coords = _aggregate(neighbor_coords, weights, aggregation)`
with:

```python
    if needs_full:
        if progress is not None:
            progress(1.0, f'Assigning {n_query:,} cells to distinct spots')
        assigned = injective_assignment(full)
        coords = ref_coords[assigned].astype(float)
        if sections_out is not None:
            # The assignment ranges over the whole reference, so the honest
            # label is the assigned spot's own section rather than the majority
            # of the k neighbours — that majority exists to stop an *average*
            # landing in the gap between cuts, and nothing is being averaged.
            sections_out = ref_sections[assigned]
    else:
        coords = _aggregate(neighbor_coords, weights, aggregation)
```

- [ ] **Step 4: Run the tests**

```bash
cd backend && pixi run -e dev pytest tests/test_localize.py -q
```

Expected: all pass, including `test_every_metric_and_aggregation_produces_a_usable_map`,
which now covers `injective` across all three metrics for free (its fixture is
400 reference cells against 120 query cells, so injective is feasible there).
If that one fails only for `euclidean/injective`, do **not** loosen it — check
that `full` is being filled with the same `block` the top-k path uses, since
euclidean's block is `-distance` and a sign error there is silent everywhere
else.

- [ ] **Step 5: Commit**

```bash
git add backend/xcell/localize.py backend/tests/test_localize.py
git commit -m "feat(localize): injective assignment — one distinct spot per cell"
```

---

### Task 2: reachable through the API, and refused before the task starts

`prepare_localize` already validates `aggregation` against `lz.AGGREGATIONS`, so
`injective` flows through the route the moment Task 1 lands. What does not flow
through is the size guard: `project_knn` raises it inside `compute_fn`, which is
a background task that dies a minute later rather than a 400 the user can read.

**Files:**
- Modify: `backend/xcell/adaptor.py` — `prepare_localize` (~line 8035)
- Modify: `backend/tests/test_localize_adaptor.py`

**Interfaces:**
- Consumes: `localize.injective_feasibility` (Task 1).
- Produces: no new names. `prepare_localize` raises `ValueError` for an
  infeasible injective run, which the existing route maps to 400.

- [ ] **Step 1: Write the failing tests**

Append to `backend/tests/test_localize_adaptor.py`:

```python
# --- injective assignment -------------------------------------------------

def test_injective_gives_every_cell_its_own_spot():
    ref, query, _ = _pair()
    bundle = ref.spatial_reference_bundle()
    _run(query, bundle, k=10, aggregation='injective')
    coords = query.adata.obsm['X_spatial_pred']
    assert len(np.unique(coords, axis=0)) == query.n_cells


def test_injective_is_a_validation_error_not_a_dead_background_task():
    """The house rule: bad input is a 400 from prepare_*, never a task that
    fails a minute in. A 200-cell query cannot be injectively assigned to a
    180-spot reference, and the user has to learn that immediately."""
    ref_ad, coords = _spatial()
    big_q, _ = _query_from(ref_ad, coords, n=200)
    ref = DataAdaptor('spatial.h5ad', adata=ref_ad)
    query = DataAdaptor('scrna.h5ad', adata=big_q)
    bundle = ref.spatial_reference_bundle()
    with pytest.raises(ValueError, match='at least one reference spot'):
        query.prepare_localize(bundle, k=10, aggregation='injective')


def test_the_other_aggregations_do_not_care_about_the_size_ratio():
    """The gate is injective's alone — nothing else acquired a constraint."""
    ref_ad, coords = _spatial()
    big_q, _ = _query_from(ref_ad, coords, n=200)
    ref = DataAdaptor('spatial.h5ad', adata=ref_ad)
    query = DataAdaptor('scrna.h5ad', adata=big_q)
    out = _run(query, ref.spatial_reference_bundle(), k=10,
               aggregation='weighted_mean')
    assert out['n_cells'] == 200
```

- [ ] **Step 2: Run them and watch the middle one fail**

```bash
cd backend && pixi run -e dev pytest tests/test_localize_adaptor.py -q -k injective
```

Expected: `test_injective_is_a_validation_error_not_a_dead_background_task`
fails — `prepare_localize` returns the two closures happily, and nothing raises
until `compute_fn()` is called. The other two should already pass on Task 1's
work.

- [ ] **Step 3: Implement**

In `backend/xcell/adaptor.py`, in `prepare_localize`, immediately after the
existing aggregation check:

```python
        if aggregation not in lz.AGGREGATIONS:
            raise ValueError(f"aggregation must be one of {list(lz.AGGREGATIONS)}")
        if aggregation == 'injective':
            # Same check the estimator makes, made here so an impossible run is
            # a 400 the user reads rather than a task that dies a minute in.
            problem = lz.injective_feasibility(
                int(self.n_cells), int(len(bundle['coords'])),
            )
            if problem:
                raise ValueError(problem)
```

- [ ] **Step 4: Run the tests**

```bash
cd backend && pixi run -e dev pytest tests/test_localize_adaptor.py tests/test_localize_routes.py -q
```

Expected: all pass.

- [ ] **Step 5: Commit**

```bash
git add backend/xcell/adaptor.py backend/tests/test_localize_adaptor.py
git commit -m "feat(localize): refuse an infeasible injective run with a 400"
```

---

### Task 3: `mean_collapse_risk` — the metric that reads the reference alone

The spec asks the parameter advice to warn "when the aggregation is
`weighted_mean` and the reference has any strongly peripheral or annular
population, which the spatial-pattern metric can detect on the reference alone."

This is that detector, distilled. Simulating a whole run against the reference
would need a *k* and a metric and would then be a run; the failure does not
depend on either. `weighted_mean` predicts, for every query cell of a type,
roughly the centroid of that type's reference members — so whether that centroid
is a place the type occupies is a property of the reference's geometry and
nothing else. **The mean of a ring is its hole.**

**Files:**
- Modify: `backend/xcell/localize_metrics.py`
- Modify: `backend/tests/test_localize_metrics.py`

**Interfaces:**
- Consumes: `_f` (already in the module).
- Produces: `mean_collapse_risk(coords, scores, *, top: float = 0.2) -> dict`
  with keys `risk` (`float|None`), `centroid_membership` (`float|None`),
  `population_fraction` (`float`), `centroid` (`[float, float] | None`),
  `n_population` (`int`).

- [ ] **Step 1: Write the failing tests**

Append to `backend/tests/test_localize_metrics.py`. `_disc` is already defined
at the top of that file — a filled unit disc centred at the origin — so every
case below is geometry whose answer is known by construction.

```python
from xcell.localize_metrics import mean_collapse_risk


def test_collapse_risk_flags_an_annular_population():
    """The limb failure, reduced to geometry. The epidermis is a ring; the mean
    of a ring is its hole; weighted_mean predicts every epidermal cell into it."""
    ref = _disc(seed=0)
    out = mean_collapse_risk(ref, np.linalg.norm(ref, axis=1))   # high on the rim
    assert out['risk'] > 0.9
    assert out['centroid_membership'] == pytest.approx(0.0, abs=0.05)


def test_collapse_risk_flags_a_population_in_two_distant_patches():
    """The other shape weighted_mean cannot represent: the mean lands in the
    gap between the patches, which is a location the population never occupies."""
    ref = _disc(seed=0)
    out = mean_collapse_risk(ref, np.abs(ref[:, 0]))             # both left and right
    assert out['risk'] > 0.9


def test_collapse_risk_clears_a_compact_population():
    ref = _disc(seed=0)
    blob = -np.linalg.norm(ref - np.array([0.4, 0.3]), axis=1)
    out = mean_collapse_risk(ref, blob)
    assert out['risk'] < 0.1
    assert out['centroid_membership'] > 0.9


def test_collapse_risk_does_not_cry_wolf_on_a_scattered_population():
    """A type spread through the tissue is not what this catches — the mean
    lands in the middle, which is as good a guess as any. Advice that fires on
    an ordinary population is noise, and would train the user to ignore it."""
    ref = _disc(seed=0)
    rng = np.random.default_rng(4)
    assert mean_collapse_risk(ref, rng.random(len(ref)))['risk'] < 0.5


def test_collapse_risk_reports_where_the_mean_would_land():
    """The centroid is returned because the warning is about a *place*, and a
    user who wants to check it needs the coordinate."""
    ref = _disc(seed=0)
    out = mean_collapse_risk(ref, np.linalg.norm(ref, axis=1))
    assert out['centroid'] == pytest.approx([0.0, 0.0], abs=0.1)
    assert out['n_population'] == pytest.approx(0.2 * len(ref), rel=0.02)


def test_collapse_risk_is_json_safe_on_degenerate_input():
    import json
    ref = _disc(n=200, seed=0)
    json.dumps(mean_collapse_risk(ref, np.ones(len(ref))))       # no variation
    json.dumps(mean_collapse_risk(_disc(n=5, seed=0), np.arange(5.0)))
    assert mean_collapse_risk(ref, np.ones(len(ref)))['risk'] is None
```

- [ ] **Step 2: Run them and watch them fail**

```bash
cd backend && pixi run -e dev pytest tests/test_localize_metrics.py -q -k collapse
```

Expected: `ImportError: cannot import name 'mean_collapse_risk'`.

- [ ] **Step 3: Implement**

Append to `backend/xcell/localize_metrics.py`:

```python
def mean_collapse_risk(coords, scores, *, top: float = 0.2) -> dict[str, Any]:
    """Would averaging this population's positions land outside the population?

    Reference-only, and that is the point: ``weighted_mean`` predicts, for every
    query cell of a type, roughly the centroid of that type's reference members.
    Whether that centroid is a place the type actually occupies depends on the
    reference's geometry and on nothing else — no query, no parameters, no run.
    So the warning can be raised while the parameters are still being chosen
    rather than after a map has been made and believed.

    **The mean of a ring is its hole.** The population is the top ``top``
    fraction of cells by score. Around its own centroid, take the cells in the
    nearest 2% of the tissue and ask how many belong to the population: at
    chance that fraction is ``top``, for a compact blob it approaches 1, and for
    a ring or a rim it is 0.

    What it does **not** catch: a population scattered through a *ring-shaped
    tissue*, where the mean is off-tissue but not off-population. That one needs
    an actual run, and ``dispersion``'s ``frac_outside`` is what reports it.

    Args:
        coords: ``(n, 2)`` reference coordinates.
        scores: ``(n,)`` marker score per reference cell.
        top: what fraction of the reference counts as the population.

    Returns:
        risk: ``1 - membership/fraction`` clipped to ``[0, 1]``. 0 means the
            mean lands among the population at least as often as chance would;
            1 means none of the cells around it belong to it. ``None`` when
            there is nothing to measure.
        centroid_membership: the raw fraction, so the UI can quote it.
        population_fraction: what chance would be, i.e. the realized ``top``.
        centroid: ``[x, y]`` — where the estimator would put every one of them.
        n_population: how many reference cells the population has.
    """
    from scipy.spatial import cKDTree

    C = np.asarray(coords, dtype=float)[:, :2]
    s = np.asarray(scores, dtype=float).ravel()
    n = len(C)

    empty: dict[str, Any] = {
        'risk': None, 'centroid_membership': None,
        'population_fraction': _f(top), 'centroid': None, 'n_population': 0,
    }
    # Under ~20 cells the local window is the whole tissue and the fraction is a
    # coin flip; a constant score has no population to speak of.
    if n < 20 or len(s) != n or not np.isfinite(s).any() or np.allclose(s, s[0]):
        return empty

    n_pop = max(3, int(round(float(top) * n)))
    member = np.zeros(n, dtype=bool)
    member[np.argsort(-s, kind='stable')[:n_pop]] = True
    fraction = n_pop / n
    centroid = C[member].mean(axis=0)

    # 2% of the tissue around the mean: small enough to describe the point the
    # estimator actually returns, large enough that the fraction is not decided
    # by one or two cells. Chance level stays exactly `fraction` at any size.
    k_local = max(10, int(round(0.02 * n)))
    _, idx = cKDTree(C).query(centroid, k=k_local)
    membership = float(member[np.asarray(idx).ravel()].mean())

    return {
        'risk': _f(np.clip(1.0 - membership / fraction, 0.0, 1.0)),
        'centroid_membership': _f(membership),
        'population_fraction': _f(fraction),
        'centroid': [_f(centroid[0]), _f(centroid[1])],
        'n_population': int(n_pop),
    }
```

- [ ] **Step 4: Run the tests**

```bash
cd backend && pixi run -e dev pytest tests/test_localize_metrics.py -q
```

Expected: all pass. If `test_collapse_risk_does_not_cry_wolf_on_a_scattered_population`
is borderline, the local window is too small for the reference size — check
`k_local`, do **not** raise the threshold in the test. A metric that fires on an
ordinary population is worse than no metric, because the user learns to skip
the line.

- [ ] **Step 5: Commit**

```bash
git add backend/xcell/localize_metrics.py backend/tests/test_localize_metrics.py
git commit -m "feat(localize): detect populations a mean estimator would collapse"
```

---

### Task 4: the reference-geometry route

**Files:**
- Modify: `backend/xcell/adaptor.py` — add `localize_reference_geometry` after
  `evaluate_localization_maps` (~line 8248)
- Modify: `backend/xcell/api/routes.py` — add after `/localize/evaluate_map`
- Modify: `backend/tests/test_localize_metrics_routes.py`

**Interfaces:**
- Consumes: `localize_metrics.mean_collapse_risk` (Task 3).
- Produces:
  - `DataAdaptor.localize_reference_geometry(gene_sets, layer=None) -> {'populations': [...], 'skipped_gene_sets': [...], 'n_reference_cells': int}`
    where each population is `{'name': str, **mean_collapse_risk(...)}`, sorted
    riskiest first.
  - `POST /api/localize/reference_geometry`

Note the asymmetry with `evaluate_map`: this one is called **on the reference
adaptor about itself**, so it needs no bundle. `spatial_reference_bundle` exists
so the *query* never reaches into the reference's AnnData; here there is no
query involved at all.

- [ ] **Step 1: Add the adaptor method**

In `backend/xcell/adaptor.py`, after `evaluate_localization_maps`:

```python
    def localize_reference_geometry(
        self, gene_sets: dict[str, list[str]], layer: str | None = None,
    ) -> dict[str, Any]:
        """Which of these populations would a mean-of-neighbours estimator lose?

        Called on the **reference** adaptor, about itself — there is no query and
        no prediction, which is what lets the answer be shown while the
        parameters are still being chosen. See
        :func:`xcell.localize_metrics.mean_collapse_risk` for what the number is.

        Args:
            gene_sets: population name -> gene list. Genes absent from this
                dataset are dropped; a set left with none is skipped and named
                in ``skipped_gene_sets`` rather than failing the whole call.
            layer: which matrix to score from; ``None`` is ``.X``.
        """
        from xcell import localize_metrics as lm

        spatial_key = self._get_spatial_key()
        if spatial_key is None:
            raise ValueError(
                'This dataset has no spatial coordinates, so it cannot act as a '
                "reference. Expected .obsm['spatial'] or .obsm['X_spatial']."
            )
        coords = np.asarray(self.adata.obsm[spatial_key], dtype=float)[:, :2]
        matrix = self._resolve_source_matrix(layer)

        populations: list[dict[str, Any]] = []
        skipped: list[str] = []
        for name, genes in gene_sets.items():
            present, _ = self._split_present_genes(list(genes))
            if not present:
                skipped.append(str(name))
                continue
            block = matrix[:, [self.adata.var_names.get_loc(g) for g in present]]
            block = block.toarray() if hasattr(block, 'toarray') else np.asarray(block)
            scores = np.asarray(block, dtype=float).mean(axis=1)
            populations.append({
                'name': str(name),
                'n_genes_used': len(present),
                **lm.mean_collapse_risk(coords, scores),
            })

        # Riskiest first: the UI names the top few, and which few matters.
        populations.sort(key=lambda p: (p['risk'] is None, -(p['risk'] or 0.0)))
        return {
            'populations': populations,
            'skipped_gene_sets': skipped,
            'n_reference_cells': int(coords.shape[0]),
        }
```

- [ ] **Step 2: Write the failing route tests**

Append to `backend/tests/test_localize_metrics_routes.py`. That file's
`_spatial_ref()` already builds exactly the fixture this needs — a 20×20 grid
where the gene `Rim` is high on the outer quarter, i.e. an annular population —
and `_install()` puts it in `primary` with the query in `secondary`.

```python
# --- reference geometry: the warning weighted_mean has earned ---------------

def _geometry(gene_sets):
    return client.post('/api/localize/reference_geometry?dataset=secondary',
                       json={'reference': 'primary', 'dataset': 'secondary',
                             'gene_sets': gene_sets})


def test_reference_geometry_flags_the_annular_population():
    _install()
    r = _geometry({'rim': ['Rim'], 'noise': ['B']})
    assert r.status_code == 200, r.text
    pops = {p['name']: p for p in r.json()['populations']}
    assert pops['rim']['risk'] > 0.8
    assert pops['noise']['risk'] < 0.5


def test_reference_geometry_returns_the_riskiest_first():
    """The modal names the top few, so the order is part of the contract."""
    _install()
    names = [p['name'] for p in _geometry({'noise': ['B'], 'rim': ['Rim']}).json()['populations']]
    assert names == ['rim', 'noise']


def test_a_gene_set_with_no_usable_genes_is_reported_not_fatal():
    _install()
    body = _geometry({'rim': ['Rim'], 'nonsense': ['NotAGene']}).json()
    assert body['skipped_gene_sets'] == ['nonsense']
    assert [p['name'] for p in body['populations']] == ['rim']


def test_reference_geometry_is_json_safe():
    import json
    _install()
    json.dumps(_geometry({'rim': ['Rim']}).json())


def test_the_reference_cannot_be_the_dataset_being_localized():
    _install()
    r = client.post('/api/localize/reference_geometry?dataset=secondary',
                    json={'reference': 'secondary', 'dataset': 'secondary',
                          'gene_sets': {'rim': ['Rim']}})
    assert r.status_code == 400
```

- [ ] **Step 3: Run them and watch them fail**

```bash
cd backend && pixi run -e dev pytest tests/test_localize_metrics_routes.py -q
```

Expected: 404 on the new tests — the route does not exist.

- [ ] **Step 4: Add the route**

In `backend/xcell/api/routes.py`, after `localize_evaluate_map`:

```python
class ReferenceGeometryRequest(BaseModel):
    reference: str = "primary"
    gene_sets: dict[str, list[str]] = {}
    layer: str | None = None
    dataset: str | None = None


@router.post("/localize/reference_geometry")
def localize_reference_geometry(
    request: ReferenceGeometryRequest, dataset: str | None = Query(None),
):
    """Which populations in the reference a mean estimator would collapse.

    Reads the reference and nothing else, so the answer is available before any
    map exists — which is the point. A user who learns after the run that
    ``weighted_mean`` puts their epidermis in the middle of the bud has already
    believed the picture.
    """
    query_slot = request.dataset or dataset
    reference = _resolve_reference(request.reference, query_slot)
    try:
        return reference.localize_reference_geometry(
            request.gene_sets, layer=request.layer,
        )
    except HTTPException:
        raise
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except KeyError as e:
        raise HTTPException(status_code=404, detail=str(e))
```

- [ ] **Step 5: Run the file, then the whole suite**

```bash
cd backend && pixi run -e dev pytest tests/test_localize_metrics_routes.py -q
cd backend && pixi run -e dev pytest -q
```

Expected: all pass, no regressions against the 678 already there.

- [ ] **Step 6: Commit**

```bash
git add backend/xcell/adaptor.py backend/xcell/api/routes.py \
        backend/tests/test_localize_metrics_routes.py
git commit -m "feat(localize): reference geometry route for pre-run advice"
```

---

### Task 5: the modal — an option, and a warning with a name in it

**Files:**
- Create: `frontend/src/lib/localizeAdvice.ts`
- Create: `frontend/src/lib/localizeAdvice.test.ts`
- Modify: `frontend/src/components/LocalizeModal.tsx`

**Interfaces:**
- Consumes: `POST /api/localize/reference_geometry` (Task 4);
  `roles.referenceSlot` / `roles.querySlot`, `refs`, and the `geneSets` memo —
  all already in the modal.
- Produces: `adviseParameters`, `COLLAPSE_RISK`, `Advice`,
  `PopulationGeometry`, `LocalizeSettings` from `src/lib/localizeAdvice`.

`adviseParameters` currently lives inside `LocalizeModal.tsx` (lines ~113–170)
and is untested. This task moves it to `src/lib/` — where `contourAdvice.ts` and
`localizeRoles.ts` already live for the same reason — because it is about to
acquire logic worth testing.

- [ ] **Step 1: Write the failing tests**

Create `frontend/src/lib/localizeAdvice.test.ts`:

```ts
import { describe, it, expect } from 'vitest'
import {
  adviseParameters, COLLAPSE_RISK,
  type LocalizeSettings, type PopulationGeometry,
} from './localizeAdvice'

const OK: LocalizeSettings = {
  k: 15, transform: 'zscore', metric: 'correlation', aggregation: 'best_match',
  minConfidence: 0, nReferenceCells: 9145, nQueryCells: 2683,
  nSharedGenes: 12449, populations: [],
}

const pop = (name: string, risk: number | null): PopulationGeometry => ({
  name, risk, centroid_membership: 0, population_fraction: 0.2, n_population: 100,
})

const texts = (a: { text: string }[]) => a.map((x) => x.text).join(' | ')

describe('the weighted_mean collapse warning', () => {
  it('names the populations this reference would lose', () => {
    const a = adviseParameters({
      ...OK, aggregation: 'weighted_mean',
      populations: [pop('skin', 0.98), pop('muscle', 0.02)],
    })
    expect(texts(a)).toContain('skin')
    expect(texts(a)).not.toContain('muscle')
    expect(a.some((x) => x.level === 'warn')).toBe(true)
  })

  it('stays quiet when no population is hollow', () => {
    const a = adviseParameters({
      ...OK, aggregation: 'weighted_mean', populations: [pop('muscle', 0.02)],
    })
    expect(a.filter((x) => x.level === 'warn')).toHaveLength(0)
  })

  it('stays quiet for an aggregation that does not average', () => {
    const a = adviseParameters({
      ...OK, aggregation: 'best_match', populations: [pop('skin', 0.98)],
    })
    expect(texts(a)).not.toContain('skin')
  })

  it('ignores a population whose risk could not be computed', () => {
    const a = adviseParameters({
      ...OK, aggregation: 'weighted_mean', populations: [pop('tiny', null)],
    })
    expect(a.filter((x) => x.level === 'warn')).toHaveLength(0)
  })

  it('caps the list rather than printing every population', () => {
    const many = ['a', 'b', 'c', 'd', 'e'].map((n) => pop(n, 0.9))
    const t = texts(adviseParameters({ ...OK, aggregation: 'weighted_mean', populations: many }))
    expect(t).toContain('+2 more')
  })

  it('uses the documented threshold', () => {
    const just = adviseParameters({
      ...OK, aggregation: 'weighted_mean', populations: [pop('edge', COLLAPSE_RISK)],
    })
    const under = adviseParameters({
      ...OK, aggregation: 'weighted_mean', populations: [pop('edge', COLLAPSE_RISK - 0.01)],
    })
    expect(just.some((x) => x.level === 'warn')).toBe(true)
    expect(under.some((x) => x.level === 'warn')).toBe(false)
  })
})

describe('injective', () => {
  it('says what one-to-one asserts about the query', () => {
    const t = texts(adviseParameters({ ...OK, aggregation: 'injective' }))
    expect(t).toContain('composition')
  })

  it('does not claim to be better than best match', () => {
    const t = texts(adviseParameters({ ...OK, aggregation: 'injective' }))
    expect(t).toContain('pile-up, not placement')
  })

  it('warns when there are more cells than spots', () => {
    const a = adviseParameters({
      ...OK, aggregation: 'injective', nReferenceCells: 900, nQueryCells: 2683,
    })
    expect(a.some((x) => x.level === 'warn')).toBe(true)
    expect(texts(a)).toContain('2,683')
  })

  it('does not warn about size when the reference is big enough', () => {
    const a = adviseParameters({ ...OK, aggregation: 'injective' })
    expect(a.filter((x) => x.level === 'warn')).toHaveLength(0)
  })
})

describe('the rules that were already there', () => {
  it('still catches a k that is too small', () => {
    expect(texts(adviseParameters({ ...OK, k: 3 }))).toContain('k=3')
  })

  it('still catches none + euclidean', () => {
    const t = texts(adviseParameters({ ...OK, transform: 'none', metric: 'euclidean' }))
    expect(t).toContain('sequencing depth')
  })

  it('says nothing at all about sensible settings', () => {
    expect(adviseParameters(OK)).toHaveLength(0)
  })
})
```

- [ ] **Step 2: Run them and watch them fail**

```bash
cd frontend && npm test -- localizeAdvice
```

Expected: `Failed to resolve import "./localizeAdvice"`.

- [ ] **Step 3: Create the module**

Create `frontend/src/lib/localizeAdvice.ts`. The first three rule groups are
moved verbatim from `LocalizeModal.tsx` lines 113–170; the `weighted_mean` and
`injective` blocks are the new work.

```ts
/**
 * What is wrong with the current Localize settings, given this reference.
 *
 * Only fires on the current selection — a static table of caveats gets skimmed;
 * a line that appears when you pick the thing does not. Kept pure and out of the
 * modal so each rule can be tested, including the cases where it must stay
 * silent: advice that fires on ordinary settings trains the user to ignore it.
 */

export interface Advice { level: 'warn' | 'info'; text: string }

/** One population's geometry in the reference, from
 *  `POST /api/localize/reference_geometry`. */
export interface PopulationGeometry {
  name: string
  risk: number | null
  centroid_membership: number | null
  population_fraction: number
  n_population: number
}

export interface LocalizeSettings {
  k: number
  transform: string
  metric: string
  aggregation: string
  minConfidence: number
  nReferenceCells: number
  nQueryCells: number
  nSharedGenes: number | null
  populations: PopulationGeometry[]
}

/**
 * Above this, a population's own mean is somewhere that population is not.
 *
 * At chance the risk is 0 by construction, and sampling noise in the metric's
 * 2%-of-tissue window keeps an ordinary scattered population under about 0.3.
 * 0.5 is clear of that noise and still catches anything meaningfully hollow —
 * the limb's epidermis scores ~1.0.
 */
export const COLLAPSE_RISK = 0.5

export function adviseParameters(o: LocalizeSettings): Advice[] {
  const out: Advice[] = []

  if (o.k < 5) {
    out.push({ level: 'warn', text:
      `k=${o.k} is too few to estimate confidence from — the spread of ${o.k} points is noisy, so the score will be unreliable in both directions.` })
  }
  if (o.nReferenceCells > 0 && o.k > 0.1 * o.nReferenceCells) {
    out.push({ level: 'warn', text:
      `k=${o.k} is over a tenth of the ${o.nReferenceCells.toLocaleString()} reference cells. Every prediction averages a large slice of the tissue, so all of them drift toward its centre.` })
  }

  if (o.transform === 'none' && o.metric === 'euclidean') {
    out.push({ level: 'warn', text:
      'none + euclidean compares raw magnitudes. Across two platforms, sequencing depth will dominate the distance and swamp the biology.' })
  } else if (o.transform === 'none') {
    out.push({ level: 'warn', text:
      'No per-dataset transform. Only safe when both datasets are already normalized the same way — otherwise per-gene capture differences drive the matches.' })
  }

  if (o.aggregation === 'densest' && o.k < 12) {
    out.push({ level: 'info', text:
      `densest picks the tightest cluster among the neighbours, and with k=${o.k} there is not much to cluster — it will behave much like the mean. Raise k to about 20 to get the benefit.` })
  }
  if (o.aggregation === 'best_match') {
    out.push({ level: 'info', text:
      'best_match takes the single nearest cell’s position, so k no longer affects the coordinate — only the confidence score, which is still measured over k neighbours.' })
  }

  if (o.aggregation === 'injective') {
    if (o.nQueryCells > o.nReferenceCells) {
      out.push({ level: 'warn', text:
        `injective needs at least one reference spot per query cell, and there are ${o.nQueryCells.toLocaleString()} cells against ${o.nReferenceCells.toLocaleString()} spots. Subsample the query, or use best match.` })
    }
    out.push({ level: 'info', text:
      'injective gives every cell a different spot, which asserts that the query’s composition matches the tissue’s. Dissociation biases which cell types survive, so an over-represented type gets pushed into locations it does not belong. It fixes pile-up, not placement — its axis correlations are as flat as best match’s, so compare them under Map quality rather than assuming it is an improvement.' })
  }

  // The reference's own geometry, which is enough to know this in advance: the
  // mean of a ring is its hole, and weighted_mean returns roughly that mean for
  // every query cell of the type.
  const hollow = o.populations.filter((p) => p.risk != null && p.risk >= COLLAPSE_RISK)
  if (o.aggregation === 'weighted_mean' && hollow.length > 0) {
    const named = hollow.slice(0, 3).map((p) => p.name).join(', ')
    const more = hollow.length > 3 ? ` (+${hollow.length - 3} more)` : ''
    const one = hollow.length === 1
    out.push({ level: 'warn', text:
      `In this reference ${named}${more} ${one ? 'forms a ring or hugs the tissue edge' : 'form rings or hug the tissue edge'}, so the mean of ${one ? 'its' : 'their'} positions is a place none of those cells occupy. weighted mean will predict every query cell of ${one ? 'that type' : 'those types'} into that hole — this is what inverted the epidermis on an E11.5 limb, placing the embryo’s outermost tissue in the middle of the bud. best match or densest keep predictions on real tissue, at the cost of the gradients; score both under Map quality before choosing.` })
  } else if (o.aggregation === 'weighted_mean') {
    out.push({ level: 'info', text:
      'weighted_mean can land in a gap the tissue does not have, most visibly for a cell type present in two separate regions. densest or best_match keep predictions on real tissue.' })
  }

  if (o.nSharedGenes != null && o.nSharedGenes < 30) {
    out.push({ level: 'warn', text:
      `Only ${o.nSharedGenes} genes drive the similarity. That is enough to run but not to separate fine structure; treat the map as coarse and check the accuracy figures.` })
  }
  if (o.minConfidence > 0) {
    out.push({ level: 'info', text:
      `Cells scoring under ${o.minConfidence} will get no coordinate at all rather than a guessed one. Confidence depends on the gene basis, so a threshold tuned for one basis does not carry to another.` })
  }
  return out
}
```

- [ ] **Step 4: Run the tests**

```bash
cd frontend && npm test -- localizeAdvice
```

Expected: all pass. Note `says nothing at all about sensible settings` uses
`aggregation: 'best_match'`, which does emit its info line — if that case fails,
change `OK` to `aggregation: 'median'`, which has no rule of its own, rather
than weakening the assertion.

- [ ] **Step 5: Wire the modal to it**

In `frontend/src/components/LocalizeModal.tsx`:

**(a)** Delete the local `interface Advice` and the whole `adviseParameters`
function (lines ~113–170) and import them instead, beside the existing
`localizeRoles` import at line 4:

```tsx
import {
  adviseParameters, type PopulationGeometry,
} from '../lib/localizeAdvice'
```

**(b)** Add state beside `mapMetrics` (~line 200):

```tsx
  const [populations, setPopulations] = useState<PopulationGeometry[]>([])
```

**(c)** Add the fetch after the existing `useEffect`s that load slots and
suggestions:

```tsx
  // Which of the user's populations this reference would collapse under an
  // averaging estimator. Reference-only, so it runs before any prediction
  // exists — the whole point is to warn while the parameters are being chosen.
  useEffect(() => {
    if (!isOpen || !reference || !querySlot || geneSets.length === 0) {
      setPopulations([])
      return
    }
    const sets: Record<string, string[]> = {}
    geneSets.forEach((g) => { if (g.genes.length) sets[g.name] = g.genes })
    if (Object.keys(sets).length === 0) { setPopulations([]); return }

    let cancelled = false
    fetch(appendDataset(`${API}/localize/reference_geometry`, querySlot as DatasetSlot), {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ reference, dataset: querySlot, gene_sets: sets }),
    })
      .then((r) => (r.ok ? r.json() : null))
      .then((body) => { if (!cancelled && body) setPopulations(body.populations || []) })
      // Advice is a bonus. A failure here must not surface as a modal error and
      // must not block the run the user came to make.
      .catch(() => {})
    return () => { cancelled = true }
  }, [isOpen, reference, querySlot, geneSets])
```

**(d)** Extend the `adviseParameters` call (~line 421):

```tsx
  const advice = adviseParameters({
    k: Number(k) || 0,
    transform, metric, aggregation,
    minConfidence: Number(minConfidence) || 0,
    nReferenceCells: chosenRef?.n_cells ?? 0,
    nQueryCells: queryInfo?.n_cells ?? 0,
    nSharedGenes: basisGenes,
    populations,
  })
```

**(e)** Add the option to the Aggregation select (~line 577), disabled when the
reference cannot supply a spot per cell — the same gate the backend enforces,
stated where the choice is made:

```tsx
              <option value="best_match">best match</option>
              <option value="injective"
                      disabled={(queryInfo?.n_cells ?? 0) > (refInfo?.n_cells ?? 0)}>
                injective (a distinct spot each)
              </option>
```

**(f)** Update `TIPS.aggregation` (~line 84) to mention it:

```tsx
  aggregation: 'How the k neighbours become one point. weighted_mean is the classic estimator; densest lands the cell in real tissue when its neighbours sit in two separate patches, though it cannot say which patch; best_match snaps to a real reference cell; injective does the same but gives every cell a different one, which removes pile-up and asserts that the query’s composition matches the tissue’s.',
```

**(g)** In `ParameterGuide`, extend the **Aggregation** bullet (~line 822) —
after the `densest` sentence, before `median`:

```tsx
          <code>injective</code> is <code>best match</code> solved as a set
          rather than one cell at a time, so no reference spot absorbs many
          cells (measured on an E11.5 limb: 2,683 cells onto 2,683 distinct
          spots, up from 865, for a 7% drop in mean similarity). It needs at
          least as many spots as cells, and it assumes the query's composition
          matches the tissue's — which dissociation makes untrue in a way that
          pushes over-represented types where they do not belong. It fixes
          pile-up, not placement.
```

and add to the **Avoid** list (~line 851):

```tsx
        <li>Reading <code>injective</code> or <code>best match</code> as "more accurate" because the map fills the tissue. Filling the tissue and carrying a gradient are the two ends of one trade-off: on the limb pair, <code>weighted mean</code> kept the proximodistal gradient at 0.24 of a 0.34 ceiling while collapsing to 15% of the area, and both single-spot methods held the full area with essentially no gradient at all. Score them under <b>Map quality</b> and pick against what you need.</li>
```

- [ ] **Step 6: Typecheck and run the full frontend suite**

```bash
cd frontend && npx tsc --noEmit && npm test
```

Expected: clean typecheck, all tests pass (65 existing + the new file).

- [ ] **Step 7: Commit**

```bash
git add frontend/src/lib/localizeAdvice.ts frontend/src/lib/localizeAdvice.test.ts \
        frontend/src/components/LocalizeModal.tsx
git commit -m "feat(localize): injective option, and a named warning for weighted_mean"
```

---

### Task 6: Verify on the limb pair, document, merge

Both of this phase's claims are empirical and neither is proven by pytest: that
injective removes the pile-up on real data without buying placement, and that
the collapse warning fires on the epidermis and stays quiet elsewhere. Every
notable bug in this codebase passed the suite and died in the browser.

**Files:**
- Modify: `README.md`, `CHANGELOG.md`

- [ ] **Step 1: Start both servers**

```bash
pixi run backend      # :8000
pixi run dev          # :5173
```

Stop them later with `pkill -f "uvicorn xcell.main:app"; pkill -f vite` — killing
by port leaves a reloader that re-binds and serves stale data. **Editing a
backend file restarts the server and drops both loaded datasets**, so batch any
fixes rather than fixing one thing at a time.

- [ ] **Step 2: Load the limb pair**

Reference `E11.5_best_102424.h5ad` (X_spatial, 9,145 spots), query
`GSM4227224_E11_112125.h5ad` (2,683 cells), both under
`~/CahanLab Dropbox/.../projects/SJD/data/`. Remember `POST /api/load` takes the
slot as a **body** field (`{"file_path": ..., "slot": "secondary"}`);
`?dataset=secondary` there is silently ignored and loads into primary.

Seed the `skin` gene set through the API rather than by clicking — clicking the
Gene Panel creates empty sets:

```bash
curl -s -X PUT localhost:8000/api/gene_sets -H 'Content-Type: application/json' \
  -d '{"categories": {...}}'   # check the current blob with GET first and edit it
```

**Send the whole blob.** A partial `/api/gene_sets` body blanks the app —
`GenePanel.flattenGeneSets` and `MultiContourModal.getAllGeneSets` walk a fixed
category list. (Known, unfixed, out of scope here.)

- [ ] **Step 3: Confirm the warning fires, and only where it should**

Open Localize with the limb pair loaded. With **Aggregation = weighted mean**,
the advice must name `skin` and say the mean of its positions is a place none of
those cells occupy. Switch to **best match** and the line must disappear.

Cross-check the number outside the UI:

```bash
curl -s -X POST "localhost:8000/api/localize/reference_geometry?dataset=secondary" \
  -H 'Content-Type: application/json' \
  -d '{"reference":"primary","dataset":"secondary",
       "gene_sets":{"skin":["Perp","Krt18","Krt14","Krt5","Epcam","Cdh1"],
                    "muscle":["Myog","Myod1","Pax7","Des"]}}' \
  | python3 -m json.tool
```

Expected: `skin` with `risk` near 1 and `centroid_membership` near 0, ordered
first; `muscle` — which is a compact proximal mass, not a rim — well below it.
If `skin` does not score high, the metric is not detecting what Phase 1 already
measured as a −0.11 inverted pattern score, and that is a bug to investigate
before accepting the phase, not a threshold to lower.

- [ ] **Step 4: Run injective and score it against the maps already there**

In Localize, set **Aggregation = injective**, key `X_spatial_pred_inj`, and run.
Then press **Score predicted maps** and read the table next to the two variants
Phase 1 measured.

Expected, from the spec's measured frontier:

| embedding | area | spots used | skin |
|---|---|---|---|
| `X_spatial_pred` (weighted_mean) | ~0.15 | few | **negative** |
| `X_spatial_pred_best` (best_match) | ~1.0 | ~865–1,749 of 2,683 | positive |
| `X_spatial_pred_inj` (injective) | ~1.0 | **2,683 of 2,683** | positive |

The one number that must hold is `spots used` reaching 2,683 — that is the whole
claim. Hovering a marker cell shows the axis correlations; injective's are
expected to be as flat as best_match's, and **that is the honest result**, not a
failure to fix.

- [ ] **Step 5: Check the refusal path in the browser**

Swap the roles so the 2,683-cell dataset is the reference and the 9,145-spot one
is the query (**swap** button) — or load any pair where the query is larger. The
`injective` option must be disabled, and hitting the route directly must return
a readable 400 rather than a task that fails later:

```bash
curl -s -X POST "localhost:8000/api/localize/prepare?dataset=primary" \
  -H 'Content-Type: application/json' \
  -d '{"reference":"secondary","aggregation":"injective","k":15}' | python3 -m json.tool
```

- [ ] **Step 6: Stop the servers and write the docs**

```bash
pkill -f "uvicorn xcell.main:app"; pkill -f vite
```

`README.md` — in the Localize section, add `injective` to the aggregation list
with its gate and its caveat, and note that the parameter advice now reads the
reference's geometry before a run.

`CHANGELOG.md` — one entry under **Added** in `[Unreleased]`, matching the
house voice of the Map quality entry above it: what the pile-up was (865 spots,
15 cells on one), what injective costs (0.163 → 0.151 mean similarity), the
composition assumption that keeps it off by default, that it moves along the
frontier rather than off it, and the warning — what it measures, why it can be
computed from the reference alone, and the threshold's reasoning. Record the
real numbers from Step 4, not the spec's prototype numbers, and say if they
differ.

- [ ] **Step 7: Commit and merge**

```bash
git add README.md CHANGELOG.md
git commit -m "docs: injective assignment and the weighted_mean collapse warning"
cd backend && pixi run -e dev pytest -q
cd ../frontend && npx tsc --noEmit && npm test
git checkout main
git merge --no-ff feat/localize-injective
```

---

## Self-Review

**Spec coverage.** §3 Injective assignment: `linear_sum_assignment` → Task 1;
"available only when the reference has at least as many spots as the query has
cells" → `injective_feasibility`, enforced in the estimator (Task 1), in
`prepare_localize` as a 400 (Task 2), and as a disabled option (Task 5);
"off by default" → `AGGREGATIONS` gains a member, no default changes anywhere;
"the UI must say this rather than presenting it as strictly better" → the
`injective` advice line, its guide bullet, and the new Avoid item, each tested
or read in the browser. §4 Change the default: the spec moves the default to OT
**once it exists**, so Phase 2 owns only "until then, the existing parameter
advice should warn when the aggregation is `weighted_mean` and the reference has
any strongly peripheral or annular population, which the spatial-pattern metric
can detect on the reference alone" → Tasks 3–5. **`weighted_mean` remains the
default**, deliberately; moving it belongs to Phase 3.

**One spec sentence implemented differently, on purpose.** The spec suggests the
spatial-pattern metric can detect the annular case on the reference alone.
Reusing `spatial_pattern_fidelity` directly would mean simulating a run against
the reference, which needs a *k* and a metric and would then *be* a run. The
failure does not depend on either, so Task 3 distills it: the population's own
centroid, and whether the population is there. Same detection, no parameters,
and testable on geometry with a known answer.

**One thing deliberately not built.** `mean_collapse_risk` does not catch a
population scattered through a *ring-shaped tissue*, where the mean lands
off-tissue but not off-population. Adding a second "is the centroid on tissue at
all" statistic was considered and dropped: it would put two numbers with
different meanings behind one warning, and `dispersion`'s `frac_outside` already
reports that case after a run. The limitation is stated in the docstring rather
than left for a reader to discover.

**Placeholder scan.** No TBDs. Every code step carries the code; every test step
carries the test; every threshold (`_MAX_INJECTIVE_ENTRIES`, `top=0.2`, the 2%
window, `COLLAPSE_RISK`) carries the reasoning for its value in a comment.

**Type consistency.** `injective_feasibility` / `injective_assignment` are
spelled identically in `localize.py`, its tests, and `adaptor.py`.
`mean_collapse_risk`'s five keys (`risk`, `centroid_membership`,
`population_fraction`, `centroid`, `n_population`) are produced in Task 3,
spread into the route payload in Task 4, and typed as `PopulationGeometry` in
Task 5 — the TypeScript interface omits `n_genes_used`, which the adaptor adds
and the frontend does not read; that is intentional and not a mismatch.
`populations` is the payload key in Task 4 and the state name in Task 5.

**Known soft spots**, each with an in-plan instruction rather than a silent
assumption: whether `test_every_metric_and_aggregation_produces_a_usable_map`
passes for `euclidean/injective` (Task 1 Step 4 says what to check and what not
to loosen); whether the `OK` fixture in the frontend tests is genuinely
advice-free (Task 5 Step 4); the `k_local` window versus reference size if the
cry-wolf test is borderline (Task 3 Step 4); and the real limb numbers, which
Task 6 records rather than assumes — the spec's own caveat is that its values
came from a prototype whose preprocessing differs from the user's saved runs.
