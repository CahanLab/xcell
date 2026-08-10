# Injective Assignment Without the Whole Similarity Matrix — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Let `injective` run on datasets it currently refuses, by solving the assignment over each cell's best few candidate spots instead of the entire query × reference similarity matrix.

**Architecture:** `project_knn`'s chunk loop already reduces each similarity block to a top-k; the injective path will keep a wider top-`c` per row instead of copying every block into a full matrix. A greedy injective seed is added to the candidate graph so a full matching provably exists, and `scipy.sparse.csgraph.min_weight_full_bipartite_matching` solves it. Problems small enough to afford the dense solver keep it and stay exactly optimal; the result says which path ran.

**Tech Stack:** Python 3 / numpy / scipy (`optimize.linear_sum_assignment`, `sparse.csgraph.min_weight_full_bipartite_matching`); FastAPI; React 18 + TypeScript; pytest.

**Spec:** `docs/superpowers/specs/2026-08-10-injective-candidate-assignment-design.md`

## Global Constraints

- Branch `feat/injective-candidates` off `main` (already created); merge back with `--no-ff`.
- TDD: write the failing test, run it, watch it fail, then implement.
- Backend tests from `backend/`: `pixi run -e dev pytest`. Frontend: `cd frontend && npx tsc --noEmit && npm test`.
- `backend/xcell/localize.py` is **pure**: arrays in, arrays and plain dicts out. No AnnData, no adaptor import.
- Heavy imports (`scipy.optimize`, `scipy.sparse`) stay **function-local**.
- Everything crossing the API must be JSON-serializable: no `inf`, no `NaN`.
- `backend/tests/` has **no `conftest.py`** — each file is self-contained, seeded `np.random.default_rng(0)`.
- Comments explain *why*, especially where a threshold was chosen.
- **Verify in the browser, not only in pytest.** The measurements this plan is built on came from a prototype; the shipped path has to reproduce them on the real limb pair.

**The claim this must not overstate.** The candidate path is near-optimal, not
optimal: measured on the limb pair at `c=128` the objective is 0.18% below the
dense optimum and 87% of cells receive the identical spot. Every user-visible
surface says which path ran. Nothing anywhere may call the candidate result
"optimal".

**Two mistakes that look like a worse algorithm rather than a bug**, both
measured while designing this:

1. Building the CSR with duplicate `(row, col)` pairs **sums** them, doubling
   the cost of every seed edge that is also a candidate, so the solver avoids
   exactly the edges it should prefer. Cost: 12% of the objective at `c=128`.
2. A stored cost of exactly zero reads as "no edge" to
   `min_weight_full_bipartite_matching`. Every kept edge carries `+1e-9`.

## File Structure

| file | responsibility |
|---|---|
| `backend/xcell/localize.py` (modify) | `candidate_assignment`, `_feasible_seed`, `_chunk_rows`, the dense/candidate switch in `project_knn`, two new `Projection` fields |
| `backend/tests/test_localize.py` (modify) | the estimator's behaviour, including the adversarial clones fixture |
| `backend/xcell/adaptor.py` (modify) | `prepare_localize` reports which path ran |
| `backend/tests/test_localize_adaptor.py` (modify) | that report reaches the run result |
| `frontend/src/components/LocalizeModal.tsx` (modify) | says so after a run |

**Module API**, fixed here so every task agrees:

```python
_DENSE_MAX_ENTRIES  = 40_000_000     # dense path ceiling (~480 MB peak)
_CANDIDATES         = 128            # candidates per cell
_MAX_CANDIDATE_EDGES = 50_000_000    # candidate graph ceiling
_MAX_BLOCK_ENTRIES  = 20_000_000     # transient chunk block ceiling

def injective_feasibility(n_query: int, n_ref: int) -> str | None
def injective_assignment(similarity: np.ndarray) -> np.ndarray            # exists, unchanged
def candidate_assignment(cand_idx, cand_sim, n_ref: int) -> np.ndarray    # new
def _feasible_seed(cand_idx: np.ndarray, best_sim: np.ndarray, n_ref: int) -> np.ndarray
def _chunk_rows(n_ref: int) -> int
```

`Projection` gains `assignment: str | None = None` (`'exact'` / `'candidate'`)
and `n_candidates: int | None = None`.

---

### Task 1: `candidate_assignment` — the solver that does not hold the matrix

**Files:**
- Modify: `backend/xcell/localize.py`
- Modify: `backend/tests/test_localize.py`

**Interfaces:**
- Consumes: `injective_feasibility`, `injective_assignment` (both exist).
- Produces:
  - `_feasible_seed(cand_idx, best_sim, n_ref) -> np.ndarray` — `(n_query,)` distinct column per row.
  - `candidate_assignment(cand_idx, cand_sim, n_ref) -> np.ndarray` — `(n_query,)` distinct column per row, `int64`.
  - `injective_feasibility` re-scoped: it now bounds the **candidate graph**, not the dense matrix.

- [ ] **Step 1: Write the failing tests**

Append to `backend/tests/test_localize.py`, and extend the import at the top of
the file (line 16) so it reads:

```python
from xcell.localize import (
    AGGREGATIONS,
    METRICS,
    TRANSFORMS,
    candidate_assignment,
    cross_validate_localization,
    evaluate_localization,
    injective_assignment,
    injective_feasibility,
    project_knn,
    transform_expression,
)
```

Then append:

```python
# --- assignment without the whole matrix ----------------------------------

def _candidates(S, c):
    """Each row's top c, sorted — what project_knn's chunk loop will keep."""
    c = min(c, S.shape[1])
    part = np.argpartition(-S, c - 1, axis=1)[:, :c]
    rows = np.arange(S.shape[0])[:, None]
    vals = S[rows, part]
    order = np.argsort(-vals, axis=1)
    return part[rows, order], vals[rows, order]


def test_candidate_assignment_gives_every_cell_a_distinct_spot():
    S = np.random.default_rng(0).random((40, 60))
    idx, sim = _candidates(S, 16)
    out = candidate_assignment(idx, sim, 60)
    assert out.shape == (40,)
    assert len(set(out.tolist())) == 40


def test_candidate_assignment_matches_the_exact_answer_when_it_can_see_everything():
    """With every column a candidate the two solvers are the same problem, so
    any disagreement is a bug in the sparse construction — most likely costs
    doubled by duplicate edges."""
    S = np.random.default_rng(1).random((25, 40))
    idx, sim = _candidates(S, 40)
    exact = injective_assignment(S)
    approx = candidate_assignment(idx, sim, 40)
    assert float(S[np.arange(25), approx].sum()) == pytest.approx(
        float(S[np.arange(25), exact].sum()), abs=1e-9)


def test_candidate_assignment_beats_taking_each_rows_best():
    """The same 2x2 the exact solver is held to: both rows prefer column 0."""
    S = np.array([[0.9, 0.8], [1.0, 0.1]])
    idx, sim = _candidates(S, 2)
    assert list(candidate_assignment(idx, sim, 2)) == [1, 0]


def test_candidate_assignment_stays_solvable_when_every_cell_wants_the_same_spots():
    """The case that makes a plain top-c graph unsolvable — and it is not
    hypothetical: on the real limb pair no full matching existed for any c up to
    64, because many cells rank the same well-recovered spots highest. That is
    the pile-up injective exists to remove. The greedy seed is what makes the
    matching exist by construction rather than by luck.
    """
    rng = np.random.default_rng(2)
    nq, nr = 200, 2000
    base = rng.normal(size=20)
    Q = base + 0.01 * rng.normal(size=(nq, 20))
    R = rng.normal(size=(nr, 20))
    R[:5] = base + 0.005 * rng.normal(size=(5, 20))   # only 5 spots anyone wants
    Q /= np.linalg.norm(Q, axis=1, keepdims=True)
    R /= np.linalg.norm(R, axis=1, keepdims=True)
    S = (Q @ R.T).astype(np.float32)

    idx, sim = _candidates(S, 8)          # far too few for a matching to exist
    out = candidate_assignment(idx, sim, nr)
    assert len(set(out.tolist())) == nq


def test_the_seed_alone_is_already_a_valid_assignment():
    """It has to be: it is what makes the matching a theorem rather than a hope."""
    from xcell.localize import _feasible_seed
    S = np.random.default_rng(3).random((30, 45))
    idx, sim = _candidates(S, 4)
    seed = _feasible_seed(idx, sim.max(axis=1), 45)
    assert len(set(seed.tolist())) == 30
    assert seed.min() >= 0


def test_candidate_assignment_needs_a_spot_for_every_cell():
    S = np.random.default_rng(4).random((9, 5))
    idx, sim = _candidates(S, 5)
    with pytest.raises(ValueError, match='at least one reference spot'):
        candidate_assignment(idx, sim, 5)


def test_the_size_limit_now_bounds_the_candidate_graph():
    """The ceiling moved from the dense matrix to the candidate graph, so what
    used to be refused now runs: 200,000 cells against 200,000 spots is 4e10
    dense entries and 2.6e7 candidate edges."""
    assert injective_feasibility(200_000, 200_000) is None
    assert 'memory' in (injective_feasibility(1_000_000, 1_000_000) or '')
```

Then **replace** the existing test that asserted the old ceiling. Find
`test_a_similarity_matrix_too_large_to_hold_is_refused_by_size` and delete it —
`test_the_size_limit_now_bounds_the_candidate_graph` above is its replacement,
and leaving both means asserting two different ceilings.

- [ ] **Step 2: Run them and watch them fail**

```bash
cd backend && pixi run -e dev pytest tests/test_localize.py -q
```

Expected: `ImportError: cannot import name 'candidate_assignment'`.

- [ ] **Step 3: Implement**

In `backend/xcell/localize.py`, replace the `_MAX_INJECTIVE_ENTRIES` constant
(around line 47) with:

```python
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

# The candidate graph's own ceiling: n_query x candidates edges, each an int32
# index plus a float64 cost. 5e7 edges is ~600 MB, so the refusal now fires
# around 400,000 cells rather than around 10,000.
_MAX_CANDIDATE_EDGES = 50_000_000
```

Replace `injective_feasibility` with:

```python
def injective_feasibility(n_query: int, n_ref: int) -> str | None:
    """Why injective assignment cannot run at these sizes, or None if it can.

    One message, three callers: the estimators themselves, ``project_knn`` —
    which checks before doing a minute of matching work it would then throw
    away — and the adaptor, which turns it into a 400 rather than a background
    task that dies later.
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
```

Add both new functions immediately after `injective_assignment`:

```python
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
        free = np.flatnonzero(~used)[:len(stranded)]
        out[stranded] = free
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
```

- [ ] **Step 4: Run the tests**

```bash
cd backend && pixi run -e dev pytest tests/test_localize.py -q
```

Expected: all pass. If
`test_candidate_assignment_matches_the_exact_answer_when_it_can_see_everything`
fails by a small amount, the duplicate-edge dedup is wrong — check that
`np.unique` is being applied to the combined `row * n_ref + col` key and that
`sims` is indexed by the same `keep`. Do **not** loosen the tolerance; that test
exists to catch precisely this.

- [ ] **Step 5: Commit**

```bash
git add backend/xcell/localize.py backend/tests/test_localize.py
git commit -m "feat(localize): assignment over candidate spots instead of the whole matrix"
```

---

### Task 2: Wire it into `project_knn`, and say which path ran

**Files:**
- Modify: `backend/xcell/localize.py` — `Projection`, `project_knn`
- Modify: `backend/tests/test_localize.py`

**Interfaces:**
- Consumes: `candidate_assignment`, `_DENSE_MAX_ENTRIES`, `_CANDIDATES` (Task 1).
- Produces: `Projection.assignment` (`'exact'` / `'candidate'` / `None`) and `Projection.n_candidates` (`int | None`).

- [ ] **Step 1: Write the failing tests**

Append to `backend/tests/test_localize.py`:

```python
def test_a_small_problem_still_gets_the_exact_assignment():
    """Existing runs must not change answers: the dense solver is affordable
    here and provably optimal, so it stays."""
    ref_expr, ref_coords, q_expr, _ = _matched()
    p = project_knn(q_expr, ref_expr, ref_coords, k=10, aggregation='injective')
    assert p.assignment == 'exact'
    assert p.n_candidates is None


def test_a_large_problem_takes_the_candidate_path(monkeypatch):
    """Forced by lowering the ceiling rather than by building a real matrix that
    would not fit — the point is the branch, not the arithmetic."""
    import xcell.localize as lz
    monkeypatch.setattr(lz, '_DENSE_MAX_ENTRIES', 10)
    ref_expr, ref_coords, q_expr, q_coords = _matched()
    p = project_knn(q_expr, ref_expr, ref_coords, k=10, aggregation='injective')
    assert p.assignment == 'candidate'
    assert p.n_candidates == min(lz._CANDIDATES, ref_expr.shape[0])
    assert len(np.unique(p.coords, axis=0)) == len(q_expr)
    centroid_error = float(np.median(
        np.linalg.norm(q_coords - ref_coords.mean(axis=0), axis=1)
    ))
    assert _median_error(p.coords, q_coords) < centroid_error / 2


def test_the_candidate_path_still_spreads_what_best_match_piles_up(monkeypatch):
    import xcell.localize as lz
    monkeypatch.setattr(lz, '_DENSE_MAX_ENTRIES', 10)
    ref_expr, ref_coords, q_expr = _clones()
    p = project_knn(q_expr, ref_expr, ref_coords, k=15,
                    transform='none', aggregation='injective')
    assert len(np.unique(p.coords, axis=0)) == 60


def test_the_other_aggregations_report_no_assignment_path():
    """Only injective assigns; nothing else should claim to."""
    ref_expr, ref_coords, q_expr, _ = _matched()
    p = project_knn(q_expr, ref_expr, ref_coords, k=10, aggregation='weighted_mean')
    assert p.assignment is None and p.n_candidates is None


def test_both_paths_agree_on_this_problem(monkeypatch):
    """Not guaranteed in general — the candidate path is near-optimal — but on a
    problem this size with 128 candidates out of 400 spots it should land on the
    same answer, and a large disagreement means the wiring is wrong rather than
    the approximation biting."""
    import xcell.localize as lz
    ref_expr, ref_coords, q_expr, _ = _matched()
    exact = project_knn(q_expr, ref_expr, ref_coords, k=10, aggregation='injective')
    monkeypatch.setattr(lz, '_DENSE_MAX_ENTRIES', 10)
    approx = project_knn(q_expr, ref_expr, ref_coords, k=10, aggregation='injective')
    agree = float(np.mean(np.all(exact.coords == approx.coords, axis=1)))
    assert agree > 0.9, agree
```

- [ ] **Step 2: Run them and watch them fail**

```bash
cd backend && pixi run -e dev pytest tests/test_localize.py -q -k "exact or candidate_path or both_paths or no_assignment_path"
```

Expected: `AttributeError: 'Projection' object has no attribute 'assignment'`.

- [ ] **Step 3: Add the two fields to `Projection`**

In `backend/xcell/localize.py`, extend the dataclass (around line 54). Defaults
keep every existing construction valid:

```python
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
```

- [ ] **Step 4: Replace the full-matrix path in `project_knn`**

Replace the block that currently reads:

```python
    n_query = q.shape[0]
    idx = np.empty((n_query, k), dtype=np.int64)
    sims = np.empty((n_query, k), dtype=float)

    # The chunked path keeps only the top k, so injective assignment — which is
    # a global decision — needs the blocks kept as they are computed. Storing
    # them costs memory, not a second pass.
    full = np.empty((n_query, r.shape[0]), dtype=np.float32) if needs_full else None

    for start in range(0, n_query, _CHUNK):
        stop = min(start + _CHUNK, n_query)
        block = _similarity_block(q[start:stop], r, metric)
        if full is not None:
            full[start:stop] = block
        # argpartition for the top-k, then order those k properly.
        part = np.argpartition(-block, k - 1, axis=1)[:, :k]
        rows = np.arange(stop - start)[:, None]
        part_sims = block[rows, part]
        order = np.argsort(-part_sims, axis=1)
        idx[start:stop] = part[rows, order]
        sims[start:stop] = part_sims[rows, order]
        if progress is not None:
            progress(stop / n_query, f'Matched {stop:,} of {n_query:,} cells')
```

with:

```python
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

    for start in range(0, n_query, _CHUNK):
        stop = min(start + _CHUNK, n_query)
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
```

- [ ] **Step 5: Branch the assignment itself**

Replace the block that currently reads:

```python
    if needs_full:
        if progress is not None:
            progress(1.0, f'Assigning {n_query:,} cells to distinct spots')
        assigned = injective_assignment(full)
```

with:

```python
    assignment_kind: str | None = None
    n_candidates_used: int | None = None
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
```

and extend the `Projection(...)` construction at the end of the function with
the two new fields:

```python
    return Projection(
        coords=coords,
        confidence=confidence,
        similarity=similarity,
        neighbor_indices=idx,
        sections=sections_out,
        n_unplaced=n_unplaced,
        assignment=assignment_kind,
        n_candidates=n_candidates_used,
    )
```

- [ ] **Step 6: Run the whole localize file, then the suite**

```bash
cd backend && pixi run -e dev pytest tests/test_localize.py -q
cd backend && pixi run -e dev pytest -q
```

Expected: all pass, no regressions. `test_every_metric_and_aggregation_produces_a_usable_map`
still covers `injective` on the dense path.

- [ ] **Step 7: Commit**

```bash
git add backend/xcell/localize.py backend/tests/test_localize.py
git commit -m "feat(localize): take the candidate path when the matrix will not fit"
```

---

### Task 3: The other memory wall — an adaptive chunk

`_CHUNK` is a fixed 2,048 rows, so the transient similarity block is
`2048 × n_ref`: 820 MB against a 100,000-spot reference. That blocks the
*reference* direction for **every** aggregation, not only injective, and it
would still block it after Task 2.

**Files:**
- Modify: `backend/xcell/localize.py`
- Modify: `backend/tests/test_localize.py`

**Interfaces:**
- Consumes: nothing from earlier tasks.
- Produces: `_chunk_rows(n_ref: int) -> int`.

- [ ] **Step 1: Write the failing tests**

Append to `backend/tests/test_localize.py`:

```python
def test_the_chunk_shrinks_as_the_reference_grows():
    """A fixed 2,048-row chunk is 820 MB of transient block against a
    100,000-spot reference, which blocks every aggregation and not just the
    assigning one."""
    from xcell.localize import _chunk_rows
    assert _chunk_rows(1_000) == 2048           # small reference: unchanged
    assert _chunk_rows(100_000) == 200
    assert _chunk_rows(50_000_000) == 1         # never zero


def test_chunking_does_not_change_the_answer(monkeypatch):
    """Whatever the chunk, the map is the same — otherwise the loop is carrying
    state it should not."""
    import xcell.localize as lz
    ref_expr, ref_coords, q_expr, _ = _matched(n_query=97)
    big = project_knn(q_expr, ref_expr, ref_coords, k=10)
    monkeypatch.setattr(lz, '_MAX_BLOCK_ENTRIES', 400)     # forces a tiny chunk
    small = project_knn(q_expr, ref_expr, ref_coords, k=10)
    assert np.allclose(big.coords, small.coords)
    assert np.allclose(big.confidence, small.confidence)
```

- [ ] **Step 2: Run them and watch them fail**

```bash
cd backend && pixi run -e dev pytest tests/test_localize.py -q -k chunk
```

Expected: `ImportError: cannot import name '_chunk_rows'`.

- [ ] **Step 3: Implement**

In `backend/xcell/localize.py`, replace the `_CHUNK` constant and its comment:

```python
# Query cells per chunk. Keeps the similarity block bounded regardless of how
# many cells are being mapped, and gives progress something to report.
_CHUNK = 2048

# ...and bounded regardless of how many reference spots there are, which the row
# count alone does not do: 2,048 rows against a 100,000-spot reference is a
# 820 MB transient block, for every aggregation rather than only the assigning
# one. 2e7 float32 entries is 80 MB.
_MAX_BLOCK_ENTRIES = 20_000_000


def _chunk_rows(n_ref: int) -> int:
    """How many query cells to score at once against this many spots."""
    return max(1, min(_CHUNK, _MAX_BLOCK_ENTRIES // max(int(n_ref), 1)))
```

In `project_knn`, replace `for start in range(0, n_query, _CHUNK):` with:

```python
    chunk = _chunk_rows(n_ref)
    for start in range(0, n_query, chunk):
        stop = min(start + chunk, n_query)
```

(The `stop = ...` line already exists directly below; replace it rather than
duplicating it.)

- [ ] **Step 4: Run the tests**

```bash
cd backend && pixi run -e dev pytest tests/test_localize.py -q
```

Expected: all pass.

- [ ] **Step 5: Commit**

```bash
git add backend/xcell/localize.py backend/tests/test_localize.py
git commit -m "perf(localize): bound the similarity block by reference size too"
```

---

### Task 4: Report it through the API and the modal

**Files:**
- Modify: `backend/xcell/adaptor.py` — `prepare_localize`'s `apply_fn`
- Modify: `backend/tests/test_localize_adaptor.py`
- Modify: `frontend/src/components/LocalizeModal.tsx`

**Interfaces:**
- Consumes: `Projection.assignment`, `Projection.n_candidates` (Task 2).
- Produces: `assignment` and `n_candidates` keys on the localize run result.

- [ ] **Step 1: Write the failing test**

Append to `backend/tests/test_localize_adaptor.py`:

```python
def test_the_run_says_how_the_assignment_was_solved():
    """A near-optimal answer must never be readable as an exact one."""
    ref, query, _ = _pair()
    out = _run(query, ref.spatial_reference_bundle(), k=10, aggregation='injective')
    assert out['assignment'] == 'exact'
    assert out['n_candidates'] is None


def test_a_non_assigning_aggregation_claims_nothing():
    ref, query, _ = _pair()
    out = _run(query, ref.spatial_reference_bundle(), k=10, aggregation='weighted_mean')
    assert out['assignment'] is None


def test_the_candidate_path_is_reported_as_such(monkeypatch):
    import xcell.localize as lz
    monkeypatch.setattr(lz, '_DENSE_MAX_ENTRIES', 10)
    ref, query, _ = _pair()
    out = _run(query, ref.spatial_reference_bundle(), k=10, aggregation='injective')
    assert out['assignment'] == 'candidate'
    assert out['n_candidates'] > 0
    json.dumps(out, allow_nan=False)
```

- [ ] **Step 2: Run it and watch it fail**

```bash
cd backend && pixi run -e dev pytest tests/test_localize_adaptor.py -q -k assignment
```

Expected: `KeyError: 'assignment'`.

- [ ] **Step 3: Add the two keys to the result**

In `backend/xcell/adaptor.py`, in `prepare_localize`'s `apply_fn`, extend the
`out` dict — put them next to `n_reference_cells`:

```python
                'n_reference_cells': snap['n_reference_cells'],
                # Which solver produced the assignment. 'candidate' is
                # near-optimal, and saying so is the difference between a
                # documented approximation and a silent one.
                'assignment': projection.assignment,
                'n_candidates': projection.n_candidates,
```

- [ ] **Step 4: Run the backend suite**

```bash
cd backend && pixi run -e dev pytest -q
```

Expected: all pass.

- [ ] **Step 5: Show it in the modal**

In `frontend/src/components/LocalizeModal.tsx`, add the two fields to the
`LocalizeResult` interface (around line 64):

```tsx
interface LocalizeResult {
  embedding_name: string
  n_cells: number
  n_unplaced: number
  n_shared_genes: number
  n_missing_genes: number
  n_reference_cells: number
  section_col: string | null
  assignment: string | null
  n_candidates: number | null
  confidence_distribution: {
    median: number
    q25: number
    q75: number
    'n_below_0.2': number
    'n_below_0.5': number
  }
}
```

and, inside the result box (after the "Select … in the embedding picker" line),
add:

```tsx
              {result.assignment === 'candidate' && (
                <div style={{ fontSize: 11, color: '#f0c987', marginTop: 5 }}>
                  Too large to assign exactly, so each cell chose among its{' '}
                  {result.n_candidates} best-matching spots rather than all{' '}
                  {result.n_reference_cells.toLocaleString()}. Every cell still
                  has its own spot; the total similarity is within a fraction of
                  a percent of the exact answer, but it is <b>near-optimal, not
                  optimal</b>.
                </div>
              )}
```

- [ ] **Step 6: Typecheck and test the frontend**

```bash
cd frontend && npx tsc --noEmit && npm test
```

Expected: clean typecheck, all tests pass.

- [ ] **Step 7: Commit**

```bash
git add backend/xcell/adaptor.py backend/tests/test_localize_adaptor.py \
        frontend/src/components/LocalizeModal.tsx
git commit -m "feat(localize): report which solver produced the assignment"
```

---

### Task 5: Verify on the limb pair, document, merge

The measurements this plan is built on came from a prototype. The shipped path
has to reproduce them, and the dense path has to still produce exactly what it
produced yesterday.

**Files:**
- Modify: `README.md`, `CHANGELOG.md`

- [ ] **Step 1: Start both servers**

```bash
pixi run backend      # :8000
pixi run dev          # :5173
```

Stop them later with `pkill -f "uvicorn xcell.main:app"; pkill -f vite`. Editing
a backend file restarts the server and drops the loaded datasets, so batch any
fixes.

- [ ] **Step 2: Load the limb pair**

Reference `E11.5_best_102424.h5ad` (9,773 spots) at
`~/CahanLab Dropbox/Patrick Cahan/PC/PC/projects/SJD/data/spatial/xcell/h5ad_empty_removed/best_per_stage/`,
query `GSM4227224_E11_112125.h5ad` (2,683 cells) at
`~/CahanLab Dropbox/Patrick Cahan/PC/PC/projects/SJD/data/scRNAseq/Kelly_Guilak_2020/`.

`POST /api/load` takes the slot as a **body** field
(`{"file_path": ..., "slot": "secondary"}`); `?dataset=` there is ignored.

- [ ] **Step 3: Confirm the dense path is unchanged on this pair**

2,683 × 9,773 = 2.6e7 entries, below `_DENSE_MAX_ENTRIES` (4e7), so this pair
keeps the exact solver and must produce exactly what it produced before this
change. Run injective and check the reported fields:

```bash
curl -s -X POST "localhost:8000/api/localize/prepare?dataset=<query slot>" \
  -H 'Content-Type: application/json' \
  -d '{"reference":"<ref slot>","k":50,"aggregation":"injective",
       "key_added":"X_spatial_pred_inj"}'
# poll /api/tasks/<id> until completed, then read the result
```

Expected: `assignment` is `"exact"`, `n_candidates` is `null`, and the run
completes in a few seconds — unchanged from before this work.

To exercise the *candidate* path on real data, re-run with
`_DENSE_MAX_ENTRIES` temporarily lowered (edit the constant, which restarts the
backend and drops the datasets, so do this last and reload). Expect
`assignment: "candidate"`, `n_candidates: 128`.

- [ ] **Step 4: Confirm the map is what it was**

Score it against the reference and compare with the numbers this design was
built on — `X_spatial_pred_inj` must still use **2,683 distinct spots of 2,683**
with hull area ~1.00, on either path:

```bash
curl -s -X POST "localhost:8000/api/localize/evaluate_map?dataset=<query slot>" \
  -H 'Content-Type: application/json' \
  -d '{"reference":"<ref slot>","embeddings":["X_spatial_pred_inj"],
       "gene_sets":{"skin":["Perp","Krt18","Krt14","Krt5","Epcam","Cdh1"]}}' \
  | python3 -m json.tool
```

Expected: `n_distinct` 2,683 of 2,683, `area_ratio` ~1.0, skin pattern
correlation positive (the prototype measured +0.285 at c=128; anything in
0.25–0.40 is the same answer given how much that metric moves).

If `n_distinct` is below 2,683 the assignment is not injective and something is
wrong — that property is the whole point and holds at every candidate count
measured.

- [ ] **Step 5: Check the modal says which path ran**

Open Localize, run injective on the pair, and confirm the result box carries the
near-optimal notice with the candidate count. Then stop the servers.

- [ ] **Step 6: Write the docs**

`README.md` — in the Localize section, beside the existing `injective`
paragraph, note that on a large pair it assigns over each cell's best candidate
spots rather than every spot, that this is near-optimal rather than optimal, and
that the run says which happened.

`CHANGELOG.md` — an entry under **Fixed** or **Changed** in `[Unreleased]`
covering: the old cap and why it was really 1.2 GB rather than 400 MB (scipy's
float64 copy); the candidate restriction and the measured 77× reduction; that
the map-quality metrics cannot separate it from exact; that plain top-k had no
full matching at any c ≤ 64 on real data and the greedy seed is what fixes that;
the two construction mistakes worth naming (summed duplicate edges, stored
zeros); and the adaptive chunk.

- [ ] **Step 7: Commit and merge**

```bash
git add README.md CHANGELOG.md
git commit -m "docs: injective assignment over candidate spots"
cd backend && pixi run -e dev pytest -q
cd ../frontend && npx tsc --noEmit && npm test
git checkout main
git merge --no-ff feat/injective-candidates
```

---

## Self-Review

**Spec coverage.** §1 keep candidates not the matrix → Task 2 Step 4 (the chunk
loop) with the solver in Task 1. §2 feasible seed, edge dedup, epsilon → Task 1
Step 3, with the dedup and zero-cost traps called out in Global Constraints and
tested by `test_candidate_assignment_matches_the_exact_answer_when_it_can_see_everything`.
§3 dense stays for problems that fit, and the run reports which → Task 2 (branch
plus `Projection` fields) and Task 4 (result dict and modal). §3's moved ceiling
→ Task 1's re-scoped `injective_feasibility` and
`test_the_size_limit_now_bounds_the_candidate_graph`. §4 adaptive chunk →
Task 3. Out-of-scope items (no UI dial for `c`, no fallback if the solver
raises) are respected: `_CANDIDATES` stays a module constant and no `try` wraps
`min_weight_full_bipartite_matching`.

**The ceiling was corrected while writing this plan.** At the 2e7 first
proposed, the limb pair (2.6e7 entries) would have fallen onto the candidate
path — a currently-exact, currently-working case made approximate, which is a
regression rather than a saving. `_DENSE_MAX_ENTRIES` is 4e7 so everything that
runs dense today still does, and the candidate path takes over exactly where the
alternative used to be a refusal.

**Placeholder scan.** No TBDs. Every code step carries its code, every test step
its test, and every constant its measured justification.

**Type consistency.** `candidate_assignment(cand_idx, cand_sim, n_ref)` is
spelled identically in Task 1's implementation, Task 1's tests and Task 2's call
site. `_feasible_seed(cand_idx, best_sim, n_ref)` likewise. `Projection`'s two
new fields (`assignment`, `n_candidates`) are produced in Task 2, consumed in
Task 4's adaptor dict under the same names, and typed as `string | null` /
`number | null` in the modal. The four constants (`_DENSE_MAX_ENTRIES`,
`_CANDIDATES`, `_MAX_CANDIDATE_EDGES`, `_MAX_BLOCK_ENTRIES`) are named the same
everywhere they are read or monkeypatched.

**Known soft spots**, each with an in-plan instruction: whether `np.unique`
keeps first occurrences as the dedup relies on (Task 1 Step 4 names the failure
and what to check); whether the candidate and dense paths agree closely enough
on the small fixture (Task 2's `test_both_paths_agree_on_this_problem` asserts
>90% rather than equality, because the method is near-optimal by construction);
and the exact skin-fidelity value on real data, which Task 5 Step 4 brackets
rather than pins, since that metric moved 0.285–0.365 across variants during
design.
