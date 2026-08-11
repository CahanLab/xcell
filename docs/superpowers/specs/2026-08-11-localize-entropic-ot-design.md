# Entropic optimal transport for Localize

Phase 3 of `2026-08-09-localize-calibration-and-transport-design.md`. Phases 1
(the Map quality harness) and 2 (injective assignment) have shipped; this is the
piece they were built to make decidable.

## The problem this estimator exists for

Every aggregation Localize currently offers sits at one end of a single
trade-off, measured on the E11.5 limb pair (2,683 cells against 9,773 spots):

| aggregation | hull area | epidermis pattern | axes |
|---|---|---|---|
| `weighted_mean` (default) | 0.15 | **−0.11 — inverted** | intact |
| `best_match` / `injective` | 1.00 | +0.29 | gone |

Averaging keeps the gradients and collapses the map; taking one location fills
the tissue and destroys the gradients. Neither is a bug to be fixed — they are
the two ends of the same frontier, and Phase 1 exists so that this can be stated
as a measurement rather than an impression.

**Optimal transport is the only estimator that moves along that frontier rather
than sitting at an end of it.** A marginal constraint on the reference side
forces the predicted population to occupy the tissue, while the entropy term
still lets each cell hedge across several spots, so averaging's noise
cancellation survives. ε is the dial between the two.

## What ε actually competes against

The parent spec quotes a frontier at ε = 0.02–0.05. Those numbers came from a
prototype using `normalize_total`+`log1p` on all shared genes. Measured on the
real pair (`E11.5_best_102424.h5ad` against `GSM4227224_E11_112125.h5ad`, 22,016
shared genes, 600 × 2,500 subsample), the cost distribution depends heavily on
the transform and metric chosen:

| transform | metric | cost spread (σ) | what ε=0.05 means |
|---|---|---|---|
| `none` | `cosine` — the settings actually used | 0.0827 | 0.60 spreads |
| `none` | `correlation` | 0.0779 | 0.64 spreads |
| `zscore` | `correlation` — the prototype's | 0.0463 | 1.08 spreads |
| `zscore` | `cosine` | 0.0699 | 0.72 spreads |
| `rank` | `cosine` | 0.0239 | 2.09 spreads |

Sinkhorn forms K = exp(−C/ε), so what ε trades against is the **spread** of C,
not its absolute scale. The useful range of ε in the parent spec spans 2.5×
(0.02 → 0.05, taking hull area 0.59 → 0.22). The transform/metric choice moves
the effective ε across **3.5×** (0.60 → 2.09 spreads). *The preprocessing choice
moves the dial further than the entire dial travels*, so a shipped ε default is
meaningless unless it is expressed relative to the cost spread.

This also corrects an assumption worth recording: the settings in use produce a
cost band **1.8× wider** than the prototype's, not narrower. The spec's ε would
under-regularize here rather than collapse the map. The direction of the error
was wrong; the need to normalize was not.

## Approach

### 1. The estimator

Solve for a coupling **P** (n_query × n_ref) minimizing Σ Pᵢⱼ·Cᵢⱼ − ε·H(P)
subject to a uniform row marginal (each cell fully placed, 1/n) and a uniform
column marginal (each spot receives 1/m).

**The column marginal is the whole point, and it carries an assumption.** It is
what forces the population onto the tissue instead of into its centroid. It also
asserts that the query's composition matches the tissue's — the same claim that
makes `injective` off by default, since dissociation biases which cell types
survive. OT holds it *softly*: entropy lets each cell hedge, so nothing is forced
one-to-one. The UI must state this rather than let it pass silently. Relaxing it
properly is unbalanced OT, which stays out of scope.

**Cost is normalized.** C = −similarity, shifted so its minimum is 0, then
divided by its standard deviation. The shift is free — entropic OT is provably
invariant to a constant added to C, because the factor exp(−a/ε) is absorbed into
the scaling vectors. The division is what makes ε dimensionless, so one default
survives `none`+`cosine` and `zscore`+`correlation` alike. **The normalizer is
computed on the matrix actually solved**, so the subsampled path cannot silently
shift the dial relative to the dense one.

**ε default: 0.5 spreads**, favouring area, since collapse is the failure that
motivated this work. In normalized units the parent spec's frontier becomes
≈0.43 spreads → area 0.59 and ≈1.08 spreads → area 0.22, so the useful band is
roughly 0.4–1.1. This number is a starting point to be re-derived by an ε sweep
through the Phase 1 harness during implementation — it is not yet a measurement
on this preprocessing, and the spec should not pretend otherwise.

**Prediction is the barycentric projection**, coordsᵢ = Σⱼ Pᵢⱼ·coordsⱼ,
row-normalized. Taking the row argmax instead returns to the single-location
regime and loses the axes (measured: 0.98 area, ~0 correlation), so barycentric
is the one that ships.

**Multi-section references reuse the existing treatment.** `project_knn` already
assigns a weighted-majority section and zeroes the other sections' weights,
because averaging coordinates across sections lands in the gap between cuts.
Barycentric projection has exactly that failure mode — a row spread over two
sections averages into empty space — so P gets the same handling: take the row's
dominant section by transported mass, renormalize within it, then project.

### 2. Scale and numerics

**Log-domain always.** The naive kernel form underflows at ε≤0.05 on this data.
The mechanism is not the kernel entries themselves but the scaling vectors, which
grow and shrink without bound over iterations. A single stable code path is worth
more than the 2–3× per-iteration saving of a form that fails at exactly the
settings the estimator is most interesting at.

**Dense while affordable, spatially-stratified subsample above it.** This mirrors
the pattern Phase 2 established and proved: exact where exact is cheap, an
approximation above that, and *report which one ran*. The reference subsample is
stratified over the tissue rather than uniform, so coverage of the spatial extent
— the thing the column marginal is enforcing — survives the reduction:
bin the reference coordinates on a regular grid and draw from each occupied bin
in proportion to its occupancy, with at least one spot per occupied bin, so no
region of the tissue can drop out of the target marginal entirely.

The dense ceiling is set by **what already works**, as `_DENSE_MAX_ENTRIES` was:
the limb pair is 26.2M entries and must continue to run dense. A lower ceiling
would push a currently-affordable case onto the approximate path, which is a
regression dressed as a memory saving.

**Bounded iterations with an honest convergence report.** Default budget 200
iterations, early stop when `marginal_error` — the maximum absolute deviation of
a column sum from its target 1/m, expressed as a fraction of that target — falls
below 1e-4. The run reports iterations used and the final marginal error,
because a Sinkhorn that ran out of budget has *not* satisfied the constraint that
justifies the method, and that must be visible rather than inferred. The parent
spec measured the full reference at 120 iterations exceeding two minutes, so this
runs as a cancellable background task with per-iteration progress.

**Rejected: sparse candidate support.** Restricting P to each cell's top-128
spots would reuse the `cand_idx`/`cand_sim` arrays Phase 2 already builds and
would be far faster. Phase 2 is also why it is rejected: a plain top-*c*
candidate graph had **no full matching at all** for every *c* ≤ 64 on this exact
pair, because many cells rank the same well-recovered spots highest. The
reference marginal is a strictly harder constraint than a matching — where the
candidate sets fail to cover the tissue it is infeasible, and Sinkhorn diverges
rather than failing cleanly. Making feasibility a theorem, as Phase 2 had to,
is a project of its own.

### 3. What it returns

`Projection` gains four fields, following the precedent of `assignment` /
`n_candidates`: `transport` (`'full' | 'subsampled'`), `n_ref_used`,
`sinkhorn_iterations`, `marginal_error`.

**Confidence comes from the posterior, in the same units as before.** Each row of
P is a spatial posterior — the honest output for a cell that genuinely could be
in several places. Confidence becomes the posterior's weighted RMS spread about
the prediction, over the tissue radius: *the same formula the k-NN path already
uses*, evaluated over P instead of over the k neighbours. Row entropy was the
other candidate and is rejected: it is not in spatial units, so it would not be
comparable with the confidence reported by every other aggregation, and the
Map quality panel exists to compare runs.

`similarity` is unchanged — it answers a different question (does this cell
resemble the reference at all) and stays orthogonal.

### 4. Surface

`'transport'` joins `AGGREGATIONS`. New parameters `epsilon` (default 0.5) and
`max_iterations` (default 200) flow through `prepare_localize` and the route,
validated synchronously so a bad ε is a 400 rather than a background task that
dies a minute in.

`src/lib/localizeAdvice.ts` gains guidance for the new aggregation: which
direction ε moves the map, the composition assumption behind the column marginal,
and a warning when a run hit its iteration budget without converging. The Map
quality panel needs no change — it already ranks whatever embeddings exist, which
is exactly how this estimator is meant to be judged.

### 5. Testing

`backend/tests/test_localize_transport.py`, self-contained, seeded, `csr_matrix`
+ `float32` per the house pattern. The properties worth pinning:

1. **Marginals are satisfied** to tolerance, both row and column.
2. **Scale invariance** — the same P for C and 2C at the same ε. This is the
   entire justification for normalizing the cost, so it is the test that would
   fail if the normalization were dropped.
3. **Shift invariance** — the same P for C and C + a.
4. **ε monotonicity** — larger ε gives more diffuse rows and predictions closer
   to the tissue centroid; smaller ε fills more of it.
5. **Log-domain survives where naive underflows** — agreement with a float64
   naive implementation at moderate ε, and a finite, marginal-satisfying answer
   at an ε where the naive form does not produce one.
6. **Barycentric projection** against a synthetic case with a known answer.
7. **Sections** — a row split across two sections must not land in the gap.
8. **The subsampled path** reports `transport='subsampled'`, and its stratified
   subsample covers the tissue extent.
9. **Stats crossing the API are JSON-serializable** — no `NaN`, no `inf`.

## Success criteria

Explicitly **not** a pass/fail gate. OT ships as an option and the Phase 1
harness ranks it per dataset, because the right estimator may differ by tissue
and by preprocessing — which the cost-spread measurement above is itself evidence
for. The obligation is that it is correct, measurable, and comparable, not that
it wins.

## Out of scope

Unbalanced OT for composition mismatch. Gromov-Wasserstein. Reference section
re-orientation — sections are assumed to arrive correctly oriented, as they are
today by hand. Changing the default aggregation away from `weighted_mean`: the
parent spec's phase 4, and it needs the ε sweep's results first.
