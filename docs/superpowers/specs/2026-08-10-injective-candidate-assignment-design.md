# Injective assignment without the whole similarity matrix

**Date:** 2026-08-10
**Status:** Approved, ready to plan

## Problem

`injective` is the aggregation users like, and it is the one that refuses to
run. It solves for a set of assignments subject to distinct reference spots, and
`scipy.optimize.linear_sum_assignment` needs the whole query × reference
similarity matrix in memory at once — which the chunked top-k path that feeds
every other aggregation never materializes. `localize.py` therefore caps it at
`_MAX_INJECTIVE_ENTRIES = 1e8` and refuses anything larger.

The cap blocks ordinary datasets. A 50,000-cell query against a 20,000-spot
reference is 1e9 entries; at 12 bytes each that is 12 GB.

**Twelve bytes, not four.** The current comment claims 1e8 float32 entries is
400 MB. Measured, scipy converts its input to float64 internally, so peak
residency is the float32 matrix *plus* a float64 copy:

| input dtype | peak RSS over baseline, 2,683 × 9,773 |
|---|---|
| float32 | **+210 MB** on top of the 105 MB array |
| float64 | +0 MB (no copy needed) |

So today's ceiling is really ~1.2 GB, and the limb pair costs 315 MB rather
than the 105 MB the code implies.

## The measurement that shapes the design

Everything below was measured on Patrick's E11.5 limb pair — reference
`E11.5_best_102424.h5ad` (9,773 spots), query `GSM4227224_E11_112125.h5ad`
(2,683 cells), 22,016 shared genes, `zscore` + `correlation`, i.e. the real
similarity matrix `project_knn` builds.

**The optimal assignment overwhelmingly uses each cell's few best candidates.**
Restricting the problem to each row's top `c` spots and solving the sparse
bipartite matching gives:

| method | memory | area | distinct spots | skin pattern | mean similarity |
|---|---|---|---|---|---|
| dense (exact) | **315 MB** | 1.00 | 2,683 / 2,683 | +0.298 | 0.1508 |
| candidates c=32 | 1.0 MB | 0.99 | 2,683 / 2,683 | +0.306 | 0.1354 |
| candidates c=64 | 2.1 MB | 1.00 | 2,683 / 2,683 | +0.365 | 0.1440 |
| candidates c=128 | **4.1 MB** | 0.99 | 2,683 / 2,683 | +0.285 | 0.1505 |
| candidates c=256 | 8.2 MB | 1.00 | 2,683 / 2,683 | +0.299 | 0.1508 |
| greedy `best_match` | — | 1.00 | 1,522 / 2,683 | +0.393 | 0.1627 |

Three facts, and they decide the design.

**Every candidate variant delivers what injective is for.** All 2,683 cells land
on distinct spots at every `c`, against 1,522 for greedy, and hull area stays at
1.00. The property is not a casualty of the approximation.

**The map-quality metrics cannot separate the approximation from the exact
answer.** Skin pattern fidelity wanders between 0.285 and 0.365 with no ordering
by objective — `c=64` scores *above* the exact solution — across variants whose
objective differs by 15%. Those differences are the metric's own noise. Mean
similarity, which is what the objective actually optimizes, converges
monotonically and reaches the exact value by `c=256`. At `c=128` the objective
is 0.18% below optimal and 87% of cells receive the identical spot.

**Feasibility has to be constructed, not hoped for.** Plain top-k had *no full
matching at all* on this data for every `c ≤ 64`:

| c | 4 | 8 | 16 | 32 | 64 | 128 |
|---|---|---|---|---|---|---|
| full matching exists? | no | no | no | no | no | yes |

The reason is the phenomenon injective exists to fix: many query cells rank the
same well-recovered spots highest, so the union of their candidates is smaller
than the number of cells and Hall's condition fails. On a synthetic matrix with
well-separated types, `c=16` already reaches the exact optimum — real data is
harder, and tuning `c` against synthetic geometry would have been misleading.

## Approach

### 1. Keep candidates, not the matrix

`project_knn` already computes `block = _similarity_block(q[chunk], r, metric)`
per chunk and reduces it to the top `k` for the confidence score. The injective
path currently copies every block into a full `(n_query, n_ref)` array instead.

It will keep each row's top `c` candidates — indices and similarities,
`(n_query, c)` — reducing each block as it is produced. Peak residency becomes
the transient chunk block, which already exists for every aggregation, plus
`n_query × c`.

`c` and the user's `k` are different quantities: `k` is how many neighbours vote
on confidence, `c` is how many spots the assignment may choose among. One
`argpartition` at `max(c, k)` serves both.

### 2. A seed that makes the matching provably exist

Before solving, build a greedy injective assignment: each cell, in descending
order of its best similarity, takes its best unused candidate; a cell whose
candidates are all taken receives an arbitrary unused spot, which exists because
`n_ref ≥ n_query` is already required. Those edges are added to the candidate
graph.

A full matching therefore exists by construction and the solver cannot fail —
which is the difference between shipping this and shipping something that
raises on the datasets it was built for. The optimizer can only improve on the
seed, so the seed costs nothing but the edges it adds.

**Deduplicate edges by `(row, col)` before building the CSR.** Constructing
`csr_matrix((vals, (rows, cols)))` with repeats *sums* them, and most seed edges
are already candidates, so their costs silently double and the solver avoids
exactly the edges it should prefer. Measured, that mistake alone cost 12% of the
objective at `c=128` — it looks like a worse algorithm rather than a bug.

The solve maximizes similarity by minimizing `max(similarity) - similarity`,
plus a small positive constant so no kept edge stores an exact zero, which
`min_weight_full_bipartite_matching` would read as "no edge".

### 3. Dense stays for problems that fit, and the run says which ran

`n_query × n_ref ≤ 2e7` (~240 MB peak) keeps `linear_sum_assignment`, so
existing runs stay bit-identical and provably optimal. Above that the candidate
path runs instead of the current refusal.

`Projection` gains `assignment` (`'exact'` or `'candidate'`) and
`n_candidates`, which reach the run result. A near-optimal answer must never be
readable as an exact one — the same reason axis fidelity is always reported
against the reference's ceiling.

The size ceiling does not disappear, it moves: the candidate graph is
`n_query × c` edges, so the refusal now fires around 400,000 cells rather than
around 10,000.

### 4. The other memory wall

`_CHUNK` is a fixed 2,048 rows, so the transient block is `2048 × n_ref` —
820 MB against a 100,000-spot reference, which would block the reference
direction even after this change, and does so for *every* aggregation, not only
injective. Choosing the chunk so `chunk × n_ref` stays under ~2e7 entries
removes it.

## Out of scope

Exposing `c` in the UI. It is a quality/memory dial with a defensible default
and nothing yet demands per-run control; a module constant and a function
parameter are enough.

A fallback if the solver raises despite the seed. The seed makes a full matching
a theorem, so a raise there is a bug and should surface as one rather than be
silently absorbed.

Approximate solvers (auction, ε-scaling) and blocked assignment. The candidate
restriction already buys 77× with a measured quality cost of nothing the metrics
can see; a second approximation would be harder to justify and harder to explain.

## Testing

- Known-optimum tiny matrices, including the 2×2 where greedy and optimal differ.
- The adversarial clones fixture — near-identical query cells, which is where
  plain top-k provably has no matching — must stay feasible at small `c`.
- Distinct spots always equal `n_query`, on both paths.
- The candidate path agrees exactly with the dense one at large `c` on a small
  problem.
- The switch reports which path ran, and the dense path still runs for the sizes
  it runs for today.
- Verified in the browser on the limb pair, since the numbers above come from a
  prototype and the shipped path has to reproduce them.
