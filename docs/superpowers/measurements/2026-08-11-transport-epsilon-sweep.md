# What ε does on the E11.5 limb pair

Reference `E11.5_best_102424.h5ad` (9,773 spots), query
`GSM4227224_E11_112125.h5ad` (2,683 cells), 22,016 shared genes.
Settings as actually used: `transform='none'`, `metric='cosine'`.
Scored with the Phase 1 harness (`xcell.localize_metrics.evaluate_map`), so
these are the numbers the Map quality panel reports.

Marker sets: **proximal** `Meis1, Meis2, Irx3, Pbx1` and **distal** `Hoxd12,
Hoxd13` — the genes Patrick uses to confirm proximodistal orientation — plus
**epidermis** `Krt5, Krt14, Krt15, Trp63, Wnt6`, the population whose inversion
started this line of work. All eleven genes are present in both datasets.

## The sweep

Runs at 150 iterations, on the full reference (26.2M entries, dense path).

| method | area | distinct spots | proximal pattern | distal pattern | epidermis pattern | runtime |
|---|---|---|---|---|---|---|
| `weighted_mean` | 0.086 | 626 | +0.307 | **−0.532** | **−0.256** | 2.5 s |
| `best_match` | 0.496 | **22** | +0.202 | +0.342 | +0.191 | 2.6 s |
| transport ε=0.03 | 0.477 | 1819 | **+0.623** | **+0.420** | **+0.328** | 306 s ✗ |
| **transport ε=0.05** | **0.437** | **1774** | +0.588 | +0.353 | +0.271 | 203 s |
| transport ε=0.07 | 0.396 | 1673 | +0.593 | +0.259 | +0.219 | 147 s |
| transport ε=0.1 | 0.342 | 1560 | +0.591 | +0.162 | +0.214 | 101 s |
| transport ε=0.2 | 0.193 | 1098 | +0.526 | −0.104 | +0.023 | 63 s |
| transport ε=0.3 | 0.114 | 775 | +0.439 | −0.379 | +0.030 | 45 s |
| transport ε=0.5 | 0.055 | 454 | +0.323 | −0.628 | −0.129 | 25 s |
| transport ε=0.8 | 0.027 | 244 | +0.466 | −0.639 | −0.161 | 19 s |
| transport ε=1.2 | 0.013 | 148 | +0.558 | −0.700 | +0.160 | 19 s |
| transport ε=2.0 | 0.004 | 72 | +0.524 | −0.706 | +0.783 | 16 s |

✗ = did not reach its convergence tolerance within the iteration budget.

Axis fidelity, prediction against the reference's own ceiling:

| method | proximal axis | distal axis |
|---|---|---|
| `weighted_mean` | +0.166 of +0.365 | +0.440 of +0.311 |
| `best_match` | **−0.036** of +0.365 | +0.231 of +0.311 |
| transport ε=0.05 | +0.205 of +0.365 | +0.230 of +0.311 |
| transport ε=0.1 | +0.215 of +0.365 | +0.259 of +0.311 |

## What it says

**Transport beats both baselines on almost everything.** At ε=0.05 it reaches
five times `weighted_mean`'s area (0.437 against 0.086) and 88% of
`best_match`'s, while using 1,774 distinct spots against `weighted_mean`'s 626
and `best_match`'s **22** — that estimator puts 2,683 cells onto twenty-two
spots. It scores better than either baseline on all three marker patterns, and
**reverses both inversions**: epidermis −0.256 → +0.271, distal −0.532 → +0.353.

That is the frontier claim in the parent spec, holding on the real pair: not
"fills the tissue" or "keeps the gradient" but both at once.

That is the frontier claim in the parent spec, holding on the real pair: not
"fills the tissue" or "keeps the gradient" but both at once.

**The dial is monotone and steep.** Area falls from 0.477 to 0.004 as ε goes
0.03 → 2.0, and distinct spots from 1,819 to 72. There is no plateau; every step
costs.

**Pattern fidelity is not readable on a collapsed map.** The epidermis score at
ε=2.0 is +0.783, the highest in the table, on a map covering 0.4% of the tissue
with 72 distinct spots — every cell is in a handful of bins and the correlation
is measuring almost nothing. Read pattern together with area, never alone. This
is worth stating because it is the metric most likely to be quoted on its own.

**Cost and iterations move together and against ε.** ε=2.0 converged in 10
iterations and 16 s; ε=0.05 needed 300 and 203 s; ε=0.03 had still not met
tolerance after 400 iterations and 306 s. Small ε is where the estimator is good
*and* where it is slow, which is why the run reports its iteration count and
marginal error rather than implying convergence.

## The default

**ε = 0.05, `max_iterations` = 300**, with a useful band of roughly 0.03–0.3.

The spec's provisional 0.5 was ten times too smooth and would have shipped a map
covering 5.5% of the tissue with the epidermis still inverted (−0.129) — the
exact failure this estimator was built to fix.

**0.03 is not the default even though it scores highest**, because it did not
meet its convergence tolerance in 400 iterations (final marginal error 4.4e-3,
44× the tolerance). A default whose marginals were never satisfied would
contradict the reason the reference-side marginal is in the design at all. It is
inside the advertised band for anyone who wants it and is willing to raise the
budget.

`max_iterations` moves 200 → 300 with it: at 200 the new default would not
converge, and a default that cannot reach its own tolerance is not a default.

## The bug this sweep found

The first run of this sweep produced hull areas of 0.013 down to 0.000 at every
ε, all converging in ten iterations. The cause was in `normalize_cost`, not in
the solver.

Entropic OT is invariant to `C_ij → C_ij + a_i + b_j`: both offsets are absorbed
into the potentials, so neither can move the coupling. The normalizer divided by
the *global* standard deviation, which counts them. On this pair:

| component of the cost spread | std |
|---|---|
| global | 0.0827 |
| row effects | 0.0115 |
| **column effects** | **0.0808** |
| double-centred | **0.0134** |

The column effect is 98% of the global spread — reference spots that are better
recovered and so similar to *everything*, the same artifact that makes
`best_match` pile 2,683 cells onto 22 spots. Normalizing by it inflated the
scale **6.2×**, so every ε was effectively six times larger than requested and
the barycentric projection collapsed. Double-centring before measuring the
spread is the fix, and
`test_normalization_ignores_row_and_column_offsets` is the test that would have
caught it: it asserts the coupling is unchanged when the cost gains per-cell and
per-spot offsets.
