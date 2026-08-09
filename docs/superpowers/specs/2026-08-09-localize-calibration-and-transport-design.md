# Localize: measuring the map, and an estimator that can spread

**Date:** 2026-08-09
**Status:** Approved, ready to plan

## Problem

Localize's kNN projection produces maps that are qualitatively wrong on real
data, and nothing in the tool says so. On an E11.5 mouse limb pair
(`E11.5_best_102424.h5ad`, 9,145 spots, as reference; `GSM4227224_E11_112125.h5ad`,
2,683 cells, as query) the default settings invert the tissue's most basic
anatomy:

| | Spearman(epidermis score, distance to tissue outline) |
|---|---|
| reference | **−0.116** — epidermis hugs the outline, correctly |
| prediction (`weighted_mean`, k=50) | **+0.166** — epidermis flees the outline |

The epidermis is the outermost tissue in the embryo and is predicted to be the
most interior. The predicted cloud also occupies **0.25×** the reference's hull
area. Both the cross-validation panel and the confidence score were "working" —
mean confidence 0.19 correctly reported that the neighbours were scattered —
yet nothing named the failure, and the user had run eight parameter variants
without a way to tell which was better.

**The mean of a ring is its hole.** Every epidermal query cell matches epidermal
reference spots; those are spread around the whole perimeter; their weighted
mean is the centre of the bud, for every one of them. `weighted_mean` cannot
represent a non-convex domain, and an annulus is the worst case.

## The measured frontier

Measured on the limb pair with `normalize_total` + `log1p`, all 22,016 shared
genes, `zscore` + `correlation`. Axis fidelity is Spearman between a marker
set's score and the predicted coordinate along the relevant axis; the reference
row is that same quantity computed on the reference itself, i.e. the ceiling.

| method | hull area | distal | proximal | anterior | posterior |
|---|---|---|---|---|---|
| **reference (ceiling)** | 1.00 | −0.327 | +0.316 | +0.306 | −0.178 |
| `weighted_mean` k=50 | **0.15** | −0.413 | +0.158 | +0.250 | −0.090 |
| `best_match` (greedy) | 0.98 | −0.045 | +0.016 | +0.055 | −0.017 |
| injective assignment | 0.99 | −0.056 | +0.017 | +0.033 | −0.027 |
| sample from neighbours | 0.98 | −0.091 | +0.014 | +0.074 | +0.018 |
| OT barycentric ε=0.05 | 0.22 | −0.226 | +0.254 | +0.396 | +0.095 |
| **OT barycentric ε=0.02** | **0.59** | −0.173 | +0.119 | +0.163 | +0.007 |

Two facts fall out, and they shape everything below.

**Every per-cell point estimator sits at one end of a frontier.** Averaging k
neighbours cancels noise, so the systematic positional component survives — all
four axis signs are correct — but the estimate is shrunk to 15% of the tissue.
Any method that picks a single location per cell (`best_match`, injective,
sampling) keeps the full extent and retains no signal at all. This is textbook
regression dilution: the mean is a shrinkage estimator with the right direction
and the wrong scale; a single draw is unbiased with variance that swamps it.

**Rescaling cannot fix it.** A per-axis affine rescale is monotone, so it
provably leaves every Spearman value unchanged while fixing the scale — and it
does: rescaling `weighted_mean` to the reference's per-axis standard deviation
keeps (−0.408, +0.124, +0.244, −0.054) exactly. But hull area overshoots to
**2.93** and 14.8% of cells land outside the tissue, because matching the
variance of a compact blob to that of a bounded paddle is not the same as
matching its shape. Calibration of the marginal, not of the variance, is what
is needed.

**Optimal transport is the only estimator that moves along the frontier.** A
marginal constraint on the reference side forces the predicted population to
occupy the tissue while the entropy term still lets each cell hedge across
several spots, so averaging's noise cancellation survives. ε is the dial: 0.05
gives area 0.22 with strong axes, 0.02 gives area 0.59 with moderate ones.

**Nothing reaches the ceiling at full area.** OT is a better trade, not a
solution, and this design does not claim otherwise. The posterior (`Hand2`,
`Shh`, `Grem1`) axis is weak under every method and OT gets its sign wrong;
AP position should be trusted much less than PD.

> **Caveat on these numbers.** They come from a prototype that normalizes both
> matrices and uses all shared genes. The user's loaded reference is filtered to
> 12,449 genes and carries SCT columns, and on *that* preprocessing the same
> comparison ranks differently (`best_match` scored ρ=−0.239 on the rim test,
> `weighted_mean` +0.166). The direction of the frontier is robust; the exact
> values are not. That sensitivity is itself an argument for the harness.

## Approach

Four pieces, in dependency order. The first is the one that makes the rest
decidable.

### 1. A calibration harness

Right now "the map looks wrong" has no number, so parameter exploration is
guesswork — eight saved variants with no way to rank them. Every metric used in
the diagnosis above was invented ad hoc for it. They become a reported panel,
computed against the reference as the ceiling, so a run can be judged and two
runs compared.

Four metrics, each answering a question a user actually asks:

**Spatial pattern fidelity — "does this cell type land where it lives?"** For a
marker set, bin the tissue on a common grid; compute the marker-weighted cell
density from the reference and from the prediction; report the correlation
between the two density maps. One number per set, 1.0 is perfect. This is the
general form of the epidermis test and the headline metric — it is the one that
would have caught the inversion immediately.

**Axis fidelity — "are the gradients real?"** For a marker set, find on the
*reference* the unit direction **u** maximizing |corr(score, coords·**u**)|,
then report that correlation on the reference and on the prediction along the
same **u**. Deriving **u** from the reference makes it orientation-free: the
user never has to declare which way is distal, and the sign convention comes
from the data. Reported as a pair (prediction, ceiling), never as a bare number.

**Dispersion — "does the map fill the tissue?"** Hull area of the predictions
over hull area of the reference, computed on the inner 95% of points so a few
outliers cannot dominate, plus the per-axis standard-deviation ratio. Catches
the 0.15 collapse and the 2.93 overshoot alike.

**Occupancy — "is it piling up?"** Distinct reference spots used as nearest
match, maximum cells on one spot, and the effective number of spots (exponential
of the assignment entropy). Catches the technical artifact where a
better-recovered spot correlates well with everything: greedy `best_match` on
HVG put 2,683 cells onto 865 spots, 15 on a single one.

The harness is a pure module taking arrays, so it is testable without AnnData
and reusable by the cross-validation path that already exists.

### 2. Optimal transport

A new aggregation that solves for a coupling **P** between query cells and
reference spots minimizing transport cost against expression dissimilarity,
subject to marginals: each query cell fully placed, and the reference-side
marginal matching spot density. Entropic regularization (Sinkhorn, log-domain
for stability — the naive form underflows at ε≤0.05 on this data) with ε
exposed as the spread/fidelity dial.

Prediction is the barycentric projection Σⱼ Pᵢⱼ·coordsⱼ. Taking the row argmax
instead returns to the single-location regime and loses the axes (measured:
0.98 area, ~0 correlation), so barycentric is the one to ship.

Cost: 4 s for 2,683 × 4,000 at 60 iterations. Full 9,145 spots at 120
iterations exceeded two minutes in the prototype, so the implementation needs
either float32 with a fixed iteration budget or reference subsampling, and must
run as a cancellable background task with progress.

**Each row of P is a spatial posterior** — the honest output for a cell that
genuinely could be in several places. That also gives a principled replacement
for the current heuristic confidence (spread of the neighbour cloud): the
entropy of the row, or the posterior's spatial variance.

### 3. Injective assignment

`best_match` takes each cell's single best spot independently, so a spot that
correlates well with everything — better recovery, less noise — absorbs many
cells. Optimal assignment maximizes total similarity subject to distinct spots,
via `scipy.optimize.linear_sum_assignment`: **0.5 s** at 2,683 × 9,773. It takes
distinct spots to 2,683 out of 2,683 in every configuration measured — from
1,749 (all shared genes) and from 865 (HVG). The cost is small: on the
all-shared run, mean similarity falls 0.163 → 0.151.

Available only when the reference has at least as many spots as the query has
cells, and off by default, because injectivity asserts that the query's
composition matches the tissue's. Dissociation biases which cell types survive,
so forcing one-to-one pushes over-represented types into locations they do not
belong. The UI must say this rather than presenting it as strictly better.

### 4. Change the default

`weighted_mean` is the default aggregation and produces a 15%-area map that
inverts the epidermis on this data. Given the harness makes the alternatives
comparable, the default should move — to OT once it exists. Until then, the
existing parameter advice should warn when the aggregation is `weighted_mean`
and the reference has any strongly peripheral or annular population, which the
spatial-pattern metric can detect on the reference alone.

## Phasing

Each phase produces something usable on its own.

1. **Harness** — metrics module, wired into the existing cross-validation panel
   and into a comparison view for embeddings already produced. Immediately lets
   the user rank the eight variants they have.
2. **Injective assignment + default warning** — small, and phase 1 proves
   whether it helps on their data rather than on the prototype's preprocessing.
3. **Optimal transport** — the largest piece, and the one whose value phase 1
   is required to demonstrate.

## Out of scope

Gromov-Wasserstein (preserving pairwise geometry rather than matching to
absolute coordinates, as novoSpaRc does) — a reasonable next step if entropic OT
proves insufficient, but it is a different objective and a much larger
implementation. Unbalanced OT for composition mismatch. Deep methods (Tangram's
learned mapping). Conformal prediction regions for calibrated per-cell
uncertainty — attractive once a posterior exists, but it needs the posterior
first. 3-D reconstruction. Mapping expression in the other direction.
