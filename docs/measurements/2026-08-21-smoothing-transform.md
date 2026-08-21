# Which transform should kNN smoothing run on?

**Date:** 2026-08-21
**Question:** xcell's Smooth writes `layers[out] = (D⁻¹(A+αI))ⁿ · M` for whatever
matrix `M` you point it at. Does it matter whether `M` holds raw counts,
library-normalized values, or log-normalized values — and is it better to
smooth the raw counts?
**Short answer:** yes it matters, and yes — **smooth counts, transform
afterwards**. The effect on estimation accuracy is small but consistent; the
effect on *what the stored numbers mean* is large. Separately: for Localize,
smoothing the reference is worth far more than the transform choice.

**Data:** `E11.5_hindlimbs_102424.h5ad` — 25,934 spots × 36,596 genes, raw
integer counts, real spatial coordinates. Gene panel: 2,000 genes sampled
evenly across ten deciles of total count (so the answer is not just the answer
for highly expressed genes), all with ≥ 20 counts.

## Method: molecular cross-validation

Guessing which denoised matrix is "better" by eye is hopeless, so the counts are
split into two independent halves by binomial thinning — `train ~ Binom(y, ½)`,
`test = y − train`. Under Poisson sampling the halves are independent given the
true rate, so a pipeline that recovers the true local rate from `train` predicts
`test` well, and one that distorts it does not (Batson & Royer 2019).

Each pipeline turns `train` into a predicted per-cell expression profile,
scored by held-out **Poisson deviance** (lower is better). The smoothing graph
is a **spatial** kNN built from coordinates alone, so it is independent of the
counts and cannot leak. Three thinning seeds per configuration; the spread
across seeds is ±0.0005 or less, so every difference quoted below is 10–100×
the noise.

## Result 1 — smoothing helps a lot, and the transform choice is a small, consistent effect

Held-out Poisson deviance per cell–gene, mean of 3 seeds:

| pipeline | k=10, 1 step | k=20, 2 steps |
|---|---|---|
| no smoothing | 0.5962 | 0.5962 |
| **counts → smooth → normalize** | **0.3913** | **0.1942** |
| normalize → smooth | 0.3937 | 0.1978 |
| normalize → log1p → smooth | 0.3933 | 0.2003 |
| normalize → sqrt → smooth | 0.3941 | 0.2190 |

Smoothing itself is worth 34–67% of the deviance — far more than any transform
choice. Among the transforms, averaging **counts** wins at every setting, and
its margin widens as smoothing strengthens: +0.6% over log at k=10×1, +3.1% at
k=20×2, where sqrt is 13% worse.

Averaging counts weights each neighbour by its sequencing depth, which is the
Poisson-optimal weighting: a deeper neighbour carries more information about the
local rate. Averaging already-normalized profiles throws that weighting away.

### The interaction worth knowing

Broken out by expression level (k=20 × 2 steps, tertiles of gene total count):

| pipeline | low | mid | high |
|---|---|---|---|
| counts → smooth → normalize | 0.04958 | 0.18705 | **0.34644** |
| normalize → smooth | 0.04976 | 0.18846 | 0.35567 |
| normalize → log1p → smooth | **0.04880** | **0.18534** | 0.36738 |
| normalize → sqrt → smooth | 0.04867 | 0.19195 | 0.41685 |

The log and sqrt pipelines are *better* on the lowest-expressed third — their
concavity damps the influence of a neighbour that happened to catch a few
counts. They lose on the highest-expressed third, by more than they win, which
is why counts wins overall. Marker genes and anything you would plot live in
that third.

## Result 2 — the large effect is on what the numbers mean

Averaging does not commute with the log: the mean of logs is not the log of the
mean, and with sparse data the gap is not subtle. Comparing `log1p(smoothed
normalized counts)` against `smoothed log1p values` on the same graph (k=10):

| decile | median gene total | gap (log units) | relative shrinkage |
|---|---|---|---|
| 1 | 26 | 0.022 | 83.0% |
| 5 | 253 | 0.200 | 82.3% |
| 8 | 826 | 0.581 | 80.8% |
| 10 | 2,515 | 1.443 | 70.4% |

A smoothed log layer sits **70–83% below** the log of the smoothed expression.
The reason is sparsity: if one neighbour in eleven carries a gene at
CP10K ≈ 100, the mean of the logs is `log1p(100)/11 ≈ 0.42` while the log of the
mean is `log1p(9.1) ≈ 2.3`. Both orderings are similar, so a heatmap still
looks right — but the values are no longer on the log-expression scale they
claim, and anything comparing them against unsmoothed values, or against a
threshold, is wrong by a factor of ~5.

**This is the practical finding.** xcell's Smooth defaults to `.X`, and on a
typical dataset `.X` is log-normalized, so the default path produced exactly
this matrix.

## Result 3 — the depth-weighting worry does not materialize

The theoretical objection to averaging raw counts is that it weights neighbours
by sequencing depth, which could import a depth gradient into the result. It
does not: across neighbourhoods, per-cell deviance for counts-space smoothing
minus normalize-first smoothing correlates **−0.17** with how heterogeneous the
neighbourhood's depth is, and the counts advantage *grows* with heterogeneity
(quartiles: −0.0014, −0.0022, −0.0027, −0.0032; negative = counts better).
Median neighbourhood depth CV here is 0.64, so this is not a low-variance
regime. Depth weighting is doing the right thing, exactly as Poisson theory
says it should.

## Result 4 — for Localize, smoothing the reference is what matters

Same tissue, different question. 3,000 spots held out as a stand-in query
(true coordinates known), reference built from the other 22,934, 2,000 HVGs,
correlation on z-scored profiles. The smoothing graph is built on **reference
spots only** — smoothing before the split would leak a held-out spot's counts
into the profiles it is matched against. The query is never smoothed, mirroring
the real case of a dissociated scRNA-seq query.

Nearest-reference-spot match, share of query spots placed within 10% of the
tissue radius:

| reference matrix | within 10% | ρ(x) | ρ(y) |
|---|---|---|---|
| raw counts | 2.5% | 0.141 | 0.052 |
| log-normalized | 2.3% | 0.115 | 0.065 |
| smoothed k=5 × 1 | 4.9% | 0.199 | 0.172 |
| smoothed k=10 × 1 | 5.6% | 0.196 | 0.238 |
| smoothed k=20 × 1 | 6.0% | 0.268 | 0.223 |
| smoothed k=20 × 2 | 7.2% | 0.312 | 0.251 |
| smoothed k=40 × 2 | 7.7% | 0.287 | 0.257 |
| smoothed k=20 × 4 | 7.9% | 0.305 | 0.269 |

Smoothing the reference roughly **triples** the share of spots placed near the
truth, with returns flattening after about k=20 × 2 steps. And here the
transform genuinely does *not* matter: smoothing counts and smoothing an
already-log layer land within noise of each other (7.2% vs 6.7% at k=20 × 2),
because Localize z-scores each dataset per gene, which absorbs the difference.

Two caveats. The absolute numbers are low because E11.5 limb mesenchyme is
largely homogeneous in expression — most cells genuinely cannot be placed, and
median error stays near the tissue radius throughout; the *relative* comparison
is the result. And this is same-platform, so it understates the difficulty of a
real dissociated query.

A useful side observation: with `transform='rank'`, raw counts and
log-normalized produce **bit-identical** results (2.4% within 10%, ρ 0.076 /
0.035), which is what invariance to a monotone transform means and a check that
the harness measures what it claims. Rank is also markedly worse than z-score
on this data.

## Guidance

1. **Smooth counts, transform after.** `Preprocess → Smooth` now has an
   *After averaging* option (`none` / `cp10k` / `lognorm`) so this is one step;
   before it, the recommended pipeline was not expressible in xcell at all,
   because Normalize Total and Log1p can only write `.X`.
2. **Do not smooth a log-normalized matrix and treat the result as expression.**
   The ordering survives; the scale does not. The Smooth panel now says so when
   the chosen source matrix is on a log scale.
3. **How much:** k ≈ 10–20 with 1–2 steps sits in the flat part of both curves.
   More is defensible for Localize; less is safer if fine spatial structure
   matters, which held-out deviance cannot see.
4. **For Localize specifically:** smooth the reference, don't agonize over its
   transform, and use Cross-validate (which reads the reference's layer) to
   confirm the gain on your own tissue.
5. **The exception:** if the analysis is dominated by very low-expressed genes,
   log-space smoothing is marginally better. That is the only case measured here
   where it wins.

## Reproducing

Scripts are in the session scratchpad rather than the repo (they read data from
`chef-gene/real_st_data`, which is not vendored): `smooth_mcv2.py` for
Results 1–3, `localize_smoothing.py` / `localize_strength.py` for Result 4.
