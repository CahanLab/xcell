# Why xcell's NMF specificity weight defaults to 1.0, not GeneNMF's 5.0

**Date:** 2026-08-19
**Data:** `E11.5_hindlimbs_102424_xcell_qced-08132026.h5ad` — 24,172 cells ×
12,236 genes, 5,001 highly variable. Values are log1p-normalized (min non-zero
0.693), 2.6% dense.
**Code:** `xcell/gene_nmf.py`, k=12, seed 0, `tol=1e-6`, `weight_explained=0.5`.

## The finding

GeneNMF's `getNMFgenes(specificity.weight = 5)` collapses a single-sample
factorization. Sweeping the exponent:

| specificity_weight | programs kept | median genes | sizes |
|---|---|---|---|
| 0.0 | 12/12 | 200 | all capped |
| 0.5 | 12/12 | 200 | `200×8, 46, 17, 200, 43` |
| **1.0** | **11/12** | **200** | `200×5, 128, 194, 200, 20, 71, 2` |
| 2.0 | 8/12 | 34 | `200, 54, 47, 104, 22, 11, 1, 8` |
| 3.0 | 5/12 | 6 | `83, 1, 2, 33, 6` |
| 5.0 (GeneNMF default) | **4/12** | **4** | `6, 1, 3, 5` |

## Mechanism

`wgtLoad` scores each gene by `spec^w`, where `spec` is the largest share any
one program takes of that gene's total loading. On real data almost nothing is
exclusive — measured specificity percentiles were
`[0.16, 0.21, 0.27, 0.36, 0.57]`, i.e. the median gene's best program owns just
27% of it. At `w=5` that spread becomes `(0.57/0.16)^5 ≈ 600×`, so the single
most exclusive gene swamps the column: after re-normalizing, one gene held
60–98% of each program's mass. `weightCumul`'s strict `cumsum < weight.explained`
then keeps *zero* genes, and the program is dropped.

Raw loadings before weighting were perfectly healthy — 115–592 genes to reach
50% of a column, top gene 0.3–30%. The factorization is not the problem.

**Why GeneNMF gets away with 5.0:** it never applies this to one factorization.
`getNMFgenes` is called on `multiNMF` output, and the meta-program path
(`get_metaprogram_consensus`) *averages* gene weights over every program in a
cluster before the cutoff. That averaging flattens the distribution. The
single-sample path has no such smoothing step.

## Why 1.0 rather than 0.0

Weighting still earns its place — it just has to be gentle. At `w=1` the distal
program's top genes are `Nrp2, Cyp26b1, Hoxd13, Hoxd11, Hoxa11os, Grem1`; at
`w=0` the same factor leads with `Stip1, Nrp2, Sulf2, Ccnd2, Nolc1, Lig1` and
the Hox genes fall out of the top 12 entirely. Sharper, and still 11–12 of 12
programs survive.

## Biology at the chosen default (k=12, w=1.0)

Recovered programs read as E11.5 limb bud:

- `Hoxd13, Hoxd11, Hoxd10, Hoxa11os, Grem1, Cyp26b1` — distal/posterior autopod
- `Sox9, Col2a1, Shox2, Vcan, Cdkn1c` — chondrogenic condensation
- `Col1a1, Col1a2, Col3a1, Postn, Fbn2, Tgfbi` — connective tissue
- `Mki67, Cenpe, Cenpf, Birc5` — cell cycle
- `Hba-x, Hba-a1, Hbb-bh1, Slc25a37, Alas2` — primitive erythroid
- `Meis1, Pbx1, Zfhx3, Hoxc10` — proximal identity

## Factorization quality (same data, k=12)

Not a tuning question — recorded because it sets what "converged" means here.

| solver | time | iters | relative residual |
|---|---|---|---|
| `xcell.gene_nmf` tol=1e-4 (default) | 0.5 s | 27 | 0.90754 |
| `xcell.gene_nmf` tol=1e-6 | 2.5 s | 130 | 0.90687 |
| sklearn `mu` | 3.6 s | 80 | 0.90710 |
| sklearn `cd` | 6.5 s | 164 | 0.90670 |

A ~0.91 residual is normal for NMF on sparse log-normalized counts — most of
the Frobenius norm is per-cell sampling noise no low-rank model captures. We
land within 0.02% of sklearn's best at 2.5× its speed.

## Test coverage

`tests/test_gene_nmf.py::test_default_specificity_weight_keeps_programs_usable`
guards the default using `_realistic_loadings()`, a synthetic W calibrated to
the specificity percentiles above. Note the synthetic reproduces *shrinkage*
(median 61 genes at w=1 → 10 at w=5) but not the full collapse to 4/12 seen on
real data; it is a floor, not a faithful replica. Disjoint-block fixtures give
every gene specificity 1.0 and cannot exercise this path at all.
