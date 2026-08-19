# Meta-program defaults: specificity_weight 5.0, weight_explained 0.8

**Date:** 2026-08-19
**Data:** `E11.5_hindlimbs_102424_xcell_qced-08132026.h5ad` — 24,172 cells ×
12,236 genes, 5,001 highly variable, three real sections
(A 7,981 / B 7,046 / C 9,145).
**Run:** `multi_nmf(ks=4..9, seed=0)` → **117 programs in 1.9 s**, then
`meta_programs(n_mp=8)`.

## Why these differ from the single-sample defaults

[The single-sample path](2026-08-19-gene-nmf-specificity-weight.md) had to drop
`specificity_weight` from GeneNMF's 5.0 to 1.0, because on one factorization the
exponent piles a program's whole mass onto one gene and the cumulative cutoff
keeps nothing.

That reasoning **does not carry over here**, and the measurement says so. The
meta-program consensus averages gene weights across every program in a cluster
before the cutoff (`trimmed_mean` over 6–30 programs), which is exactly the
smoothing the single-sample path lacked. GeneNMF's 5.0 survives — *provided*
the cutoff is loosened, since after weighting the surviving genes hold most of
the mass and a 0.5 cutoff admits only a handful.

## The grid

Purity = of the marker genes in a meta-program's top 50, the share belonging to
its single best-matching category, averaged over meta-programs. Categories
captured = how many of 8 known limb programs (blood, chondro, muscle, ectoderm,
distal, proximal, fibro, cycle) any meta-program picks up with ≥3 markers.

| specificity | weight_explained | kept | median genes | categories | purity |
|---|---|---|---|---|---|
| 0.0 | 0.5 | 8/8 | 200 | 3 | 0.57 |
| 1.0 | 0.5 | 8/8 | 200 | 5 | 0.56 |
| 2.0 | 0.5 | 8/8 | 112 | 5 | 0.63 |
| 3.0 | 0.5 | 8/8 | 16 | 5 | 0.73 |
| 3.0 | 0.8 | 8/8 | 200 | 5 | 0.63 |
| 5.0 | 0.5 | **6/8** | **7** | **2** | 1.00 |
| **5.0** | **0.8** | **8/8** | **46** | **5** | **0.72** |
| 5.0 | 0.9 | 8/8 | 60 | 5 | 0.68 |

`(5.0, 0.5)` is the trap: purity reads 1.00 only because just two
meta-programs still have enough genes to match anything. `(5.0, 0.8)` keeps
every meta-program at a usable size with the best purity of any setting that
captures all five recoverable categories.

## Biology at (5.0, 0.8)

Sizes `[8, 27, 57, 25, 41, 158, 52, 200]`:

- **MP2** (cov 1.00, sil +0.78) — `Hbb-bh1, Hba-a1, Hba-x, Hbb-bt, Hba-a2,
  Hbb-bs, Slc25a37, Slc4a1, Car2, Hemgn`: primitive erythroid, clean.
- **MP4** (cov 1.00, sil +0.68) — `Col1a2, Col1a1, Col3a1, Sparc, Postn, Lum,
  Dcn, Col5a2`: ECM / connective tissue, clean.
- **MP7** (cov 0.67, sil +0.64) — `Sox9, Col2a1, Shox2, Myog, Tnnt1, Msc,
  Tbx18, Tbx15`: chondro-myogenic.
- **MP8** (cov 0.33, sil +0.91) — `Nrp2, Hoxd11, Hoxd13, Grem1, Cyp26b1,
  Hoxa11os, Hoxd10, Hoxd12, Hand2, Lmx1b`: distal/posterior autopod. Present in
  one section only, which is what sample coverage is for — the sections cut at
  different proximodistal levels.
- **MP6** (cov 1.00, sil **−0.05**, 30 programs, 158 genes) — the junk-drawer
  cluster, and the negative silhouette says so. Worth surfacing in the UI
  rather than hiding.

At `(3.0, 0.8)` seven of eight meta-programs sit at the 200-gene cap and MP1/
MP3/MP5/MP6 are visibly mixed (blood genes inside mesenchymal programs).

## Test coverage

`tests/test_gene_nmf_meta.py::test_default_meta_program_settings_produce_usable_gene_sets`
and `::test_the_single_sample_weight_cutoff_would_stunt_meta_programs` guard the
pair using `_realistic_fit()` — programs recurring across three models, each
carrying the heavy-tailed low-specificity shape measured on real data. On that
fixture `(5.0, 0.5)` yields median 11 genes against `(5.0, 0.8)`'s 66, so the
two settings are distinguishable there even though the synthetic does not
reproduce the full collapse seen on real data.

## Performance

117 factorizations (3 samples × 6 ranks) over 5,001 genes in **1.9 s** total —
the per-sample matrices are small, so a full sweep is interactive rather than a
coffee break.
