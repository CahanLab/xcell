# CNV inference against copy-number truth

Date: 2026-10-13. Branch `feat/cnv-benchmark`. Harness:
`backend/benchmarks/cnv/` (README there lists the downloads). Raw results:
the harness writes `results.tsv`; every number below comes from it.

## Question

Does `xcell.cnv` (inferCNV-style, shipped 2026-10-13) recover the copy-number
profile and the sub-clones of real tumor data, and which of its defaults
should change?

## Gold standard

Schmid et al., *Benchmarking scRNA-seq copy number variation callers*, Nat
Commun 2025 (doi:10.1038/s41467-025-62359-9), is the reference evaluation:
six callers, 21 datasets, truth from (sc)WGS or WES, ground truth deposited
at Zenodo 20260649. Two other 2025 benchmarks (Briefings in Bioinformatics
bbaf076, paired scRNA + scDNA from the same cells; Precision Clinical
Medicine pbaf011, HCC1395 and the Tian lung-line mixtures) reach the same
broad conclusions: no expression-only caller clearly leads, inferCNV and
CopyKAT are the usual references, allele-aware Numbat is best when BAMs are
available. Schmid's processed truth plus public count matrices made it the
only one I could reproduce without realigning reads.

Subset chosen (six datasets, about 600 MB), each for a reason:

| Dataset | Cells (tumor / reference) | Truth | Why |
|---|---|---|---|
| SNU601, gastric cancer line (Andor et al. 2020, GSE142750) + 5 gastric tissue samples as external reference (Kim et al., GSE150290, Pat25–29 A; Schmid's choice) | 5,186 / 3,949 | scWGS pseudobulk, 100 kb bins, fraction of cells gained / lost | Highly aneuploid, with sub-clonal events in 20–80 % of cells on 16 chromosomes |
| NUGC4, gastric cancer line, same reference | 3,797 / 3,949 | scWGS pseudobulk | Hyperdiploid: whole-chromosome gains, almost no losses |
| MM199, multiple myeloma bone marrow (Mahdipour-Shirayeh et al., GSM4200471) | 1,608 plasma / 1,205 B, NK, T, monocytes (my marker-based annotation; 429 erythroid cells neither) | WES, GATK calls | A patient sample with its own normal cells |
| BCC06, basal cell carcinoma su006 pre-treatment (Yost et al., GSE123813) | 1,403 / 33 (endothelial, melanocytes, myofibroblasts) | WES | The benchmark's hardest case: tiny reference |
| BCC mix: tumor cells of su005, su006, su007, su008 plus all their normals | 2,521 / 398 | Patient identity | Clone recovery: four patients are four CNV clones |
| PBMC 10k (10x, healthy donor); CD4 T cells as reference | 0 / 3,442 (7,148 other normal cells) | Every bin diploid | False-positive level |

Cells were kept at ≥ 1,000 UMIs and ≤ 10–20 % mitochondrial counts. The
gene filter, window, reference and clustering were otherwise the method's
own.

## Metrics

Schmid's profile metrics, reimplemented in `evaluate.py`: the tumor cells'
pseudobulk per window is spread onto 100 kb bins; **max F1** is the macro F1
over loss / base / gain at the best pair of thresholds on the method's own
scale; Pearson with the mean copy state; AUC for gains and for losses.
scWGS truth calls a state when more than half the cells carry it, WES truth
is GATK's call. Per cell: AUROC of `cnv_score` for tumor versus reference
cells; ARI and homogeneity of clones against patient in the BCC mix; for
SNU601 the correlation, over bins where 20–80 % of DNA cells carry an event,
between that fraction and the fraction of RNA cells called. For PBMC the RMSE
of non-reference cells from zero, divided by the SNU601 tumor SD of the same
run (a tool that smooths over unexpressed genes shrinks every value, so the
raw RMSE is not comparable across tools).

Baseline: infercnvpy 0.6.1 with the same gene positions, reference
categories, window 100 / step 10, dynamic threshold 1.5, Leiden resolution
1.0 (its defaults), installed beside the harness.

## Results

Shipped defaults (`default`: min_mean 0.1, resolution 1.0, clustering on the
thresholded matrix, all reference categories set the bounds, lineage genes
kept) versus the final defaults (`final`) and infercnvpy:

| Dataset | Metric | default | final | infercnvpy |
|---|---|---|---|---|
| SNU601 | max F1 / Pearson / AUC gain / AUC loss | 0.654 / 0.549 / 0.838 / 0.829 | **0.684 / 0.597 / 0.903 / 0.845** | 0.608 / 0.547 / 0.889 / 0.801 |
| NUGC4 | max F1 / Pearson / AUC gain | 0.586 / 0.560 / 0.939 | 0.594 / 0.571 / 0.925 | **0.624** / 0.541 / 0.919 |
| MM199 | max F1 / Pearson / AUC gain / AUC loss | 0.614 / 0.445 / 0.812 / 0.772 | **0.667 / 0.571 / 0.825 / 0.815** | 0.605 / 0.482 / 0.765 / 0.806 |
| BCC06 | max F1 / Pearson | 0.406 / 0.223 | 0.403 / 0.230 | **0.426** / 0.219 |
| mean of four | max F1 | 0.565 | **0.587** | 0.566 |
| MM199 | tumor-vs-reference AUROC of cnv_score | 0.457 | **0.946** | 0.251 |
| BCC06 | same | 0.988 | **0.997** | 0.905 |
| SNU601 | same | 0.999 | 1.000 | 0.924 |
| BCC mix | ARI / homogeneity vs patient, clones | 0.398 / 0.802, 11 | **0.977 / 0.930**, 5 | 0.338 / 0.900, 10 |
| SNU601 | r(DNA fraction, RNA fraction) sub-clonal bins, gain / loss | 0.474 / 0.738 | 0.538 / 0.611 | 0.483 / 0.604 |
| PBMC | normalized false-positive level (lower is better), clones | 0.643, 18 | 0.552, 5 | **0.422**, 13 |
| SNU601 | seconds (9,135 cells) | 2.0 | 2.7 | 20.4 |

Schmid et al. report 0.57–0.59 as the mean max F1 of the best expression
callers over their 15 droplet datasets; the mean over these four is in that
range for all three rows. NUGC4 has losses in 2.7 % of bins, so its loss
metrics are noise; the F1 gap to infercnvpy there is loss sensitivity 0.17
versus 0.14 at lower precision. BCC06 is the dataset Schmid found every
caller fails on (44 % tumor / normal accuracy); 33 reference cells of three
other lineages is not a reference.

### Sub-clones

SNU601 splits into two RNA clones, 3,231 and 1,953 cells (37.7 %). The
smaller one carries a chr20 gain (+0.094 above the other clone) and a chr11
gain (+0.068) that scWGS puts in 39.5 % and 30.1 % of cells, and a stronger
chr18 loss; the two agree on the clonal chr6, chr12 and chr19 gains and chr9
loss. In the BCC mix all 398 normal cells fall in `clone_1`, su006 (1,402 of
1,403) and su008 (811 of 816) each get a clone, su007 (48 cells) joins
su005's main clone, and su005 splits 178 / 72 (not checked against DNA).
MM199's plasma cells split 1,312 / 292.

### What failed and why

- **A contaminated reference category erased the per-cell signal.** My
  MM199 annotation's 44-cell "T cell" cluster scored high on plasma
  markers too (doublets or misassigned cells). As one of four categories it
  set a bound, and because its mean was plasma-like, tumor values fell
  inside the bounds and were zeroed: tumor-vs-normal AUROC 0.457 with it,
  0.945 without (pseudobulk F1 barely moved, 0.614 vs 0.626). The 13-cell
  monocyte category did no harm. Any small category's mean is noisy and
  widens the min / max bounds outward, so the fix is a cell floor
  (`min_reference_cells`, 50) below which a category stays a reference
  category but does not set bounds; with no qualifying category every
  reference cell is pooled into one mean (BCC06: identical F1 either way,
  0.416 pooled vs 0.406). infercnvpy has the same exposure (AUROC 0.251).
- **Thresholding before clustering destroyed clone structure.** Clones
  from the noise-thresholded matrix: BCC-mix ARI 0.54–0.76; from the
  un-thresholded matrix 0.97–0.98. The 1.5 SD threshold zeroes the weak,
  consistent shifts that distinguish clones, leaving per-cell noise. The
  threshold stays for the stored matrix, the heatmap and the per-cell score,
  where it does help (tumor-vs-reference AUROC 0.95 vs 0.88 on the BCC mix).
- **Resolution 1.0 over-splits.** 24 clones on 9,135 cells, 18 on PBMC.
  0.5 gives 3–5 and the right patient partition.
- **The expression floor was too strict.** `min_mean` on mean log
  expression is not inferCNV's `cutoff` on mean counts: 0.1 kept 4,181
  MM199 genes (203 windows, one window for all of chr13) where inferCNV's
  rule keeps 5,717. Sweeping 0, 0.01, 0.02, 0.05, 0.1: mean max F1 0.568,
  0.587, 0.587, 0.581, 0.574; per-cell AUROC peaks at 0.01–0.02. No filter
  at all gives the lowest PBMC false-positive level (0.41, infercnvpy's) but
  the worst F1: unexpressed genes dilute signal and noise alike. 0.02 is
  the default.
- **Immunoglobulin and TCR loci.** Dropping IG / TR / HLA genes raised the
  MM199 AUROC from 0.945 to 0.973 and SNU601 F1 by 0.013, and left BCC06
  and PBMC unchanged; on by default (`exclude_lineage_genes`).
- **Per-cell median centering** is the most important step: off, mean F1
  0.486 and the PBMC false-positive level doubles.
- **Window 50 ≈ 100 > 200** on F1; 50 is noisier per cell. 100 stays.

### Limits

- Erythroid cells in MM199 (not in the reference) score as high as the
  tumor (0.017 vs 0.017) and cluster with it. Any normal lineage missing
  from the reference reads as aneuploid: the reference must cover the
  normal cell types present, which the modal already says.
- The remaining false-positive gap to infercnvpy on PBMC (0.55 vs 0.42)
  is the gene floor; closing it costs profile accuracy.
- Truth is pseudobulk; the SNU601 sub-clone check rests on per-chromosome
  fractions, not per-cell DNA labels. Andor et al.'s per-cell scDNA clone
  assignments would make a stricter test.
- Six datasets, two tissues of cell lines, two patient samples. Defaults
  tuned here could overfit; each change is also the one the literature
  recommends (inferCNV's own gene cutoff, clustering before denoising).

## Decisions

Defaults: `min_mean` 0.02, `resolution` 0.5, `min_reference_cells` 50,
`exclude_lineage_genes` on. Clones come from the un-thresholded matrix.
`window` 100, `step` 10, `lfc_clip` 3, `noise_threshold` 1.5,
`center_cells` on are unchanged. `CnvRequest`, `prepare_cnv`, the modal
defaults and `test_defaults_agree_between_adaptor_route_and_record` all
carry the same values.

## Not done

- Two-pass reference refinement (CopyKAT, SCEVAN; the BiB benchmark found
  it helps inferCNV in 8 of 9 cases): add the quietest non-reference cells
  to the reference and rerun. Risky on cell lines with no normal cells.
- HMM or segmentation of clone profiles for sharper breakpoints.
- Allele-aware mode (Numbat) needs BAMs.
