# CNV inference benchmark

Scores `xcell.cnv` against orthogonal copy-number truth, following the
evaluation of Schmid et al., *Benchmarking scRNA-seq copy number variation
callers*, Nat Commun 2025 (doi:10.1038/s41467-025-62359-9). Results and the
decisions taken from them: `docs/measurements/2026-10-13-cnv-benchmark.md`.

Nothing here runs under pytest; the data are downloaded, not committed.

## Data (about 600 MB)

Make a directory `<bench>` with `raw/`, `truth/`, `h5ad/`, `runs/`.

| Dataset | scRNA-seq | Truth |
|---|---|---|
| SNU601, NUGC4 (gastric cancer lines, 10x) | GEO GSE142750, `GSM4238687_SNU-601_*`, `GSM4238684_NUGC-4_*` into `raw/snu601`, `raw/nugc4` | scWGS pseudobulk, Zenodo 20260649 `SNU601_wgs_results_formated.csv`, `NUGC4_...` |
| Gastric reference for the cell lines | GEO GSE150290 `GSM4546347..51_Pat25..29-A.txt.gz` into `raw/gastric_ref` | — |
| MM199 (multiple myeloma marrow, 10x) | GEO GSM4200471 both `.txt.gz` into `raw/mm` | WES, Zenodo `MM199Tumor_clean.called.seg` |
| BCC06 and the four-patient mix (basal cell carcinoma) | GEO GSE123813 `GSE123813_bcc_scRNA_counts.txt.gz`, `..._all_metadata.txt.gz` into `raw/bcc` | WES, Zenodo `BCC06_tumor_clean.called.seg`; patient identity for the mix |
| PBMC 10k (diploid control) | 10x `pbmc_10k_v3_filtered_feature_bc_matrix.h5` into `raw/pbmc` | all bins diploid |

Zenodo files: `https://zenodo.org/api/records/20260649/files/<name>/content`.

## Run

```bash
cd backend
for d in SNU601 NUGC4 MM BCC PBMC; do pixi run -e dev python -I benchmarks/cnv/load_data.py $d <bench>; done
for d in snu601 nugc4 MM199 BCC06 BCCmix PBMC; do
  pixi run -e dev python -I benchmarks/cnv/run_cnv.py <bench>/h5ad/$d.h5ad <bench>/runs mytag [--param key=value ...]
done
pixi run -e dev python -I benchmarks/cnv/evaluate.py <bench> '*__mytag'     # table + results.tsv
pixi run -e dev python -I benchmarks/cnv/sweep.py <bench> sw '[{"window":50},{"window":100}]' snu601 MM199
```

`run_cnv.py --tool infercnvpy` runs the infercnvpy baseline; install it
beside the scripts without touching the pixi environment:
`pixi run -e dev python -m pip install --target benchmarks/cnv/pylib --no-deps infercnvpy`
and add an empty `pylib/pyreadr.py` (only its SCEVAN reader needs pyreadr).

## Metrics (evaluate.py)

Profile accuracy is scored on the pseudobulk of the tumor cells spread onto
100 kb bins: Pearson with the truth's mean copy state, AUC for gains and for
losses, and the paper's **max F1** (macro F1 over loss / base / gain at the
best pair of thresholds). scWGS truth calls a state when more than half the
cells carry it; WES calls are GATK's `+`, `-`, `0`. Per cell: AUROC of
`cnv_score` for tumor against reference cells; ARI and homogeneity of clones
against patient identity in the BCC mix; for SNU601 and NUGC4 the
correlation, over bins where 20–80 % of scWGS cells carry an event, between
that DNA fraction and the fraction of RNA cells called. PBMC: RMSE of the
non-reference cells from zero, to be divided by the SNU601 tumor SD of the
same run before comparing tools, because a tool that smooths over unexpressed
genes shrinks every value.
