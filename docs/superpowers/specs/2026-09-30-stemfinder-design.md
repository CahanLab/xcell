# Differentiation potency (PyStemFinder) — design

Analyze → Cells → **Differentiation (stemFinder)** scores how far each cell has
progressed through differentiation with [PyStemFinder](https://github.com/CahanLab/PyStemFinder)
(Noller & Cahan, *Brief. Bioinform.* 2024). Less differentiated cells vary more in
their expression of cell cycle genes than their neighbours do; stemFinder scores
that heterogeneity within each cell's kNN neighbourhood.

Decisions were made without a review round (the session ran on "don't ask
unless needed"); each is recorded with its reason so it can be revisited.

## What it computes

Four per-cell `.obs` columns, each a checkbox:

| Metric | PyStemFinder call | Column(s) | Orientation |
| --- | --- | --- | --- |
| stemFinder (default on) | `psf.stemfinder(markers, method, threshold)` | `stemfinder`, `stemfinder_raw` | `stemfinder` like pseudotime: lower = less differentiated |
| diffOmeter (default on) | `psf.diffometer(markers, threshold, weight_by, include_self)` | `diffometer` | higher = less differentiated |
| Cell-cycle expression (baseline) | `psf.gene_set_score(markers)` | `stemfinder_cc_mean` | higher = more proliferative |
| Expressed TFs (baseline) | `count_expressed_genes(psf.transcription_factors(species))` | `stemfinder_n_TFs` | higher = less differentiated (CytoTRACE's logic) |

The two baselines are what the R package benchmarks against; they are off by
default. `compute_performance` / `pct_recover` need an ordinal ground-truth
column and are left out (a benchmarking tool, not an exploration one).

An optional **suffix** is appended to every column so runs with different
settings can sit side by side. When the mask is a saved subset the suffix
defaults to `_<subset>`, matching the subset-scoped naming elsewhere.

## Inputs

**Markers.** Default: `psf.cell_cycle_genes(species)` (S + G2M), species from
`guess_species()` (mouse / human; C. elegans selectable). Alternatively any
Gene-panel set. Markers are matched to `var_names` case-insensitively and stored
in the dataset's spelling, so a human list finds `Mcm2` in mouse data; the
result reports how many were found and which were missing. Zero found is a 400.

**Expression.** stemFinder's `gini` wants *scaled* expression (threshold 0 then
splits each gene at its mean); `stdev` / `variance`, the cell-cycle mean and the
TF count want *log-normalized* expression. xcell prepares both from `adata.X`
(or a chosen layer) for the marker columns only:

- `layer_scale.assess_matrix_scale` says what the matrix is. Raw counts and
  linear-normalized data are normalized to 1e4 over **all** genes (PyStemFinder's
  `recipe_stemfinder` target) and `log1p`'d; log-scale data is used as is.
  Z-scored input is used as the scaled matrix and the log-normalized metrics
  are refused with a message (their values would be meaningless).
- Scaled = `sc.pp.scale(max_value=10)` of the log-normalized marker columns over
  the scoped cells. Scaling is per gene, so scaling only the markers equals
  scaling the whole matrix and reading the markers.

Binarizing for gini / diffOmeter can instead use log-normalized expression
(threshold 0 = detected), an advanced option. `weight_by='expression'` is only
offered then, since mean scaled expression is ~0.

**Neighbourhood.** Two choices:

1. *Build a kNN graph for stemFinder* (default): exact kNN distances
   (scikit-learn) on a PC embedding (`X_pca`, a PC subset, or the active
   subset's `X_pca_<name>`), `n_pcs` default min(32, available),
   **k = round(√n)** as the method prescribes, counting the cell itself as
   scanpy does. Discarded after the run — an intermediate of the score, not a
   dataset graph. `sc.pp.neighbors` was the first version: the same
   neighbours below 4,096 cells, but ~100 s at 60k cells and k = 245, three
   quarters of it UMAP connectivities stemFinder never reads; this is ~3 s.
2. *Use an existing graph*: any `obsp` `*_connectivities`; its `*_distances`
   partner is preferred (the directed kNN), otherwise the connectivities'
   sparsity. Only the sparsity pattern matters to PyStemFinder.

**Cells.** The cell mask scopes the run: scores are computed among the masked
cells only (their own scaling, graph and √n) and cells outside get NaN, like
UMAP on a subset. Cells left with no neighbours (an existing graph cut by the
mask) get NaN for every neighbourhood score — PyStemFinder would divide by
zero, or, counting the cell itself, score it 0, i.e. fully differentiated —
and the result warns when neighbourhoods are small.

**Warnings** travel with the result: an expression scale that could not be
told, markers that are highly variable genes (they shaped the PCA the graph is
built on, which damps the heterogeneity measured), small neighbourhoods.

## Shape

- `backend/xcell/stemfinder.py` — pure, the `pyscn.py` pattern: never imports
  `pystemfinder` at module scope, `availability()`, `ValueError` with install
  instructions. `match_genes`, `prepare_expression`, `build_knn`,
  `graph_from_existing`, `compute_scores`, `group_summary`.
- `DataAdaptor.stemfinder_status()` and `prepare_stemfinder(...) -> (compute_fn,
  apply_fn)`; apply writes the columns, logs `stemfinder`, returns stats and
  (optionally) the per-group medians of a chosen categorical column, sorted.
- `GET /api/stemfinder/status`, `POST /api/stemfinder` (202 + task).
- codegen `stemfinder`: `_xcell_run(xa.prepare_stemfinder(...))`, fidelity xcell.
- `StemFinderModal.tsx`, a global modal launched from Analyze → Cells; pure
  request builder in `lib/stemfinder.ts`. Refreshes schema + obs summaries and
  colours by the first written score.
- Dependency: `pystemfinder>=0.5` in `pixi.toml`'s default `pypi-dependencies`
  (its own deps — anndata, numpy, pandas, scanpy, scipy — are already there).
  The code still degrades to install instructions for environments that
  predate it.

## Tests

Backend: the pure module against PyStemFinder's documented identities
(`diffometer(include_self=False) == 2·stemfinder_raw/len(markers)`), marker
matching, count vs log-normalized preparation giving identical scores, the mask
(NaN outside, scores inside equal a run on the subset alone), the existing-graph
path, the adaptor's write-back and record, the route, and absence of the package
(status says unavailable, the route 400s with the hint). Package-dependent tests
skip when it is absent. Frontend: the request builder.
