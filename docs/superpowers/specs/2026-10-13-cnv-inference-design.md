# CNV inference and subclone calling (Analyze → Cells → CNV)

Date: 2026-10-13. Branch `feat/cnv`.

## Goal

Infer copy-number variation from expression, inferCNV-style, so tumor cells
can be separated from normal cells and split into sub-clones by their CNV
profile. Deliverables: a pure backend module, a background-task adaptor
method, a modal, a CNV heatmap figure kind, and per-cell outputs that the
existing coloring and selection tools pick up.

Scope is the expression-only method (inferCNV / infercnvpy). Allele-aware
methods (Numbat) need BAMs and are out of scope; CopyKAT's integer-ploidy
segmentation is not attempted.

## Method

Given log-normalized expression and gene positions:

1. **Expression.** `layer` or `.X`. Raw counts are normalized to 1e4 per
   cell and log1p'd; log-normalized data is used as is; z-scored data is
   refused (`layer_scale.assess_matrix_scale`).
2. **Gene positions.** Either two `.var` columns the user names
   (`chromosome`, `start`), or the shipped GENCODE table for the species
   (`data/gene_positions_{human,mouse}.tsv.gz`, built by
   `build_gene_positions.py`), matched by Ensembl ID, then exact symbol,
   then upper-cased symbol. Species comes from `guess_species` unless given.
3. **Gene filter.** Genes with a position on an included chromosome
   (default: autosomes only; X and Y excluded as inferCNVpy does, because a
   sex-mismatched reference reads as a whole-chromosome event), not an
   immunoglobulin / TCR / MHC gene (`exclude_lineage_genes`, default on),
   and with mean log expression at or above `min_mean` (default 0.02,
   lowered from 0.1 by the benchmark) across the cells being analysed.
4. **Reference.** `reference_column` + `reference_categories`: the cells in
   those categories define normal expression. With one category, each
   gene's reference mean is subtracted. With several, inferCNVpy's bounds
   rule: a value between the lowest and highest category mean becomes 0,
   otherwise the nearest bound is subtracted. Only categories with at least
   `min_reference_cells` (default 50) cells set the bounds; smaller ones
   remain reference cells, and when none qualifies all reference cells are
   pooled into one mean (with a warning). With no reference, the mean
   over all analysed cells is subtracted (works when most cells are normal
   or CNVs vary between clones; stated in the modal).
5. **Clip** to ±`lfc_clip` (default 3).
6. **Smooth** along the genome: within each chromosome, genes sorted by
   start, running mean over `window` genes (default 100) taken every `step`
   genes (default 10). A chromosome with fewer genes than `window` gives
   one window. Windows are the matrix columns.
7. **Center cells.** Subtract each cell's median across windows (inferCNV's
   step; removes residual depth effects). `center_cells`, default on.
8. **Noise filter.** Values with |x| below `noise_threshold` × the global
   standard deviation are set to 0 (default 1.5; 0 disables).
9. **Score.** `cnv_score` = mean |x| over windows per cell.
10. **Clones.** `cluster`: `leiden` (default), `hierarchical`, or `none`.
    PCA of the *un-thresholded* window matrix (`n_pcs`, default 30,
    clamped), then Leiden on an exact kNN graph (k = 15, `resolution`
    default 0.5) or Ward linkage cut at `n_clusters` (default 4). Clones are renamed by rising
    mean `cnv_score`: `clone_1` is the quietest. Per-clone summary: cell
    count, mean score, fraction of reference cells, dominant
    `reference_column` category, mean CNV per chromosome.

Computation runs in cell chunks of 2000 so a 60k × 10k matrix is never
densified at once; `report(fraction, message)` after each chunk.

## Outputs (adaptor writes, all under one `key`, default `cnv`)

| Where | What |
|---|---|
| `obsm['X_<key>']` | cells × windows, float32, NaN rows outside the cell mask |
| `obsm['X_<key>_chrom']` + `uns['xcell_score_matrices']` | cells × chromosomes mean CNV, columns = chromosome names, so the embedding can be colored by one chromosome |
| `obs['<key>_score']` | mean absolute CNV per cell |
| `obs['<key>_clone']` + `uns['<key>_clone_colors']` | categorical clone labels, `unassigned` outside the mask; absent when `cluster='none'` |
| `uns['xcell_cnv'][key]` | JSON string: windows (`chromosome`, `start`, `end`, `n_genes`), chromosomes (`name`, `start`, `size`), params, gene counts, reference, clone summary |

`_log_action('cnv', ...)` with a `codegen.REGISTRY` entry
(`_two_phase_on_selection('prepare_cnv')`).

## API

- `GET /api/cnv/status?dataset=` → species guess, whether a shipped table
  exists for it, numeric and string `.var` columns (position candidates),
  categorical `.obs` columns with their categories, layers, x_scale,
  existing CNV keys.
- `POST /api/cnv` → 202 `{task_id}`; `CnvRequest` mirrors `prepare_cnv`.
- `GET /api/cnv/results?dataset=` → the stored `uns['xcell_cnv']` records
  (for the modal's result pane and the figure input picker).

Errors: ValueError → 400 (no positions matched, too few genes, bad column,
z-scored input, reference category not in column).

## Figure kind `cnv_heatmap`

Inputs: `key`, `obs_column` (row grouping; the clone column by default;
null = ungrouped), plus the standard figure cell inputs. Params
(`config.yaml figures.cnv_heatmap`): `row_mode` (`cells` | `groups`),
`max_rows` (400; cells are binned within each group, rows allotted in
proportion to group size, at least one each), `group_order` (`cnv_score` |
`category`), `vmax` (null = 99th percentile of |x|), `colormap`
(`rdbu` | `prgn` | `roma`).

`figure_data` returns `matrix`, `row_labels`, `row_groups`,
`column_groups` (chromosomes: `name`, `start`, `size`), `n_cells`, `vmax`,
`cnv_key`. Pure assembly in `cnv.heatmap_data`. The frontend renders it on a
canvas (`components/figures/CnvHeatmapFigure.tsx`) with a diverging ramp,
chromosome separators and labels, and a group color bar; exports as PNG like
the expression heatmap.

## Frontend

- `CnvModal.tsx`, global modal gated on `isCnvModalOpen`; ScanpyModal gets a
  `cnv` entry under Cells with an "Open CNV tool…" button.
- Controls: positions source (shipped table + species, or two `.var`
  columns), reference column + category checklist, clustering method and its
  one parameter, key, advanced (window, step, min_mean, lfc_clip,
  noise_threshold, include X/Y, center_cells, layer). Scoped to the cell
  mask like stemFinder.
- Result pane: gene counts, chromosomes used, warnings, clone table; chips
  *Color by clone*, *Color by CNV score*; button *Heatmap figure…* creates
  the figure and switches to the Figures tab.
- `lib/cnv.ts` builds the request from the modal state (vitest).
- Figures: `FigureKind` union, label, schema (pinned to config), default
  title, `FiguresView` dispatch and canvas export path.
- After the run: `refreshSchema()` and `refreshObsSummaries()`.

## Tests

- `test_cnv.py`: position matching (ID, symbol, case), chromosome ordering,
  reference subtraction incl. bounds rule, window layout, chunked == whole,
  noise threshold, score, and a synthetic tumor: 3 clones with distinct
  chromosome gains/losses plus a normal reference; recovered clones must
  match the planted ones (ARI > 0.9) and the gained windows must be
  positive.
- `test_cnv_adaptor.py`: validation 400s, written keys and NaN/unassigned
  outside a mask, registry entry, action logged, codegen entry, figure
  create/data, `.var` position columns path, species-table path (skipped
  when the table is absent).
- `test_cnv_routes.py`: status, 202 submit, results.
- Frontend: `lib/cnv.test.ts`, `figureSchemas.test.ts` extended.

## Verification

Browser run on a real dataset. No local tumor dataset is known, so a
synthetic one with planted CNVs is written from a real mouse dataset's
genes (`ad_Grosswendt_demo.h5ad`) to exercise the shipped position table;
clones must appear in the heatmap and on the embedding.

## Benchmark (added 2026-10-13)

Defaults and the four method changes above (raw-matrix clustering, the
expression floor, the resolution, the reference floor, lineage genes) come
from `docs/measurements/2026-10-13-cnv-benchmark.md`.
