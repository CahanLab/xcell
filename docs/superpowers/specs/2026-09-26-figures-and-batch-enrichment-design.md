# Figures: a managed, reproducible home for non-interactive plots — starting with per-cluster enrichment

**Date:** 2026-09-26
**Status:** approved by delegation ("proceed without my feedback unless needed")
**Builds on:** `2026-09-26-enrichment-design.md` (merged at `7b328aa`)

## The problem

The typical loop is QC → cluster → **Compare Cells** → interpret each
cluster's gene signature. Interpretation today means reading marker tables
or running one enrichment per cluster by hand, and there is no compact
picture of "which programs distinguish which clusters". Once such a picture
exists it is a manuscript figure, and a manuscript figure has three
demands the app does not meet:

1. **Exploration then export.** Thresholds, top-N, colour scale and layout
   are tuned interactively, then the result leaves as SVG or PNG.
2. **Persistence.** The current Heatmap, Barplot and Figure tabs each hold
   one unsaved config in memory; a reload loses it and a second figure
   replaces the first.
3. **Provenance.** The only link between a figure and the analysis is a
   PNG attached to a record step by index. Nothing stored can *regenerate*
   the figure, so "how was Fig. 3C made?" has no answer six months later.

This spec adds (A) batch enrichment across the groups of a column, (B) a
**figure system** — declarative figure records stored in the AnnData,
rendered from stored results, editable, exportable, and logged in the
analysis record — with enrichment heatmap and network as the first kinds,
and (C) brings the existing composition barplot and expression heatmap
under the same system.

## Design principles

- **A figure is data + a spec, not pixels.** A figure record names its
  *inputs* (stored result keys, `.obs` columns) and its *params* (thresholds,
  ordering, colours). Rendering is a pure function of the two. Pixels are an
  export, never the source of truth.
- **Inputs are stored results.** A figure can only reference things that
  persist in `uns` under a key (enrichment results, batch collections,
  `.obs` columns). If an analysis is not stored, it cannot be a figure input
  — which is why marker-gene batch enrichment stores its results.
- **The figure's data table is a backend function.** `figure_data(id)`
  assembles the plotted matrix/graph from the stored inputs and params in
  Python, so the notebook export can reproduce the exact table with one
  `xa.figure_data(...)` call. The browser only draws.
- **One renderer per kind, in the browser (SVG).** Vector output for
  manuscripts, PNG by rasterising the same SVG. No server-side matplotlib.
- **Provenance is the analysis record.** Creating a figure is a logged
  step (`figure_create`) whose params are the full spec; editing is a
  logged step; the record's own PNG attachment mechanism is reused for
  "put this image in the notebook". A figure also stores the indices of the
  record steps that produced its inputs, resolved at creation time.
- **Reproduction path:** load an exported h5ad → the Figures tab lists the
  stored figures → each re-renders from stored inputs. The notebook carries
  the spec and the data call.

## Part A — batch enrichment across a column's groups

### Backend

`prepare_gsea_batch(obs_column, *, groups=None, reference='rest',
libraries, sets, gene_subset, method='wilcoxon', metric='score',
cell_subset=None, n_perm, min_set_size, max_set_size, weight, seed,
key=None) -> (compute_fn, apply_fn)`

- Validates synchronously exactly as `prepare_gsea` does, once per group
  (column exists, ≥ 2 groups, each side ≥ 2 cells; a group that fails the
  cell count is *skipped with a note*, not fatal, so one tiny cluster does
  not block the run). `reference='rest'` is the default; with exactly two
  groups and `reference='rest'` the second contrast is the mirror of the
  first, which is fine — the heatmap shows both columns.
- Snapshots one throwaway AnnData of the involved cells × universe once,
  and builds each group's ranking inside `compute_fn` from that copy
  (`sc.tl.rank_genes_groups` with `groups=[g], reference='rest'|other`).
  Progress: `report(i/n_groups, f'{group} ({i}/{n})')`.
- Resolves the library sets once; runs `preranked_gsea` per group with the
  same permutation seed (so identical sets get comparable nulls across
  groups; the pool is per call anyway because the ranked universe can
  differ).
- `apply_fn` stores **one result per group** under
  `gsea_<column>_<group>_vs_<reference>` (exactly the single-run shape, so
  each is a valid "previous run") and **one collection** under
  `gsea_<column>_batch` (suffix rule applies):
  ```
  {kind: 'gsea_batch', label: 'GSEA: <column> (N groups vs rest)',
   created_at, obs_column, reference, groups: [...], members: {group: key},
   skipped: {group: reason}, n_perm, universe_size, gene_subset_type,
   n_sets_tested, n_significant, params}
  ```
  and logs one step `enrichment_gsea_batch` with `{key, members,
  n_significant}`.

`run_overlap_enrichment_batch(obs_column, *, groups=None, top_n=100,
min_in_group_fraction, max_out_group_fraction, min_fold_change,
libraries, sets, gene_subset, min_set_size, max_set_size, min_overlap,
key=None) -> dict` — synchronous. Runs `run_marker_genes` (one-vs-rest,
top N per group, the existing filters) and then `run_overlap_enrichment`
for each group's marker list; stores per-group results as
`ora_<column>_<group>_markers` and a collection `ora_<column>_batch` with
`kind: 'ora_batch'`, the marker lists (`markers: {group: [genes]}`) so the
figure's input is self-contained, and logs `enrichment_ora_batch`. It
does **not** log a separate `marker_genes` step (the batch step's params
carry the marker settings).

`get_enrichment_results()` lists collections alongside single runs; the
summary gains `n_groups` for collections. `get_enrichment_result(key)` on a
collection returns it with `members` expanded (`member_results: {group:
result}`) so one GET feeds a figure or the modal.

Routes: `POST /enrichment/gsea_batch` (202), `POST /enrichment/ora_batch`
(200). Request models mirror the single-run ones plus `obs_column`,
`groups`, `reference`, and the marker filters. `gene_subset` keeps the full
union type.

Codegen: `enrichment_gsea_batch` (two-phase `prepare_gsea_batch`) and
`enrichment_ora_batch` (direct), XCELL tier.

### Frontend

- **EnrichmentModal, GSEA tab:** the group select gains a first option
  *All groups (one vs rest)*; choosing it hides the reference select and
  runs the batch. **ORA tab:** the query select gains *Marker genes of each
  group in …* with a column picker and the marker-gene filters (top N,
  min in-group fraction, max out-group fraction, min fold change, seeded
  from `config.yaml marker_genes`).
- **Batch results view:** the summary line, a group selector (chips with
  each group's `n_significant`), the existing table for the selected
  group, and two new footer buttons: **Heatmap figure…** and **Network
  figure…** which create a figure (Part B) from the collection and switch
  the centre panel to the Figures tab. *Add to gene sets* files one folder
  per group.
- **Compare Cells hand-off:** `MarkerGenesModal` and `DiffExpModal` gain an
  **Enrichment…** button that opens the modal pre-set to the batch mode for
  that column (marker modal) or to the two-group contrast (diffexp modal:
  `group` = group 1, `reference` = group 2). The ScanpyModal Compare section
  gains a third action next to the existing run: *Enrichment for these
  groups…*, which opens the modal directly in batch mode with the checked
  groups.
- `EnrichmentSource` grows: `{kind: 'gsea', obs_column?, groups?, reference?}`
  and `{kind: 'ora', markers_of?: {obs_column, groups?}}`.

## Part B — the figure system

### Figure record

```
uns['xcell_figures'][id] = json.dumps({
  id: 'fig_<n>',                 # allocated by the adaptor, never reused
  kind: 'enrichment_heatmap' | 'enrichment_network' | 'composition_barplot' | 'expression_heatmap',
  title, caption,
  created_at, updated_at,
  inputs: {...},                 # kind-specific references to stored things
  params: {...},                 # kind-specific display parameters
  provenance: {
    steps: [int, ...],           # record step indices that produced the inputs (best effort)
    created_step: int,           # index of this figure's own figure_create step
    source: <adaptor source path>,
  },
})
```

JSON strings, as for enrichment results. `xcell_figures_seq` in `uns`
holds the id counter so ids survive deletion.

### Kinds (this spec)

**`enrichment_heatmap`** — rows are gene sets, columns are groups.
- inputs: `{enrichment_keys: [collection key] | [single keys...]}`. A
  collection expands to its members; a list of single results works too
  (e.g. three hand-run contrasts), with column labels from each result's
  ranking label / query name.
- params: `value: 'nes' | 'signed_logp' | 'es' | 'fold_enrichment'`
  (`signed_logp` = −log10(padj) × sign(NES or fold enrichment > 1);
  ORA collections default to `signed_logp`, GSEA to `nes`),
  `padj_max: 0.05` (cells above it are set to 0, PySCN's rule),
  `top_n: 5` (per column, positive and negative separately; rows are the
  union), `direction: 'both' | 'up' | 'down'`, `collapse_jaccard: 0.5 |
  null` (greedy: walk sets by best |value|; a set joins an earlier
  representative when the Jaccard index of their **member genes in the
  universe** ≥ threshold; the representative's row stays, members are
  listed in the row tooltip and the data table), `row_order: 'peak' |
  'cluster' | 'input'` (`peak` = by the column of the row's largest |value|
  then by value; `cluster` = average-linkage on euclidean distance via
  scipy), `col_order: 'input' | 'cluster'`, `colormap: 'rdbu' | 'prgn' |
  'roma'` (diverging, centred on 0), `vmax: null | number` (symmetric),
  `show_values: false`, `label_max_chars: 40`, `cell_size: 14`.
- data (`figure_data`): `{rows: [{name, library, members: [names],
  n_set}], cols: [{label, key}], values: [[float]], padj: [[float]],
  value_label: 'NES', n_rows_total, n_collapsed}`.

**`enrichment_network`** — a bipartite graph of groups and gene sets.
- inputs: as above.
- params: `padj_max`, `top_n`, `direction`, `value`, `set_edge_jaccard:
  0.3 | null` (adds set–set edges when the Jaccard of member genes ≥ it),
  `layout: 'force' | 'bipartite'`, `seed: 0`, `node_size_by: 'degree' |
  'n_set' | 'constant'`, `edge_width_by: 'value' | 'constant'`,
  `label_max_chars: 30`, `colormap`.
- data: `{nodes: [{id, kind: 'group' | 'set', label, x, y, size, degree,
  library?}], edges: [{source, target, kind: 'enrichment' | 'overlap',
  value, padj?}], bounds}`. Positions come from a numpy Fruchterman–
  Reingold layout (`enrichment_figures.force_layout`, seeded, ~200
  iterations, group nodes weighted heavier) so the same spec gives the
  same picture; `bipartite` places groups on a left column and sets on the
  right, ordered to minimise crossings by barycentre.

**`composition_barplot`** (Part C) — inputs `{column_a, column_b,
cell_subset?}`; params = today's `BarplotConfig` minus the columns. Data
= today's `/obs/crosstab` computation, run through the adaptor.

**`expression_heatmap`** (Part C) — inputs `{gene_sets: [{name, genes}],
obs_column?, line_name?, cell_subset?}`; params = today's `HeatmapConfig`
display fields. Data = `heatmap.compute_heatmap_data`.

### Backend

`backend/xcell/enrichment_figures.py` (pure): `assemble_matrix(results_by_col,
*, value, padj_max, top_n, direction, collapse_jaccard, row_order,
col_order, universe_sets)` → the heatmap data; `assemble_network(...)` →
nodes/edges; `force_layout(n_nodes, edges, weights, seed, iterations)`;
`collapse_sets(sets_by_name, order, jaccard)`; `signed_logp(padj, sign)`
with a floor at 300 so `-log10(0)` never becomes inf.

Adaptor (`figures` section):
- `FIGURES_UNS_KEY = 'xcell_figures'`, `create_figure(kind, *, title=None,
  inputs, params=None) -> dict` (validates `kind` against
  `FIGURE_KINDS`, validates inputs exist — enrichment keys, obs columns,
  subset names — fills param defaults from `config.yaml figures.<kind>`,
  resolves provenance steps by scanning the record for steps whose
  `result.key` / `result.members` / `params.key_added` match the inputs,
  stores, logs `figure_create` with the full record, returns it with
  `provenance.created_step` filled in).
- `update_figure(id, *, title=None, caption=None, params=None) -> dict`
  (merges, bumps `updated_at`, logs `figure_update` with only the changed
  fields). `delete_figure(id)` logs `figure_delete`.
- `list_figures() -> [summary]` (id, kind, title, created_at, updated_at,
  inputs summary, n provenance steps), `get_figure(id)`.
- `figure_data(id, params_override=None) -> dict` — dispatches on kind;
  `params_override` lets the UI preview a parameter change before saving
  it (no logging). Enrichment kinds load the input results, resolve
  members, and call `enrichment_figures`. Part C kinds call the existing
  crosstab / heatmap computations.
- `attach_figure_to_record(id, png_b64, caption=None)` — reuses
  `analysis_record.add_figure`, links it to the figure's `created_step`,
  and stores `figure_id` on the record `Figure` (new optional field, with
  `from_dict` tolerance) so the notebook can cite the spec.

Routes: `GET /figures`, `POST /figures` (200, returns the record),
`GET /figures/{id}`, `PUT /figures/{id}`, `DELETE /figures/{id}`,
`POST /figures/{id}/data` (body: optional `params` override),
`POST /figures/{id}/attach` (png_b64, caption). Models above the routes.
Errors: unknown kind / missing input → 400, unknown id → 404.

Codegen: `figure_create` → `xa.create_figure(kind=…, title=…, inputs=…,
params=…)` plus a second line `xa.figure_data('<id>')` (exact tier —
both calls exist verbatim); `figure_update` → `xa.update_figure(...)`;
`figure_delete` → `xa.delete_figure(...)`. The notebook export's figure
payload gains `figure_id` next to the PNG so a reader can find the spec.

Config: `figures:` section with per-kind param defaults (the values above).

### Frontend

- **Centre tab "Figures"** replaces the current "Figure" tab label; the
  existing multi-panel embedding compositor stays reachable from its
  "Create figure" button and renders in the same tab as an unsaved
  workspace item labelled *Embedding panels (unsaved)* — it is not
  migrated in this spec (follow-up: make it a persisted kind).
- `FiguresView.tsx`: left rail = gallery (title, kind icon, created date,
  a provenance badge "step 12"), sorted newest first, with *New figure…*
  (kind picker → input picker: enrichment collections/results for the
  enrichment kinds; columns for the barplot; gene sets for the heatmap);
  centre = the renderer at fit-to-width with pan/zoom via CSS transform;
  right rail = **title/caption**, a **params form generated from a per-kind
  schema** (`lib/figureSchemas.ts`: field name, label, type
  number/select/bool/nullable-number, min/max/step, options, help), and a
  **Provenance** box listing the linked steps (label + index, click opens
  the Analysis record panel scrolled to it). Param edits preview
  immediately (data fetched with the override, debounced 300 ms) and a
  **Save** button writes them (`PUT`). Toolbar: **Export SVG**, **Export
  PNG** (2× by default, 1–4× select), **Attach to record** (rasterise →
  POST attach), **Duplicate**, **Delete** (confirm).
- Renderers (pure React, SVG, `data-figure-svg` on the root): 
  `EnrichmentHeatmapFigure` (cells, diverging scale legend, row labels
  with collapsed-member counts "(+3)", column labels rotated 45°,
  optional value text, hover tooltip with padj and members),
  `EnrichmentNetworkFigure` (group nodes as squares, set nodes as circles
  sized by the chosen measure, enrichment edges coloured by sign and
  widthed by |value|, overlap edges dashed grey, labels with collision
  offset, legend), `CompositionBarplotFigure` (extracted from
  `BarplotView`), `ExpressionHeatmapFigure` (Part C).
- `lib/figures.ts`: types (`FigureRecord`, `FigureSummary`, per-kind
  `Params`/`Inputs`), `svgToPng(svgEl, scale) -> Promise<Blob>`, filename
  helpers, `defaultTitle(kind, inputs)`. `lib/figureSchemas.ts`: the param
  schemas. `hooks/useData.ts`: the API client.
- Store (per dataset): `figures: FigureSummary[]`, `activeFigureId`,
  `figuresVersion` (bump to refetch). Refresh after any figure mutation
  and after an enrichment run (new inputs become available).
- **Entry points:** EnrichmentModal batch footer buttons (Part A) create a
  figure and switch to the Figures tab; *New figure…* in the tab;
  `AnalysisRecordPanel` figure rows show "open figure" when a record PNG
  carries a `figure_id`.

## Part C — the existing barplot and expression heatmap join

- **Barplot tab:** unchanged as the interactive editor; gains *Save as
  figure* (creates a `composition_barplot` figure from the current config)
  and its SVG drawing moves into `CompositionBarplotFigure` so both places
  render identically. The Figures tab renders saved barplots with the
  shared component and the generic params form (schema for `order`,
  `shareOf`, `normalize`, `minCells`, `showValues`).
- **Heatmap tab:** *Save as figure* creates an `expression_heatmap` figure
  from `heatmapConfig`. `HeatmapView`'s canvas drawing is refactored to
  take `config` and `data` as props (`ExpressionHeatmapFigure`), used by
  both the tab and the Figures tab. Export for this kind is PNG from the
  canvas at 1–4×, plus SVG for the labels/legend layer *only if* cheap —
  otherwise PNG only, stated in the toolbar.
- Both migrations reuse the existing backend computations through
  `figure_data`; nothing numerical changes.

## Non-goals

- Server-side matplotlib rendering, PDF export, multi-panel composition of
  saved figures (the embedding compositor stays as is), figure versioning
  beyond the record's edit steps, pairwise (all-vs-all) batch contrasts.

## Error handling

- Batch: a group with < 2 cells on either side is skipped and named in
  `skipped`; if every group is skipped → 400. An uncached library → 400
  before any work.
- Figures: unknown kind → 400 listing kinds; an input key that no longer
  exists (result deleted) → `figure_data` returns 409 with a message the
  renderer shows in place of the plot, and the figure stays listed so its
  spec is not lost. Param validation per schema → 400.
- Empty matrix after thresholds → data with zero rows and a `note`; the
  renderer shows the note rather than an empty SVG.

## Testing

Backend: `test_enrichment_batch.py` (per-group keys + collection; skipped
groups; two-group mirror; ORA batch stores markers; codegen entries),
`test_enrichment_figures.py` (matrix assembly against a hand-built
example: thresholds, top-N union, directions, signed_logp floor, collapse
by Jaccard with member listing, peak vs cluster ordering; network nodes/
edges/overlap edges; force layout determinism and finite coordinates;
bipartite barycentre order), `test_figures_adaptor.py` (create validates
kind/inputs, ids never reused, provenance steps resolved to the enrichment
step indices, update logs only changes, delete, figure_data for each kind
incl. override without logging, attach links figure_id, uns round-trip),
`test_figures_routes.py` (CRUD + data + 409 on a deleted input + attach).
Frontend: `figures.test.ts` (schemas cover every param the backend
accepts, default title, PNG helper input validation), `enrichment.test.ts`
additions for the batch source shape. Browser: batch GSEA on the toy
`cell_type` → heatmap figure → edit padj → save → export SVG/PNG → attach →
record shows it → reload → figure re-renders; network figure; barplot
save-as-figure; heatmap save-as-figure.
