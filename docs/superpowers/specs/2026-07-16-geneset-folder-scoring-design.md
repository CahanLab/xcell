# Score a gene-set folder into an `.obsm` matrix

Date: 2026-07-16

## Goal

Let a user score every gene set in a gene-set folder at once, store the results
in an `.obsm` matrix, and then explore them: embed cells on two chosen scores,
color by any score, draw a line on that embedding, and run the existing line →
gene-association to find genes correlated along the axis (e.g. a
progenitor→differentiated trajectory).

Most of the downstream already exists (`create_obs_embedding`, embedding
selection, line drawing, line/gene association). The new work is: batch folder
scoring into `.obsm`, embedding from two `.obsm` columns, and color-by-score.

## User flow

1. Genes panel → a gene-set folder → **Score sets** action → opens the **Score
   Gene Sets** modal.
2. Modal fields (defaults pulled from Display settings; all editable):
   - **Per-gene normalization** across cells: `none | zscore_mad | zscore_sd | minmax | rank`
   - **Per-gene clip**: shown only when normalization is `minmax`
   - **Aggregation** across genes per cell: `mean | median | sum | max`
   - **`.obsm` slot name**: auto-filled `geneset_scores_<folder>` (sanitized), editable
   - (No UCell — the mean pipeline only.)
3. Submit → each set scored with the mean pipeline → results stacked into one
   `n_cells × K` matrix written to `adata.obsm[<slot>]`; set→column names and the
   scoring params recorded in `adata.uns`. Skipped sets (no usable genes)
   reported.
4. Success step offers **Create embedding from two scores**: pick two columns
   (e.g. `progenitorState` × `differentiatedState`) → a 2-column `.obsm`
   embedding is written and auto-selected. (A 2-set folder's matrix is itself a
   valid 2D embedding and also appears in the embedding dropdown.)
5. The embedding is colorable by every existing variable **and** by any score
   column. User draws a line and runs line/gene association (unchanged).

## Data model

- `adata.obsm[<slot>]`: `float64`, shape `n_cells × K` (K = number of scored sets
  that had usable genes).
- Column names + provenance recorded under a registry in `.uns` so unnamed
  `.obsm` columns can be resolved back to set names:
  - `adata.uns['xcell_score_matrices'][<slot>] = { "columns": [set names],
    "per_gene_norm": ..., "per_gene_clip": ..., "aggregation": ...,
    "layer": ..., "source_folder": ... }`
- The matrix participates in the normal embeddings list (any `.obsm` key that is
  2D with ≥2 columns), so no schema change is needed to make it selectable.

## Backend (adaptor + routes)

1. `score_gene_sets_matrix(sets, per_gene_norm, per_gene_clip, aggregation,
   obsm_name, layer=None, transform=None, overwrite=False)`
   - For each set: `valid = [g for g in genes if g in var.index]`, then
     `valid, n_masked = self._filter_to_visible(valid)` (identical gene-mask
     handling to `get_multi_gene_expression`). Empty → skipped (not a column).
   - Score each remaining set via `_aggregate_gene_set_scores(...)` using the
     same source-matrix resolution as coloring (`layer`/`transform` → X /
     normalized / layer). Stack kept sets → `n_cells × K` → `adata.obsm[obsm_name]`.
   - Record columns + params in `.uns['xcell_score_matrices'][obsm_name]`.
   - Collision: if `obsm_name` exists and not `overwrite` → `ValueError` (→ 400);
     with `overwrite` → replace.
   - Returns `{obsm_name, columns, n_cells, n_sets, skipped:[{name,reason}],
     per_column:{name:{min,max,n_masked_excluded}}}`.
   - Route: `POST /api/scanpy/score_gene_sets_matrix`.
2. `get_obsm_column(obsm_name, column)` → resolve column index via the `.uns`
   registry (fallback: integer index), return `{values, min, max}` for coloring.
   - Route: `POST /api/obsm/column` (or `GET /api/obsm/{name}/column/{column}`).
3. `create_obsm_embedding(obsm_name, col_x, col_y, log_axes='none', name=None)`
   → read two named columns of `adata.obsm[obsm_name]`, `np.column_stack` → write
   a 2-column `adata.obsm[X_<name>]` (mirrors `create_obs_embedding`, incl.
   `log_axes` and name-collision rules). Returns `{embedding_name, n_cells}`.
   - Route: `POST /api/scanpy/embedding_from_obsm`.
4. `/schema` gains `score_matrices: { slot: [column names] }` (from the `.uns`
   registry) so the UI can list score columns for coloring and the embed picker.

## Frontend

1. **Score Gene Sets modal** (new component): opened from a folder's "Score sets"
   action in the Genes panel. Fields as above, defaults from Display settings
   (`geneSetPerGeneNorm`, `geneSetPerGeneClip`, `geneSetAggregation`); auto slot
   name; overwrite checkbox shown if the name already exists. Submits the
   folder's sets (`{name, genes}`) + params + current display `layer`/`transform`.
2. **Embed-from-two-scores picker** in the modal's success step: two dropdowns of
   the just-created columns → `embedding_from_obsm` → refresh schema + auto-select.
3. **Color-by-score**: score columns (from `/schema.score_matrices`) are listed as
   colorable variables; selecting one fetches `get_obsm_column` and drives the
   existing continuous-color state (same rendering as gene/obs continuous color).
4. Existing coloring by obs/gene/gene-set and line drawing / association work on
   the new embedding with no change.

## Reused unchanged

Mean scoring math (`_aggregate_gene_set_scores`, `_normalize_gene_column`,
`_aggregate_across_genes`), gene-mask filter (`_filter_to_visible`), embedding
storage/selection, line drawing, line/gene association, continuous-color
rendering.

## Testing

- Backend pytest: `score_gene_sets_matrix` — matrix shape `n_cells × K`, `.uns`
  column names, per-column values equal single-set `_aggregate_gene_set_scores`;
  gene mask excludes masked genes (n_masked_excluded > 0) and a fully-masked set
  is skipped; collision without/with overwrite. `get_obsm_column` resolves by
  name. `create_obsm_embedding` writes a 2-col obsm and appears in schema
  embeddings.
- Frontend `tsc`.
- End-to-end (Playwright): define a folder with ≥2 sets → score → `.obsm` matrix
  present → create embedding from two columns → color by a third score → draw a
  line → run association.

## Out of scope (YAGNI)

UCell batch (too slow — excluded per user), UMAP/PCA on the full score matrix
(the matrix is in `.obsm`, so that's a later Scanpy neighbors/umap step — not
built now), mirroring scores into `.obs`.
