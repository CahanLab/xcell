# Named cell subsets: sub-clustering without losing the dataset

Date: 2026-09-19. Status: designed and built without review — Patrick asked
for the mask bug and the sub-clustering workflow to be fixed without his
feedback. Decisions he did not make are marked **[decision]**.

## The bug

Patrick masked some cells, clustered the rest, and could never unmask them:
"the interface to achieve this in Cells does not appear".

The mask (`activeCellMask`) lived only in the browser's store. A masked run of
HVG → PCA → Neighbors → UMAP → Leiden wrote its results *over* the dataset's
own keys: `X_pca` and `X_umap` with NaN rows for the masked cells,
`connectivities` pruned to the subset, `leiden` with `unassigned` for the rest,
and the pooled `highly_variable` column replaced. A NaN coordinate crosses the
wire as `null` and deck.gl draws nothing, so once the store forgot the mask
(reload, HMR, or Reset Mask itself) the masked cells were invisible and the
only control that could have restored them — the mask bar — was gone with it.
Reproduced on the toy dataset: 500 of 1,000 `X_umap` rows NaN after one
masked chain.

## What we built

**A mask is now something you can name, and a named subset is data.**

- `create_cell_subset(name, indices)` writes `obs['subset_<name>']` (bool) and
  a registry entry in `uns['xcell_cell_subsets']`. It survives a reload,
  exports with the h5ad, and reads naturally in scanpy
  (`adata[adata.obs['subset_chondro']]`).
- The clustering chain takes `cell_subset='<name>'` and writes to keys
  suffixed with the name, in exactly the shape
  `prepare_cluster_cells_by_gene_set` already used:

  | operation | writes | leaves alone |
  |---|---|---|
  | highly_variable_genes | `var['highly_variable__<name>']` | `var['highly_variable']` |
  | pca | `obsm['X_pca_<name>']`, `varm['PCs_<name>']`, `uns['pca_<name>']` | `X_pca` |
  | neighbors | `obsp['<name>_connectivities' / '_distances']`, `uns['<name>']` | `connectivities` |
  | umap | `obsm['X_umap_<name>']` (NaN outside) | `X_umap` |
  | leiden | `obs['leiden_<name>']` (`unassigned` outside) | `leiden` |

  Each step defaults to the previous step's subset output when it exists —
  PCA to `highly_variable__<name>`, Neighbors to `X_pca_<name>`, UMAP and
  Leiden to `<name>_connectivities` — and otherwise to the dataset's own,
  sliced to the subset. So the sub-clustering workflow is: mask → name → run
  the five steps in order, each picking up the last.

- **[decision]** The output name always carries the subset's name, whatever
  graph is chosen. The browser sends the default graph explicitly as
  `connectivities`, which contributes no suffix, and the first version of the
  rule then wrote a subset's Leiden over the dataset's `leiden` — caught in the
  browser, not by the tests. Another named graph is appended
  (`leiden_chondro_spatial`); the subset's own adds nothing more.

- The ad-hoc `active_cell_indices` path is unchanged for everything else
  (normalize, log1p, filter, QC, contourize…) — those act on the cells in
  place and have no global key to protect — and is still accepted by the five
  chain operations for the analysis record and the exported notebook. The
  browser no longer uses it for them.

## The interface

- **Cells panel, mask bar:** an unsaved mask shows a name field (pre-filled
  `sub1`, `sub2`, …) and *Save as subset*; a saved one is titled
  `Cell Mask · subset <name>`. Reset Mask still clears everything.
- **Cells panel, Subsets section:** every saved subset with its cell count,
  badges for what has been computed on it (HVG · PCA · kNN · UMAP ·
  `leiden_<name>`), *Activate* (the mask comes back from the backend's
  indices — this is the "unmask" that was missing), and ✕ with a choice of
  deleting the subset only or also its results.
- **Analyze modal:** on a saved subset the banner names what the operation
  writes and what it spares; on an unsaved mask it asks for a name inline and
  saves the subset before running, so the result has somewhere scoped to go.
  The graph picker defaults to the subset's own graph, the PC source to its
  own PCA, and the output-name field is pre-filled with the suffixed name.
  Prerequisites accept the subset's own keys (`?cell_subset=`).
- **Barplot** honours the mask: the crosstab takes the indices in a POST body
  and the toolbar says `300 of 1,000 cells (mask)`.

## Two bugs found on the way

- **PCA silently intersected an explicit gene subset with `highly_variable`.**
  scanpy's `mask_var` defaults to that column whenever it exists, so
  `gene_subset='spatially_variable'` used only the genes that were also pooled
  HVGs and failed outright when none were. Fixed with `mask_var=None` on the
  explicit-subset path; the exported notebook emits the same.
- **The component clamp ignored the HVG mask.** `n_comps` was clamped against
  all genes while scanpy used only the HVGs, so a 76-gene toy with 40 HVGs
  asked arpack for 50 components. Clamped against `n_genes_used`.

## Record and notebook

Scoped steps record `cell_subset` in their params and the selection alongside;
codegen emits them through the xcell API (`xa.run_pca(..., cell_subset='chondro')`,
fidelity `xcell`) without the "runs on the whole dataset" caveat, since the
call honours the subset. `create_cell_subset` is emitted from the recorded
selection (`SELECTIONS['step_N']`).

## Out of scope

- Marker genes and the heatmap still group by an `.obs` column without a
  cell-index list (a known gap since the highlight-overlay work).
- Automatically re-activating the last subset after a reload; the Subsets
  section makes it one click and keeps the choice explicit.
- Showing which embedding is a subset embedding in the embedding picker; the
  `X_umap_<name>` name says it.
