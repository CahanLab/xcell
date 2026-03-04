# Changelog

All notable changes to xcell are documented in this file.
Format follows [Keep a Changelog](https://keepachangelog.com/).

## [Unreleased]

### Added
- Gene Neighbors basis selection: choose between gene PCA embeddings or raw expression as the basis for computing gene-gene neighbors. When using expression basis, optional gene subset filtering and z-score scaling are available. This lets you compare PCA-based and expression-based gene similarity results.
- Quilt mode undo: after rearranging cells in quilt mode, click the "Undo" button or press Ctrl/Cmd+Z to revert the last transform. Supports multiple levels of undo within a quilt session. The undo stack resets when exiting and re-entering quilt mode.
- Quilt mode: a new interaction mode for rearranging tissue pieces in spatial transcriptomics data. Click "Quilt" in the toolbar, lasso a group of cells, then drag to translate, shift+drag to rotate, or use flip buttons to reposition the selected patch. Press Escape to re-select, or Escape again to exit. Transforms persist in h5ad export.
- Side-by-side dual scatter plot view: click "Split" in the toolbar to view both primary and secondary datasets simultaneously. Click a plot to make it the active dataset. Each plot has independent pan/zoom, its own embedding selector, and its own legend. Interaction tools (lasso, draw, adjust) work on whichever plot you click. Gene and gene set coloring applies to both plots simultaneously.
- Embeddings created by Scanpy (PCA, UMAP) are now automatically selected for viewing, even when the dataset had no embeddings initially.
- Backend multi-dataset support: load multiple datasets into named slots (`primary`, `secondary`, etc.). All API endpoints accept an optional `?dataset=` query parameter. New `GET /datasets` and `DELETE /datasets/{slot}` endpoints for managing loaded datasets. Existing single-dataset usage is unchanged.
- Dataset slot selector in Load modal — load a second h5ad file into the Secondary slot without losing the Primary dataset.
- Dataset switcher dropdown in header toolbar (appears when two datasets are loaded) — switch between primary and secondary views seamlessly.
- Contourize spatial analysis: assign cells to spatial expression contour levels based on a gene set. Select genes in the Gene Panel, open Scanpy Modal > Spatial Analysis > Contourize, configure smoothing and contour levels, and run. Creates a new categorical obs column that can be used for coloring and downstream analysis.
- One-vs-rest marker gene analysis: click "Markers" on any categorical column in the Cell Panel to identify marker genes for each group using Wilcoxon rank-sum test. Select groups, set top N genes, and optionally apply fold-change and expression fraction filters. Results can be added as gene sets to the new "Marker Genes" category in the Gene Panel.
- Additive lasso selection: hold Shift while drawing multiple lasso shapes to select non-adjacent groups of cells without losing the previous selection.
- Embedding rotation and reflection controls ("Adjust" interaction mode): shift+drag to rotate, flip X/Y buttons. Transforms modify adata.obsm in place and persist in h5ad export.
- JSON import support in Import Gene Lists modal (`.json` files round-trip with export).
- Preferred embedding auto-selection on dataset load: spatial > umap > pca (case-insensitive substring match).

### Changed
- Loading a dataset no longer reloads the page; the store is updated in-place for seamless transitions.
- All frontend API calls now include `?dataset=` query parameter when targeting the secondary dataset slot.
- Replaced "Markers" button and "G1"/"G2" group buttons on categorical columns with checkboxes and a unified "Compare" button in the header toolbar. Check 2 categories for pairwise differential expression, or 3+ for one-vs-rest marker gene analysis. Checked categories are pre-selected in the Marker Genes modal.
- Renamed "Shapes" panel to "Lines" panel.
- Line tools (smoothing, gene association, projection embedding) moved from an inline section that obscured the panel into a dedicated modal, launched via a gear button on each line row.
- Lines panel is now collapsible (click the header to toggle).
- Cells (left) and Genes (right) panels are now horizontally collapsible. Collapsing either or both gives more space to the scatter plot.

### Fixed
- Gene set export now includes genes from all categories and folders (previously only exported the legacy flat list, which was often empty).
