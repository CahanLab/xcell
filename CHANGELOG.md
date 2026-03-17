# Changelog

All notable changes to xcell are documented in this file.
Format follows [Keep a Changelog](https://keepachangelog.com/).

## [Unreleased]

### Added
- Load 10x CellRanger matrix folders (`filtered_feature_bc_matrix/` etc.) directly from the file browser. Folders containing `matrix.mtx(.gz)`, `barcodes.tsv(.gz)`, and `features.tsv(.gz)` or `genes.tsv(.gz)` appear as loadable items.
- Exclude Genes in Scanpy → Preprocessing: remove genes by exact name or regex pattern (e.g. `^mt-` for mitochondrial, `^Gm\d+` for predicted genes). Enter gene names (one per line) and/or comma-separated regex patterns.
- PCA variance bar chart in Scanpy → Cell Analysis → Neighbors: shows % variance explained per PC with cumulative line and elbow detection to help choose the number of PCs for kNN computation.
- Gene identifier column switching: dropdown in Gene Panel header to switch between .var columns (e.g., Ensembl IDs vs gene symbols) when alternatives are available. Gene sets and selections are automatically remapped.
- CahanLab logo in header now links to https://cahanlab.org/

### Changed
- Load modal redesigned with Finder-inspired two-column layout: sidebar with quick-access shortcuts (Home, Desktop, Documents, Downloads) and recently loaded files, breadcrumb path navigation for clicking any ancestor directory
- Gene Panel: more compact layout with reduced padding, Browse button moved to header next to Import, empty auto-generated categories (Gene Clusters, Similar Genes, etc.) hidden until populated
- Cell Panel: lasso selection is now cleared after running a lasso-based comparison (Set as Group 1/2 → Compare)

### Fixed
- Expression coloring errors (e.g., "Gene not found") no longer block the scatter plot. The error appears as a dismissable toast that auto-clears after 5 seconds, and the color mode resets so the app remains usable.
- When a dataset is loaded with no embeddings, the message now says to use Scanpy to create embeddings, instead of the misleading "No data loaded" message.
- Cell Panel now properly refreshes when switching between datasets or loading a new dataset
- Support for loading `.rds` files containing Seurat objects. Requires R and the Seurat/SeuratDisk R packages to be installed. RDS files are automatically converted to h5ad format on load.
- Load 10x Genomics Cell Ranger `.h5` files in addition to `.h5ad` files. The file browser now shows both formats, and the Load modal accepts either.
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
