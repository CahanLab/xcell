# Changelog

All notable changes to xcell are documented in this file.
Format follows [Keep a Changelog](https://keepachangelog.com/).

## [Unreleased]

### Fixed
- Single-line gene association (gear icon in Shapes) now uses the line's projected/linked cells instead of the global cell mask, so only cells actually linked to the line are tested
- Fixed B-spline regression in line gene association: removed redundant intercept column that caused multicollinearity with the B-spline partition of unity, producing unreliable F-statistics and missing significant genes

### Added
- Support for loading prefixed 10x file trios (e.g. GSM1234_barcodes.tsv.gz, GSM1234_features.tsv.gz, GSM1234_matrix.mtx.gz) common in GEO accessions — detected automatically in the file browser and loadable like standard 10x matrix folders
- Header now shows the loaded data filename above cell/gene counts, with stacked stats layout
- GitHub icon link and Docs link in the header for quick access to the xcell repository
- Compare Cells feature in Analyze > Cell Analysis: select an .obs column, check 2+ groups, and run pairwise differential expression or marker gene analysis directly from the Analyze modal
- Lasso draw tool: draw closed freehand shapes in Draw mode (like pencil but always creates a closed shape)
- Polygon selection tool for Select mode: click vertices to define a polygon selection area, double-click to close and select cells inside
- Select button dropdown (replaces Lasso button): choose between Lasso (freehand) and Polygon (click-based) selection tools
- Adjust button dropdown: combines Rotate (formerly Adjust) and Quilt into a single toolbar dropdown

### Changed
- Renamed "Lasso" toolbar button to "Select" with tool dropdown for lasso and polygon selection
- Renamed "Lines" panel to "Shapes" with updated empty-state messages
- Renamed Adjust mode to "Rotate" within the new Adjust dropdown
- Merged Adjust and Quilt buttons into a single "Adjust" dropdown
- Cell Panel checkboxes now select cells (like lasso) instead of marking for comparison
- Removed standalone "Compare" toolbar button; comparison feature is now accessed via Analyze > Cell Analysis > Compare Cells
- Fixed Draw button dropdown arrow alignment (was appearing below the button)
- Standardized draw tool instructions across all tools to consistent "Click and drag to..." / "Click to add..., double-click to..." format
- Backend schema now includes `filename` field

### Previously Added
- Cancel button for long-running analysis operations (gene neighbors, spatial neighbors, spatial autocorrelation, contourize, line gene association). Operations run in the background and can be stopped without corrupting session data.
- Line Association: analysis parameters (spline knots, FDR threshold, max genes per module) are now configurable in the Line Tools modal before running "Find Associated Genes"
- Line Association results modal: interactive filter controls for minimum R², minimum amplitude, maximum FDR, and pattern type toggles to refine results after analysis
- Multi-line combined association analysis: check multiple lines with projected cells in the Lines panel, then run a pooled "Find Associated Genes" across tissue sections or replicates. Per-line direction reversal ensures consistent biological axis alignment.
- Cell Panel: clicking a category value to select cells now highlights that row with a teal left border, making the selection source visible

### Previously Added
- Draw tool subtypes: pencil (freehand), polygon (click vertices, double-click to close), segmented line (click points, double-click to finish), and smooth curve (Catmull-Rom spline through control points). Select via dropdown arrow next to the Draw button.
- Per-shape appearance customization: stroke color, line width, fill color (with transparency), and open/closed toggle. Accessible via the gear icon (Line Tools) on each line in the Lines panel. Changes apply immediately and persist.
- Gene subset selection across all gene-level analyses: Differential Expression, Marker Genes, Gene PCA, Build Gene Graph, Gene Neighbors, and Spatial Autocorrelation all now default to using highly variable genes (HVG) if defined, with an option to switch to all genes. This reduces multiple testing burden and makes results more interpretable.
- Pearson correlation metric for Gene Neighbors analysis
- Differential Expression modal now has customizable parameters for scanpy's `rank_genes_groups` (test method, p-value correction) and `filter_rank_genes_groups` (min fold change, max p-adj, min/max group fractions), with a "Re-run" button to adjust and re-analyze
- Load 10x CellRanger matrix folders (`filtered_feature_bc_matrix/` etc.) directly from the file browser. Folders containing `matrix.mtx(.gz)`, `barcodes.tsv(.gz)`, and `features.tsv(.gz)` or `genes.tsv(.gz)` appear as loadable items.
- Exclude Genes in Scanpy → Preprocessing: remove genes by exact name or regex pattern (e.g. `^mt-` for mitochondrial, `^Gm\d+` for predicted genes). Enter gene names (one per line) and/or comma-separated regex patterns.
- PCA variance bar chart in Scanpy → Cell Analysis → Neighbors: shows % variance explained per PC with cumulative line and elbow detection to help choose the number of PCs for kNN computation.
- Gene identifier column switching: dropdown in Gene Panel header to switch between .var columns (e.g., Ensembl IDs vs gene symbols) when alternatives are available. Gene sets and selections are automatically remapped.
- CahanLab logo in header now links to https://cahanlab.org/

### Changed
- Renamed "Scanpy" toolbar button to "Analyze"
- Preprocessing function order: Exclude Genes → Filter Cells → Filter Genes → Normalize → Log1p → HVG
- Updated defaults: Exclude Genes regex patterns (`^mt-, ^Gm\d+, ^Rps, ^Rpl`), HVG max mean (6), Gene Neighbors (basis: expression, neighbors: 10, default subset: HVG), Cluster Genes (resolution: 2.0, column: gmod)
- Compare button now resets to inactive state after returning results
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
