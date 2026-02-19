# Changelog

All notable changes to xcell are documented in this file.
Format follows [Keep a Changelog](https://keepachangelog.com/).

## [Unreleased]

### Added
- One-vs-rest marker gene analysis: click "Markers" on any categorical column in the Cell Panel to identify marker genes for each group using Wilcoxon rank-sum test. Select groups, set top N genes, and optionally apply fold-change and expression fraction filters. Results can be added as gene sets to the new "Marker Genes" category in the Gene Panel.
- Additive lasso selection: hold Shift while drawing multiple lasso shapes to select non-adjacent groups of cells without losing the previous selection.
- Embedding rotation and reflection controls ("Adjust" interaction mode): shift+drag to rotate, flip X/Y buttons. Transforms modify adata.obsm in place and persist in h5ad export.
- JSON import support in Import Gene Lists modal (`.json` files round-trip with export).
- Preferred embedding auto-selection on dataset load: spatial > umap > pca (case-insensitive substring match).

### Changed
- Replaced "Markers" button and "G1"/"G2" group buttons on categorical columns with checkboxes and a unified "Compare" button in the header toolbar. Check 2 categories for pairwise differential expression, or 3+ for one-vs-rest marker gene analysis. Checked categories are pre-selected in the Marker Genes modal.
- Renamed "Shapes" panel to "Lines" panel.
- Line tools (smoothing, gene association, projection embedding) moved from an inline section that obscured the panel into a dedicated modal, launched via a gear button on each line row.
- Lines panel is now collapsible (click the header to toggle).
- Cells (left) and Genes (right) panels are now horizontally collapsible. Collapsing either or both gives more space to the scatter plot.

### Fixed
- Gene set export now includes genes from all categories and folders (previously only exported the legacy flat list, which was often empty).
