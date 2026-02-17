# Changelog

All notable changes to xcell are documented in this file.
Format follows [Keep a Changelog](https://keepachangelog.com/).

## [Unreleased]

### Added
- Embedding rotation and reflection controls ("Adjust" interaction mode): shift+drag to rotate, flip X/Y buttons. Transforms modify adata.obsm in place and persist in h5ad export.
- JSON import support in Import Gene Lists modal (`.json` files round-trip with export).
- Preferred embedding auto-selection on dataset load: spatial > umap > pca (case-insensitive substring match).

### Changed
- Renamed "Shapes" panel to "Lines" panel.
- Line tools (smoothing, gene association, projection embedding) moved from an inline section that obscured the panel into a dedicated modal, launched via a gear button on each line row.
- Lines panel is now collapsible (click the header to toggle).
- Cells (left) and Genes (right) panels are now horizontally collapsible. Collapsing either or both gives more space to the scatter plot.

### Fixed
- Gene set export now includes genes from all categories and folders (previously only exported the legacy flat list, which was often empty).
