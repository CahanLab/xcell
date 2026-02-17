# Changelog

All notable changes to xcell are documented in this file.
Format follows [Keep a Changelog](https://keepachangelog.com/).

## [Unreleased]

### Added
- Embedding rotation and reflection controls ("Adjust" interaction mode): shift+drag to rotate, flip X/Y buttons. Transforms modify adata.obsm in place and persist in h5ad export.
- JSON import support in Import Gene Lists modal (`.json` files round-trip with export).
- Preferred embedding auto-selection on dataset load: spatial > umap > pca (case-insensitive substring match).

### Fixed
- Gene set export now includes genes from all categories and folders (previously only exported the legacy flat list, which was often empty).
