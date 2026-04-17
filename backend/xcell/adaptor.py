"""DataAdaptor class for wrapping AnnData objects.

This module provides a clean interface for accessing single-cell data,
following the adaptor pattern used in excellxgene. It abstracts away
direct AnnData access and is designed for easy integration with scanpy
analysis functions.
"""

from pathlib import Path
from typing import Any, Callable

import anndata
import numpy as np
import pandas as pd
import scanpy as sc

from .diffexp import compute_diffexp


class DataAdaptor:
    """Wraps an AnnData object and provides accessor methods.

    This class follows the adaptor pattern to:
    - Provide clean accessor methods for embeddings, metadata, etc.
    - Enable future scanpy integration for analysis features
    - Abstract away AnnData implementation details from API routes

    Attributes:
        adata: The underlying AnnData object
        filepath: Path to the loaded h5ad file
    """

    def __init__(self, filepath: str | Path):
        """Load an h5ad, 10x h5, or 10x mtx directory and initialize the adaptor.

        Args:
            filepath: Path to the .h5ad/.h5 file, 10x CellRanger matrix directory,
                      or a *_matrix.mtx(.gz) file from a prefixed file trio
        """
        self.filepath = Path(filepath)
        trio = self._find_10x_trio_files(self.filepath)
        if self.filepath.is_dir():
            self.adata = sc.read_10x_mtx(self.filepath)
            self.adata.var_names_make_unique()
        elif trio is not None:
            self._load_10x_mtx_trio(*trio)
        elif self.filepath.suffix == '.h5':
            self.adata = sc.read_10x_h5(self.filepath)
            self.adata.var_names_make_unique()
        else:
            self.adata = anndata.read_h5ad(self.filepath)
        self._normalized_adata: anndata.AnnData | None = None
        self._drawn_lines: list[dict[str, Any]] = []  # Stored lines from frontend
        self._action_history: list[dict[str, Any]] = []  # Track scanpy operations
        self._embedding_undo_stacks: dict[str, list[np.ndarray]] = {}  # Undo stacks for quilt transforms
        # Gene mask state — None means no mask is active.
        # See get_gene_mask / set_gene_mask / clear_gene_mask.
        self._gene_mask_config: dict[str, Any] | None = None
        self._visible_gene_mask: np.ndarray | None = None  # bool array, shape (n_genes,)

    @staticmethod
    def _find_10x_trio_files(filepath: Path) -> tuple[Path, Path, Path] | None:
        """Check if filepath is a prefixed 10x matrix file with companion files.

        Detects GEO-style file trios like prefix_barcodes.tsv.gz,
        prefix_features.tsv.gz, prefix_matrix.mtx.gz.

        Returns:
            (matrix, barcodes, features) paths if valid trio, else None.
        """
        import re
        m = re.match(r'^(.+)_matrix\.mtx(\.gz)?$', filepath.name)
        if not m:
            return None
        prefix = m.group(1)
        parent = filepath.parent

        barcodes = None
        for ext in ('.tsv.gz', '.tsv'):
            candidate = parent / f'{prefix}_barcodes{ext}'
            if candidate.exists():
                barcodes = candidate
                break

        features = None
        for feat in ('features', 'genes'):
            for ext in ('.tsv.gz', '.tsv'):
                candidate = parent / f'{prefix}_{feat}{ext}'
                if candidate.exists():
                    features = candidate
                    break
            if features:
                break

        if barcodes and features:
            return (filepath, barcodes, features)
        return None

    def _load_10x_mtx_trio(self, matrix: Path, barcodes: Path, features: Path) -> None:
        """Load a prefixed 10x file trio via a temp directory with symlinks."""
        import os
        import shutil
        import tempfile

        # Build standard filenames preserving .gz extension
        mtx_name = 'matrix.mtx.gz' if matrix.name.endswith('.gz') else 'matrix.mtx'
        bar_name = 'barcodes.tsv.gz' if barcodes.name.endswith('.gz') else 'barcodes.tsv'
        if '_genes.' in features.name:
            feat_name = 'genes.tsv.gz' if features.name.endswith('.gz') else 'genes.tsv'
        else:
            feat_name = 'features.tsv.gz' if features.name.endswith('.gz') else 'features.tsv'

        tmpdir = tempfile.mkdtemp(prefix='xcell_10x_')
        try:
            os.symlink(matrix, os.path.join(tmpdir, mtx_name))
            os.symlink(barcodes, os.path.join(tmpdir, bar_name))
            os.symlink(features, os.path.join(tmpdir, feat_name))
            self.adata = sc.read_10x_mtx(tmpdir)
            self.adata.var_names_make_unique()
        finally:
            shutil.rmtree(tmpdir, ignore_errors=True)

    @property
    def n_cells(self) -> int:
        """Number of cells (observations) in the dataset."""
        return self.adata.n_obs

    @property
    def normalized_adata(self) -> anndata.AnnData:
        """Get normalized and log1p-transformed version of the data.

        Lazily computes and caches a copy of the AnnData with:
        - sc.pp.normalize_total (count depth scaling)
        - sc.pp.log1p transformation

        Returns:
            AnnData object with normalized expression values
        """
        if self._normalized_adata is None:
            # Create a copy to avoid modifying original data
            self._normalized_adata = self.adata.copy()
            # Apply count depth normalization (scales each cell to same total counts)
            sc.pp.normalize_total(self._normalized_adata)
            # Apply log1p transformation: log(x + 1)
            sc.pp.log1p(self._normalized_adata)
        return self._normalized_adata

    @property
    def n_genes(self) -> int:
        """Number of genes (variables) in the dataset."""
        return self.adata.n_vars

    def get_schema(self) -> dict[str, Any]:
        """Get dataset schema including available embeddings and metadata.

        Returns:
            Dictionary containing:
            - n_cells: Number of cells
            - n_genes: Number of genes
            - embeddings: List of available embedding names from .obsm
            - obs_columns: List of cell metadata column names from .obs
            - obs_dtypes: Dictionary mapping column names to their dtypes
        """
        # Get embedding names (keys in obsm that are 2D array-likes)
        embeddings = []
        for key in self.adata.obsm.keys():
            arr = self.adata.obsm[key]
            if hasattr(arr, 'shape') and len(arr.shape) == 2 and arr.shape[1] >= 2:
                embeddings.append(key)

        # Get obs column info
        obs_columns = list(self.adata.obs.columns)
        obs_dtypes = {}
        for col in obs_columns:
            dtype = self.adata.obs[col].dtype
            if pd.api.types.is_categorical_dtype(dtype):
                obs_dtypes[col] = "category"
            elif pd.api.types.is_numeric_dtype(dtype):
                obs_dtypes[col] = "numeric"
            else:
                obs_dtypes[col] = "string"

        return {
            "n_cells": self.n_cells,
            "n_genes": self.n_genes,
            "n_genes_visible": (
                int(self._visible_gene_mask.sum())
                if self._visible_gene_mask is not None
                else self.n_genes
            ),
            "embeddings": embeddings,
            "obs_columns": obs_columns,
            "obs_dtypes": obs_dtypes,
            "filename": self.filepath.name,
        }

    def get_embedding(self, name: str) -> dict[str, Any]:
        """Get embedding coordinates by name.

        Args:
            name: Name of the embedding (e.g., 'X_umap', 'X_pca')

        Returns:
            Dictionary containing:
            - name: The embedding name
            - coordinates: List of [x, y] coordinate pairs

        Raises:
            KeyError: If embedding name not found in .obsm
        """
        if name not in self.adata.obsm:
            raise KeyError(f"Embedding '{name}' not found. Available: {list(self.adata.obsm.keys())}")

        coords = self.adata.obsm[name]

        # Take first 2 dimensions for visualization
        coords_2d = coords[:, :2]

        # Convert to list for JSON serialization
        return {
            "name": name,
            "coordinates": coords_2d.tolist(),
        }

    def transform_embedding(
        self,
        name: str,
        rotation_degrees: float = 0,
        reflect_x: bool = False,
        reflect_y: bool = False,
        cell_indices: list[int] | None = None,
        translate_x: float = 0.0,
        translate_y: float = 0.0,
    ) -> dict[str, Any]:
        """Apply rotation, reflection, and/or translation to an embedding in-place.

        Transforms are applied around the centroid (of the subset if cell_indices
        is provided, otherwise of all cells): reflections first, then rotation,
        then translation.

        Args:
            name: Name of the embedding in .obsm
            rotation_degrees: Counter-clockwise rotation angle in degrees
            reflect_x: If True, negate y-coordinates (reflect about x-axis)
            reflect_y: If True, negate x-coordinates (reflect about y-axis)
            cell_indices: Optional list of cell indices to transform (subset mode).
                          If None, all cells are transformed.
            translate_x: Translation offset along x-axis
            translate_y: Translation offset along y-axis

        Returns:
            Updated embedding dict (same format as get_embedding)
        """
        if name not in self.adata.obsm:
            raise KeyError(f"Embedding '{name}' not found. Available: {list(self.adata.obsm.keys())}")

        if cell_indices is not None:
            # Snapshot for undo before quilt transform
            stack = self._embedding_undo_stacks.setdefault(name, [])
            stack.append(np.array(self.adata.obsm[name][:, :2], copy=True))

            # Subset mode: only transform specified cells
            idx = np.array(cell_indices, dtype=int)
            coords = np.array(self.adata.obsm[name][idx, :2], dtype=np.float64)
            centroid = coords.mean(axis=0)

            coords -= centroid

            if reflect_y:
                coords[:, 0] *= -1
            if reflect_x:
                coords[:, 1] *= -1

            if rotation_degrees != 0:
                theta = np.radians(rotation_degrees)
                cos_t, sin_t = np.cos(theta), np.sin(theta)
                rot = np.array([[cos_t, -sin_t], [sin_t, cos_t]])
                coords = coords @ rot.T

            coords += centroid

            # Apply translation
            coords[:, 0] += translate_x
            coords[:, 1] += translate_y

            # Write back only the subset
            self.adata.obsm[name][idx, :2] = coords
        else:
            # Full-embedding mode (original behavior)
            coords = np.array(self.adata.obsm[name][:, :2], dtype=np.float64)
            centroid = coords.mean(axis=0)

            coords -= centroid

            if reflect_y:
                coords[:, 0] *= -1
            if reflect_x:
                coords[:, 1] *= -1

            if rotation_degrees != 0:
                theta = np.radians(rotation_degrees)
                cos_t, sin_t = np.cos(theta), np.sin(theta)
                rot = np.array([[cos_t, -sin_t], [sin_t, cos_t]])
                coords = coords @ rot.T

            coords += centroid

            # Apply translation (for full embedding too, though less common)
            if translate_x != 0 or translate_y != 0:
                coords[:, 0] += translate_x
                coords[:, 1] += translate_y

            self.adata.obsm[name][:, :2] = coords

        # Clear normalized cache (it may share obsm references)
        self._normalized_adata = None

        result = self.get_embedding(name)
        result["undo_depth"] = len(self._embedding_undo_stacks.get(name, []))
        return result

    def undo_transform_embedding(self, name: str) -> dict[str, Any]:
        """Undo the last quilt transform for an embedding.

        Pops the most recent snapshot from the undo stack and restores the
        embedding coordinates.

        Args:
            name: Name of the embedding in .obsm

        Returns:
            Updated embedding dict with undo_depth
        """
        stack = self._embedding_undo_stacks.get(name, [])
        if not stack:
            raise ValueError(f"No undo history for embedding '{name}'")
        coords = stack.pop()
        self.adata.obsm[name][:, :2] = coords
        self._normalized_adata = None
        result = self.get_embedding(name)
        result["undo_depth"] = len(stack)
        return result

    def get_obs_column(self, name: str) -> dict[str, Any]:
        """Get cell metadata column values.

        Args:
            name: Name of the column in .obs

        Returns:
            Dictionary containing:
            - name: The column name
            - values: List of values for each cell
            - dtype: Data type ('category', 'numeric', or 'string')
            - categories: List of category names (only for categorical columns)

        Raises:
            KeyError: If column name not found in .obs
        """
        if name not in self.adata.obs.columns:
            raise KeyError(f"Column '{name}' not found. Available: {list(self.adata.obs.columns)}")

        series = self.adata.obs[name]
        dtype = series.dtype

        result: dict[str, Any] = {
            "name": name,
        }

        if pd.api.types.is_categorical_dtype(dtype):
            result["dtype"] = "category"
            result["values"] = series.cat.codes.tolist()
            result["categories"] = series.cat.categories.tolist()
        elif pd.api.types.is_numeric_dtype(dtype):
            result["dtype"] = "numeric"
            # Handle NaN values by converting to None
            values = series.tolist()
            result["values"] = [None if pd.isna(v) else v for v in values]
        else:
            result["dtype"] = "string"
            result["values"] = series.astype(str).tolist()

        return result

    def get_cell_indices(self) -> list[str]:
        """Get cell index names/barcodes.

        Returns:
            List of cell identifiers from .obs.index
        """
        return self.adata.obs.index.tolist()

    def get_gene_names(self) -> list[str]:
        """Get gene names.

        Returns:
            List of gene names from .var.index
        """
        return self.adata.var.index.tolist()

    def get_var_identifier_columns(self) -> dict[str, Any]:
        """Get .var columns that could serve as gene identifiers.

        Returns columns with string/object dtype and >90% unique values.
        Includes '_index' representing the current index.

        Returns:
            Dictionary with 'columns' (list of column names including '_index')
            and 'current' (name of the current index, or '_index' if unnamed).
        """
        candidates = ['_index']  # Always include current index
        n_genes = self.adata.n_vars
        if n_genes == 0:
            index_name = self.adata.var.index.name or '_index'
            return {'columns': candidates, 'current': index_name if index_name != '_index' else '_index'}

        for col in self.adata.var.columns:
            series = self.adata.var[col]
            # Must be string/object dtype (including categorical with string categories)
            if hasattr(series, 'cat'):
                if not pd.api.types.is_string_dtype(series.cat.categories):
                    continue
            elif series.dtype not in ('object', 'string', 'str'):
                if not pd.api.types.is_string_dtype(series):
                    continue
            # Skip boolean-like columns
            unique_vals = series.dropna().unique()
            if len(unique_vals) <= 2 and set(str(v).lower() for v in unique_vals).issubset({'true', 'false', '0', '1', 'yes', 'no'}):
                continue
            # Must have >90% unique values
            n_unique = series.nunique()
            if n_unique / n_genes > 0.9:
                candidates.append(col)

        current = self.adata.var.index.name or '_index'
        return {'columns': candidates, 'current': current}

    def swap_var_index(self, column_name: str) -> dict[str, Any]:
        """Swap the .var index with another column.

        Moves the current index into a .var column and promotes the
        specified column to the index. Clears expression caches.

        Args:
            column_name: Name of the .var column to use as the new index.

        Returns:
            Updated schema dict (same format as get_schema()).

        Raises:
            KeyError: If column_name is not in .var columns.
        """
        if column_name not in self.adata.var.columns:
            raise KeyError(f"Column '{column_name}' not found in .var")

        # Save current index as a column
        old_index_name = self.adata.var.index.name or '_prev_index'
        # Avoid collision if column already exists
        save_name = old_index_name
        if save_name in self.adata.var.columns:
            save_name = f"{save_name}_orig"
        self.adata.var[save_name] = self.adata.var.index

        # Set new index
        self.adata.var.index = self.adata.var[column_name].values
        self.adata.var.index.name = column_name
        # Remove the column (it's now the index)
        self.adata.var.drop(columns=[column_name], inplace=True)

        # Handle duplicates
        self.adata.var_names_make_unique()

        # Clear caches
        self._normalized_adata = None

        # Regenerate the visible-gene mask since .var axis may have changed.
        # If referenced columns no longer exist, the mask is cleared.
        self._regenerate_gene_mask_after_var_change()

        return self.get_schema()

    def search_genes(self, query: str, limit: int = 20) -> list[str]:
        """Search for genes by name prefix.

        Args:
            query: Search query (case-insensitive prefix match)
            limit: Maximum number of results to return

        Returns:
            List of matching gene names (restricted to visible genes when
            a gene mask is active)
        """
        query_lower = query.lower()
        gene_names = self.get_visible_gene_names()

        # Find genes that start with the query (case-insensitive)
        matches = [g for g in gene_names if g.lower().startswith(query_lower)]

        # If not enough prefix matches, also include substring matches
        if len(matches) < limit:
            substring_matches = [
                g for g in gene_names
                if query_lower in g.lower() and g not in matches
            ]
            matches.extend(substring_matches)

        return matches[:limit]

    def get_expression(self, gene: str, transform: str | None = None) -> dict[str, Any]:
        """Get expression values for a single gene across all cells.

        Args:
            gene: Gene name
            transform: Optional transformation to apply. Supported values:
                - None: Raw expression values
                - "log1p": Apply normalize_total followed by log1p transformation

        Returns:
            Dictionary containing:
            - gene: The gene name
            - values: List of expression values for each cell
            - min: Minimum expression value
            - max: Maximum expression value
            - transform: The transformation applied (if any)

        Raises:
            KeyError: If gene not found in .var
        """
        if gene not in self.adata.var.index:
            raise KeyError(f"Gene '{gene}' not found in dataset")

        # Get gene index
        gene_idx = self.adata.var.index.get_loc(gene)

        # Select data source based on transform
        if transform == "log1p":
            adata_source = self.normalized_adata
        else:
            adata_source = self.adata

        # Get expression values from X matrix
        # Handle both dense and sparse matrices
        X = adata_source.X
        if hasattr(X, 'toarray'):
            # Sparse matrix
            values = X[:, gene_idx].toarray().flatten()
        else:
            # Dense matrix
            values = X[:, gene_idx].flatten()

        # Convert to regular Python floats and handle NaN
        values_list = []
        for v in values:
            if np.isnan(v):
                values_list.append(None)
            else:
                values_list.append(float(v))

        # Calculate min/max excluding None values
        valid_values = [v for v in values_list if v is not None]
        min_val = min(valid_values) if valid_values else 0
        max_val = max(valid_values) if valid_values else 0

        result = {
            "gene": gene,
            "values": values_list,
            "min": min_val,
            "max": max_val,
        }
        if transform:
            result["transform"] = transform
        return result

    def get_multi_gene_expression(
        self,
        genes: list[str],
        transform: str | None = None,
        scoring_method: str = 'mean',
        clip_percentile: float = 1.0,
    ) -> dict[str, Any]:
        """Get aggregated expression values for multiple genes across all cells.

        Args:
            genes: List of gene names
            transform: Optional transformation to apply. Supported values:
                - None: Raw expression values
                - "log1p": Apply normalize_total followed by log1p transformation
            scoring_method: How to aggregate expression across genes:
                - "mean": Simple average of expression values (default)
                - "zscore": Mean-center each gene, scale by MAD, then average
            clip_percentile: For zscore method, percentile for symmetric clipping (default 1.0)

        Returns:
            Dictionary containing:
            - genes: List of gene names used
            - values: List of aggregated expression values for each cell
            - min: Minimum value
            - max: Maximum value
            - transform: The transformation applied (if any)
            - scoring_method: The scoring method used

        """
        # Filter to only genes present in the dataset (silently skip missing),
        # then drop any genes currently masked by the gene mask.
        valid_genes = [g for g in genes if g in self.adata.var.index]
        valid_genes, n_masked_excluded = self._filter_to_visible(valid_genes)

        if len(valid_genes) == 0:
            return {
                "genes": [],
                "values": [0.0] * self.n_cells,
                "min": 0.0,
                "max": 0.0,
                "n_masked_excluded": n_masked_excluded,
            }

        genes = valid_genes

        # Select data source based on transform
        if transform == "log1p":
            adata_source = self.normalized_adata
        else:
            adata_source = self.adata

        if scoring_method == 'zscore':
            # Z-score method: mean-center each gene, scale by MAD, average
            scaled_arrays = []

            for gene in genes:
                gene_idx = adata_source.var.index.get_loc(gene)
                X = adata_source.X

                if hasattr(X, 'toarray'):
                    values = X[:, gene_idx].toarray().flatten().astype(np.float64)
                else:
                    values = X[:, gene_idx].flatten().astype(np.float64)

                # Mean-center the gene
                gene_mean = np.nanmean(values)
                centered = values - gene_mean

                # Scale by MAD (median absolute deviation) for robustness
                mad = np.nanmedian(np.abs(centered - np.nanmedian(centered)))
                if mad > 0:
                    scaled = centered / (mad * 1.4826)
                else:
                    sd = np.nanstd(values)
                    if sd > 0:
                        scaled = centered / sd
                    else:
                        scaled = np.zeros_like(values)

                scaled_arrays.append(scaled)

            # Stack and compute mean across genes
            stacked = np.vstack(scaled_arrays).T
            mean_scores = np.nanmean(stacked, axis=1)

            # Clip extreme values symmetrically
            valid_scores = mean_scores[~np.isnan(mean_scores)]
            lo = np.percentile(valid_scores, clip_percentile)
            hi = np.percentile(valid_scores, 100 - clip_percentile)
            clipped = np.clip(mean_scores, lo, hi)

            # Convert to Python floats (keep the z-score scale, don't normalize to 0-1)
            values_list = [float(v) if not np.isnan(v) else 0.0 for v in clipped]
            min_val = float(lo)
            max_val = float(hi)

        else:
            # Mean method: simple average
            gene_indices = [adata_source.var.index.get_loc(g) for g in genes]
            X = adata_source.X
            if hasattr(X, 'toarray'):
                expr_matrix = X[:, gene_indices].toarray()
            else:
                expr_matrix = X[:, gene_indices]

            mean_expr = np.nanmean(expr_matrix, axis=1)

            values_list = []
            for v in mean_expr:
                if np.isnan(v):
                    values_list.append(None)
                else:
                    values_list.append(float(v))

            valid_values = [v for v in values_list if v is not None]
            min_val = min(valid_values) if valid_values else 0
            max_val = max(valid_values) if valid_values else 0

        result = {
            "genes": genes,
            "values": values_list,
            "min": min_val,
            "max": max_val,
            "scoring_method": scoring_method,
            "n_masked_excluded": n_masked_excluded,
        }
        if transform:
            result["transform"] = transform
        return result

    def get_bivariate_expression(
        self,
        genes1: list[str],
        genes2: list[str],
        transform: str | None = None,
        scoring_method: str = 'zscore',
        clip_percentile: float = 1.0,
    ) -> dict[str, Any]:
        """Get normalized expression values for two gene sets for bivariate coloring.

        Args:
            genes1: List of gene names for the first set (maps to red/x-axis)
            genes2: List of gene names for the second set (maps to blue/y-axis)
            transform: Optional transformation ('log1p' for normalize_total + log1p)
            scoring_method: How to aggregate expression across genes:
                - "mean": Simple average, then min-max normalize to [0,1]
                - "zscore": Mean-center each gene, scale by MAD, average, then normalize to [0,1]
            clip_percentile: Percentile for symmetric clipping (default 1.0 = clip at 1st/99th)

        Returns:
            Dictionary containing:
            - genes1: List of gene names for set 1
            - genes2: List of gene names for set 2
            - values1: Normalized [0,1] expression values for gene set 1
            - values2: Normalized [0,1] expression values for gene set 2
            - transform: The transformation applied (if any)
            - scoring_method: The scoring method used

        """
        # Filter to only genes present in the dataset (silently skip missing)
        genes1 = [g for g in genes1 if g in self.adata.var.index]
        genes2 = [g for g in genes2 if g in self.adata.var.index]

        if len(genes1) == 0 or len(genes2) == 0:
            raise ValueError("No valid genes found in one or both gene sets after filtering to dataset genes")

        # Select data source based on transform
        if transform == "log1p":
            adata_source = self.normalized_adata
        else:
            adata_source = self.adata

        def summarize_geneset_zscore(genes: list[str]) -> list[float]:
            """Summarize expression using z-score method (mean-centered, MAD-scaled)."""
            scaled_arrays = []

            for gene in genes:
                gene_idx = adata_source.var.index.get_loc(gene)
                X = adata_source.X

                if hasattr(X, 'toarray'):
                    values = X[:, gene_idx].toarray().flatten().astype(np.float64)
                else:
                    values = X[:, gene_idx].flatten().astype(np.float64)

                # Mean-center the gene
                gene_mean = np.nanmean(values)
                centered = values - gene_mean

                # Scale by MAD for robustness
                mad = np.nanmedian(np.abs(centered - np.nanmedian(centered)))
                if mad > 0:
                    scaled = centered / (mad * 1.4826)
                else:
                    sd = np.nanstd(values)
                    if sd > 0:
                        scaled = centered / sd
                    else:
                        scaled = np.zeros_like(values)

                scaled_arrays.append(scaled)

            # Stack and compute mean across genes
            stacked = np.vstack(scaled_arrays).T
            mean_scores = np.nanmean(stacked, axis=1)

            # Clip extreme values symmetrically
            valid_scores = mean_scores[~np.isnan(mean_scores)]
            lo = np.percentile(valid_scores, clip_percentile)
            hi = np.percentile(valid_scores, 100 - clip_percentile)
            clipped = np.clip(mean_scores, lo, hi)

            # Rescale to [0, 1]
            if hi > lo:
                normalized = (clipped - lo) / (hi - lo)
            else:
                normalized = np.full_like(clipped, 0.5)

            return [float(v) if not np.isnan(v) else 0.5 for v in normalized]

        def summarize_geneset_mean(genes: list[str]) -> list[float]:
            """Summarize expression using simple mean, then min-max normalize."""
            gene_indices = [adata_source.var.index.get_loc(g) for g in genes]
            X = adata_source.X

            if hasattr(X, 'toarray'):
                expr_matrix = X[:, gene_indices].toarray()
            else:
                expr_matrix = X[:, gene_indices]

            mean_expr = np.nanmean(expr_matrix, axis=1)

            # Clip to percentiles
            valid_values = mean_expr[~np.isnan(mean_expr)]
            lo = np.percentile(valid_values, clip_percentile)
            hi = np.percentile(valid_values, 100 - clip_percentile)
            clipped = np.clip(mean_expr, lo, hi)

            # Min-max normalize to [0, 1]
            if hi > lo:
                normalized = (clipped - lo) / (hi - lo)
            else:
                normalized = np.full_like(clipped, 0.5)

            return [float(v) if not np.isnan(v) else 0.5 for v in normalized]

        # Choose summarization function based on scoring method
        if scoring_method == 'zscore':
            summarize = summarize_geneset_zscore
        else:
            summarize = summarize_geneset_mean

        values1 = summarize(genes1)
        values2 = summarize(genes2)

        result = {
            "genes1": genes1,
            "genes2": genes2,
            "values1": values1,
            "values2": values2,
            "scoring_method": scoring_method,
        }
        if transform:
            result["transform"] = transform

        return result

    def get_obs_column_summary(self, name: str) -> dict[str, Any]:
        """Get summary statistics for a cell metadata column.

        For categorical columns: returns categories with cell counts.
        For numeric columns: returns min, max, mean.

        Args:
            name: Name of the column in .obs

        Returns:
            Dictionary containing:
            - name: The column name
            - dtype: Data type ('category', 'numeric', or 'string')
            - For categorical: categories (list of {value, count} objects)
            - For numeric: min, max, mean

        Raises:
            KeyError: If column name not found in .obs
        """
        if name not in self.adata.obs.columns:
            raise KeyError(f"Column '{name}' not found. Available: {list(self.adata.obs.columns)}")

        series = self.adata.obs[name]
        dtype = series.dtype

        result: dict[str, Any] = {
            "name": name,
        }

        if pd.api.types.is_categorical_dtype(dtype):
            result["dtype"] = "category"
            # Get value counts
            value_counts = series.value_counts()
            result["categories"] = [
                {"value": str(val), "count": int(count)}
                for val, count in value_counts.items()
            ]
        elif pd.api.types.is_numeric_dtype(dtype):
            result["dtype"] = "numeric"
            result["min"] = float(series.min()) if not pd.isna(series.min()) else None
            result["max"] = float(series.max()) if not pd.isna(series.max()) else None
            result["mean"] = float(series.mean()) if not pd.isna(series.mean()) else None
        else:
            result["dtype"] = "string"
            # For string columns, get unique values with counts
            value_counts = series.value_counts()
            result["categories"] = [
                {"value": str(val), "count": int(count)}
                for val, count in value_counts.head(50).items()  # Limit to 50 for strings
            ]

        return result

    def get_all_obs_summaries(self) -> list[dict[str, Any]]:
        """Get summary statistics for all cell metadata columns.

        Returns:
            List of summary dictionaries for each obs column.
        """
        summaries = []
        for col in self.adata.obs.columns:
            try:
                summary = self.get_obs_column_summary(col)
                summaries.append(summary)
            except Exception:
                # Skip columns that fail
                pass
        return summaries

    # =========================================================================
    # User annotation methods
    # =========================================================================

    def create_annotation(self, name: str, default_value: str = "unassigned") -> dict[str, Any]:
        """Create a new categorical annotation column.

        Args:
            name: Name of the new annotation column
            default_value: Default value for all cells

        Returns:
            Dictionary with the new column summary

        Raises:
            ValueError: If column already exists
        """
        if name in self.adata.obs.columns:
            raise ValueError(f"Annotation '{name}' already exists")

        # Create categorical column with default value
        self.adata.obs[name] = pd.Categorical(
            [default_value] * self.n_cells,
            categories=[default_value]
        )

        return self.get_obs_column_summary(name)

    def add_label_to_annotation(self, annotation: str, label: str) -> dict[str, Any]:
        """Add a new label/category to an existing annotation column.

        Args:
            annotation: Name of the annotation column
            label: New label to add

        Returns:
            Updated column summary

        Raises:
            KeyError: If annotation doesn't exist
        """
        if annotation not in self.adata.obs.columns:
            raise KeyError(f"Annotation '{annotation}' not found")

        series = self.adata.obs[annotation]
        if not pd.api.types.is_categorical_dtype(series):
            raise ValueError(f"Annotation '{annotation}' is not categorical")

        # Add new category if it doesn't exist
        if label not in series.cat.categories:
            self.adata.obs[annotation] = series.cat.add_categories([label])

        return self.get_obs_column_summary(annotation)

    def label_cells(
        self, annotation: str, label: str, cell_indices: list[int]
    ) -> dict[str, Any]:
        """Assign a label to specific cells in an annotation column.

        Args:
            annotation: Name of the annotation column
            label: Label to assign
            cell_indices: List of cell indices to label

        Returns:
            Updated column summary

        Raises:
            KeyError: If annotation doesn't exist
        """
        if annotation not in self.adata.obs.columns:
            raise KeyError(f"Annotation '{annotation}' not found")

        series = self.adata.obs[annotation]
        if not pd.api.types.is_categorical_dtype(series):
            raise ValueError(f"Annotation '{annotation}' is not categorical")

        # Add label as category if needed
        if label not in series.cat.categories:
            self.adata.obs[annotation] = series.cat.add_categories([label])

        # Assign label to specified cells
        self.adata.obs.iloc[cell_indices, self.adata.obs.columns.get_loc(annotation)] = label

        return self.get_obs_column_summary(annotation)

    def delete_annotation(self, name: str) -> None:
        """Delete an annotation column.

        Args:
            name: Name of the annotation column to delete

        Raises:
            KeyError: If annotation doesn't exist
        """
        if name not in self.adata.obs.columns:
            raise KeyError(f"Annotation '{name}' not found")

        self.adata.obs.drop(columns=[name], inplace=True)

    def get_user_annotations(self) -> list[str]:
        """Get list of user-created annotation columns.

        For now, returns all categorical columns. In the future,
        could track which columns were created by users.

        Returns:
            List of annotation column names
        """
        return [
            col for col in self.adata.obs.columns
            if pd.api.types.is_categorical_dtype(self.adata.obs[col])
        ]

    def export_annotations(self, columns: list[str] | None = None) -> str:
        """Export cell annotations as TSV string.

        Args:
            columns: List of column names to export. If None, exports all.

        Returns:
            TSV-formatted string with cell indices and annotation values
        """
        if columns is None:
            df = self.adata.obs.copy()
        else:
            # Validate columns exist
            missing = [c for c in columns if c not in self.adata.obs.columns]
            if missing:
                raise KeyError(f"Columns not found: {missing}")
            df = self.adata.obs[columns].copy()

        return df.to_csv(sep="\t")

    # =========================================================================
    # Differential expression analysis
    # =========================================================================

    def run_diffexp(
        self,
        group1_indices: list[int],
        group2_indices: list[int],
        top_n: int = 10,
        method: str = "wilcoxon",
        corr_method: str = "benjamini-hochberg",
        min_fold_change: float | None = None,
        min_in_group_fraction: float | None = None,
        max_out_group_fraction: float | None = None,
        max_pval_adj: float | None = None,
        gene_subset: str | list[str] | dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Run differential expression analysis between two cell groups.

        Uses scanpy's rank_genes_groups with Wilcoxon rank-sum test.

        Args:
            group1_indices: Cell indices for group 1
            group2_indices: Cell indices for group 2
            top_n: Number of top genes to return for each direction
            method: Statistical method for rank_genes_groups
            corr_method: P-value correction method
            min_fold_change: Minimum fold change for filtering
            min_in_group_fraction: Minimum fraction of cells in group expressing gene
            max_out_group_fraction: Maximum fraction of cells outside group expressing gene
            max_pval_adj: Maximum adjusted p-value for filtering
            gene_subset: Gene filtering specification (str column name, list of genes, or dict spec)

        Returns:
            Dictionary containing:
            - positive: Top N genes upregulated in group1
            - negative: Top N genes upregulated in group2
            - group1_count: Number of cells in group 1
            - group2_count: Number of cells in group 2

        Raises:
            ValueError: If indices are invalid or groups too small
        """
        # Validate indices
        max_idx = self.n_cells - 1
        for idx in group1_indices:
            if idx < 0 or idx > max_idx:
                raise ValueError(f"Invalid cell index: {idx}")
        for idx in group2_indices:
            if idx < 0 or idx > max_idx:
                raise ValueError(f"Invalid cell index: {idx}")

        # Check for overlap
        set1 = set(group1_indices)
        set2 = set(group2_indices)
        overlap = set1 & set2
        if overlap:
            raise ValueError(f"Groups have {len(overlap)} overlapping cells")

        # Resolve gene subset
        if gene_subset is not None:
            gene_mask, subset_type, _ = self._resolve_gene_mask(gene_subset)
            work_adata = self.adata[:, gene_mask].copy()
        else:
            work_adata = self.adata
            subset_type = 'all'

        result = compute_diffexp(
            adata=work_adata,
            group1_indices=group1_indices,
            group2_indices=group2_indices,
            top_n=top_n,
            method=method,
            corr_method=corr_method,
            min_fold_change=min_fold_change,
            min_in_group_fraction=min_in_group_fraction,
            max_out_group_fraction=max_out_group_fraction,
            max_pval_adj=max_pval_adj,
        )
        result['gene_subset_type'] = subset_type
        result['n_genes_tested'] = work_adata.n_vars
        return result

    # =========================================================================
    # Drawn lines / trajectory methods
    # =========================================================================

    def set_lines(self, lines: list[dict[str, Any]]) -> None:
        """Store drawn lines from the frontend.

        Args:
            lines: List of line objects with keys:
                - name: Line name
                - embeddingName: Which embedding this was drawn on
                - points: Raw line points [[x, y], ...]
                - smoothedPoints: Smoothed line points (optional)
        """
        self._drawn_lines = lines

    def get_lines(self) -> list[dict[str, Any]]:
        """Get stored drawn lines."""
        return self._drawn_lines

    def _project_cells_onto_line(
        self,
        line_points: list[list[float]],
        embedding_coords: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Project all cells onto a line and compute position/distance.

        Uses the smoothed line if available, otherwise raw points.
        For each cell, finds the closest point on the polyline and computes:
        - Position along line (0 = start, 1 = end, normalized by arc length)
        - Perpendicular distance to the line

        Args:
            line_points: List of [x, y] points defining the line
            embedding_coords: Embedding coordinates for all cells (n_cells x 2)

        Returns:
            Tuple of (positions, distances) arrays, each of shape (n_cells,)
        """
        if len(line_points) < 2:
            return np.zeros(len(embedding_coords)), np.full(len(embedding_coords), np.nan)

        line_pts = np.array(line_points)
        n_cells = len(embedding_coords)

        # Compute cumulative arc length along line
        segment_lengths = np.sqrt(np.sum(np.diff(line_pts, axis=0) ** 2, axis=1))
        cumulative_lengths = np.concatenate([[0], np.cumsum(segment_lengths)])
        total_length = cumulative_lengths[-1]

        positions = np.zeros(n_cells)
        distances = np.zeros(n_cells)

        for i, cell_pt in enumerate(embedding_coords):
            best_dist = np.inf
            best_pos = 0.0

            # Check each line segment
            for j in range(len(line_pts) - 1):
                p1 = line_pts[j]
                p2 = line_pts[j + 1]
                seg_vec = p2 - p1
                seg_len_sq = np.dot(seg_vec, seg_vec)

                if seg_len_sq < 1e-12:
                    # Degenerate segment
                    t = 0.0
                else:
                    # Project point onto segment
                    t = max(0, min(1, np.dot(cell_pt - p1, seg_vec) / seg_len_sq))

                # Closest point on this segment
                closest = p1 + t * seg_vec
                dist = np.sqrt(np.sum((cell_pt - closest) ** 2))

                if dist < best_dist:
                    best_dist = dist
                    # Position along total line
                    if total_length > 1e-12:
                        best_pos = (cumulative_lengths[j] + t * segment_lengths[j]) / total_length
                    else:
                        best_pos = 0.0

            positions[i] = best_pos
            distances[i] = best_dist

        return positions, distances

    def compute_line_projections(self) -> dict[str, dict[str, np.ndarray]]:
        """Compute cell projections for all stored lines.

        Returns:
            Dictionary mapping line name to {positions, distances} arrays
        """
        projections = {}

        for line in self._drawn_lines:
            embedding_name = line.get('embeddingName', '')
            line_name = line.get('name', 'unnamed')

            # Get the appropriate embedding coordinates
            if embedding_name not in self.adata.obsm:
                continue

            coords = self.adata.obsm[embedding_name][:, :2]

            # Use smoothed points if available, otherwise raw points
            line_points = line.get('smoothedPoints') or line.get('points', [])
            if not line_points:
                continue

            positions, distances = self._project_cells_onto_line(line_points, coords)
            projections[line_name] = {
                'positions': positions,
                'distances': distances,
            }

        return projections

    def prepare_export_with_lines(self) -> anndata.AnnData:
        """Prepare AnnData for export, including line data.

        Stores line metadata as JSON string in .uns['xcell_lines_json'] and
        cell projections in .obsm['X_{line_name}_projection'].

        The JSON contains an array of line objects with:
        - name: Line name
        - embedding: Embedding name the line was drawn on
        - points: Array of [x, y] coordinates
        - smoothed_points: Array of smoothed [x, y] coordinates (if exists)

        Returns:
            Copy of adata with lines and projections added
        """
        import json

        # Work with a copy to avoid modifying the live data
        adata_export = self.adata.copy()

        if not self._drawn_lines:
            return adata_export

        # Store line metadata as JSON string (h5ad-safe)
        line_metadata = []
        for line in self._drawn_lines:
            line_info = {
                'name': line.get('name', 'unnamed'),
                'embedding': line.get('embeddingName', ''),
                'points': line.get('points', []),
            }
            smoothed = line.get('smoothedPoints')
            if smoothed:
                line_info['smoothed_points'] = smoothed
            line_metadata.append(line_info)

        adata_export.uns['xcell_lines_json'] = json.dumps(line_metadata)

        # Compute and store projections
        projections = self.compute_line_projections()

        for line_name, proj_data in projections.items():
            # Sanitize line name for use as key (replace spaces, special chars)
            safe_name = line_name.replace(' ', '_').replace('-', '_')
            safe_name = ''.join(c for c in safe_name if c.isalnum() or c == '_')

            key = f'X_{safe_name}_projection'

            # Store as n_cells x 2 matrix: [position, distance]
            proj_matrix = np.column_stack([
                proj_data['positions'],
                proj_data['distances'],
            ])
            adata_export.obsm[key] = proj_matrix

        return adata_export

    def _run_spline_association(
        self,
        test_values: np.ndarray,
        cell_indices: np.ndarray,
        gene_mask: np.ndarray,
        n_spline_knots: int = 5,
        fdr_threshold: float = 0.05,
        top_n: int = 50,
    ) -> dict[str, Any]:
        """Core spline regression engine for line association analysis.

        Fits cubic B-spline models for each gene against the provided test
        values and tests significance via F-test (spline model vs intercept-only).

        This is a reusable helper called by both test_line_association (single
        line) and test_multi_line_association (pooled multi-line).

        Args:
            test_values: 1-D array of test variable values (e.g. position along
                        line), one per cell. Should be in [0, 1].
            cell_indices: 1-D integer array of cell indices into self.adata.
            gene_mask: Boolean array of length n_genes selecting which genes
                      to test.
            n_spline_knots: Number of interior knots for the B-spline basis.
            fdr_threshold: FDR threshold for significance.
            top_n: Number of top genes to return for each direction.

        Returns:
            Dict with keys: positive, negative, modules, n_cells,
            n_significant, n_positive, n_negative, n_modules, fdr_threshold,
            diagnostics. Callers should add line_name, test_variable, etc.
        """
        from scipy.interpolate import BSpline
        from scipy.stats import f as f_dist
        from statsmodels.stats.multitest import multipletests

        n_cells_used = len(cell_indices)

        # Get expression matrix for selected cells and genes
        # (trusts that user has already preprocessed: normalize, log1p, etc.)
        X = self.adata.X[cell_indices][:, gene_mask]

        # Convert sparse to dense if needed
        if hasattr(X, 'toarray'):
            X = X.toarray()
        X = np.asarray(X, dtype=np.float64)

        n_genes = X.shape[1]
        gene_names = self.adata.var_names[gene_mask].tolist()

        # Build B-spline basis matrix
        # Use quantile-based knots for even coverage of cells
        pos = test_values

        # Create knot vector for cubic B-spline
        # Interior knots at quantiles, plus boundary knots
        degree = 3
        n_interior = n_spline_knots
        interior_knots = np.quantile(pos, np.linspace(0, 1, n_interior + 2)[1:-1])

        # Full knot vector: degree+1 copies at boundaries, interior knots in between
        knots = np.concatenate([
            np.repeat(0.0, degree + 1),
            interior_knots,
            np.repeat(1.0, degree + 1),
        ])

        # Number of basis functions
        n_basis = len(knots) - degree - 1

        # Evaluate B-spline basis at all positions
        B = np.zeros((n_cells_used, n_basis))
        for i in range(n_basis):
            # Create coefficient vector with 1 at position i
            c = np.zeros(n_basis)
            c[i] = 1.0
            spline = BSpline(knots, c, degree)
            B[:, i] = spline(pos)

        # Design matrix: B-spline basis only (no separate intercept).
        # B-spline basis functions satisfy the partition of unity (sum to 1),
        # so the constant function is already in their span.
        design = B
        k = n_basis - 1  # Extra df beyond intercept (one basis function spans the constant)

        # Solve OLS for all genes at once: beta = (X'X)^{-1} X' Y
        # where Y is expression matrix (n_cells x n_genes)
        try:
            XtX = design.T @ design
            XtX_inv = np.linalg.inv(XtX)
            beta = XtX_inv @ design.T @ X  # n_basis x n_genes
        except np.linalg.LinAlgError:
            raise ValueError("Singular design matrix. Try fewer spline knots.")

        # Compute residuals and RSS
        predicted = design @ beta
        residuals = X - predicted
        rss_full = np.sum(residuals ** 2, axis=0)  # RSS per gene

        # Null model: intercept only
        gene_means = X.mean(axis=0)
        rss_null = np.sum((X - gene_means) ** 2, axis=0)

        # F-test: F = [(RSS_null - RSS_full) / k] / [RSS_full / (n - k - 1)]
        df1 = k  # Numerator df (spline coefficients)
        df2 = n_cells_used - k - 1  # Denominator df

        # Avoid division by zero
        rss_full_safe = np.maximum(rss_full, 1e-10)

        f_stat = ((rss_null - rss_full) / df1) / (rss_full_safe / df2)
        f_stat = np.maximum(f_stat, 0)  # F-stat can't be negative

        # P-values from F distribution
        p_values = 1 - f_dist.cdf(f_stat, df1, df2)

        # FDR correction
        _, fdr, _, _ = multipletests(p_values, method='fdr_bh')

        # Compute effect size metrics
        # R-squared: variance explained
        r_squared = 1 - rss_full / np.maximum(rss_null, 1e-10)
        r_squared = np.clip(r_squared, 0, 1)

        # Amplitude: range of predicted expression along the line
        amplitude = predicted.max(axis=0) - predicted.min(axis=0)

        # Direction: correlation of predicted with position (for ranking)
        # Positive = expression increases along line, Negative = decreases
        pos_centered = pos - pos.mean()
        pred_centered = predicted - predicted.mean(axis=0)
        direction = np.zeros(n_genes)
        for g in range(n_genes):
            if np.std(pred_centered[:, g]) > 1e-10:
                corr = np.corrcoef(pos_centered, pred_centered[:, g])[0, 1]
                direction[g] = corr if not np.isnan(corr) else 0
            else:
                direction[g] = 0

        # Build results DataFrame
        results = pd.DataFrame({
            'gene': gene_names,
            'f_stat': f_stat,
            'pval': p_values,
            'fdr': fdr,
            'r_squared': r_squared,
            'amplitude': amplitude,
            'direction': direction,
        })

        # Sort by significance (combining FDR and amplitude)
        results['score'] = -np.log10(results['fdr'] + 1e-300) * results['amplitude']

        # Get significant genes
        sig_mask = results['fdr'] < fdr_threshold
        n_significant = sig_mask.sum()

        # Split into positive (increasing) and negative (decreasing) direction
        # (kept for backward compatibility)
        pos_mask = sig_mask & (results['direction'] > 0)
        neg_mask = sig_mask & (results['direction'] < 0)

        positive_genes = (
            results[pos_mask]
            .nlargest(top_n, 'score')
            [['gene', 'f_stat', 'pval', 'fdr', 'r_squared', 'amplitude', 'direction']]
            .to_dict('records')
        )

        negative_genes = (
            results[neg_mask]
            .nlargest(top_n, 'score')
            [['gene', 'f_stat', 'pval', 'fdr', 'r_squared', 'amplitude', 'direction']]
            .to_dict('records')
        )

        # ---- Module-based clustering of ALL significant genes ----
        # Evaluate spline profiles at evenly-spaced positions
        n_profile_points = 50
        profile_positions = np.linspace(0.0, 1.0, n_profile_points)
        profile_design = np.zeros((n_profile_points, n_basis))
        for i in range(n_basis):
            c = np.zeros(n_basis)
            c[i] = 1.0
            spline = BSpline(knots, c, degree)
            profile_design[:, i] = spline(profile_positions)
        profile_design_full = profile_design

        sig_indices = np.where(sig_mask.values)[0]
        modules = []

        if len(sig_indices) > 0:
            # Predicted profiles for significant genes (n_profile_points x n_sig_genes)
            sig_profiles = profile_design_full @ beta[:, sig_indices]

            # Min-max normalize each gene's profile to [0, 1]
            prof_min = sig_profiles.min(axis=0)
            prof_max = sig_profiles.max(axis=0)
            prof_range = prof_max - prof_min
            prof_range[prof_range < 1e-10] = 1.0  # avoid division by zero
            norm_profiles = (sig_profiles - prof_min) / prof_range  # (n_points, n_sig)

            if len(sig_indices) == 1:
                # Single gene: one module
                cluster_labels = np.array([0])
            else:
                # Hierarchical clustering with correlation distance
                from scipy.cluster.hierarchy import linkage, fcluster
                from scipy.spatial.distance import pdist

                # Transpose so each row is a gene's profile
                profile_matrix = norm_profiles.T  # (n_sig, n_points)

                # Correlation distance; clip to avoid numerical issues
                dists = pdist(profile_matrix, metric='correlation')
                dists = np.clip(dists, 0, 2)

                Z = linkage(dists, method='average')
                # Cut tree: use distance threshold of 0.5 (correlation-based)
                # This gives reasonable module granularity
                cluster_labels = fcluster(Z, t=0.5, criterion='distance') - 1  # 0-indexed

            sig_results = results.iloc[sig_indices].reset_index(drop=True)
            n_modules = int(cluster_labels.max()) + 1

            # Compute per-gene peak positions for all significant genes
            gene_peak_positions = np.argmax(norm_profiles, axis=0) / max(n_profile_points - 1, 1)

            for mod_idx in range(n_modules):
                member_mask = cluster_labels == mod_idx
                member_genes = sig_results[member_mask]
                member_profiles = norm_profiles[:, member_mask]  # (n_points, n_members)
                member_peak_positions = gene_peak_positions[member_mask]

                # Representative profile: mean of normalized profiles in this module
                rep_profile = member_profiles.mean(axis=1)

                # Classify pattern shape
                pattern = self._classify_profile_pattern(rep_profile, profile_positions)

                # Sort genes within module by peak position along the line
                member_genes = member_genes.copy()
                member_genes['peak_position'] = member_peak_positions
                member_genes_sorted = member_genes.sort_values('peak_position')
                if len(member_genes_sorted) > top_n:
                    member_genes_sorted = member_genes_sorted.head(top_n)

                # Build gene records with per-gene profiles
                gene_records = []
                for row_idx, (orig_sig_idx, row) in enumerate(member_genes_sorted.iterrows()):
                    # orig_sig_idx is the index into norm_profiles columns
                    # (sig_results was built with reset_index, so index = column position)
                    gene_profile = norm_profiles[:, orig_sig_idx].tolist()
                    gene_records.append({
                        'gene': row['gene'],
                        'f_stat': row['f_stat'],
                        'pval': row['pval'],
                        'fdr': row['fdr'],
                        'r_squared': row['r_squared'],
                        'amplitude': row['amplitude'],
                        'direction': row['direction'],
                        'profile': gene_profile,
                        'peak_position': float(row['peak_position']),
                    })

                modules.append({
                    'module_id': mod_idx,
                    'pattern': pattern,
                    'n_genes': int(member_mask.sum()),
                    'representative_profile': rep_profile.tolist(),
                    'profile_positions': profile_positions.tolist(),
                    'genes': gene_records,
                })

            # Sort modules: increasing first, then decreasing, then peak, trough, complex
            pattern_order = {'increasing': 0, 'decreasing': 1, 'peak': 2, 'trough': 3, 'complex': 4}
            modules.sort(key=lambda m: (pattern_order.get(m['pattern'], 5), -m['n_genes']))

        # Compute diagnostic statistics
        n_pval_below_05 = int((p_values < 0.05).sum())
        n_pval_below_01 = int((p_values < 0.01).sum())

        # Check expression matrix properties
        expr_min = float(X.min())
        expr_max = float(X.max())
        expr_mean = float(X.mean())
        n_zero_genes = int((X.sum(axis=0) == 0).sum())

        # Position statistics
        pos_min = float(pos.min())
        pos_max = float(pos.max())
        pos_std = float(pos.std())

        return {
            'positive': positive_genes,
            'negative': negative_genes,
            'modules': modules,
            'n_cells': n_cells_used,
            'n_significant': int(n_significant),
            'n_positive': int(pos_mask.sum()),
            'n_negative': int(neg_mask.sum()),
            'n_modules': len(modules),
            'fdr_threshold': fdr_threshold,
            'diagnostics': {
                'n_genes_tested': n_genes,
                'n_pval_below_05': n_pval_below_05,
                'n_pval_below_01': n_pval_below_01,
                'position_range': [pos_min, pos_max],
                'position_std': pos_std,
                'expression_range': [expr_min, expr_max],
                'expression_mean': expr_mean,
                'n_zero_genes': n_zero_genes,
                'spline_df': k,
            },
        }

    def prepare_line_association(
        self,
        line_name: str,
        cell_indices: list[int] | None = None,
        gene_subset: str | list[str] | dict[str, Any] | None = None,
        test_variable: str = 'position',
        n_spline_knots: int = 5,
        min_cells: int = 20,
        fdr_threshold: float = 0.05,
        top_n: int = 50,
    ) -> tuple[Callable[[], dict[str, Any]], Callable[[dict[str, Any]], None]]:
        """Prepare line association computation (cancellable).

        Validates that the named line exists (fail fast), then returns a pair of
        functions: ``compute_fn`` (calls test_line_association, read-only) and
        ``apply_fn`` (no-op since this operation doesn't write to adata).

        Args:
            Same as test_line_association.

        Returns:
            Tuple of (compute_fn, apply_fn)

        Raises:
            ValueError: If line not found
        """
        # Fail fast: validate line exists
        line_found = False
        for l in self._drawn_lines:
            if l.get('name') == line_name:
                line_found = True
                break
        if not line_found:
            raise ValueError(f"Line '{line_name}' not found")

        # Snapshot parameters
        snap_line_name = line_name
        snap_cell_indices = cell_indices
        snap_gene_subset = gene_subset
        snap_test_variable = test_variable
        snap_n_spline_knots = n_spline_knots
        snap_min_cells = min_cells
        snap_fdr_threshold = fdr_threshold
        snap_top_n = top_n

        def compute_fn() -> dict[str, Any]:
            return self.test_line_association(
                line_name=snap_line_name,
                cell_indices=snap_cell_indices,
                gene_subset=snap_gene_subset,
                test_variable=snap_test_variable,
                n_spline_knots=snap_n_spline_knots,
                min_cells=snap_min_cells,
                fdr_threshold=snap_fdr_threshold,
                top_n=snap_top_n,
            )

        def apply_fn(result: dict[str, Any]) -> dict[str, Any]:
            return result  # Read-only operation, result is already serializable

        return compute_fn, apply_fn

    def test_line_association(
        self,
        line_name: str,
        cell_indices: list[int] | None = None,
        gene_subset: str | list[str] | dict[str, Any] | None = None,
        test_variable: str = 'position',
        n_spline_knots: int = 5,
        min_cells: int = 20,
        fdr_threshold: float = 0.05,
        top_n: int = 50,
    ) -> dict[str, Any]:
        """Test genes for association with position along or distance from a line.

        Uses cubic B-spline regression to model gene expression as a function
        of a spatial variable derived from the line, then tests whether the
        spline model explains significantly more variance than an intercept-only
        model (F-test).

        Args:
            line_name: Name of the line to test against
            cell_indices: Optional list of cell indices to use. If None, uses
                         all cells (projected based on distance threshold).
            gene_subset: Optional gene filter. Can be a boolean column name (str),
                        a list of gene names, or a dict for combining columns.
            test_variable: 'position' to test against position along the line,
                          'distance' to test against perpendicular distance from
                          the line.
            n_spline_knots: Number of interior knots for the B-spline basis.
                           Total df = n_spline_knots + 2 (for cubic splines).
            min_cells: Minimum number of cells required for testing.
            fdr_threshold: FDR threshold for significance.
            top_n: Number of top genes to return for each direction.

        Returns:
            Dict containing:
            - modules: Gene modules clustered by expression profile shape
            - positive: Genes with expression increasing along variable
            - negative: Genes with expression decreasing along variable
            - n_cells: Number of cells used
            - n_significant: Total significant genes at FDR threshold
            - line_name: The line name used

        Raises:
            ValueError: If line not found or too few cells
        """
        # Find the line
        line = None
        for l in self._drawn_lines:
            if l.get('name') == line_name:
                line = l
                break

        if line is None:
            raise ValueError(f"Line '{line_name}' not found")

        embedding_name = line.get('embeddingName', '')
        if embedding_name not in self.adata.obsm:
            raise ValueError(f"Embedding '{embedding_name}' not found")

        # Get line points (smoothed if available)
        line_points = line.get('smoothedPoints') or line.get('points', [])
        if len(line_points) < 2:
            raise ValueError("Line must have at least 2 points")

        # Get embedding coordinates
        coords = self.adata.obsm[embedding_name][:, :2]

        # Project cells onto line
        positions, distances = self._project_cells_onto_line(line_points, coords)

        # Determine which cells to use
        if cell_indices is not None:
            # Use provided cell indices
            cell_mask = np.zeros(self.n_cells, dtype=bool)
            cell_mask[cell_indices] = True
        else:
            # Use all cells (could filter by distance threshold in future)
            cell_mask = np.ones(self.n_cells, dtype=bool)

        # Get positions and distances for selected cells
        selected_indices = np.where(cell_mask)[0]
        selected_positions = positions[cell_mask]
        selected_distances = distances[cell_mask]
        n_cells_used = len(selected_indices)

        if n_cells_used < min_cells:
            raise ValueError(
                f"Too few cells ({n_cells_used}). Need at least {min_cells}."
            )

        # Select the test variable
        if test_variable == 'distance':
            # Normalize distances to [0, 1] for spline fitting
            d_min = selected_distances.min()
            d_max = selected_distances.max()
            if d_max - d_min < 1e-10:
                raise ValueError(
                    "All cells have the same distance from the line. "
                    "Cannot test distance association."
                )
            test_values = (selected_distances - d_min) / (d_max - d_min)
        else:
            test_values = selected_positions

        # Resolve gene subset if provided
        if gene_subset is not None:
            gene_mask, _, _ = self._resolve_gene_mask(gene_subset)
        else:
            gene_mask = np.ones(self.n_genes, dtype=bool)

        # Delegate to the shared spline regression engine
        result = self._run_spline_association(
            test_values=test_values,
            cell_indices=selected_indices,
            gene_mask=gene_mask,
            n_spline_knots=n_spline_knots,
            fdr_threshold=fdr_threshold,
            top_n=top_n,
        )

        # Add line-specific metadata
        result['line_name'] = line_name
        result['test_variable'] = test_variable

        return result

    def prepare_multi_line_association(
        self,
        lines: list[dict[str, Any]],
        gene_subset: str | list[str] | dict[str, Any] | None = None,
        test_variable: str = 'position',
        n_spline_knots: int = 5,
        min_cells: int = 20,
        fdr_threshold: float = 0.05,
        top_n: int = 50,
    ) -> tuple[Callable[[], dict[str, Any]], Callable[[dict[str, Any]], None]]:
        """Prepare multi-line association computation (cancellable).

        Validates that all named lines exist (fail fast), then returns a pair of
        functions: ``compute_fn`` (calls test_multi_line_association, read-only)
        and ``apply_fn`` (no-op since this operation doesn't write to adata).

        Args:
            Same as test_multi_line_association.

        Returns:
            Tuple of (compute_fn, apply_fn)

        Raises:
            ValueError: If any line not found
        """
        # Fail fast: validate all lines exist
        line_names_set = {l.get('name') for l in self._drawn_lines}
        for entry in lines:
            if entry['name'] not in line_names_set:
                raise ValueError(f"Line '{entry['name']}' not found")

        # Snapshot parameters
        snap_lines = lines
        snap_gene_subset = gene_subset
        snap_test_variable = test_variable
        snap_n_spline_knots = n_spline_knots
        snap_min_cells = min_cells
        snap_fdr_threshold = fdr_threshold
        snap_top_n = top_n

        def compute_fn() -> dict[str, Any]:
            return self.test_multi_line_association(
                lines=snap_lines,
                gene_subset=snap_gene_subset,
                test_variable=snap_test_variable,
                n_spline_knots=snap_n_spline_knots,
                min_cells=snap_min_cells,
                fdr_threshold=snap_fdr_threshold,
                top_n=snap_top_n,
            )

        def apply_fn(result: dict[str, Any]) -> dict[str, Any]:
            return result  # Read-only operation, result is already serializable

        return compute_fn, apply_fn

    def test_multi_line_association(
        self,
        lines: list[dict[str, Any]],
        gene_subset: str | list[str] | dict[str, Any] | None = None,
        test_variable: str = 'position',
        n_spline_knots: int = 5,
        min_cells: int = 20,
        fdr_threshold: float = 0.05,
        top_n: int = 50,
    ) -> dict[str, Any]:
        """Test genes for association across multiple lines (pooled analysis).

        Projects cells from each line entry onto their respective line geometry,
        normalizes positions per-line (optionally reversing direction), pools all
        cells, and runs the shared spline regression engine.

        Args:
            lines: List of dicts, each with:
                - name (str): Name of a drawn line in self._drawn_lines
                - cell_indices (list[int]): Cell indices to use for this line
                - reversed (bool): If True, flip positions (1 - pos) for this line
            gene_subset: Optional gene filter (boolean column name, gene list, or dict).
            test_variable: 'position' or 'distance'.
            n_spline_knots: Number of interior knots for the B-spline basis.
            min_cells: Minimum number of pooled cells required.
            fdr_threshold: FDR threshold for significance.
            top_n: Number of top genes to return per direction.

        Returns:
            Dict with spline association results plus multi-line metadata:
            line_name, test_variable, n_lines, lines_used.

        Raises:
            ValueError: If any line not found, embedding missing, or too few cells.
        """
        all_test_values = []
        all_cell_indices = []
        lines_used = []

        for entry in lines:
            line_name = entry['name']
            entry_cell_indices = entry['cell_indices']
            is_reversed = entry.get('reversed', False)

            # Look up line geometry
            line = None
            for l in self._drawn_lines:
                if l.get('name') == line_name:
                    line = l
                    break
            if line is None:
                raise ValueError(f"Line '{line_name}' not found")

            embedding_name = line.get('embeddingName', '')
            if embedding_name not in self.adata.obsm:
                raise ValueError(f"Embedding '{embedding_name}' not found")

            line_points = line.get('smoothedPoints') or line.get('points', [])
            if len(line_points) < 2:
                raise ValueError(f"Line '{line_name}' must have at least 2 points")

            coords = self.adata.obsm[embedding_name][:, :2]

            # Project all cells onto the line
            positions, distances = self._project_cells_onto_line(line_points, coords)

            # Select this entry's cells
            idx_array = np.array(entry_cell_indices, dtype=int)
            if test_variable == 'distance':
                selected = distances[idx_array]
                d_min = selected.min()
                d_max = selected.max()
                if d_max - d_min < 1e-10:
                    raise ValueError(
                        f"All cells for line '{line_name}' have the same distance. "
                        "Cannot test distance association."
                    )
                vals = (selected - d_min) / (d_max - d_min)
            else:
                vals = positions[idx_array]

            # Reverse direction if requested
            if is_reversed:
                vals = 1.0 - vals

            all_test_values.append(vals)
            all_cell_indices.append(idx_array)
            lines_used.append(line_name)

        # Pool across all lines
        pooled_test_values = np.concatenate(all_test_values)
        pooled_cell_indices = np.concatenate(all_cell_indices)

        if len(pooled_cell_indices) < min_cells:
            raise ValueError(
                f"Too few pooled cells ({len(pooled_cell_indices)}). "
                f"Need at least {min_cells}."
            )

        # Resolve gene subset
        if gene_subset is not None:
            gene_mask, _, _ = self._resolve_gene_mask(gene_subset)
        else:
            gene_mask = np.ones(self.n_genes, dtype=bool)

        # Run spline association on pooled data
        result = self._run_spline_association(
            test_values=pooled_test_values,
            cell_indices=pooled_cell_indices,
            gene_mask=gene_mask,
            n_spline_knots=n_spline_knots,
            fdr_threshold=fdr_threshold,
            top_n=top_n,
        )

        # Add multi-line metadata
        result['line_name'] = ' + '.join(lines_used)
        result['test_variable'] = test_variable
        result['n_lines'] = len(lines_used)
        result['lines_used'] = lines_used

        return result

    @staticmethod
    def _classify_profile_pattern(
        profile: np.ndarray,
        positions: np.ndarray,
    ) -> str:
        """Classify the shape of a gene expression profile along a line.

        Args:
            profile: Normalized expression profile (0-1 scale), shape (n_points,)
            positions: Corresponding position values along the line (0-1)

        Returns:
            One of: 'increasing', 'decreasing', 'peak', 'trough', 'complex'
        """
        # Correlation with position
        corr = np.corrcoef(positions, profile)[0, 1]
        if np.isnan(corr):
            corr = 0.0

        # Strong monotonic trend
        if corr > 0.7:
            return 'increasing'
        if corr < -0.7:
            return 'decreasing'

        # Check for peak or trough: location of max/min relative to endpoints
        argmax = np.argmax(profile)
        argmin = np.argmin(profile)
        n = len(profile)
        interior_fraction = 0.15  # consider first/last 15% as "edges"
        edge_low = int(n * interior_fraction)
        edge_high = int(n * (1 - interior_fraction))

        max_is_interior = edge_low <= argmax <= edge_high
        min_is_interior = edge_low <= argmin <= edge_high

        # Peak: max in interior and higher than both endpoints
        edge_mean = (profile[:edge_low].mean() + profile[edge_high:].mean()) / 2
        center_val = profile[argmax] if max_is_interior else profile[argmin]

        if max_is_interior and profile[argmax] > edge_mean + 0.2:
            return 'peak'
        if min_is_interior and profile[argmin] < edge_mean - 0.2:
            return 'trough'

        # Fallback: use monotonicity for weak trends
        if corr > 0.3:
            return 'increasing'
        if corr < -0.3:
            return 'decreasing'

        return 'complex'

    def create_line_projection_embedding(
        self,
        line_name: str,
        cell_indices: list[int] | None = None,
    ) -> dict[str, Any]:
        """Create an embedding based on cell projections onto a line.

        This creates a new embedding in .obsm where:
        - X-axis: position along the line (0 = start, 1 = end)
        - Y-axis: distance from the line (with jitter for visualization)

        Only cells that are projected are included; others get NaN.

        Args:
            line_name: Name of the line to create embedding from
            cell_indices: Optional cell indices to include. If None and line
                         has projections stored, uses those. Otherwise uses all.

        Returns:
            Dict with embedding name and cell count
        """
        # Find the line
        line = None
        for l in self._drawn_lines:
            if l.get('name') == line_name:
                line = l
                break

        if line is None:
            raise ValueError(f"Line '{line_name}' not found")

        embedding_name = line.get('embeddingName', '')
        if embedding_name not in self.adata.obsm:
            raise ValueError(f"Embedding '{embedding_name}' not found")

        # Get line points (smoothed if available)
        line_points = line.get('smoothedPoints') or line.get('points', [])
        if len(line_points) < 2:
            raise ValueError("Line must have at least 2 points")

        # Get embedding coordinates
        coords = self.adata.obsm[embedding_name][:, :2]

        # Project all cells onto line
        positions, distances = self._project_cells_onto_line(line_points, coords)

        # Determine which cells to include
        if cell_indices is not None:
            cell_mask = np.zeros(self.n_cells, dtype=bool)
            cell_mask[cell_indices] = True
        else:
            # Use all cells
            cell_mask = np.ones(self.n_cells, dtype=bool)

        # Create the projection embedding
        # X = position along line (0-1), Y = distance from line (normalized to 0-1)
        proj_embedding = np.full((self.n_cells, 2), np.nan)
        proj_embedding[cell_mask, 0] = positions[cell_mask]

        # Normalize distances to 0-1 range
        masked_distances = distances[cell_mask]
        dist_min = masked_distances.min()
        dist_max = masked_distances.max()
        if dist_max > dist_min:
            normalized_distances = (masked_distances - dist_min) / (dist_max - dist_min)
        else:
            # All cells same distance from line
            normalized_distances = np.zeros_like(masked_distances)
        proj_embedding[cell_mask, 1] = normalized_distances

        # Sanitize line name for embedding key
        safe_name = line_name.replace(' ', '_').replace('-', '_')
        safe_name = ''.join(c for c in safe_name if c.isalnum() or c == '_')
        emb_key = f'X_{safe_name}_proj'

        # Store in adata.obsm
        self.adata.obsm[emb_key] = proj_embedding

        n_cells_projected = int(cell_mask.sum())

        return {
            'embedding_name': emb_key,
            'n_cells': n_cells_projected,
            'position_range': [float(positions[cell_mask].min()), float(positions[cell_mask].max())],
            'distance_range_original': [float(dist_min), float(dist_max)],
            'distance_range_normalized': [0.0, 1.0],
        }

    # =========================================================================
    # Scanpy analysis methods
    # =========================================================================

    def get_action_history(self) -> list[dict[str, Any]]:
        """Get the history of scanpy operations performed."""
        return self._action_history

    def _log_action(self, action: str, params: dict[str, Any], result: dict[str, Any]) -> None:
        """Log a scanpy action to the history."""
        import datetime
        self._action_history.append({
            'action': action,
            'params': params,
            'result': result,
            'timestamp': datetime.datetime.now().isoformat(),
        })

    def _validate_cell_indices(
        self, active_cell_indices: list[int] | None,
    ) -> np.ndarray | None:
        """Validate active_cell_indices and return as numpy array.

        Returns None if no subsetting is needed (indices is None or covers all cells).
        """
        if active_cell_indices is None:
            return None

        indices = np.array(active_cell_indices)

        if len(indices) == 0:
            raise ValueError("active_cell_indices is empty — no cells selected.")

        if indices.max() >= self.n_cells or indices.min() < 0:
            raise ValueError(
                f"active_cell_indices out of range: max index {indices.max()}, "
                f"n_cells {self.n_cells}"
            )

        if len(indices) == self.n_cells:
            return None

        return indices

    def _get_active_adata(
        self, active_cell_indices: list[int] | None,
    ) -> tuple[anndata.AnnData, np.ndarray | None]:
        """Get an AnnData subset copy for active cells.

        Args:
            active_cell_indices: List of cell indices to include, or None for all.

        Returns:
            Tuple of (adata_subset_or_full, indices_array_or_None).
            If indices is None or covers all cells, returns (self.adata, None).
        """
        indices = self._validate_cell_indices(active_cell_indices)
        if indices is None:
            return self.adata, None
        return self.adata[indices].copy(), indices

    def check_prerequisites(self, action: str) -> dict[str, Any]:
        """Check if prerequisites are met for a scanpy action.

        Args:
            action: The scanpy action to check

        Returns:
            Dict with 'satisfied' (bool) and 'missing' (list of missing prereqs)
        """
        prereqs = {
            # Cell analysis
            'filter_genes': [],
            'exclude_genes': [],
            'filter_cells': [],
            'normalize_total': [],
            'log1p': [],
            'pca': [],
            'neighbors': ['pca'],
            'umap': ['neighbors'],
            'leiden': ['neighbors'],
            'pca_loadings': ['pca_with_loadings'],
            # Gene analysis
            'gene_pca': [],
            'gene_neighbors': [],
            'find_similar_genes': ['gene_neighbors'],
            'cluster_genes': ['gene_neighbors'],
            'build_gene_graph': [],  # Convenience function, no prereqs
            # Spatial analysis
            'spatial_neighbors': ['has_spatial'],
            'spatial_autocorr': ['spatial_neighbors'],
            'contourize': ['has_spatial'],
        }

        required = prereqs.get(action, [])
        missing = []

        for prereq in required:
            if prereq == 'pca':
                if 'X_pca' not in self.adata.obsm:
                    missing.append('pca')
            elif prereq == 'neighbors':
                if 'connectivities' not in self.adata.obsp:
                    missing.append('neighbors')
            elif prereq == 'gene_pca':
                if 'X_gene_pca' not in self.adata.varm:
                    missing.append('gene_pca')
            elif prereq == 'gene_neighbors':
                if 'gene_connectivities' not in self.adata.varp:
                    missing.append('gene_neighbors')
            elif prereq == 'pca_with_loadings':
                if 'pca' not in self.adata.uns or 'PCs' not in self.adata.varm:
                    missing.append('pca_with_loadings')
            elif prereq == 'has_spatial':
                if not self._has_spatial_coordinates():
                    missing.append('has_spatial')
            elif prereq == 'spatial_neighbors':
                if 'spatial_connectivities' not in self.adata.obsp:
                    missing.append('spatial_neighbors')

        return {
            'satisfied': len(missing) == 0,
            'missing': missing,
        }

    def _has_spatial_coordinates(self) -> bool:
        """Check if spatial coordinates are available.

        Looks for common spatial coordinate keys in .obsm.

        Returns:
            True if spatial coordinates exist
        """
        spatial_keys = ['spatial', 'X_spatial']
        for key in spatial_keys:
            if key in self.adata.obsm:
                arr = self.adata.obsm[key]
                if isinstance(arr, np.ndarray) and arr.ndim == 2 and arr.shape[1] >= 2:
                    return True
        return False

    def _get_spatial_key(self) -> str | None:
        """Get the key for spatial coordinates in .obsm.

        Returns:
            The key name, or None if not found
        """
        spatial_keys = ['spatial', 'X_spatial']
        for key in spatial_keys:
            if key in self.adata.obsm:
                arr = self.adata.obsm[key]
                if isinstance(arr, np.ndarray) and arr.ndim == 2 and arr.shape[1] >= 2:
                    return key
        return None

    def _column_to_bool_array(self, col_name: str) -> np.ndarray:
        """Convert a .var column to a boolean numpy array.

        Accepts bool columns and numeric 0/1 columns (matching the same
        rules as get_var_boolean_columns). Raises ValueError if the
        column is not bool-like.
        """
        if col_name not in self.adata.var.columns:
            raise ValueError(f"Column '{col_name}' not found in .var")
        series = self.adata.var[col_name]
        dtype = series.dtype
        if dtype == bool:
            return np.asarray(series.values, dtype=bool)
        if pd.api.types.is_numeric_dtype(dtype):
            unique_vals = set(series.dropna().unique())
            if unique_vals.issubset({0, 1, 0.0, 1.0, True, False}):
                return np.asarray((series == 1) | (series == True), dtype=bool)
        raise ValueError(f"Column '{col_name}' is not a boolean-like column")

    def _compute_visible_mask(
        self,
        keep_columns: list[str],
        hide_columns: list[str],
        keep_combine_mode: str,
    ) -> np.ndarray:
        """Compute the final visible mask from a config.

        Formula:
            visible = keep_mask AND NOT hide_mask
            keep_mask = all-True if no keep columns
                      = OR(columns) if keep_combine_mode == 'or'
                      = AND(columns) if keep_combine_mode == 'and'
            hide_mask = all-False if no hide columns
                      = OR(columns) otherwise
        """
        n = self.n_genes
        if keep_columns:
            arrays = [self._column_to_bool_array(c) for c in keep_columns]
            if keep_combine_mode == 'and':
                keep_mask = arrays[0].copy()
                for a in arrays[1:]:
                    keep_mask &= a
            else:  # 'or' (default)
                keep_mask = arrays[0].copy()
                for a in arrays[1:]:
                    keep_mask |= a
        else:
            keep_mask = np.ones(n, dtype=bool)

        if hide_columns:
            arrays = [self._column_to_bool_array(c) for c in hide_columns]
            hide_mask = arrays[0].copy()
            for a in arrays[1:]:
                hide_mask |= a
        else:
            hide_mask = np.zeros(n, dtype=bool)

        return keep_mask & ~hide_mask

    def get_gene_mask(self) -> dict[str, Any]:
        """Return the current mask config + counts.

        Always returns a dict, even when no mask is active.
        """
        n_total = self.n_genes
        if self._gene_mask_config is None or self._visible_gene_mask is None:
            return {
                'active': False,
                'keep_columns': [],
                'hide_columns': [],
                'keep_combine_mode': 'or',
                'n_visible': n_total,
                'n_total': n_total,
                'visible_gene_names': None,
            }
        n_visible = int(self._visible_gene_mask.sum())
        visible_gene_names = self.adata.var.index[self._visible_gene_mask].tolist()
        return {
            'active': True,
            'keep_columns': list(self._gene_mask_config.get('keep_columns', [])),
            'hide_columns': list(self._gene_mask_config.get('hide_columns', [])),
            'keep_combine_mode': self._gene_mask_config.get('keep_combine_mode', 'or'),
            'n_visible': n_visible,
            'n_total': n_total,
            'visible_gene_names': visible_gene_names,
        }

    def set_gene_mask(
        self,
        keep_columns: list[str],
        hide_columns: list[str],
        keep_combine_mode: str = 'or',
    ) -> dict[str, Any]:
        """Apply a gene mask.

        - Validates all referenced columns exist and are bool-like.
        - Empty keep_columns + empty hide_columns clears the mask.
        - Raises ValueError if the resulting mask leaves 0 visible genes.

        Returns the same shape as get_gene_mask().
        """
        if keep_combine_mode not in ('or', 'and'):
            raise ValueError(f"keep_combine_mode must be 'or' or 'and', got {keep_combine_mode!r}")

        # Empty config = clear
        if not keep_columns and not hide_columns:
            return self.clear_gene_mask()

        # Validate columns (raises ValueError if any are missing/non-bool)
        for c in keep_columns:
            self._column_to_bool_array(c)
        for c in hide_columns:
            self._column_to_bool_array(c)

        mask = self._compute_visible_mask(keep_columns, hide_columns, keep_combine_mode)
        if mask.sum() == 0:
            raise ValueError("Gene mask would leave 0 visible genes")

        self._gene_mask_config = {
            'keep_columns': list(keep_columns),
            'hide_columns': list(hide_columns),
            'keep_combine_mode': keep_combine_mode,
        }
        self._visible_gene_mask = mask
        return self.get_gene_mask()

    def clear_gene_mask(self) -> dict[str, Any]:
        """Clear the gene mask state."""
        self._gene_mask_config = None
        self._visible_gene_mask = None
        return self.get_gene_mask()

    def _regenerate_gene_mask_after_var_change(self) -> bool:
        """Rebuild _visible_gene_mask against the current .var axis.

        Called after operations that drop genes from .var or change the
        gene index. Behaviour:
          - No active mask → returns False, no-op.
          - All referenced columns still exist → recomputes the mask
            in place; returns False.
          - One or more referenced columns are gone → clears the mask;
            returns True.
        """
        if self._gene_mask_config is None:
            return False
        cfg = self._gene_mask_config
        try:
            self._visible_gene_mask = self._compute_visible_mask(
                keep_columns=cfg['keep_columns'],
                hide_columns=cfg['hide_columns'],
                keep_combine_mode=cfg['keep_combine_mode'],
            )
            # If everything got filtered out, clear rather than crash.
            if self._visible_gene_mask.sum() == 0:
                self.clear_gene_mask()
                return True
            return False
        except ValueError:
            # A referenced column no longer exists — clear the mask.
            self.clear_gene_mask()
            return True

    def get_visible_gene_names(self) -> list[str]:
        """Return gene names where _visible_gene_mask is True.

        Returns all gene names when no mask is active.
        """
        if self._visible_gene_mask is None:
            return self.adata.var.index.tolist()
        return self.adata.var.index[self._visible_gene_mask].tolist()

    def _filter_to_visible(self, genes: list[str]) -> tuple[list[str], int]:
        """Split a gene list into (visible_genes, n_excluded).

        When no mask is active, returns (genes, 0) unchanged.
        """
        if self._visible_gene_mask is None:
            return list(genes), 0
        visible_set = set(self.adata.var.index[self._visible_gene_mask].tolist())
        kept = [g for g in genes if g in visible_set]
        return kept, len(genes) - len(kept)

    def get_var_boolean_columns(self) -> list[dict[str, Any]]:
        """Get list of boolean columns in .var that can be used for gene filtering.

        Returns:
            List of dicts with column name, description, and count of True values
        """
        bool_columns = []
        for col in self.adata.var.columns:
            # Check if column is boolean or can be treated as boolean
            dtype = self.adata.var[col].dtype
            if dtype == bool or (dtype == 'bool'):
                n_true = self.adata.var[col].sum()
                bool_columns.append({
                    'name': col,
                    'n_true': int(n_true),
                    'n_total': self.n_genes,
                })
            # Also check for columns that look boolean (0/1 or True/False)
            elif pd.api.types.is_numeric_dtype(dtype):
                unique_vals = self.adata.var[col].dropna().unique()
                if len(unique_vals) <= 2 and set(unique_vals).issubset({0, 1, 0.0, 1.0, True, False}):
                    n_true = int((self.adata.var[col] == 1).sum() | (self.adata.var[col] == True).sum())
                    bool_columns.append({
                        'name': col,
                        'n_true': n_true,
                        'n_total': self.n_genes,
                    })
        return bool_columns

    def _resolve_gene_mask(
        self,
        gene_subset: str | list[str] | dict[str, Any] | None,
    ) -> tuple[np.ndarray, str, dict[str, Any]]:
        """Resolve a gene_subset specification into a boolean mask.

        Args:
            gene_subset: Gene subset specification. Can be:
                - None: all genes
                - str: single boolean column name from .var (e.g., 'highly_variable')
                - list[str]: explicit list of gene names
                - dict: {'columns': [...], 'operation': 'intersection'|'union'}

        Returns:
            Tuple of (boolean mask, subset_type string, metadata dict)
        """
        if gene_subset is None:
            return (
                np.ones(self.n_genes, dtype=bool),
                'all',
                {'n_genes': self.n_genes},
            )

        # Single column name
        if isinstance(gene_subset, str):
            if gene_subset not in self.adata.var.columns:
                raise ValueError(f"Column '{gene_subset}' not found in .var")
            col_values = self.adata.var[gene_subset]
            # Convert to boolean mask
            if col_values.dtype == bool:
                mask = col_values.values
            else:
                mask = (col_values == 1) | (col_values == True)
            mask = np.asarray(mask, dtype=bool)
            if mask.sum() == 0:
                raise ValueError(f"No genes found with {gene_subset}=True")
            return (
                mask,
                f'column:{gene_subset}',
                {'column': gene_subset, 'n_genes': int(mask.sum())},
            )

        # Explicit list of gene names
        if isinstance(gene_subset, list) and len(gene_subset) > 0 and isinstance(gene_subset[0], str):
            # Check if it looks like gene names (not column names)
            # If all items are in var_names, treat as gene list
            # If all items are in var.columns, treat as column list with default intersection
            all_genes = all(g in self.adata.var_names for g in gene_subset)
            all_columns = all(c in self.adata.var.columns for c in gene_subset)

            if all_genes and not all_columns:
                # Explicit gene list
                mask = self.adata.var_names.isin(gene_subset)
                if mask.sum() == 0:
                    raise ValueError("None of the specified genes found in dataset")
                return (
                    mask,
                    'gene_list',
                    {'n_genes': int(mask.sum()), 'genes_requested': len(gene_subset)},
                )
            elif all_columns:
                # Treat as column list with intersection
                gene_subset = {'columns': gene_subset, 'operation': 'intersection'}
            else:
                # Mixed - try as gene list
                mask = self.adata.var_names.isin(gene_subset)
                if mask.sum() == 0:
                    raise ValueError("None of the specified genes found in dataset")
                return (
                    mask,
                    'gene_list',
                    {'n_genes': int(mask.sum()), 'genes_requested': len(gene_subset)},
                )

        # Dict with columns and operation
        if isinstance(gene_subset, dict):
            columns = gene_subset.get('columns', [])
            operation = gene_subset.get('operation', 'intersection')

            if not columns:
                raise ValueError("No columns specified in gene_subset")

            if operation not in ('intersection', 'union'):
                raise ValueError("operation must be 'intersection' or 'union'")

            # Validate columns exist
            missing = [c for c in columns if c not in self.adata.var.columns]
            if missing:
                raise ValueError(f"Columns not found in .var: {missing}")

            # Build combined mask
            masks = []
            for col in columns:
                col_values = self.adata.var[col]
                if col_values.dtype == bool:
                    m = col_values.values
                else:
                    m = (col_values == 1) | (col_values == True)
                masks.append(np.asarray(m, dtype=bool))

            if operation == 'intersection':
                combined_mask = np.all(masks, axis=0)
                op_symbol = 'AND'
            else:  # union
                combined_mask = np.any(masks, axis=0)
                op_symbol = 'OR'

            if combined_mask.sum() == 0:
                raise ValueError(f"No genes found matching {op_symbol} of {columns}")

            return (
                combined_mask,
                f'{operation}:{"+".join(columns)}',
                {
                    'columns': columns,
                    'operation': operation,
                    'n_genes': int(combined_mask.sum()),
                    'individual_counts': {c: int(m.sum()) for c, m in zip(columns, masks)},
                },
            )

        raise ValueError(
            "gene_subset must be None, a column name (str), a list of gene names, "
            "or a dict with 'columns' and 'operation'"
        )

    def run_filter_genes(
        self,
        min_counts: int | None = None,
        max_counts: int | None = None,
        min_cells: int | None = None,
        max_cells: int | None = None,
        active_cell_indices: list[int] | None = None,
    ) -> dict[str, Any]:
        """Filter genes based on counts or number of cells expressing.

        Args:
            min_counts: Minimum total counts for a gene
            max_counts: Maximum total counts for a gene
            min_cells: Minimum number of cells expressing the gene
            max_cells: Maximum number of cells expressing the gene
            active_cell_indices: If provided, compute gene stats using only these cells

        Returns:
            Dict with before/after gene counts
        """
        n_genes_before = self.n_genes

        # Build kwargs for scanpy
        kwargs = {}
        if min_counts is not None:
            kwargs['min_counts'] = min_counts
        if max_counts is not None:
            kwargs['max_counts'] = max_counts
        if min_cells is not None:
            kwargs['min_cells'] = min_cells
        if max_cells is not None:
            kwargs['max_cells'] = max_cells

        if kwargs:
            adata_sub, indices = self._get_active_adata(active_cell_indices)
            if indices is not None:
                # Run filter on subset to find surviving genes
                # (adata_sub is already a copy from _get_active_adata)
                sc.pp.filter_genes(adata_sub, **kwargs)
                surviving = set(adata_sub.var_names)
                self.adata = self.adata[:, self.adata.var_names.isin(surviving)].copy()
            else:
                sc.pp.filter_genes(self.adata, **kwargs)

        n_genes_after = self.n_genes

        # Invalidate normalized cache since data changed
        self._normalized_adata = None

        mask_cleared = self._regenerate_gene_mask_after_var_change()

        result = {
            'n_genes_before': n_genes_before,
            'n_genes_after': n_genes_after,
            'n_genes_removed': n_genes_before - n_genes_after,
            'gene_mask_cleared': mask_cleared,
        }
        self._log_action('filter_genes', kwargs, result)
        return result

    def run_exclude_genes(
        self,
        gene_names: list[str] | None = None,
        patterns: list[str] | None = None,
    ) -> dict[str, Any]:
        """Remove genes by exact name or regex pattern.

        Args:
            gene_names: List of gene names to remove (exact match)
            patterns: List of regex patterns to match against gene names
                (e.g. "^mt-" for mitochondrial, "^Gm\\d+" for predicted genes)

        Returns:
            Dict with before/after gene counts and removed gene names
        """
        import re

        n_genes_before = self.n_genes
        names = self.adata.var_names

        mask = np.zeros(len(names), dtype=bool)

        # Exact name matches
        if gene_names:
            name_set = set(gene_names)
            mask |= names.isin(name_set)

        # Pattern matches
        if patterns:
            for pattern in patterns:
                try:
                    mask |= names.str.match(pattern)
                except re.error as e:
                    raise ValueError(f"Invalid regex pattern '{pattern}': {e}")

        removed_genes = names[mask].tolist()
        n_removed = int(mask.sum())

        if n_removed > 0:
            self.adata = self.adata[:, ~mask].copy()
            self._normalized_adata = None

        mask_cleared = self._regenerate_gene_mask_after_var_change()

        result = {
            'n_genes_before': n_genes_before,
            'n_genes_after': self.n_genes,
            'n_genes_removed': n_removed,
            'removed_genes': removed_genes[:100],  # Cap list for large removals
            'removed_genes_total': n_removed,
            'gene_mask_cleared': mask_cleared,
        }
        self._log_action('exclude_genes', {
            'gene_names': gene_names,
            'patterns': patterns,
        }, result)
        return result

    def run_filter_cells(
        self,
        min_counts: int | None = None,
        max_counts: int | None = None,
        min_genes: int | None = None,
        max_genes: int | None = None,
        active_cell_indices: list[int] | None = None,
    ) -> dict[str, Any]:
        """Filter cells based on counts or number of genes expressed.

        Args:
            min_counts: Minimum total counts for a cell
            max_counts: Maximum total counts for a cell
            min_genes: Minimum number of genes expressed in the cell
            max_genes: Maximum number of genes expressed in the cell
            active_cell_indices: If provided, only evaluate these cells for filtering

        Returns:
            Dict with before/after cell counts
        """
        n_cells_before = self.n_cells

        kwargs = {}
        if min_counts is not None:
            kwargs['min_counts'] = min_counts
        if max_counts is not None:
            kwargs['max_counts'] = max_counts
        if min_genes is not None:
            kwargs['min_genes'] = min_genes
        if max_genes is not None:
            kwargs['max_genes'] = max_genes

        if kwargs:
            adata_sub, indices = self._get_active_adata(active_cell_indices)
            if indices is not None:
                # Run filter on subset copy to find which cells fail
                adata_test = adata_sub.copy()
                n_before_sub = adata_test.n_obs
                sc.pp.filter_cells(adata_test, **kwargs)
                n_after_sub = adata_test.n_obs
                # Identify surviving cell names
                surviving_names = set(adata_test.obs_names)
                # Find the indices in the full adata that failed
                failing_mask = np.ones(self.n_cells, dtype=bool)
                for idx in indices:
                    cell_name = self.adata.obs_names[idx]
                    if cell_name not in surviving_names:
                        failing_mask[idx] = False
                self.adata = self.adata[failing_mask].copy()
            else:
                sc.pp.filter_cells(self.adata, **kwargs)

        n_cells_after = self.n_cells

        # Invalidate normalized cache since data changed
        self._normalized_adata = None

        result = {
            'n_cells_before': n_cells_before,
            'n_cells_after': n_cells_after,
            'n_cells_removed': n_cells_before - n_cells_after,
        }
        self._log_action('filter_cells', kwargs, result)
        return result

    def delete_cells(
        self,
        cell_indices: list[int],
    ) -> dict[str, Any]:
        """Permanently remove specific cells from the dataset.

        Args:
            cell_indices: List of cell indices to remove.

        Returns:
            Dict with before/after cell counts.
        """
        if not cell_indices:
            raise ValueError("No cell indices provided.")

        indices = np.array(cell_indices)
        if indices.max() >= self.n_cells or indices.min() < 0:
            raise ValueError(
                f"Cell indices out of range [0, {self.n_cells - 1}]."
            )

        n_cells_before = self.n_cells

        # Build keep mask (True for cells to keep)
        keep_mask = np.ones(self.n_cells, dtype=bool)
        keep_mask[indices] = False

        self.adata = self.adata[keep_mask].copy()

        # Invalidate normalized cache
        self._normalized_adata = None

        n_cells_after = self.n_cells

        result = {
            'n_cells_before': n_cells_before,
            'n_cells_after': n_cells_after,
            'n_cells_deleted': n_cells_before - n_cells_after,
        }
        self._log_action('delete_cells', {'n_indices': len(cell_indices)}, result)
        return result

    def run_normalize_total(
        self,
        target_sum: float | None = None,
        active_cell_indices: list[int] | None = None,
    ) -> dict[str, Any]:
        """Normalize total counts per cell.

        Args:
            target_sum: Target sum of counts per cell. If None, uses median.
            active_cell_indices: If provided, only normalize these cells

        Returns:
            Dict with operation status
        """
        from scipy import sparse

        kwargs = {}
        if target_sum is not None:
            kwargs['target_sum'] = target_sum

        adata_sub, indices = self._get_active_adata(active_cell_indices)
        if indices is not None:
            sc.pp.normalize_total(adata_sub, **kwargs)
            # Write back — normalize_total only scales rows, preserving sparsity
            if sparse.issparse(self.adata.X):
                csr = self.adata.X.tocsr()
                sub_csr = adata_sub.X.tocsr() if sparse.issparse(adata_sub.X) else sparse.csr_matrix(adata_sub.X)
                # Vectorized: flag data entries belonging to active rows
                row_flag = np.zeros(csr.shape[0], dtype=bool)
                row_flag[indices] = True
                entry_mask = np.repeat(row_flag, np.diff(csr.indptr))
                csr.data[entry_mask] = sub_csr.data
                self.adata.X = csr
            else:
                self.adata.X[indices] = adata_sub.X if not sparse.issparse(adata_sub.X) else adata_sub.X.toarray()
        else:
            sc.pp.normalize_total(self.adata, **kwargs)

        # Invalidate normalized cache
        self._normalized_adata = None

        result = {'status': 'completed', 'target_sum': target_sum}
        self._log_action('normalize_total', kwargs, result)
        return result

    def run_log1p(
        self,
        active_cell_indices: list[int] | None = None,
    ) -> dict[str, Any]:
        """Apply log1p transformation to the data.

        Args:
            active_cell_indices: If provided, only transform these cells

        Returns:
            Dict with operation status
        """
        from scipy import sparse

        # Validate indices without copying (log1p operates in-place)
        indices = self._validate_cell_indices(active_cell_indices)
        if indices is not None:
            # In-place transform — log1p(0)=0 so sparsity is preserved
            if sparse.issparse(self.adata.X):
                csr = self.adata.X.tocsr()
                row_flag = np.zeros(csr.shape[0], dtype=bool)
                row_flag[indices] = True
                entry_mask = np.repeat(row_flag, np.diff(csr.indptr))
                np.log1p(csr.data[entry_mask], out=csr.data[entry_mask])
                self.adata.X = csr
            else:
                np.log1p(self.adata.X[indices], out=self.adata.X[indices])
        else:
            sc.pp.log1p(self.adata)

        # Invalidate normalized cache
        self._normalized_adata = None

        result = {'status': 'completed'}
        self._log_action('log1p', {}, result)
        return result

    def run_highly_variable_genes(
        self,
        n_top_genes: int | None = None,
        min_mean: float = 0.0125,
        max_mean: float = 3.0,
        min_disp: float = 0.5,
        flavor: str = 'seurat',
        n_bins: int = 20,
        subset: bool = False,
        active_cell_indices: list[int] | None = None,
    ) -> dict[str, Any]:
        """Identify highly variable genes.

        Adds 'highly_variable' boolean column to .var.

        Args:
            n_top_genes: Number of top genes to select (overrides min/max thresholds)
            min_mean: Minimum mean expression threshold
            max_mean: Maximum mean expression threshold
            min_disp: Minimum dispersion threshold
            flavor: Method ('seurat', 'cell_ranger', 'seurat_v3')
            n_bins: Number of bins for dispersion normalization
            subset: If True, subset adata to only highly variable genes (destructive)
            active_cell_indices: If provided, compute HVGs using only these cells

        Returns:
            Dict with operation status and number of HVGs
        """
        adata_sub, indices = self._get_active_adata(active_cell_indices)
        if indices is not None:
            from scipy import sparse
            # Drop genes with zero expression in the subset to avoid
            # degenerate bin edges in scanpy's HVG binning step
            if sparse.issparse(adata_sub.X):
                gene_totals = np.asarray(adata_sub.X.sum(axis=0)).ravel()
            else:
                gene_totals = np.asarray(adata_sub.X.sum(axis=0)).ravel()
            expressed_mask = gene_totals > 0
            adata_hvg = adata_sub[:, expressed_mask].copy()

            # Compute HVGs on subset (always with subset=False to get annotations)
            sc.pp.highly_variable_genes(
                adata_hvg,
                n_top_genes=n_top_genes,
                min_mean=min_mean,
                max_mean=max_mean,
                min_disp=min_disp,
                flavor=flavor,
                n_bins=n_bins,
                subset=False,
            )
            # Map results back to full gene set — unexpressed genes are not HVG
            for col in ['highly_variable', 'means', 'dispersions', 'dispersions_norm']:
                if col in adata_hvg.var.columns:
                    default = False if col == 'highly_variable' else 0.0
                    full_col = pd.Series(default, index=self.adata.var_names, dtype=adata_hvg.var[col].dtype)
                    full_col.loc[adata_hvg.var_names] = adata_hvg.var[col]
                    self.adata.var[col] = full_col.values
            # Apply subset on full adata if requested
            if subset:
                self.adata = self.adata[:, self.adata.var['highly_variable']].copy()
        else:
            sc.pp.highly_variable_genes(
                self.adata,
                n_top_genes=n_top_genes,
                min_mean=min_mean,
                max_mean=max_mean,
                min_disp=min_disp,
                flavor=flavor,
                n_bins=n_bins,
                subset=subset,
            )

        n_hvg = int(self.adata.var['highly_variable'].sum())

        result = {
            'status': 'completed',
            'n_highly_variable': n_hvg,
            'n_total_genes': self.n_genes,
            'flavor': flavor,
        }
        self._log_action('highly_variable_genes', {
            'n_top_genes': n_top_genes,
            'min_mean': min_mean,
            'max_mean': max_mean,
            'min_disp': min_disp,
            'flavor': flavor,
            'subset': subset,
        }, result)
        return result

    def run_pca(
        self,
        n_comps: int = 50,
        svd_solver: str = 'arpack',
        gene_subset: str | list[str] | dict[str, Any] | None = None,
        use_highly_variable: bool | None = None,
        active_cell_indices: list[int] | None = None,
    ) -> dict[str, Any]:
        """Run PCA dimensionality reduction.

        Args:
            n_comps: Number of principal components to compute
            svd_solver: SVD solver to use ('arpack', 'randomized', 'auto')
            gene_subset: Subset of genes to use. Can be:
                - None: use default behavior (highly_variable if available, else all)
                - str: boolean column name from .var
                - list[str]: explicit list of gene names
                - dict: {'columns': [...], 'operation': 'intersection'|'union'}
            use_highly_variable: Deprecated, use gene_subset instead.
                If True and gene_subset is None, uses 'highly_variable' column.
            active_cell_indices: If provided, compute PCA on these cells only;
                inactive cells get NaN in X_pca.

        Returns:
            Dict with operation status and variance explained
        """
        # Handle legacy use_highly_variable parameter
        if gene_subset is None and use_highly_variable is True:
            if 'highly_variable' in self.adata.var.columns:
                gene_subset = 'highly_variable'

        # Resolve gene subset
        if gene_subset is not None:
            gene_mask, subset_type, subset_metadata = self._resolve_gene_mask(gene_subset)
            n_genes_used = int(gene_mask.sum())

            # Create a temporary subset for PCA
            adata_pca = self.adata[:, gene_mask].copy()
        else:
            # Default scanpy behavior: use highly_variable if present
            adata_pca = self.adata
            subset_type = 'default'
            n_genes_used = self.n_genes
            if 'highly_variable' in self.adata.var.columns:
                n_genes_used = int(self.adata.var['highly_variable'].sum())
                subset_type = 'highly_variable (auto)'

        # Apply cell mask
        adata_sub, cell_indices = self._get_active_adata(active_cell_indices)
        if cell_indices is not None:
            # Subset cells from the (possibly gene-subsetted) adata
            if gene_subset is not None:
                adata_pca = adata_pca[cell_indices].copy()
            else:
                adata_pca = self.adata[cell_indices].copy()

        # Limit n_comps to valid range
        max_comps = min(adata_pca.n_obs - 1, adata_pca.n_vars - 1)
        n_comps = min(n_comps, max_comps)

        # Run PCA on subset
        sc.tl.pca(adata_pca, n_comps=n_comps, svd_solver=svd_solver)

        # Copy results back to main adata
        if cell_indices is not None:
            # Store X_pca with NaN for inactive cells
            full_pca = np.full((self.n_cells, n_comps), np.nan)
            full_pca[cell_indices] = adata_pca.obsm['X_pca']
            self.adata.obsm['X_pca'] = full_pca
        else:
            self.adata.obsm['X_pca'] = adata_pca.obsm['X_pca']
        self.adata.uns['pca'] = adata_pca.uns['pca']

        # Copy gene loadings back as a full-size (n_genes, n_comps) matrix
        # with NaN rows for genes not included in the subset. Downstream
        # code (get_pca_loadings, create_pca_subset) expects varm['PCs']
        # to be present and correctly shaped for self.adata.n_vars.
        if 'PCs' in adata_pca.varm:
            full_pcs = np.full((self.n_genes, n_comps), np.nan)
            if gene_subset is not None:
                full_pcs[gene_mask, :] = adata_pca.varm['PCs']
            else:
                full_pcs[:, :] = adata_pca.varm['PCs']
            self.adata.varm['PCs'] = full_pcs
            self.adata.uns['pca']['gene_subset'] = {
                'type': subset_type,
                'n_genes': n_genes_used,
            }

        # Get variance explained
        variance_ratio = self.adata.uns['pca']['variance_ratio'][:10].tolist()

        result = {
            'status': 'completed',
            'n_comps': n_comps,
            'variance_explained_top10': variance_ratio,
            'embedding_name': 'X_pca',
            'gene_subset_type': subset_type,
            'n_genes_used': n_genes_used,
        }
        # Clear derived PC subsets — they reference columns of the previous
        # X_pca and become stale on re-run. obsm and varm are NOT touched by
        # sc.tl.pca (only 'X_pca' and 'PCs' are overwritten), so we scan here.
        # sc.tl.pca does replace adata.uns['pca'] wholesale, so variance_ratio_*
        # and the 'subsets' metadata dict are already gone — the uns pops below
        # are defensive against stale obsm keys from externally loaded h5ad
        # files and to keep the invariant explicit.
        cleared_subsets: list[str] = []
        for key in list(self.adata.obsm.keys()):
            if key.startswith('X_pca_') and key != 'X_pca':
                suffix = key[len('X_pca_'):]
                self.adata.obsm.pop(key, None)
                self.adata.varm.pop(f"PCs_{suffix}", None)
                if 'pca' in self.adata.uns and isinstance(self.adata.uns['pca'], dict):
                    self.adata.uns['pca'].pop(f"variance_ratio_{suffix}", None)
                    subsets_meta = self.adata.uns['pca'].get('subsets', {})
                    if isinstance(subsets_meta, dict):
                        subsets_meta.pop(suffix, None)
                cleared_subsets.append(key)
        if cleared_subsets:
            result['cleared_subsets'] = cleared_subsets

        self._log_action('pca', {
            'n_comps': n_comps,
            'svd_solver': svd_solver,
            'gene_subset': gene_subset,
        }, result)
        return result

    def get_pca_loadings(self, top_n: int = 10) -> dict[str, Any]:
        """Return top +/- loading genes per computed PC.

        Reads self.adata.varm['PCs'] and self.adata.uns['pca']['variance_ratio'].
        Gene rows containing NaN loadings (from subset-PCA runs) are excluded
        from per-PC rankings; up to top_n valid genes are returned per side.

        Raises:
            ValueError: if PCA has not been run or loadings are missing.

        Returns:
            {
              'n_pcs': int,
              'top_n': int,
              'pcs': [
                {
                  'index': 0,                 # zero-based
                  'variance_ratio': 0.127,
                  'positive': [{'gene': 'MALAT1', 'loading': 0.18}, ...],
                  'negative': [{'gene': 'MT-CO1', 'loading': -0.15}, ...],
                }, ...
              ]
            }
        """
        if 'pca' not in self.adata.uns:
            raise ValueError("PCA has not been run. Run pca first.")
        if 'PCs' not in self.adata.varm:
            raise ValueError("PC loadings are unavailable (varm['PCs'] missing). Re-run PCA.")

        pcs_matrix = np.asarray(self.adata.varm['PCs'])
        if pcs_matrix.ndim != 2:
            raise ValueError(f"Unexpected varm['PCs'] shape: {pcs_matrix.shape}")

        n_genes, n_comps = pcs_matrix.shape
        var_ratio = np.asarray(self.adata.uns['pca'].get('variance_ratio', []))
        gene_names = list(self.adata.var_names)
        top_n = max(1, int(top_n))

        pcs_out = []
        for i in range(n_comps):
            col = pcs_matrix[:, i]
            valid = ~np.isnan(col)
            valid_indices = np.where(valid)[0]
            valid_loadings = col[valid_indices]

            # Positive side: sort descending, take top_n
            pos_order = valid_indices[np.argsort(-valid_loadings)][:top_n]
            positive = [
                {'gene': gene_names[int(j)], 'loading': float(col[int(j)])}
                for j in pos_order
                if col[int(j)] > 0
            ]

            # Negative side: sort ascending, take top_n
            neg_order = valid_indices[np.argsort(valid_loadings)][:top_n]
            negative = [
                {'gene': gene_names[int(j)], 'loading': float(col[int(j)])}
                for j in neg_order
                if col[int(j)] < 0
            ]

            pcs_out.append({
                'index': i,
                'variance_ratio': float(var_ratio[i]) if i < len(var_ratio) else None,
                'positive': positive,
                'negative': negative,
            })

        return {
            'n_pcs': n_comps,
            'top_n': top_n,
            'pcs': pcs_out,
        }

    def create_pca_subset(
        self,
        drop_pc_indices: list[int],
        suffix: str | None = None,
    ) -> dict[str, Any]:
        """Create derived PCA slots that exclude specific 1-indexed PCs.

        Writes:
          - obsm[f'X_pca_{suffix}'] — base embedding with dropped columns removed.
          - varm[f'PCs_{suffix}'] — matching loadings with dropped columns removed.
          - uns['pca'][f'variance_ratio_{suffix}'] — matching variance ratios.
          - uns['pca']['subsets'][suffix] = {'dropped_pcs': [i, j, ...]}
            (round-trips exact indices regardless of suffix).

        Raises:
            ValueError: missing PCA, empty indices, out-of-range, all-dropped.
            ValueError: suffix collision with existing obsm key.
        """
        if 'X_pca' not in self.adata.obsm:
            raise ValueError("PCA has not been run. Run pca first.")
        if not drop_pc_indices:
            raise ValueError("drop_pc_indices must contain at least one PC.")

        base_embed = np.asarray(self.adata.obsm['X_pca'])
        n_cells, n_pcs = base_embed.shape

        # Convert from 1-indexed user-facing to 0-indexed column positions.
        idx = np.asarray(drop_pc_indices, dtype=int) - 1
        if (idx < 0).any():
            raise ValueError("drop_pc_indices must be >= 1 (PC numbers are 1-indexed).")
        if (idx >= n_pcs).any():
            raise ValueError(
                f"drop_pc_indices contains entries > {n_pcs} (total PCs available)."
            )
        idx = np.unique(idx)
        keep = np.setdiff1d(np.arange(n_pcs), idx, assume_unique=False)
        if keep.size == 0:
            raise ValueError("Cannot drop all PCs.")

        dropped_1indexed = sorted(int(i + 1) for i in idx)

        if suffix is None or suffix == '':
            suffix = f"noPC{'_'.join(str(i) for i in dropped_1indexed)}"

        new_obsm_key = f"X_pca_{suffix}"
        if new_obsm_key in self.adata.obsm:
            raise ValueError(f"A PC subset named '{suffix}' already exists.")

        # Write the three companion slots.
        self.adata.obsm[new_obsm_key] = base_embed[:, keep]

        varm_key = None
        if 'PCs' in self.adata.varm:
            varm_key = f"PCs_{suffix}"
            self.adata.varm[varm_key] = np.asarray(self.adata.varm['PCs'])[:, keep]

        var_ratio_key = None
        if 'pca' in self.adata.uns and isinstance(self.adata.uns['pca'], dict):
            if 'variance_ratio' in self.adata.uns['pca']:
                var_ratio_key = f"variance_ratio_{suffix}"
                self.adata.uns['pca'][var_ratio_key] = np.asarray(
                    self.adata.uns['pca']['variance_ratio']
                )[keep]
            # Record the dropped indices for round-tripping in list_pca_subsets.
            subsets_meta = self.adata.uns['pca'].setdefault('subsets', {})
            subsets_meta[suffix] = {'dropped_pcs': dropped_1indexed}

        result = {
            'obsm_key': new_obsm_key,
            'varm_key': varm_key,
            'variance_ratio_key': var_ratio_key,
            'suffix': suffix,
            'n_pcs_kept': int(keep.size),
            'dropped_pcs': dropped_1indexed,
        }
        self._log_action('create_pca_subset', {
            'drop_pc_indices': dropped_1indexed,
            'suffix': suffix,
        }, result)
        return result

    def list_pca_subsets(self) -> list[dict[str, Any]]:
        """List every derived PC subset in adata.obsm.

        Iterates obsm keys with prefix 'X_pca_' (excluding the exact key
        'X_pca'). For each, reports obsm_key, suffix, n_pcs_kept, and
        dropped_pcs (from uns['pca']['subsets'][suffix] when present,
        otherwise []).
        """
        out: list[dict[str, Any]] = []
        subsets_meta = {}
        if 'pca' in self.adata.uns and isinstance(self.adata.uns['pca'], dict):
            subsets_meta = self.adata.uns['pca'].get('subsets', {}) or {}

        for key in sorted(self.adata.obsm.keys()):
            if not key.startswith('X_pca_') or key == 'X_pca':
                continue
            suffix = key[len('X_pca_'):]
            arr = np.asarray(self.adata.obsm[key])
            n_pcs_kept = int(arr.shape[1]) if arr.ndim == 2 else 0
            meta = subsets_meta.get(suffix, {})
            dropped = [int(x) for x in meta.get('dropped_pcs', [])]
            out.append({
                'obsm_key': key,
                'suffix': suffix,
                'n_pcs_kept': n_pcs_kept,
                'dropped_pcs': dropped,
            })
        return out

    def delete_pca_subset(self, obsm_key: str) -> None:
        """Delete a derived PC subset's obsm, varm, variance_ratio, and
        uns['pca']['subsets'] entries.

        Raises:
            ValueError: if obsm_key == 'X_pca', is missing, or doesn't start
                with 'X_pca_'.
        """
        if obsm_key == 'X_pca':
            raise ValueError("Cannot delete the base X_pca embedding.")
        if not obsm_key.startswith('X_pca_'):
            raise ValueError(f"'{obsm_key}' is not a derived PC subset.")
        if obsm_key not in self.adata.obsm:
            raise ValueError(f"'{obsm_key}' not found in obsm.")

        suffix = obsm_key[len('X_pca_'):]
        self.adata.obsm.pop(obsm_key, None)
        self.adata.varm.pop(f"PCs_{suffix}", None)
        if 'pca' in self.adata.uns and isinstance(self.adata.uns['pca'], dict):
            self.adata.uns['pca'].pop(f"variance_ratio_{suffix}", None)
            subsets_meta = self.adata.uns['pca'].get('subsets', {})
            if isinstance(subsets_meta, dict):
                subsets_meta.pop(suffix, None)
        self._log_action('delete_pca_subset', {'obsm_key': obsm_key}, None)

    def run_neighbors(
        self,
        n_neighbors: int = 15,
        n_pcs: int | None = None,
        metric: str = 'euclidean',
        use_rep: str | None = None,
        active_cell_indices: list[int] | None = None,
    ) -> dict[str, Any]:
        """Compute neighborhood graph.

        Args:
            n_neighbors: Number of neighbors to use
            n_pcs: Number of PCs to use (None = use all)
            metric: Distance metric
            use_rep: obsm key to use as the representation (e.g. 'X_pca_noPC2_5').
                None or 'X_pca' preserves the default scanpy path (uses X_pca).
                Any other value must exist in adata.obsm.
            active_cell_indices: If provided, compute neighbors on these cells only;
                results are remapped into full-size sparse matrices.

        Returns:
            Dict with operation status
        """
        from scipy.sparse import coo_matrix

        # Check prerequisites
        prereq = self.check_prerequisites('neighbors')
        if not prereq['satisfied']:
            raise ValueError(f"Prerequisites not met: {prereq['missing']}")

        kwargs = {
            'n_neighbors': n_neighbors,
            'metric': metric,
        }
        if n_pcs is not None:
            kwargs['n_pcs'] = n_pcs

        # Resolve and validate use_rep. None / 'X_pca' preserve the existing
        # default path. Any other value must exist in adata.obsm.
        rep_key = use_rep if use_rep and use_rep != 'X_pca' else None
        if rep_key is not None:
            if rep_key not in self.adata.obsm:
                raise ValueError(
                    f"use_rep '{rep_key}' not found in obsm. "
                    f"Create it via /api/scanpy/pca_subsets first."
                )
            kwargs['use_rep'] = rep_key

        cell_indices = self._validate_cell_indices(active_cell_indices)
        if cell_indices is not None:
            # Build a subset AnnData with PCA from the active cells
            import anndata as ad
            source_key = rep_key if rep_key is not None else 'X_pca'
            pca_full = self.adata.obsm[source_key]
            pca_sub = pca_full[cell_indices]
            adata_sub = ad.AnnData(obs=pd.DataFrame(index=self.adata.obs_names[cell_indices]))
            adata_sub.obsm[source_key] = pca_sub

            sc.pp.neighbors(adata_sub, **kwargs)

            # Remap sparse obsp matrices to full size
            n_full = self.n_cells
            for key in ['connectivities', 'distances']:
                if key in adata_sub.obsp:
                    sub_coo = adata_sub.obsp[key].tocoo()
                    full_rows = cell_indices[sub_coo.row]
                    full_cols = cell_indices[sub_coo.col]
                    full_mat = coo_matrix(
                        (sub_coo.data, (full_rows, full_cols)),
                        shape=(n_full, n_full),
                    )
                    self.adata.obsp[key] = full_mat.tocsr()

            # Copy uns['neighbors'] metadata
            self.adata.uns['neighbors'] = adata_sub.uns['neighbors']
        else:
            sc.pp.neighbors(self.adata, **kwargs)

        result = {'status': 'completed', 'n_neighbors': n_neighbors}
        self._log_action('neighbors', kwargs, result)
        return result

    def run_umap(
        self,
        min_dist: float = 0.5,
        spread: float = 1.0,
        n_components: int = 2,
        active_cell_indices: list[int] | None = None,
    ) -> dict[str, Any]:
        """Compute UMAP embedding.

        Args:
            min_dist: Minimum distance between points
            spread: Spread of the embedding
            n_components: Number of dimensions
            active_cell_indices: If provided, compute UMAP on these cells only;
                inactive cells get NaN coordinates.

        Returns:
            Dict with operation status and embedding name
        """
        # Check prerequisites
        prereq = self.check_prerequisites('umap')
        if not prereq['satisfied']:
            raise ValueError(f"Prerequisites not met: {prereq['missing']}")

        cell_indices = self._validate_cell_indices(active_cell_indices)
        if cell_indices is not None:
            # Build subset AnnData with PCA and neighbor graph
            import anndata as ad
            pca_full = self.adata.obsm['X_pca']
            adata_sub = ad.AnnData(obs=pd.DataFrame(index=self.adata.obs_names[cell_indices]))
            adata_sub.obsm['X_pca'] = pca_full[cell_indices]

            # Extract subset neighbor graph from full-size obsp
            for key in ['connectivities', 'distances']:
                if key in self.adata.obsp:
                    full_mat = self.adata.obsp[key].tocsr()
                    adata_sub.obsp[key] = full_mat[np.ix_(cell_indices, cell_indices)]

            if 'neighbors' in self.adata.uns:
                adata_sub.uns['neighbors'] = self.adata.uns['neighbors']

            sc.tl.umap(adata_sub, min_dist=min_dist, spread=spread, n_components=n_components)

            # Store with NaN for inactive cells
            full_umap = np.full((self.n_cells, n_components), np.nan)
            full_umap[cell_indices] = adata_sub.obsm['X_umap']
            self.adata.obsm['X_umap'] = full_umap
        else:
            sc.tl.umap(self.adata, min_dist=min_dist, spread=spread, n_components=n_components)

        result = {
            'status': 'completed',
            'embedding_name': 'X_umap',
            'n_components': n_components,
        }
        self._log_action('umap', {'min_dist': min_dist, 'spread': spread, 'n_components': n_components}, result)
        return result

    def run_leiden(
        self,
        resolution: float = 1.0,
        key_added: str = 'leiden',
        active_cell_indices: list[int] | None = None,
    ) -> dict[str, Any]:
        """Run Leiden clustering.

        Args:
            resolution: Resolution parameter (higher = more clusters)
            key_added: Key to add to obs for cluster labels
            active_cell_indices: If provided, cluster only these cells;
                inactive cells are labeled 'unassigned'.

        Returns:
            Dict with operation status and cluster info
        """
        # Check prerequisites
        prereq = self.check_prerequisites('leiden')
        if not prereq['satisfied']:
            raise ValueError(f"Prerequisites not met: {prereq['missing']}")

        cell_indices = self._validate_cell_indices(active_cell_indices)
        if cell_indices is not None:
            # Build subset AnnData with neighbor graph
            import anndata as ad
            adata_sub = ad.AnnData(obs=pd.DataFrame(index=self.adata.obs_names[cell_indices]))

            # Extract subset neighbor graph from full-size obsp
            for key in ['connectivities', 'distances']:
                if key in self.adata.obsp:
                    full_mat = self.adata.obsp[key].tocsr()
                    adata_sub.obsp[key] = full_mat[np.ix_(cell_indices, cell_indices)]

            if 'neighbors' in self.adata.uns:
                adata_sub.uns['neighbors'] = self.adata.uns['neighbors']

            sc.tl.leiden(adata_sub, resolution=resolution, key_added=key_added)

            # Map labels back with 'unassigned' for inactive cells
            sub_categories = list(adata_sub.obs[key_added].cat.categories)
            all_categories = sub_categories + ['unassigned']
            full_labels = ['unassigned'] * self.n_cells
            for i, idx in enumerate(cell_indices):
                full_labels[idx] = str(adata_sub.obs[key_added].iloc[i])
            self.adata.obs[key_added] = pd.Categorical(
                full_labels, categories=all_categories,
            )

            n_clusters = len(sub_categories)
        else:
            sc.tl.leiden(self.adata, resolution=resolution, key_added=key_added)
            n_clusters = len(self.adata.obs[key_added].cat.categories)

        result = {
            'status': 'completed',
            'key_added': key_added,
            'n_clusters': n_clusters,
            'resolution': resolution,
        }
        self._log_action('leiden', {'resolution': resolution, 'key_added': key_added}, result)
        return result

    # =========================================================================
    # Gene analysis methods
    # =========================================================================

    @staticmethod
    def _find_elbow_kneedle(values: np.ndarray, sensitivity: float = 1.0) -> int:
        """Find elbow point using Kneedle algorithm.

        Args:
            values: Array of values (e.g., variance ratios)
            sensitivity: Sensitivity parameter (higher = more sensitive)

        Returns:
            Index of the elbow point
        """
        n = len(values)
        if n < 2:
            return 0

        # Normalize x and y to [0, 1]
        x = np.arange(n)
        x_norm = (x - x.min()) / (x.max() - x.min() + 1e-10)
        y_norm = (values - values.min()) / (values.max() - values.min() + 1e-10)

        # Calculate differences from the diagonal line
        # For a decreasing curve, we look for max distance below the line
        differences = y_norm - (1 - x_norm)

        # Apply sensitivity - look for where the curve deviates significantly
        threshold = sensitivity * np.std(differences)

        # Find the elbow: first point where difference drops below threshold
        # after the initial steep decline
        for i in range(1, n - 1):
            if differences[i] < -threshold:
                # Check if we're past the steep part
                local_slope = values[i] - values[i - 1]
                next_slope = values[i + 1] - values[i] if i + 1 < n else 0
                if abs(next_slope) < abs(local_slope) * 0.5:
                    return i

        # Fallback: use maximum curvature
        if n > 2:
            second_derivative = np.diff(np.diff(values))
            return int(np.argmax(np.abs(second_derivative))) + 1

        return min(n - 1, 10)

    def run_gene_pca(
        self,
        n_comps: int | None = None,
        scale: bool = True,
        use_kneedle: bool = True,
        max_comps: int = 100,
        gene_subset: str | list[str] | dict[str, Any] | None = None,
        active_cell_indices: list[int] | None = None,
    ) -> dict[str, Any]:
        """Run PCA on genes (transposed expression matrix).

        Computes gene embeddings based on their expression patterns across cells.
        Results are stored in .varm['X_gene_pca'] and variance info in .uns['gene_pca'].

        Args:
            n_comps: Number of components. If None and use_kneedle=True, auto-detect.
            scale: Whether to z-score scale genes before PCA (recommended)
            use_kneedle: Whether to use Kneedle algorithm for auto PC selection
            max_comps: Maximum components to compute before Kneedle selection
            gene_subset: Subset of genes to use. Can be:
                - None: use all genes
                - str: boolean column name from .var (e.g., 'highly_variable', 'spatially_variable')
                - list[str]: explicit list of gene names
                - dict: {'columns': ['col1', 'col2'], 'operation': 'intersection'|'union'}
                  for combining multiple boolean columns with AND/OR logic
            active_cell_indices: If provided, use only these cells for gene PCA

        Returns:
            Dict with operation status, n_comps used, and variance explained
        """
        from scipy import sparse
        from sklearn.decomposition import PCA
        from sklearn.preprocessing import StandardScaler

        # Resolve gene subset to boolean mask
        gene_mask, subset_type, subset_metadata = self._resolve_gene_mask(gene_subset)

        # Store the gene subset info for downstream use
        self.adata.uns['gene_pca_subset'] = {
            'type': subset_type,
            'genes': self.adata.var_names[gene_mask].tolist(),
            'n_genes': int(gene_mask.sum()),
            **subset_metadata,
        }

        # Subset cells if active_cell_indices provided, then genes, then transpose
        cell_indices = self._validate_cell_indices(active_cell_indices)
        if cell_indices is not None:
            X = self.adata.X[cell_indices][:, gene_mask].T
        else:
            X = self.adata.X[:, gene_mask].T
        if sparse.issparse(X):
            X = X.toarray()
        X = np.asarray(X, dtype=np.float64)

        # Handle NaN/Inf
        X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)

        # Optional scaling (z-score each gene's expression across cells)
        if scale:
            scaler = StandardScaler()
            X = scaler.fit_transform(X)

        # Determine number of components to compute
        n_genes, n_cells = X.shape
        max_possible = min(n_genes - 1, n_cells - 1, max_comps)

        if n_comps is not None:
            # User specified
            n_comps_compute = min(n_comps, max_possible)
            n_comps_final = n_comps_compute
        elif use_kneedle:
            # Compute more PCs, then use Kneedle to select
            n_comps_compute = max_possible
        else:
            # Default to 50
            n_comps_compute = min(50, max_possible)
            n_comps_final = n_comps_compute

        # Run PCA
        pca = PCA(n_components=n_comps_compute)
        gene_pcs = pca.fit_transform(X)
        variance_ratio = pca.explained_variance_ratio_

        # Apply Kneedle if needed
        if n_comps is None and use_kneedle:
            elbow_idx = self._find_elbow_kneedle(variance_ratio)
            n_comps_final = max(elbow_idx + 1, 5)  # At least 5 PCs
            n_comps_final = min(n_comps_final, n_comps_compute)
        elif n_comps is None:
            n_comps_final = n_comps_compute

        # Store truncated results
        # If using a subset, store full-sized array with NaN for excluded genes
        full_gene_pcs = np.full((self.n_genes, n_comps_final), np.nan)
        full_gene_pcs[gene_mask, :] = gene_pcs[:, :n_comps_final]
        self.adata.varm['X_gene_pca'] = full_gene_pcs

        # Store variance info and subset metadata
        self.adata.uns['gene_pca'] = {
            'variance_ratio': variance_ratio.tolist(),
            'variance': pca.explained_variance_.tolist(),
            'n_comps': n_comps_final,
            'n_comps_computed': n_comps_compute,
            'scaled': scale,
            'elbow_index': elbow_idx if (n_comps is None and use_kneedle) else None,
            'gene_subset_type': subset_type,
            'n_genes_used': int(gene_mask.sum()),
        }

        cumulative_var = float(np.sum(variance_ratio[:n_comps_final]))

        result = {
            'status': 'completed',
            'n_comps': n_comps_final,
            'n_comps_computed': n_comps_compute,
            'cumulative_variance': cumulative_var,
            'scaled': scale,
            'elbow_detected': elbow_idx if (n_comps is None and use_kneedle) else None,
            'gene_subset_type': subset_type,
            'n_genes_used': int(gene_mask.sum()),
        }
        self._log_action('gene_pca', {
            'n_comps': n_comps,
            'scale': scale,
            'use_kneedle': use_kneedle,
            'gene_subset': gene_subset,
        }, result)
        return result

    def get_cell_pca_variance(self) -> dict[str, Any]:
        """Get cell PCA variance information for visualization.

        Returns:
            Dict with variance ratios, cumulative variance, and elbow point
        """
        if 'pca' not in self.adata.uns:
            raise ValueError("Cell PCA has not been computed. Run pca first.")

        info = self.adata.uns['pca']
        variance_ratio = np.array(info['variance_ratio'])
        cumulative = np.cumsum(variance_ratio)

        # Try elbow detection
        elbow_idx = None
        if len(variance_ratio) > 2:
            elbow_idx = self._find_elbow_kneedle(variance_ratio)

        return {
            'variance_ratio': variance_ratio.tolist(),
            'cumulative_variance': cumulative.tolist(),
            'n_comps_used': len(variance_ratio),
            'n_comps_computed': len(variance_ratio),
            'elbow_index': elbow_idx,
        }

    def get_gene_pca_variance(self) -> dict[str, Any]:
        """Get gene PCA variance information for visualization.

        Returns:
            Dict with variance ratios, cumulative variance, and elbow point
        """
        if 'gene_pca' not in self.adata.uns:
            raise ValueError("Gene PCA has not been computed. Run gene_pca first.")

        info = self.adata.uns['gene_pca']
        variance_ratio = np.array(info['variance_ratio'])
        cumulative = np.cumsum(variance_ratio)

        return {
            'variance_ratio': variance_ratio.tolist(),
            'cumulative_variance': cumulative.tolist(),
            'n_comps_used': info['n_comps'],
            'n_comps_computed': info['n_comps_computed'],
            'elbow_index': info.get('elbow_index'),
        }

    def prepare_gene_neighbors(
        self,
        n_neighbors: int = 15,
        metric: str = 'euclidean',
        basis: str = 'gene_pca',
        gene_subset: str | list[str] | dict[str, Any] | None = None,
        scale: bool = True,
        active_cell_indices: list[int] | None = None,
    ) -> tuple[Callable[[], dict[str, Any]], Callable[[dict[str, Any]], None]]:
        """Prepare gene-gene kNN graph computation (cancellable).

        Validates inputs and snapshots data upfront, then returns a pair of
        functions: ``compute_fn`` (pure, no side-effects) and ``apply_fn``
        (writes results into ``self.adata``).

        Args:
            n_neighbors: Number of neighbors per gene
            metric: Distance metric ('euclidean', 'cosine', 'pearson')
            basis: 'gene_pca' to use PCA embedding, 'expression' to use raw expression
            gene_subset: Gene filtering (only used when basis='expression').
                Can be str (boolean column), list[str] (gene names), or dict (multi-column spec).
            scale: Z-score scale genes before computing neighbors (only used when basis='expression')
            active_cell_indices: Optional cell subset (only used when basis='expression')

        Returns:
            Tuple of (compute_fn, apply_fn)
        """
        from sklearn.neighbors import NearestNeighbors
        from scipy import sparse

        n_genes_total = self.adata.n_vars
        subset_type = 'all'

        if basis == 'gene_pca':
            if 'X_gene_pca' not in self.adata.varm:
                raise ValueError("Gene PCA has not been computed. Run gene_pca first.")

            gene_pcs = self.adata.varm['X_gene_pca'].copy()
            valid_mask = ~np.isnan(gene_pcs[:, 0])
            valid_indices = np.where(valid_mask)[0]
            representation = gene_pcs[valid_mask, :]

            pca_info = self.adata.uns.get('gene_pca', {})
            subset_type = pca_info.get('gene_subset_type', 'all')

        elif basis == 'expression':
            from sklearn.preprocessing import StandardScaler

            if gene_subset is not None:
                gene_mask, subset_type, _ = self._resolve_gene_mask(gene_subset)
            else:
                gene_mask = np.ones(self.adata.n_vars, dtype=bool)

            valid_mask = gene_mask
            valid_indices = np.where(valid_mask)[0]

            cell_indices = self._validate_cell_indices(active_cell_indices)
            if cell_indices is not None:
                X = self.adata.X[cell_indices][:, gene_mask].T
            else:
                X = self.adata.X[:, gene_mask].T

            if sparse.issparse(X):
                X = X.toarray()
            X = np.asarray(X, dtype=np.float64)
            X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)

            if scale:
                scaler = StandardScaler()
                X = scaler.fit_transform(X)

            representation = X
        else:
            raise ValueError(f"Unknown basis: {basis}. Must be 'gene_pca' or 'expression'.")

        n_genes_valid = len(valid_indices)
        if n_genes_valid == 0:
            raise ValueError("No genes with valid embeddings/expression")

        n_neighbors = min(n_neighbors, n_genes_valid - 1)

        # Snapshot all data needed by compute_fn
        snap_representation = representation.copy()
        snap_valid_indices = valid_indices.copy()
        snap_n_genes_total = n_genes_total
        snap_n_genes_valid = n_genes_valid
        snap_n_neighbors = n_neighbors
        snap_metric = metric
        snap_basis = basis
        snap_subset_type = subset_type

        def compute_fn() -> dict[str, Any]:
            from sklearn.neighbors import NearestNeighbors
            from scipy import sparse

            if snap_metric == 'pearson':
                corr_matrix = np.corrcoef(snap_representation)
                corr_matrix = np.nan_to_num(corr_matrix, nan=0.0)
                dist_full = 1.0 - corr_matrix
                np.fill_diagonal(dist_full, 0.0)
                nn = NearestNeighbors(n_neighbors=snap_n_neighbors + 1, metric='precomputed')
                nn.fit(dist_full)
                distances, indices = nn.kneighbors(dist_full)
            else:
                nn = NearestNeighbors(n_neighbors=snap_n_neighbors + 1, metric=snap_metric)
                nn.fit(snap_representation)
                distances, indices = nn.kneighbors(snap_representation)

            rows = []
            cols = []
            dists = []
            for i_valid in range(snap_n_genes_valid):
                i_original = snap_valid_indices[i_valid]
                for j_idx in range(1, snap_n_neighbors + 1):
                    j_valid = indices[i_valid, j_idx]
                    j_original = snap_valid_indices[j_valid]
                    rows.append(i_original)
                    cols.append(j_original)
                    dists.append(distances[i_valid, j_idx])

            dist_matrix = sparse.csr_matrix(
                (dists, (rows, cols)),
                shape=(snap_n_genes_total, snap_n_genes_total)
            )

            conn_weights = [1.0 / (1.0 + d) for d in dists]
            conn_matrix = sparse.csr_matrix(
                (conn_weights, (rows, cols)),
                shape=(snap_n_genes_total, snap_n_genes_total)
            )

            return {
                'dist_matrix': dist_matrix,
                'conn_matrix': conn_matrix,
                'n_neighbors': snap_n_neighbors,
                'metric': snap_metric,
                'basis': snap_basis,
                'n_genes_valid': snap_n_genes_valid,
                'n_genes_total': snap_n_genes_total,
                'subset_type': snap_subset_type,
            }

        def apply_fn(result: dict[str, Any]) -> dict[str, Any]:
            self.adata.varp['gene_distances'] = result['dist_matrix']
            self.adata.varp['gene_connectivities'] = result['conn_matrix']

            self.adata.uns['gene_neighbors'] = {
                'n_neighbors': result['n_neighbors'],
                'metric': result['metric'],
                'basis': result['basis'],
                'n_genes_in_graph': result['n_genes_valid'],
                'gene_subset_type': result['subset_type'],
            }

            status_result = {
                'status': 'completed',
                'n_neighbors': result['n_neighbors'],
                'metric': result['metric'],
                'basis': result['basis'],
                'n_genes': result['n_genes_valid'],
                'n_genes_total': result['n_genes_total'],
                'gene_subset_type': result['subset_type'],
            }
            self._log_action('gene_neighbors', {
                'n_neighbors': result['n_neighbors'],
                'metric': result['metric'],
                'basis': result['basis'],
            }, status_result)
            return status_result

        return compute_fn, apply_fn

    def run_gene_neighbors(
        self,
        n_neighbors: int = 15,
        metric: str = 'euclidean',
        basis: str = 'gene_pca',
        gene_subset: str | list[str] | dict[str, Any] | None = None,
        scale: bool = True,
        active_cell_indices: list[int] | None = None,
    ) -> dict[str, Any]:
        """Compute gene-gene kNN graph from gene PCA embedding or raw expression.

        Results are stored in .varp['gene_connectivities'] and .varp['gene_distances'].

        Args:
            n_neighbors: Number of neighbors per gene
            metric: Distance metric ('euclidean', 'cosine', 'pearson')
            basis: 'gene_pca' to use PCA embedding, 'expression' to use raw expression
            gene_subset: Gene filtering (only used when basis='expression').
                Can be str (boolean column), list[str] (gene names), or dict (multi-column spec).
            scale: Z-score scale genes before computing neighbors (only used when basis='expression')
            active_cell_indices: Optional cell subset (only used when basis='expression')

        Returns:
            Dict with operation status
        """
        from sklearn.neighbors import NearestNeighbors
        from scipy import sparse

        n_genes_total = self.adata.n_vars
        subset_type = 'all'

        if basis == 'gene_pca':
            # Existing behavior: use gene PCA embedding
            if 'X_gene_pca' not in self.adata.varm:
                raise ValueError("Gene PCA has not been computed. Run gene_pca first.")

            gene_pcs = self.adata.varm['X_gene_pca']
            valid_mask = ~np.isnan(gene_pcs[:, 0])
            valid_indices = np.where(valid_mask)[0]
            representation = gene_pcs[valid_mask, :]

            # Get subset info from gene_pca
            pca_info = self.adata.uns.get('gene_pca', {})
            subset_type = pca_info.get('gene_subset_type', 'all')

        elif basis == 'expression':
            # New path: use raw expression matrix (genes x cells)
            from sklearn.preprocessing import StandardScaler

            # Resolve gene subset
            if gene_subset is not None:
                gene_mask, subset_type, _ = self._resolve_gene_mask(gene_subset)
            else:
                gene_mask = np.ones(self.adata.n_vars, dtype=bool)

            valid_mask = gene_mask
            valid_indices = np.where(valid_mask)[0]

            # Subset cells if provided
            cell_indices = self._validate_cell_indices(active_cell_indices)
            if cell_indices is not None:
                X = self.adata.X[cell_indices][:, gene_mask].T
            else:
                X = self.adata.X[:, gene_mask].T

            # Densify if sparse
            if sparse.issparse(X):
                X = X.toarray()
            X = np.asarray(X, dtype=np.float64)
            X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)

            # Optional z-score scaling
            if scale:
                scaler = StandardScaler()
                X = scaler.fit_transform(X)

            representation = X
        else:
            raise ValueError(f"Unknown basis: {basis}. Must be 'gene_pca' or 'expression'.")

        n_genes_valid = len(valid_indices)
        if n_genes_valid == 0:
            raise ValueError("No genes with valid embeddings/expression")

        # Limit n_neighbors to valid range
        n_neighbors = min(n_neighbors, n_genes_valid - 1)

        # Compute kNN on valid genes
        if metric == 'pearson':
            # Pearson correlation distance: 1 - r
            corr_matrix = np.corrcoef(representation)
            corr_matrix = np.nan_to_num(corr_matrix, nan=0.0)
            dist_full = 1.0 - corr_matrix
            np.fill_diagonal(dist_full, 0.0)
            nn = NearestNeighbors(n_neighbors=n_neighbors + 1, metric='precomputed')
            nn.fit(dist_full)
            distances, indices = nn.kneighbors(dist_full)
        else:
            nn = NearestNeighbors(n_neighbors=n_neighbors + 1, metric=metric)
            nn.fit(representation)
            distances, indices = nn.kneighbors(representation)

        # Build sparse distance matrix (exclude self)
        # Map back to original gene indices
        rows = []
        cols = []
        dists = []
        for i_valid in range(n_genes_valid):
            i_original = valid_indices[i_valid]
            for j_idx in range(1, n_neighbors + 1):  # Skip self (index 0)
                j_valid = indices[i_valid, j_idx]
                j_original = valid_indices[j_valid]
                rows.append(i_original)
                cols.append(j_original)
                dists.append(distances[i_valid, j_idx])

        dist_matrix = sparse.csr_matrix(
            (dists, (rows, cols)),
            shape=(n_genes_total, n_genes_total)
        )

        # Build connectivity matrix (1 / (1 + distance) for weights)
        conn_weights = [1.0 / (1.0 + d) for d in dists]
        conn_matrix = sparse.csr_matrix(
            (conn_weights, (rows, cols)),
            shape=(n_genes_total, n_genes_total)
        )

        # Store in .varp
        self.adata.varp['gene_distances'] = dist_matrix
        self.adata.varp['gene_connectivities'] = conn_matrix

        # Store metadata
        self.adata.uns['gene_neighbors'] = {
            'n_neighbors': n_neighbors,
            'metric': metric,
            'basis': basis,
            'n_genes_in_graph': n_genes_valid,
            'gene_subset_type': subset_type,
        }

        result = {
            'status': 'completed',
            'n_neighbors': n_neighbors,
            'metric': metric,
            'basis': basis,
            'n_genes': n_genes_valid,
            'n_genes_total': n_genes_total,
            'gene_subset_type': subset_type,
        }
        self._log_action('gene_neighbors', {
            'n_neighbors': n_neighbors,
            'metric': metric,
            'basis': basis,
        }, result)
        return result

    def run_find_similar_genes(
        self,
        gene: str,
        n_neighbors: int = 10,
        use: str = 'connectivities',
    ) -> dict[str, Any]:
        """Find genes with similar expression patterns.

        Args:
            gene: Query gene name
            n_neighbors: Number of similar genes to return
            use: 'connectivities' (similarity) or 'distances'

        Returns:
            Dict with list of similar genes and their scores
        """
        # Check prerequisites
        prereq = self.check_prerequisites('find_similar_genes')
        if not prereq['satisfied']:
            raise ValueError(f"Prerequisites not met: {prereq['missing']}")

        if use not in ('connectivities', 'distances'):
            raise ValueError("`use` must be 'connectivities' or 'distances'")

        # Find gene index
        if gene not in self.adata.var_names:
            raise KeyError(f"Gene '{gene}' not found in dataset")

        gene_idx = self.adata.var_names.get_loc(gene)

        # Get the appropriate matrix
        if use == 'connectivities':
            matrix = self.adata.varp['gene_connectivities']
        else:
            matrix = self.adata.varp['gene_distances']

        # Extract row for query gene
        row = matrix.getrow(gene_idx)
        cols = row.indices
        values = row.data

        # Exclude self
        mask = cols != gene_idx
        cols = cols[mask]
        values = values[mask]

        if len(cols) == 0:
            return {
                'query_gene': gene,
                'similar_genes': [],
                'scores': [],
                'use': use,
            }

        # Sort by similarity (descending for connectivity, ascending for distance)
        if use == 'connectivities':
            order = np.argsort(-values)
        else:
            order = np.argsort(values)

        # Get top k
        top_k = min(n_neighbors, len(order))
        top_indices = cols[order[:top_k]]
        top_scores = values[order[:top_k]]

        similar_genes = [self.adata.var_names[i] for i in top_indices]

        result = {
            'query_gene': gene,
            'similar_genes': similar_genes,
            'scores': top_scores.tolist(),
            'use': use,
        }
        return result

    def run_cluster_genes(
        self,
        resolution: float = 0.5,
        key_added: str = 'gene_cluster',
    ) -> dict[str, Any]:
        """Cluster genes into co-expression modules using Leiden.

        Results are stored in .var[key_added] and .uns['gene_modules'].

        Args:
            resolution: Leiden resolution (higher = more clusters)
            key_added: Column name in .var for cluster labels

        Returns:
            Dict with cluster info and module composition
        """
        import igraph as ig
        import leidenalg

        # Check prerequisites
        prereq = self.check_prerequisites('cluster_genes')
        if not prereq['satisfied']:
            raise ValueError(f"Prerequisites not met: {prereq['missing']}")

        # Get connectivity matrix
        conn = self.adata.varp['gene_connectivities']

        # Make symmetric (required for Leiden)
        conn_sym = conn + conn.T
        conn_sym.data = conn_sym.data / 2

        # Identify genes that are in the neighbor graph (have connections)
        # Works for both gene_pca and expression basis
        row_nnz = np.diff(conn_sym.indptr)
        valid_mask = row_nnz > 0
        valid_indices = np.where(valid_mask)[0]
        n_genes_valid = len(valid_indices)

        if n_genes_valid == 0:
            raise ValueError("No genes with valid embeddings to cluster")

        # Create mapping from original indices to subgraph indices
        original_to_sub = {orig: sub for sub, orig in enumerate(valid_indices)}
        sub_to_original = valid_indices

        # Extract subgraph for valid genes only
        conn_sub = conn_sym[valid_indices, :][:, valid_indices]
        sources_sub, targets_sub = conn_sub.nonzero()
        weights = np.array(conn_sub[sources_sub, targets_sub]).flatten()

        # Build igraph from the subgraph
        g = ig.Graph(directed=False)
        g.add_vertices(n_genes_valid)
        edges = list(zip(sources_sub.tolist(), targets_sub.tolist()))
        g.add_edges(edges)
        g.es['weight'] = weights.tolist()

        # Run Leiden
        partition = leidenalg.find_partition(
            g,
            leidenalg.RBConfigurationVertexPartition,
            weights='weight',
            resolution_parameter=resolution,
        )

        # Extract cluster assignments for valid genes
        clusters_sub = np.array(partition.membership)
        n_clusters = len(set(clusters_sub))

        # Map back to full gene set: unclustered genes get label 'unclustered'
        cluster_labels = np.full(self.n_genes, 'unclustered', dtype=object)
        for sub_idx, orig_idx in enumerate(sub_to_original):
            cluster_labels[orig_idx] = str(clusters_sub[sub_idx])

        # Store in .var
        self.adata.var[key_added] = pd.Categorical(cluster_labels)

        # Build module dictionary (only for clustered genes)
        modules = {}
        for cluster_id in range(n_clusters):
            # Find genes in this cluster
            sub_mask = clusters_sub == cluster_id
            original_indices = sub_to_original[sub_mask]
            genes_in_cluster = self.adata.var_names[original_indices].tolist()
            modules[f'module_{cluster_id}'] = genes_in_cluster

        self.adata.uns['gene_modules'] = modules

        result = {
            'status': 'completed',
            'key_added': key_added,
            'n_clusters': n_clusters,
            'n_genes_clustered': n_genes_valid,
            'n_genes_unclustered': self.n_genes - n_genes_valid,
            'resolution': resolution,
            'module_sizes': {k: len(v) for k, v in modules.items()},
        }
        self._log_action('cluster_genes', {
            'resolution': resolution,
            'key_added': key_added,
        }, result)
        return result

    def cluster_gene_set(
        self,
        gene_names: list[str],
        method: str,
        k: int,
        cell_indices: list[int] | None = None,
    ) -> list[list[str]]:
        """Cluster a set of genes by expression pattern across cells.

        Uses the normalized (normalize_total + log1p) expression matrix so
        results match the rest of the gene-analysis code path.

        Args:
            gene_names: Gene symbols to cluster. Unknown names are dropped.
            method: 'hierarchical' (Ward linkage on correlation distance)
                    or 'kmeans' (K-means on raw gene expression vectors).
            k: Number of clusters to produce. Must be at least 2.
            cell_indices: Optional subset of cells. None means all cells.

        Returns:
            A list of length up to k where each element is a list of gene
            names belonging to that cluster. Deterministic ordering by
            cluster id. K-means may return fewer than k groups if it
            collapses an empty cluster; the UI handles this.
        """
        if method not in ('hierarchical', 'kmeans'):
            raise ValueError(f"Unknown method: {method}")
        if k < 2:
            raise ValueError(f"k must be at least 2, got {k}")

        var_names = self.adata.var_names
        found_genes: list[str] = []
        gene_idx: list[int] = []
        for name in gene_names:
            if name in var_names:
                found_genes.append(name)
                gene_idx.append(int(var_names.get_loc(name)))
        if len(found_genes) < k:
            raise ValueError(
                f"cluster_gene_set: need at least {k} known genes, got {len(found_genes)}"
            )

        adata_norm = self.normalized_adata
        if cell_indices is not None:
            if len(cell_indices) == 0:
                raise ValueError("Cell subset is empty")
            cell_arr = np.asarray(cell_indices, dtype=np.int64)
            X = adata_norm.X[cell_arr, :][:, gene_idx]
        else:
            X = adata_norm.X[:, gene_idx]

        import scipy.sparse
        if scipy.sparse.issparse(X):
            X = X.toarray()
        X = np.asarray(X)
        # Shape here is (n_cells_subset, n_genes_found). Transpose so
        # each row is one gene's profile across the selected cells.
        X_genes = X.T

        if method == 'hierarchical':
            from scipy.spatial.distance import pdist
            from scipy.cluster.hierarchy import linkage, fcluster
            dist = pdist(X_genes, metric='correlation')
            # Guard against numerical issues (correlation distance can
            # produce tiny negatives or slight >2 values).
            dist = np.clip(dist, 0, 2)
            if not np.isfinite(dist).all():
                raise ValueError(
                    "Cannot cluster: one or more genes have zero variance "
                    "across the selected cells"
                )
            Z = linkage(dist, method='ward')
            labels = fcluster(Z, t=k, criterion='maxclust')  # 1-indexed
        else:  # kmeans
            from sklearn.cluster import KMeans
            km = KMeans(n_clusters=k, n_init=10, random_state=0)
            labels = km.fit_predict(X_genes)  # 0-indexed

        # Partition found_genes by cluster label.
        partition: dict[int, list[str]] = {}
        for gene, label in zip(found_genes, labels):
            partition.setdefault(int(label), []).append(gene)
        return [partition[key] for key in sorted(partition.keys())]

    def run_build_gene_graph(
        self,
        n_pcs: int | None = None,
        scale: bool = True,
        use_kneedle: bool = True,
        n_neighbors: int = 15,
        metric: str = 'euclidean',
        active_cell_indices: list[int] | None = None,
        gene_subset: str | list[str] | dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Convenience function: run gene_pca and gene_neighbors in one step.

        Args:
            n_pcs: Number of PCs (None for auto-detection)
            scale: Whether to scale genes before PCA
            use_kneedle: Whether to use Kneedle for PC selection
            n_neighbors: Number of neighbors for kNN graph
            metric: Distance metric for kNN
            active_cell_indices: If provided, use only these cells for gene PCA
            gene_subset: Gene filtering specification passed to gene_pca

        Returns:
            Dict with combined results from both steps
        """
        # Run gene PCA
        pca_result = self.run_gene_pca(
            n_comps=n_pcs,
            scale=scale,
            use_kneedle=use_kneedle,
            active_cell_indices=active_cell_indices,
            gene_subset=gene_subset,
        )

        # Run gene neighbors
        neighbors_result = self.run_gene_neighbors(
            n_neighbors=n_neighbors,
            metric=metric,
        )

        result = {
            'status': 'completed',
            'pca': pca_result,
            'neighbors': neighbors_result,
        }
        return result

    # =========================================================================
    # Spatial Analysis Methods
    # =========================================================================

    def prepare_spatial_neighbors(
        self,
        n_neighs: int = 6,
        coord_type: str | None = None,
        spatial_key: str | None = None,
        delaunay: bool = False,
        n_rings: int = 1,
        radius: float | None = None,
    ) -> tuple[Callable[[], dict[str, Any]], Callable[[dict[str, Any]], None]]:
        """Prepare spatial neighborhood graph computation (cancellable).

        Validates inputs and copies adata upfront, then returns a pair of
        functions: ``compute_fn`` (pure, no side-effects) and ``apply_fn``
        (writes results into ``self.adata``).

        Args:
            n_neighs: Number of spatial neighbors (default 6 for hexagonal grids)
            coord_type: 'grid' for Visium, 'generic' for other, None for auto-detect
            spatial_key: Key in .obsm for spatial coordinates (auto-detected if None)
            delaunay: Use Delaunay triangulation instead of kNN
            n_rings: Number of rings of neighbors for grid coordinates
            radius: Radius for generic coordinates (optional)

        Returns:
            Tuple of (compute_fn, apply_fn)
        """
        # Check prerequisites (fail fast)
        prereq = self.check_prerequisites('spatial_neighbors')
        if not prereq['satisfied']:
            raise ValueError(f"Prerequisites not met: {prereq['missing']}")

        # Auto-detect spatial key if not provided
        if spatial_key is None:
            spatial_key = self._get_spatial_key()
            if spatial_key is None:
                raise ValueError("No spatial coordinates found in .obsm")

        # Copy adata for background computation (squidpy needs full adata)
        adata_copy = self.adata.copy()

        # Snapshot parameters
        snap_n_neighs = n_neighs
        snap_coord_type = coord_type
        snap_spatial_key = spatial_key
        snap_delaunay = delaunay
        snap_n_rings = n_rings
        snap_radius = radius

        def compute_fn() -> dict[str, Any]:
            import squidpy as sq

            sq.gr.spatial_neighbors(
                adata_copy,
                n_neighs=snap_n_neighs,
                coord_type=snap_coord_type,
                spatial_key=snap_spatial_key,
                delaunay=snap_delaunay,
                n_rings=snap_n_rings,
                radius=snap_radius,
            )

            return {
                'spatial_connectivities': adata_copy.obsp['spatial_connectivities'],
                'spatial_distances': adata_copy.obsp['spatial_distances'],
                'uns_spatial_neighbors': adata_copy.uns.get('spatial_neighbors', {}),
                'n_edges': adata_copy.obsp['spatial_connectivities'].nnz,
                'spatial_key': snap_spatial_key,
                'n_neighs': snap_n_neighs,
            }

        def apply_fn(result: dict[str, Any]) -> dict[str, Any]:
            self.adata.obsp['spatial_connectivities'] = result['spatial_connectivities']
            self.adata.obsp['spatial_distances'] = result['spatial_distances']

            # Copy any uns keys squidpy set
            if result['uns_spatial_neighbors']:
                self.adata.uns['spatial_neighbors'] = result['uns_spatial_neighbors']

            status_result = {
                'status': 'completed',
                'n_cells': self.n_cells,
                'n_edges': result['n_edges'],
                'spatial_key': result['spatial_key'],
                'n_neighs': result['n_neighs'],
            }
            self._log_action('spatial_neighbors', {
                'n_neighs': result['n_neighs'],
                'coord_type': snap_coord_type,
                'spatial_key': result['spatial_key'],
                'delaunay': snap_delaunay,
            }, status_result)
            return status_result

        return compute_fn, apply_fn

    def run_spatial_neighbors(
        self,
        n_neighs: int = 6,
        coord_type: str | None = None,
        spatial_key: str | None = None,
        delaunay: bool = False,
        n_rings: int = 1,
        radius: float | None = None,
    ) -> dict[str, Any]:
        """Compute spatial neighborhood graph using Squidpy.

        Builds a graph based on spatial proximity of cells/spots.
        Results stored in .obsp['spatial_connectivities'] and .obsp['spatial_distances'].

        Args:
            n_neighs: Number of spatial neighbors (default 6 for hexagonal grids)
            coord_type: 'grid' for Visium, 'generic' for other, None for auto-detect
            spatial_key: Key in .obsm for spatial coordinates (auto-detected if None)
            delaunay: Use Delaunay triangulation instead of kNN
            n_rings: Number of rings of neighbors for grid coordinates
            radius: Radius for generic coordinates (optional)

        Returns:
            Dict with operation status and graph info
        """
        import squidpy as sq

        # Check prerequisites
        prereq = self.check_prerequisites('spatial_neighbors')
        if not prereq['satisfied']:
            raise ValueError(f"Prerequisites not met: {prereq['missing']}")

        # Auto-detect spatial key if not provided
        if spatial_key is None:
            spatial_key = self._get_spatial_key()
            if spatial_key is None:
                raise ValueError("No spatial coordinates found in .obsm")

        # Build spatial neighbors graph
        sq.gr.spatial_neighbors(
            self.adata,
            n_neighs=n_neighs,
            coord_type=coord_type,
            spatial_key=spatial_key,
            delaunay=delaunay,
            n_rings=n_rings,
            radius=radius,
        )

        # Get graph stats
        n_edges = self.adata.obsp['spatial_connectivities'].nnz

        result = {
            'status': 'completed',
            'n_cells': self.n_cells,
            'n_edges': n_edges,
            'spatial_key': spatial_key,
            'n_neighs': n_neighs,
        }
        self._log_action('spatial_neighbors', {
            'n_neighs': n_neighs,
            'coord_type': coord_type,
            'spatial_key': spatial_key,
            'delaunay': delaunay,
        }, result)
        return result

    def prepare_spatial_autocorr(
        self,
        mode: str = 'moran',
        genes: list[str] | None = None,
        n_perms: int | None = 100,
        n_jobs: int = 1,
        corr_method: str = 'fdr_bh',
        pval_threshold: float = 0.05,
        gene_subset: str | list[str] | dict[str, Any] | None = None,
    ) -> tuple[Callable[[], dict[str, Any]], Callable[[dict[str, Any]], None]]:
        """Prepare spatial autocorrelation computation (cancellable).

        Validates inputs and copies adata upfront, then returns a pair of
        functions: ``compute_fn`` (pure, no side-effects) and ``apply_fn``
        (writes results into ``self.adata``).

        Args:
            mode: 'moran' for Moran's I, 'geary' for Geary's C
            genes: Explicit subset of gene names to test (takes priority over gene_subset)
            n_perms: Number of permutations for p-value (None for analytical only)
            n_jobs: Number of parallel jobs
            corr_method: Multiple testing correction method
            pval_threshold: Threshold for marking genes as spatially variable
            gene_subset: Gene filtering via boolean column (e.g. 'highly_variable'). Ignored if genes is set.

        Returns:
            Tuple of (compute_fn, apply_fn)
        """
        # Check prerequisites (fail fast)
        prereq = self.check_prerequisites('spatial_autocorr')
        if not prereq['satisfied']:
            raise ValueError(f"Prerequisites not met: {prereq['missing']}")

        if mode not in ('moran', 'geary'):
            raise ValueError("mode must be 'moran' or 'geary'")

        # Resolve gene_subset to gene list if no explicit genes provided
        subset_type = 'all'
        if genes is None and gene_subset is not None:
            gene_mask, subset_type, _ = self._resolve_gene_mask(gene_subset)
            genes = self.adata.var_names[gene_mask].tolist()

        # Copy adata for background computation
        adata_copy = self.adata.copy()

        # Snapshot parameters
        snap_mode = mode
        snap_genes = genes
        snap_n_perms = n_perms
        snap_n_jobs = n_jobs
        snap_corr_method = corr_method
        snap_pval_threshold = pval_threshold
        snap_subset_type = subset_type
        snap_gene_subset = gene_subset

        def compute_fn() -> dict[str, Any]:
            import squidpy as sq

            sq.gr.spatial_autocorr(
                adata_copy,
                mode=snap_mode,
                genes=snap_genes,
                n_perms=snap_n_perms,
                n_jobs=snap_n_jobs,
                corr_method=snap_corr_method,
            )

            # Extract results from .uns
            uns_key = 'moranI' if snap_mode == 'moran' else 'gearyC'
            stat_col = 'I' if snap_mode == 'moran' else 'C'
            results_df = adata_copy.uns[uns_key]

            # Determine p-value column
            if f'pval_{snap_corr_method}' in results_df.columns:
                pval_col = f'pval_{snap_corr_method}'
            elif 'pval_norm_fdr_bh' in results_df.columns:
                pval_col = 'pval_norm_fdr_bh'
            elif 'pval_norm' in results_df.columns:
                pval_col = 'pval_norm'
            else:
                pval_cols = [c for c in results_df.columns if 'pval' in c]
                pval_col = pval_cols[0] if pval_cols else None

            return {
                'uns_key': uns_key,
                'stat_col': stat_col,
                'results_df': results_df,
                'pval_col': pval_col,
            }

        def apply_fn(result: dict[str, Any]) -> dict[str, Any]:
            uns_key = result['uns_key']
            stat_col = result['stat_col']
            results_df = result['results_df']
            pval_col = result['pval_col']

            # Store full results DataFrame in .uns
            self.adata.uns[uns_key] = results_df

            # Initialize .var columns with defaults for genes not tested
            self.adata.var[uns_key] = np.nan
            self.adata.var['spatial_pval_adj'] = np.nan
            self.adata.var['spatially_variable'] = False

            # Fill in values for tested genes
            for gene in results_df.index:
                if gene in self.adata.var_names:
                    self.adata.var.loc[gene, uns_key] = results_df.loc[gene, stat_col]
                    if pval_col:
                        self.adata.var.loc[gene, 'spatial_pval_adj'] = results_df.loc[gene, pval_col]

            # Mark spatially variable genes
            if pval_col:
                sv_mask = self.adata.var['spatial_pval_adj'] < snap_pval_threshold
                if snap_mode == 'moran':
                    sv_mask = sv_mask & (self.adata.var[uns_key] > 0)
                else:
                    sv_mask = sv_mask & (self.adata.var[uns_key] < 1)
                self.adata.var['spatially_variable'] = sv_mask

            n_sv_genes = self.adata.var['spatially_variable'].sum()
            n_tested = len(results_df)

            # Get top spatially variable genes
            sv_genes = self.adata.var[self.adata.var['spatially_variable']].copy()
            if snap_mode == 'moran':
                sv_genes = sv_genes.sort_values(uns_key, ascending=False)
            else:
                sv_genes = sv_genes.sort_values(uns_key, ascending=True)
            top_sv_genes = sv_genes.head(20).index.tolist()

            status_result = {
                'status': 'completed',
                'mode': snap_mode,
                'n_tested': n_tested,
                'n_spatially_variable': int(n_sv_genes),
                'pval_threshold': snap_pval_threshold,
                'top_sv_genes': top_sv_genes,
                'gene_subset_type': snap_subset_type,
            }
            self._log_action('spatial_autocorr', {
                'mode': snap_mode,
                'n_perms': snap_n_perms,
                'corr_method': snap_corr_method,
                'pval_threshold': snap_pval_threshold,
                'gene_subset': snap_gene_subset,
            }, status_result)
            return status_result

        return compute_fn, apply_fn

    def run_spatial_autocorr(
        self,
        mode: str = 'moran',
        genes: list[str] | None = None,
        n_perms: int | None = 100,
        n_jobs: int = 1,
        corr_method: str = 'fdr_bh',
        pval_threshold: float = 0.05,
        gene_subset: str | list[str] | dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Compute spatial autocorrelation to identify spatially variable genes.

        Uses Squidpy to compute Moran's I or Geary's C statistics.
        Results stored in:
        - .uns['moranI'] or .uns['gearyC']: Full DataFrame with stats
        - .var['spatially_variable']: Boolean, True if gene passes threshold
        - .var['moranI'] or .var['gearyC']: The autocorrelation statistic
        - .var['spatial_pval_adj']: Adjusted p-value

        Args:
            mode: 'moran' for Moran's I, 'geary' for Geary's C
            genes: Explicit subset of gene names to test (takes priority over gene_subset)
            n_perms: Number of permutations for p-value (None for analytical only)
            n_jobs: Number of parallel jobs
            corr_method: Multiple testing correction method
            pval_threshold: Threshold for marking genes as spatially variable
            gene_subset: Gene filtering via boolean column (e.g. 'highly_variable'). Ignored if genes is set.

        Returns:
            Dict with operation status, number of spatially variable genes
        """
        import squidpy as sq

        # Check prerequisites
        prereq = self.check_prerequisites('spatial_autocorr')
        if not prereq['satisfied']:
            raise ValueError(f"Prerequisites not met: {prereq['missing']}")

        if mode not in ('moran', 'geary'):
            raise ValueError("mode must be 'moran' or 'geary'")

        # Resolve gene_subset to gene list if no explicit genes provided
        subset_type = 'all'
        if genes is None and gene_subset is not None:
            gene_mask, subset_type, _ = self._resolve_gene_mask(gene_subset)
            genes = self.adata.var_names[gene_mask].tolist()

        # Run spatial autocorrelation
        sq.gr.spatial_autocorr(
            self.adata,
            mode=mode,
            genes=genes,
            n_perms=n_perms,
            n_jobs=n_jobs,
            corr_method=corr_method,
        )

        # Get results from .uns
        uns_key = 'moranI' if mode == 'moran' else 'gearyC'
        stat_col = 'I' if mode == 'moran' else 'C'
        results_df = self.adata.uns[uns_key]

        # Determine p-value column (depends on correction method and n_perms)
        if f'pval_{corr_method}' in results_df.columns:
            pval_col = f'pval_{corr_method}'
        elif 'pval_norm_fdr_bh' in results_df.columns:
            pval_col = 'pval_norm_fdr_bh'
        elif 'pval_norm' in results_df.columns:
            pval_col = 'pval_norm'
        else:
            # Fallback - use first pval column
            pval_cols = [c for c in results_df.columns if 'pval' in c]
            pval_col = pval_cols[0] if pval_cols else None

        # Store in .var for easy filtering
        # Initialize columns with NaN for genes not tested
        self.adata.var[uns_key] = np.nan
        self.adata.var['spatial_pval_adj'] = np.nan
        self.adata.var['spatially_variable'] = False

        # Fill in values for tested genes
        for gene in results_df.index:
            if gene in self.adata.var_names:
                self.adata.var.loc[gene, uns_key] = results_df.loc[gene, stat_col]
                if pval_col:
                    self.adata.var.loc[gene, 'spatial_pval_adj'] = results_df.loc[gene, pval_col]

        # Mark spatially variable genes
        if pval_col:
            sv_mask = self.adata.var['spatial_pval_adj'] < pval_threshold
            # For Moran's I, positive values indicate clustering
            if mode == 'moran':
                sv_mask = sv_mask & (self.adata.var[uns_key] > 0)
            # For Geary's C, values < 1 indicate positive autocorrelation
            else:
                sv_mask = sv_mask & (self.adata.var[uns_key] < 1)
            self.adata.var['spatially_variable'] = sv_mask

        n_sv_genes = self.adata.var['spatially_variable'].sum()
        n_tested = len(results_df)

        # Get top spatially variable genes
        sv_genes = self.adata.var[self.adata.var['spatially_variable']].copy()
        if mode == 'moran':
            sv_genes = sv_genes.sort_values(uns_key, ascending=False)
        else:
            sv_genes = sv_genes.sort_values(uns_key, ascending=True)
        top_sv_genes = sv_genes.head(20).index.tolist()

        result = {
            'status': 'completed',
            'mode': mode,
            'n_tested': n_tested,
            'n_spatially_variable': int(n_sv_genes),
            'pval_threshold': pval_threshold,
            'top_sv_genes': top_sv_genes,
            'gene_subset_type': subset_type,
        }
        self._log_action('spatial_autocorr', {
            'mode': mode,
            'n_perms': n_perms,
            'corr_method': corr_method,
            'pval_threshold': pval_threshold,
            'gene_subset': gene_subset,
        }, result)
        return result

    def prepare_contourize(
        self,
        genes: list[str],
        contour_levels: int = 6,
        log_transform: bool = True,
        smooth_sigma: float = 2.0,
        grid_res: int = 200,
        clip_percentiles: tuple = (1, 99),
        annotation_key: str | None = None,
    ) -> tuple[Callable[[], dict[str, Any]], Callable[[dict[str, Any]], None]]:
        """Prepare spatial expression contouring (cancellable).

        Validates inputs and snapshots data upfront, then returns a pair of
        functions: ``compute_fn`` (pure, no side-effects) and ``apply_fn``
        (writes results into ``self.adata``).

        Args:
            genes: List of gene names defining the module
            contour_levels: Number of contour thresholds
            log_transform: Apply log1p before contouring
            smooth_sigma: Gaussian smoothing strength
            grid_res: Interpolation grid size per axis
            clip_percentiles: Percentile clipping range
            annotation_key: Name for the result .obs column (auto-generated if None)

        Returns:
            Tuple of (compute_fn, apply_fn)
        """
        # Validate spatial key (fail fast)
        spatial_key = self._get_spatial_key()
        if spatial_key is None:
            raise ValueError("No spatial coordinates found")

        # Validate genes (fail fast)
        missing = [g for g in genes if g not in self.adata.var_names]
        if missing:
            raise ValueError(f"Genes not found: {missing}")

        # Auto-generate annotation_key if None
        if annotation_key is None:
            annotation_key = f"contour_{genes[0]}_{len(genes)}"

        # Snapshot spatial coordinates and gene expression data
        coords_snap = self.adata.obsm[spatial_key].copy()
        gene_expression_snap = {}
        for g in genes:
            xmat = self.adata[:, g].X
            if hasattr(xmat, 'toarray'):
                gene_expression_snap[g] = xmat.toarray().flatten().copy()
            else:
                gene_expression_snap[g] = np.asarray(xmat).flatten().copy()

        # Snapshot parameters
        snap_genes = list(genes)
        snap_contour_levels = contour_levels
        snap_log_transform = log_transform
        snap_smooth_sigma = smooth_sigma
        snap_grid_res = grid_res
        snap_clip_percentiles = clip_percentiles
        snap_annotation_key = annotation_key
        snap_n_cells = self.adata.n_obs

        def compute_fn() -> dict[str, Any]:
            from scipy.interpolate import griddata
            from scipy.ndimage import gaussian_filter

            x, y = coords_snap[:, 0], coords_snap[:, 1]

            # 1) Preprocess and normalize each gene
            normed = []
            for g in snap_genes:
                vals = gene_expression_snap[g].copy()
                if snap_log_transform:
                    vals = np.log1p(vals)
                lo, hi = np.percentile(vals, snap_clip_percentiles)
                clipped = np.clip(vals, lo, hi)
                normalized = (clipped - lo) / (hi - lo) if hi > lo else np.zeros_like(clipped)
                normed.append(normalized)

            # 2) Average across genes per cell
            M = np.column_stack(normed)
            summary = np.mean(M, axis=1)

            # 3) Interpolate onto grid
            xi = np.linspace(x.min(), x.max(), snap_grid_res)
            yi = np.linspace(y.min(), y.max(), snap_grid_res)
            Xi, Yi = np.meshgrid(xi, yi)
            Zi = griddata((x, y), summary, (Xi, Yi), method='cubic', fill_value=0.0)

            # 4) Gaussian smooth
            Zi_s = gaussian_filter(Zi, sigma=snap_smooth_sigma, mode='nearest')
            vmax = np.nanmax(Zi_s)

            # 5) Compute thresholds
            N = snap_contour_levels
            thresholds = np.linspace(0, vmax, N + 2)[1:-1]

            # 6) Sample smoothed grid at cell positions (nearest)
            pts = np.vstack((Xi.ravel(), Yi.ravel())).T
            val_grid = Zi_s.ravel()
            cell_vals = griddata(pts, val_grid, (x, y), method='nearest')

            # 7) Assign each cell the highest threshold it meets
            annotation = np.zeros(snap_n_cells, dtype=float)
            for t in sorted(thresholds):
                mask = cell_vals >= t
                annotation[mask] = t

            # 8) Build ordered categorical data
            cats = np.unique(np.concatenate(([0.0], thresholds)))

            return {
                'annotation': annotation,
                'categories': cats,
                'annotation_key': snap_annotation_key,
                'n_genes': len(snap_genes),
                'genes': snap_genes,
                'contour_levels': snap_contour_levels,
                'n_cells': snap_n_cells,
            }

        def apply_fn(result: dict[str, Any]) -> dict[str, Any]:
            annotation_cat = pd.Categorical(
                result['annotation'],
                categories=result['categories'],
                ordered=True,
            )
            self.adata.obs[result['annotation_key']] = annotation_cat

            status_result = {
                'status': 'completed',
                'annotation_key': result['annotation_key'],
                'n_genes': result['n_genes'],
                'genes': result['genes'],
                'contour_levels': result['contour_levels'],
                'n_cells': result['n_cells'],
            }
            self._log_action('contourize', {
                'genes': result['genes'],
                'contour_levels': result['contour_levels'],
                'log_transform': snap_log_transform,
                'smooth_sigma': snap_smooth_sigma,
                'grid_res': snap_grid_res,
                'annotation_key': result['annotation_key'],
            }, status_result)
            return status_result

        return compute_fn, apply_fn

    def run_contourize(
        self,
        genes: list[str],
        contour_levels: int = 6,
        log_transform: bool = True,
        smooth_sigma: float = 2.0,
        grid_res: int = 200,
        clip_percentiles: tuple = (1, 99),
        annotation_key: str | None = None,
    ) -> dict[str, Any]:
        """Compute spatial expression contours from a gene set and assign each cell a contour level.

        For each gene: extract expression, optionally log1p, percentile-clip,
        min-max normalize to [0,1]. Average across genes per cell. Interpolate
        onto a grid, Gaussian smooth, compute N thresholds, and assign each cell
        the highest threshold it meets. Result stored as an ordered categorical
        column in .obs.

        Args:
            genes: List of gene names defining the module
            contour_levels: Number of contour thresholds
            log_transform: Apply log1p before contouring
            smooth_sigma: Gaussian smoothing strength
            grid_res: Interpolation grid size per axis
            clip_percentiles: Percentile clipping range
            annotation_key: Name for the result .obs column (auto-generated if None)

        Returns:
            Dict with status, annotation_key, n_genes, genes, contour_levels, n_cells
        """
        from scipy.interpolate import griddata
        from scipy.ndimage import gaussian_filter

        # Validate genes
        missing = [g for g in genes if g not in self.adata.var_names]
        if missing:
            raise ValueError(f"Genes not found: {missing}")

        # Get spatial coordinates
        spatial_key = self._get_spatial_key()
        if spatial_key is None:
            raise ValueError("No spatial coordinates found")
        coords = self.adata.obsm[spatial_key]
        x, y = coords[:, 0], coords[:, 1]
        n_cells = x.shape[0]

        # Helper for sparse arrays
        def _get_array(xmat):
            return xmat.toarray().flatten() if hasattr(xmat, 'toarray') else xmat.flatten()

        # 1) Preprocess and normalize each gene
        normed = []
        for g in genes:
            vals = _get_array(self.adata[:, g].X)
            if log_transform:
                vals = np.log1p(vals)
            lo, hi = np.percentile(vals, clip_percentiles)
            clipped = np.clip(vals, lo, hi)
            normalized = (clipped - lo) / (hi - lo) if hi > lo else np.zeros_like(clipped)
            normed.append(normalized)

        # 2) Average across genes per cell
        M = np.column_stack(normed)
        summary = np.mean(M, axis=1)

        # 3) Interpolate onto grid
        xi = np.linspace(x.min(), x.max(), grid_res)
        yi = np.linspace(y.min(), y.max(), grid_res)
        Xi, Yi = np.meshgrid(xi, yi)
        Zi = griddata((x, y), summary, (Xi, Yi), method='cubic', fill_value=0.0)

        # 4) Gaussian smooth
        Zi_s = gaussian_filter(Zi, sigma=smooth_sigma, mode='nearest')
        vmax = np.nanmax(Zi_s)

        # 5) Compute thresholds
        N = contour_levels
        thresholds = np.linspace(0, vmax, N + 2)[1:-1]

        # 6) Sample smoothed grid at cell positions (nearest)
        pts = np.vstack((Xi.ravel(), Yi.ravel())).T
        val_grid = Zi_s.ravel()
        cell_vals = griddata(pts, val_grid, (x, y), method='nearest')

        # 7) Assign each cell the highest threshold it meets
        annotation = np.zeros(n_cells, dtype=float)
        for t in sorted(thresholds):
            mask = cell_vals >= t
            annotation[mask] = t

        # 8) Store as ordered categorical
        cats = np.unique(np.concatenate(([0.0], thresholds)))
        annotation_cat = pd.Categorical(annotation, categories=cats, ordered=True)

        if annotation_key is None:
            annotation_key = f"contour_{genes[0]}_{len(genes)}"
        self.adata.obs[annotation_key] = annotation_cat

        result = {
            'status': 'completed',
            'annotation_key': annotation_key,
            'n_genes': len(genes),
            'genes': genes,
            'contour_levels': contour_levels,
            'n_cells': n_cells,
        }
        self._log_action('contourize', {
            'genes': genes,
            'contour_levels': contour_levels,
            'log_transform': log_transform,
            'smooth_sigma': smooth_sigma,
            'grid_res': grid_res,
            'annotation_key': annotation_key,
        }, result)
        return result

    def get_spatially_variable_genes(
        self,
        top_n: int | None = None,
        pval_threshold: float | None = None,
    ) -> dict[str, Any]:
        """Get list of spatially variable genes.

        Args:
            top_n: Return only top N genes (sorted by statistic)
            pval_threshold: Override threshold for filtering

        Returns:
            Dict with gene list and statistics
        """
        if 'spatially_variable' not in self.adata.var.columns:
            raise ValueError("spatial_autocorr has not been run")

        # Determine which statistic was used
        if 'moranI' in self.adata.var.columns:
            stat_col = 'moranI'
            ascending = False
        elif 'gearyC' in self.adata.var.columns:
            stat_col = 'gearyC'
            ascending = True
        else:
            raise ValueError("No spatial autocorrelation statistics found")

        # Filter genes
        if pval_threshold is not None:
            mask = (self.adata.var['spatial_pval_adj'] < pval_threshold)
            if stat_col == 'moranI':
                mask = mask & (self.adata.var[stat_col] > 0)
            else:
                mask = mask & (self.adata.var[stat_col] < 1)
        else:
            mask = self.adata.var['spatially_variable']

        sv_genes = self.adata.var[mask].copy()
        sv_genes = sv_genes.sort_values(stat_col, ascending=ascending)

        if top_n is not None:
            sv_genes = sv_genes.head(top_n)

        genes_list = []
        for gene in sv_genes.index:
            genes_list.append({
                'gene': gene,
                'statistic': float(sv_genes.loc[gene, stat_col]),
                'pval_adj': float(sv_genes.loc[gene, 'spatial_pval_adj']),
            })

        return {
            'genes': genes_list,
            'n_total': int(mask.sum()),
            'statistic_type': stat_col,
        }

    def get_gene_modules(self) -> dict[str, Any]:
        """Get gene cluster modules from the last cluster_genes run.

        Returns:
            Dict with modules (dict of module_name -> gene list)
        """
        if 'gene_modules' not in self.adata.uns:
            raise ValueError("cluster_genes has not been run")

        return {
            'modules': self.adata.uns['gene_modules'],
            'n_modules': len(self.adata.uns['gene_modules']),
        }

    def run_marker_genes(
        self,
        obs_column: str,
        groups: list[str] | None = None,
        top_n: int = 25,
        min_in_group_fraction: float | None = None,
        max_out_group_fraction: float | None = None,
        min_fold_change: float | None = None,
        gene_subset: str | list[str] | dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Run one-vs-rest marker gene analysis using scanpy.

        Identifies marker genes for each group in a categorical obs column
        using Wilcoxon rank-sum test (one-vs-rest).

        Args:
            obs_column: Name of a categorical column in .obs
            groups: Optional list of group names to include. If None, uses all groups.
            top_n: Number of top marker genes per group
            min_in_group_fraction: Min fraction of cells in group expressing the gene
            max_out_group_fraction: Max fraction of cells outside group expressing the gene
            min_fold_change: Minimum fold change threshold
            gene_subset: Gene filtering specification (str column name, list of genes, or dict spec)

        Returns:
            Dictionary with obs_column, results (per-group gene lists), and params

        Raises:
            ValueError: If column doesn't exist, isn't categorical, or has < 2 groups
        """
        # Validate obs_column exists
        if obs_column not in self.adata.obs.columns:
            raise ValueError(f"Column '{obs_column}' not found in .obs")

        # Validate it's categorical
        dtype = self.adata.obs[obs_column].dtype
        if not pd.api.types.is_categorical_dtype(dtype) and not pd.api.types.is_string_dtype(dtype):
            raise ValueError(f"Column '{obs_column}' is not categorical (dtype: {dtype})")

        # Resolve gene subset
        if gene_subset is not None:
            gene_mask, subset_type, _ = self._resolve_gene_mask(gene_subset)
            work_adata = self.adata[:, gene_mask].copy()
        else:
            work_adata = self.adata.copy()
            subset_type = 'all'

        # Ensure the column is categorical
        if not pd.api.types.is_categorical_dtype(work_adata.obs[obs_column].dtype):
            work_adata.obs[obs_column] = pd.Categorical(work_adata.obs[obs_column])

        # If groups specified, subset to only those cells
        if groups is not None:
            all_categories = list(work_adata.obs[obs_column].cat.categories)
            invalid = [g for g in groups if g not in all_categories]
            if invalid:
                raise ValueError(f"Groups not found in column '{obs_column}': {invalid}")
            mask = work_adata.obs[obs_column].isin(groups)
            work_adata = work_adata[mask].copy()
            # Remove unused categories after subsetting
            work_adata.obs[obs_column] = work_adata.obs[obs_column].cat.remove_unused_categories()

        # Validate we have at least 2 groups
        n_groups = len(work_adata.obs[obs_column].cat.categories)
        if n_groups < 2:
            raise ValueError(f"Need at least 2 groups for marker gene analysis, got {n_groups}")

        # Run rank_genes_groups (one-vs-rest)
        sc.tl.rank_genes_groups(
            work_adata,
            groupby=obs_column,
            method='wilcoxon',
            key_added='marker_genes',
        )

        # Apply filters if specified
        has_filters = any(x is not None for x in [min_in_group_fraction, max_out_group_fraction, min_fold_change])
        if has_filters:
            filter_kwargs: dict[str, Any] = {'key': 'marker_genes', 'key_added': 'marker_genes_filtered'}
            if min_in_group_fraction is not None:
                filter_kwargs['min_in_group_fraction'] = min_in_group_fraction
            if max_out_group_fraction is not None:
                filter_kwargs['max_out_group_fraction'] = max_out_group_fraction
            if min_fold_change is not None:
                filter_kwargs['min_fold_change'] = min_fold_change
            sc.tl.filter_rank_genes_groups(work_adata, **filter_kwargs)

        # Extract results per group
        result_groups = []
        for group in work_adata.obs[obs_column].cat.categories:
            group_str = str(group)
            try:
                if has_filters:
                    df = sc.get.rank_genes_groups_df(work_adata, group=group_str, key='marker_genes_filtered')
                    # filter_rank_genes_groups sets filtered genes to NaN
                    df = df.dropna(subset=['names'])
                else:
                    df = sc.get.rank_genes_groups_df(work_adata, group=group_str, key='marker_genes')

                # Take top N
                df = df.head(top_n)

                genes = []
                for _, row in df.iterrows():
                    genes.append({
                        'gene': str(row['names']),
                        'log2fc': float(row['logfoldchanges']),
                        'pval': float(row['pvals']),
                        'pval_adj': float(row['pvals_adj']),
                    })

                result_groups.append({
                    'group': group_str,
                    'genes': genes,
                })
            except Exception:
                # If a group fails (e.g., too few cells), include empty result
                result_groups.append({
                    'group': group_str,
                    'genes': [],
                })

        self._log_action('marker_genes', {
            'obs_column': obs_column,
            'groups': groups,
            'top_n': top_n,
            'min_in_group_fraction': min_in_group_fraction,
            'max_out_group_fraction': max_out_group_fraction,
            'min_fold_change': min_fold_change,
            'gene_subset': gene_subset,
        }, {
            'n_groups': len(result_groups),
            'total_genes': sum(len(g['genes']) for g in result_groups),
        })

        return {
            'obs_column': obs_column,
            'results': result_groups,
            'gene_subset_type': subset_type,
            'n_genes_tested': work_adata.n_vars,
        }
