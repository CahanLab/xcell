"""DataAdaptor class for wrapping AnnData objects.

This module provides a clean interface for accessing single-cell data,
following the adaptor pattern used in excellxgene. It abstracts away
direct AnnData access and is designed for easy integration with scanpy
analysis functions.
"""

from pathlib import Path
from typing import Any

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
        """Load an h5ad file and initialize the adaptor.

        Args:
            filepath: Path to the .h5ad file to load
        """
        self.filepath = Path(filepath)
        self.adata = anndata.read_h5ad(self.filepath)
        self._normalized_adata: anndata.AnnData | None = None
        self._drawn_lines: list[dict[str, Any]] = []  # Stored lines from frontend
        self._action_history: list[dict[str, Any]] = []  # Track scanpy operations

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
        # Get embedding names (keys in obsm that are 2D arrays)
        embeddings = []
        for key in self.adata.obsm.keys():
            arr = self.adata.obsm[key]
            if isinstance(arr, np.ndarray) and arr.ndim == 2 and arr.shape[1] >= 2:
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
            "embeddings": embeddings,
            "obs_columns": obs_columns,
            "obs_dtypes": obs_dtypes,
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

    def search_genes(self, query: str, limit: int = 20) -> list[str]:
        """Search for genes by name prefix.

        Args:
            query: Search query (case-insensitive prefix match)
            limit: Maximum number of results to return

        Returns:
            List of matching gene names
        """
        query_lower = query.lower()
        gene_names = self.adata.var.index.tolist()

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

        Raises:
            KeyError: If any gene not found in .var
        """
        # Validate all genes exist
        missing = [g for g in genes if g not in self.adata.var.index]
        if missing:
            raise KeyError(f"Genes not found: {missing}")

        if len(genes) == 0:
            return {
                "genes": [],
                "values": [0.0] * self.n_cells,
                "min": 0.0,
                "max": 0.0,
            }

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

        Raises:
            KeyError: If any gene not found in .var
        """
        # Validate all genes exist
        all_genes = genes1 + genes2
        missing = [g for g in all_genes if g not in self.adata.var.index]
        if missing:
            raise KeyError(f"Genes not found: {missing}")

        if len(genes1) == 0 or len(genes2) == 0:
            raise ValueError("Both gene sets must contain at least one gene")

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
    ) -> dict[str, Any]:
        """Run differential expression analysis between two cell groups.

        Uses scanpy's rank_genes_groups with Wilcoxon rank-sum test.

        Args:
            group1_indices: Cell indices for group 1
            group2_indices: Cell indices for group 2
            top_n: Number of top genes to return for each direction

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

        return compute_diffexp(
            adata=self.adata,
            group1_indices=group1_indices,
            group2_indices=group2_indices,
            top_n=top_n,
        )

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
            'filter_cells': [],
            'normalize_total': [],
            'log1p': [],
            'pca': [],
            'neighbors': ['pca'],
            'umap': ['neighbors'],
            'leiden': ['neighbors'],
            # Gene analysis
            'gene_pca': [],
            'gene_neighbors': ['gene_pca'],
            'find_similar_genes': ['gene_neighbors'],
            'cluster_genes': ['gene_neighbors'],
            'build_gene_graph': [],  # Convenience function, no prereqs
            # Spatial analysis
            'spatial_neighbors': ['has_spatial'],
            'spatial_autocorr': ['spatial_neighbors'],
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
    ) -> dict[str, Any]:
        """Filter genes based on counts or number of cells expressing.

        Args:
            min_counts: Minimum total counts for a gene
            max_counts: Maximum total counts for a gene
            min_cells: Minimum number of cells expressing the gene
            max_cells: Maximum number of cells expressing the gene

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
            sc.pp.filter_genes(self.adata, **kwargs)

        n_genes_after = self.n_genes

        # Invalidate normalized cache since data changed
        self._normalized_adata = None

        result = {
            'n_genes_before': n_genes_before,
            'n_genes_after': n_genes_after,
            'n_genes_removed': n_genes_before - n_genes_after,
        }
        self._log_action('filter_genes', kwargs, result)
        return result

    def run_filter_cells(
        self,
        min_counts: int | None = None,
        max_counts: int | None = None,
        min_genes: int | None = None,
        max_genes: int | None = None,
    ) -> dict[str, Any]:
        """Filter cells based on counts or number of genes expressed.

        Args:
            min_counts: Minimum total counts for a cell
            max_counts: Maximum total counts for a cell
            min_genes: Minimum number of genes expressed in the cell
            max_genes: Maximum number of genes expressed in the cell

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

    def run_normalize_total(
        self,
        target_sum: float | None = None,
    ) -> dict[str, Any]:
        """Normalize total counts per cell.

        Args:
            target_sum: Target sum of counts per cell. If None, uses median.

        Returns:
            Dict with operation status
        """
        kwargs = {}
        if target_sum is not None:
            kwargs['target_sum'] = target_sum

        sc.pp.normalize_total(self.adata, **kwargs)

        # Invalidate normalized cache
        self._normalized_adata = None

        result = {'status': 'completed', 'target_sum': target_sum}
        self._log_action('normalize_total', kwargs, result)
        return result

    def run_log1p(self) -> dict[str, Any]:
        """Apply log1p transformation to the data.

        Returns:
            Dict with operation status
        """
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

        Returns:
            Dict with operation status and number of HVGs
        """
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

        # Limit n_comps to valid range
        max_comps = min(adata_pca.n_obs - 1, adata_pca.n_vars - 1)
        n_comps = min(n_comps, max_comps)

        # Run PCA on subset
        sc.tl.pca(adata_pca, n_comps=n_comps, svd_solver=svd_solver)

        # Copy results back to main adata
        self.adata.obsm['X_pca'] = adata_pca.obsm['X_pca']
        self.adata.uns['pca'] = adata_pca.uns['pca']
        if 'PCs' in adata_pca.varm:
            # Store loadings for the subset genes
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
        self._log_action('pca', {
            'n_comps': n_comps,
            'svd_solver': svd_solver,
            'gene_subset': gene_subset,
        }, result)
        return result

    def run_neighbors(
        self,
        n_neighbors: int = 15,
        n_pcs: int | None = None,
        metric: str = 'euclidean',
    ) -> dict[str, Any]:
        """Compute neighborhood graph.

        Args:
            n_neighbors: Number of neighbors to use
            n_pcs: Number of PCs to use (None = use all)
            metric: Distance metric

        Returns:
            Dict with operation status
        """
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

        sc.pp.neighbors(self.adata, **kwargs)

        result = {'status': 'completed', 'n_neighbors': n_neighbors}
        self._log_action('neighbors', kwargs, result)
        return result

    def run_umap(
        self,
        min_dist: float = 0.5,
        spread: float = 1.0,
        n_components: int = 2,
    ) -> dict[str, Any]:
        """Compute UMAP embedding.

        Args:
            min_dist: Minimum distance between points
            spread: Spread of the embedding
            n_components: Number of dimensions

        Returns:
            Dict with operation status and embedding name
        """
        # Check prerequisites
        prereq = self.check_prerequisites('umap')
        if not prereq['satisfied']:
            raise ValueError(f"Prerequisites not met: {prereq['missing']}")

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
    ) -> dict[str, Any]:
        """Run Leiden clustering.

        Args:
            resolution: Resolution parameter (higher = more clusters)
            key_added: Key to add to obs for cluster labels

        Returns:
            Dict with operation status and cluster info
        """
        # Check prerequisites
        prereq = self.check_prerequisites('leiden')
        if not prereq['satisfied']:
            raise ValueError(f"Prerequisites not met: {prereq['missing']}")

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

        # Subset and transpose: genes become observations
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

    def run_gene_neighbors(
        self,
        n_neighbors: int = 15,
        metric: str = 'cosine',
    ) -> dict[str, Any]:
        """Compute gene-gene kNN graph from gene PCA embedding.

        Results are stored in .varp['gene_connectivities'] and .varp['gene_distances'].
        If gene_pca was computed on a subset, neighbors are only computed for that subset.

        Args:
            n_neighbors: Number of neighbors per gene
            metric: Distance metric ('cosine', 'euclidean', 'correlation')

        Returns:
            Dict with operation status
        """
        from sklearn.neighbors import NearestNeighbors
        from scipy import sparse

        # Check prerequisites
        prereq = self.check_prerequisites('gene_neighbors')
        if not prereq['satisfied']:
            raise ValueError(f"Prerequisites not met: {prereq['missing']}")

        # Get gene PCA embedding
        gene_pcs = self.adata.varm['X_gene_pca']
        n_genes_total = gene_pcs.shape[0]

        # Identify genes with valid embeddings (not NaN)
        valid_mask = ~np.isnan(gene_pcs[:, 0])
        valid_indices = np.where(valid_mask)[0]
        n_genes_valid = len(valid_indices)

        if n_genes_valid == 0:
            raise ValueError("No genes with valid PCA embeddings")

        # Extract valid gene embeddings
        gene_pcs_valid = gene_pcs[valid_mask, :]

        # Limit n_neighbors to valid range
        n_neighbors = min(n_neighbors, n_genes_valid - 1)

        # Compute kNN on valid genes
        nn = NearestNeighbors(n_neighbors=n_neighbors + 1, metric=metric)
        nn.fit(gene_pcs_valid)
        distances, indices = nn.kneighbors(gene_pcs_valid)

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

        # Get subset info from gene_pca
        subset_info = self.adata.uns.get('gene_pca', {})
        subset_type = subset_info.get('gene_subset_type', 'all')

        # Store metadata
        self.adata.uns['gene_neighbors'] = {
            'n_neighbors': n_neighbors,
            'metric': metric,
            'n_genes_in_graph': n_genes_valid,
            'gene_subset_type': subset_type,
        }

        result = {
            'status': 'completed',
            'n_neighbors': n_neighbors,
            'metric': metric,
            'n_genes': n_genes_valid,
            'n_genes_total': n_genes_total,
            'gene_subset_type': subset_type,
        }
        self._log_action('gene_neighbors', {
            'n_neighbors': n_neighbors,
            'metric': metric,
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

        # Identify genes with valid embeddings (genes that have connections)
        # Genes outside the PCA subset will have no connections
        gene_pcs = self.adata.varm['X_gene_pca']
        valid_mask = ~np.isnan(gene_pcs[:, 0])
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

    def run_build_gene_graph(
        self,
        n_pcs: int | None = None,
        scale: bool = True,
        use_kneedle: bool = True,
        n_neighbors: int = 15,
        metric: str = 'cosine',
    ) -> dict[str, Any]:
        """Convenience function: run gene_pca and gene_neighbors in one step.

        Args:
            n_pcs: Number of PCs (None for auto-detection)
            scale: Whether to scale genes before PCA
            use_kneedle: Whether to use Kneedle for PC selection
            n_neighbors: Number of neighbors for kNN graph
            metric: Distance metric for kNN

        Returns:
            Dict with combined results from both steps
        """
        # Run gene PCA
        pca_result = self.run_gene_pca(
            n_comps=n_pcs,
            scale=scale,
            use_kneedle=use_kneedle,
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

    def run_spatial_autocorr(
        self,
        mode: str = 'moran',
        genes: list[str] | None = None,
        n_perms: int | None = 100,
        n_jobs: int = 1,
        corr_method: str = 'fdr_bh',
        pval_threshold: float = 0.05,
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
            genes: Subset of genes to test (None for all highly variable or all)
            n_perms: Number of permutations for p-value (None for analytical only)
            n_jobs: Number of parallel jobs
            corr_method: Multiple testing correction method
            pval_threshold: Threshold for marking genes as spatially variable

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
        }
        self._log_action('spatial_autocorr', {
            'mode': mode,
            'n_perms': n_perms,
            'corr_method': corr_method,
            'pval_threshold': pval_threshold,
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
