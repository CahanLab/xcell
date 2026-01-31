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

    @property
    def n_cells(self) -> int:
        """Number of cells (observations) in the dataset."""
        return self.adata.n_obs

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

    def get_expression(self, gene: str) -> dict[str, Any]:
        """Get expression values for a single gene across all cells.

        Args:
            gene: Gene name

        Returns:
            Dictionary containing:
            - gene: The gene name
            - values: List of expression values for each cell
            - min: Minimum expression value
            - max: Maximum expression value

        Raises:
            KeyError: If gene not found in .var
        """
        if gene not in self.adata.var.index:
            raise KeyError(f"Gene '{gene}' not found in dataset")

        # Get gene index
        gene_idx = self.adata.var.index.get_loc(gene)

        # Get expression values from X matrix
        # Handle both dense and sparse matrices
        X = self.adata.X
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

        return {
            "gene": gene,
            "values": values_list,
            "min": min_val,
            "max": max_val,
        }

    def get_multi_gene_expression(self, genes: list[str]) -> dict[str, Any]:
        """Get mean expression values for multiple genes across all cells.

        Args:
            genes: List of gene names

        Returns:
            Dictionary containing:
            - genes: List of gene names used
            - values: List of mean expression values for each cell
            - min: Minimum mean expression value
            - max: Maximum mean expression value

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

        # Get gene indices
        gene_indices = [self.adata.var.index.get_loc(g) for g in genes]

        # Get expression values
        X = self.adata.X
        if hasattr(X, 'toarray'):
            # Sparse matrix
            expr_matrix = X[:, gene_indices].toarray()
        else:
            # Dense matrix
            expr_matrix = X[:, gene_indices]

        # Calculate mean expression across genes
        mean_expr = np.nanmean(expr_matrix, axis=1)

        # Convert to Python floats
        values_list = []
        for v in mean_expr:
            if np.isnan(v):
                values_list.append(None)
            else:
                values_list.append(float(v))

        valid_values = [v for v in values_list if v is not None]
        min_val = min(valid_values) if valid_values else 0
        max_val = max(valid_values) if valid_values else 0

        return {
            "genes": genes,
            "values": values_list,
            "min": min_val,
            "max": max_val,
        }

    # =========================================================================
    # Future scanpy integration methods (stubs for now)
    # =========================================================================

    # def run_pca(self, n_comps: int = 50, **kwargs):
    #     """Run PCA dimensionality reduction."""
    #     import scanpy as sc
    #     sc.tl.pca(self.adata, n_comps=n_comps, **kwargs)

    # def run_umap(self, **kwargs):
    #     """Run UMAP embedding."""
    #     import scanpy as sc
    #     sc.tl.umap(self.adata, **kwargs)

    # def run_leiden(self, resolution: float = 1.0, **kwargs):
    #     """Run Leiden clustering."""
    #     import scanpy as sc
    #     sc.tl.leiden(self.adata, resolution=resolution, **kwargs)

    # def run_diffexp(self, groupby: str, groups: list[str] | None = None, **kwargs):
    #     """Run differential expression analysis."""
    #     import scanpy as sc
    #     sc.tl.rank_genes_groups(self.adata, groupby=groupby, groups=groups, **kwargs)
