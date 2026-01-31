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
