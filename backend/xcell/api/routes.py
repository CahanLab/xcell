"""API routes for XCell."""

from fastapi import APIRouter, HTTPException

from xcell.adaptor import DataAdaptor

router = APIRouter(prefix="/api")

# Global adaptor instance - set by main.py when loading data
_adaptor: DataAdaptor | None = None


def set_adaptor(adaptor: DataAdaptor) -> None:
    """Set the global data adaptor instance."""
    global _adaptor
    _adaptor = adaptor


def get_adaptor() -> DataAdaptor:
    """Get the global data adaptor instance."""
    if _adaptor is None:
        raise HTTPException(status_code=503, detail="No data loaded")
    return _adaptor


@router.get("/schema")
def get_schema():
    """Get dataset schema including available embeddings and metadata columns.

    Returns:
        JSON object containing:
        - n_cells: Number of cells
        - n_genes: Number of genes
        - embeddings: List of available embedding names
        - obs_columns: List of cell metadata column names
        - obs_dtypes: Dictionary mapping column names to their dtypes
    """
    adaptor = get_adaptor()
    return adaptor.get_schema()


@router.get("/embedding/{name}")
def get_embedding(name: str):
    """Get embedding coordinates by name.

    Args:
        name: Name of the embedding (e.g., 'X_umap', 'X_pca')

    Returns:
        JSON object containing:
        - name: The embedding name
        - coordinates: Array of [x, y] coordinate pairs
    """
    adaptor = get_adaptor()
    try:
        return adaptor.get_embedding(name)
    except KeyError as e:
        raise HTTPException(status_code=404, detail=str(e))


@router.get("/obs/{column}")
def get_obs_column(column: str):
    """Get cell metadata column values.

    Args:
        column: Name of the column in .obs

    Returns:
        JSON object containing:
        - name: The column name
        - values: Array of values for each cell
        - dtype: Data type ('category', 'numeric', or 'string')
        - categories: Array of category names (only for categorical columns)
    """
    adaptor = get_adaptor()
    try:
        return adaptor.get_obs_column(column)
    except KeyError as e:
        raise HTTPException(status_code=404, detail=str(e))


@router.get("/health")
def health_check():
    """Health check endpoint."""
    adaptor = get_adaptor()
    return {
        "status": "healthy",
        "n_cells": adaptor.n_cells,
        "n_genes": adaptor.n_genes,
    }
