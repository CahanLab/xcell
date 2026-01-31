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


# =========================================================================
# Gene endpoints
# =========================================================================


@router.get("/genes")
def get_genes():
    """Get all gene names in the dataset.

    Returns:
        JSON object containing:
        - genes: Array of all gene names
        - count: Total number of genes
    """
    adaptor = get_adaptor()
    genes = adaptor.get_gene_names()
    return {
        "genes": genes,
        "count": len(genes),
    }


@router.get("/genes/search")
def search_genes(q: str, limit: int = 20):
    """Search for genes by name.

    Args:
        q: Search query (prefix or substring match)
        limit: Maximum number of results (default 20)

    Returns:
        JSON object containing:
        - query: The search query
        - genes: Array of matching gene names
    """
    adaptor = get_adaptor()
    matches = adaptor.search_genes(q, limit=limit)
    return {
        "query": q,
        "genes": matches,
    }


@router.get("/expression/{gene}")
def get_expression(gene: str):
    """Get expression values for a single gene.

    Args:
        gene: Gene name

    Returns:
        JSON object containing:
        - gene: The gene name
        - values: Array of expression values for each cell
        - min: Minimum expression value
        - max: Maximum expression value
    """
    adaptor = get_adaptor()
    try:
        return adaptor.get_expression(gene)
    except KeyError as e:
        raise HTTPException(status_code=404, detail=str(e))


@router.post("/expression/multi")
def get_multi_expression(genes: list[str]):
    """Get mean expression values for multiple genes.

    Args:
        genes: List of gene names in request body

    Returns:
        JSON object containing:
        - genes: List of gene names used
        - values: Array of mean expression values for each cell
        - min: Minimum mean expression value
        - max: Maximum mean expression value
    """
    adaptor = get_adaptor()
    try:
        return adaptor.get_multi_gene_expression(genes)
    except KeyError as e:
        raise HTTPException(status_code=404, detail=str(e))
