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


# =========================================================================
# Cell metadata (obs) endpoints
# Note: Specific routes must come BEFORE parameterized routes
# =========================================================================


@router.get("/obs/summaries")
def get_all_obs_summaries():
    """Get summary statistics for all cell metadata columns.

    Returns:
        Array of summary objects for each obs column.
    """
    adaptor = get_adaptor()
    return adaptor.get_all_obs_summaries()


@router.get("/obs/summary/{column}")
def get_obs_summary(column: str):
    """Get summary statistics for a cell metadata column.

    For categorical columns: returns categories with cell counts.
    For numeric columns: returns min, max, mean.

    Args:
        column: Name of the column in .obs

    Returns:
        JSON object containing:
        - name: The column name
        - dtype: Data type ('category', 'numeric', or 'string')
        - For categorical: categories (array of {value, count} objects)
        - For numeric: min, max, mean
    """
    adaptor = get_adaptor()
    try:
        return adaptor.get_obs_column_summary(column)
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


# =========================================================================
# Annotation management endpoints
# =========================================================================

from pydantic import BaseModel
from fastapi.responses import PlainTextResponse


class CreateAnnotationRequest(BaseModel):
    name: str
    default_value: str = "unassigned"


class AddLabelRequest(BaseModel):
    label: str


class LabelCellsRequest(BaseModel):
    label: str
    cell_indices: list[int]


class ExportAnnotationsRequest(BaseModel):
    columns: list[str] | None = None


@router.post("/annotations")
def create_annotation(request: CreateAnnotationRequest):
    """Create a new categorical annotation column.

    Args:
        name: Name of the new annotation
        default_value: Default value for all cells (default: "unassigned")

    Returns:
        Summary of the new annotation column
    """
    adaptor = get_adaptor()
    try:
        return adaptor.create_annotation(request.name, request.default_value)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.post("/annotations/{name}/labels")
def add_label_to_annotation(name: str, request: AddLabelRequest):
    """Add a new label to an annotation column.

    Args:
        name: Name of the annotation column
        label: New label to add

    Returns:
        Updated annotation summary
    """
    adaptor = get_adaptor()
    try:
        return adaptor.add_label_to_annotation(name, request.label)
    except (KeyError, ValueError) as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.post("/annotations/{name}/label-cells")
def label_cells(name: str, request: LabelCellsRequest):
    """Assign a label to specific cells.

    Args:
        name: Name of the annotation column
        label: Label to assign
        cell_indices: List of cell indices to label

    Returns:
        Updated annotation summary
    """
    adaptor = get_adaptor()
    try:
        return adaptor.label_cells(name, request.label, request.cell_indices)
    except (KeyError, ValueError) as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.delete("/annotations/{name}")
def delete_annotation(name: str):
    """Delete an annotation column.

    Args:
        name: Name of the annotation column to delete

    Returns:
        Success message
    """
    adaptor = get_adaptor()
    try:
        adaptor.delete_annotation(name)
        return {"status": "ok", "message": f"Deleted annotation '{name}'"}
    except KeyError as e:
        raise HTTPException(status_code=404, detail=str(e))


@router.post("/annotations/export")
def export_annotations(request: ExportAnnotationsRequest):
    """Export cell annotations as TSV.

    Args:
        columns: List of column names to export. If null, exports all.

    Returns:
        TSV file as text
    """
    adaptor = get_adaptor()
    try:
        tsv = adaptor.export_annotations(request.columns)
        return PlainTextResponse(
            content=tsv,
            media_type="text/tab-separated-values",
            headers={"Content-Disposition": "attachment; filename=annotations.tsv"}
        )
    except KeyError as e:
        raise HTTPException(status_code=400, detail=str(e))


# =========================================================================
# Differential expression endpoints
# =========================================================================


class DiffExpRequest(BaseModel):
    """Request model for differential expression analysis."""
    group1: list[int]
    group2: list[int]
    top_n: int = 10


class DiffExpGene(BaseModel):
    """A single gene result from differential expression."""
    gene: str
    log2fc: float
    pval: float
    pval_adj: float


class DiffExpResponse(BaseModel):
    """Response model for differential expression analysis."""
    positive: list[DiffExpGene]
    negative: list[DiffExpGene]
    group1_count: int
    group2_count: int


@router.post("/diffexp", response_model=DiffExpResponse)
def run_diffexp(request: DiffExpRequest):
    """Run differential expression analysis between two cell groups.

    Uses Welch's t-test to identify differentially expressed genes.

    Args:
        group1: List of cell indices for group 1
        group2: List of cell indices for group 2
        top_n: Number of top genes to return for each direction (default: 10)

    Returns:
        JSON object containing:
        - positive: Top N genes upregulated in group1
        - negative: Top N genes upregulated in group2
        - group1_count: Number of cells in group 1
        - group2_count: Number of cells in group 2
    """
    adaptor = get_adaptor()
    try:
        return adaptor.run_diffexp(
            group1_indices=request.group1,
            group2_indices=request.group2,
            top_n=request.top_n,
        )
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
