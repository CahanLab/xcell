"""API routes for XCell."""

from fastapi import APIRouter, HTTPException
from fastapi.responses import PlainTextResponse
from pydantic import BaseModel

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
def get_expression(gene: str, transform: str | None = None):
    """Get expression values for a single gene.

    Args:
        gene: Gene name
        transform: Optional transformation to apply. Supported values:
            - "log1p": Apply normalize_total followed by log1p transformation

    Returns:
        JSON object containing:
        - gene: The gene name
        - values: Array of expression values for each cell
        - min: Minimum expression value
        - max: Maximum expression value
        - transform: The transformation applied (if any)
    """
    adaptor = get_adaptor()
    try:
        return adaptor.get_expression(gene, transform=transform)
    except KeyError as e:
        raise HTTPException(status_code=404, detail=str(e))


class MultiExpressionRequest(BaseModel):
    """Request model for multi-gene expression."""
    genes: list[str]
    transform: str | None = None


@router.post("/expression/multi")
def get_multi_expression(request: MultiExpressionRequest):
    """Get mean expression values for multiple genes.

    Args:
        genes: List of gene names
        transform: Optional transformation to apply. Supported values:
            - "log1p": Apply normalize_total followed by log1p transformation

    Returns:
        JSON object containing:
        - genes: List of gene names used
        - values: Array of mean expression values for each cell
        - min: Minimum mean expression value
        - max: Maximum mean expression value
        - transform: The transformation applied (if any)
    """
    adaptor = get_adaptor()
    try:
        return adaptor.get_multi_gene_expression(request.genes, transform=request.transform)
    except KeyError as e:
        raise HTTPException(status_code=404, detail=str(e))


class BivariateExpressionRequest(BaseModel):
    """Request model for bivariate gene expression."""
    genes1: list[str]
    genes2: list[str]
    transform: str | None = None
    clip_percentiles: tuple[float, float] = (0, 99)


@router.post("/expression/bivariate")
def get_bivariate_expression(request: BivariateExpressionRequest):
    """Get normalized expression for two gene sets for bivariate visualization.

    Args:
        genes1: List of gene names for set 1 (maps to red/x-axis)
        genes2: List of gene names for set 2 (maps to blue/y-axis)
        transform: Optional transformation ('log1p' for normalize_total + log1p)
        clip_percentiles: Tuple of (low, high) percentiles for clipping

    Returns:
        JSON object containing:
        - genes1: List of gene names for set 1
        - genes2: List of gene names for set 2
        - values1: Normalized [0,1] expression values for gene set 1
        - values2: Normalized [0,1] expression values for gene set 2
        - transform: The transformation applied (if any)
    """
    adaptor = get_adaptor()
    try:
        return adaptor.get_bivariate_expression(
            genes1=request.genes1,
            genes2=request.genes2,
            transform=request.transform,
            clip_percentiles=request.clip_percentiles,
        )
    except (KeyError, ValueError) as e:
        raise HTTPException(status_code=400, detail=str(e))


# =========================================================================
# Annotation management endpoints
# =========================================================================


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


# =========================================================================
# Line / trajectory endpoints
# =========================================================================


class LineData(BaseModel):
    """Data for a single drawn line."""
    name: str
    embeddingName: str
    points: list[list[float]]
    smoothedPoints: list[list[float]] | None = None


class SetLinesRequest(BaseModel):
    """Request model for setting lines."""
    lines: list[LineData]


@router.post("/lines")
def set_lines(request: SetLinesRequest):
    """Store drawn lines from the frontend.

    These lines will be included in h5ad exports with:
    - Line metadata in .uns['xcell_lines']
    - Cell projections in .obsm['X_{line_name}_projection']

    Args:
        lines: List of line objects with name, embedding, and points

    Returns:
        Confirmation with line count
    """
    adaptor = get_adaptor()
    # Convert Pydantic models to dicts
    lines_data = [line.model_dump() for line in request.lines]
    adaptor.set_lines(lines_data)
    return {"status": "ok", "line_count": len(lines_data)}


@router.get("/lines")
def get_lines():
    """Get currently stored lines.

    Returns:
        List of stored line objects
    """
    adaptor = get_adaptor()
    return {"lines": adaptor.get_lines()}


# =========================================================================
# Scanpy analysis endpoints
# =========================================================================


class FilterGenesRequest(BaseModel):
    min_counts: int | None = None
    max_counts: int | None = None
    min_cells: int | None = None
    max_cells: int | None = None


class FilterCellsRequest(BaseModel):
    min_counts: int | None = None
    max_counts: int | None = None
    min_genes: int | None = None
    max_genes: int | None = None


class NormalizeTotalRequest(BaseModel):
    target_sum: float | None = None


class PcaRequest(BaseModel):
    n_comps: int = 50
    svd_solver: str = 'arpack'


class NeighborsRequest(BaseModel):
    n_neighbors: int = 15
    n_pcs: int | None = None
    metric: str = 'euclidean'


class UmapRequest(BaseModel):
    min_dist: float = 0.5
    spread: float = 1.0
    n_components: int = 2


class LeidenRequest(BaseModel):
    resolution: float = 1.0
    key_added: str = 'leiden'


@router.get("/scanpy/history")
def get_action_history():
    """Get the history of scanpy operations performed.

    Returns:
        List of action records with timestamps
    """
    adaptor = get_adaptor()
    return {"history": adaptor.get_action_history()}


@router.get("/scanpy/prerequisites/{action}")
def check_prerequisites(action: str):
    """Check if prerequisites are met for a scanpy action.

    Args:
        action: The scanpy action to check

    Returns:
        Dict with satisfied (bool) and missing prerequisites
    """
    adaptor = get_adaptor()
    return adaptor.check_prerequisites(action)


@router.post("/scanpy/filter_genes")
def run_filter_genes(request: FilterGenesRequest):
    """Filter genes based on counts or number of cells expressing.

    Returns:
        Before/after gene counts
    """
    adaptor = get_adaptor()
    try:
        return adaptor.run_filter_genes(
            min_counts=request.min_counts,
            max_counts=request.max_counts,
            min_cells=request.min_cells,
            max_cells=request.max_cells,
        )
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.post("/scanpy/filter_cells")
def run_filter_cells(request: FilterCellsRequest):
    """Filter cells based on counts or number of genes expressed.

    Returns:
        Before/after cell counts
    """
    adaptor = get_adaptor()
    try:
        return adaptor.run_filter_cells(
            min_counts=request.min_counts,
            max_counts=request.max_counts,
            min_genes=request.min_genes,
            max_genes=request.max_genes,
        )
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.post("/scanpy/normalize_total")
def run_normalize_total(request: NormalizeTotalRequest):
    """Normalize total counts per cell.

    Returns:
        Operation status
    """
    adaptor = get_adaptor()
    try:
        return adaptor.run_normalize_total(target_sum=request.target_sum)
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.post("/scanpy/log1p")
def run_log1p():
    """Apply log1p transformation.

    Returns:
        Operation status
    """
    adaptor = get_adaptor()
    try:
        return adaptor.run_log1p()
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.post("/scanpy/pca")
def run_pca(request: PcaRequest):
    """Run PCA dimensionality reduction.

    Returns:
        Operation status and variance explained
    """
    adaptor = get_adaptor()
    try:
        return adaptor.run_pca(
            n_comps=request.n_comps,
            svd_solver=request.svd_solver,
        )
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.post("/scanpy/neighbors")
def run_neighbors(request: NeighborsRequest):
    """Compute neighborhood graph.

    Requires: PCA must be computed first.

    Returns:
        Operation status
    """
    adaptor = get_adaptor()
    try:
        return adaptor.run_neighbors(
            n_neighbors=request.n_neighbors,
            n_pcs=request.n_pcs,
            metric=request.metric,
        )
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/scanpy/umap")
def run_umap(request: UmapRequest):
    """Compute UMAP embedding.

    Requires: Neighbors must be computed first.

    Returns:
        Operation status and embedding name
    """
    adaptor = get_adaptor()
    try:
        return adaptor.run_umap(
            min_dist=request.min_dist,
            spread=request.spread,
            n_components=request.n_components,
        )
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/scanpy/leiden")
def run_leiden(request: LeidenRequest):
    """Run Leiden clustering.

    Requires: Neighbors must be computed first.

    Returns:
        Operation status and cluster info
    """
    adaptor = get_adaptor()
    try:
        return adaptor.run_leiden(
            resolution=request.resolution,
            key_added=request.key_added,
        )
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# =========================================================================
# Export endpoints
# =========================================================================

from fastapi.responses import FileResponse
import tempfile
import os


@router.get("/export/h5ad")
def export_h5ad():
    """Export the current AnnData object as an h5ad file.

    This includes:
    - Any new annotation columns that were created
    - Drawn lines stored in .uns['xcell_lines']
    - Cell projections onto lines in .obsm['X_{line_name}_projection']

    Returns:
        The h5ad file as a download
    """
    adaptor = get_adaptor()
    try:
        # Create a temporary file
        fd, temp_path = tempfile.mkstemp(suffix='.h5ad')
        os.close(fd)

        # Get adata with lines and projections included
        adata_export = adaptor.prepare_export_with_lines()

        # Write to the temp file
        adata_export.write_h5ad(temp_path)

        return FileResponse(
            path=temp_path,
            filename="xcell_export.h5ad",
            media_type="application/octet-stream",
            background=None,  # Don't delete file in background task
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
