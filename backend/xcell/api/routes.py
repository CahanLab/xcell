"""API routes for XCell."""

from pathlib import Path
import shutil
import subprocess
import tempfile
from typing import Any

from fastapi import APIRouter, HTTPException, Query
from fastapi.responses import PlainTextResponse
from pydantic import BaseModel

from xcell.adaptor import DataAdaptor
from xcell.task_manager import task_manager

router = APIRouter(prefix="/api")

# Multi-dataset registry - keyed by slot name (e.g. "primary", "secondary")
_adaptors: dict[str, DataAdaptor] = {}


def set_adaptor(adaptor: DataAdaptor, slot: str = "primary") -> None:
    """Store a DataAdaptor in a named slot."""
    _adaptors[slot] = adaptor


def get_adaptor(slot: str | None = None) -> DataAdaptor:
    """Get the DataAdaptor for a slot. Defaults to 'primary'."""
    key = slot or "primary"
    if key not in _adaptors:
        raise HTTPException(status_code=503, detail=f"No data loaded for slot '{key}'")
    return _adaptors[key]


def remove_adaptor(slot: str) -> None:
    """Remove a DataAdaptor from a slot."""
    _adaptors.pop(slot, None)


def list_adaptors() -> dict[str, dict]:
    """Return info for each loaded dataset slot."""
    return {
        slot: {
            "filepath": str(a.filepath),
            "n_cells": a.adata.n_obs,
            "n_genes": a.adata.n_vars,
        }
        for slot, a in _adaptors.items()
    }


@router.get("/browse")
def browse_filesystem(path: str | None = None):
    """List directories and .h5ad/.h5/.rds files at the given path.

    Args:
        path: Directory path to list. Defaults to the user's home directory.

    Returns:
        JSON object with current path, parent path, and entries (dirs + h5ad/h5 files).
    """
    if path:
        directory = Path(path).expanduser().resolve()
    else:
        directory = Path.home()

    if not directory.is_dir():
        raise HTTPException(status_code=400, detail=f"Not a directory: {directory}")

    parent = str(directory.parent) if directory != directory.parent else None

    entries = []
    try:
        for item in sorted(directory.iterdir(), key=lambda p: (not p.is_dir(), p.name.lower())):
            if item.name.startswith('.'):
                continue
            if item.is_dir():
                # Check if this is a 10x CellRanger matrix folder
                children = {c.name for c in item.iterdir()} if item.is_dir() else set()
                has_matrix = bool(children & {'matrix.mtx', 'matrix.mtx.gz'})
                has_barcodes = bool(children & {'barcodes.tsv', 'barcodes.tsv.gz'})
                has_features = bool(children & {'features.tsv', 'features.tsv.gz', 'genes.tsv', 'genes.tsv.gz'})
                if has_matrix and has_barcodes and has_features:
                    entries.append({"name": item.name, "type": "10x_mtx", "path": str(item)})
                else:
                    entries.append({"name": item.name, "type": "directory", "path": str(item)})
            elif item.suffix in ('.h5ad', '.h5', '.rds'):
                try:
                    size = item.stat().st_size
                except OSError:
                    size = None
                entries.append({"name": item.name, "type": "file", "path": str(item), "size": size})
    except PermissionError:
        raise HTTPException(status_code=403, detail=f"Permission denied: {directory}")

    # Standard quick-access locations
    home = Path.home()
    shortcut_defs = [
        ("Home", home),
        ("Desktop", home / "Desktop"),
        ("Documents", home / "Documents"),
        ("Downloads", home / "Downloads"),
    ]
    shortcuts = [
        {"name": name, "path": str(p)}
        for name, p in shortcut_defs
        if p.is_dir()
    ]

    return {
        "current": str(directory),
        "parent": parent,
        "entries": entries,
        "shortcuts": shortcuts,
    }


def _convert_rds_to_h5ad(rds_path: Path) -> Path:
    """Convert a Seurat .rds file to .h5ad via Rscript subprocess.

    Returns the path to the converted .h5ad file (in a temp directory).
    Caller is responsible for cleanup only on error; the file is needed long-term.
    """
    if not shutil.which("Rscript"):
        raise HTTPException(
            status_code=400,
            detail="R is not installed. Install R and the Seurat/SeuratDisk packages to load .rds files.",
        )

    r_script = Path(__file__).resolve().parent.parent / "convert_seurat.R"
    if not r_script.exists():
        raise HTTPException(status_code=500, detail="convert_seurat.R not found in backend package.")

    # Create temp dir for intermediate and output files
    tmp_dir = tempfile.mkdtemp(prefix="xcell_rds_")
    h5seurat_path = Path(tmp_dir) / "converted.h5seurat"
    h5ad_path = Path(tmp_dir) / "converted.h5ad"

    try:
        result = subprocess.run(
            ["Rscript", str(r_script), str(rds_path), str(h5seurat_path)],
            capture_output=True,
            text=True,
            timeout=600,
        )
        if result.returncode != 0:
            raise HTTPException(
                status_code=500,
                detail=f"RDS conversion failed:\n{result.stderr}",
            )
        if not h5ad_path.exists():
            raise HTTPException(
                status_code=500,
                detail="RDS conversion produced no output. Check that Seurat and SeuratDisk R packages are installed.",
            )
        # Clean up intermediate .h5seurat file
        if h5seurat_path.exists():
            h5seurat_path.unlink()
        return h5ad_path
    except subprocess.TimeoutExpired:
        raise HTTPException(status_code=500, detail="RDS conversion timed out (>10 minutes).")
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"RDS conversion error: {e}")


class LoadRequest(BaseModel):
    file_path: str
    slot: str = "primary"


@router.post("/load")
def load_dataset(request: LoadRequest, dataset: str | None = Query(None)):
    """Load a new dataset from a server-side file path.

    Supports .h5ad, .h5, and .rds (Seurat) files. RDS files are converted
    to h5ad via Rscript before loading.

    Args:
        file_path: Absolute path to an .h5ad, .h5, or .rds file on the server filesystem
        slot: Named slot to load into (default: 'primary')

    Returns:
        The schema of the newly loaded dataset plus the slot name
    """
    target_slot = request.slot
    path = Path(request.file_path)
    if path.is_dir():
        # Check for 10x CellRanger matrix folder
        children = {c.name for c in path.iterdir()}
        has_matrix = bool(children & {'matrix.mtx', 'matrix.mtx.gz'})
        has_barcodes = bool(children & {'barcodes.tsv', 'barcodes.tsv.gz'})
        has_features = bool(children & {'features.tsv', 'features.tsv.gz', 'genes.tsv', 'genes.tsv.gz'})
        if not (has_matrix and has_barcodes and has_features):
            raise HTTPException(status_code=400, detail="Directory is not a valid 10x CellRanger matrix folder")
    elif path.suffix not in ('.h5ad', '.h5', '.rds'):
        raise HTTPException(status_code=400, detail="File must have .h5ad, .h5, or .rds extension")
    elif not path.exists():
        raise HTTPException(status_code=404, detail=f"File not found: {request.file_path}")
    try:
        load_path = path
        if path.suffix == '.rds':
            load_path = _convert_rds_to_h5ad(path)
        adaptor = DataAdaptor(load_path)
        # Store original filepath for display purposes
        adaptor.filepath = path
        set_adaptor(adaptor, slot=target_slot)
        return {"slot": target_slot, **adaptor.get_schema()}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to load file: {e}")


@router.get("/datasets")
def get_datasets():
    """List all loaded dataset slots with basic info."""
    return list_adaptors()


@router.delete("/datasets/{slot}")
def unload_dataset(slot: str):
    """Unload a dataset from a slot."""
    if slot not in _adaptors:
        raise HTTPException(status_code=404, detail=f"No dataset in slot '{slot}'")
    remove_adaptor(slot)
    return {"status": "ok", "slot": slot}


@router.get("/schema")
def get_schema(dataset: str | None = Query(None)):
    """Get dataset schema including available embeddings and metadata columns.

    Returns:
        JSON object containing:
        - n_cells: Number of cells
        - n_genes: Number of genes
        - embeddings: List of available embedding names
        - obs_columns: List of cell metadata column names
        - obs_dtypes: Dictionary mapping column names to their dtypes
    """
    adaptor = get_adaptor(dataset)
    return adaptor.get_schema()


@router.get("/embedding/{name}")
def get_embedding(name: str, dataset: str | None = Query(None)):
    """Get embedding coordinates by name.

    Args:
        name: Name of the embedding (e.g., 'X_umap', 'X_pca')

    Returns:
        JSON object containing:
        - name: The embedding name
        - coordinates: Array of [x, y] coordinate pairs
    """
    adaptor = get_adaptor(dataset)
    try:
        return adaptor.get_embedding(name)
    except KeyError as e:
        raise HTTPException(status_code=404, detail=str(e))


class TransformEmbeddingRequest(BaseModel):
    rotation_degrees: float = 0
    reflect_x: bool = False
    reflect_y: bool = False
    cell_indices: list[int] | None = None
    translate_x: float = 0.0
    translate_y: float = 0.0


@router.post("/embedding/{name}/transform")
def transform_embedding(name: str, request: TransformEmbeddingRequest, dataset: str | None = Query(None)):
    """Apply rotation and/or reflection to an embedding.

    Transforms are applied in-place around the centroid (reflections first, then rotation).

    Args:
        name: Name of the embedding
        request: Rotation angle (degrees) and reflection flags

    Returns:
        Updated embedding coordinates
    """
    adaptor = get_adaptor(dataset)
    try:
        return adaptor.transform_embedding(
            name,
            rotation_degrees=request.rotation_degrees,
            reflect_x=request.reflect_x,
            reflect_y=request.reflect_y,
            cell_indices=request.cell_indices,
            translate_x=request.translate_x,
            translate_y=request.translate_y,
        )
    except KeyError as e:
        raise HTTPException(status_code=404, detail=str(e))


@router.post("/embedding/{name}/undo")
def undo_transform_embedding(name: str, dataset: str | None = Query(None)):
    """Undo the last quilt transform for an embedding."""
    adaptor = get_adaptor(dataset)
    try:
        return adaptor.undo_transform_embedding(name)
    except (KeyError, ValueError) as e:
        raise HTTPException(status_code=400, detail=str(e))


# =========================================================================
# Cell metadata (obs) endpoints
# Note: Specific routes must come BEFORE parameterized routes
# =========================================================================


@router.get("/obs/summaries")
def get_all_obs_summaries(dataset: str | None = Query(None)):
    """Get summary statistics for all cell metadata columns.

    Returns:
        Array of summary objects for each obs column.
    """
    adaptor = get_adaptor(dataset)
    return adaptor.get_all_obs_summaries()


@router.get("/obs/summary/{column}")
def get_obs_summary(column: str, dataset: str | None = Query(None)):
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
    adaptor = get_adaptor(dataset)
    try:
        return adaptor.get_obs_column_summary(column)
    except KeyError as e:
        raise HTTPException(status_code=404, detail=str(e))


@router.get("/obs/{column}")
def get_obs_column(column: str, dataset: str | None = Query(None)):
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
    adaptor = get_adaptor(dataset)
    try:
        return adaptor.get_obs_column(column)
    except KeyError as e:
        raise HTTPException(status_code=404, detail=str(e))


@router.get("/health")
def health_check(dataset: str | None = Query(None)):
    """Health check endpoint."""
    adaptor = get_adaptor(dataset)
    return {
        "status": "healthy",
        "n_cells": adaptor.n_cells,
        "n_genes": adaptor.n_genes,
    }


# =========================================================================
# Gene endpoints
# =========================================================================


@router.get("/genes")
def get_genes(dataset: str | None = Query(None)):
    """Get all gene names in the dataset.

    Returns:
        JSON object containing:
        - genes: Array of all gene names
        - count: Total number of genes
    """
    adaptor = get_adaptor(dataset)
    genes = adaptor.get_gene_names()
    return {
        "genes": genes,
        "count": len(genes),
    }


@router.get("/genes/browse")
def browse_genes(offset: int = 0, limit: int = 50, dataset: str | None = Query(None)):
    """Browse genes with pagination (sorted alphabetically).

    Args:
        offset: Starting index (default 0)
        limit: Number of genes to return (default 50)

    Returns:
        JSON object containing:
        - genes: Array of gene names for this page
        - offset: The starting index used
        - limit: The page size used
        - total: Total number of genes
    """
    adaptor = get_adaptor(dataset)
    all_genes = sorted(adaptor.get_gene_names(), key=str.lower)
    total = len(all_genes)
    page = all_genes[offset:offset + limit]
    return {
        "genes": page,
        "offset": offset,
        "limit": limit,
        "total": total,
    }


@router.get("/genes/search")
def search_genes(q: str, limit: int = 20, dataset: str | None = Query(None)):
    """Search for genes by name.

    Args:
        q: Search query (prefix or substring match)
        limit: Maximum number of results (default 20)

    Returns:
        JSON object containing:
        - query: The search query
        - genes: Array of matching gene names
    """
    adaptor = get_adaptor(dataset)
    matches = adaptor.search_genes(q, limit=limit)
    return {
        "query": q,
        "genes": matches,
    }


@router.get("/expression/{gene}")
def get_expression(gene: str, transform: str | None = None, dataset: str | None = Query(None)):
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
    adaptor = get_adaptor(dataset)
    try:
        return adaptor.get_expression(gene, transform=transform)
    except KeyError as e:
        raise HTTPException(status_code=404, detail=str(e))


class MultiExpressionRequest(BaseModel):
    """Request model for multi-gene expression."""
    genes: list[str]
    transform: str | None = None
    scoring_method: str = 'mean'  # 'mean' or 'zscore'


@router.post("/expression/multi")
def get_multi_expression(request: MultiExpressionRequest, dataset: str | None = Query(None)):
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
    adaptor = get_adaptor(dataset)
    try:
        return adaptor.get_multi_gene_expression(
            request.genes,
            transform=request.transform,
            scoring_method=request.scoring_method,
        )
    except KeyError as e:
        raise HTTPException(status_code=404, detail=str(e))


class BivariateExpressionRequest(BaseModel):
    """Request model for bivariate gene expression."""
    genes1: list[str]
    genes2: list[str]
    transform: str | None = None
    clip_percentile: float = 1.0  # Symmetric percentile clipping (1.0 = clip at 1st/99th)
    scoring_method: str = 'zscore'  # 'mean' or 'zscore'


@router.post("/expression/bivariate")
def get_bivariate_expression(request: BivariateExpressionRequest, dataset: str | None = Query(None)):
    """Get normalized expression for two gene sets for bivariate visualization.

    Uses robust scoring: mean-centers each gene, scales by MAD to handle outliers,
    clips extreme values, then averages across genes.

    Args:
        genes1: List of gene names for set 1 (maps to red/x-axis)
        genes2: List of gene names for set 2 (maps to blue/y-axis)
        transform: Optional transformation ('log1p' for normalize_total + log1p)
        clip_percentile: Symmetric percentile for clipping (1.0 = clip at 1st/99th)

    Returns:
        JSON object containing:
        - genes1: List of gene names for set 1
        - genes2: List of gene names for set 2
        - values1: Normalized [0,1] expression values for gene set 1
        - values2: Normalized [0,1] expression values for gene set 2
        - transform: The transformation applied (if any)
    """
    adaptor = get_adaptor(dataset)
    try:
        return adaptor.get_bivariate_expression(
            genes1=request.genes1,
            genes2=request.genes2,
            transform=request.transform,
            clip_percentile=request.clip_percentile,
            scoring_method=request.scoring_method,
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
def create_annotation(request: CreateAnnotationRequest, dataset: str | None = Query(None)):
    """Create a new categorical annotation column.

    Args:
        name: Name of the new annotation
        default_value: Default value for all cells (default: "unassigned")

    Returns:
        Summary of the new annotation column
    """
    adaptor = get_adaptor(dataset)
    try:
        return adaptor.create_annotation(request.name, request.default_value)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.post("/annotations/{name}/labels")
def add_label_to_annotation(name: str, request: AddLabelRequest, dataset: str | None = Query(None)):
    """Add a new label to an annotation column.

    Args:
        name: Name of the annotation column
        label: New label to add

    Returns:
        Updated annotation summary
    """
    adaptor = get_adaptor(dataset)
    try:
        return adaptor.add_label_to_annotation(name, request.label)
    except (KeyError, ValueError) as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.post("/annotations/{name}/label-cells")
def label_cells(name: str, request: LabelCellsRequest, dataset: str | None = Query(None)):
    """Assign a label to specific cells.

    Args:
        name: Name of the annotation column
        label: Label to assign
        cell_indices: List of cell indices to label

    Returns:
        Updated annotation summary
    """
    adaptor = get_adaptor(dataset)
    try:
        return adaptor.label_cells(name, request.label, request.cell_indices)
    except (KeyError, ValueError) as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.delete("/annotations/{name}")
def delete_annotation(name: str, dataset: str | None = Query(None)):
    """Delete an annotation column.

    Args:
        name: Name of the annotation column to delete

    Returns:
        Success message
    """
    adaptor = get_adaptor(dataset)
    try:
        adaptor.delete_annotation(name)
        return {"status": "ok", "message": f"Deleted annotation '{name}'"}
    except KeyError as e:
        raise HTTPException(status_code=404, detail=str(e))


@router.post("/annotations/export")
def export_annotations(request: ExportAnnotationsRequest, dataset: str | None = Query(None)):
    """Export cell annotations as TSV.

    Args:
        columns: List of column names to export. If null, exports all.

    Returns:
        TSV file as text
    """
    adaptor = get_adaptor(dataset)
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
# Cell deletion endpoint
# =========================================================================


class DeleteCellsRequest(BaseModel):
    """Request model for permanently deleting cells."""
    cell_indices: list[int]


class DeleteCellsResponse(BaseModel):
    """Response model for cell deletion."""
    n_cells_before: int
    n_cells_after: int
    n_cells_deleted: int


@router.post("/cells/delete", response_model=DeleteCellsResponse)
def delete_cells(request: DeleteCellsRequest, dataset: str | None = Query(None)):
    """Permanently remove specific cells from the dataset.

    This is irreversible within the current session. The cells are removed
    from the underlying AnnData object.

    Args:
        cell_indices: List of cell indices to delete

    Returns:
        Before/after cell counts and number deleted
    """
    adaptor = get_adaptor(dataset)
    try:
        return adaptor.delete_cells(cell_indices=request.cell_indices)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))


# =========================================================================
# Differential expression endpoints
# =========================================================================


class DiffExpRequest(BaseModel):
    """Request model for differential expression analysis."""
    group1: list[int]
    group2: list[int]
    top_n: int = 10
    method: str = "wilcoxon"
    corr_method: str = "benjamini-hochberg"
    min_fold_change: float | None = None
    min_in_group_fraction: float | None = None
    max_out_group_fraction: float | None = None
    max_pval_adj: float | None = None
    gene_subset: str | None = None


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
def run_diffexp(request: DiffExpRequest, dataset: str | None = Query(None)):
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
    adaptor = get_adaptor(dataset)
    try:
        return adaptor.run_diffexp(
            group1_indices=request.group1,
            group2_indices=request.group2,
            top_n=request.top_n,
            method=request.method,
            corr_method=request.corr_method,
            min_fold_change=request.min_fold_change,
            min_in_group_fraction=request.min_in_group_fraction,
            max_out_group_fraction=request.max_out_group_fraction,
            max_pval_adj=request.max_pval_adj,
            gene_subset=request.gene_subset,
        )
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))


# =========================================================================
# Marker genes (one-vs-rest) endpoints
# =========================================================================


class MarkerGenesRequest(BaseModel):
    """Request model for one-vs-rest marker gene analysis."""
    obs_column: str
    groups: list[str] | None = None
    top_n: int = 25
    min_in_group_fraction: float | None = None
    max_out_group_fraction: float | None = None
    min_fold_change: float | None = None
    gene_subset: str | None = None


class MarkerGeneEntry(BaseModel):
    """A single gene result from marker gene analysis."""
    gene: str
    log2fc: float
    pval: float
    pval_adj: float


class MarkerGenesGroupResult(BaseModel):
    """Results for one group in marker gene analysis."""
    group: str
    genes: list[MarkerGeneEntry]


class MarkerGenesResponse(BaseModel):
    """Response model for marker gene analysis."""
    obs_column: str
    results: list[MarkerGenesGroupResult]


@router.post("/marker-genes", response_model=MarkerGenesResponse)
def run_marker_genes(request: MarkerGenesRequest, dataset: str | None = Query(None)):
    """Run one-vs-rest marker gene analysis for groups in a categorical column.

    Uses scanpy's rank_genes_groups with Wilcoxon rank-sum test to identify
    marker genes for each group (one-vs-rest).

    Args:
        obs_column: Categorical column in .obs to group by
        groups: Optional subset of groups to include (default: all)
        top_n: Number of top marker genes per group (default: 25)
        min_in_group_fraction: Min fraction of cells in group expressing gene
        max_out_group_fraction: Max fraction of cells outside group expressing gene
        min_fold_change: Minimum fold change threshold

    Returns:
        Per-group lists of marker genes with statistics
    """
    adaptor = get_adaptor(dataset)
    try:
        return adaptor.run_marker_genes(
            obs_column=request.obs_column,
            groups=request.groups,
            top_n=request.top_n,
            min_in_group_fraction=request.min_in_group_fraction,
            max_out_group_fraction=request.max_out_group_fraction,
            min_fold_change=request.min_fold_change,
            gene_subset=request.gene_subset,
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
def set_lines(request: SetLinesRequest, dataset: str | None = Query(None)):
    """Store drawn lines from the frontend.

    These lines will be included in h5ad exports with:
    - Line metadata in .uns['xcell_lines']
    - Cell projections in .obsm['X_{line_name}_projection']

    Args:
        lines: List of line objects with name, embedding, and points

    Returns:
        Confirmation with line count
    """
    adaptor = get_adaptor(dataset)
    # Convert Pydantic models to dicts
    lines_data = [line.model_dump() for line in request.lines]
    adaptor.set_lines(lines_data)
    return {"status": "ok", "line_count": len(lines_data)}


@router.get("/lines")
def get_lines(dataset: str | None = Query(None)):
    """Get currently stored lines.

    Returns:
        List of stored line objects
    """
    adaptor = get_adaptor(dataset)
    return {"lines": adaptor.get_lines()}


@router.get("/lines/debug/{line_name}")
def debug_line_projection(line_name: str, dataset: str | None = Query(None)):
    """Debug endpoint to inspect line projection data.

    Returns detailed information about the line and sample projections.
    """
    import numpy as np

    adaptor = get_adaptor(dataset)

    # Find the line
    line = None
    for l in adaptor._drawn_lines:
        if l.get('name') == line_name:
            line = l
            break

    if line is None:
        raise HTTPException(status_code=404, detail=f"Line '{line_name}' not found")

    # Get line points
    points = line.get('points', [])
    smoothed = line.get('smoothedPoints')
    line_points = smoothed if smoothed else points

    # Get embedding
    embedding_name = line.get('embeddingName', '')
    if embedding_name not in adaptor.adata.obsm:
        raise HTTPException(status_code=400, detail=f"Embedding '{embedding_name}' not found")

    coords = adaptor.adata.obsm[embedding_name][:, :2]

    # Compute projections
    positions, distances = adaptor._project_cells_onto_line(line_points, coords)

    # Sample some cells
    sample_indices = [0, 100, 500, 1000, 2000] if len(positions) > 2000 else list(range(min(10, len(positions))))
    sample_indices = [i for i in sample_indices if i < len(positions)]

    return {
        "line_name": line_name,
        "embedding_name": embedding_name,
        "n_line_points": len(line_points),
        "line_points_sample": line_points[:5] if len(line_points) > 5 else line_points,
        "line_points_range": {
            "x": [float(min(p[0] for p in line_points)), float(max(p[0] for p in line_points))],
            "y": [float(min(p[1] for p in line_points)), float(max(p[1] for p in line_points))],
        } if line_points else None,
        "embedding_range": {
            "x": [float(coords[:, 0].min()), float(coords[:, 0].max())],
            "y": [float(coords[:, 1].min()), float(coords[:, 1].max())],
        },
        "n_cells": len(positions),
        "position_stats": {
            "min": float(positions.min()),
            "max": float(positions.max()),
            "mean": float(positions.mean()),
            "std": float(positions.std()),
            "unique_count": len(np.unique(positions)),
        },
        "distance_stats": {
            "min": float(distances.min()),
            "max": float(distances.max()),
            "mean": float(distances.mean()),
        },
        "sample_projections": [
            {
                "cell_idx": int(i),
                "cell_coords": [float(coords[i, 0]), float(coords[i, 1])],
                "position": float(positions[i]),
                "distance": float(distances[i]),
            }
            for i in sample_indices
        ],
    }


class LineAssociationRequest(BaseModel):
    """Request model for line association testing."""
    line_name: str
    cell_indices: list[int] | None = None
    gene_subset: str | list[str] | None = None
    test_variable: str = 'position'  # 'position' or 'distance'
    n_spline_knots: int = 5
    min_cells: int = 20
    fdr_threshold: float = 0.05
    top_n: int = 50


class LineAssociationGene(BaseModel):
    """A gene result from line association testing."""
    gene: str
    f_stat: float
    pval: float
    fdr: float
    r_squared: float
    amplitude: float
    direction: float
    profile: list[float] | None = None  # Smoothed expression profile (normalized 0-1)
    peak_position: float | None = None  # Position along line where expression peaks


class LineAssociationDiagnostics(BaseModel):
    """Diagnostic information from line association testing."""
    n_genes_tested: int
    n_pval_below_05: int
    n_pval_below_01: int
    position_range: list[float]
    position_std: float
    expression_range: list[float]
    expression_mean: float
    n_zero_genes: int
    spline_df: int


class LineAssociationModule(BaseModel):
    """A module of genes with similar expression profiles along a line."""
    module_id: int
    pattern: str                          # 'increasing', 'decreasing', 'peak', 'trough', 'complex'
    n_genes: int
    representative_profile: list[float]   # normalized 0-1 profile at evenly-spaced positions
    profile_positions: list[float]        # corresponding position values (0-1)
    genes: list[LineAssociationGene]


class LineAssociationResponse(BaseModel):
    """Response model for line association testing."""
    positive: list[LineAssociationGene]
    negative: list[LineAssociationGene]
    modules: list[LineAssociationModule] = []
    n_cells: int
    n_significant: int
    n_positive: int
    n_negative: int
    n_modules: int = 0
    line_name: str
    test_variable: str = 'position'
    fdr_threshold: float
    n_lines: int = 1
    lines_used: list[str] = []
    diagnostics: LineAssociationDiagnostics | None = None


@router.post("/lines/association", response_model=LineAssociationResponse)
def test_line_association(request: LineAssociationRequest, dataset: str | None = Query(None)):
    """Test genes for association with position along a line.

    Uses cubic B-spline regression to model gene expression as a function
    of position along the line, then tests significance via F-test.

    Args:
        line_name: Name of the line to test against
        cell_indices: Optional list of cell indices to use
        n_spline_knots: Number of interior knots for spline (default: 5)
        min_cells: Minimum cells required (default: 20)
        fdr_threshold: FDR significance threshold (default: 0.05)
        top_n: Number of top genes per direction (default: 50)
        use_log1p: Apply log1p transform (default: True)

    Returns:
        Genes with expression associated with line position:
        - positive: genes increasing along line
        - negative: genes decreasing along line
    """
    adaptor = get_adaptor(dataset)
    try:
        return adaptor.test_line_association(
            line_name=request.line_name,
            cell_indices=request.cell_indices,
            gene_subset=request.gene_subset,
            test_variable=request.test_variable,
            n_spline_knots=request.n_spline_knots,
            min_cells=request.min_cells,
            fdr_threshold=request.fdr_threshold,
            top_n=request.top_n,
        )
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))


class MultiLineEntry(BaseModel):
    """A single line entry for multi-line association testing."""
    name: str
    cell_indices: list[int]
    reversed: bool = False


class MultiLineAssociationRequest(BaseModel):
    """Request model for multi-line association testing."""
    lines: list[MultiLineEntry]
    gene_subset: str | list[str] | None = None
    test_variable: str = 'position'
    n_spline_knots: int = 5
    min_cells: int = 20
    fdr_threshold: float = 0.05
    top_n: int = 50


@router.post("/lines/multi-association", response_model=LineAssociationResponse)
def test_multi_line_association(request: MultiLineAssociationRequest, dataset: str | None = Query(None)):
    """Test genes for association across multiple lines (pooled analysis).

    Projects cells from each line onto their respective geometry, pools them,
    and runs cubic B-spline regression to find genes associated with position
    along or distance from the combined trajectory.

    Returns:
        Genes with expression associated with pooled line position.
    """
    adaptor = get_adaptor(dataset)
    try:
        return adaptor.test_multi_line_association(
            lines=[{'name': e.name, 'cell_indices': e.cell_indices, 'reversed': e.reversed} for e in request.lines],
            gene_subset=request.gene_subset,
            test_variable=request.test_variable,
            n_spline_knots=request.n_spline_knots,
            min_cells=request.min_cells,
            fdr_threshold=request.fdr_threshold,
            top_n=request.top_n,
        )
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))


class CreateLineEmbeddingRequest(BaseModel):
    """Request model for creating a line projection embedding."""
    line_name: str
    cell_indices: list[int] | None = None


class CreateLineEmbeddingResponse(BaseModel):
    """Response model for line projection embedding creation."""
    embedding_name: str
    n_cells: int
    position_range: list[float]
    distance_range_original: list[float]
    distance_range_normalized: list[float]


@router.post("/lines/create-embedding", response_model=CreateLineEmbeddingResponse)
def create_line_embedding(request: CreateLineEmbeddingRequest, dataset: str | None = Query(None)):
    """Create an embedding from cell projections onto a line.

    Creates a new embedding in .obsm where:
    - X-axis: position along the line (0-1)
    - Y-axis: distance from the line (normalized to 0-1)

    This allows visualizing cells by their position along a trajectory
    and coloring by gene expression.

    Args:
        line_name: Name of the line to project onto
        cell_indices: Optional cell indices to include

    Returns:
        The new embedding name and statistics
    """
    adaptor = get_adaptor(dataset)
    try:
        # First sync lines (they need to exist on the backend)
        result = adaptor.create_line_projection_embedding(
            line_name=request.line_name,
            cell_indices=request.cell_indices,
        )
        return result
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))


# =========================================================================
# Heatmap endpoint
# =========================================================================


class HeatmapGeneSetGroup(BaseModel):
    name: str
    genes: list[str]


class HeatmapRequest(BaseModel):
    """Request model for heatmap data computation."""
    genes: list[str]
    gene_set_groups: list[HeatmapGeneSetGroup] | None = None
    aggregate_gene_sets: bool = False
    cell_ordering: str = "none"
    obs_column: str | None = None
    line_name: str | None = None
    gene_ordering: str = "as_provided"
    n_bins: int = 0
    transform: str | None = None
    cell_indices: list[int] | None = None


@router.post("/heatmap/data")
def get_heatmap_data(request: HeatmapRequest, dataset: str | None = Query(None)):
    """Compute expression heatmap matrix.

    Returns a per-row normalized expression matrix with cells ordered
    and optionally binned. Used by the Heatmap tab in the center panel.
    """
    from xcell.heatmap import compute_heatmap_data

    adaptor = get_adaptor(dataset)
    try:
        gene_set_groups = None
        if request.gene_set_groups:
            gene_set_groups = [
                {"name": g.name, "genes": g.genes} for g in request.gene_set_groups
            ]
        return compute_heatmap_data(
            adaptor,
            genes=request.genes,
            gene_set_groups=gene_set_groups,
            aggregate_gene_sets=request.aggregate_gene_sets,
            cell_ordering=request.cell_ordering,
            obs_column=request.obs_column,
            line_name=request.line_name,
            gene_ordering=request.gene_ordering,
            n_bins=request.n_bins,
            transform=request.transform,
            cell_indices=request.cell_indices,
        )
    except (ValueError, KeyError) as e:
        raise HTTPException(status_code=400, detail=str(e))


# =========================================================================
# Scanpy analysis endpoints
# =========================================================================


class ExcludeGenesRequest(BaseModel):
    gene_names: list[str] | None = None
    patterns: list[str] | None = None


class FilterGenesRequest(BaseModel):
    min_counts: int | None = None
    max_counts: int | None = None
    min_cells: int | None = None
    max_cells: int | None = None
    active_cell_indices: list[int] | None = None


class FilterCellsRequest(BaseModel):
    min_counts: int | None = None
    max_counts: int | None = None
    min_genes: int | None = None
    max_genes: int | None = None
    active_cell_indices: list[int] | None = None


class NormalizeTotalRequest(BaseModel):
    target_sum: float | None = None
    active_cell_indices: list[int] | None = None


class HighlyVariableGenesRequest(BaseModel):
    n_top_genes: int | None = None
    min_mean: float = 0.0125
    max_mean: float = 3.0
    min_disp: float = 0.5
    flavor: str = 'seurat'
    n_bins: int = 20
    subset: bool = False
    active_cell_indices: list[int] | None = None


class GeneSubsetSpec(BaseModel):
    """Specification for combining multiple boolean columns."""
    columns: list[str]
    operation: str = 'intersection'  # 'intersection' (AND) or 'union' (OR)


class PcaRequest(BaseModel):
    n_comps: int = 50
    svd_solver: str = 'arpack'
    # Gene subset can be:
    # - None: default behavior (use highly_variable if available)
    # - str: boolean column name (e.g., 'highly_variable', 'spatially_variable')
    # - list[str]: explicit gene names
    # - GeneSubsetSpec: combine multiple columns with AND/OR
    gene_subset: str | list[str] | GeneSubsetSpec | None = None
    active_cell_indices: list[int] | None = None


class NeighborsRequest(BaseModel):
    n_neighbors: int = 15
    n_pcs: int | None = None
    metric: str = 'euclidean'
    active_cell_indices: list[int] | None = None


class UmapRequest(BaseModel):
    min_dist: float = 0.5
    spread: float = 1.0
    n_components: int = 2
    active_cell_indices: list[int] | None = None


class LeidenRequest(BaseModel):
    resolution: float = 1.0
    key_added: str = 'leiden'
    active_cell_indices: list[int] | None = None


@router.get("/scanpy/history")
def get_action_history(dataset: str | None = Query(None)):
    """Get the history of scanpy operations performed.

    Returns:
        List of action records with timestamps
    """
    adaptor = get_adaptor(dataset)
    return {"history": adaptor.get_action_history()}


@router.get("/scanpy/prerequisites/{action}")
def check_prerequisites(action: str, dataset: str | None = Query(None)):
    """Check if prerequisites are met for a scanpy action.

    Args:
        action: The scanpy action to check

    Returns:
        Dict with satisfied (bool) and missing prerequisites
    """
    adaptor = get_adaptor(dataset)
    return adaptor.check_prerequisites(action)


@router.post("/scanpy/exclude_genes")
def run_exclude_genes(request: ExcludeGenesRequest, dataset: str | None = Query(None)):
    """Remove genes by exact name or regex pattern.

    Returns:
        Before/after gene counts and list of removed genes
    """
    adaptor = get_adaptor(dataset)
    try:
        return adaptor.run_exclude_genes(
            gene_names=request.gene_names,
            patterns=request.patterns,
        )
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.post("/scanpy/filter_genes")
def run_filter_genes(request: FilterGenesRequest, dataset: str | None = Query(None)):
    """Filter genes based on counts or number of cells expressing.

    Returns:
        Before/after gene counts
    """
    adaptor = get_adaptor(dataset)
    try:
        return adaptor.run_filter_genes(
            min_counts=request.min_counts,
            max_counts=request.max_counts,
            min_cells=request.min_cells,
            max_cells=request.max_cells,
            active_cell_indices=request.active_cell_indices,
        )
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.post("/scanpy/filter_cells")
def run_filter_cells(request: FilterCellsRequest, dataset: str | None = Query(None)):
    """Filter cells based on counts or number of genes expressed.

    Returns:
        Before/after cell counts
    """
    adaptor = get_adaptor(dataset)
    try:
        return adaptor.run_filter_cells(
            min_counts=request.min_counts,
            max_counts=request.max_counts,
            min_genes=request.min_genes,
            max_genes=request.max_genes,
            active_cell_indices=request.active_cell_indices,
        )
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.post("/scanpy/normalize_total")
def run_normalize_total(request: NormalizeTotalRequest, dataset: str | None = Query(None)):
    """Normalize total counts per cell.

    Returns:
        Operation status
    """
    adaptor = get_adaptor(dataset)
    try:
        return adaptor.run_normalize_total(target_sum=request.target_sum, active_cell_indices=request.active_cell_indices)
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


class Log1pRequest(BaseModel):
    active_cell_indices: list[int] | None = None


@router.post("/scanpy/log1p")
def run_log1p(request: Log1pRequest = Log1pRequest(), dataset: str | None = Query(None)):
    """Apply log1p transformation.

    Returns:
        Operation status
    """
    adaptor = get_adaptor(dataset)
    try:
        return adaptor.run_log1p(active_cell_indices=request.active_cell_indices)
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.post("/scanpy/highly_variable_genes")
def run_highly_variable_genes(request: HighlyVariableGenesRequest, dataset: str | None = Query(None)):
    """Identify highly variable genes.

    Adds 'highly_variable' boolean column to .var.

    Returns:
        Operation status and number of HVGs
    """
    adaptor = get_adaptor(dataset)
    try:
        return adaptor.run_highly_variable_genes(
            n_top_genes=request.n_top_genes,
            min_mean=request.min_mean,
            max_mean=request.max_mean,
            min_disp=request.min_disp,
            flavor=request.flavor,
            n_bins=request.n_bins,
            subset=request.subset,
            active_cell_indices=request.active_cell_indices,
        )
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.post("/scanpy/pca")
def run_pca(request: PcaRequest, dataset: str | None = Query(None)):
    """Run PCA dimensionality reduction.

    Args (via request):
        gene_subset: Gene filtering specification (see GenePcaRequest for format)

    Returns:
        Operation status and variance explained
    """
    adaptor = get_adaptor(dataset)
    try:
        # Convert Pydantic model to dict if needed
        gene_subset = request.gene_subset
        if isinstance(gene_subset, GeneSubsetSpec):
            gene_subset = {'columns': gene_subset.columns, 'operation': gene_subset.operation}

        return adaptor.run_pca(
            n_comps=request.n_comps,
            svd_solver=request.svd_solver,
            gene_subset=gene_subset,
            active_cell_indices=request.active_cell_indices,
        )
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.post("/scanpy/neighbors")
def run_neighbors(request: NeighborsRequest, dataset: str | None = Query(None)):
    """Compute neighborhood graph.

    Requires: PCA must be computed first.

    Returns:
        Operation status
    """
    adaptor = get_adaptor(dataset)
    try:
        return adaptor.run_neighbors(
            n_neighbors=request.n_neighbors,
            n_pcs=request.n_pcs,
            metric=request.metric,
            active_cell_indices=request.active_cell_indices,
        )
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/scanpy/umap")
def run_umap(request: UmapRequest, dataset: str | None = Query(None)):
    """Compute UMAP embedding.

    Requires: Neighbors must be computed first.

    Returns:
        Operation status and embedding name
    """
    adaptor = get_adaptor(dataset)
    try:
        return adaptor.run_umap(
            min_dist=request.min_dist,
            spread=request.spread,
            n_components=request.n_components,
            active_cell_indices=request.active_cell_indices,
        )
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/scanpy/leiden")
def run_leiden(request: LeidenRequest, dataset: str | None = Query(None)):
    """Run Leiden clustering.

    Requires: Neighbors must be computed first.

    Returns:
        Operation status and cluster info
    """
    adaptor = get_adaptor(dataset)
    try:
        return adaptor.run_leiden(
            resolution=request.resolution,
            key_added=request.key_added,
            active_cell_indices=request.active_cell_indices,
        )
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# =========================================================================
# Gene analysis endpoints
# =========================================================================


class GenePcaRequest(BaseModel):
    n_comps: int | None = None
    scale: bool = True
    use_kneedle: bool = True
    max_comps: int = 100
    # Gene subset can be:
    # - None: all genes
    # - str: single boolean column name (e.g., 'highly_variable', 'spatially_variable')
    # - list[str]: explicit list of gene names
    # - GeneSubsetSpec: combine multiple columns with AND/OR
    gene_subset: str | list[str] | GeneSubsetSpec | None = None
    active_cell_indices: list[int] | None = None


class GeneNeighborsRequest(BaseModel):
    n_neighbors: int = 15
    metric: str = 'euclidean'
    basis: str = 'gene_pca'
    gene_subset: str | list[str] | GeneSubsetSpec | None = None
    scale: bool = True
    active_cell_indices: list[int] | None = None


class FindSimilarGenesRequest(BaseModel):
    gene: str
    n_neighbors: int = 10
    use: str = 'connectivities'


class ClusterGenesRequest(BaseModel):
    resolution: float = 0.5
    key_added: str = 'gene_cluster'


class BuildGeneGraphRequest(BaseModel):
    n_pcs: int | None = None
    scale: bool = True
    use_kneedle: bool = True
    n_neighbors: int = 15
    metric: str = 'euclidean'
    active_cell_indices: list[int] | None = None
    gene_subset: str | list[str] | GeneSubsetSpec | None = None


@router.get("/var/boolean_columns")
def get_var_boolean_columns(dataset: str | None = Query(None)):
    """Get list of boolean columns in .var that can be used for gene filtering.

    Returns:
        List of columns with name, count of True values, and total genes
    """
    adaptor = get_adaptor(dataset)
    return adaptor.get_var_boolean_columns()


class SwapVarIndexRequest(BaseModel):
    column: str


@router.get("/var/identifier_columns")
def get_var_identifier_columns(dataset: str | None = Query(None)):
    """Get .var columns suitable as gene identifiers."""
    adaptor = get_adaptor(dataset)
    return adaptor.get_var_identifier_columns()


@router.post("/var/swap_index")
def swap_var_index(request: SwapVarIndexRequest, dataset: str | None = Query(None)):
    """Swap the .var index with another column.

    Returns:
        Updated schema after swap, plus old and new gene lists for remapping.
    """
    adaptor = get_adaptor(dataset)
    # Capture old gene names (positional order)
    old_genes = adaptor.get_gene_names()
    try:
        schema = adaptor.swap_var_index(request.column)
    except KeyError as e:
        raise HTTPException(status_code=400, detail=str(e))
    # Capture new gene names (same positional order)
    new_genes = adaptor.get_gene_names()
    return {
        'schema': schema,
        'old_genes': old_genes,
        'new_genes': new_genes,
    }


@router.post("/scanpy/gene_pca")
def run_gene_pca(request: GenePcaRequest, dataset: str | None = Query(None)):
    """Run PCA on genes (transposed expression matrix).

    Computes gene embeddings based on expression patterns.
    Results stored in .varm['X_gene_pca'].

    Args (via request):
        gene_subset: Gene filtering specification. Can be:
            - None: all genes
            - str: boolean column name (e.g., 'highly_variable')
            - list[str]: explicit gene names
            - {columns: [...], operation: 'intersection'|'union'}: combine columns

    Returns:
        Operation status, n_comps, variance explained, subset info
    """
    adaptor = get_adaptor(dataset)
    try:
        # Convert Pydantic model to dict if needed
        gene_subset = request.gene_subset
        if isinstance(gene_subset, GeneSubsetSpec):
            gene_subset = {'columns': gene_subset.columns, 'operation': gene_subset.operation}

        return adaptor.run_gene_pca(
            n_comps=request.n_comps,
            scale=request.scale,
            use_kneedle=request.use_kneedle,
            max_comps=request.max_comps,
            gene_subset=gene_subset,
            active_cell_indices=request.active_cell_indices,
        )
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.get("/scanpy/cell_pca_variance")
def get_cell_pca_variance(dataset: str | None = Query(None)):
    """Get cell PCA variance information for visualization.

    Returns:
        Variance ratios, cumulative variance, elbow point
    """
    adaptor = get_adaptor(dataset)
    try:
        return adaptor.get_cell_pca_variance()
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.get("/scanpy/gene_pca_variance")
def get_gene_pca_variance(dataset: str | None = Query(None)):
    """Get gene PCA variance information for visualization.

    Returns:
        Variance ratios, cumulative variance, elbow point
    """
    adaptor = get_adaptor(dataset)
    try:
        return adaptor.get_gene_pca_variance()
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.post("/scanpy/gene_neighbors", status_code=202)
def run_gene_neighbors(request: GeneNeighborsRequest, dataset: str | None = Query(None)):
    """Compute gene-gene kNN graph (cancellable background task)."""
    adaptor = get_adaptor(dataset)
    try:
        gene_subset = request.gene_subset
        if isinstance(gene_subset, GeneSubsetSpec):
            gene_subset = {'columns': gene_subset.columns, 'operation': gene_subset.operation}

        compute_fn, apply_fn = adaptor.prepare_gene_neighbors(
            n_neighbors=request.n_neighbors,
            metric=request.metric,
            basis=request.basis,
            gene_subset=gene_subset,
            scale=request.scale,
            active_cell_indices=request.active_cell_indices,
        )
        task_id = task_manager.submit(compute_fn, apply_fn)
        return {"task_id": task_id, "status": "running"}
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/scanpy/find_similar_genes")
def run_find_similar_genes(request: FindSimilarGenesRequest, dataset: str | None = Query(None)):
    """Find genes with similar expression patterns.

    Requires: gene_neighbors must be computed first.

    Returns:
        List of similar genes with scores
    """
    adaptor = get_adaptor(dataset)
    try:
        return adaptor.run_find_similar_genes(
            gene=request.gene,
            n_neighbors=request.n_neighbors,
            use=request.use,
        )
    except (ValueError, KeyError) as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.post("/scanpy/cluster_genes")
def run_cluster_genes(request: ClusterGenesRequest, dataset: str | None = Query(None)):
    """Cluster genes into co-expression modules using Leiden.

    Requires: gene_neighbors must be computed first.
    Results stored in .var[key_added] and .uns['gene_modules'].

    Returns:
        Cluster info and module composition
    """
    adaptor = get_adaptor(dataset)
    try:
        return adaptor.run_cluster_genes(
            resolution=request.resolution,
            key_added=request.key_added,
        )
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/scanpy/gene_modules")
def get_gene_modules(dataset: str | None = Query(None)):
    """Get gene modules from the last cluster_genes run.

    Returns:
        Dict with modules (module_name -> gene list)
    """
    adaptor = get_adaptor(dataset)
    try:
        return adaptor.get_gene_modules()
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.post("/scanpy/build_gene_graph")
def run_build_gene_graph(request: BuildGeneGraphRequest, dataset: str | None = Query(None)):
    """Convenience: run gene_pca and gene_neighbors in one step.

    Returns:
        Combined results from both steps
    """
    adaptor = get_adaptor(dataset)
    try:
        gene_subset = request.gene_subset
        if isinstance(gene_subset, GeneSubsetSpec):
            gene_subset = {'columns': gene_subset.columns, 'operation': gene_subset.operation}
        return adaptor.run_build_gene_graph(
            n_pcs=request.n_pcs,
            scale=request.scale,
            use_kneedle=request.use_kneedle,
            n_neighbors=request.n_neighbors,
            metric=request.metric,
            active_cell_indices=request.active_cell_indices,
            gene_subset=gene_subset,
        )
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


# =========================================================================
# Spatial Analysis endpoints
# =========================================================================

class SpatialNeighborsRequest(BaseModel):
    n_neighs: int = 6
    coord_type: str | None = None
    spatial_key: str | None = None
    delaunay: bool = False
    n_rings: int = 1
    radius: float | None = None


class SpatialAutocorrRequest(BaseModel):
    mode: str = 'moran'
    genes: list[str] | None = None
    n_perms: int | None = 100
    n_jobs: int = 1
    corr_method: str = 'fdr_bh'
    pval_threshold: float = 0.05
    gene_subset: str | None = None


class GetSpatiallyVariableGenesRequest(BaseModel):
    top_n: int | None = None
    pval_threshold: float | None = None


class ContourizeRequest(BaseModel):
    genes: list[str]
    contour_levels: int = 6
    log_transform: bool = True
    smooth_sigma: float = 2.0
    grid_res: int = 200
    annotation_key: str | None = None


@router.get("/scanpy/has_spatial")
def check_has_spatial(dataset: str | None = Query(None)):
    """Check if spatial coordinates are available.

    Returns:
        Dict with has_spatial (bool) and spatial_key if found
    """
    adaptor = get_adaptor(dataset)
    has_spatial = adaptor._has_spatial_coordinates()
    spatial_key = adaptor._get_spatial_key() if has_spatial else None
    return {
        'has_spatial': has_spatial,
        'spatial_key': spatial_key,
    }


@router.post("/scanpy/spatial_neighbors", status_code=202)
def run_spatial_neighbors(request: SpatialNeighborsRequest, dataset: str | None = Query(None)):
    """Compute spatial neighborhood graph (cancellable background task)."""
    adaptor = get_adaptor(dataset)
    try:
        compute_fn, apply_fn = adaptor.prepare_spatial_neighbors(
            n_neighs=request.n_neighs,
            coord_type=request.coord_type,
            spatial_key=request.spatial_key,
            delaunay=request.delaunay,
            n_rings=request.n_rings,
            radius=request.radius,
        )
        task_id = task_manager.submit(compute_fn, apply_fn)
        return {"task_id": task_id, "status": "running"}
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/scanpy/spatial_autocorr", status_code=202)
def run_spatial_autocorr(request: SpatialAutocorrRequest, dataset: str | None = Query(None)):
    """Compute spatial autocorrelation (cancellable background task).

    Requires: spatial_neighbors must be computed first.

    Returns:
        Task ID and status for polling
    """
    adaptor = get_adaptor(dataset)
    try:
        compute_fn, apply_fn = adaptor.prepare_spatial_autocorr(
            mode=request.mode,
            genes=request.genes,
            n_perms=request.n_perms,
            n_jobs=request.n_jobs,
            corr_method=request.corr_method,
            pval_threshold=request.pval_threshold,
            gene_subset=request.gene_subset,
        )
        task_id = task_manager.submit(compute_fn, apply_fn)
        return {"task_id": task_id, "status": "running"}
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/scanpy/spatially_variable_genes")
def get_spatially_variable_genes(request: GetSpatiallyVariableGenesRequest, dataset: str | None = Query(None)):
    """Get list of spatially variable genes.

    Requires: spatial_autocorr must be computed first.

    Returns:
        List of genes with statistics
    """
    adaptor = get_adaptor(dataset)
    try:
        return adaptor.get_spatially_variable_genes(
            top_n=request.top_n,
            pval_threshold=request.pval_threshold,
        )
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.post("/scanpy/contourize")
def run_contourize(request: ContourizeRequest, dataset: str | None = Query(None)):
    """Compute spatial expression contours from a gene set.

    Requires: spatial coordinates in .obsm

    Returns:
        Operation status, annotation key, contour info
    """
    adaptor = get_adaptor(dataset)
    try:
        return adaptor.run_contourize(
            genes=request.genes,
            contour_levels=request.contour_levels,
            log_transform=request.log_transform,
            smooth_sigma=request.smooth_sigma,
            grid_res=request.grid_res,
            annotation_key=request.annotation_key,
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
def export_h5ad(dataset: str | None = Query(None)):
    """Export the current AnnData object as an h5ad file.

    This includes:
    - Any new annotation columns that were created
    - Drawn lines stored in .uns['xcell_lines']
    - Cell projections onto lines in .obsm['X_{line_name}_projection']

    Returns:
        The h5ad file as a download
    """
    adaptor = get_adaptor(dataset)
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


# --- Background task management ---

@router.get("/tasks/{task_id}")
def get_task_status(task_id: str):
    """Poll the status of a background task."""
    entry = task_manager.get_status(task_id)
    if entry is None:
        raise HTTPException(status_code=404, detail="Task not found")
    response: dict[str, Any] = {
        "task_id": entry.id,
        "status": entry.status,
    }
    if entry.result is not None:
        response["result"] = entry.result
    if entry.error is not None:
        response["error"] = entry.error
    return response


@router.post("/tasks/{task_id}/cancel")
def cancel_task(task_id: str):
    """Cancel a running background task."""
    entry = task_manager.get_status(task_id)
    if entry is None:
        raise HTTPException(status_code=404, detail="Task not found")
    task_manager.cancel(task_id)
    return {"task_id": task_id, "status": "cancelling"}
