"""API routes for XCell."""

from pathlib import Path
import shutil
import subprocess
import tempfile
from typing import Any

import numpy as np

from fastapi import APIRouter, HTTPException, Query
from fastapi.responses import PlainTextResponse
from pydantic import BaseModel

from xcell.adaptor import DataAdaptor, combine_datasets, describe_combine_columns
from xcell.task_manager import task_manager
from xcell import config as user_config
from xcell import gene_set_store
from xcell import gene_set_library

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


def _resolve_cell_context(
    adaptor,
    context: str,
    indices: list[int] | None,
    annotation_column: str | None,
    annotation_values: list[str] | None,
) -> list[int] | None:
    """Translate a cluster_gene_set cell_context request into a concrete
    list of cell indices (or None for "all cells")."""
    if context == 'all':
        return None
    if context == 'selection':
        if not indices:
            raise HTTPException(
                status_code=400,
                detail="cell_context='selection' requires non-empty cell_indices",
            )
        return indices
    if context == 'annotation':
        if not annotation_column or not annotation_values:
            raise HTTPException(
                status_code=400,
                detail="cell_context='annotation' requires annotation_column and annotation_values",
            )
        col = adaptor.adata.obs[annotation_column]
        mask = col.isin(annotation_values)
        import numpy as np
        resolved = np.flatnonzero(mask.values).tolist()
        if not resolved:
            raise HTTPException(
                status_code=400,
                detail=f"No cells matched {annotation_column}={annotation_values}",
            )
        return resolved
    raise HTTPException(status_code=400, detail=f"Unknown cell_context: {context}")


#: Which file extensions each browse `kind` surfaces. Directories are always
#: listed so navigation works the same way whatever you are looking for.
BROWSE_KINDS = {
    'data': ('.h5ad', '.h5', '.rds'),
    'classifier': ('.pkl', '.pickle'),
    # Export targets. Each kind remembers its own directory, so where you keep
    # exported tables is not forced to be where you keep h5ads.
    'tabular': ('.tsv', '.csv', '.txt'),
    'geneset': ('.json', '.gmt'),
}


@router.get("/browse")
def browse_filesystem(path: str | None = None, kind: str = 'data'):
    """List directories and files of interest at the given path.

    Args:
        path: Directory path to list. Defaults to the user's home directory.
        kind: Which files to surface — 'data' (default) for loadable datasets,
            including 10x matrix folders and file trios, or 'classifier' for
            pickled PySingleCellNet classifiers. Everything else is hidden;
            listing a home directory's worth of unrelated files would bury
            the handful that matter.

    Returns:
        JSON object with current path, parent path, and entries (dirs + files).
    """
    if kind not in BROWSE_KINDS:
        raise HTTPException(
            status_code=400,
            detail=f"Unknown kind '{kind}'. Expected one of: {', '.join(sorted(BROWSE_KINDS))}.",
        )
    suffixes = BROWSE_KINDS[kind]

    if path:
        directory = Path(path).expanduser().resolve()
    else:
        directory = Path.home()

    if not directory.is_dir():
        raise HTTPException(status_code=400, detail=f"Not a directory: {directory}")

    parent = str(directory.parent) if directory != directory.parent else None

    entries = []
    try:
        import re
        # 10x matrix folders and file trios are datasets; irrelevant otherwise.
        detect_10x = kind == 'data'
        # Collect all filenames for trio detection
        all_names = {item.name for item in directory.iterdir() if item.is_file()}
        # Track prefixes that form complete trios so we don't also list them as raw files
        trio_matrix_names: set[str] = set()
        for name in all_names:
            m = re.match(r'^(.+)_matrix\.mtx(\.gz)?$', name)
            if not m:
                continue
            prefix = m.group(1)
            has_bar = any(f'{prefix}_barcodes{ext}' in all_names for ext in ('.tsv.gz', '.tsv'))
            has_feat = any(f'{prefix}_{f}{ext}' in all_names for f in ('features', 'genes') for ext in ('.tsv.gz', '.tsv'))
            if has_bar and has_feat:
                trio_matrix_names.add(name)

        for item in sorted(directory.iterdir(), key=lambda p: (not p.is_dir(), p.name.lower())):
            if item.name.startswith('.'):
                continue
            if item.is_dir():
                # Check if this is a 10x CellRanger matrix folder. A child dir we
                # can't read (e.g. macOS TCC-protected ~/Documents) must not abort
                # the whole listing — treat it as a plain directory instead.
                try:
                    children = {c.name for c in item.iterdir()}
                except (PermissionError, OSError):
                    children = set()
                has_matrix = bool(children & {'matrix.mtx', 'matrix.mtx.gz'})
                has_barcodes = bool(children & {'barcodes.tsv', 'barcodes.tsv.gz'})
                has_features = bool(children & {'features.tsv', 'features.tsv.gz', 'genes.tsv', 'genes.tsv.gz'})
                if detect_10x and has_matrix and has_barcodes and has_features:
                    entries.append({"name": item.name, "type": "10x_mtx", "path": str(item)})
                else:
                    entries.append({"name": item.name, "type": "directory", "path": str(item)})
            elif detect_10x and item.name in trio_matrix_names:
                # Prefixed 10x file trio (e.g. GSM1234_matrix.mtx.gz with companions)
                prefix = re.match(r'^(.+)_matrix\.mtx', item.name).group(1)
                entries.append({"name": prefix, "type": "10x_mtx_trio", "path": str(item)})
            elif item.suffix in suffixes:
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

    Supports .h5ad, .h5, .rds (Seurat), 10x CellRanger matrix directories,
    and prefixed 10x file trios (e.g. GSM1234_matrix.mtx.gz with companion
    barcodes and features files, common in GEO accessions).

    Args:
        file_path: Absolute path to a data file, 10x matrix directory, or
                   prefixed *_matrix.mtx(.gz) file
        slot: Named slot to load into (default: 'primary')

    Returns:
        The schema of the newly loaded dataset plus the slot name
    """
    target_slot = request.slot
    path = Path(request.file_path)
    if not path.exists():
        raise HTTPException(status_code=404, detail=f"File not found: {request.file_path}")
    if path.is_dir():
        # Check for 10x CellRanger matrix folder
        children = {c.name for c in path.iterdir()}
        has_matrix = bool(children & {'matrix.mtx', 'matrix.mtx.gz'})
        has_barcodes = bool(children & {'barcodes.tsv', 'barcodes.tsv.gz'})
        has_features = bool(children & {'features.tsv', 'features.tsv.gz', 'genes.tsv', 'genes.tsv.gz'})
        if not (has_matrix and has_barcodes and has_features):
            raise HTTPException(status_code=400, detail="Directory is not a valid 10x CellRanger matrix folder")
    elif DataAdaptor._find_10x_trio_files(path) is not None:
        pass  # Valid prefixed 10x file trio — DataAdaptor handles loading
    elif path.suffix not in ('.h5ad', '.h5', '.rds'):
        raise HTTPException(status_code=400, detail="File must have .h5ad, .h5, or .rds extension")
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


class CombineSpatialFile(BaseModel):
    file_path: str
    label: str | None = None


class CombineSpatialRequest(BaseModel):
    files: list[CombineSpatialFile]
    slot: str = "primary"
    gap_fraction: float = 0.05
    #: Per-column handling; see combine_datasets. Omitted columns take the
    #: suggestion /combine/columns reports.
    obs_policy: dict[str, str] | None = None
    var_policy: dict[str, str] | None = None


class CombineColumnsRequest(BaseModel):
    files: list[CombineSpatialFile]


def _dataset_label(path: Path) -> str:
    """Default `sample` label for a combine input: the dataset's bare name.

    A 10x matrix folder is named by the folder; a prefixed trio by its
    prefix (GSM1234_matrix.mtx.gz -> GSM1234); a file by its stem.
    """
    if path.is_dir():
        return path.name
    import re
    m = re.match(r'^(.+)_matrix\.mtx(\.gz)?$', path.name)
    if m:
        return m.group(1)
    return path.stem


def _resolve_combine_inputs(files) -> tuple[list[Path], list[str]]:
    """Validate combine inputs and return (load paths, sample labels).

    Shared by /combine and /combine/columns so the column picker can never
    accept a set of files the combine itself would reject.
    """
    if len(files) < 2:
        raise HTTPException(status_code=400, detail="At least 2 files required to combine")
    paths: list[Path] = []
    labels: list[str] = []
    for entry in files:
        p = Path(entry.file_path)
        if not p.exists():
            raise HTTPException(status_code=404, detail=f"File not found: {entry.file_path}")
        # Same accepted kinds as /api/load, and the same .rds conversion.
        if p.is_dir():
            children = {c.name for c in p.iterdir()}
            has_matrix = bool(children & {'matrix.mtx', 'matrix.mtx.gz'})
            has_barcodes = bool(children & {'barcodes.tsv', 'barcodes.tsv.gz'})
            has_features = bool(children & {'features.tsv', 'features.tsv.gz', 'genes.tsv', 'genes.tsv.gz'})
            if not (has_matrix and has_barcodes and has_features):
                raise HTTPException(
                    status_code=400,
                    detail=f"{p.name} is not a valid 10x CellRanger matrix folder")
        elif DataAdaptor._find_10x_trio_files(p) is not None:
            pass  # Valid prefixed 10x file trio — the loader handles it
        elif p.suffix not in ('.h5ad', '.h5', '.rds'):
            raise HTTPException(
                status_code=400,
                detail=f"Cannot combine {p.name}: expected .h5ad, .h5, .rds, "
                       "a 10x matrix folder, or a *_matrix.mtx trio")
        load_path = _convert_rds_to_h5ad(p) if p.suffix == '.rds' else p
        paths.append(load_path)
        labels.append((entry.label or _dataset_label(p)).strip() or _dataset_label(p))

    return paths, labels


@router.post("/combine/columns")
def combine_columns(request: CombineColumnsRequest):
    """Every .obs / .var column across the inputs, with a suggested policy.

    Called before /combine so the column choices are made with the collision
    map in hand — which datasets carry each name, whether their .var values
    already agree, and why the suggestion is what it is. Reads annotations
    only; an h5ad is opened backed rather than loaded.
    """
    paths, labels = _resolve_combine_inputs(request.files)
    try:
        return describe_combine_columns(paths, labels)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Could not read columns: {e}")


@router.post("/combine")
def combine(request: CombineSpatialRequest):
    """Combine multiple datasets into one adata and load it into a slot.

    Inputs may be any format File -> Load accepts: .h5ad, 10x .h5 (including
    Visium HD feature_slice.h5), .rds (Seurat, converted via R), 10x
    CellRanger matrix directories, and prefixed *_matrix.mtx(.gz) trios.

    When every input has spatial coordinates, sections are laid out
    left-to-right along the x-axis with a small gap (``mode: "spatial"``).
    Otherwise rows are concatenated with no geometry invented, keeping the
    .obsm arrays every input shares (``mode: "concat"``). Either way: gene
    index = intersection across inputs; a new ``sample`` categorical .obs
    column tags each cell with its source file label.

    Args:
        files: List of {file_path, label?} entries. >=2 existing datasets.
        slot: Named slot to load the combined adata into.
        gap_fraction: Gap between adjacent sections as a fraction of mean
                      section width (default 0.05 = 5%). Spatial mode only.
    """
    paths, labels = _resolve_combine_inputs(request.files)

    try:
        combined = combine_datasets(
            paths, labels, gap_fraction=request.gap_fraction,
            obs_policy=request.obs_policy, var_policy=request.var_policy,
        )
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Combine failed: {e}")

    mode = combined.uns['xcell_combine']['mode']
    # Wrap the in-memory adata in a DataAdaptor and store under the slot.
    # The "filepath" is synthetic — used only for display in the header bar.
    virtual_path = Path(f"combined_{len(paths)}_datasets.h5ad")
    adaptor = DataAdaptor(virtual_path, adata=combined)
    adaptor.filepath = virtual_path
    set_adaptor(adaptor, slot=request.slot)
    return {
        "slot": request.slot,
        "n_sections": len(paths),
        "labels": labels,
        "mode": mode,
        **adaptor.get_schema(),
    }


@router.post("/combine_spatial")
def combine_spatial(request: CombineSpatialRequest):
    """Deprecated alias for POST /combine, kept for old clients."""
    return combine(request)


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
def get_embedding(
    name: str,
    dim_x: int = Query(0),
    dim_y: int = Query(1),
    dim_z: int | None = Query(None),
    dataset: str | None = Query(None),
):
    """Get embedding coordinates by name, viewing two (or three, via dim_z) .obsm columns.

    dim_x / dim_y (default 0, 1) pick which columns of a >2-dimensional matrix
    (PCA, gene-set scores) are shown as x / y. dim_z is optional; when provided
    the response additionally includes a "z" array and "dim_z".
    """
    adaptor = get_adaptor(dataset)
    try:
        return adaptor.get_embedding(name, dim_x=dim_x, dim_y=dim_y, dim_z=dim_z)
    except KeyError as e:
        raise HTTPException(status_code=404, detail=str(e))


class TransformEmbeddingRequest(BaseModel):
    rotation_degrees: float = 0
    reflect_x: bool = False
    reflect_y: bool = False
    cell_indices: list[int] | None = None
    translate_x: float = 0.0
    translate_y: float = 0.0
    dim_x: int = 0
    dim_y: int = 1


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
            dim_x=request.dim_x,
            dim_y=request.dim_y,
        )
    except KeyError as e:
        raise HTTPException(status_code=404, detail=str(e))


@router.post("/embedding/{name}/undo")
def undo_transform_embedding(
    name: str,
    dim_x: int = Query(0),
    dim_y: int = Query(1),
    dataset: str | None = Query(None),
):
    """Undo the last transform for an embedding (returns the requested dims view)."""
    adaptor = get_adaptor(dataset)
    try:
        return adaptor.undo_transform_embedding(name, dim_x=dim_x, dim_y=dim_y)
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


@router.get("/obs/crosstab")
def get_obs_crosstab(
    a: str = Query(...), b: str = Query(...), dataset: str | None = Query(None),
):
    """Count cells by two .obs columns at once, for the stacked barplot.

    Args:
        a: column whose categories become the bars
        b: column whose categories split each bar

    Returns:
        counts as rows of a_categories by columns of b_categories, the category
        lists, and each column's scanpy colors where the dataset carries them
    """
    adaptor = get_adaptor(dataset)
    try:
        return adaptor.crosstab(a, b)
    except KeyError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))


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


class RenameObsLabelRequest(BaseModel):
    old_label: str
    new_label: str


@router.post("/obs/{column}/rename_label")
def rename_obs_label(
    column: str,
    request: RenameObsLabelRequest,
    dataset: str | None = Query(None),
):
    """Rename a single category value in a categorical or string .obs column."""
    adaptor = get_adaptor(dataset)
    try:
        return adaptor.rename_obs_label(column, request.old_label, request.new_label)
    except KeyError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except ValueError as e:
        msg = str(e)
        # Collisions get a 409 so the frontend can offer "merge instead".
        if "already exists" in msg:
            raise HTTPException(status_code=409, detail=msg)
        raise HTTPException(status_code=400, detail=msg)


class MergeObsLabelsRequest(BaseModel):
    labels: list[str]
    new_label: str


@router.post("/obs/{column}/merge_labels")
def merge_obs_labels(
    column: str,
    request: MergeObsLabelsRequest,
    dataset: str | None = Query(None),
):
    """Merge two or more category values in a categorical or string .obs column."""
    adaptor = get_adaptor(dataset)
    try:
        return adaptor.merge_obs_labels(column, request.labels, request.new_label)
    except KeyError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))


class TransferObsLabelsRequest(BaseModel):
    target_column: str
    source_column: str
    out_column: str
    rename_mode: str = "replace"
    sep: str = "."
    prefix: str = ""
    unassigned_values: list[str] | None = None


@router.post("/obs/transfer_labels")
def transfer_obs_labels(
    request: TransferObsLabelsRequest,
    dataset: str | None = Query(None),
):
    """Fold labels from a (partial) source column into a parent target column."""
    adaptor = get_adaptor(dataset)
    try:
        return adaptor.transfer_obs_labels(
            target_column=request.target_column,
            source_column=request.source_column,
            out_column=request.out_column,
            rename_mode=request.rename_mode,
            sep=request.sep,
            prefix=request.prefix,
            unassigned_values=request.unassigned_values,
        )
    except KeyError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.get("/gene_sets")
def get_gene_sets_state():
    """Return the currently-persisted gene-set state (opaque JSON blob).

    The shape matches the frontend's ``geneSetCategories`` dict. The server
    doesn't inspect the contents — it just round-trips them so reloads in
    the browser can restore the user's sets. Empty dict means nothing has
    been saved this server lifetime.
    """
    return {"gene_sets": gene_set_store.get_gene_sets()}


class GeneSetsPutRequest(BaseModel):
    gene_sets: dict[str, Any]


@router.put("/gene_sets")
def put_gene_sets_state(request: GeneSetsPutRequest):
    """Replace the persisted gene-set state. Called by the frontend on any
    mutation (debounced). Server-lifetime storage — cleared on restart."""
    try:
        gene_set_store.set_gene_sets(request.gene_sets)
        return {"status": "ok"}
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.get("/gene_sets/library")
def get_gene_sets_library():
    """List the curated gene-set bundles shipped with xcell (read-only).

    Files live in ``xcell/data/gene_sets/*.json``. The frontend shows these in
    the Import dialog; the user loads a bundle into their own editable sets on
    demand — this endpoint never mutates the user's gene-set state.
    """
    return {"bundles": gene_set_library.list_bundles()}


@router.get("/config/defaults")
def get_config_defaults():
    """Return the raw user-config dict (loaded from ~/.xcell/config.* at startup).

    The frontend merges these values over its hardcoded defaults when
    initializing modal param forms. Restarting the backend picks up edits.
    """
    return {
        "config": user_config.get_user_config(),
        "meta": user_config.get_config_meta(),
    }


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
    all_genes = sorted(adaptor.get_visible_gene_names(), key=str.lower)
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
def get_expression(
    gene: str,
    transform: str | None = None,
    clip_percentile: float = 0.0,
    layer: str | None = None,
    dataset: str | None = Query(None),
):
    """Get expression values for a single gene.

    Args:
        gene: Gene name
        transform: Optional transformation to apply. Supported values:
            - "log1p": Apply normalize_total followed by log1p transformation
        clip_percentile: Optional symmetric percentile clip (0 = off).

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
        return adaptor.get_expression(
            gene,
            transform=transform,
            clip_percentile=clip_percentile,
            layer=layer,
        )
    except KeyError as e:
        raise HTTPException(status_code=404, detail=str(e))


class MultiExpressionRequest(BaseModel):
    """Request model for multi-gene expression aggregation.

    The pipeline is: source → per-gene normalize → aggregate across genes
    → optional symmetric percentile clip on per-cell scores.
    """
    genes: list[str]
    transform: str | None = None
    # 'none' | 'zscore_mad' | 'zscore_sd' | 'minmax' | 'rank'
    per_gene_norm: str = 'zscore_mad'
    per_gene_clip: float = 0.0  # used by 'minmax' per-gene norm
    # 'mean' | 'median' | 'sum' | 'max'
    aggregation: str = 'mean'
    clip_percentile: float = 1.0
    # Optional layer name in adata.layers; None / 'X' / omitted → adata.X.
    # Overrides `transform` when set (the layer is treated as authoritative).
    layer: str | None = None


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
            per_gene_norm=request.per_gene_norm,
            per_gene_clip=request.per_gene_clip,
            aggregation=request.aggregation,
            clip_percentile=request.clip_percentile,
            layer=request.layer,
        )
    except KeyError as e:
        raise HTTPException(status_code=404, detail=str(e))


class UcellExpressionRequest(BaseModel):
    up: list[str]
    down: list[str] = []
    layer: str = 'counts'
    max_rank: int = 1500
    w_neg: float = 1.0


@router.post("/expression/ucell")
def expression_ucell(request: UcellExpressionRequest, dataset: str | None = Query(None)):
    """Non-persisted per-cell UCell score for one directional set (interactive coloring)."""
    adaptor = get_adaptor(dataset)
    try:
        return adaptor.ucell_score_values(
            up=request.up, down=request.down, layer=request.layer,
            max_rank=request.max_rank, w_neg=request.w_neg,
        )
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


class BivariateExpressionRequest(BaseModel):
    """Request model for bivariate gene expression.

    Same per-gene-norm + aggregation pipeline as ``MultiExpressionRequest``,
    applied independently per axis.
    """
    genes1: list[str]
    genes2: list[str]
    transform: str | None = None
    per_gene_norm: str = 'zscore_mad'
    per_gene_clip: float = 0.0
    aggregation: str = 'mean'
    clip_percentile: float = 1.0
    layer: str | None = None


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
            per_gene_norm=request.per_gene_norm,
            per_gene_clip=request.per_gene_clip,
            aggregation=request.aggregation,
            clip_percentile=request.clip_percentile,
            layer=request.layer,
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


class ClusterGeneSetRequest(BaseModel):
    gene_names: list[str]
    method: str  # 'hierarchical' | 'kmeans' | 'dbscan'
    k: int | None = None  # required for hierarchical/kmeans, ignored for dbscan
    cell_context: str  # 'all' | 'selection' | 'annotation'
    cell_indices: list[int] | None = None
    annotation_column: str | None = None
    annotation_values: list[str] | None = None
    # DBSCAN-only knobs (ignored for other methods).
    eps: float = 0.3
    min_samples: int = 3
    # Optional layer name. None / 'X' / omitted → adata.X (passes through the
    # lazy normalize_total + log1p snapshot, the historical default). When
    # set to a layer name, that layer is read directly without renormalization
    # — pass the output of run_smooth here to cluster on smoothed expression.
    layer: str | None = None
    # When True, restrict the clustered genes to those visible under the active
    # .var gene mask (no-op if no mask is active).
    use_gene_mask: bool = False
    # method='auto' (co-expression modules) knobs; ignored by other methods.
    metric: str = "bicor"            # 'bicor' | 'pearson' | 'spearman'
    min_genes: int = 5               # min genes per surviving module
    merge_threshold: float = 0.8     # eigengene-corr above which modules merge
    purity_threshold: float = 0.5    # eigengene PVE below which a module splits
    max_split_depth: int = 2         # recursion cap on splitting
    min_module_corr: float = 0.2     # min best-partner corr to join a module
                                     # (below -> grey/unassigned)


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
    # Which two .obsm columns of `embeddingName` the line was drawn against, so
    # projection/association use the same axes the user saw (default first two).
    dimX: int = 0
    dimY: int = 1


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

    coords = adaptor._line_view_coords(line)

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
    cluster_genes: bool = False


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
    all_genes: list[LineAssociationGene] = []
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


@router.post("/lines/association", status_code=202)
def test_line_association(request: LineAssociationRequest, dataset: str | None = Query(None)):
    """Test genes for association with a line (cancellable background task)."""
    adaptor = get_adaptor(dataset)
    try:
        compute_fn, apply_fn = adaptor.prepare_line_association(
            line_name=request.line_name,
            cell_indices=request.cell_indices,
            gene_subset=request.gene_subset,
            test_variable=request.test_variable,
            n_spline_knots=request.n_spline_knots,
            min_cells=request.min_cells,
            fdr_threshold=request.fdr_threshold,
            top_n=request.top_n,
            cluster_genes=request.cluster_genes,
        )
        task_id = task_manager.submit(compute_fn, apply_fn)
        return {"task_id": task_id, "status": "running"}
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
    cluster_genes: bool = False


@router.post("/lines/multi-association", status_code=202)
def test_multi_line_association(request: MultiLineAssociationRequest, dataset: str | None = Query(None)):
    """Test genes for association across multiple lines (cancellable background task)."""
    adaptor = get_adaptor(dataset)
    try:
        compute_fn, apply_fn = adaptor.prepare_multi_line_association(
            lines=[{'name': e.name, 'cell_indices': e.cell_indices, 'reversed': e.reversed} for e in request.lines],
            gene_subset=request.gene_subset,
            test_variable=request.test_variable,
            n_spline_knots=request.n_spline_knots,
            min_cells=request.min_cells,
            fdr_threshold=request.fdr_threshold,
            top_n=request.top_n,
            cluster_genes=request.cluster_genes,
        )
        task_id = task_manager.submit(compute_fn, apply_fn)
        return {"task_id": task_id, "status": "running"}
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
    #: Detect HVGs within each group of this .obs column instead of pooled.
    split_by: str | None = None
    add_union: bool = False
    add_intersection: bool = False
    min_cells_per_group: int = 10


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
    use_rep: str | None = None
    active_cell_indices: list[int] | None = None


class CombineNeighborsSource(BaseModel):
    key: str
    weight: float = 1.0


class CombineNeighborsRequest(BaseModel):
    sources: list[CombineNeighborsSource]
    target_key: str = 'connectivities'


class CreatePcaSubsetRequest(BaseModel):
    drop_pc_indices: list[int]
    suffix: str | None = None


class UmapRequest(BaseModel):
    min_dist: float = 0.5
    spread: float = 1.0
    n_components: int = 2
    graph_key: str | None = None
    key_added: str | None = None
    active_cell_indices: list[int] | None = None


class LeidenRequest(BaseModel):
    resolution: float = 1.0
    key_added: str = 'leiden'
    graph_key: str | None = None
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


@router.get("/scanpy/filter_cells/qc")
def filter_cells_qc(dataset: str | None = Query(None)):
    """Per-cell counts/genes distributions, for threshold histograms.

    Returns:
        {'counts': [...], 'genes': [...]} — one value per cell, from .X
    """
    adaptor = get_adaptor(dataset)
    try:
        return adaptor.filter_cells_qc()
    except HTTPException:
        raise
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

    Adds a 'highly_variable' boolean column to .var. With ``split_by``, the
    detection runs *within* each group of that .obs column instead, writing
    ``highly_variable__<group>`` plus an optional union / intersection — and
    leaving the pooled ``highly_variable`` column untouched.

    Returns:
        Operation status and number of HVGs; a split run also returns the
        per-group counts, every column written, and any groups skipped.
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
            # The generic Scanpy panel always sends the field; empty means
            # "no split", not a column whose name is the empty string.
            split_by=request.split_by or None,
            add_union=request.add_union,
            add_intersection=request.add_intersection,
            min_cells_per_group=request.min_cells_per_group,
        )
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


class EmbeddingFromObsRequest(BaseModel):
    col_x: str
    col_y: str
    log_axes: str = "none"
    name: str | None = None


@router.post("/scanpy/embedding_from_obs")
def scanpy_embedding_from_obs(
    request: EmbeddingFromObsRequest, dataset: str | None = Query(None)
):
    """Build a 2-D embedding from two numeric .obs columns."""
    adaptor = get_adaptor(dataset)
    try:
        return adaptor.create_obs_embedding(
            col_x=request.col_x, col_y=request.col_y,
            log_axes=request.log_axes, name=request.name,
        )
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))


class ScoreGeneSetsMatrixRequest(BaseModel):
    sets: list[dict[str, Any]]  # [{name, genes:[...]}]
    per_gene_norm: str = "zscore_mad"
    per_gene_clip: float = 0.0
    aggregation: str = "mean"
    obsm_name: str = "geneset_scores"
    layer: str | None = None
    transform: str | None = None
    overwrite: bool = False


@router.post("/scanpy/score_gene_sets_matrix")
def scanpy_score_gene_sets_matrix(
    request: ScoreGeneSetsMatrixRequest, dataset: str | None = Query(None)
):
    """Score every gene set in a folder into one .obsm matrix (mean pipeline)."""
    adaptor = get_adaptor(dataset)
    try:
        return adaptor.score_gene_sets_matrix(
            sets=request.sets,
            per_gene_norm=request.per_gene_norm,
            per_gene_clip=request.per_gene_clip,
            aggregation=request.aggregation,
            obsm_name=request.obsm_name,
            layer=request.layer,
            transform=request.transform,
            overwrite=request.overwrite,
        )
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))


class ObsmColumnRequest(BaseModel):
    obsm_name: str
    column: str


@router.post("/obsm/column")
def obsm_column(request: ObsmColumnRequest, dataset: str | None = Query(None)):
    """Per-cell values of a named column of an .obsm matrix (for coloring by score)."""
    adaptor = get_adaptor(dataset)
    try:
        return adaptor.get_obsm_column(request.obsm_name, request.column)
    except KeyError as e:
        raise HTTPException(status_code=404, detail=str(e))



class SumCountsRequest(BaseModel):
    pattern: str
    match_mode: str = "prefix"
    obs_name: str | None = None
    layer: str = "counts"


@router.post("/scanpy/sum_counts_by_pattern")
def scanpy_sum_counts_by_pattern(
    request: SumCountsRequest, dataset: str | None = Query(None)
):
    """Sum counts of genes matching a prefix/regex into a new .obs column."""
    adaptor = get_adaptor(dataset)
    try:
        return adaptor.sum_counts_by_pattern(
            pattern=request.pattern, match_mode=request.match_mode,
            obs_name=request.obs_name, layer=request.layer,
        )
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))


class AssignSpeciesRequest(BaseModel):
    count_columns: str  # comma-separated obs column names
    labels: str | None = None  # comma-separated, optional
    obs_name: str = "species"
    threshold: float = 0.9


@router.post("/scanpy/assign_species")
def scanpy_assign_species(
    request: AssignSpeciesRequest, dataset: str | None = Query(None)
):
    """Assign each cell a species from per-species count columns."""
    adaptor = get_adaptor(dataset)
    try:
        return adaptor.assign_species(
            count_columns=request.count_columns, labels=request.labels,
            obs_name=request.obs_name, threshold=request.threshold,
        )
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))


class DetectSpeciesPrefixesRequest(BaseModel):
    min_fraction: float = 0.01


@router.post("/scanpy/detect_species_prefixes")
def scanpy_detect_species_prefixes(
    request: DetectSpeciesPrefixesRequest, dataset: str | None = Query(None)
):
    """Report CellRanger genome prefixes on gene names. Read-only preview."""
    adaptor = get_adaptor(dataset)
    try:
        return adaptor.detect_species_prefixes(min_fraction=request.min_fraction)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))


class AddVarSpeciesColumnRequest(BaseModel):
    species_column: str = "species"
    prefixes: str | None = None  # comma-separated, optional (auto-detect)
    labels: str | None = None  # comma-separated, optional
    min_fraction: float = 0.01
    unknown_label: str = "unknown"
    overwrite: bool = False


@router.post("/scanpy/add_var_species_column")
def scanpy_add_var_species_column(
    request: AddVarSpeciesColumnRequest, dataset: str | None = Query(None)
):
    """Add a species .var column derived from genome prefixes on gene names."""
    adaptor = get_adaptor(dataset)
    try:
        return adaptor.add_var_species_column(
            species_column=request.species_column, prefixes=request.prefixes,
            labels=request.labels, min_fraction=request.min_fraction,
            unknown_label=request.unknown_label, overwrite=request.overwrite,
        )
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))


class RenameGenesRequest(BaseModel):
    pattern: str
    replacement: str = ""
    match_mode: str = "regex"
    make_unique: bool = False


@router.post("/scanpy/rename_genes")
def scanpy_rename_genes(
    request: RenameGenesRequest, dataset: str | None = Query(None)
):
    """Find/replace across gene symbols; an empty replacement strips the match."""
    adaptor = get_adaptor(dataset)
    try:
        return adaptor.rename_genes(
            pattern=request.pattern, replacement=request.replacement,
            match_mode=request.match_mode, make_unique=request.make_unique,
        )
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))


class SumCountsBySpeciesRequest(BaseModel):
    species_column: str = "species"
    layer: str = "counts"
    suffix: str = "_counts"
    include_unknown: bool = False
    unknown_label: str = "unknown"


@router.post("/scanpy/sum_counts_by_species")
def scanpy_sum_counts_by_species(
    request: SumCountsBySpeciesRequest, dataset: str | None = Query(None)
):
    """Sum per-cell UMIs per species using the .var species column."""
    adaptor = get_adaptor(dataset)
    try:
        return adaptor.sum_counts_by_species(
            species_column=request.species_column, layer=request.layer,
            suffix=request.suffix, include_unknown=request.include_unknown,
            unknown_label=request.unknown_label,
        )
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))


class AddVarBooleanRequest(BaseModel):
    name: str
    pattern: str
    match_mode: str = "prefix"


@router.post("/scanpy/add_var_boolean")
def scanpy_add_var_boolean(
    request: AddVarBooleanRequest, dataset: str | None = Query(None)
):
    """Add a boolean .var column from a gene-name prefix/regex."""
    adaptor = get_adaptor(dataset)
    try:
        return adaptor.add_var_boolean_column(
            name=request.name, pattern=request.pattern, match_mode=request.match_mode,
        )
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))


class CalculateQcMetricsRequest(BaseModel):
    qc_vars: str | None = None      # comma-separated boolean .var columns
    percent_top: str | None = None  # comma-separated ints; blank -> None
    log1p: bool = True


@router.post("/scanpy/calculate_qc_metrics")
def scanpy_calculate_qc_metrics(
    request: CalculateQcMetricsRequest, dataset: str | None = Query(None)
):
    """sc.pp.calculate_qc_metrics with user-selected qc_vars."""
    adaptor = get_adaptor(dataset)
    try:
        return adaptor.run_calculate_qc_metrics(
            qc_vars=request.qc_vars, percent_top=request.percent_top,
            log1p=request.log1p,
        )
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))


class ScoreGenesUcellRequest(BaseModel):
    sets: list[dict[str, Any]]
    layer: str = 'counts'
    max_rank: int = 1500
    w_neg: float = 1.0


@router.post("/scanpy/score_genes_ucell")
def scanpy_score_genes_ucell(request: ScoreGenesUcellRequest, dataset: str | None = Query(None)):
    """Score directional gene sets with UCell; writes UCell_<name> .obs columns."""
    adaptor = get_adaptor(dataset)
    try:
        return adaptor.score_gene_sets_ucell(
            sets=request.sets, layer=request.layer,
            max_rank=request.max_rank, w_neg=request.w_neg,
        )
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


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
            use_rep=request.use_rep,
            active_cell_indices=request.active_cell_indices,
        )
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/scanpy/neighbor_graphs")
def list_neighbor_graphs(dataset: str | None = Query(None)):
    """List available cell connectivity graphs in obsp (combinable via combine_neighbors)."""
    adaptor = get_adaptor(dataset)
    try:
        return {'graphs': adaptor.list_neighbor_graphs()}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/scanpy/combine_neighbors")
def combine_neighbors(request: CombineNeighborsRequest, dataset: str | None = Query(None)):
    """Combine multiple cell connectivity graphs with user-defined weights.

    Writes the combined graph to ``obsp[target_key]`` (default 'connectivities'),
    so downstream Leiden/UMAP operate on the combined graph automatically.
    """
    adaptor = get_adaptor(dataset)
    try:
        sources = [s.dict() for s in request.sources]
        return adaptor.combine_neighbor_graphs(
            sources=sources,
            target_key=request.target_key,
        )
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/scanpy/layers")
def list_layers(dataset: str | None = Query(None)):
    """List the readable expression matrices for downstream gene analyses.

    Returns a synthetic 'X' entry first (always present) followed by the
    available adata.layers keys; used by Gene PCA / Gene Neighbors / Cluster
    Genes "Source matrix" dropdowns.
    """
    adaptor = get_adaptor(dataset)
    try:
        return {'layers': adaptor.list_layers()}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


class SmoothRequest(BaseModel):
    graph_key: str
    n_steps: int = 1
    source_layer: str | None = None
    output_layer: str = 'smoothed'
    self_loop_weight: float = 1.0
    post_transform: str = 'none'


@router.post("/scanpy/smooth")
def run_smooth(request: SmoothRequest, dataset: str | None = Query(None)):
    """Smooth expression over a kNN graph; result lands in adata.layers[output_layer]."""
    adaptor = get_adaptor(dataset)
    try:
        return adaptor.run_smooth(
            graph_key=request.graph_key,
            n_steps=request.n_steps,
            source_layer=request.source_layer,
            output_layer=request.output_layer,
            self_loop_weight=request.self_loop_weight,
            post_transform=request.post_transform,
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
            graph_key=request.graph_key,
            key_added=request.key_added,
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
            graph_key=request.graph_key,
            active_cell_indices=request.active_cell_indices,
        )
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/scanpy/pca_loadings")
def get_pca_loadings(
    top_n: int = Query(10, ge=1, le=500),
    dataset: str | None = Query(None),
):
    """Return top +/- loading genes per computed PC."""
    adaptor = get_adaptor(dataset)
    try:
        return adaptor.get_pca_loadings(top_n=top_n)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.get("/scanpy/pca_subsets")
def list_pca_subsets(dataset: str | None = Query(None)):
    """List derived PC subsets (X_pca_no* obsm slots)."""
    adaptor = get_adaptor(dataset)
    try:
        return {'subsets': adaptor.list_pca_subsets()}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/scanpy/pca_subsets")
def create_pca_subset(
    request: CreatePcaSubsetRequest,
    dataset: str | None = Query(None),
):
    """Create a derived PC subset that excludes the given (1-indexed) PCs."""
    adaptor = get_adaptor(dataset)
    try:
        return adaptor.create_pca_subset(
            drop_pc_indices=request.drop_pc_indices,
            suffix=request.suffix,
        )
    except ValueError as e:
        # Use 409 for suffix collision so the UI can show a specific toast.
        if 'already exists' in str(e):
            raise HTTPException(status_code=409, detail=str(e))
        raise HTTPException(status_code=400, detail=str(e))


@router.delete("/scanpy/pca_subsets/{obsm_key}")
def delete_pca_subset(
    obsm_key: str,
    dataset: str | None = Query(None),
):
    """Delete a derived PC subset by its obsm key (e.g., X_pca_noPC2_5)."""
    adaptor = get_adaptor(dataset)
    try:
        adaptor.delete_pca_subset(obsm_key)
        return {'status': 'deleted', 'obsm_key': obsm_key}
    except ValueError as e:
        if 'not found' in str(e):
            raise HTTPException(status_code=404, detail=str(e))
        raise HTTPException(status_code=400, detail=str(e))


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
    # Optional layer name in adata.layers to read from instead of adata.X.
    # None / 'X' / omitted → adata.X (the default).
    layer: str | None = None


class GeneNeighborsRequest(BaseModel):
    n_neighbors: int = 15
    metric: str = 'euclidean'
    basis: str = 'gene_pca'
    gene_subset: str | list[str] | GeneSubsetSpec | None = None
    scale: bool = True
    active_cell_indices: list[int] | None = None
    # Only used when basis='expression'. None / 'X' / omitted → adata.X.
    layer: str | None = None


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


@router.get("/var/boolean_column_values")
def get_var_boolean_column_values(dataset: str | None = Query(None)):
    """Return per-column True-index lists for boolean .var columns.

    Used by the frontend Gene Mask modal for client-side preview count
    computation (fetched once on modal open; no per-toggle round-trips).

    Returns:
        {
            "n_genes": int,
            "columns": {
                "<column_name>": [positional_index, positional_index, ...],
                ...
            }
        }
    """
    adaptor = get_adaptor(dataset)
    bool_cols = adaptor.get_var_boolean_columns()
    result: dict[str, Any] = {}
    for col_info in bool_cols:
        name = col_info['name']
        arr = adaptor._column_to_bool_array(name)
        result[name] = [int(i) for i in np.nonzero(arr)[0]]
    return {
        'n_genes': adaptor.n_genes,
        'columns': result,
    }


@router.get("/var/column_genes")
def get_var_column_genes(
    column: str = Query(...), dataset: str | None = Query(None)
):
    """Gene names where a boolean .var column is True."""
    adaptor = get_adaptor(dataset)
    try:
        genes = adaptor.column_to_gene_names(column)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    return {"column": column, "genes": genes}


class GeneMaskRequest(BaseModel):
    keep_columns: list[str] = []
    hide_columns: list[str] = []
    keep_combine_mode: str = 'or'


@router.get("/gene_mask")
def get_gene_mask(dataset: str | None = Query(None)):
    """Get the current gene mask config for a dataset."""
    adaptor = get_adaptor(dataset)
    return adaptor.get_gene_mask()


@router.post("/gene_mask")
def set_gene_mask(request: GeneMaskRequest, dataset: str | None = Query(None)):
    """Apply a gene mask.

    Raises 400 if the config references missing/non-bool columns or would
    leave zero visible genes.
    """
    adaptor = get_adaptor(dataset)
    try:
        return adaptor.set_gene_mask(
            keep_columns=request.keep_columns,
            hide_columns=request.hide_columns,
            keep_combine_mode=request.keep_combine_mode,
        )
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.delete("/gene_mask")
def clear_gene_mask(dataset: str | None = Query(None)):
    """Clear the gene mask."""
    adaptor = get_adaptor(dataset)
    return adaptor.clear_gene_mask()


class SwapVarIndexRequest(BaseModel):
    column: str


@router.get("/var/identifier_columns")
def get_var_identifier_columns(dataset: str | None = Query(None)):
    """Get .var columns suitable as gene identifiers."""
    adaptor = get_adaptor(dataset)
    return adaptor.get_var_identifier_columns()


@router.get("/var/symbol_mapping_preview")
def symbol_mapping_preview(dataset: str | None = Query(None)):
    """What mapping Ensembl ids to official symbols would do here.

    Mutates nothing, so the cost is visible while the decision is still the
    user's.
    """
    adaptor = get_adaptor(dataset)
    try:
        return adaptor.preview_gene_symbol_mapping()
    except HTTPException:
        raise
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))


class MapGeneSymbolsRequest(BaseModel):
    column: str = "gene_symbol"
    set_as_index: bool = False


@router.post("/var/map_symbols")
def map_gene_symbols(request: MapGeneSymbolsRequest,
                     dataset: str | None = Query(None)):
    """Write official gene symbols into .var, optionally as the gene index."""
    adaptor = get_adaptor(dataset)
    try:
        return adaptor.map_gene_symbols(
            column=request.column, set_as_index=request.set_as_index)
    except HTTPException:
        raise
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except KeyError as e:
        raise HTTPException(status_code=404, detail=str(e))


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
            layer=request.layer,
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
            layer=request.layer,
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
    section_col: str | None = None


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
    section_col: str | None = None


class MultiContourPrepareRequest(BaseModel):
    gene_sets: dict[str, list[str]]
    contour_levels: int = 3
    log_transform: bool = True
    grid_res: int | None = None
    smooth_sigma: float | None = None
    section_col: str | None = None


class MultiContourFinalizeRequest(BaseModel):
    token: str
    cutoffs: dict[str, float]
    profile_k: int = 15
    out_name: str = "tissue"
    save_qc: bool = False
    params: dict | None = None


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


@router.get("/spatial_key")
def get_spatial_key(dataset: str | None = Query(None)):
    """Which .obsm arrays could be spatial coordinates, and which one is active.

    The picker this feeds exists because a Localize map lands in
    ``X_spatial_pred``, which auto-detection cannot see — so the spatial tools
    refused to run on coordinates the user had just made.
    """
    return get_adaptor(dataset).spatial_key_options()


class SpatialKeyRequest(BaseModel):
    key: str | None = None


@router.put("/spatial_key")
def put_spatial_key(request: SpatialKeyRequest, dataset: str | None = Query(None)):
    """Choose the .obsm array that acts as spatial coordinates. None resets."""
    adaptor = get_adaptor(dataset)
    try:
        return adaptor.set_spatial_key(request.key)
    except HTTPException:
        raise
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except KeyError as e:
        raise HTTPException(status_code=404, detail=str(e))


@router.get("/spatial_scale")
def get_spatial_scale(dataset: str | None = Query(None)):
    """How many µm one spatial coordinate unit spans, if known."""
    return get_adaptor(dataset).spatial_scale()


class SpatialScaleRequest(BaseModel):
    um_per_unit: float | None = None


@router.put("/spatial_scale")
def put_spatial_scale(request: SpatialScaleRequest, dataset: str | None = Query(None)):
    """Set the physical scale of the spatial coordinates. None clears it."""
    adaptor = get_adaptor(dataset)
    try:
        return adaptor.spatial_scale_set(request.um_per_unit)
    except HTTPException:
        raise
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))


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
            section_col=request.section_col,
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


@router.post("/scanpy/contourize", status_code=202)
def run_contourize(request: ContourizeRequest, dataset: str | None = Query(None)):
    """Compute spatial expression contours from a gene set (cancellable background task).

    Requires: spatial coordinates in .obsm

    Returns:
        Task ID and status for polling
    """
    adaptor = get_adaptor(dataset)
    try:
        compute_fn, apply_fn = adaptor.prepare_contourize(
            genes=request.genes,
            contour_levels=request.contour_levels,
            log_transform=request.log_transform,
            smooth_sigma=request.smooth_sigma,
            grid_res=request.grid_res,
            annotation_key=request.annotation_key,
            section_col=request.section_col,
        )
        task_id = task_manager.submit(compute_fn, apply_fn)
        return {"task_id": task_id, "status": "running"}
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/scanpy/contour_suggest")
def contour_suggest(dataset: str | None = Query(None)):
    """Data-aware suggested contour params (grid_res, smooth_sigma) for prefilling
    the contour UI. Works for both single- and multi-gene-set contouring."""
    adaptor = get_adaptor(dataset)
    try:
        return adaptor.suggest_contour_params()
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.post("/scanpy/multicontour/prepare", status_code=202)
def multicontour_prepare(request: MultiContourPrepareRequest, dataset: str | None = Query(None)):
    """Phase 1 of multi-contour: score each gene-set module (cancellable task).

    Validates prerequisites synchronously (X_pca, spatial coords, >=2 gene sets)
    so failures return 400 immediately; the heavy scoring runs in the background.

    Returns:
        Task ID + status. Poll /tasks/{id}; the result holds token + modules + params.
    """
    adaptor = get_adaptor(dataset)
    try:
        adaptor.check_multicontour_prereqs(request.gene_sets)

        def compute_fn():
            return adaptor.prepare_multicontour(
                gene_sets=request.gene_sets,
                contour_levels=request.contour_levels,
                log_transform=request.log_transform,
                grid_res=request.grid_res,
                smooth_sigma=request.smooth_sigma,
                section_col=request.section_col,
            )

        def apply_fn(result):
            return result  # scores already cached on the adaptor in compute_fn

        task_id = task_manager.submit(compute_fn, apply_fn)
        return {"task_id": task_id, "status": "running"}
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.post("/scanpy/multicontour/finalize")
def multicontour_finalize(request: MultiContourFinalizeRequest, dataset: str | None = Query(None)):
    """Phase 2 of multi-contour: binarize, assign, resolve conflicts, write column."""
    adaptor = get_adaptor(dataset)
    try:
        return adaptor.finalize_multicontour(
            token=request.token,
            cutoffs=request.cutoffs,
            profile_k=request.profile_k,
            out_name=request.out_name,
            save_qc=request.save_qc,
            params=request.params,
        )
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))


# =========================================================================
# Ligand-receptor spatial signaling (CytoSignal-style)
# =========================================================================
class LigRecPrepareRequest(BaseModel):
    radius: float | None = None
    sigma: float | None = None
    n_perm: int = 100
    min_cells: int = 10
    p_thresh: float = 0.05
    recep_smooth: bool = False
    smooth: bool = True
    types: list[str] | None = None
    section_col: str | None = None
    max_pairs: int = 400
    gene_subset: str | None = None


class LigRecFinalizeRequest(BaseModel):
    interactions: list[str]
    write_significance: bool = False


@router.get("/scanpy/ligrec/suggest")
def ligrec_suggest(dataset: str | None = Query(None)):
    """Data-driven default parameters (radius, n_perm, min_cells) for the LR tool."""
    adaptor = get_adaptor(dataset)
    try:
        return adaptor.suggest_ligrec_params()
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.get("/scanpy/ligrec/result")
def ligrec_result(dataset: str | None = Query(None)):
    """Return a previously-computed ligand-receptor result (for re-selection),
    or null if none has been run on this dataset."""
    adaptor = get_adaptor(dataset)
    return adaptor.get_ligrec_result()


@router.post("/scanpy/ligrec/prepare", status_code=202)
def ligrec_prepare(request: LigRecPrepareRequest, dataset: str | None = Query(None)):
    """Phase 1 of ligand-receptor scoring: score + test every usable pair.

    Validates spatial coordinates synchronously (400 on failure); the heavy
    scoring + permutation test runs in the background and reports progress. Poll
    /tasks/{id}; the result holds the ranked summary + params, and the full score
    matrices are persisted to the dataset.
    """
    adaptor = get_adaptor(dataset)
    try:
        # Fail fast if there are no spatial coordinates.
        adaptor._ligrec_spatial_coords()

        def compute_fn(report):
            return adaptor.prepare_ligrec(
                radius=request.radius,
                sigma=request.sigma,
                n_perm=request.n_perm,
                min_cells=request.min_cells,
                p_thresh=request.p_thresh,
                recep_smooth=request.recep_smooth,
                smooth=request.smooth,
                types=request.types,
                section_col=request.section_col,
                max_pairs=request.max_pairs,
                gene_subset=request.gene_subset,
                progress_callback=report,
            )

        def apply_fn(result):
            return result  # score matrices persisted to the adata in compute_fn

        task_id = task_manager.submit(compute_fn, apply_fn)
        return {"task_id": task_id, "status": "running"}
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.post("/scanpy/ligrec/finalize")
def ligrec_finalize(request: LigRecFinalizeRequest, dataset: str | None = Query(None)):
    """Phase 2 of ligand-receptor scoring: write selected score columns to .obs."""
    adaptor = get_adaptor(dataset)
    try:
        return adaptor.finalize_ligrec(
            interactions=request.interactions,
            write_significance=request.write_significance,
        )
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))


class NeighborhoodRequest(BaseModel):
    column: str
    mode: str = "knn"
    n_neighs: int = 10
    radius: float | None = None
    n_perms: int = 1000
    section_col: str | None = None
    seed: int = 0


@router.get("/scanpy/neighborhood/result")
def neighborhood_result(dataset: str | None = Query(None)):
    """Return the stored neighborhood enrichment result, or null if none."""
    adaptor = get_adaptor(dataset)
    return adaptor.get_neighborhood_result()


@router.post("/scanpy/neighborhood/run", status_code=202)
def neighborhood_run(request: NeighborhoodRequest, dataset: str | None = Query(None)):
    """Cell-type neighborhood composition + co-location enrichment.

    Validates synchronously (400 on bad column / graph params); the
    permutation test runs in the background with progress. Poll /tasks/{id};
    the result holds the types x types composition and enrichment matrices,
    and the per-cell composition is persisted to the dataset.
    """
    adaptor = get_adaptor(dataset)
    try:
        compute_fn, apply_fn = adaptor.prepare_neighborhood(
            column=request.column,
            mode=request.mode,
            n_neighs=request.n_neighs,
            radius=request.radius,
            n_perms=request.n_perms,
            section_col=request.section_col,
            seed=request.seed,
        )
        task_id = task_manager.submit(compute_fn, apply_fn)
        return {"task_id": task_id, "status": "running"}
    except HTTPException:
        raise
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# =========================================================================
# NMF gene programs
# =========================================================================


class GeneNmfRequest(BaseModel):
    k: int = 10
    gene_subset: str | list[str] | None = None
    layer: str | None = None
    transform: str = 'log1p'
    key: str = 'NMF'
    l1_w: float = 0.0
    l1_h: float = 0.0
    max_iter: int = 500
    tol: float = 1e-4
    seed: int = 0
    specificity_weight: float = 1.0
    weight_explained: float = 0.5
    max_genes: int = 200
    overwrite: bool = False
    # Cell scoping, same vocabulary as /cluster_gene_set.
    cell_context: str = 'all'
    cell_indices: list[int] | None = None
    annotation_column: str | None = None
    annotation_values: list[str] | None = None


@router.get("/gene_nmf/result")
def gene_nmf_result(key: str = Query('NMF'), dataset: str | None = Query(None)):
    """Return a stored NMF run, or null if that key has never been run."""
    adaptor = get_adaptor(dataset)
    return adaptor.get_gene_nmf_result(key)


@router.get("/gene_nmf/runs")
def gene_nmf_runs(dataset: str | None = Query(None)):
    """Keys of every NMF run stored on this dataset."""
    adaptor = get_adaptor(dataset)
    return {"keys": adaptor.list_gene_nmf_runs()}


@router.post("/gene_nmf/run", status_code=202)
def gene_nmf_run(request: GeneNmfRequest, dataset: str | None = Query(None)):
    """Factorize expression into gene programs (GeneNMF, single sample).

    Validates synchronously (400 on a bad k, gene subset, layer or key); the
    factorization runs in the background with progress. Poll /tasks/{id}; the
    result holds the per-program gene sets, and the per-cell program usage is
    persisted to the dataset as a score matrix.
    """
    adaptor = get_adaptor(dataset)
    cell_indices = _resolve_cell_context(
        adaptor,
        context=request.cell_context,
        indices=request.cell_indices,
        annotation_column=request.annotation_column,
        annotation_values=request.annotation_values,
    )
    try:
        compute_fn, apply_fn = adaptor.prepare_gene_nmf(
            k=request.k,
            gene_subset=request.gene_subset,
            cell_indices=cell_indices,
            layer=request.layer,
            transform=request.transform,
            key=request.key,
            l1_w=request.l1_w,
            l1_h=request.l1_h,
            max_iter=request.max_iter,
            tol=request.tol,
            seed=request.seed,
            specificity_weight=request.specificity_weight,
            weight_explained=request.weight_explained,
            max_genes=request.max_genes,
            overwrite=request.overwrite,
        )
        task_id = task_manager.submit(compute_fn, apply_fn)
        return {"task_id": task_id, "status": "running"}
    except HTTPException:
        raise
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


class MetaProgramsRequest(BaseModel):
    sample_column: str
    ks: list[int] = [4, 5, 6]
    n_mp: int = 10
    gene_subset: str | list[str] | None = None
    layer: str | None = None
    transform: str = 'log1p'
    key: str = 'MP'
    min_cells: int = 10
    l1_w: float = 0.0
    l1_h: float = 0.0
    max_iter: int = 500
    tol: float = 1e-4
    seed: int = 0
    specificity_weight: float = 5.0
    weight_explained: float = 0.8
    max_genes: int = 200
    metric: str = 'cosine'
    min_confidence: float = 0.5
    overwrite: bool = False


@router.get("/gene_nmf/meta/columns")
def meta_program_columns(dataset: str | None = Query(None)):
    """Categorical .obs columns that could identify samples."""
    adaptor = get_adaptor(dataset)
    return {"columns": adaptor.sample_column_candidates()}


@router.get("/gene_nmf/meta/result")
def meta_programs_result(key: str = Query('MP'), dataset: str | None = Query(None)):
    """A stored meta-program run, or null if that key has never been run."""
    adaptor = get_adaptor(dataset)
    return adaptor.get_meta_programs_result(key)


@router.get("/gene_nmf/meta/runs")
def meta_programs_runs(dataset: str | None = Query(None)):
    """Keys of every meta-program run stored on this dataset."""
    adaptor = get_adaptor(dataset)
    return {"keys": adaptor.list_meta_program_runs()}


@router.post("/gene_nmf/meta/run", status_code=202)
def meta_programs_run(request: MetaProgramsRequest, dataset: str | None = Query(None)):
    """Meta-programs: NMF per sample per rank, then consensus (GeneNMF).

    Validates synchronously (400 on a bad sample column, k list or key); the
    sweep runs in the background with progress. Poll /tasks/{id}; the result
    holds the meta-program gene sets, their metrics and per-sample
    composition, and the program similarity matrix for the heatmap.
    """
    adaptor = get_adaptor(dataset)
    try:
        compute_fn, apply_fn = adaptor.prepare_meta_programs(
            sample_column=request.sample_column,
            ks=request.ks,
            n_mp=request.n_mp,
            gene_subset=request.gene_subset,
            layer=request.layer,
            transform=request.transform,
            key=request.key,
            min_cells=request.min_cells,
            l1_w=request.l1_w,
            l1_h=request.l1_h,
            max_iter=request.max_iter,
            tol=request.tol,
            seed=request.seed,
            specificity_weight=request.specificity_weight,
            weight_explained=request.weight_explained,
            max_genes=request.max_genes,
            metric=request.metric,
            min_confidence=request.min_confidence,
            overwrite=request.overwrite,
        )
        task_id = task_manager.submit(compute_fn, apply_fn)
        return {"task_id": task_id, "status": "running"}
    except HTTPException:
        raise
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


#: What ``POST /export/save`` knows how to write.
_EXPORT_SAVE_KINDS = ('h5ad', 'metadata', 'gene_sets')


class ExportSaveRequest(BaseModel):
    path: str
    kind: str = "h5ad"
    # Only gene sets carry content. The flattening that gives that file its
    # shape lives in the frontend, so reproducing it here would let the two
    # drift; h5ad and metadata the backend can produce itself.
    content: str | None = None


@router.post("/export/save")
def export_save(request: ExportSaveRequest, dataset: str | None = Query(None)):
    """Write an export to a path the user picked, rather than to Downloads.

    The streaming ``GET /export/h5ad`` stays: downloading through the browser is
    still right when the backend is not on your machine. This is for the usual
    case, where it is, and where a multi-gigabyte h5ad has no business making
    the round trip.
    """
    if request.kind not in _EXPORT_SAVE_KINDS:
        raise HTTPException(
            status_code=400,
            detail=f"kind must be one of {list(_EXPORT_SAVE_KINDS)}, "
                   f"got '{request.kind}'",
        )

    target = Path(request.path).expanduser()
    if target.is_dir():
        raise HTTPException(
            status_code=400,
            detail=f"That path is a directory, not a file: {target}",
        )
    if not target.parent.is_dir():
        raise HTTPException(
            status_code=400, detail=f"No such directory: {target.parent}",
        )

    adaptor = get_adaptor(dataset)
    try:
        if request.kind == 'h5ad':
            adaptor.prepare_export_with_lines().write_h5ad(target)
        elif request.kind == 'metadata':
            target.write_text(adaptor.export_annotations(None))
        else:
            if request.content is None:
                raise HTTPException(
                    status_code=400,
                    detail="gene_sets needs 'content' — the sets are assembled "
                           "in the browser, so it sends the bytes to write",
                )
            target.write_text(request.content)
    except HTTPException:
        raise
    except (OSError, ValueError, KeyError) as e:
        raise HTTPException(status_code=400, detail=str(e))

    return {
        "path": str(target),
        "kind": request.kind,
        "n_bytes": int(target.stat().st_size),
    }


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
    if entry.progress is not None:
        response["progress"] = entry.progress
    if entry.message is not None:
        response["message"] = entry.message
    return response


@router.post("/tasks/{task_id}/cancel")
def cancel_task(task_id: str):
    """Cancel a running background task."""
    entry = task_manager.get_status(task_id)
    if entry is None:
        raise HTTPException(status_code=404, detail="Task not found")
    task_manager.cancel(task_id)
    return {"task_id": task_id, "status": "cancelling"}


@router.post("/cluster_gene_set")
def cluster_gene_set_route(req: ClusterGeneSetRequest, dataset: str | None = Query(None)):
    adaptor = get_adaptor(dataset)
    cell_indices = _resolve_cell_context(
        adaptor,
        context=req.cell_context,
        indices=req.cell_indices,
        annotation_column=req.annotation_column,
        annotation_values=req.annotation_values,
    )
    try:
        if req.method == "auto":
            # Auto returns modules-only clusters plus a separate unassigned
            # group and per-module diagnostics, so the UI can label and explain.
            report = adaptor.auto_coexpression_report(
                gene_names=req.gene_names,
                cell_indices=cell_indices,
                layer=req.layer,
                use_gene_mask=req.use_gene_mask,
                metric=req.metric,
                min_genes=req.min_genes,
                merge_threshold=req.merge_threshold,
                purity_threshold=req.purity_threshold,
                max_split_depth=req.max_split_depth,
                min_module_corr=req.min_module_corr,
            )
            return {
                "clusters": report["modules"],
                "unassigned": report["unassigned"],
                "diagnostics": report["diagnostics"],
            }
        clusters = adaptor.cluster_gene_set(
            gene_names=req.gene_names,
            method=req.method,
            k=req.k,
            cell_indices=cell_indices,
            eps=req.eps,
            min_samples=req.min_samples,
            layer=req.layer,
            use_gene_mask=req.use_gene_mask,
        )
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    return {"clusters": clusters}


# --- PySingleCellNet cell-type classification (optional dependency) ---

class PyscnInspectRequest(BaseModel):
    path: str


class PyscnClassifyRequest(BaseModel):
    path: str
    key: str = 'SCN'
    layer: str | None = None
    case_insensitive: bool = False
    categorize: bool = True
    quantile: float = 0.05


class PyscnTrainRequest(BaseModel):
    groupby: str
    out_path: str
    n_cells_per_type: int | None = 100
    n_top_genes: int = 30
    n_top_gene_pairs: int = 40
    n_trees: int = 1000
    n_rand: int | None = None
    n_comps: int = 30
    layer: str | None = None
    # None auto-detects via xcell.layer_scale; set explicitly to override.
    source_scale: str | None = None


@router.get("/pyscn/status")
def pyscn_status(dataset: str | None = Query(None)):
    """Whether PySingleCellNet is installed, plus this dataset's pickers.

    Always 200 — "not installed" is a state the UI renders, not an error.
    """
    adaptor = get_adaptor(dataset)
    try:
        return adaptor.pyscn_status()
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/pyscn/inspect_classifier")
def pyscn_inspect_classifier(request: PyscnInspectRequest, dataset: str | None = Query(None)):
    """Describe a classifier pickle and how well its genes cover this dataset."""
    adaptor = get_adaptor(dataset)
    try:
        return adaptor.pyscn_inspect_classifier(request.path)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/pyscn/classify", status_code=202)
def pyscn_classify(request: PyscnClassifyRequest, dataset: str | None = Query(None)):
    """Classify cells against a trained classifier (cancellable background task)."""
    adaptor = get_adaptor(dataset)
    try:
        compute_fn, apply_fn = adaptor.prepare_pyscn_classify(
            request.path,
            key=request.key,
            layer=request.layer,
            case_insensitive=request.case_insensitive,
            categorize=request.categorize,
            quantile=request.quantile,
        )
        task_id = task_manager.submit(compute_fn, apply_fn)
        return {"task_id": task_id, "status": "running"}
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/pyscn/train", status_code=202)
def pyscn_train(request: PyscnTrainRequest, dataset: str | None = Query(None)):
    """Train a classifier from this dataset's labels (cancellable background task)."""
    adaptor = get_adaptor(dataset)
    try:
        compute_fn, apply_fn = adaptor.prepare_pyscn_train(
            request.groupby,
            request.out_path,
            n_cells_per_type=request.n_cells_per_type,
            n_top_genes=request.n_top_genes,
            n_top_gene_pairs=request.n_top_gene_pairs,
            n_trees=request.n_trees,
            n_rand=request.n_rand,
            n_comps=request.n_comps,
            layer=request.layer,
            source_scale=request.source_scale,
        )
        task_id = task_manager.submit(compute_fn, apply_fn)
        return {"task_id": task_id, "status": "running"}
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# =========================================================================
# Analysis record — reproducible export to Jupyter / Markdown
# =========================================================================

def _record_payload(adaptor: DataAdaptor) -> dict[str, Any]:
    """The record, with each step already translated and figures listed by id.

    Translation happens here rather than in the frontend so there is exactly one
    definition of what a step means and how faithfully it reproduces. Figure
    payloads are omitted — a dozen base64 PNGs would make every poll megabytes.
    """
    from xcell import codegen
    from xcell.notebook_export import report_counts

    record = adaptor.analysis_record
    steps = []
    for step in record.steps:
        t = codegen.translate(step)
        steps.append({
            "index": step.index,
            "action": step.action,
            "title": t.title,
            "summary": t.summary,
            "fidelity": t.fidelity,
            "warnings": t.warnings,
            "code": t.code,
            "params": step.params,
            "timestamp": step.timestamp,
            "note": step.note,
            "figure_ids": step.figure_ids,
            "n_active": step.n_active,
            "n_total": step.n_total,
            "in_report": step.index >= record.report_start,
        })
    return {
        "title": record.title,
        "abstract": record.abstract,
        "source": record.source,
        "report_start": record.report_start,
        "steps": steps,
        "figures": [
            {"id": f.id, "caption": f.caption, "step_index": f.step_index,
             "timestamp": f.timestamp}
            for f in record.figures.values()
        ],
        "counts": report_counts(record),
    }


@router.get("/record")
def get_analysis_record(dataset: str | None = Query(None)):
    """The analysis record: every recorded step, translated, plus figure metadata."""
    return _record_payload(get_adaptor(dataset))


class RecordMetaRequest(BaseModel):
    title: str = ""
    abstract: str = ""


@router.put("/record/meta")
def set_record_meta(request: RecordMetaRequest, dataset: str | None = Query(None)):
    """Set the exported document's title and abstract."""
    record = get_adaptor(dataset).analysis_record
    record.title = request.title
    record.abstract = request.abstract
    return {"title": record.title, "abstract": record.abstract}


@router.post("/record/mark")
def mark_record_start(dataset: str | None = Query(None)):
    """Treat everything from here on as the report. Discards nothing."""
    return {"report_start": get_adaptor(dataset).analysis_record.mark_start()}


class StepNoteRequest(BaseModel):
    note: str = ""


@router.post("/record/step/{index}/note")
def set_step_note(index: int, request: StepNoteRequest, dataset: str | None = Query(None)):
    """Annotate one step. An empty note clears it."""
    record = get_adaptor(dataset).analysis_record
    try:
        step = record.set_note(index, request.note)
    except IndexError:
        raise HTTPException(status_code=404, detail=f"No step at index {index}")
    return {"index": step.index, "note": step.note}


class RecordFigureRequest(BaseModel):
    png_b64: str
    caption: str = ""
    step_index: int | None = None


@router.post("/record/figure")
def add_record_figure(request: RecordFigureRequest, dataset: str | None = Query(None)):
    """Attach a PNG captured in the browser. Defaults to the most recent step."""
    import base64
    import binascii

    payload = request.png_b64
    # canvas.toDataURL() returns 'data:image/png;base64,....'
    if payload.startswith("data:"):
        _, _, payload = payload.partition(",")
    try:
        base64.b64decode(payload, validate=True)
    except (binascii.Error, ValueError):
        raise HTTPException(status_code=400, detail="Figure is not valid base64 PNG data")

    record = get_adaptor(dataset).analysis_record
    try:
        figure = record.add_figure(
            payload, caption=request.caption, step_index=request.step_index,
        )
    except IndexError as e:
        raise HTTPException(status_code=404, detail=str(e))
    return {"id": figure.id, "caption": figure.caption, "step_index": figure.step_index}


@router.delete("/record/figure/{figure_id}")
def delete_record_figure(figure_id: str, dataset: str | None = Query(None)):
    """Drop a captured figure."""
    try:
        get_adaptor(dataset).analysis_record.remove_figure(figure_id)
    except KeyError as e:
        raise HTTPException(status_code=404, detail=str(e))
    return {"removed": figure_id}


@router.delete("/record")
def clear_analysis_record(dataset: str | None = Query(None)):
    """Clear the history, keeping the load step so an export still opens the
    right file."""
    adaptor = get_adaptor(dataset)
    adaptor.analysis_record.clear()
    return _record_payload(adaptor)


class RecordExportRequest(BaseModel):
    output_dir: str
    filename: str = "analysis"
    format: str = "ipynb"          # ipynb | md | both
    include_figures: bool = True
    include_code: bool = True


_EXPORT_FORMATS = {"ipynb", "md", "both"}


def _safe_stem(filename: str) -> str:
    """A filename from a text box must be a name, not a path.

    Rejects separators and traversal outright rather than sanitizing silently —
    a user who typed a path deserves to be told it isn't one.
    """
    stem = filename.strip()
    for suffix in (".ipynb", ".md"):
        if stem.endswith(suffix):
            stem = stem[: -len(suffix)]
    if not stem or stem in (".", ".."):
        raise HTTPException(status_code=400, detail="Filename must not be empty")
    if "/" in stem or "\\" in stem or stem.startswith("."):
        raise HTTPException(
            status_code=400,
            detail="Filename must be a name, not a path (no / or \\)",
        )
    return stem


@router.post("/record/export")
def export_analysis_record(request: RecordExportRequest, dataset: str | None = Query(None)):
    """Render the record to disk as a notebook and/or markdown.

    Writes server-side, into a directory the user picks with the file browser:
    a notebook has to live next to the data to be re-runnable, and it keeps
    figure sidecars simple. The rendered markdown comes back in the response so
    the panel can preview what was written.
    """
    import json as _json

    from xcell.notebook_export import (
        figure_payloads,
        report_counts,
        selections_payload,
        to_markdown,
        to_notebook,
    )

    if request.format not in _EXPORT_FORMATS:
        raise HTTPException(
            status_code=400,
            detail=f"format must be one of {sorted(_EXPORT_FORMATS)}",
        )
    out_dir = Path(request.output_dir).expanduser()
    if not out_dir.is_dir():
        raise HTTPException(
            status_code=400, detail=f"Not a directory: {request.output_dir}",
        )
    stem = _safe_stem(request.filename)

    adaptor = get_adaptor(dataset)
    record = adaptor.analysis_record
    opts = {
        "include_figures": request.include_figures,
        "include_code": request.include_code,
        "notebook_name": stem,
    }
    wants_md = request.format in ("md", "both")
    figures = figure_payloads(record) if request.include_figures else {}
    # Markdown links figures from a sibling directory; a notebook carries them
    # inline as base64 and stays a single file.
    figure_dir = f"{stem}_figures" if (wants_md and figures) else None

    written: list[dict[str, Any]] = []
    try:
        selections = selections_payload(record)
        if selections and request.include_code:
            path = out_dir / f"{stem}_selections.json"
            path.write_text(_json.dumps(selections))
            written.append({"path": str(path), "kind": "selections"})

        if request.format in ("ipynb", "both"):
            path = out_dir / f"{stem}.ipynb"
            path.write_text(_json.dumps(to_notebook(record, **opts), indent=1))
            written.append({"path": str(path), "kind": "notebook"})

        markdown = to_markdown(record, figure_dir=figure_dir, **opts)
        if wants_md:
            path = out_dir / f"{stem}.md"
            path.write_text(markdown)
            written.append({"path": str(path), "kind": "markdown"})

        if figure_dir:
            fig_root = out_dir / figure_dir
            fig_root.mkdir(exist_ok=True)
            for name, data in figures.items():
                path = fig_root / name
                path.write_bytes(data)
                written.append({"path": str(path), "kind": "figure"})
    except OSError as e:
        raise HTTPException(status_code=400, detail=f"Could not write export: {e}")

    return {
        "files": written,
        "markdown": markdown,
        "counts": report_counts(record),
    }


# =========================================================================
# Territories — hand-drawn regions that annotate cells by occupancy
# =========================================================================


class TerritoryCut(BaseModel):
    id: str
    points: list[list[float]]
    closed: bool = False


class TerritoryAnchor(BaseModel):
    name: str
    x: float
    y: float


class TerritoryFacesRequest(BaseModel):
    ring: list[list[float]]
    cuts: list[TerritoryCut] = []
    anchors: list[TerritoryAnchor] = []


@router.post("/territories/faces")
def territory_faces(request: TerritoryFacesRequest):
    """Derive the regions a set of cuts divides a ring into.

    Deliberately stateless — no dataset, no adaptor. The drawing panel calls
    this on every edit, and a preview that could observe (or block on) live
    dataset state would couple a drag gesture to whatever else is running.
    """
    from xcell import territories as terr

    if len(request.ring) < 3:
        raise HTTPException(status_code=400, detail='A ring needs at least 3 points')
    try:
        cuts = [c.model_dump() for c in request.cuts]
        faces = terr.derive_faces(request.ring, cuts)
        names = terr.name_faces(faces, [a.model_dump() for a in request.anchors])
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    return {
        'faces': [
            {
                'polygon': [[float(x), float(y)] for x, y in f.exterior.coords],
                'name': n,
                'area': float(f.area),
                # Where the panel would drop an anchor to name this face, so it
                # needs no geometry of its own.
                'anchor': [float(f.representative_point().x),
                           float(f.representative_point().y)],
            }
            for f, n in zip(faces, names)
        ]
    }


class TerritorySection(BaseModel):
    ring: list[list[float]]
    cuts: list[TerritoryCut] = []
    anchors: list[TerritoryAnchor] = []


class TerritoryTypeRequest(BaseModel):
    embedding: str = 'spatial'
    section_col: str | None = None
    sections: dict[str, TerritorySection]
    source: str = 'drawn'


@router.get("/territories")
def list_territories(dataset: str | None = Query(None)):
    """Every territory type saved on this dataset."""
    adaptor = get_adaptor(dataset)
    return {"territories": adaptor.get_territories()}


@router.put("/territories/{type_name}")
def put_territory(
    type_name: str,
    request: TerritoryTypeRequest,
    dataset: str | None = Query(None),
):
    """Save (or replace) one territory type on this dataset."""
    adaptor = get_adaptor(dataset)
    try:
        return adaptor.save_territories(type_name, request.model_dump())
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.delete("/territories/{type_name}")
def delete_territory(type_name: str, dataset: str | None = Query(None)):
    """Forget one territory type. Any .obs columns it produced are left alone."""
    adaptor = get_adaptor(dataset)
    try:
        return adaptor.delete_territories(type_name)
    except KeyError as e:
        raise HTTPException(status_code=404, detail=str(e))


class TerritoryAssignRequest(BaseModel):
    types: list[str]
    combine: bool = False
    embedding: str | None = None


@router.post("/territories/assign")
def assign_territories(
    request: TerritoryAssignRequest,
    dataset: str | None = Query(None),
):
    """Annotate cells by which territory they occupy."""
    adaptor = get_adaptor(dataset)
    try:
        return adaptor.assign_territories(
            types=request.types,
            combine=request.combine,
            embedding=request.embedding,
        )
    except KeyError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))


# =========================================================================
# Localize — predicting spatial coordinates from a spatial reference
# =========================================================================

def _resolve_reference(slot: str | None, query_slot: str | None) -> DataAdaptor:
    """Resolve the spatial reference, which is a *different* slot to the query.

    The only place in the API where two datasets are resolved at once. Mapping a
    dataset onto itself would return each cell's own coordinate and look
    flawless, so it is refused rather than allowed to mislead.
    """
    reference_slot = slot or 'secondary'
    if reference_slot == (query_slot or 'primary'):
        raise HTTPException(
            status_code=400,
            detail='The spatial reference must be a different dataset from the '
                   'one being localized.',
        )
    return get_adaptor(reference_slot)


@router.get("/localize/suggest")
def localize_suggest(
    reference: str | None = Query(None),
    gene_subset: str | None = Query(None),
    dataset: str | None = Query(None),
):
    """What can act as a spatial reference, and how well it would match.

    The gene-overlap preview is the point: proceeding on a handful of shared
    genes yields a map that looks smooth and means nothing, and the user should
    learn that before committing to a run rather than after.
    """
    query = get_adaptor(dataset)
    references = []
    for slot, adaptor in _adaptors.items():
        spatial_key = adaptor._get_spatial_key()
        references.append({
            "slot": slot,
            "filename": str(adaptor.filepath.name),
            "n_cells": int(adaptor.n_cells),
            "n_genes": int(adaptor.n_genes),
            "has_spatial": spatial_key is not None,
            "spatial_key": spatial_key,
            "is_query": slot == (dataset or "primary"),
        })

    out: dict[str, Any] = {
        "references": references,
        # k ~ sqrt(n) is the usual rule of thumb, bounded to stay interpretable.
        "suggested_k": int(max(5, min(50, round(query.n_cells ** 0.5)))),
    }

    if reference:
        try:
            ref = _resolve_reference(reference, dataset)
            bundle = ref.spatial_reference_bundle(gene_subset=gene_subset)
            out["overlap"] = query.localize_gene_overlap(bundle)
            out["reference_sections"] = [
                c for c, d in ref.get_schema()["obs_dtypes"].items() if d == "category"
            ]
            # The reference owns which genes carry positional signal, so the
            # bases on offer (spatially_variable, highly_variable, ...) are its
            # .var flags, not the query's.
            out["reference_gene_columns"] = ref.get_var_boolean_columns()
        except HTTPException:
            raise
        except ValueError as e:
            raise HTTPException(status_code=400, detail=str(e))
    return out


class LocalizeRequest(BaseModel):
    reference: str = "secondary"
    k: int = 15
    metric: str = "correlation"
    transform: str = "zscore"
    aggregation: str = "weighted_mean"
    min_confidence: float = 0.0
    epsilon: float = 0.05
    max_iterations: int = 300
    gene_subset: str | list[str] | dict[str, Any] | None = None
    section_col: str | None = None
    layer: str | None = None
    reference_layer: str | None = None
    key_added: str = "X_spatial_pred"
    import_territories: bool = False
    assign_territories: bool = False
    dataset: str | None = None


@router.post("/localize/prepare", status_code=202)
def localize_prepare(request: LocalizeRequest, dataset: str | None = Query(None)):
    """Predict spatial coordinates for the query from a spatial reference."""
    query_slot = request.dataset or dataset
    query = get_adaptor(query_slot)
    reference = _resolve_reference(request.reference, query_slot)
    try:
        bundle = reference.spatial_reference_bundle(
            gene_subset=request.gene_subset,
            layer=request.reference_layer,
            section_col=request.section_col,
        )
        compute_fn, apply_fn = query.prepare_localize(
            bundle,
            k=request.k,
            metric=request.metric,
            transform=request.transform,
            aggregation=request.aggregation,
            min_confidence=request.min_confidence,
            epsilon=request.epsilon,
            max_iterations=request.max_iterations,
            layer=request.layer,
            key_added=request.key_added,
            import_territories=request.import_territories,
            assign_territories=request.assign_territories,
        )
        task_id = task_manager.submit(compute_fn, apply_fn)
        return {"task_id": task_id, "status": "running"}
    except HTTPException:
        raise
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except KeyError as e:
        raise HTTPException(status_code=404, detail=str(e))


class LocalizeCrossValidateRequest(BaseModel):
    reference: str = "secondary"
    k: int = 15
    metric: str = "correlation"
    transform: str = "zscore"
    aggregation: str = "weighted_mean"
    holdout_fraction: float = 0.2
    gene_subset: str | list[str] | dict[str, Any] | None = None
    section_col: str | None = None
    layer: str | None = None
    groupby: str | None = None
    seed: int = 0


@router.post("/localize/cross_validate", status_code=202)
def localize_cross_validate(
    request: LocalizeCrossValidateRequest, dataset: str | None = Query(None),
):
    """Hold out part of the reference and predict it from the rest.

    A measurement, not an analysis step: it writes nothing. The result carries
    ``same_platform: true`` because there is no platform gap to cross within one
    dataset, so the number is an upper bound on what the real mapping achieves.
    """
    reference = get_adaptor(request.reference)
    try:
        compute_fn, apply_fn = reference.prepare_localize_cross_validation(
            k=request.k,
            metric=request.metric,
            transform=request.transform,
            aggregation=request.aggregation,
            holdout_fraction=request.holdout_fraction,
            gene_subset=request.gene_subset,
            layer=request.layer,
            section_col=request.section_col,
            groupby=request.groupby,
            seed=request.seed,
        )
        task_id = task_manager.submit(compute_fn, apply_fn)
        return {"task_id": task_id, "status": "running"}
    except HTTPException:
        raise
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))


class LocalizeEvaluateRequest(BaseModel):
    predicted_key: str = "X_spatial_pred"
    truth_key: str = "spatial_true"
    groupby: str | None = None
    confidence_column: str | None = None


@router.post("/localize/evaluate")
def localize_evaluate(
    request: LocalizeEvaluateRequest, dataset: str | None = Query(None),
):
    """Score a prediction against coordinates already known to be true."""
    adaptor = get_adaptor(dataset)
    try:
        return adaptor.evaluate_localization(
            request.predicted_key,
            request.truth_key,
            groupby=request.groupby,
            confidence_column=request.confidence_column,
        )
    except KeyError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))


class EvaluateMapRequest(BaseModel):
    reference: str = "primary"
    embeddings: list[str]
    gene_sets: dict[str, list[str]] = {}
    dataset: str | None = None


@router.post("/localize/evaluate_map")
def localize_evaluate_map(
    request: EvaluateMapRequest, dataset: str | None = Query(None),
):
    """Score predicted maps against the spatial reference.

    Several embeddings at once, because these numbers are only meaningful
    comparatively — a user arrives with a handful of saved variants and no way
    to rank them.
    """
    query_slot = request.dataset or dataset
    query = get_adaptor(query_slot)
    reference = _resolve_reference(request.reference, query_slot)
    try:
        bundle = reference.spatial_reference_bundle()
        return query.evaluate_localization_maps(
            bundle, request.embeddings, request.gene_sets,
        )
    except HTTPException:
        raise
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))


class ReferenceGeometryRequest(BaseModel):
    reference: str = "primary"
    gene_sets: dict[str, list[str]] = {}
    layer: str | None = None
    dataset: str | None = None


@router.post("/localize/reference_geometry")
def localize_reference_geometry(
    request: ReferenceGeometryRequest, dataset: str | None = Query(None),
):
    """Which populations in the reference a mean estimator would collapse.

    Reads the reference and nothing else, so the answer is available before any
    map exists — which is the point. A user who learns after the run that
    ``weighted_mean`` puts their epidermis in the middle of the bud has already
    believed the picture.
    """
    query_slot = request.dataset or dataset
    reference = _resolve_reference(request.reference, query_slot)
    try:
        return reference.localize_reference_geometry(
            request.gene_sets, layer=request.layer,
        )
    except HTTPException:
        raise
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except KeyError as e:
        raise HTTPException(status_code=404, detail=str(e))
