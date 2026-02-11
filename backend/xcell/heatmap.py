"""Heatmap data computation for XCell.

Standalone module for computing expression heatmap matrices with cell ordering,
binning, gene grouping, and normalization.

Rollback: delete this file and remove the /api/heatmap route from routes.py.
"""

import numpy as np
from typing import Any, TYPE_CHECKING

if TYPE_CHECKING:
    from xcell.adaptor import DataAdaptor


def compute_heatmap_data(
    adaptor: "DataAdaptor",
    genes: list[str],
    gene_set_groups: list[dict[str, Any]] | None = None,
    aggregate_gene_sets: bool = False,
    cell_ordering: str = "none",
    obs_column: str | None = None,
    line_name: str | None = None,
    gene_ordering: str = "as_provided",
    n_bins: int = 0,
    transform: str | None = None,
    cell_indices: list[int] | None = None,
) -> dict[str, Any]:
    """Compute expression heatmap matrix.

    Args:
        adaptor: DataAdaptor instance
        genes: List of gene names to include
        gene_set_groups: Gene set grouping: [{"name": "set1", "genes": ["g1", "g2"]}, ...]
        aggregate_gene_sets: If True, each row is a gene set mean instead of per-gene
        cell_ordering: "none", "category", "line_position", "line_distance",
                       or "category_then_position"
        obs_column: Obs column name for category-based ordering
        line_name: Line name for position/distance-based ordering
        gene_ordering: "as_provided" or "peak_position"
        n_bins: Number of bins to average cells into (0 = no binning)
        transform: Optional "log1p" transformation
        cell_indices: Optional subset of cell indices

    Returns:
        Dictionary with matrix, row_labels, row_groups, column_groups, n_bins, n_cells
    """
    adata = adaptor.adata

    # Filter to valid genes
    valid_genes = [g for g in genes if g in adata.var.index]
    if not valid_genes:
        return {
            "matrix": [],
            "row_labels": [],
            "row_groups": [],
            "column_groups": [],
            "n_bins": 0,
            "n_cells": 0,
        }

    # Data source
    adata_src = adaptor.normalized_adata if transform == "log1p" else adata

    # Cell subset
    if cell_indices is not None:
        cell_idx = np.array(cell_indices, dtype=int)
    else:
        cell_idx = np.arange(adata.n_obs)

    # Cell ordering
    ordered_idx, col_groups = _order_cells(
        adaptor, cell_idx, cell_ordering, obs_column, line_name
    )

    # Extract expression matrix (ordered_cells x genes)
    gene_col_indices = [adata_src.var.index.get_loc(g) for g in valid_genes]
    X = adata_src.X
    if hasattr(X, "toarray"):
        expr = X[ordered_idx][:, gene_col_indices].toarray().astype(np.float64)
    else:
        expr = np.array(X[ordered_idx][:, gene_col_indices], dtype=np.float64)

    n_cells = expr.shape[0]

    # Gene set aggregation
    if aggregate_gene_sets and gene_set_groups:
        expr, row_labels, row_groups = _aggregate_gene_sets(
            expr, valid_genes, gene_set_groups
        )
    else:
        row_labels = list(valid_genes)
        row_groups = _label_genes_by_set(valid_genes, gene_set_groups)

    # Bin cells
    effective_bins = n_cells
    if n_bins > 0 and n_cells > n_bins:
        expr, col_groups = _bin_cells(expr, n_bins, col_groups, n_cells)
        effective_bins = n_bins

    # Transpose to (genes x bins)
    mat = expr.T

    # Handle empty matrix
    if mat.size == 0:
        return {
            "matrix": [],
            "row_labels": row_labels,
            "row_groups": row_groups,
            "column_groups": col_groups,
            "n_bins": effective_bins,
            "n_cells": n_cells,
        }

    # Per-row min-max normalization
    rmin = np.nanmin(mat, axis=1, keepdims=True)
    rmax = np.nanmax(mat, axis=1, keepdims=True)
    rng = rmax - rmin
    rng[rng == 0] = 1
    mat = (mat - rmin) / rng

    # Gene ordering by peak position
    if gene_ordering == "peak_position" and mat.shape[1] > 0:
        peaks = np.argmax(mat, axis=1)
        if (
            row_groups
            and any(g is not None for g in row_groups)
            and not aggregate_gene_sets
        ):
            order = _sort_genes_within_groups(peaks, row_groups)
        else:
            order = np.argsort(peaks)
        mat = mat[order]
        row_labels = [row_labels[i] for i in order]
        if row_groups:
            row_groups = [row_groups[i] for i in order]

    # Serialize
    matrix_out = []
    for row in mat:
        matrix_out.append(
            [round(float(v), 4) if not np.isnan(v) else 0.0 for v in row]
        )

    return {
        "matrix": matrix_out,
        "row_labels": row_labels,
        "row_groups": row_groups,
        "column_groups": col_groups,
        "n_bins": effective_bins,
        "n_cells": n_cells,
    }


# ---------------------------------------------------------------------------
# Cell ordering helpers
# ---------------------------------------------------------------------------


def _order_cells(
    adaptor: "DataAdaptor",
    cell_idx: np.ndarray,
    ordering: str,
    obs_column: str | None,
    line_name: str | None,
) -> tuple[np.ndarray, list[dict]]:
    """Compute cell ordering and return (ordered_indices, column_groups)."""
    if ordering == "none":
        return cell_idx, []

    if ordering == "category" and obs_column:
        return _order_by_category(adaptor.adata, cell_idx, obs_column)

    if ordering in ("line_position", "line_distance") and line_name:
        return _order_by_line(
            adaptor, cell_idx, line_name, use_distance=(ordering == "line_distance")
        )

    if ordering == "category_then_position" and obs_column and line_name:
        return _order_category_then_position(
            adaptor, cell_idx, obs_column, line_name
        )

    return cell_idx, []


def _order_by_category(adata, cell_idx, obs_column):
    if obs_column not in adata.obs.columns:
        return cell_idx, []

    vals = adata.obs[obs_column].values[cell_idx]
    if hasattr(vals, "cat"):
        cats = vals.cat.categories.tolist()
        codes = vals.cat.codes.values.copy()
    else:
        cats = sorted(set(str(v) for v in vals))
        cmap = {c: i for i, c in enumerate(cats)}
        codes = np.array([cmap.get(str(v), -1) for v in vals])

    order = np.argsort(codes, kind="stable")
    ordered = cell_idx[order]
    sorted_codes = codes[order]

    groups = []
    pos = 0
    for ci, cn in enumerate(cats):
        cnt = int(np.sum(sorted_codes == ci))
        if cnt > 0:
            groups.append({"name": str(cn), "start": pos, "size": cnt})
            pos += cnt

    return ordered, groups


def _order_by_line(adaptor, cell_idx, line_name, use_distance):
    line = _find_line(adaptor, line_name)
    if line is None:
        return cell_idx, []

    pts = line.get("smoothedPoints") or line["points"]
    projs = adaptor._project_cells_onto_line(np.array(pts), line["embeddingName"])

    key = "distanceToLine" if use_distance else "positionOnLine"
    all_vals = np.array([p[key] for p in projs])
    subset_vals = all_vals[cell_idx]
    order = np.argsort(subset_vals)
    return cell_idx[order], []


def _order_category_then_position(adaptor, cell_idx, obs_column, line_name):
    adata = adaptor.adata
    if obs_column not in adata.obs.columns:
        return cell_idx, []

    vals = adata.obs[obs_column].values[cell_idx]
    if hasattr(vals, "cat"):
        cats = vals.cat.categories.tolist()
        codes = vals.cat.codes.values.copy()
    else:
        cats = sorted(set(str(v) for v in vals))
        cmap = {c: i for i, c in enumerate(cats)}
        codes = np.array([cmap.get(str(v), -1) for v in vals])

    line = _find_line(adaptor, line_name)
    if line is None:
        return _order_by_category(adata, cell_idx, obs_column)

    pts = line.get("smoothedPoints") or line["points"]
    projs = adaptor._project_cells_onto_line(np.array(pts), line["embeddingName"])
    positions = np.array([p["positionOnLine"] for p in projs])
    subset_pos = positions[cell_idx]

    order = np.lexsort((subset_pos, codes))
    ordered = cell_idx[order]
    sorted_codes = codes[order]

    groups = []
    pos = 0
    for ci, cn in enumerate(cats):
        cnt = int(np.sum(sorted_codes == ci))
        if cnt > 0:
            groups.append({"name": str(cn), "start": pos, "size": cnt})
            pos += cnt

    return ordered, groups


def _find_line(adaptor, line_name):
    for dl in adaptor._drawn_lines:
        if dl["name"] == line_name:
            return dl
    return None


# ---------------------------------------------------------------------------
# Gene grouping helpers
# ---------------------------------------------------------------------------


def _aggregate_gene_sets(expr, gene_names, gene_set_groups):
    """Aggregate expression by gene set mean. expr is (cells, genes)."""
    name_to_idx = {g: i for i, g in enumerate(gene_names)}
    rows, labels, groups = [], [], []

    for gs in gene_set_groups:
        idxs = [name_to_idx[g] for g in gs["genes"] if g in name_to_idx]
        if not idxs:
            continue
        rows.append(np.nanmean(expr[:, idxs], axis=1))
        labels.append(gs["name"])
        groups.append(gs["name"])

    if not rows:
        return np.zeros((expr.shape[0], 0)), [], []
    return np.column_stack(rows), labels, groups


def _label_genes_by_set(gene_names, gene_set_groups):
    """Map each gene to its gene set name (first match)."""
    if not gene_set_groups:
        return [None] * len(gene_names)
    g2s: dict[str, str] = {}
    for gs in gene_set_groups:
        for g in gs["genes"]:
            if g not in g2s:
                g2s[g] = gs["name"]
    return [g2s.get(g) for g in gene_names]


# ---------------------------------------------------------------------------
# Binning
# ---------------------------------------------------------------------------


def _bin_cells(expr, n_bins, col_groups, n_cells):
    """Bin cells by averaging. expr is (cells, genes)."""
    edges = np.linspace(0, n_cells, n_bins + 1, dtype=int)
    binned = np.zeros((n_bins, expr.shape[1]))
    for i in range(n_bins):
        s, e = edges[i], edges[i + 1]
        if e > s:
            binned[i] = np.nanmean(expr[s:e], axis=0)

    if col_groups:
        adj = []
        for g in col_groups:
            bs = int(g["start"] / n_cells * n_bins)
            be = int((g["start"] + g["size"]) / n_cells * n_bins)
            adj.append({"name": g["name"], "start": bs, "size": max(1, be - bs)})
        return binned, adj

    return binned, col_groups


# ---------------------------------------------------------------------------
# Gene sorting
# ---------------------------------------------------------------------------


def _sort_genes_within_groups(peaks, row_groups):
    """Sort genes by peak position within each gene set group."""
    seen: set[str | None] = set()
    group_order: list[str | None] = []
    for g in row_groups:
        if g not in seen:
            seen.add(g)
            group_order.append(g)

    result: list[int] = []
    for gname in group_order:
        group_indices = [i for i, g in enumerate(row_groups) if g == gname]
        group_peaks = peaks[group_indices]
        sorted_within = np.argsort(group_peaks)
        result.extend(group_indices[j] for j in sorted_within)

    return np.array(result)
