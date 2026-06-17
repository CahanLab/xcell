"""spatial_neighbors should optionally build a per-section (block-diagonal) graph."""

import numpy as np
import anndata
from scipy.sparse import csr_matrix

from xcell.adaptor import DataAdaptor


def _two_section_adata():
    cells, secs = [], []
    for i in range(8):
        for j in range(8):
            cells.append((float(i), float(j)))
            secs.append("A" if i < 4 else "B")
    coords = np.array(cells, dtype=np.float32)
    n = coords.shape[0]
    X = csr_matrix(np.ones((n, 3), dtype=np.float32))
    ad = anndata.AnnData(X=X)
    ad.var_names = ["g0", "g1", "g2"]
    ad.obsm["spatial"] = coords
    ad.obs["section"] = np.array(secs)
    return ad


def _cross_section_edges(adata):
    conn = adata.obsp["spatial_connectivities"].tocoo()
    sec = np.asarray(adata.obs["section"].astype(str).values)
    return int(np.sum(sec[conn.row] != sec[conn.col]))


def test_delaunay_bridges_sections_without_section_col():
    a = DataAdaptor("x.h5ad", adata=_two_section_adata())
    a.run_spatial_neighbors(delaunay=True, coord_type="generic")
    assert _cross_section_edges(a.adata) > 0  # convex-hull triangulation bridges


def test_section_col_yields_block_diagonal_graph():
    a = DataAdaptor("x.h5ad", adata=_two_section_adata())
    a.run_spatial_neighbors(delaunay=True, coord_type="generic", section_col="section")
    assert _cross_section_edges(a.adata) == 0
    # within-section edges still exist
    assert a.adata.obsp["spatial_connectivities"].nnz > 0
