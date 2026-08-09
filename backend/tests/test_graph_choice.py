"""Choosing which obsp connectivity graph UMAP and Leiden run on.

The point of these tests is that picking a non-default graph must not disturb
anything else: the default path stays byte-identical, the temporary neighbors
entry never survives the call, and scanpy is never given the chance to compute
a PCA nobody asked for.
"""

import numpy as np
import anndata
import pytest
from scipy.sparse import csr_matrix

from xcell.adaptor import DataAdaptor, _default_output_name, _graph_suffix


def _adata(n_genes=8):
    """A small grid of cells with spatial coords and two gene blocks."""
    rng = np.random.default_rng(0)
    coords = np.array([[float(i), float(j)] for i in range(6) for j in range(6)])
    n = coords.shape[0]
    X = rng.poisson(1.0, (n, n_genes)).astype(np.float32)
    # Left half expresses the first block, right half the second, so a graph
    # built on either expression or space has real structure to cluster.
    X[coords[:, 0] <= 2, : n_genes // 2] += 10.0
    X[coords[:, 0] > 2, n_genes // 2:] += 10.0
    ad = anndata.AnnData(X=csr_matrix(X))
    ad.var_names = [f"g{i}" for i in range(n_genes)]
    ad.obsm["spatial"] = coords
    return ad


def _adaptor(**kw):
    return DataAdaptor("x.h5ad", adata=_adata(**kw))


def test_graph_suffix_rules():
    assert _graph_suffix(None) == ""
    assert _graph_suffix("") == ""
    # The default graph contributes no suffix, so X_umap keeps its name.
    assert _graph_suffix("connectivities") == ""
    assert _graph_suffix("spatial_connectivities") == "spatial"
    assert _graph_suffix("expr_plus_space_connectivities") == "expr_plus_space"
    # A key that does not follow the convention is used whole rather than dropped.
    assert _graph_suffix("weird") == "weird"


def test_default_output_name():
    assert _default_output_name("X_umap", None) == "X_umap"
    assert _default_output_name("X_umap", "connectivities") == "X_umap"
    assert _default_output_name("X_umap", "spatial_connectivities") == "X_umap_spatial"
    assert _default_output_name("leiden", "spatial_connectivities") == "leiden_spatial"


def test_list_neighbor_graphs_reports_suffix():
    a = _adaptor()
    a.run_pca(n_comps=5)
    a.run_neighbors(n_neighbors=5)
    a.run_spatial_neighbors(n_neighs=6, coord_type="generic")

    graphs = {g["key"]: g for g in a.list_neighbor_graphs()}
    assert graphs["connectivities"]["suffix"] == ""
    assert graphs["spatial_connectivities"]["suffix"] == "spatial"
