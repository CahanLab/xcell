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


# --- UMAP -----------------------------------------------------------------

def _spatial_only():
    """Visium-shaped: spatial graph, no PCA, no expression kNN.

    n_vars is above scanpy's N_PCS (50) on purpose — that is the branch where
    _choose_representation would otherwise compute a PCA behind our back.
    """
    a = DataAdaptor("x.h5ad", adata=_adata(n_genes=60))
    a.run_spatial_neighbors(n_neighs=6, coord_type="generic")
    return a


def _expression_and_spatial():
    a = _adaptor()
    a.run_pca(n_comps=5)
    a.run_neighbors(n_neighbors=5)
    a.run_spatial_neighbors(n_neighs=6, coord_type="generic")
    return a


def test_umap_on_spatial_graph_writes_its_own_embedding():
    a = _adaptor()
    a.run_pca(n_comps=5)
    a.run_neighbors(n_neighbors=5)
    a.run_umap()                       # the default one
    a.run_spatial_neighbors(n_neighs=6, coord_type="generic")

    res = a.run_umap(graph_key="spatial_connectivities")

    assert res["embedding_name"] == "X_umap_spatial"
    assert res["graph_key"] == "spatial_connectivities"
    assert a.adata.obsm["X_umap_spatial"].shape == (a.n_cells, 2)
    # The expression UMAP must survive untouched — coexisting is the point.
    assert "X_umap" in a.adata.obsm
    assert not np.allclose(a.adata.obsm["X_umap"], a.adata.obsm["X_umap_spatial"])


def test_umap_leaves_no_temporary_neighbors_entry():
    from xcell.adaptor import _GRAPH_META_KEY

    a = _expression_and_spatial()
    a.run_umap(graph_key="spatial_connectivities")

    assert _GRAPH_META_KEY not in a.adata.uns


def test_temporary_entry_is_removed_even_when_umap_raises(monkeypatch):
    from xcell import adaptor as adaptor_mod
    from xcell.adaptor import _GRAPH_META_KEY

    a = _expression_and_spatial()

    def boom(*args, **kwargs):
        raise RuntimeError("umap exploded")

    monkeypatch.setattr(adaptor_mod.sc.tl, "umap", boom)
    with pytest.raises(RuntimeError):
        a.run_umap(graph_key="spatial_connectivities")

    # A fabricated neighbors entry left behind would be silently picked up by
    # the next tool that reads uns.
    assert _GRAPH_META_KEY not in a.adata.uns


def test_umap_on_a_graph_with_no_uns_entry_at_all():
    """combine_neighbor_graphs writes only the matrix.

    Constructing a minimal uns entry is not enough either — NeighborsView
    requires distances_key to be present, though the matrix it names is never
    read.
    """
    a = _expression_and_spatial()
    a.combine_neighbor_graphs(
        sources=[{"key": "connectivities", "weight": 1.0},
                 {"key": "spatial_connectivities", "weight": 1.0}],
        target_key="blend_connectivities",
    )
    assert "blend_connectivities" in a.adata.obsp

    res = a.run_umap(graph_key="blend_connectivities")

    assert res["embedding_name"] == "X_umap_blend"
    assert not np.isnan(a.adata.obsm["X_umap_blend"]).any()


def test_umap_on_spatial_graph_without_pca_does_not_invent_one():
    """scanpy's _choose_representation runs sc.pp.pca when X_pca is missing.

    That would put an embedding into the object that no recorded step created,
    so the exported notebook would not reproduce the state. use_rep='X' takes
    the branch that just reads .X.
    """
    a = _spatial_only()
    a.run_umap(graph_key="spatial_connectivities")

    assert "X_umap_spatial" in a.adata.obsm
    assert "X_pca" not in a.adata.obsm
    assert "pca" not in a.adata.uns


def test_umap_keeps_a_real_uns_entrys_recorded_params():
    from xcell.adaptor import _umap_neighbors_meta

    a = _adaptor()
    a.run_spatial_neighbors(n_neighs=6, coord_type="generic")
    meta = _umap_neighbors_meta(a.adata, "spatial_connectivities")

    assert meta["connectivities_key"] == "spatial_connectivities"
    assert meta["distances_key"] == "spatial_distances"
    # scanpy reads params['method'] unguarded; squidpy never writes it.
    assert meta["params"]["method"] == "umap"
    # squidpy's own record survives, so the exported notebook is not invented.
    assert meta["params"]["n_neighbors"] == 6


def test_umap_default_path_is_unchanged():
    a = _adaptor()
    a.run_pca(n_comps=5)
    a.run_neighbors(n_neighbors=5)
    res = a.run_umap(min_dist=0.4, spread=1.2)

    assert res["embedding_name"] == "X_umap"
    assert "X_umap" in a.adata.obsm
    step = a._action_history[-1]
    # Recording graph_key/key_added on the default path would change every
    # existing export; it is only recorded when it differs from the default.
    assert step["params"] == {"min_dist": 0.4, "spread": 1.2, "n_components": 2}


def test_umap_custom_name_on_the_default_graph_is_recorded():
    a = _adaptor()
    a.run_pca(n_comps=5)
    a.run_neighbors(n_neighbors=5)
    res = a.run_umap(key_added="X_umap_v2")

    assert res["embedding_name"] == "X_umap_v2"
    assert a._action_history[-1]["params"]["key_added"] == "X_umap_v2"


def test_umap_blank_name_falls_back_to_the_derived_default():
    a = _expression_and_spatial()

    assert a.run_umap(graph_key="spatial_connectivities",
                      key_added="   ")["embedding_name"] == "X_umap_spatial"


def test_umap_rejects_an_unknown_graph():
    a = _adaptor()
    a.run_pca(n_comps=5)
    a.run_neighbors(n_neighbors=5)
    with pytest.raises(ValueError) as e:
        a.run_umap(graph_key="nope_connectivities")
    assert "nope_connectivities" in str(e.value)
    assert "connectivities" in str(e.value)   # lists what is available


def test_a_wrong_shaped_graph_cannot_reach_the_adaptor():
    """_require_graph checks the shape, but nothing can violate it.

    n_cells is adata.n_obs and anndata rejects a non-square obsp on assignment,
    so the guard is unreachable defence — kept because combine_neighbor_graphs
    carries the same check. This test records why there is no test driving it
    through run_umap.
    """
    a = _adaptor()
    with pytest.raises(ValueError) as e:
        a.adata.obsp["bad_connectivities"] = csr_matrix((a.n_cells, a.n_cells - 1))
    assert "shape" in str(e.value)

    # The guard itself still answers correctly when handed one directly.
    a.adata.obsp["ok_connectivities"] = csr_matrix((a.n_cells, a.n_cells))
    a._require_graph("ok_connectivities")   # does not raise


def test_umap_on_a_selection_slices_the_chosen_graph():
    a = _expression_and_spatial()
    active = list(range(20))

    res = a.run_umap(graph_key="spatial_connectivities", active_cell_indices=active)

    E = a.adata.obsm[res["embedding_name"]]
    assert not np.isnan(E[active]).any()
    assert np.isnan(E[20:]).all()


# --- Leiden ---------------------------------------------------------------

def test_leiden_on_spatial_graph_writes_its_own_column():
    a = _adaptor()
    a.run_pca(n_comps=5)
    a.run_neighbors(n_neighbors=5)
    a.run_leiden(resolution=0.5)
    a.run_spatial_neighbors(n_neighs=6, coord_type="generic")

    res = a.run_leiden(resolution=0.5, graph_key="spatial_connectivities")

    assert res["key_added"] == "leiden_spatial"
    assert res["graph_key"] == "spatial_connectivities"
    assert "leiden_spatial" in a.adata.obs
    assert "leiden" in a.adata.obs           # the expression clustering survives
    assert res["n_clusters"] >= 2


def test_leiden_on_spatial_graph_without_pca_or_expression_knn():
    a = _spatial_only()
    res = a.run_leiden(resolution=0.5, graph_key="spatial_connectivities")
    assert res["n_clusters"] >= 2
    assert "connectivities" not in a.adata.obsp


def test_leiden_explicit_name_wins_over_the_derived_one():
    a = _adaptor()
    a.run_spatial_neighbors(n_neighs=6, coord_type="generic")
    res = a.run_leiden(resolution=0.5, key_added="domains",
                       graph_key="spatial_connectivities")
    assert res["key_added"] == "domains"
    assert "domains" in a.adata.obs


def test_leiden_default_path_is_unchanged():
    a = _adaptor()
    a.run_pca(n_comps=5)
    a.run_neighbors(n_neighbors=5)
    res = a.run_leiden(resolution=0.5)

    assert res["key_added"] == "leiden"
    assert a._action_history[-1]["params"] == {"resolution": 0.5, "key_added": "leiden"}


def test_leiden_rejects_an_unknown_graph():
    a = _adaptor()
    a.run_pca(n_comps=5)
    a.run_neighbors(n_neighbors=5)
    with pytest.raises(ValueError):
        a.run_leiden(graph_key="nope_connectivities")


def test_leiden_on_a_selection_slices_the_chosen_graph():
    a = _adaptor()
    a.run_spatial_neighbors(n_neighs=6, coord_type="generic")
    active = list(range(20))
    res = a.run_leiden(resolution=0.5, graph_key="spatial_connectivities",
                       active_cell_indices=active)
    labels = a.adata.obs[res["key_added"]]
    assert set(labels.iloc[20:]) == {"unassigned"}
    assert "unassigned" not in set(labels.iloc[:20])
