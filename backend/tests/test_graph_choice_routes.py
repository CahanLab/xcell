"""Routes for choosing the connectivity graph UMAP and Leiden run on."""

import numpy as np
import anndata
from fastapi.testclient import TestClient
from scipy.sparse import csr_matrix

from xcell.main import app
from xcell.api import routes
from xcell.adaptor import DataAdaptor

client = TestClient(app)


def _adata():
    rng = np.random.default_rng(0)
    coords = np.array([[float(i), float(j)] for i in range(6) for j in range(6)])
    n = coords.shape[0]
    X = rng.poisson(1.0, (n, 8)).astype(np.float32)
    X[coords[:, 0] <= 2, :4] += 10.0
    X[coords[:, 0] > 2, 4:] += 10.0
    ad = anndata.AnnData(X=csr_matrix(X))
    ad.var_names = [f"g{i}" for i in range(8)]
    ad.obsm["spatial"] = coords
    return ad


def _install():
    a = DataAdaptor("x.h5ad", adata=_adata())
    a.run_pca(n_comps=5)
    a.run_neighbors(n_neighbors=5)
    a.run_spatial_neighbors(n_neighs=6, coord_type="generic")
    routes.set_adaptor(a, slot="primary")
    return a


def test_neighbor_graphs_lists_a_suffix():
    _install()
    body = client.get("/api/scanpy/neighbor_graphs").json()
    graphs = {g["key"]: g for g in body["graphs"]}
    assert graphs["spatial_connectivities"]["suffix"] == "spatial"
    assert graphs["connectivities"]["suffix"] == ""


def test_umap_route_accepts_a_graph_key():
    a = _install()
    r = client.post("/api/scanpy/umap", json={"graph_key": "spatial_connectivities"})
    assert r.status_code == 200, r.text
    assert r.json()["embedding_name"] == "X_umap_spatial"
    assert "X_umap_spatial" in a.adata.obsm


def test_umap_route_accepts_an_explicit_name():
    _install()
    r = client.post("/api/scanpy/umap",
                    json={"graph_key": "spatial_connectivities", "key_added": "X_domains"})
    assert r.json()["embedding_name"] == "X_domains"


def test_umap_route_rejects_an_unknown_graph_with_400():
    _install()
    r = client.post("/api/scanpy/umap", json={"graph_key": "nope_connectivities"})
    assert r.status_code == 400
    assert "nope_connectivities" in r.json()["detail"]


def test_leiden_route_accepts_a_graph_key():
    a = _install()
    r = client.post("/api/scanpy/leiden",
                    json={"resolution": 0.5, "graph_key": "spatial_connectivities"})
    assert r.status_code == 200, r.text
    assert r.json()["key_added"] == "leiden_spatial"
    assert "leiden_spatial" in a.adata.obs


def test_umap_route_default_body_still_works():
    _install()
    r = client.post("/api/scanpy/umap", json={})
    assert r.status_code == 200
    assert r.json()["embedding_name"] == "X_umap"
