"""Custom 2-D embedding from two numeric .obs columns."""
import numpy as np
import anndata
import pytest
from scipy.sparse import csr_matrix
from fastapi.testclient import TestClient

from xcell.adaptor import DataAdaptor
from xcell.main import app
from xcell.api import routes


def _adata():
    rng = np.random.default_rng(0)
    ad = anndata.AnnData(X=csr_matrix(rng.random((30, 4)).astype(np.float32)))
    ad.var_names = [f"g{i}" for i in range(4)]
    ad.obs["total_counts"] = rng.integers(1, 1000, 30).astype(float)
    ad.obs["n_genes"] = rng.integers(1, 500, 30).astype(float)
    ad.obs["score"] = rng.standard_normal(30)        # has negatives
    ad.obs["celltype"] = ["a", "b"] * 15             # categorical
    return ad


def test_create_obs_embedding_stores_2col_obsm():
    a = DataAdaptor("x.h5ad", adata=_adata())
    r = a.create_obs_embedding("total_counts", "n_genes")
    key = r["embedding_name"]
    assert key == "X_total_counts_vs_n_genes"
    assert a.adata.obsm[key].shape == (30, 2)
    assert np.allclose(a.adata.obsm[key][:, 0], a.adata.obs["total_counts"].to_numpy())


def test_create_obs_embedding_logs_requested_axis():
    a = DataAdaptor("x.h5ad", adata=_adata())
    r = a.create_obs_embedding("total_counts", "n_genes", log_axes="x")
    coords = a.adata.obsm[r["embedding_name"]]
    assert np.allclose(coords[:, 0], np.log1p(a.adata.obs["total_counts"].to_numpy()))
    assert np.allclose(coords[:, 1], a.adata.obs["n_genes"].to_numpy())  # y untouched


def test_create_obs_embedding_rejects_non_numeric():
    a = DataAdaptor("x.h5ad", adata=_adata())
    with pytest.raises(ValueError):
        a.create_obs_embedding("total_counts", "celltype")


def test_create_obs_embedding_rejects_log_on_negative():
    a = DataAdaptor("x.h5ad", adata=_adata())
    with pytest.raises(ValueError):
        a.create_obs_embedding("score", "n_genes", log_axes="x")


def test_create_obs_embedding_rejects_duplicate_name():
    a = DataAdaptor("x.h5ad", adata=_adata())
    a.create_obs_embedding("total_counts", "n_genes", name="X_mine")
    with pytest.raises(ValueError):
        a.create_obs_embedding("total_counts", "n_genes", name="X_mine")


def test_route_embedding_from_obs(monkeypatch):
    a = DataAdaptor("x.h5ad", adata=_adata())
    monkeypatch.setattr(routes, "get_adaptor", lambda dataset=None: a)
    client = TestClient(app)
    resp = client.post("/api/scanpy/embedding_from_obs", json={
        "col_x": "total_counts", "col_y": "n_genes", "log_axes": "both",
    })
    assert resp.status_code == 200, resp.text
    assert resp.json()["embedding_name"] == "X_total_counts_vs_n_genes"


def test_route_embedding_from_obs_bad_column(monkeypatch):
    a = DataAdaptor("x.h5ad", adata=_adata())
    monkeypatch.setattr(routes, "get_adaptor", lambda dataset=None: a)
    client = TestClient(app)
    resp = client.post("/api/scanpy/embedding_from_obs", json={
        "col_x": "total_counts", "col_y": "celltype",
    })
    assert resp.status_code == 400
