"""Pattern-based boolean .var columns + calculate_qc_metrics."""
import numpy as np
import anndata
import pytest
from scipy.sparse import csr_matrix
from fastapi.testclient import TestClient

from xcell.adaptor import DataAdaptor
from xcell.main import app
from xcell.api import routes


def _adata():
    genes = ["mt-Nd1", "mt-Co1", "Actb", "Gapdh", "Rpl13", "Rps6"]
    rng = np.random.default_rng(0)
    ad = anndata.AnnData(X=csr_matrix(rng.integers(0, 30, (12, 6)).astype(np.float32)))
    ad.var_names = genes
    return ad


def test_add_var_boolean_prefix():
    a = DataAdaptor("x.h5ad", adata=_adata())
    r = a.add_var_boolean_column("mt", "mt-", match_mode="prefix")
    assert r["n_genes_matched"] == 2
    assert a.adata.var["mt"].dtype == bool
    assert a.adata.var["mt"].tolist() == [True, True, False, False, False, False]
    # appears in the boolean-columns list used by the qc_vars picker
    assert "mt" in {c["name"] for c in a.get_var_boolean_columns()}


def test_add_var_boolean_regex():
    a = DataAdaptor("x.h5ad", adata=_adata())
    r = a.add_var_boolean_column("ribo", "^Rp[ls]", match_mode="regex")
    assert r["n_genes_matched"] == 2
    assert a.adata.var["ribo"].tolist() == [False, False, False, False, True, True]


def test_add_var_boolean_rejects_empty_and_nomatch():
    a = DataAdaptor("x.h5ad", adata=_adata())
    with pytest.raises(ValueError):
        a.add_var_boolean_column("x", "", match_mode="prefix")
    with pytest.raises(ValueError):
        a.add_var_boolean_column("x", "ZZZ", match_mode="prefix")


def test_add_var_boolean_rejects_nonbool_name_collision():
    ad = _adata()
    ad.var["mean_counts"] = np.arange(6, dtype=float)  # numeric column
    a = DataAdaptor("x.h5ad", adata=ad)
    with pytest.raises(ValueError):
        a.add_var_boolean_column("mean_counts", "mt-", match_mode="prefix")


def test_route_add_var_boolean(monkeypatch):
    a = DataAdaptor("x.h5ad", adata=_adata())
    monkeypatch.setattr(routes, "get_adaptor", lambda dataset=None: a)
    client = TestClient(app)
    resp = client.post("/api/scanpy/add_var_boolean", json={
        "name": "mt", "pattern": "mt-", "match_mode": "prefix",
    })
    assert resp.status_code == 200, resp.text
    assert resp.json()["n_genes_matched"] == 2


def test_calculate_qc_metrics_adds_columns():
    a = DataAdaptor("x.h5ad", adata=_adata())
    a.add_var_boolean_column("mt", "mt-", match_mode="prefix")
    r = a.run_calculate_qc_metrics(qc_vars=["mt"], log1p=True)
    obs = a.adata.obs.columns
    assert "total_counts" in obs
    assert "n_genes_by_counts" in obs
    assert "log1p_total_counts" in obs
    assert "pct_counts_mt" in obs
    assert "total_counts_mt" in obs
    assert r["qc_vars"] == ["mt"]


def test_calculate_qc_metrics_log1p_false():
    a = DataAdaptor("x.h5ad", adata=_adata())
    a.run_calculate_qc_metrics(qc_vars=None, log1p=False)
    assert "total_counts" in a.adata.obs.columns
    assert "log1p_total_counts" not in a.adata.obs.columns


def test_calculate_qc_metrics_rejects_non_boolean_qc_var():
    ad = _adata()
    ad.var["mean_counts"] = np.arange(6, dtype=float)
    a = DataAdaptor("x.h5ad", adata=ad)
    with pytest.raises(ValueError):
        a.run_calculate_qc_metrics(qc_vars=["mean_counts"])


def test_calculate_qc_metrics_accepts_comma_string():
    a = DataAdaptor("x.h5ad", adata=_adata())
    a.add_var_boolean_column("mt", "mt-", match_mode="prefix")
    r = a.run_calculate_qc_metrics(qc_vars="mt", log1p=True)
    assert "pct_counts_mt" in a.adata.obs.columns
    assert r["qc_vars"] == ["mt"]


def test_route_calculate_qc_metrics(monkeypatch):
    a = DataAdaptor("x.h5ad", adata=_adata())
    a.add_var_boolean_column("mt", "mt-", match_mode="prefix")
    monkeypatch.setattr(routes, "get_adaptor", lambda dataset=None: a)
    client = TestClient(app)
    resp = client.post("/api/scanpy/calculate_qc_metrics", json={
        "qc_vars": "mt", "log1p": True,
    })
    assert resp.status_code == 200, resp.text
    assert "pct_counts_mt" in a.adata.obs.columns
