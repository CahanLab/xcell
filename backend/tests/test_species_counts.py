"""Per-species count summation by gene-name pattern + species assignment."""
import numpy as np
import anndata
import pytest
from scipy.sparse import csr_matrix
from fastapi.testclient import TestClient

from xcell.adaptor import DataAdaptor
from xcell.main import app
from xcell.api import routes


def _adata():
    # 4 human (GRCh38_) + 3 mouse (mm10___) genes; integer-ish counts in .X.
    genes = ["GRCh38_A1BG", "GRCh38_TP53", "GRCh38_EGFR", "GRCh38_MYC",
             "mm10___Xkr4", "mm10___Sox2", "mm10___Actb"]
    rng = np.random.default_rng(1)
    X = rng.integers(0, 20, size=(10, 7)).astype(np.float32)
    ad = anndata.AnnData(X=csr_matrix(X))
    ad.var_names = genes
    return ad, X


def test_sum_counts_prefix_sums_matching_genes():
    ad, X = _adata()
    a = DataAdaptor("x.h5ad", adata=ad)
    r = a.sum_counts_by_pattern("GRCh38_", match_mode="prefix")
    assert r["n_genes_matched"] == 4
    assert r["obs_name"] == "GRCh38_counts"
    expected = X[:, :4].sum(axis=1)
    assert np.allclose(a.adata.obs["GRCh38_counts"].to_numpy(), expected)


def test_sum_counts_regex():
    ad, X = _adata()
    a = DataAdaptor("x.h5ad", adata=ad)
    r = a.sum_counts_by_pattern("^mm10", match_mode="regex", obs_name="mouse")
    assert r["n_genes_matched"] == 3
    assert np.allclose(a.adata.obs["mouse"].to_numpy(), X[:, 4:].sum(axis=1))


def test_sum_counts_no_match_raises():
    ad, _ = _adata()
    a = DataAdaptor("x.h5ad", adata=ad)
    with pytest.raises(ValueError):
        a.sum_counts_by_pattern("ZZZ_", match_mode="prefix")


def test_assign_species_threshold():
    ad, _ = _adata()
    a = DataAdaptor("x.h5ad", adata=ad)
    # Plant explicit count columns: cell0 human-pure, cell1 mouse-pure, cell2 mixed, cell3 empty.
    a.adata.obs["human_counts"] = np.array([100.0, 1, 50] + [10] * 7)
    a.adata.obs["mouse_counts"] = np.array([0.0, 100, 50] + [10] * 7)
    a.adata.obs.loc[a.adata.obs.index[3], "human_counts"] = 0.0
    a.adata.obs.loc[a.adata.obs.index[3], "mouse_counts"] = 0.0
    r = a.assign_species(["human_counts", "mouse_counts"], threshold=0.9)
    sp = a.adata.obs[r["obs_name"]].astype(str).to_numpy()
    assert sp[0] == "human"
    assert sp[1] == "mouse"
    assert sp[2] == "mixed"
    assert sp[3] == "unassigned"
    assert r["obs_name"] == "species"


def test_assign_species_requires_two_columns():
    ad, _ = _adata()
    a = DataAdaptor("x.h5ad", adata=ad)
    a.adata.obs["human_counts"] = np.ones(10)
    with pytest.raises(ValueError):
        a.assign_species(["human_counts"])


def test_route_sum_counts(monkeypatch):
    ad, _ = _adata()
    a = DataAdaptor("x.h5ad", adata=ad)
    monkeypatch.setattr(routes, "get_adaptor", lambda dataset=None: a)
    client = TestClient(app)
    resp = client.post("/api/scanpy/sum_counts_by_pattern", json={
        "pattern": "GRCh38_", "match_mode": "prefix",
    })
    assert resp.status_code == 200, resp.text
    assert resp.json()["n_genes_matched"] == 4


def test_route_assign_species(monkeypatch):
    ad, _ = _adata()
    a = DataAdaptor("x.h5ad", adata=ad)
    a.adata.obs["human_counts"] = np.array([100.0] + [10] * 9)
    a.adata.obs["mouse_counts"] = np.array([0.0] + [10] * 9)
    monkeypatch.setattr(routes, "get_adaptor", lambda dataset=None: a)
    client = TestClient(app)
    resp = client.post("/api/scanpy/assign_species", json={
        "count_columns": "human_counts, mouse_counts", "threshold": 0.9,
    })
    assert resp.status_code == 200, resp.text
    assert "counts" in resp.json()
