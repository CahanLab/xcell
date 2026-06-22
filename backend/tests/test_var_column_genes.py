"""Resolve a boolean .var column to its gene names."""
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
    ad = anndata.AnnData(X=csr_matrix(rng.random((20, 6)).astype(np.float32)))
    ad.var_names = [f"g{i}" for i in range(6)]
    ad.var["highly_variable"] = [True, False, True, True, False, False]
    ad.var["means"] = rng.random(6)  # numeric, not boolean
    return ad


def test_column_to_gene_names_returns_true_genes_in_order():
    a = DataAdaptor("x.h5ad", adata=_adata())
    assert a.column_to_gene_names("highly_variable") == ["g0", "g2", "g3"]


def test_column_to_gene_names_rejects_missing_column():
    a = DataAdaptor("x.h5ad", adata=_adata())
    with pytest.raises(ValueError):
        a.column_to_gene_names("nope")


def test_column_to_gene_names_rejects_non_boolean_column():
    a = DataAdaptor("x.h5ad", adata=_adata())
    with pytest.raises(ValueError):
        a.column_to_gene_names("means")


def test_route_var_column_genes(monkeypatch):
    a = DataAdaptor("x.h5ad", adata=_adata())
    monkeypatch.setattr(routes, "get_adaptor", lambda dataset=None: a)
    client = TestClient(app)
    resp = client.get("/api/var/column_genes", params={"column": "highly_variable"})
    assert resp.status_code == 200, resp.text
    assert resp.json() == {"column": "highly_variable", "genes": ["g0", "g2", "g3"]}


def test_route_var_column_genes_bad_column(monkeypatch):
    a = DataAdaptor("x.h5ad", adata=_adata())
    monkeypatch.setattr(routes, "get_adaptor", lambda dataset=None: a)
    client = TestClient(app)
    resp = client.get("/api/var/column_genes", params={"column": "means"})
    assert resp.status_code == 400
