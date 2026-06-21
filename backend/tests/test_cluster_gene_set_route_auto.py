"""POST /cluster_gene_set accepts and forwards the auto-method params."""
import numpy as np
import anndata
from scipy.sparse import csr_matrix
from fastapi.testclient import TestClient

from xcell.main import app
from xcell.api import routes


def _adata():
    rng = np.random.default_rng(0)
    n_cells = 200
    f1, f2 = rng.standard_normal(n_cells), rng.standard_normal(n_cells)
    cols, names = [], []
    for i in range(8):
        cols.append(f1 + rng.standard_normal(n_cells) * 0.15); names.append(f"A{i}")
    for i in range(8):
        cols.append(f2 + rng.standard_normal(n_cells) * 0.15); names.append(f"B{i}")
    X = np.vstack(cols).T
    X = X - X.min()
    ad = anndata.AnnData(X=csr_matrix(X.astype(np.float32)))
    ad.var_names = names
    ad.layers["raw"] = ad.X.copy()
    return ad, names


def test_route_auto_method(monkeypatch):
    from xcell.adaptor import DataAdaptor
    ad, names = _adata()
    adaptor = DataAdaptor("x.h5ad", adata=ad)
    monkeypatch.setattr(routes, "get_adaptor", lambda dataset=None: adaptor)
    client = TestClient(app)
    resp = client.post("/api/cluster_gene_set", json={
        "gene_names": names,
        "method": "auto",
        "cell_context": "all",
        "metric": "pearson",
        "min_genes": 4,
        "merge_threshold": 0.8,
        "purity_threshold": 0.5,
        "max_split_depth": 2,
        "layer": "raw",
    })
    assert resp.status_code == 200, resp.text
    body = resp.json()
    # auto returns modules-only clusters + a separate unassigned group + diagnostics
    assert "unassigned" in body and "diagnostics" in body
    flat = [g for c in body["clusters"] for g in c] + list(body["unassigned"])
    assert sorted(flat) == sorted(names)
    diag = body["diagnostics"]
    assert len(diag["module_coherence"]) == len(body["clusters"])


def test_route_forwards_auto_params(monkeypatch):
    """The auto knobs in the request body must reach the report call."""
    from xcell.adaptor import DataAdaptor
    ad, names = _adata()
    adaptor = DataAdaptor("x.h5ad", adata=ad)
    monkeypatch.setattr(routes, "get_adaptor", lambda dataset=None: adaptor)

    captured = {}
    real = adaptor.auto_coexpression_report

    def spy(*args, **kwargs):
        captured.update(kwargs)
        return real(*args, **kwargs)

    monkeypatch.setattr(adaptor, "auto_coexpression_report", spy)
    client = TestClient(app)
    resp = client.post("/api/cluster_gene_set", json={
        "gene_names": names,
        "method": "auto",
        "cell_context": "all",
        "metric": "spearman",
        "min_genes": 7,
        "merge_threshold": 0.66,
        "purity_threshold": 0.42,
        "max_split_depth": 3,
        "min_module_corr": 0.27,
        "layer": "raw",
    })
    assert resp.status_code == 200, resp.text
    assert captured.get("metric") == "spearman"
    assert captured.get("min_genes") == 7
    assert captured.get("merge_threshold") == 0.66
    assert captured.get("purity_threshold") == 0.42
    assert captured.get("max_split_depth") == 3
    assert captured.get("min_module_corr") == 0.27


def test_route_non_auto_unchanged(monkeypatch):
    """Non-auto methods keep the plain {clusters} response (no unassigned)."""
    from xcell.adaptor import DataAdaptor
    ad, names = _adata()
    adaptor = DataAdaptor("x.h5ad", adata=ad)
    monkeypatch.setattr(routes, "get_adaptor", lambda dataset=None: adaptor)
    client = TestClient(app)
    resp = client.post("/api/cluster_gene_set", json={
        "gene_names": names,
        "method": "kmeans",
        "k": 3,
        "cell_context": "all",
        "layer": "raw",
    })
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert "clusters" in body
    assert "unassigned" not in body
    assert "diagnostics" not in body
