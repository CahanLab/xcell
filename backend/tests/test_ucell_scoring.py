"""UCell-style directional gene-set scoring."""
import numpy as np
import anndata
import pytest
import scipy.sparse as sp
from fastapi.testclient import TestClient

from xcell.adaptor import DataAdaptor
from xcell.main import app
from xcell.api import routes


def _adata():
    # 2 cells x 4 genes, hand-chosen so descending ranks are exact.
    # Cell0: A=10,B=5,C=1,D=0 -> ranks A=1,B=2,C=3,D=4
    # Cell1: A=0,B=1,C=5,D=10 -> ranks D=1,C=2,B=3,A=4
    X = np.array([[10, 5, 1, 0],
                  [0, 1, 5, 10]], dtype=np.float32)
    ad = anndata.AnnData(X=sp.csr_matrix(X))
    ad.var_names = ["A", "B", "C", "D"]
    return ad


def test_ucell_ranks_caps_to_n_genes_and_caches():
    a = DataAdaptor("x.h5ad", adata=_adata())
    ranks = a._ucell_ranks("X", 1500)   # maxRank auto-capped to n_genes=4
    assert sp.issparse(ranks)
    assert ranks.shape == (2, 4)
    # rank>=maxRank(4) dropped to 0; cell0 keeps A=1,B=2,C=3, drops D
    dense = ranks.toarray()
    assert dense[0, 0] == 1 and dense[0, 1] == 2 and dense[0, 2] == 3
    assert dense[0, 3] == 0   # D rank 4 == maxRank -> dropped
    # identical call returns the SAME cached object
    assert a._ucell_ranks("X", 1500) is ranks


def test_ucell_up_only_scores():
    a = DataAdaptor("x.h5ad", adata=_adata())
    r = a.ucell_score_values(["A", "B"], [], layer="X", max_rank=1500, w_neg=1.0)
    # Cell0: ranks A=1,B=2 -> u=1.0 ; Cell1: ranks A=4,B=3 -> u=0.2
    assert np.allclose(r["values"], [1.0, 0.2])
    assert r["n_up_used"] == 2 and r["n_down_used"] == 0


def test_ucell_up_and_down_subtracts_and_clips():
    a = DataAdaptor("x.h5ad", adata=_adata())
    r = a.ucell_score_values(["A", "B"], ["D"], layer="X", max_rank=1500, w_neg=1.0)
    # u_p=[1.0,0.2]; u_n(D)=[0,1]; max(u_p-u_n,0)=[1.0, 0.0]
    assert np.allclose(r["values"], [1.0, 0.0])
    assert r["n_down_used"] == 1


def test_ucell_down_only_is_zero():
    a = DataAdaptor("x.h5ad", adata=_adata())
    r = a.ucell_score_values([], ["D"], layer="X", max_rank=1500, w_neg=1.0)
    assert np.allclose(r["values"], [0.0, 0.0])
    assert r["n_up_used"] == 0


def test_ucell_skips_missing_genes():
    a = DataAdaptor("x.h5ad", adata=_adata())
    r = a.ucell_score_values(["A", "B", "ZZZ"], [], layer="X", max_rank=1500)
    assert r["n_up_used"] == 2          # ZZZ filtered out
    assert np.allclose(r["values"], [1.0, 0.2])


def test_score_gene_sets_writes_obs_columns():
    a = DataAdaptor("x.h5ad", adata=_adata())
    out = a.score_gene_sets_ucell(
        [{"name": "Sig A", "up": ["A", "B"], "down": ["D"]},
         {"name": "Sig C", "up": ["C", "D"]}],
        layer="X", max_rank=1500, w_neg=1.0,
    )
    cols = {r["name"]: r for r in out["results"]}
    assert "UCell_Sig_A" in a.adata.obs.columns
    assert cols["Sig A"]["obs_column"] == "UCell_Sig_A"
    assert np.allclose(a.adata.obs["UCell_Sig_A"].to_numpy(), [1.0, 0.0])
    assert cols["Sig A"]["n_up_used"] == 2 and cols["Sig A"]["n_down_used"] == 1


def test_score_gene_sets_skips_down_only_set():
    a = DataAdaptor("x.h5ad", adata=_adata())
    out = a.score_gene_sets_ucell([{"name": "DownOnly", "up": [], "down": ["D"]}])
    r = out["results"][0]
    assert r.get("skipped")
    assert "UCell_DownOnly" not in a.adata.obs.columns


def test_score_gene_sets_obs_name_collision_suffixes():
    ad = _adata()
    ad.obs["UCell_Dup"] = [9.0, 9.0]
    a = DataAdaptor("x.h5ad", adata=ad)
    out = a.score_gene_sets_ucell([{"name": "Dup", "up": ["A"]}])
    assert out["results"][0]["obs_column"] == "UCell_Dup_1"
    assert "UCell_Dup_1" in a.adata.obs.columns


def test_route_score_genes_ucell(monkeypatch):
    a = DataAdaptor("x.h5ad", adata=_adata())
    monkeypatch.setattr(routes, "get_adaptor", lambda dataset=None: a)
    client = TestClient(app)
    resp = client.post("/api/scanpy/score_genes_ucell", json={
        "sets": [{"name": "Sig A", "up": ["A", "B"], "down": ["D"]}],
        "layer": "X", "max_rank": 1500, "w_neg": 1.0,
    })
    assert resp.status_code == 200
    body = resp.json()
    assert body["results"][0]["obs_column"] == "UCell_Sig_A"
    assert "UCell_Sig_A" in a.adata.obs.columns


def test_route_expression_ucell(monkeypatch):
    a = DataAdaptor("x.h5ad", adata=_adata())
    monkeypatch.setattr(routes, "get_adaptor", lambda dataset=None: a)
    client = TestClient(app)
    resp = client.post("/api/expression/ucell", json={
        "up": ["A", "B"], "down": ["D"], "layer": "X",
    })
    assert resp.status_code == 200
    body = resp.json()
    assert np.allclose(body["values"], [1.0, 0.0])
    # interactive endpoint must NOT persist a column
    assert not any(c.startswith("UCell_") for c in a.adata.obs.columns)
