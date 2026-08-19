"""Route tests for /api/gene_nmf run + result + runs."""

import time

import anndata
import numpy as np
import pandas as pd
from fastapi.testclient import TestClient
from scipy.sparse import csr_matrix

from xcell.main import app
from xcell.api import routes
from xcell.adaptor import DataAdaptor


N_CELLS, N_GENES, K = 120, 24, 3
BLOCK = N_GENES // K


def _install():
    rng = np.random.default_rng(0)
    W = np.zeros((N_GENES, K))
    for j in range(K):
        W[j * BLOCK:(j + 1) * BLOCK, j] = rng.uniform(4.0, 9.0, BLOCK)
    H = rng.uniform(0.0, 0.2, size=(N_CELLS, K))
    dominant = rng.integers(0, K, N_CELLS)
    H[np.arange(N_CELLS), dominant] += rng.uniform(2.0, 6.0, N_CELLS)
    counts = rng.poisson(H @ W.T).astype(np.float32)

    ad = anndata.AnnData(X=csr_matrix(counts))
    ad.var_names = [f"G{i:03d}" for i in range(N_GENES)]
    ad.var["highly_variable"] = [True] * N_GENES
    ad.obs["group"] = pd.Categorical(["a" if d == 0 else "b" for d in dominant])
    routes.set_adaptor(DataAdaptor("x.h5ad", adata=ad), slot="primary")


def _poll(client, task_id, timeout=30.0):
    deadline = time.time() + timeout
    while time.time() < deadline:
        r = client.get(f"/api/tasks/{task_id}").json()
        if r["status"] in ("completed", "error", "cancelled"):
            return r
        time.sleep(0.05)
    raise AssertionError("task did not finish")


def _run(client, **body):
    body.setdefault("k", K)
    body.setdefault("weight_explained", 0.8)
    r = client.post("/api/gene_nmf/run", json=body)
    assert r.status_code == 202, r.text
    done = _poll(client, r.json()["task_id"])
    assert done["status"] == "completed", done
    return done["result"]


def test_result_is_null_before_any_run():
    _install()
    c = TestClient(app)
    r = c.get("/api/gene_nmf/result")
    assert r.status_code == 200
    assert r.json() is None


def test_run_returns_202_with_a_task_id():
    _install()
    c = TestClient(app)
    r = c.post("/api/gene_nmf/run", json={"k": K, "weight_explained": 0.8})
    assert r.status_code == 202
    assert "task_id" in r.json()


def test_run_and_result_roundtrip():
    _install()
    c = TestClient(app)
    result = _run(c)
    assert result["n_programs"] >= 1
    assert result["obsm_key"] == "NMF"

    stored = c.get("/api/gene_nmf/result").json()
    assert stored["key"] == "NMF"
    assert [p["name"] for p in stored["programs"]] == [
        p["name"] for p in result["programs"]
    ]


def test_programs_show_up_in_the_schema_as_a_score_matrix():
    _install()
    c = TestClient(app)
    result = _run(c)
    schema = c.get("/api/schema").json()
    assert schema["score_matrices"]["NMF"] == [
        p["name"] for p in result["programs"]
    ]


def test_runs_endpoint_lists_stored_keys():
    _install()
    c = TestClient(app)
    assert c.get("/api/gene_nmf/runs").json()["keys"] == []
    _run(c, key="NMF")
    _run(c, key="NMF_b", seed=2)
    assert c.get("/api/gene_nmf/runs").json()["keys"] == ["NMF", "NMF_b"]


def test_result_accepts_a_key_query_parameter():
    _install()
    c = TestClient(app)
    _run(c, key="alt", seed=5)
    assert c.get("/api/gene_nmf/result", params={"key": "alt"}).json()["key"] == "alt"
    assert c.get("/api/gene_nmf/result", params={"key": "nope"}).json() is None


def test_bad_k_is_a_400_not_a_failed_task():
    """Validation lives in prepare_*, so the caller learns immediately."""
    _install()
    c = TestClient(app)
    r = c.post("/api/gene_nmf/run", json={"k": 1})
    assert r.status_code == 400
    assert "k" in r.json()["detail"]


def test_unknown_gene_subset_column_is_a_400():
    _install()
    c = TestClient(app)
    r = c.post("/api/gene_nmf/run", json={"k": K, "gene_subset": "nope"})
    assert r.status_code == 400


def test_existing_key_without_overwrite_is_a_400():
    _install()
    c = TestClient(app)
    _run(c, key="NMF")
    r = c.post("/api/gene_nmf/run", json={"k": K, "key": "NMF"})
    assert r.status_code == 400
    assert "overwrite" in r.json()["detail"]


def test_annotation_cell_context_restricts_the_fit():
    _install()
    c = TestClient(app)
    result = _run(
        c,
        cell_context="annotation",
        annotation_column="group",
        annotation_values=["a"],
    )
    assert 0 < result["n_cells_used"] < N_CELLS


def test_selection_cell_context_requires_indices():
    _install()
    c = TestClient(app)
    r = c.post(
        "/api/gene_nmf/run", json={"k": K, "cell_context": "selection"}
    )
    assert r.status_code == 400
