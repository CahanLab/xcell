"""Route tests for /api/scanpy/ligrec suggest + prepare + finalize."""

import time

import numpy as np
import anndata
from scipy.sparse import csr_matrix
from fastapi.testclient import TestClient

from xcell.main import app
from xcell.api import routes
from xcell.adaptor import DataAdaptor


def _install():
    w = 30
    xs, ys = np.meshgrid(np.arange(w), np.arange(w))
    coords = np.column_stack([xs.ravel(), ys.ravel()]).astype(float)
    c = np.array([15.0, 15.0])
    d = np.linalg.norm(coords - c, axis=1)
    L = np.where(d <= 2.0, 20.0, 0.0)
    Rnear = np.where((d > 2.0) & (d <= 3.7), 20.0, 0.0)
    rng = np.random.default_rng(0)
    N = rng.poisson(1.0, coords.shape[0]).astype(float)
    X = np.column_stack([L, Rnear, N]).astype(np.float32)
    ad = anndata.AnnData(X=csr_matrix(X))
    # Use real CellPhoneDB gene symbols so the shipped database matches a pair.
    ad.var_names = ["ADCYAP1", "ADCYAP1R1", "GAPDH"]
    ad.obsm["spatial"] = coords
    routes.set_adaptor(DataAdaptor("x.h5ad", adata=ad), slot="primary")


def _poll(client, task_id, timeout=20.0):
    deadline = time.time() + timeout
    while time.time() < deadline:
        r = client.get(f"/api/tasks/{task_id}").json()
        if r["status"] in ("completed", "error", "cancelled"):
            return r
        time.sleep(0.05)
    raise AssertionError("task did not finish")


def test_suggest_returns_radius():
    _install()
    c = TestClient(app)
    r = c.get("/api/scanpy/ligrec/suggest")
    assert r.status_code == 200, r.text
    assert r.json()["radius"] > 0


def test_prepare_and_finalize_roundtrip():
    _install()
    c = TestClient(app)
    pr = c.post("/api/scanpy/ligrec/prepare", json={
        "radius": 4.0, "n_perm": 80, "min_cells": 2,
    })
    assert pr.status_code == 202, pr.text
    task = _poll(c, pr.json()["task_id"])
    assert task["status"] == "completed", task
    body = task["result"]
    assert body["n_tested"] >= 1
    top = body["summary"][0]["interaction"]

    fr = c.post("/api/scanpy/ligrec/finalize", json={
        "interactions": [top], "write_significance": False,
    })
    assert fr.status_code == 200, fr.text
    assert fr.json()["annotation_key"].startswith("LR_")

    # The result persists for re-selection without re-running.
    rr = c.get("/api/scanpy/ligrec/result")
    assert rr.status_code == 200
    assert rr.json()["n_tested"] >= 1


def test_prepare_without_spatial_400():
    w = 5
    X = csr_matrix(np.ones((w * w, 3), dtype=np.float32))
    ad = anndata.AnnData(X=X)
    ad.var_names = ["L", "Rnear", "N"]
    routes.set_adaptor(DataAdaptor("x.h5ad", adata=ad), slot="primary")
    c = TestClient(app)
    r = c.post("/api/scanpy/ligrec/prepare", json={"n_perm": 10})
    assert r.status_code == 400
