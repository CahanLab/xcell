"""Route tests for /api/scanpy/neighborhood run + result."""

import time

import anndata
import numpy as np
import pandas as pd
from fastapi.testclient import TestClient
from scipy.sparse import csr_matrix

from xcell.main import app
from xcell.api import routes
from xcell.adaptor import DataAdaptor


def _install():
    xs, ys = np.meshgrid(np.arange(8, dtype=float), np.arange(8, dtype=float))
    ab = np.column_stack([xs.ravel(), ys.ravel()])
    c = ab + np.array([1000.0, 0.0])
    coords = np.vstack([ab, c])
    parity = (ab[:, 0] + ab[:, 1]) % 2
    labels = ["A" if p == 0 else "B" for p in parity] + ["C"] * len(c)
    rng = np.random.default_rng(0)
    X = csr_matrix(rng.poisson(1.0, size=(len(coords), 4)).astype(np.float32))
    ad = anndata.AnnData(X=X)
    ad.obs["cell_type"] = pd.Categorical(labels)
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


def test_result_is_null_before_any_run():
    _install()
    c = TestClient(app)
    r = c.get("/api/scanpy/neighborhood/result")
    assert r.status_code == 200
    assert r.json() is None


def test_run_and_result_roundtrip():
    _install()
    c = TestClient(app)
    r = c.post("/api/scanpy/neighborhood/run",
               json={"column": "cell_type", "n_neighs": 6, "n_perms": 80})
    assert r.status_code == 202, r.text
    task = _poll(c, r.json()["task_id"])
    assert task["status"] == "completed", task
    result = task["result"]
    assert result["categories"] == ["A", "B", "C"]
    z = np.array(result["zscores"])
    assert z[0, 1] > 2.0

    stored = c.get("/api/scanpy/neighborhood/result").json()
    assert stored is not None
    assert stored["categories"] == ["A", "B", "C"]
    assert stored["params"]["column"] == "cell_type"


def test_unknown_column_is_400():
    _install()
    c = TestClient(app)
    r = c.post("/api/scanpy/neighborhood/run", json={"column": "nope"})
    assert r.status_code == 400
    assert "nope" in r.json()["detail"]


def test_radius_mode_without_radius_is_400():
    _install()
    c = TestClient(app)
    r = c.post("/api/scanpy/neighborhood/run",
               json={"column": "cell_type", "mode": "radius"})
    assert r.status_code == 400
