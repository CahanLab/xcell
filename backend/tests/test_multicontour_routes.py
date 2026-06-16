"""Route tests for /api/scanpy/multicontour prepare + finalize."""

import time

import numpy as np
import anndata
from scipy.sparse import csr_matrix
from fastapi.testclient import TestClient

from xcell.main import app
from xcell.api import routes
from xcell.adaptor import DataAdaptor


def _install_adaptor():
    coords = np.array([[float(i), float(j)] for i in range(5) for j in range(5)])
    n = coords.shape[0]
    X = np.zeros((n, 4), dtype=np.float32)
    for k, (i, j) in enumerate([(i, j) for i in range(5) for j in range(5)]):
        if i <= 1:
            X[k, 0] = X[k, 1] = 10.0
        if i >= 3:
            X[k, 2] = X[k, 3] = 10.0
    ad = anndata.AnnData(X=csr_matrix(X))
    ad.var_names = ["a0", "a1", "b0", "b1"]
    ad.obsm["spatial"] = coords
    ad.obsm["X_pca"] = coords.copy()
    routes.set_adaptor(DataAdaptor("x.h5ad", adata=ad), slot="primary")


def _poll(client, task_id, timeout=10.0):
    deadline = time.time() + timeout
    while time.time() < deadline:
        r = client.get(f"/api/tasks/{task_id}").json()
        if r["status"] in ("completed", "error", "cancelled"):
            return r
        time.sleep(0.05)
    raise AssertionError("task did not finish")


def test_prepare_and_finalize_roundtrip():
    _install_adaptor()
    c = TestClient(app)
    pr = c.post("/api/scanpy/multicontour/prepare", json={
        "gene_sets": {"A": ["a0", "a1"], "B": ["b0", "b1"]},
        "contour_levels": 3, "grid_res": 40, "smooth_sigma": 2.0})
    assert pr.status_code == 202
    task = _poll(c, pr.json()["task_id"])
    assert task["status"] == "completed", task
    body = task["result"]
    cutoffs = {m["name"]: m["auto_cutoff"] for m in body["modules"]}

    fr = c.post("/api/scanpy/multicontour/finalize", json={
        "token": body["token"], "cutoffs": cutoffs, "profile_k": 5,
        "out_name": "tissue", "save_qc": False, "params": body["params"]})
    assert fr.status_code == 200, fr.text
    assert fr.json()["annotation_key"] == "tissue"
    assert "unassigned" in fr.json()["categories"]


def test_prepare_missing_pca_returns_400():
    coords = np.array([[float(i), float(j)] for i in range(4) for j in range(4)])
    ad = anndata.AnnData(X=csr_matrix(np.ones((coords.shape[0], 2), dtype=np.float32)))
    ad.var_names = ["a0", "a1"]
    ad.obsm["spatial"] = coords
    routes.set_adaptor(DataAdaptor("x.h5ad", adata=ad), slot="primary")
    c = TestClient(app)
    pr = c.post("/api/scanpy/multicontour/prepare", json={
        "gene_sets": {"A": ["a0"], "B": ["a1"]}, "contour_levels": 3})
    assert pr.status_code == 400
    assert "PCA" in pr.json()["detail"]
