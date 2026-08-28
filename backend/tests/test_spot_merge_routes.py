"""Route tests for POST /api/spots/merge.

The route's whole job is to leave the source alone and hand back a sibling
dataset, so that is what these check — plus that a preview really previews.
"""

import anndata
import numpy as np
import pandas as pd
from fastapi.testclient import TestClient
from scipy.sparse import csr_matrix

from xcell.main import app
from xcell.api import routes
from xcell.adaptor import DataAdaptor


def _install(n_side: int = 6, pitch: float = 15.0, scale: bool = True):
    for key in list(routes._adaptors):
        routes.remove_adaptor(key)
    rng = np.random.default_rng(0)
    xs, ys = np.meshgrid(np.arange(n_side) * pitch, np.arange(n_side) * pitch)
    coords = np.column_stack([xs.ravel(), ys.ravel()]).astype(float)
    counts = (rng.poisson(5.0, size=(len(coords), 8)) + 1).astype(np.float32)
    ad = anndata.AnnData(X=csr_matrix(counts))
    ad.var_names = [f"g{i}" for i in range(8)]
    ad.layers["counts"] = csr_matrix(counts)
    ad.obs["cell_type"] = pd.Categorical(["hyper"] * len(coords))
    ad.obsm["spatial"] = coords
    if scale:
        ad.uns["xcell_spatial_scale"] = {"um_per_unit": 1.0}
    routes.set_adaptor(DataAdaptor("src.h5ad", adata=ad), slot="primary")
    return np.where((coords[:, 0] < 40) & (coords[:, 1] < 40))[0].tolist()


def test_dry_run_reports_without_creating_a_slot():
    region = _install()
    c = TestClient(app)
    r = c.post("/api/spots/merge", json={"region_indices": region, "dry_run": True})
    assert r.status_code == 200, r.text
    assert r.json()["n_merged_spots"] < len(region)
    assert list(c.get("/api/datasets").json()) == ["primary"]


def test_committing_creates_the_sibling_dataset():
    region = _install()
    c = TestClient(app)
    r = c.post("/api/spots/merge", json={
        "region_indices": region, "dry_run": False, "slot": "slot3",
    })
    assert r.status_code == 200, r.text
    assert r.json()["slot"] == "slot3"
    assert set(c.get("/api/datasets").json()) == {"primary", "slot3"}
    assert c.get("/api/schema?dataset=slot3").json()["n_cells"] < 36


def test_the_source_is_left_alone():
    region = _install()
    c = TestClient(app)
    c.post("/api/spots/merge", json={
        "region_indices": region, "dry_run": False, "slot": "slot3",
    })
    assert c.get("/api/schema").json()["n_cells"] == 36


def test_committing_without_a_slot_is_a_400():
    region = _install()
    r = TestClient(app).post("/api/spots/merge", json={
        "region_indices": region, "dry_run": False,
    })
    assert r.status_code == 400
    assert "slot" in r.json()["detail"].lower()


def test_no_region_is_a_400():
    _install()
    r = TestClient(app).post("/api/spots/merge", json={"region_indices": []})
    assert r.status_code == 400
    assert "at least two" in r.json()["detail"]


def test_missing_spatial_scale_is_a_400_that_says_so():
    region = _install(scale=False)
    r = TestClient(app).post("/api/spots/merge", json={"region_indices": region})
    assert r.status_code == 400
    assert "coordinate unit" in r.json()["detail"]


def test_internal_labels_do_not_leak_into_the_response():
    region = _install()
    body = TestClient(app).post(
        "/api/spots/merge", json={"region_indices": region}
    ).json()
    assert "_labels" not in body


def test_a_gene_set_narrows_the_veto():
    region = _install()
    r = TestClient(app).post("/api/spots/merge", json={
        "region_indices": region, "gene_subset": ["g0", "g1", "g2"],
    })
    assert r.status_code == 200, r.text
    assert r.json()["gene_subset"]["n_genes"] == 3
