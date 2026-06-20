"""Route test for POST /api/obs/transfer_labels."""

import numpy as np
import anndata
import pandas as pd
from scipy.sparse import csr_matrix
from fastapi.testclient import TestClient

from xcell.main import app
from xcell.api import routes
from xcell.adaptor import DataAdaptor


def _install():
    ad = anndata.AnnData(X=csr_matrix(np.ones((6, 2), dtype=np.float32)))
    ad.var_names = ["g0", "g1"]
    ad.obs["cell_type"] = pd.Categorical(["A", "A", "A", "A", "B", "B"])
    ad.obs["sub"] = pd.Categorical(
        ["0", "0", "1", "1", "unassigned", "unassigned"],
        categories=["0", "1", "unassigned"],
    )
    routes.set_adaptor(DataAdaptor("x.h5ad", adata=ad), slot="primary")


def test_transfer_labels_route_parent_prefix():
    _install()
    c = TestClient(app)
    r = c.post("/api/obs/transfer_labels", json={
        "target_column": "cell_type",
        "source_column": "sub",
        "out_column": "cell_type_refined",
        "rename_mode": "parent_prefix",
    })
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["n_overridden"] == 4
    assert body["n_kept"] == 2
    assert set(body["categories"]) == {"A.0", "A.1", "B"}


def test_transfer_labels_route_missing_column_404():
    _install()
    c = TestClient(app)
    r = c.post("/api/obs/transfer_labels", json={
        "target_column": "nope",
        "source_column": "sub",
        "out_column": "refined",
    })
    assert r.status_code == 404
