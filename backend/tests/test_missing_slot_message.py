"""A 503 for a slot that isn't there should say what *is*.

"No data loaded for slot 'primary'" is true and useless when two datasets are
loaded under other names — it reads as "nothing is loaded", which is what sent
2026-08-25's debugging down the wrong path.
"""

import anndata
import numpy as np
from fastapi.testclient import TestClient
from scipy.sparse import csr_matrix

from xcell.main import app
from xcell.api import routes
from xcell.adaptor import DataAdaptor


def _install(*slots):
    for key in list(routes._adaptors):
        routes.remove_adaptor(key)
    rng = np.random.default_rng(0)
    for slot in slots:
        ad = anndata.AnnData(X=csr_matrix(rng.random((6, 4), dtype=np.float32)))
        ad.var_names = [f"g{i}" for i in range(4)]
        routes.set_adaptor(DataAdaptor(f"{slot}.h5ad", adata=ad), slot=slot)


def test_names_the_slots_that_are_loaded():
    _install("slot3", "slot4")
    r = TestClient(app).get("/api/schema")
    assert r.status_code == 503
    detail = r.json()["detail"]
    assert "primary" in detail
    assert "slot3" in detail and "slot4" in detail


def test_says_nothing_is_loaded_when_that_is_true():
    _install()
    r = TestClient(app).get("/api/schema")
    assert r.status_code == 503
    detail = r.json()["detail"].lower()
    assert "no data" in detail
    assert "slot3" not in detail


def test_a_named_slot_that_is_absent_says_so_too():
    _install("slot3")
    r = TestClient(app).get("/api/schema?dataset=nope")
    assert r.status_code == 503
    assert "nope" in r.json()["detail"]
    assert "slot3" in r.json()["detail"]
