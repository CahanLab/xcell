"""/localize/suggest must answer "what can act as a reference" unconditionally.

Reported 2026-08-25: with four datasets loaded and the first-loaded one closed,
Localize said "No dataset is loaded" while two healthy datasets were in the
registry. The route resolved the *query* adaptor on its first line, and an
unnamed query means the literal slot 'primary' — which is exactly the key that
gets freed when you close the dataset you loaded first.
"""

import anndata
import numpy as np
from fastapi.testclient import TestClient
from scipy.sparse import csr_matrix

from xcell.main import app
from xcell.api import routes
from xcell.adaptor import DataAdaptor


def _adata(spatial: bool):
    rng = np.random.default_rng(0)
    n_cells, n_genes = 40, 8
    X = csr_matrix(rng.integers(0, 5, size=(n_cells, n_genes)).astype(np.float32))
    ad = anndata.AnnData(X=X)
    ad.var_names = [f"g{i}" for i in range(n_genes)]
    if spatial:
        ad.obsm["X_spatial"] = rng.normal(size=(n_cells, 2))
    return ad


def _install(slots):
    """Load the named slots, and only those."""
    for key in list(routes._adaptors):
        routes.remove_adaptor(key)
    for slot, spatial in slots:
        routes.set_adaptor(DataAdaptor("x.h5ad", adata=_adata(spatial)), slot=slot)


TWO_PAIRS = [("primary", True), ("secondary", False), ("slot3", True), ("slot4", False)]


def test_lists_every_loaded_dataset():
    _install(TWO_PAIRS)
    c = TestClient(app)
    r = c.get("/api/localize/suggest?dataset=slot4")
    assert r.status_code == 200, r.text
    assert [s["slot"] for s in r.json()["references"]] == ["primary", "secondary", "slot3", "slot4"]
    assert [s["has_spatial"] for s in r.json()["references"]] == [True, False, True, False]


def test_answers_without_a_query_slot_named():
    """The unconditional first call names no dataset — that must not 503."""
    _install(TWO_PAIRS)
    c = TestClient(app)
    r = c.get("/api/localize/suggest")
    assert r.status_code == 200, r.text
    assert len(r.json()["references"]) == 4


def test_answers_after_the_primary_slot_is_closed():
    """Closing the first-loaded dataset frees 'primary'. The slot list survives."""
    _install(TWO_PAIRS)
    c = TestClient(app)
    assert c.delete("/api/datasets/primary").status_code == 200
    assert c.delete("/api/datasets/secondary").status_code == 200

    r = c.get("/api/localize/suggest")
    assert r.status_code == 200, r.text
    assert [s["slot"] for s in r.json()["references"]] == ["slot3", "slot4"]


def test_suggested_k_is_omitted_rather_than_guessed_without_a_query():
    _install(TWO_PAIRS)
    c = TestClient(app)
    c.delete("/api/datasets/primary")
    r = c.get("/api/localize/suggest")
    assert r.status_code == 200, r.text
    body = r.json()
    # No query resolved, so there is no n to take a square root of. Saying
    # nothing beats returning a number derived from an arbitrary dataset.
    assert body.get("suggested_k") is None

    named = c.get("/api/localize/suggest?dataset=slot4").json()
    assert named["suggested_k"] == max(5, min(50, round(40 ** 0.5)))


def test_overlap_still_needs_both_sides():
    _install(TWO_PAIRS)
    c = TestClient(app)
    body = c.get("/api/localize/suggest?reference=slot3&dataset=slot4").json()
    assert "overlap" in body
    assert body["overlap"]["n_shared"] == 8


def test_naming_a_reference_without_a_query_is_a_400_not_a_503():
    """A reference is only meaningful against a query — say so plainly."""
    _install(TWO_PAIRS)
    c = TestClient(app)
    c.delete("/api/datasets/primary")
    r = c.get("/api/localize/suggest?reference=slot3")
    assert r.status_code == 400, r.text
    assert "which dataset" in r.json()["detail"].lower()


def test_nothing_loaded_at_all_is_still_the_house_503():
    """The narrow fix: a *missing primary* must not kill the answer. An empty
    registry is a different thing and keeps the convention every route uses."""
    _install([])
    c = TestClient(app)
    assert c.get("/api/localize/suggest").status_code == 503
