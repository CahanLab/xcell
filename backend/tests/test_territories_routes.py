"""Territory routes. The faces endpoint is deliberately stateless: it is called
on every edit while drawing, and must not be able to observe or mutate a
dataset mid-drag."""
import anndata
import numpy as np
import pandas as pd
from fastapi.testclient import TestClient
from scipy.sparse import csr_matrix

from xcell.adaptor import DataAdaptor
from xcell.api import routes
from xcell.main import app

client = TestClient(app)
SQUARE = [[0.0, 0.0], [10.0, 0.0], [10.0, 10.0], [0.0, 10.0]]


def _install(n=40):
    rng = np.random.default_rng(0)
    ad = anndata.AnnData(X=csr_matrix(rng.poisson(1.0, (n, 5)).astype(np.float32)))
    ad.var_names = [f"g{i}" for i in range(5)]
    ad.obs_names = [f"c{i}" for i in range(n)]
    ad.obsm["spatial"] = np.column_stack([rng.uniform(0.5, 9.5, n),
                                          rng.uniform(0.5, 9.5, n)])
    ad.obs["section"] = pd.Categorical(["A"] * n)
    a = DataAdaptor("t.h5ad", adata=ad)
    routes.set_adaptor(a, slot="primary")
    return a


def _payload():
    return {"embedding": "spatial", "section_col": "section", "source": "drawn",
            "sections": {"A": {"ring": SQUARE,
                               "cuts": [{"id": "c1", "points": [[1.0, 5.0], [9.0, 5.0]],
                                         "closed": False}],
                               "anchors": [{"name": "proximal", "x": 5.0, "y": 8.0},
                                           {"name": "distal", "x": 5.0, "y": 2.0}]}}}


def test_the_faces_preview_needs_no_dataset():
    resp = client.post("/api/territories/faces", json={
        "ring": SQUARE,
        "cuts": [{"id": "c1", "points": [[1.0, 5.0], [9.0, 5.0]], "closed": False}],
        "anchors": [{"name": "proximal", "x": 5.0, "y": 8.0}],
    })
    assert resp.status_code == 200
    faces = resp.json()["faces"]
    assert len(faces) == 2
    assert faces[0]["name"] == "proximal"
    assert faces[1]["name"] is None
    assert faces[0]["polygon"][0] == faces[0]["polygon"][-1]  # closed ring


def test_the_faces_preview_reports_area_as_a_finite_number():
    resp = client.post("/api/territories/faces",
                       json={"ring": SQUARE, "cuts": [], "anchors": []})
    area = resp.json()["faces"][0]["area"]
    assert isinstance(area, float) and np.isfinite(area)


def test_the_faces_preview_returns_an_anchor_inside_each_face():
    """The panel names a face without doing any geometry of its own, so the
    point it would drop an anchor on has to come back with the face."""
    resp = client.post("/api/territories/faces", json={
        "ring": SQUARE,
        "cuts": [{"id": "c1", "points": [[1.0, 5.0], [9.0, 5.0]], "closed": False}],
        "anchors": [],
    })
    faces = resp.json()["faces"]
    assert faces[0]["anchor"][1] > 5.0
    assert faces[1]["anchor"][1] < 5.0


def test_a_ring_with_too_few_points_is_a_400():
    resp = client.post("/api/territories/faces",
                       json={"ring": [[0.0, 0.0]], "cuts": [], "anchors": []})
    assert resp.status_code == 400


def test_put_then_get_round_trips_a_type():
    _install()
    assert client.put("/api/territories/prox-dist", json=_payload()).status_code == 200
    got = client.get("/api/territories").json()["territories"]
    assert list(got) == ["prox-dist"]


def test_put_with_an_unknown_embedding_is_a_400():
    _install()
    payload = _payload()
    payload["embedding"] = "X_nope"
    resp = client.put("/api/territories/prox-dist", json=payload)
    assert resp.status_code == 400
    assert "X_nope" in resp.json()["detail"]


def test_deleting_an_unknown_type_is_a_404():
    _install()
    assert client.delete("/api/territories/nope").status_code == 404


def test_assign_writes_the_column_and_reports_counts():
    a = _install()
    client.put("/api/territories/prox-dist", json=_payload())
    resp = client.post("/api/territories/assign",
                       json={"types": ["prox-dist"], "combine": False})
    assert resp.status_code == 200
    assert "territory_prox-dist" in a.adata.obs.columns
    assert resp.json()["types"]["prox-dist"]["counts"]


def test_assigning_an_unknown_type_is_a_404():
    _install()
    resp = client.post("/api/territories/assign", json={"types": ["nope"]})
    assert resp.status_code == 404
