"""The crosstab route: slot resolution and how bad input is reported."""
import anndata
import numpy as np
import pandas as pd
from fastapi.testclient import TestClient
from scipy.sparse import csr_matrix

from xcell.adaptor import DataAdaptor
from xcell.api import routes
from xcell.main import app


def _adaptor(obs: pd.DataFrame) -> DataAdaptor:
    rng = np.random.default_rng(0)
    ad = anndata.AnnData(
        X=csr_matrix(rng.poisson(1.0, size=(len(obs), 4)).astype(np.float32)),
        obs=obs,
        var=pd.DataFrame(index=[f'g{i}' for i in range(4)]),
    )
    return DataAdaptor("x.h5ad", adata=ad)


def _install(monkeypatch, adaptor):
    monkeypatch.setattr(routes, "get_adaptor", lambda dataset=None: adaptor)


OBS = pd.DataFrame({
    'leiden': pd.Categorical(['0', '0', '1']),
    'territory': pd.Categorical(['prox', 'dist', 'dist']),
    'n_counts': [1.0, 2.0, 3.0],
})


def test_returns_the_table(monkeypatch):
    _install(monkeypatch, _adaptor(OBS))
    r = TestClient(app).get("/api/obs/crosstab?a=leiden&b=territory")
    assert r.status_code == 200
    body = r.json()
    assert body['a_categories'] == ['0', '1']
    assert body['counts'] == [[1, 1], [1, 0]]


def test_missing_column_is_a_404(monkeypatch):
    _install(monkeypatch, _adaptor(OBS))
    r = TestClient(app).get("/api/obs/crosstab?a=leiden&b=nope")
    assert r.status_code == 404


def test_continuous_column_is_a_400_that_says_what_to_do(monkeypatch):
    _install(monkeypatch, _adaptor(OBS))
    r = TestClient(app).get("/api/obs/crosstab?a=leiden&b=n_counts")
    assert r.status_code == 400
    assert 'continuous' in r.json()['detail']


def test_reads_the_slot_it_was_asked_about(monkeypatch):
    seen = {}
    def fake(dataset=None):
        seen['dataset'] = dataset
        return _adaptor(OBS)
    monkeypatch.setattr(routes, "get_adaptor", fake)
    TestClient(app).get("/api/obs/crosstab?a=leiden&b=territory&dataset=slot3")
    assert seen['dataset'] == 'slot3'
