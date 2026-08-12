import anndata as ad
import numpy as np
import pandas as pd
from fastapi.testclient import TestClient
from scipy.sparse import csr_matrix

from xcell.adaptor import DataAdaptor
from xcell.api import routes
from xcell.main import app

MOUSE = ['ENSMUSG00000002603', 'ENSMUSG00000051951', 'ENSMUSG00000000001']


def _install(monkeypatch, names):
    rng = np.random.default_rng(0)
    X = csr_matrix(rng.poisson(1.0, size=(10, len(names))).astype(np.float32))
    a = ad.AnnData(X=X, obs=pd.DataFrame(index=[f'c{i}' for i in range(10)]),
                   var=pd.DataFrame(index=pd.Index(list(names))))
    adaptor = DataAdaptor('x.h5ad', adata=a)
    # The lambda must accept dataset=None: every data-touching route passes it.
    monkeypatch.setattr(routes, 'get_adaptor', lambda dataset=None: adaptor)
    return adaptor


def test_preview_route_reports_the_species(monkeypatch):
    _install(monkeypatch, MOUSE)
    r = TestClient(app).get('/api/var/symbol_mapping_preview')
    assert r.status_code == 200
    body = r.json()
    assert body['species'] == 'mouse'
    assert body['applicable'] is True


def test_preview_route_mutates_nothing(monkeypatch):
    adaptor = _install(monkeypatch, MOUSE)
    TestClient(app).get('/api/var/symbol_mapping_preview')
    assert 'gene_symbol' not in adaptor.adata.var.columns
    assert list(adaptor.adata.var_names) == MOUSE


def test_map_route_writes_the_column(monkeypatch):
    adaptor = _install(monkeypatch, MOUSE)
    r = TestClient(app).post('/api/var/map_symbols', json={})
    assert r.status_code == 200
    assert r.json()['n_mapped'] >= 2
    assert 'gene_symbol' in adaptor.adata.var.columns
    assert list(adaptor.adata.var_names) == MOUSE       # index untouched


def test_map_route_can_promote_to_the_index(monkeypatch):
    adaptor = _install(monkeypatch, MOUSE)
    r = TestClient(app).post('/api/var/map_symbols', json={'set_as_index': True})
    assert r.status_code == 200
    assert 'Tgfb1' in list(adaptor.adata.var_names)


def test_map_route_maps_a_bad_species_to_400(monkeypatch):
    _install(monkeypatch, ['ENSDARG00000000001', 'ENSDARG00000000002'])
    r = TestClient(app).post('/api/var/map_symbols', json={})
    assert r.status_code == 400
    assert 'zebrafish' in r.json()['detail']


def test_map_route_maps_already_symbols_to_400(monkeypatch):
    _install(monkeypatch, ['Tgfb1', 'Xkr4', 'Shh'])
    r = TestClient(app).post('/api/var/map_symbols', json={})
    assert r.status_code == 400
    assert 'ensembl' in r.json()['detail'].lower()
