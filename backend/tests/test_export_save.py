import json

import anndata as ad
import numpy as np
import pandas as pd
from fastapi.testclient import TestClient
from scipy.sparse import csr_matrix

from xcell.adaptor import DataAdaptor
from xcell.api import routes
from xcell.main import app


def _install(monkeypatch):
    rng = np.random.default_rng(0)
    X = csr_matrix(rng.poisson(1.0, size=(12, 6)).astype(np.float32))
    a = ad.AnnData(
        X=X,
        obs=pd.DataFrame({'cell_type': ['a', 'b'] * 6},
                         index=[f'c{i}' for i in range(12)]),
        var=pd.DataFrame(index=[f'g{i}' for i in range(6)]),
    )
    adaptor = DataAdaptor('x.h5ad', adata=a)
    monkeypatch.setattr(routes, 'get_adaptor', lambda dataset=None: adaptor)
    return adaptor


def test_saves_an_h5ad_that_reads_back(monkeypatch, tmp_path):
    _install(monkeypatch)
    out = tmp_path / 'exported.h5ad'
    r = TestClient(app).post('/api/export/save',
                             json={'path': str(out), 'kind': 'h5ad'})
    assert r.status_code == 200, r.text
    assert out.exists()
    body = r.json()
    assert body['n_bytes'] > 0 and body['path'] == str(out)
    back = ad.read_h5ad(out)
    assert back.shape == (12, 6)
    assert 'cell_type' in back.obs.columns


def test_saves_metadata_as_tsv(monkeypatch, tmp_path):
    _install(monkeypatch)
    out = tmp_path / 'meta.tsv'
    r = TestClient(app).post('/api/export/save',
                             json={'path': str(out), 'kind': 'metadata'})
    assert r.status_code == 200, r.text
    text = out.read_text()
    assert '\t' in text and 'cell_type' in text


def test_saves_gene_sets_from_client_content(monkeypatch, tmp_path):
    # The sets are flattened in the browser, so the client supplies the exact
    # bytes rather than the server reproducing that shape.
    _install(monkeypatch)
    out = tmp_path / 'sets.json'
    payload = json.dumps([{'name': 'skin', 'genes': ['Krt5']}])
    r = TestClient(app).post('/api/export/save',
                             json={'path': str(out), 'kind': 'gene_sets',
                                   'content': payload})
    assert r.status_code == 200, r.text
    assert json.loads(out.read_text())[0]['name'] == 'skin'


def test_gene_sets_without_content_is_400(monkeypatch, tmp_path):
    _install(monkeypatch)
    r = TestClient(app).post('/api/export/save',
                             json={'path': str(tmp_path / 's.json'),
                                   'kind': 'gene_sets'})
    assert r.status_code == 400
    assert 'content' in r.json()['detail'].lower()


def test_a_missing_directory_is_400_naming_it(monkeypatch, tmp_path):
    _install(monkeypatch)
    bad = tmp_path / 'nope' / 'x.h5ad'
    r = TestClient(app).post('/api/export/save',
                             json={'path': str(bad), 'kind': 'h5ad'})
    assert r.status_code == 400
    assert 'nope' in r.json()['detail']


def test_a_directory_as_the_path_is_400(monkeypatch, tmp_path):
    _install(monkeypatch)
    r = TestClient(app).post('/api/export/save',
                             json={'path': str(tmp_path), 'kind': 'h5ad'})
    assert r.status_code == 400
    assert 'directory' in r.json()['detail'].lower()


def test_an_unknown_kind_is_400_listing_the_valid_ones(monkeypatch, tmp_path):
    _install(monkeypatch)
    r = TestClient(app).post('/api/export/save',
                             json={'path': str(tmp_path / 'x'), 'kind': 'nonsense'})
    assert r.status_code == 400
    assert 'h5ad' in r.json()['detail']


def test_overwriting_replaces_rather_than_appends(monkeypatch, tmp_path):
    _install(monkeypatch)
    out = tmp_path / 'meta.tsv'
    out.write_text('stale content that must not survive\n')
    r = TestClient(app).post('/api/export/save',
                             json={'path': str(out), 'kind': 'metadata'})
    assert r.status_code == 200
    assert 'stale content' not in out.read_text()


def test_browse_kinds_cover_the_new_shapes():
    assert set(routes.BROWSE_KINDS) >= {'data', 'classifier', 'tabular', 'geneset'}
