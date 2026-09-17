"""Routes over external gene-set sources, the overlap check, and the species guess."""
import time

import numpy as np
import anndata
import pytest
from scipy.sparse import csr_matrix
from fastapi.testclient import TestClient

from xcell.main import app
from xcell.api import routes
from xcell.adaptor import DataAdaptor
from xcell import gene_set_sources as gss


STATS = {"statistics": [
    {"libraryName": "MSigDB_Hallmark_2020", "numTerms": 50, "geneCoverage": 4000, "genesPerTerm": 80, "link": "", "categoryId": 2},
]}
GMT = "Hallmark Apoptosis\t\tBAX,1.0\tBAK1,1.0\tCOL1A1,1.0\nHallmark Myogenesis\t\tMYOD1,1.0\tCOL1A1,1.0\n"


@pytest.fixture(autouse=True)
def _offline(tmp_path, monkeypatch):
    gss.set_cache_dir(tmp_path / 'cache')
    monkeypatch.setattr(gss, 'fetch_json', lambda url, **kw: STATS if 'datasetStatistics' in url else (_ for _ in ()).throw(AssertionError(url)))
    monkeypatch.setattr(gss, 'fetch_text', lambda url, **kw: GMT if 'geneSetLibrary' in url else (_ for _ in ()).throw(AssertionError(url)))
    yield
    gss.set_cache_dir(None)


def _adata():
    rng = np.random.default_rng(0)
    genes = ['Col1a1', 'Bax', 'Sox9', 'Runx2']
    ad = anndata.AnnData(X=csr_matrix(rng.poisson(1.0, size=(10, len(genes))).astype(np.float32)))
    ad.var_names = genes
    ad.var['highly_variable'] = [True, False, True, False]
    return ad


def _install(monkeypatch):
    a = DataAdaptor('x.h5ad', adata=_adata())
    monkeypatch.setattr(routes, 'get_adaptor', lambda dataset=None: a)
    return a


def _poll(client, task_id, timeout=10.0):
    t0 = time.time()
    while time.time() - t0 < timeout:
        st = client.get(f'/api/tasks/{task_id}').json()
        if st['status'] in ('completed', 'error', 'cancelled'):
            return st
        time.sleep(0.05)
    raise AssertionError('task did not finish')


def test_sources_endpoint_lists_sources_and_cache():
    client = TestClient(app)
    r = client.get('/api/gene_set_sources')
    assert r.status_code == 200
    body = r.json()
    assert set(body['sources']) >= {'msigdb', 'enrichr', 'string'}
    assert body['sources']['string']['kind'] == 'query'
    assert body['cached'] == []


def test_catalogue_route_and_bad_source():
    client = TestClient(app)
    r = client.get('/api/gene_set_sources/enrichr/libraries?species=human')
    assert r.status_code == 200
    assert r.json()['libraries'][0]['id'] == 'MSigDB_Hallmark_2020'
    assert r.json()['species'] == 'human'
    assert client.get('/api/gene_set_sources/nope/libraries?species=human').status_code == 400
    assert client.get('/api/gene_set_sources/enrichr/libraries?species=cat').status_code == 400


def test_fetch_route_runs_as_task_then_sets_are_searchable():
    client = TestClient(app)
    # Not fetched yet: searching is a 404 that says so.
    r = client.get('/api/gene_set_sources/enrichr/libraries/MSigDB_Hallmark_2020/sets?species=human')
    assert r.status_code == 404 and 'not been fetched' in r.json()['detail']

    r = client.post('/api/gene_set_sources/enrichr/libraries/MSigDB_Hallmark_2020/fetch', json={'species': 'human'})
    assert r.status_code == 202, r.text
    st = _poll(client, r.json()['task_id'])
    assert st['status'] == 'completed', st
    assert st['result']['n_sets'] == 2 and st['result']['id'] == 'MSigDB_Hallmark_2020'

    r = client.get('/api/gene_set_sources/enrichr/libraries/MSigDB_Hallmark_2020/sets?species=human&gene=col1a1')
    assert r.status_code == 200
    body = r.json()
    assert body['total'] == 2 and body['library']['name'] == 'MSigDB Hallmark 2020'
    r = client.get('/api/gene_set_sources/enrichr/libraries/MSigDB_Hallmark_2020/sets?species=human&q=myo')
    assert [s['name'] for s in r.json()['sets']] == ['Hallmark Myogenesis']
    # The catalogue now reports the library as cached.
    cat = client.get('/api/gene_set_sources/enrichr/libraries?species=human').json()['libraries']
    assert cat[0]['cached'] is True and cat[0]['n_sets'] == 2


def test_fetch_route_rejects_unknown_library_synchronously():
    client = TestClient(app)
    r = client.post('/api/gene_set_sources/msigdb/libraries/zz.nope/fetch', json={'species': 'mouse'})
    assert r.status_code == 400 and 'not a MSigDB collection' in r.json()['detail']


def test_overlap_route_uses_dataset_and_columns(monkeypatch):
    _install(monkeypatch)
    client = TestClient(app)
    r = client.post('/api/gene_sets/overlap', json={
        'sets': [{'name': 'apoptosis', 'genes': ['BAX', 'BAK1', 'COL1A1']}],
        'columns': ['highly_variable'],
    })
    assert r.status_code == 200, r.text
    s = r.json()['sets'][0]
    assert s['n_present'] == 2 and s['genes_resolved'] == ['Bax', 'Col1a1']
    assert s['columns'] == {'highly_variable': 1}
    assert client.post('/api/gene_sets/overlap', json={'sets': [], 'columns': ['nope']}).status_code == 400


def test_species_guess_route(monkeypatch):
    _install(monkeypatch)
    client = TestClient(app)
    r = client.get('/api/gene_sets/species_guess')
    assert r.status_code == 200 and r.json()['species'] == 'mouse'


def test_string_routes(monkeypatch):
    def fake_json(url, **kw):
        if 'get_string_ids' in url:
            return [{"queryItem": "Col1a1", "stringId": "10090.A", "preferredName": "Col1a1"}]
        if 'interaction_partners' in url:
            return [{"preferredName_A": "Col1a1", "preferredName_B": "Col3a1", "score": 0.99}]
        if 'network' in url:
            return [{"preferredName_A": "Col1a1", "preferredName_B": "Col3a1", "score": 0.99}]
        raise AssertionError(url)
    monkeypatch.setattr(gss, 'fetch_json', fake_json)
    client = TestClient(app)
    r = client.post('/api/gene_set_sources/string/partners', json={'genes': ['Col1a1'], 'species': 'mouse', 'limit': 5})
    assert r.status_code == 200, r.text
    assert r.json()['sets'][0]['genes'] == ['Col1a1', 'Col3a1']
    r = client.post('/api/gene_set_sources/string/network', json={'genes': ['Col1a1', 'Col3a1'], 'species': 'mouse'})
    assert r.status_code == 200 and len(r.json()['edges']) == 1
    r = client.post('/api/gene_set_sources/string/network', json={'genes': ['Col1a1'], 'species': 'cat'})
    assert r.status_code == 400
