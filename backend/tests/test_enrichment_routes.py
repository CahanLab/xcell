"""Route tests for /api/enrichment/*."""
import time

import anndata
import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient
from scipy.sparse import csr_matrix

from xcell import gene_set_sources as gss
from xcell.adaptor import DataAdaptor
from xcell.api import routes
from xcell.main import app

GENES = ['Col1a1', 'Col1a2', 'Col3a1', 'Ptprc', 'Cd3e', 'Cd19'] + [f'G{i}' for i in range(14)]


@pytest.fixture(autouse=True)
def _cache(tmp_path, monkeypatch):
    gss.set_cache_dir(tmp_path / 'cache')
    monkeypatch.setattr(gss, 'fetch_text', lambda url, **kw: (_ for _ in ()).throw(AssertionError(url)))
    monkeypatch.setattr(gss, 'fetch_json', lambda url, **kw: (_ for _ in ()).throw(AssertionError(url)))
    gss.save_library({'source': 'msigdb', 'id': 'toy', 'name': 'Toy', 'species': 'mouse', 'version': '1', 'n_sets': 2,
                      'sets': [{'name': 'COLLAGEN', 'genes': ['Col1a1', 'Col1a2', 'Col3a1', 'G0', 'G1']},
                               {'name': 'IMMUNE', 'genes': ['Ptprc', 'Cd3e', 'Cd19', 'G2', 'G3']}]})
    yield
    gss.set_cache_dir(None)


def _install():
    rng = np.random.default_rng(0)
    n = 60
    lam = np.full((n, len(GENES)), 1.0)
    grp = np.array(['a'] * 30 + ['b'] * 30)
    for g in ('Col1a1', 'Col1a2', 'Col3a1'):
        lam[grp == 'a', GENES.index(g)] = 8.0
    ad = anndata.AnnData(X=csr_matrix(rng.poisson(lam).astype(np.float32)))
    ad.var_names = GENES
    ad.var['hv'] = [True] * 10 + [False] * 10
    ad.var['panel'] = [True] * 20
    ad.obs['grp'] = pd.Categorical(grp)
    routes.set_adaptor(DataAdaptor('x.h5ad', adata=ad), slot='primary')


def _poll(client, task_id, timeout=30.0):
    deadline = time.time() + timeout
    while time.time() < deadline:
        s = client.get(f'/api/tasks/{task_id}').json()
        if s['status'] in ('completed', 'failed', 'cancelled'):
            return s
        time.sleep(0.05)
    raise AssertionError('task did not finish')


def test_overlap_route_accepts_dict_gene_subset_and_returns_result():
    _install()
    c = TestClient(app)
    body = {'genes': ['Col1a1', 'Col1a2', 'Col3a1'], 'name': 'q',
            'libraries': [{'source': 'msigdb', 'id': 'toy'}],
            'gene_subset': {'columns': ['hv', 'panel'], 'operation': 'intersection'},
            'min_set_size': 2, 'min_overlap': 1}
    r = c.post('/api/enrichment/overlap', json=body)
    assert r.status_code == 200, r.text
    d = r.json()
    assert d['universe_size'] == 10 and d['results'][0]['name'] == 'COLLAGEN'
    assert d['gene_subset_type'].startswith('intersection')
    lst = c.get('/api/enrichment/results').json()['results']
    assert lst[0]['key'] == d['key'] and lst[0]['kind'] == 'ora'
    assert c.get(f"/api/enrichment/results/{d['key']}").json()['results'][0]['name'] == 'COLLAGEN'
    assert c.delete(f"/api/enrichment/results/{d['key']}").json() == {'deleted': d['key'], 'also_deleted': []}
    assert c.get(f"/api/enrichment/results/{d['key']}").status_code == 404


def test_overlap_route_error_mapping():
    _install()
    c = TestClient(app)
    r = c.post('/api/enrichment/overlap', json={'genes': ['Col1a1', 'Col1a2'],
                                                'libraries': [{'source': 'msigdb', 'id': 'missing'}]})
    assert r.status_code == 400 and 'not cached' in r.json()['detail']
    r = c.post('/api/enrichment/overlap', json={'genes': ['Col1a1', 'Col1a2']})
    assert r.status_code == 400


def test_gsea_route_runs_as_task():
    _install()
    c = TestClient(app)
    body = {'ranking': {'kind': 'diffexp', 'obs_column': 'grp', 'group': 'a', 'reference': 'rest'},
            'libraries': [{'source': 'msigdb', 'id': 'toy'}], 'n_perm': 50, 'min_set_size': 2}
    r = c.post('/api/enrichment/gsea', json=body)
    assert r.status_code == 202, r.text
    s = _poll(c, r.json()['task_id'])
    assert s['status'] == 'completed', s
    res = s['result']
    assert res['kind'] == 'gsea' and res['results']
    assert res['ranking']['genes'][0] in ('Col1a1', 'Col1a2', 'Col3a1')
    bad = dict(body, ranking={'kind': 'diffexp', 'obs_column': 'grp', 'group': 'zzz'})
    assert c.post('/api/enrichment/gsea', json=bad).status_code == 400
    ghost = dict(body, ranking={'kind': 'diffexp', 'obs_column': 'grp', 'group': 'a', 'cell_subset': 'ghost'})
    assert c.post('/api/enrichment/gsea', json=ghost).status_code == 404


def test_gsea_batch_route_and_collection_get():
    _install()
    c = TestClient(app)
    body = {'obs_column': 'grp', 'libraries': [{'source': 'msigdb', 'id': 'toy'}], 'n_perm': 40, 'min_set_size': 2}
    r = c.post('/api/enrichment/gsea_batch', json=body)
    assert r.status_code == 202, r.text
    s = _poll(c, r.json()['task_id'])
    assert s['status'] == 'completed', s
    col = s['result']
    assert col['kind'] == 'gsea_batch' and set(col['members']) == {'a', 'b'}
    full = c.get(f"/api/enrichment/results/{col['key']}").json()
    assert set(full['member_results']) == {'a', 'b'}
    listed = [x for x in c.get('/api/enrichment/results').json()['results'] if x['key'] == col['key']]
    assert listed[0]['n_groups'] == 2
    assert c.post('/api/enrichment/gsea_batch', json=dict(body, obs_column='nope')).status_code == 400


def test_ora_batch_route():
    _install()
    c = TestClient(app)
    r = c.post('/api/enrichment/ora_batch', json={
        'obs_column': 'grp', 'top_n': 5, 'min_overlap': 1, 'min_set_size': 2,
        'libraries': [{'source': 'msigdb', 'id': 'toy'}],
        'gene_subset': {'columns': ['panel'], 'operation': 'union'}})
    assert r.status_code == 200, r.text
    d = r.json()
    assert d['kind'] == 'ora_batch' and 'a' in d['markers'] and d['members']['a'].startswith('ora_grp_a')
