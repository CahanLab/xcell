"""Route tests for /api/figures/*."""
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
    grp = np.array(['a'] * 30 + ['b'] * 30)
    lam = np.full((grp.size, len(GENES)), 1.0)
    for g in ('Col1a1', 'Col1a2', 'Col3a1'):
        lam[grp == 'a', GENES.index(g)] = 8.0
    for g in ('Ptprc', 'Cd3e', 'Cd19'):
        lam[grp == 'b', GENES.index(g)] = 8.0
    ad = anndata.AnnData(X=csr_matrix(rng.poisson(lam).astype(np.float32)))
    ad.var_names = GENES
    ad.obs['grp'] = pd.Categorical(grp)
    ad.obs['batch'] = pd.Categorical(['x', 'y'] * 30)
    routes.set_adaptor(DataAdaptor('x.h5ad', adata=ad), slot='primary')


def _poll(client, task_id, timeout=30.0):
    deadline = time.time() + timeout
    while time.time() < deadline:
        s = client.get(f'/api/tasks/{task_id}').json()
        if s['status'] in ('completed', 'failed', 'cancelled'):
            return s
        time.sleep(0.05)
    raise AssertionError('task did not finish')


def _batch(c):
    r = c.post('/api/enrichment/gsea_batch', json={'obs_column': 'grp', 'libraries': [{'source': 'msigdb', 'id': 'toy'}],
                                                   'n_perm': 40, 'min_set_size': 2})
    assert r.status_code == 202, r.text
    s = _poll(c, r.json()['task_id'])
    assert s['status'] == 'completed', s
    return s['result']


def test_figures_crud_data_and_409_after_input_deleted():
    _install()
    c = TestClient(app)
    col = _batch(c)
    r = c.post('/api/figures', json={'kind': 'enrichment_heatmap', 'inputs': {'enrichment_keys': [col['key']]},
                                     'params': {'padj_max': 1.0}})
    assert r.status_code == 200, r.text
    fig = r.json()
    assert fig['id'] == 'fig_1' and fig['params']['padj_max'] == 1.0 and fig['provenance']['created_step'] is not None
    assert [f['id'] for f in c.get('/api/figures').json()['figures']] == ['fig_1']
    assert c.get('/api/figures/fig_1').json()['kind'] == 'enrichment_heatmap'
    d = c.post('/api/figures/fig_1/data', json={}).json()
    assert [x['label'] for x in d['cols']] == ['a', 'b'] and d['rows']
    d2 = c.post('/api/figures/fig_1/data', json={'params': {'top_n': 1}}).json()
    assert len(d2['rows']) <= len(d['rows'])
    assert c.post('/api/figures/fig_1/data', json={'params': {'bogus': 1}}).status_code == 400
    u = c.put('/api/figures/fig_1', json={'title': 'T', 'params': {'top_n': 2}})
    assert u.status_code == 200 and u.json()['title'] == 'T' and u.json()['params']['top_n'] == 2
    n = c.post('/api/figures', json={'kind': 'enrichment_network', 'inputs': {'enrichment_keys': [col['key']]}, 'params': {'padj_max': 1.0}})
    assert n.status_code == 200 and c.post(f"/api/figures/{n.json()['id']}/data", json={}).json()['nodes']
    assert c.post('/api/figures', json={'kind': 'pie', 'inputs': {}}).status_code == 400
    assert c.post('/api/figures', json={'kind': 'enrichment_heatmap', 'inputs': {'enrichment_keys': ['ghost']}}).status_code == 400
    assert c.get('/api/figures/fig_99').status_code == 404
    # attach → record figure carries figure_id
    a = c.post('/api/figures/fig_1/attach', json={'png_b64': 'iVBORw0KGgo=', 'caption': 'Fig 1'})
    assert a.status_code == 200, a.text
    rec = c.get('/api/record').json()
    assert [f for f in rec['figures'] if f['figure_id'] == 'fig_1']
    assert c.post('/api/figures/fig_1/attach', json={'png_b64': '***'}).status_code == 400
    # deleting the input → 409 on data, figure still listed
    assert c.delete(f"/api/enrichment/results/{col['key']}").status_code == 200
    r409 = c.post('/api/figures/fig_1/data', json={})
    assert r409.status_code == 409 and col['key'] in r409.json()['detail']
    assert 'fig_1' in [f['id'] for f in c.get('/api/figures').json()['figures']]
    assert c.delete('/api/figures/fig_1').json() == {'deleted': 'fig_1'}
    assert c.delete('/api/figures/fig_1').status_code == 404
