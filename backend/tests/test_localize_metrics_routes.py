"""Route contract for scoring predicted Localize maps."""

import numpy as np
import anndata
from fastapi.testclient import TestClient
from scipy.sparse import csr_matrix

from xcell.main import app
from xcell.api import routes
from xcell.adaptor import DataAdaptor

client = TestClient(app)


def _spatial_ref(n_side=20):
    coords = np.array([[float(i), float(j)]
                       for i in range(n_side) for j in range(n_side)])
    n = len(coords)
    rng = np.random.default_rng(0)
    X = rng.poisson(1.0, (n, 4)).astype(np.float32)
    r = np.linalg.norm(coords - coords.mean(0), axis=1)
    X[r > np.quantile(r, 0.75), 0] += 20.0          # 'rim' gene on the outside
    ad = anndata.AnnData(X=csr_matrix(X))
    ad.var_names = ['Rim', 'B', 'C', 'D']
    ad.obsm['X_spatial'] = coords
    return ad


def _query(n=200):
    rng = np.random.default_rng(1)
    X = rng.poisson(1.0, (n, 4)).astype(np.float32)
    X[: n // 4, 0] += 20.0                          # a quarter are 'rim' cells
    ad = anndata.AnnData(X=csr_matrix(X))
    ad.var_names = ['Rim', 'B', 'C', 'D']
    ad.obsm['X_good'] = np.column_stack([
        rng.random(n) * 19, rng.random(n) * 19,     # spread over the tissue
    ])
    ad.obsm['X_collapsed'] = np.full((n, 2), 9.5) + rng.normal(0, 0.3, (n, 2))
    return ad


def _install():
    routes.set_adaptor(DataAdaptor('ref.h5ad', adata=_spatial_ref()), slot='primary')
    routes.set_adaptor(DataAdaptor('qry.h5ad', adata=_query()), slot='secondary')


def _body(embeddings):
    return {'reference': 'primary', 'dataset': 'secondary',
            'embeddings': embeddings, 'gene_sets': {'rim': ['Rim']}}


def test_evaluate_map_scores_each_embedding():
    _install()
    r = client.post('/api/localize/evaluate_map?dataset=secondary',
                    json=_body(['X_good', 'X_collapsed']))
    assert r.status_code == 200, r.text
    body = r.json()
    assert [m['embedding'] for m in body['maps']] == ['X_good', 'X_collapsed']
    for m in body['maps']:
        assert set(m) >= {'embedding', 'dispersion', 'occupancy', 'markers'}


def test_a_collapsed_map_scores_a_lower_area_ratio():
    _install()
    r = client.post('/api/localize/evaluate_map?dataset=secondary',
                    json=_body(['X_good', 'X_collapsed'])).json()
    good, bad = r['maps'][0]['dispersion'], r['maps'][1]['dispersion']
    assert bad['area_ratio'] < good['area_ratio']


def test_unknown_embedding_is_a_400():
    _install()
    r = client.post('/api/localize/evaluate_map?dataset=secondary',
                    json=_body(['X_nope']))
    assert r.status_code == 400
    assert 'X_nope' in r.json()['detail']


def test_a_gene_set_with_no_usable_genes_is_reported_not_fatal():
    _install()
    r = client.post('/api/localize/evaluate_map?dataset=secondary', json={
        'reference': 'primary', 'dataset': 'secondary', 'embeddings': ['X_good'],
        'gene_sets': {'rim': ['Rim'], 'nonsense': ['NotAGene']},
    })
    assert r.status_code == 200, r.text
    assert r.json()['skipped_gene_sets'] == ['nonsense']
    assert [m['name'] for m in r.json()['maps'][0]['markers']] == ['rim']


def test_response_is_json_safe():
    import json
    _install()
    r = client.post('/api/localize/evaluate_map?dataset=secondary',
                    json=_body(['X_good']))
    json.dumps(r.json())
