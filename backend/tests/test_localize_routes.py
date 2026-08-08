"""Route tests for /api/localize — the first routes resolving two slots."""
import json
import time

import anndata
import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient
from scipy.sparse import csr_matrix

from xcell.adaptor import DataAdaptor
from xcell.api import routes
from xcell.main import app


def _spatial(n=160, n_genes=24, seed=0):
    rng = np.random.default_rng(seed)
    coords = rng.uniform(0, 100, (n, 2))
    u = (coords[:, 0] - 50) / 50
    v = (coords[:, 1] - 50) / 50
    planes = rng.normal(0, 1, (3, n_genes))
    X = np.column_stack([u, v, np.ones(n)]) @ planes
    X = np.maximum(X + 3.0, 0.0) + rng.normal(0, 0.1, (n, n_genes))
    ad = anndata.AnnData(X=csr_matrix(np.maximum(X, 0).astype(np.float32)))
    ad.var_names = [f'g{i}' for i in range(n_genes)]
    ad.obs_names = [f'spot{i}' for i in range(n)]
    ad.obsm['spatial'] = coords
    ad.obs['region'] = pd.Categorical(np.where(coords[:, 0] < 50, 'left', 'right'))
    # The flags a real reference carries after highly_variable_genes /
    # spatial_autocorr — the two principled bases for a similarity gene set.
    ad.var['highly_variable'] = [i < 16 for i in range(n_genes)]
    ad.var['spatially_variable'] = [i < 12 for i in range(n_genes)]
    return ad, coords


def _query_from(ref_ad, coords, n=50, seed=1):
    rng = np.random.default_rng(seed)
    picks = rng.integers(0, ref_ad.n_obs, n)
    X = np.asarray(ref_ad.X.todense())[picks] * 3.0 + rng.normal(0, 0.2, (n, ref_ad.n_vars))
    ad = anndata.AnnData(X=csr_matrix(np.maximum(X, 0).astype(np.float32)))
    ad.var_names = list(ref_ad.var_names)
    ad.obs_names = [f'cell{i}' for i in range(n)]
    ad.obs['cell_type'] = pd.Categorical(['a', 'b'] * (n // 2))
    ad.obsm['spatial_true'] = coords[picks]
    return ad


def _install():
    """Spatial reference in `secondary`, scRNA-seq query in `primary`."""
    ref_ad, coords = _spatial()
    q_ad = _query_from(ref_ad, coords)
    routes.set_adaptor(DataAdaptor('scrna.h5ad', adata=q_ad), slot='primary')
    routes.set_adaptor(DataAdaptor('spatial.h5ad', adata=ref_ad), slot='secondary')
    return TestClient(app)


def _poll(client, task_id, timeout=30.0):
    deadline = time.time() + timeout
    while time.time() < deadline:
        body = client.get(f'/api/tasks/{task_id}').json()
        if body['status'] in ('completed', 'error', 'cancelled'):
            return body
        time.sleep(0.05)
    raise AssertionError('task did not finish')


def _run(client, **overrides):
    payload = {'reference': 'secondary', 'k': 10, **overrides}
    resp = client.post('/api/localize/prepare', json=payload)
    assert resp.status_code == 202, resp.text
    return _poll(client, resp.json()['task_id'])


# --- suggest --------------------------------------------------------------

def test_suggest_lists_which_slots_can_act_as_a_reference():
    client = _install()
    body = client.get('/api/localize/suggest').json()
    refs = {r['slot']: r for r in body['references']}
    assert refs['secondary']['has_spatial'] is True
    assert refs['secondary']['n_cells'] == 160


def test_suggest_marks_a_slot_without_coordinates():
    client = _install()
    refs = {r['slot']: r for r in client.get('/api/localize/suggest').json()['references']}
    # The query itself has no .obsm['spatial'] — only the answer key.
    assert refs['primary']['has_spatial'] is False


def test_suggest_previews_the_gene_overlap():
    """So the panel can warn before the user commits to a run."""
    client = _install()
    body = client.get('/api/localize/suggest?reference=secondary').json()
    assert body['overlap']['n_shared'] == 24
    assert body['overlap']['severity'] == 'ok'


def test_suggest_proposes_a_k():
    client = _install()
    assert client.get('/api/localize/suggest').json()['suggested_k'] >= 1


def test_suggest_without_data_is_a_503():
    routes.remove_adaptor('primary')
    routes.remove_adaptor('secondary')
    assert TestClient(app).get('/api/localize/suggest').status_code == 503


# --- prepare --------------------------------------------------------------

def test_a_projection_writes_an_embedding_into_the_query():
    client = _install()
    result = _run(client)
    assert result['status'] == 'completed', result
    assert result['result']['embedding_name'] == 'X_spatial_pred'
    assert 'X_spatial_pred' in client.get('/api/schema').json()['embeddings']


def test_the_prediction_is_close_to_the_truth_the_query_carries():
    client = _install()
    _run(client)
    a = routes.get_adaptor('primary')
    err = np.median(np.linalg.norm(
        a.adata.obsm['X_spatial_pred'] - a.adata.obsm['spatial_true'], axis=1))
    assert err < 20


def test_the_confidence_columns_are_visible_to_the_cell_manager():
    client = _install()
    _run(client)
    dtypes = client.get('/api/schema').json()['obs_dtypes']
    assert dtypes['X_spatial_pred_confidence'] == 'numeric'
    assert dtypes['X_spatial_pred_similarity'] == 'numeric'


def test_the_new_embedding_can_actually_be_fetched():
    client = _install()
    _run(client, min_confidence=0.9)
    resp = client.get('/api/embedding/X_spatial_pred')
    assert resp.status_code == 200, resp.text
    json.loads(resp.text)  # strict: unplaced cells must be null, not NaN


def test_a_section_column_is_honoured():
    client = _install()
    result = _run(client, section_col='region')
    assert result['result']['section_col'] == 'region'
    assert client.get('/api/schema').json()['obs_dtypes']['X_spatial_pred_section'] \
        == 'category'


def test_the_reference_slot_must_exist():
    client = _install()
    resp = client.post('/api/localize/prepare', json={'reference': 'tertiary'})
    assert resp.status_code in (400, 503), resp.text


def test_a_reference_without_coordinates_is_a_400():
    client = _install()
    resp = client.post('/api/localize/prepare', json={'reference': 'primary'})
    assert resp.status_code == 400
    assert 'spatial' in resp.json()['detail'].lower()


def test_localizing_a_dataset_onto_itself_is_refused():
    """It would return each cell's own position and look flawless."""
    client = _install()
    resp = client.post('/api/localize/prepare',
                       json={'reference': 'primary', 'dataset': 'primary'})
    assert resp.status_code == 400


def test_a_bad_parameter_is_a_400_not_a_failed_task():
    client = _install()
    resp = client.post('/api/localize/prepare',
                       json={'reference': 'secondary', 'metric': 'mahalanobis'})
    assert resp.status_code == 400


def test_too_little_gene_overlap_is_a_400():
    client = _install()
    a = routes.get_adaptor('primary')
    routes.set_adaptor(DataAdaptor('q.h5ad', adata=a.adata[:, :4].copy()), slot='primary')
    resp = client.post('/api/localize/prepare', json={'reference': 'secondary'})
    assert resp.status_code == 400
    assert 'overlap' in resp.json()['detail'].lower()


def test_two_runs_can_coexist_under_different_keys():
    client = _install()
    _run(client, k=5, key_added='X_k5')
    _run(client, k=25, key_added='X_k25')
    embeddings = set(client.get('/api/schema').json()['embeddings'])
    assert {'X_k5', 'X_k25'} <= embeddings


# --- cross-validation -----------------------------------------------------

def test_cross_validation_scores_the_reference_against_itself():
    client = _install()
    resp = client.post('/api/localize/cross_validate',
                       json={'reference': 'secondary', 'k': 10})
    assert resp.status_code == 202, resp.text
    body = _poll(client, resp.json()['task_id'])
    assert body['status'] == 'completed', body
    metrics = body['result']
    assert metrics['same_platform'] is True
    assert metrics['median_error'] < metrics['baseline_centroid']


def test_cross_validation_can_group_by_an_obs_column():
    client = _install()
    resp = client.post('/api/localize/cross_validate',
                       json={'reference': 'secondary', 'k': 10, 'groupby': 'region'})
    metrics = _poll(client, resp.json()['task_id'])['result']
    assert set(metrics['per_group']) == {'left', 'right'}


def test_cross_validation_writes_nothing():
    """It is a measurement, not an analysis step."""
    client = _install()
    before = set(client.get('/api/schema').json()['embeddings'])
    resp = client.post('/api/localize/cross_validate',
                       json={'reference': 'secondary', 'k': 10})
    _poll(client, resp.json()['task_id'])
    assert set(client.get('/api/schema').json()['embeddings']) == before


# --- evaluate -------------------------------------------------------------

def test_a_prediction_can_be_scored_against_stored_truth():
    client = _install()
    _run(client)
    resp = client.post('/api/localize/evaluate', json={
        'predicted_key': 'X_spatial_pred',
        'truth_key': 'spatial_true',
        'groupby': 'cell_type',
    })
    assert resp.status_code == 200, resp.text
    metrics = resp.json()
    assert metrics['median_error'] < metrics['baseline_centroid']
    assert set(metrics['per_group']) == {'a', 'b'}
    json.dumps(metrics, allow_nan=False)


def test_scoring_against_a_missing_key_is_a_404():
    client = _install()
    _run(client)
    resp = client.post('/api/localize/evaluate', json={
        'predicted_key': 'X_spatial_pred', 'truth_key': 'nope',
    })
    assert resp.status_code == 404


# --- choosing the genes that similarity is computed over ------------------

def test_suggest_lists_the_reference_gene_flags():
    """SVG / HVG live on the reference — it owns which genes carry position."""
    client = _install()
    body = client.get('/api/localize/suggest?reference=secondary').json()
    columns = {c['name']: c['n_true'] for c in body['reference_gene_columns']}
    assert columns['spatially_variable'] == 12
    assert columns['highly_variable'] == 16


def test_suggest_recomputes_the_overlap_for_a_gene_subset():
    """Narrowing the basis changes how many genes are actually shared, so the
    preview has to follow the choice rather than describe the whole panel."""
    client = _install()
    full = client.get('/api/localize/suggest?reference=secondary').json()
    svg = client.get(
        '/api/localize/suggest?reference=secondary&gene_subset=spatially_variable'
    ).json()
    assert full['overlap']['n_shared'] == 24
    assert svg['overlap']['n_shared'] == 12


def test_a_gene_flag_can_be_the_similarity_basis():
    client = _install()
    result = _run(client, gene_subset='highly_variable')
    assert result['status'] == 'completed', result
    assert result['result']['n_shared_genes'] == 16
    assert result['result']['gene_subset_type'] == 'column:highly_variable'


def test_an_explicit_gene_list_can_be_the_similarity_basis():
    """A set curated in the Gene panel, passed through as names."""
    client = _install()
    genes = [f'g{i}' for i in range(12)]
    result = _run(client, gene_subset=genes)
    assert result['status'] == 'completed', result
    assert result['result']['n_shared_genes'] == 12


def test_a_gene_list_naming_nothing_real_is_a_400():
    client = _install()
    resp = client.post('/api/localize/prepare', json={
        'reference': 'secondary', 'gene_subset': ['nosuchgene1', 'nosuchgene2'],
    })
    assert resp.status_code == 400


def test_a_basis_too_narrow_to_predict_from_is_a_400():
    """Twelve spatially-variable genes is fine; a two-gene basis is not."""
    client = _install()
    resp = client.post('/api/localize/prepare', json={
        'reference': 'secondary', 'gene_subset': ['g0', 'g1'],
    })
    assert resp.status_code == 400
    assert 'overlap' in resp.json()['detail'].lower()


def test_the_basis_is_recorded_in_the_analysis_record():
    client = _install()
    _run(client, gene_subset='spatially_variable')
    step = routes.get_adaptor('primary').analysis_record.steps[-1]
    assert step.action == 'localize'
    assert step.params['gene_subset_type'] == 'column:spatially_variable'


def test_cross_validation_honours_the_same_basis():
    """Otherwise the accuracy check measures different parameters than the run."""
    client = _install()
    resp = client.post('/api/localize/cross_validate', json={
        'reference': 'secondary', 'k': 10, 'gene_subset': 'highly_variable',
    })
    assert resp.status_code == 202, resp.text
    body = _poll(client, resp.json()['task_id'])
    assert body['status'] == 'completed', body
