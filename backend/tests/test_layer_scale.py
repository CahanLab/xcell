"""Layer scale detection — "is this matrix raw counts or already normalized?"

The user-facing question these tests pin down: given an arbitrary .h5ad, which
of adata.X / adata.layers[*] hold raw counts, which hold library-size
normalized values, and which have been log-transformed or z-scored. The answer
has to be derivable from the numbers themselves, because provenance is usually
missing from files that arrive from collaborators.
"""
import numpy as np
import anndata
import pandas as pd
import pytest
from scipy.sparse import csr_matrix
from fastapi.testclient import TestClient

from xcell.layer_scale import assess_matrix_scale, SCALE_LABELS
from xcell.adaptor import DataAdaptor
from xcell.main import app
from xcell.api import routes


# --------------------------------------------------------------------------
# Matrix builders — each produces a matrix with a known, unambiguous scale.
# --------------------------------------------------------------------------

def _counts(n_obs=60, n_var=40, seed=0):
    """Raw UMI counts: non-negative integers, variable library size."""
    rng = np.random.default_rng(seed)
    # Poisson with a per-cell size factor so library sizes genuinely vary.
    size = rng.uniform(0.5, 4.0, size=(n_obs, 1))
    return rng.poisson(size * 3.0, size=(n_obs, n_var)).astype(np.float32)


def _cpm(target=1e4, **kw):
    """normalize_total: floats whose row sums are all exactly `target`."""
    X = _counts(**kw)
    sums = X.sum(axis=1, keepdims=True)
    sums[sums == 0] = 1.0
    return (X / sums * target).astype(np.float32)


def _lognorm(target=1e4, **kw):
    """log1p(normalize_total(...)) — the scanpy default expression scale."""
    return np.log1p(_cpm(target=target, **kw)).astype(np.float32)


def _log_counts(**kw):
    """log1p of raw counts — log scale, but library size was never equalized."""
    return np.log1p(_counts(**kw)).astype(np.float32)


def _scaled(**kw):
    """sc.pp.scale: per-gene z-scores, so negatives appear."""
    X = _lognorm(**kw)
    mu = X.mean(axis=0, keepdims=True)
    sd = X.std(axis=0, keepdims=True)
    sd[sd == 0] = 1.0
    return ((X - mu) / sd).astype(np.float32)


# --------------------------------------------------------------------------
# Core classification
# --------------------------------------------------------------------------

def test_raw_counts_detected():
    r = assess_matrix_scale(_counts())
    assert r['verdict'] == 'raw_counts'
    assert r['confidence'] == 'high'
    assert r['stats']['integer_valued'] is True
    assert r['stats']['has_negative'] is False


def test_raw_counts_detected_when_sparse():
    r = assess_matrix_scale(csr_matrix(_counts()))
    assert r['verdict'] == 'raw_counts'


def test_normalized_linear_detected():
    r = assess_matrix_scale(_cpm(target=1e4))
    assert r['verdict'] == 'normalized_linear'
    assert r['confidence'] == 'high'
    # The inferred library-size target is the actionable detail here.
    assert r['stats']['row_sum_median'] == pytest.approx(1e4, rel=1e-3)
    assert r['stats']['row_sum_cv'] < 0.01


def test_log_normalized_detected():
    r = assess_matrix_scale(_lognorm(target=1e4))
    assert r['verdict'] == 'log_normalized'
    assert r['confidence'] == 'high'
    # expm1 of the row values sums back to the normalization target.
    assert r['stats']['expm1_row_sum_median'] == pytest.approx(1e4, rel=1e-3)


def test_log_transformed_without_normalization():
    """log1p(counts) is on a log scale but library sizes were never equalized."""
    r = assess_matrix_scale(_log_counts())
    assert r['verdict'] == 'log_transformed'
    assert r['stats']['has_negative'] is False


def test_z_scored_detected():
    r = assess_matrix_scale(_scaled())
    assert r['verdict'] == 'z_scored'
    assert r['stats']['has_negative'] is True


def test_lognorm_survives_gene_subsetting():
    """HVG subsetting destroys the constant-row-sum signal but not the log scale.

    This is the common real-world case: someone normalized, logged, then kept
    2000 HVGs. We must not fall back to 'unknown'.
    """
    X = _lognorm(n_var=400)
    subset = X[:, :50]  # keep an arbitrary slice of genes
    r = assess_matrix_scale(subset)
    assert r['verdict'] in ('log_normalized', 'log_transformed')
    assert 'log' in SCALE_LABELS[r['verdict']].lower()


def test_binary_matrix_detected():
    X = (_counts() > 2).astype(np.float32)
    r = assess_matrix_scale(X)
    assert r['verdict'] == 'binary'


def test_empty_matrix_is_reported_not_crashed():
    r = assess_matrix_scale(np.zeros((10, 5), dtype=np.float32))
    assert r['verdict'] == 'empty'
    assert r['confidence'] == 'low'


def test_every_verdict_has_a_human_label_and_reasons():
    for build in (_counts, _cpm, _lognorm, _log_counts, _scaled):
        r = assess_matrix_scale(build())
        assert r['label'] == SCALE_LABELS[r['verdict']]
        assert r['reasons'], f"{r['verdict']} produced no reasons"
        assert all(isinstance(s, str) for s in r['reasons'])


def test_sampling_caps_work_and_are_reported():
    r = assess_matrix_scale(_counts(n_obs=5000), max_cells=100)
    assert r['stats']['n_cells_sampled'] == 100
    assert r['verdict'] == 'raw_counts'


def test_sampling_is_deterministic():
    X = _counts(n_obs=5000)
    a = assess_matrix_scale(X, max_cells=100)
    b = assess_matrix_scale(X, max_cells=100)
    assert a['stats'] == b['stats']


# --------------------------------------------------------------------------
# Adaptor integration — per-layer report with provenance
# --------------------------------------------------------------------------

def _adata_with_layers():
    ad = anndata.AnnData(X=csr_matrix(_lognorm()))
    ad.layers['counts'] = csr_matrix(_counts())
    ad.layers['cpm'] = csr_matrix(_cpm())
    ad.var_names = [f'g{i}' for i in range(ad.n_vars)]
    ad.obs_names = [f'c{i}' for i in range(ad.n_obs)]
    return ad


def test_list_layers_includes_scale_for_every_entry():
    a = DataAdaptor('x.h5ad', adata=_adata_with_layers())
    layers = {L['name']: L for L in a.list_layers()}
    assert set(layers) == {'X', 'counts', 'cpm'}
    assert layers['X']['scale']['verdict'] == 'log_normalized'
    assert layers['counts']['scale']['verdict'] == 'raw_counts'
    assert layers['cpm']['scale']['verdict'] == 'normalized_linear'
    # The pre-existing shape/density fields must survive.
    assert layers['X']['is_default'] is True
    assert 'density' in layers['X']


def test_scale_reports_scanpy_log1p_provenance():
    ad = _adata_with_layers()
    ad.uns['log1p'] = {'base': None}  # what sc.pp.log1p leaves behind
    a = DataAdaptor('x.h5ad', adata=ad)
    x = next(L for L in a.list_layers() if L['name'] == 'X')
    assert any('log1p' in p for p in x['scale']['provenance'])


def test_scale_reports_xcell_action_history_provenance():
    ad = anndata.AnnData(X=csr_matrix(_counts()))
    a = DataAdaptor('x.h5ad', adata=ad)
    a.run_normalize_total(target_sum=1e4)
    a.run_log1p()
    x = next(L for L in a.list_layers() if L['name'] == 'X')
    joined = ' '.join(x['scale']['provenance'])
    assert 'normalize_total' in joined
    assert 'log1p' in joined


def test_counts_layer_snapshot_is_flagged_as_inferred():
    """The auto-snapshot at load time is a guess; say so rather than imply truth."""
    ad = anndata.AnnData(X=csr_matrix(_counts()))
    a = DataAdaptor('x.h5ad', adata=ad)
    counts = next(L for L in a.list_layers() if L['name'] == 'counts')
    assert any('xcell' in p.lower() for p in counts['scale']['provenance'])


def test_scale_assessment_never_blocks_layer_listing():
    """A pathological layer must degrade to 'unknown', not raise."""
    ad = _adata_with_layers()
    ad.layers['weird'] = np.full((ad.n_obs, ad.n_vars), np.nan, dtype=np.float32)
    a = DataAdaptor('x.h5ad', adata=ad)
    names = {L['name'] for L in a.list_layers()}
    assert 'weird' in names
    weird = next(L for L in a.list_layers() if L['name'] == 'weird')
    assert weird['scale']['verdict'] in ('unknown', 'empty')


# --------------------------------------------------------------------------
# Route
# --------------------------------------------------------------------------

def test_route_layers_exposes_scale(monkeypatch):
    a = DataAdaptor('x.h5ad', adata=_adata_with_layers())
    monkeypatch.setattr(routes, 'get_adaptor', lambda dataset=None: a)
    client = TestClient(app)
    resp = client.get('/api/scanpy/layers')
    assert resp.status_code == 200
    layers = {L['name']: L for L in resp.json()['layers']}
    assert layers['counts']['scale']['verdict'] == 'raw_counts'
    assert layers['counts']['scale']['label'] == 'raw counts'
    assert isinstance(layers['X']['scale']['reasons'], list)
    assert isinstance(layers['X']['scale']['stats']['max'], float)
