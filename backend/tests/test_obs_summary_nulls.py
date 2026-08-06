"""Numeric obs summaries when there is nothing to summarize.

A numeric column with no usable values has no min, max or mean. JSON has no
NaN, so those come back as null — and every consumer has to cope with that.
These tests pin the contract so it can't drift into NaN (unserializable) or
0.0 (a lie).
"""
import anndata
import numpy as np
import pandas as pd
import pytest
from scipy.sparse import csr_matrix
from fastapi.testclient import TestClient

from xcell.adaptor import DataAdaptor
from xcell.main import app
from xcell.api import routes


def _adata(n_cells=20, n_genes=6):
    rng = np.random.default_rng(0)
    ad = anndata.AnnData(X=csr_matrix(rng.integers(0, 10, (n_cells, n_genes)).astype(np.float32)))
    ad.var_names = [f'g{i}' for i in range(n_genes)]
    ad.obs_names = [f'c{i}' for i in range(n_cells)]
    return ad


def test_all_nan_numeric_column_summarizes_to_nulls():
    """The reported crash came from a column like this."""
    ad = _adata()
    ad.obs['empty_qc'] = np.full(ad.n_obs, np.nan, dtype=float)
    a = DataAdaptor('x.h5ad', adata=ad)
    s = a.get_obs_column_summary('empty_qc')
    assert s['dtype'] == 'numeric'
    assert s['min'] is None
    assert s['max'] is None
    assert s['mean'] is None


def test_partially_nan_column_still_reports_statistics():
    """pandas skips NaN, so a partly-missing column is still summarizable."""
    ad = _adata()
    values = np.full(ad.n_obs, np.nan, dtype=float)
    values[:5] = [1.0, 2.0, 3.0, 4.0, 5.0]
    ad.obs['partial'] = values
    a = DataAdaptor('x.h5ad', adata=ad)
    s = a.get_obs_column_summary('partial')
    assert s['min'] == 1.0
    assert s['max'] == 5.0
    assert s['mean'] == pytest.approx(3.0)


def test_nulls_survive_json_serialization():
    """NaN would serialize to invalid JSON; null is the whole point."""
    ad = _adata()
    ad.obs['empty_qc'] = np.full(ad.n_obs, np.nan, dtype=float)
    a = DataAdaptor('x.h5ad', adata=ad)
    routes.set_adaptor(a, slot='primary')
    client = TestClient(app)

    resp = client.get('/api/obs/summary/empty_qc')
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body['min'] is None and body['max'] is None and body['mean'] is None

    # The panel that crashed reads the bulk endpoint, not the single one.
    resp = client.get('/api/obs/summaries')
    assert resp.status_code == 200, resp.text
    payload = resp.json()
    summaries = payload['summaries'] if isinstance(payload, dict) else payload
    entry = next(s for s in summaries if s['name'] == 'empty_qc')
    assert entry['mean'] is None


def test_single_valued_column_has_a_defined_mean():
    """Guards against an over-eager "if min == max, null it out" fix."""
    ad = _adata()
    ad.obs['constant'] = np.full(ad.n_obs, 7.0)
    a = DataAdaptor('x.h5ad', adata=ad)
    s = a.get_obs_column_summary('constant')
    assert (s['min'], s['max'], s['mean']) == (7.0, 7.0, 7.0)


def test_all_nan_column_is_still_listed_as_numeric():
    """It must remain selectable — the UI has to render it, not hide it."""
    ad = _adata()
    ad.obs['empty_qc'] = np.full(ad.n_obs, np.nan, dtype=float)
    a = DataAdaptor('x.h5ad', adata=ad)
    assert a.get_schema()['obs_dtypes']['empty_qc'] == 'numeric'
