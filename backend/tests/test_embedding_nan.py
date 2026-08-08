"""Embeddings that contain NaN coordinates.

NaN is a legitimate value in an `.obsm` embedding and xcell writes it in three
places: `run_pca` and `run_umap` on an active cell selection leave inactive
cells NaN, and Localize leaves unplaceable cells NaN rather than fabricating a
position. But NaN is not valid JSON, and starlette serializes with the strict
encoder — so the embedding endpoint used to raise
``ValueError: Out of range float values are not JSON compliant`` and return a
500, making the result of a subset UMAP impossible to display at all.

These tests pin the contract: non-finite coordinates cross the wire as ``null``.
"""
import json

import anndata
import numpy as np
import pytest
from fastapi.testclient import TestClient
from scipy.sparse import csr_matrix

from xcell.adaptor import DataAdaptor
from xcell.api import routes
from xcell.main import app


def _adaptor(n=12, n_genes=5):
    rng = np.random.default_rng(0)
    ad = anndata.AnnData(X=csr_matrix(rng.random((n, n_genes)).astype(np.float32)))
    ad.var_names = [f'g{i}' for i in range(n_genes)]
    ad.obs_names = [f'c{i}' for i in range(n)]
    coords = np.full((n, 3), np.nan)
    coords[: n // 2] = rng.random((n // 2, 3))
    ad.obsm['X_partial'] = coords
    return DataAdaptor('x.h5ad', adata=ad)


def test_a_partly_nan_embedding_is_json_serializable():
    body = _adaptor().get_embedding('X_partial')
    json.dumps(body, allow_nan=False)


def test_missing_coordinates_come_through_as_null():
    coords = _adaptor().get_embedding('X_partial')['coordinates']
    assert coords[0][0] is not None
    assert coords[-1] == [None, None]


def test_a_missing_third_dimension_is_null_too():
    z = _adaptor().get_embedding('X_partial', dim_z=2)['z']
    assert z[0] is not None and z[-1] is None


def test_the_route_serves_it_instead_of_raising():
    """The actual regression: this used to 500."""
    routes.set_adaptor(_adaptor(), slot='primary')
    resp = TestClient(app).get('/api/embedding/X_partial')
    assert resp.status_code == 200, resp.text
    json.loads(resp.text)  # strict — 'NaN' is not valid JSON
    assert resp.json()['coordinates'][-1] == [None, None]


def test_a_fully_finite_embedding_is_unchanged():
    """The common path keeps returning plain numbers."""
    rng = np.random.default_rng(0)
    ad = anndata.AnnData(X=csr_matrix(rng.random((6, 3)).astype(np.float32)))
    ad.obsm['X_ok'] = rng.random((6, 2))
    coords = DataAdaptor('x.h5ad', adata=ad).get_embedding('X_ok')['coordinates']
    assert all(isinstance(v, float) for row in coords for v in row)


def test_a_subset_umap_can_actually_be_displayed():
    """The user-facing shape of the bug: cluster on a lasso selection, then try
    to look at the result."""
    rng = np.random.default_rng(0)
    ad = anndata.AnnData(X=csr_matrix(rng.random((40, 8)).astype(np.float32)))
    ad.var_names = [f'g{i}' for i in range(8)]
    a = DataAdaptor('x.h5ad', adata=ad)
    active = list(range(20))
    a.run_pca(n_comps=4, active_cell_indices=active)
    a.run_neighbors(n_neighbors=5, active_cell_indices=active)
    a.run_umap(active_cell_indices=active)

    routes.set_adaptor(a, slot='primary')
    resp = TestClient(app).get('/api/embedding/X_umap')
    assert resp.status_code == 200, resp.text
    coords = resp.json()['coordinates']
    assert coords[0][0] is not None      # an active cell has a position
    assert coords[-1] == [None, None]    # an inactive one does not
