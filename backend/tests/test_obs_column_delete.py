"""Deleting an .obs column from the Cells panel.

A column is dropped with its scanpy colour list; if it was a named subset's
membership column the registry entry goes with it (a subset with no column
cannot be activated again), and the step is recorded so the notebook replays
it.
"""
import anndata
import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient
from scipy.sparse import csr_matrix

from xcell.adaptor import DataAdaptor
from xcell.api import routes
from xcell.codegen import EXACT, translate
from xcell.main import app


def _adata(n_cells=30, n_genes=20, seed=0):
    rng = np.random.default_rng(seed)
    ad = anndata.AnnData(X=csr_matrix(rng.random((n_cells, n_genes)).astype(np.float32)))
    ad.var_names = [f'g{i}' for i in range(n_genes)]
    ad.obs_names = [f'c{i}' for i in range(n_cells)]
    ad.obs['cell_type'] = pd.Categorical(['a', 'b', 'c'] * (n_cells // 3))
    ad.obs['score'] = rng.random(n_cells)
    ad.uns['cell_type_colors'] = ['#111111', '#222222', '#333333']
    return ad


def _adaptor():
    return DataAdaptor('x.h5ad', adata=_adata())


def test_deleting_drops_the_column_and_its_colours_and_records_the_step():
    a = _adaptor()
    out = a.delete_obs_column('cell_type')
    assert out == {'column': 'cell_type', 'dropped_colors': True, 'subset_removed': None}
    assert 'cell_type' not in a.adata.obs.columns
    assert 'cell_type_colors' not in a.adata.uns
    step = a.analysis_record.steps[-1]
    assert step.action == 'delete_obs_column' and step.params == {'column': 'cell_type'}


def test_deleting_a_numeric_column_reports_no_colours():
    a = _adaptor()
    assert a.delete_obs_column('score')['dropped_colors'] is False
    assert 'score' not in a.adata.obs.columns


def test_an_unknown_column_is_a_key_error():
    with pytest.raises(KeyError):
        _adaptor().delete_obs_column('nope')


def test_deleting_a_subsets_column_removes_its_registry_entry():
    a = _adaptor()
    a.create_cell_subset('chondro', list(range(0, 30, 2)))
    out = a.delete_obs_column('subset_chondro')
    assert out['subset_removed'] == 'chondro'
    assert 'chondro' not in a.adata.uns['xcell_cell_subsets']
    assert a.list_cell_subsets() == []


def test_delete_annotation_still_works_as_an_alias():
    a = _adaptor()
    a.delete_annotation('score')
    assert 'score' not in a.adata.obs.columns


def test_the_notebook_replays_the_deletion():
    a = _adaptor()
    a.delete_obs_column('score')
    t = translate(a.analysis_record.steps[-1])
    assert t.fidelity == EXACT
    assert t.code == ["del adata.obs['score']"]


def test_the_route_deletes_and_maps_a_missing_column_to_404(monkeypatch):
    a = _adaptor()
    monkeypatch.setattr(routes, 'get_adaptor', lambda dataset=None: a)
    client = TestClient(app)
    r = client.delete('/api/obs/cell_type')
    assert r.status_code == 200 and r.json()['column'] == 'cell_type'
    assert 'cell_type' not in a.adata.obs.columns
    assert client.delete('/api/obs/cell_type').status_code == 404
