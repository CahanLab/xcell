"""The subset registry as a tree, with its embeddings and decorations.

Yesterday's named subsets wrote to suffixed keys and inferred what had been
computed from the names. These tests pin the record: a subset knows its
parent (by containment), what each scoped step wrote and with which
parameters, every embedding it owns, and the shapes and territories drawn on
those embeddings — and deleting it with its results takes all of that with it.
"""
import json

import anndata
import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient
from scipy.sparse import csr_matrix

from xcell.adaptor import DataAdaptor
from xcell.api import routes
from xcell.codegen import translate
from xcell.main import app


def _adata(n_cells=60, n_genes=80, seed=0):
    rng = np.random.default_rng(seed)
    ad = anndata.AnnData(X=csr_matrix(rng.random((n_cells, n_genes)).astype(np.float32)))
    ad.var_names = [f'g{i}' for i in range(n_genes)]
    ad.obs_names = [f'c{i}' for i in range(n_cells)]
    ad.obs['cell_type'] = pd.Categorical(['a', 'b', 'c'] * (n_cells // 3))
    return ad


def _adaptor(**kw):
    return DataAdaptor('x.h5ad', adata=_adata(**kw))


def _global_pipeline(a, n_comps=5):
    a.run_highly_variable_genes(n_top_genes=20)
    a.run_pca(n_comps=n_comps)
    a.run_neighbors(n_neighbors=5)
    a.run_umap()
    a.run_leiden(resolution=1.0)


def _chain(a, name):
    a.run_highly_variable_genes(n_top_genes=10, cell_subset=name)
    a.run_pca(n_comps=4, cell_subset=name)
    a.run_neighbors(n_neighbors=5, cell_subset=name)
    a.run_umap(min_dist=0.3, cell_subset=name)
    a.run_leiden(resolution=0.8, cell_subset=name)


def _subset(a, name):
    return next(s for s in a.list_cell_subsets() if s['name'] == name)


ACTIVE = list(range(0, 60, 2))  # every other cell — 30 of 60


# --- parent, origin, ordering --------------------------------------------------

def test_a_subset_inside_another_gets_it_as_parent():
    a = _adaptor()
    a.create_cell_subset('chondro', ACTIVE)
    out = a.create_cell_subset('prolif', [0, 2, 4, 6])
    assert out['parent'] == 'chondro' and out['depth'] == 1
    assert a.adata.uns['xcell_cell_subsets']['prolif']['parent'] == 'chondro'
    chondro = _subset(a, 'chondro')
    assert chondro['children'] == ['prolif'] and chondro['parent'] is None and chondro['depth'] == 0


def test_the_smallest_containing_subset_wins():
    a = _adaptor()
    a.create_cell_subset('big', list(range(40)))
    a.create_cell_subset('mid', list(range(20)))
    assert a.create_cell_subset('small', [1, 2, 3])['parent'] == 'mid'


def test_a_selection_not_inside_any_subset_is_a_root():
    a = _adaptor()
    a.create_cell_subset('chondro', ACTIVE)
    assert a.create_cell_subset('other', [1, 3, 5])['parent'] is None


def test_a_twin_with_the_same_cells_is_not_a_parent():
    a = _adaptor()
    a.create_cell_subset('chondro', ACTIVE)
    assert a.create_cell_subset('twin', ACTIVE)['parent'] is None


def test_an_explicit_parent_is_honoured_and_validated():
    a = _adaptor()
    a.create_cell_subset('big', list(range(40)))
    a.create_cell_subset('mid', list(range(20)))
    assert a.create_cell_subset('x', [1, 2], parent='big')['parent'] == 'big'
    with pytest.raises(ValueError):
        a.create_cell_subset('y', [45, 46], parent='mid')   # not inside mid
    with pytest.raises(ValueError):
        a.create_cell_subset('z', [1], parent='nope')


def test_origin_is_stored_and_returned():
    a = _adaptor()
    out = a.create_cell_subset('chondro', ACTIVE,
                               origin={'kind': 'selection', 'embedding': 'X_umap'})
    assert out['origin'] == {'kind': 'selection', 'embedding': 'X_umap'}
    assert _subset(a, 'chondro')['origin'] == {'kind': 'selection', 'embedding': 'X_umap'}
    assert a.create_cell_subset('bare', [1, 3])['origin'] is None


def test_the_listing_is_depth_first_with_parents_before_children():
    a = _adaptor()
    a.create_cell_subset('b_root', [40, 41, 42])
    a.create_cell_subset('a_root', list(range(30)))
    a.create_cell_subset('kid2', [5, 6])
    a.create_cell_subset('kid1', [1, 2])
    a.create_cell_subset('grandkid', [1])
    # All created within the same second, so siblings fall back to name order.
    names = [(s['name'], s['depth']) for s in a.list_cell_subsets()]
    assert names == [('a_root', 0), ('kid1', 1), ('grandkid', 2), ('kid2', 1), ('b_root', 0)]


def test_deleting_a_parent_reparents_its_children():
    a = _adaptor()
    a.create_cell_subset('big', list(range(40)))
    a.create_cell_subset('mid', list(range(20)))
    a.create_cell_subset('small', [1, 2])
    a.delete_cell_subset('mid')
    small = _subset(a, 'small')
    assert small['parent'] == 'big' and small['depth'] == 1
    a.delete_cell_subset('big')
    assert _subset(a, 'small')['parent'] is None


def test_a_parent_that_no_longer_exists_is_treated_as_none():
    a = _adaptor()
    a.create_cell_subset('chondro', ACTIVE)
    a.create_cell_subset('prolif', [0, 2])
    del a.adata.obs['subset_chondro']          # dropped in scanpy, say
    assert _subset(a, 'prolif')['parent'] is None


def test_the_registry_survives_an_h5ad_round_trip(tmp_path):
    a = _adaptor()
    a.create_cell_subset('chondro', ACTIVE, origin={'kind': 'selection', 'embedding': 'X_umap'})
    a.create_cell_subset('prolif', [0, 2])
    p = tmp_path / 'x.h5ad'
    a.adata.write_h5ad(p)
    b = DataAdaptor(str(p))
    prolif = _subset(b, 'prolif')
    assert prolif['parent'] == 'chondro' and prolif['children'] == []
    assert _subset(b, 'chondro')['origin']['embedding'] == 'X_umap'
    json.dumps(b.list_cell_subsets())


def test_the_route_accepts_parent_and_origin(monkeypatch):
    a = _adaptor()
    monkeypatch.setattr(routes, 'get_adaptor', lambda dataset=None: a)
    client = TestClient(app)
    client.post('/api/cell_subsets', json={'name': 'chondro', 'cell_indices': ACTIVE})
    r = client.post('/api/cell_subsets', json={
        'name': 'prolif', 'cell_indices': [0, 2], 'parent': 'chondro',
        'origin': {'kind': 'selection', 'embedding': 'X_umap'}})
    assert r.status_code == 200 and r.json()['parent'] == 'chondro'
    assert r.json()['origin'] == {'kind': 'selection', 'embedding': 'X_umap'}
    bad = client.post('/api/cell_subsets', json={'name': 'q', 'cell_indices': [1], 'parent': 'nope'})
    assert bad.status_code == 400


def test_the_notebook_emits_the_parent():
    a = _adaptor()
    a.create_cell_subset('chondro', ACTIVE)
    a.create_cell_subset('prolif', [0, 2])
    step = a.analysis_record.steps[-1]
    t = translate(step)
    assert t.code == [f"xa.create_cell_subset('prolif', SELECTIONS['step_{step.index}'], parent='chondro')"]
    root = translate(a.analysis_record.steps[-2])
    assert 'parent' not in root.code[0]
