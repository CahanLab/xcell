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


# --- what each scoped step wrote, recorded at run time -------------------------

def test_each_scoped_step_records_its_key_and_params_in_the_registry():
    a = _adaptor(); _global_pipeline(a)
    a.create_cell_subset('chondro', ACTIVE); _chain(a, 'chondro')
    d = a.adata.uns['xcell_cell_subsets']['chondro']['derived']
    assert d['hvg']['key'] == 'highly_variable__chondro' and d['hvg']['params']['n_top_genes'] == 10
    assert d['pca']['key'] == 'X_pca_chondro' and d['pca']['params']['n_comps'] == 4
    assert d['graph']['key'] == 'chondro_connectivities' and d['graph']['params']['n_neighbors'] == 5
    assert d['graph']['params']['use_rep'] == 'X_pca_chondro'
    assert d['umap'] == {'X_umap_chondro': {
        'min_dist': 0.3, 'spread': 1.0, 'n_components': 2, 'graph_key': 'chondro_connectivities'}}
    assert d['leiden']['leiden_chondro'] == {'resolution': 0.8, 'graph_key': 'chondro_connectivities'}


def test_a_second_umap_over_another_graph_is_recorded_beside_the_first():
    a = _adaptor(); _global_pipeline(a)
    a.create_cell_subset('chondro', ACTIVE); _chain(a, 'chondro')
    a.run_umap(cell_subset='chondro', graph_key='connectivities', key_added='X_umap_chondro_alt')
    s = _subset(a, 'chondro')
    assert s['derived']['umap'] == ['X_umap_chondro', 'X_umap_chondro_alt']
    assert s['embeddings'] == ['X_pca_chondro', 'X_umap_chondro', 'X_umap_chondro_alt']
    assert set(s['steps']['umap']) == {'X_umap_chondro', 'X_umap_chondro_alt'}


def test_rerunning_a_step_overwrites_its_record():
    a = _adaptor(); _global_pipeline(a)
    a.create_cell_subset('chondro', ACTIVE); _chain(a, 'chondro')
    a.run_leiden(resolution=2.0, cell_subset='chondro')
    s = _subset(a, 'chondro')
    assert s['derived']['leiden'] == ['leiden_chondro']
    assert s['steps']['leiden']['leiden_chondro']['resolution'] == 2.0


def test_a_key_that_vanished_is_not_reported_even_if_recorded():
    a = _adaptor(); _global_pipeline(a)
    a.create_cell_subset('chondro', ACTIVE); _chain(a, 'chondro')
    del a.adata.obsm['X_umap_chondro']
    s = _subset(a, 'chondro')
    assert s['derived']['umap'] == [] and 'X_umap_chondro' not in s['embeddings']


def test_a_file_from_before_the_record_is_still_detected_by_name():
    a = _adaptor(); _global_pipeline(a)
    a.create_cell_subset('chondro', ACTIVE); _chain(a, 'chondro')
    a.adata.uns['xcell_cell_subsets']['chondro'].pop('derived')
    s = _subset(a, 'chondro')
    assert s['derived']['pca'] == 'X_pca_chondro' and s['derived']['leiden'] == ['leiden_chondro']
    assert s['derived']['umap'] == ['X_umap_chondro'] and s['steps'] == {}


def test_the_record_round_trips_through_h5ad(tmp_path):
    a = _adaptor(); _global_pipeline(a)
    a.create_cell_subset('chondro', ACTIVE); _chain(a, 'chondro')
    p = tmp_path / 'x.h5ad'; a.adata.write_h5ad(p)
    b = DataAdaptor(str(p))
    s = _subset(b, 'chondro')
    assert s['steps']['umap']['X_umap_chondro']['min_dist'] == 0.3
    assert s['derived']['pca_subsets'] == []
    json.dumps(s)


def test_deleting_with_results_drops_every_recorded_umap():
    a = _adaptor(); _global_pipeline(a)
    a.create_cell_subset('chondro', ACTIVE); _chain(a, 'chondro')
    a.run_umap(cell_subset='chondro', graph_key='connectivities', key_added='X_umap_chondro_alt')
    out = a.delete_cell_subset('chondro', drop_derived=True)
    assert 'X_umap_chondro' not in a.adata.obsm and 'X_umap_chondro_alt' not in a.adata.obsm
    assert {'X_umap_chondro', 'X_umap_chondro_alt'} <= set(out['dropped'])


# --- shapes and lines persist; decorations resolve by embedding ---------------

LINE = {'id': 'line_1', 'name': 'ridge', 'embeddingName': 'X_umap_chondro', 'dimX': 0, 'dimY': 1,
        'points': [[0.0, 0.0], [1.0, 1.0]], 'smoothedPoints': None, 'drawType': 'pencil',
        'closed': False, 'visible': True, 'strokeColor': '#4ecdc4', 'strokeWidth': 2,
        'fillColor': None}

RING = {'ring': [[0.0, 0.0], [1.0, 0.0], [1.0, 1.0]], 'cuts': [], 'anchors': []}


def test_lines_are_written_to_uns_and_survive_a_reload(tmp_path):
    a = _adaptor()
    a.set_lines([LINE])
    assert json.loads(a.adata.uns['xcell_lines_json'])[0]['name'] == 'ridge'
    p = tmp_path / 'x.h5ad'
    a.adata.write_h5ad(p)
    b = DataAdaptor(str(p))
    assert b.get_lines() == [LINE]


def test_clearing_the_lines_clears_the_file_too():
    a = _adaptor()
    a.set_lines([LINE])
    a.set_lines([])
    assert a.get_lines() == [] and json.loads(a.adata.uns['xcell_lines_json']) == []


def test_an_old_export_with_embedding_instead_of_embedding_name_still_loads():
    ad = _adata()
    ad.uns['xcell_lines_json'] = json.dumps([
        {'name': 'old', 'embedding': 'X_umap', 'points': [[0, 0], [1, 1]]},
        {'name': 'broken', 'points': []},
    ])
    a = DataAdaptor('x.h5ad', adata=ad)
    lines = a.get_lines()
    assert len(lines) == 1
    line = lines[0]
    assert line['embeddingName'] == 'X_umap' and line['visible'] is True
    assert line['drawType'] == 'pencil' and line['id']
    assert 'embedding' not in line


def test_a_corrupt_lines_blob_does_not_block_the_load():
    ad = _adata()
    ad.uns['xcell_lines_json'] = '{not json'
    assert DataAdaptor('x.h5ad', adata=ad).get_lines() == []


def test_the_export_carries_the_full_lines():
    a = _adaptor(); _global_pipeline(a)
    a.set_lines([{**LINE, 'embeddingName': 'X_umap'}])
    exported = a.prepare_export_with_lines()
    stored = json.loads(exported.uns['xcell_lines_json'])
    assert stored[0]['strokeColor'] == '#4ecdc4' and stored[0]['id'] == 'line_1'


def test_the_listing_names_the_shapes_and_territories_on_a_subsets_embeddings():
    a = _adaptor(); _global_pipeline(a)
    a.create_cell_subset('chondro', ACTIVE); _chain(a, 'chondro')
    a.set_lines([LINE, {**LINE, 'id': 'line_2', 'name': 'elsewhere', 'embeddingName': 'X_umap'}])
    a.save_territories('zones', {'embedding': 'X_umap_chondro', 'sections': {'all': RING}})
    a.save_territories('other', {'embedding': 'X_umap', 'sections': {'all': RING}})
    s = _subset(a, 'chondro')
    assert s['decorations'] == {'lines': ['ridge'], 'territories': ['zones']}


def test_a_subset_with_no_embeddings_has_no_decorations():
    a = _adaptor()
    a.create_cell_subset('chondro', ACTIVE)
    a.set_lines([LINE])
    assert _subset(a, 'chondro')['decorations'] == {'lines': [], 'territories': []}


def test_the_lines_route_round_trips_every_field(monkeypatch):
    a = _adaptor()
    monkeypatch.setattr(routes, 'get_adaptor', lambda dataset=None: a)
    client = TestClient(app)
    assert client.post('/api/lines', json={'lines': [LINE]}).status_code == 200
    assert client.get('/api/lines').json()['lines'] == [LINE]
    # The pre-existing minimal payload still works, defaults filled in.
    minimal = {'name': 'm', 'embeddingName': 'X_umap', 'points': [[0, 0], [1, 1]]}
    assert client.post('/api/lines', json={'lines': [minimal]}).status_code == 200
    got = client.get('/api/lines').json()['lines'][0]
    assert got['visible'] is True and got['closed'] is False and got['id']
