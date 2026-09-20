"""Named cell subsets: the persisted, scoped form of the active cell mask.

The mask used to live only in the browser. A masked sub-clustering run wrote
its results *over* the dataset's own ``X_pca`` / ``connectivities`` /
``X_umap`` / ``leiden`` with NaN rows and ``unassigned`` labels for the masked
cells, so the moment the browser forgot the mask (reload, Reset Mask) those
cells were invisible with nothing left to restore. These tests pin the
replacement: a subset is a named boolean ``.obs`` column plus a registry
entry, and every clustering-chain operation run on it writes to keys suffixed
with its name, leaving the full-dataset results exactly as they were.
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
from xcell.codegen import XCELL, translate
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


ACTIVE = list(range(0, 60, 2))  # every other cell — 30 of 60


# --- create / list / delete ---------------------------------------------------

def test_creating_a_subset_writes_a_boolean_obs_column_and_a_registry_entry():
    a = _adaptor()
    out = a.create_cell_subset('chondro', ACTIVE)

    assert out['name'] == 'chondro'
    assert out['obs_key'] == 'subset_chondro'
    assert out['n_cells'] == 30 and out['n_total'] == 60
    col = a.adata.obs['subset_chondro']
    assert col.dtype == bool
    assert col.values[ACTIVE].all() and not col.values[1::2].any()
    assert 'chondro' in a.adata.uns['xcell_cell_subsets']


def test_listing_reports_each_subset_with_its_live_cell_count():
    a = _adaptor()
    a.create_cell_subset('chondro', ACTIVE)
    a.create_cell_subset('rest', [1, 3, 5])
    names = {s['name']: s for s in a.list_cell_subsets()}
    assert set(names) == {'chondro', 'rest'}
    assert names['rest']['n_cells'] == 3
    assert names['chondro']['derived'] == {
        'hvg': None, 'pca': None, 'graph': None, 'umap': [], 'leiden': [], 'pca_subsets': [],
    }


def test_the_indices_round_trip():
    a = _adaptor()
    a.create_cell_subset('chondro', ACTIVE)
    assert a.get_cell_subset_indices('chondro') == ACTIVE


def test_a_name_is_sanitised_to_something_that_can_suffix_a_key():
    a = _adaptor()
    out = a.create_cell_subset('my chondrocytes (E11.5)', ACTIVE)
    assert out['name'] == 'my_chondrocytes_E11_5'
    assert 'subset_my_chondrocytes_E11_5' in a.adata.obs


@pytest.mark.parametrize('bad', ['', '   ', '(((', 'unassigned'])
def test_an_unusable_name_is_a_value_error(bad):
    with pytest.raises(ValueError):
        _adaptor().create_cell_subset(bad, ACTIVE)


def test_a_duplicate_name_is_refused_unless_overwriting():
    a = _adaptor()
    a.create_cell_subset('chondro', ACTIVE)
    with pytest.raises(ValueError, match='already exists'):
        a.create_cell_subset('chondro', [0, 1, 2])
    out = a.create_cell_subset('chondro', [0, 1, 2], overwrite=True)
    assert out['n_cells'] == 3
    assert a.get_cell_subset_indices('chondro') == [0, 1, 2]


def test_empty_out_of_range_and_whole_dataset_selections_are_refused():
    a = _adaptor()
    with pytest.raises(ValueError):
        a.create_cell_subset('x', [])
    with pytest.raises(ValueError):
        a.create_cell_subset('x', [0, 999])
    with pytest.raises(ValueError, match='every cell'):
        a.create_cell_subset('x', list(range(60)))


def test_an_unknown_subset_is_a_key_error():
    with pytest.raises(KeyError):
        _adaptor().get_cell_subset_indices('nope')


def test_deleting_removes_the_column_and_the_entry():
    a = _adaptor()
    a.create_cell_subset('chondro', ACTIVE)
    a.delete_cell_subset('chondro')
    assert 'subset_chondro' not in a.adata.obs
    assert 'chondro' not in a.adata.uns['xcell_cell_subsets']
    assert a.list_cell_subsets() == []


def test_deleting_can_also_drop_everything_derived_from_it():
    a = _adaptor()
    _global_pipeline(a)
    a.create_cell_subset('chondro', ACTIVE)
    a.run_highly_variable_genes(n_top_genes=10, cell_subset='chondro')
    a.run_pca(n_comps=4, cell_subset='chondro')
    a.run_neighbors(n_neighbors=5, cell_subset='chondro')
    a.run_umap(cell_subset='chondro')
    a.run_leiden(resolution=1.0, cell_subset='chondro')

    out = a.delete_cell_subset('chondro', drop_derived=True)

    assert 'highly_variable__chondro' not in a.adata.var
    assert 'X_pca_chondro' not in a.adata.obsm and 'PCs_chondro' not in a.adata.varm
    assert 'chondro_connectivities' not in a.adata.obsp and 'chondro' not in a.adata.uns
    assert 'X_umap_chondro' not in a.adata.obsm
    assert 'leiden_chondro' not in a.adata.obs
    assert set(out['dropped']) >= {'highly_variable__chondro', 'X_pca_chondro',
                                   'chondro_connectivities', 'X_umap_chondro', 'leiden_chondro'}
    # And the dataset's own results are still there.
    assert 'X_pca' in a.adata.obsm and 'connectivities' in a.adata.obsp
    assert 'X_umap' in a.adata.obsm and 'leiden' in a.adata.obs


def test_deleting_without_the_flag_keeps_the_derived_results():
    a = _adaptor()
    a.create_cell_subset('chondro', ACTIVE)
    a.run_pca(n_comps=4, cell_subset='chondro')
    a.delete_cell_subset('chondro')
    assert 'X_pca_chondro' in a.adata.obsm


def test_the_subset_follows_the_cells_when_some_are_deleted():
    a = _adaptor()
    a.create_cell_subset('chondro', ACTIVE)
    a.delete_cells([0, 2, 4])  # three subset members
    entry = next(s for s in a.list_cell_subsets() if s['name'] == 'chondro')
    assert entry['n_cells'] == 27 and entry['n_total'] == 57


# --- the clustering chain on a subset ------------------------------------------

def test_hvg_on_a_subset_writes_its_own_column_and_leaves_the_pooled_one_alone():
    a = _adaptor()
    a.run_highly_variable_genes(n_top_genes=20)
    pooled = a.adata.var['highly_variable'].values.copy()
    a.create_cell_subset('chondro', ACTIVE)

    out = a.run_highly_variable_genes(n_top_genes=10, cell_subset='chondro')

    assert out['column'] == 'highly_variable__chondro' and out['cell_subset'] == 'chondro'
    assert out['n_highly_variable'] == 10
    assert int(a.adata.var['highly_variable__chondro'].sum()) == 10
    np.testing.assert_array_equal(a.adata.var['highly_variable'].values, pooled)


def test_pca_on_a_subset_writes_a_suffixed_embedding_with_nan_outside_it():
    a = _adaptor()
    a.run_pca(n_comps=5)
    before = a.adata.obsm['X_pca'].copy()
    a.create_cell_subset('chondro', ACTIVE)

    out = a.run_pca(n_comps=4, cell_subset='chondro')

    assert out['embedding_name'] == 'X_pca_chondro'
    sub = a.adata.obsm['X_pca_chondro']
    assert sub.shape == (60, 4)
    assert np.isfinite(sub[ACTIVE]).all()
    assert np.isnan(sub[1::2]).all()
    assert a.adata.varm['PCs_chondro'].shape == (80, 4)
    assert 'variance_ratio' in a.adata.uns['pca_chondro']
    np.testing.assert_array_equal(a.adata.obsm['X_pca'], before)
    assert not np.isnan(a.adata.obsm['X_pca']).any()


def test_pca_on_a_subset_prefers_the_subsets_own_hvg_column():
    a = _adaptor()
    a.run_highly_variable_genes(n_top_genes=30)
    a.create_cell_subset('chondro', ACTIVE)
    a.run_highly_variable_genes(n_top_genes=12, cell_subset='chondro')

    out = a.run_pca(n_comps=4, cell_subset='chondro')

    assert out['n_genes_used'] == 12
    assert out['gene_subset_type'] == 'highly_variable__chondro (auto)'
    finite_rows = ~np.isnan(a.adata.varm['PCs_chondro'][:, 0])
    assert finite_rows.sum() == 12


def test_pca_on_a_subset_falls_back_to_the_pooled_hvg_then_to_all_genes():
    a = _adaptor()
    a.create_cell_subset('chondro', ACTIVE)
    assert a.run_pca(n_comps=4, cell_subset='chondro')['n_genes_used'] == 80
    a.run_highly_variable_genes(n_top_genes=30)
    assert a.run_pca(n_comps=4, cell_subset='chondro')['n_genes_used'] == 30


def test_rerunning_the_dataset_pca_does_not_clear_a_subset_pca():
    """run_pca wipes X_pca_* PC subsets, which look just like a subset PCA."""
    a = _adaptor()
    a.run_pca(n_comps=5)
    a.create_pca_subset(drop_pc_indices=[1])
    a.create_cell_subset('chondro', ACTIVE)
    a.run_pca(n_comps=4, cell_subset='chondro')

    out = a.run_pca(n_comps=5)

    assert 'X_pca_chondro' in a.adata.obsm and 'PCs_chondro' in a.adata.varm
    assert 'X_pca_chondro' not in out.get('cleared_subsets', [])
    assert any(k.startswith('X_pca_noPC') for k in out.get('cleared_subsets', []))


def test_neighbors_on_a_subset_writes_a_named_graph_and_leaves_the_default_alone():
    a = _adaptor()
    a.run_pca(n_comps=5)
    a.run_neighbors(n_neighbors=5)
    before_nnz = a.adata.obsp['connectivities'].nnz
    a.create_cell_subset('chondro', ACTIVE)
    a.run_pca(n_comps=4, cell_subset='chondro')

    out = a.run_neighbors(n_neighbors=5, cell_subset='chondro')

    assert out['graph_key'] == 'chondro_connectivities'
    assert out['use_rep'] == 'X_pca_chondro'
    conn = a.adata.obsp['chondro_connectivities'].tocoo()
    assert conn.shape == (60, 60) and conn.nnz > 0
    inactive = set(range(1, 60, 2))
    assert not (set(conn.row.tolist()) | set(conn.col.tolist())) & inactive
    assert 'chondro_distances' in a.adata.obsp
    meta = a.adata.uns['chondro']
    assert meta['connectivities_key'] == 'chondro_connectivities'
    assert meta['distances_key'] == 'chondro_distances'
    assert meta['params']['use_rep'] == 'X_pca_chondro'
    assert a.adata.obsp['connectivities'].nnz == before_nnz
    graphs = {g['key']: g for g in a.list_neighbor_graphs()}
    assert graphs['chondro_connectivities']['suffix'] == 'chondro'
    assert 'chondro' in graphs['chondro_connectivities']['label']


def test_neighbors_on_a_subset_uses_the_dataset_pca_when_the_subset_has_none():
    a = _adaptor()
    a.run_pca(n_comps=5)
    a.create_cell_subset('chondro', ACTIVE)
    out = a.run_neighbors(n_neighbors=5, cell_subset='chondro')
    assert out['use_rep'] == 'X_pca'
    assert 'chondro_connectivities' in a.adata.obsp


def test_an_explicit_x_pca_on_a_subset_wins_over_the_subsets_own_pca():
    a = _adaptor()
    a.run_pca(n_comps=5)
    a.create_cell_subset('chondro', ACTIVE)
    a.run_pca(n_comps=4, cell_subset='chondro')
    out = a.run_neighbors(n_neighbors=5, use_rep='X_pca', cell_subset='chondro')
    assert out['use_rep'] == 'X_pca'
    assert a.adata.uns['chondro']['params']['use_rep'] == 'X_pca'


def test_neighbors_on_a_subset_refuses_a_representation_with_no_values_for_its_cells():
    """The old overwrite path leaves NaN rows in X_pca; scanpy would choke on them."""
    a = _adaptor()
    a.run_pca(n_comps=5, active_cell_indices=list(range(30)))  # NaN for cells 30..59
    a.create_cell_subset('late', list(range(30, 60)))
    with pytest.raises(ValueError, match='no values'):
        a.run_neighbors(n_neighbors=5, cell_subset='late')


def test_umap_on_a_subset_embeds_its_own_graph_and_leaves_x_umap_alone():
    a = _adaptor()
    _global_pipeline(a)
    before = a.adata.obsm['X_umap'].copy()
    a.create_cell_subset('chondro', ACTIVE)
    a.run_pca(n_comps=4, cell_subset='chondro')
    a.run_neighbors(n_neighbors=5, cell_subset='chondro')

    out = a.run_umap(cell_subset='chondro')

    assert out['embedding_name'] == 'X_umap_chondro'
    assert out['graph_key'] == 'chondro_connectivities'
    um = a.adata.obsm['X_umap_chondro']
    assert np.isfinite(um[ACTIVE]).all() and np.isnan(um[1::2]).all()
    np.testing.assert_array_equal(a.adata.obsm['X_umap'], before)


def test_umap_on_a_subset_without_its_own_graph_slices_the_default_one():
    a = _adaptor()
    _global_pipeline(a)
    a.create_cell_subset('chondro', ACTIVE)
    out = a.run_umap(cell_subset='chondro')
    assert out['embedding_name'] == 'X_umap_chondro'
    assert out['graph_key'] is None
    assert np.isfinite(a.adata.obsm['X_umap_chondro'][ACTIVE]).all()


def test_leiden_on_a_subset_writes_a_suffixed_column_with_unassigned_outside():
    a = _adaptor()
    _global_pipeline(a)
    before = a.adata.obs['leiden'].copy()
    a.create_cell_subset('chondro', ACTIVE)
    a.run_pca(n_comps=4, cell_subset='chondro')
    a.run_neighbors(n_neighbors=5, cell_subset='chondro')

    out = a.run_leiden(resolution=1.0, cell_subset='chondro')

    assert out['key_added'] == 'leiden_chondro'
    assert out['graph_key'] == 'chondro_connectivities'
    col = a.adata.obs['leiden_chondro']
    assert (col.values[1::2] == 'unassigned').all()
    assert not (col.values[ACTIVE] == 'unassigned').any()
    assert out['n_clusters'] == len(set(col.values[ACTIVE]))
    pd.testing.assert_series_equal(a.adata.obs['leiden'], before)


def test_a_custom_leiden_name_is_honoured_on_a_subset():
    a = _adaptor()
    _global_pipeline(a)
    a.create_cell_subset('chondro', ACTIVE)
    out = a.run_leiden(resolution=1.0, key_added='fine', cell_subset='chondro')
    assert out['key_added'] == 'fine' and 'fine' in a.adata.obs


def test_the_listing_reports_what_has_been_derived():
    a = _adaptor()
    _global_pipeline(a)
    a.create_cell_subset('chondro', ACTIVE)
    a.run_highly_variable_genes(n_top_genes=10, cell_subset='chondro')
    a.run_pca(n_comps=4, cell_subset='chondro')
    a.run_neighbors(n_neighbors=5, cell_subset='chondro')
    a.run_umap(cell_subset='chondro')
    a.run_leiden(resolution=1.0, cell_subset='chondro')
    a.run_leiden(resolution=2.0, key_added='leiden_chondro_r2', cell_subset='chondro')

    entry = next(s for s in a.list_cell_subsets() if s['name'] == 'chondro')
    assert entry['derived'] == {
        'hvg': 'highly_variable__chondro',
        'pca': 'X_pca_chondro',
        'graph': 'chondro_connectivities',
        'umap': ['X_umap_chondro'],
        'leiden': ['leiden_chondro', 'leiden_chondro_r2'],
        'pca_subsets': [],
    }


def test_prerequisites_accept_the_subsets_own_keys():
    a = _adaptor()
    a.create_cell_subset('chondro', ACTIVE)
    assert a.check_prerequisites('neighbors', cell_subset='chondro')['satisfied'] is False
    a.run_pca(n_comps=4, cell_subset='chondro')
    assert a.check_prerequisites('neighbors', cell_subset='chondro')['satisfied'] is True
    assert a.check_prerequisites('neighbors')['satisfied'] is False
    assert a.check_prerequisites('leiden', cell_subset='chondro')['satisfied'] is False
    a.run_neighbors(n_neighbors=5, cell_subset='chondro')
    assert a.check_prerequisites('leiden', cell_subset='chondro')['satisfied'] is True
    assert a.check_prerequisites('umap', cell_subset='chondro')['satisfied'] is True


def test_an_unknown_subset_name_on_an_operation_is_a_key_error():
    a = _adaptor()
    with pytest.raises(KeyError):
        a.run_pca(n_comps=4, cell_subset='nope')


def test_the_whole_chain_is_json_serialisable_and_recorded_with_its_selection():
    a = _adaptor()
    _global_pipeline(a)
    a.create_cell_subset('chondro', ACTIVE)
    results = [
        a.run_highly_variable_genes(n_top_genes=10, cell_subset='chondro'),
        a.run_pca(n_comps=4, cell_subset='chondro'),
        a.run_neighbors(n_neighbors=5, cell_subset='chondro'),
        a.run_umap(cell_subset='chondro'),
        a.run_leiden(resolution=1.0, cell_subset='chondro'),
    ]
    for r in results:
        json.dumps(r, allow_nan=False)
    steps = a.analysis_record.steps[-5:]
    assert [s.action for s in steps] == ['highly_variable_genes', 'pca', 'neighbors', 'umap', 'leiden']
    for s in steps:
        assert s.params['cell_subset'] == 'chondro'
        assert s.selection == ACTIVE
    created = next(s for s in a.analysis_record.steps if s.action == 'create_cell_subset')
    assert created.selection == ACTIVE and created.params['name'] == 'chondro'


# --- codegen ------------------------------------------------------------------

def test_a_scoped_step_is_exported_through_the_xcell_api_naming_the_subset():
    from xcell.analysis_record import AnalysisRecord
    r = AnalysisRecord()
    step = r.add_step('pca', {'n_comps': 4, 'svd_solver': 'arpack', 'gene_subset': None,
                              'cell_subset': 'chondro'}, {}, selection=ACTIVE, n_total=60)
    t = translate(step)
    assert t.fidelity == XCELL
    assert t.code == ["xa.run_pca(n_comps=4, svd_solver='arpack', cell_subset='chondro')"]
    # The code honours the subset, so the "runs on the whole dataset" caveat
    # would be wrong here.
    assert not any('whole dataset' in w for w in t.warnings)


def test_creating_a_subset_is_exported_from_the_recorded_selection():
    from xcell.analysis_record import AnalysisRecord
    r = AnalysisRecord()
    step = r.add_step('create_cell_subset', {'name': 'chondro'}, {'n_cells': 30},
                      selection=ACTIVE, n_total=60)
    t = translate(step)
    assert t.fidelity == XCELL
    assert t.code == [f"xa.create_cell_subset('chondro', SELECTIONS['step_{step.index}'])"]


# --- routes -------------------------------------------------------------------

def _install():
    a = _adaptor()
    routes.set_adaptor(a, slot='primary')
    return a, TestClient(app)


def test_the_subset_routes_create_list_fetch_and_delete():
    a, c = _install()
    r = c.post('/api/cell_subsets', json={'name': 'chondro', 'cell_indices': ACTIVE})
    assert r.status_code == 200, r.text
    assert r.json()['n_cells'] == 30

    r = c.get('/api/cell_subsets')
    assert r.status_code == 200 and [s['name'] for s in r.json()['subsets']] == ['chondro']

    r = c.get('/api/cell_subsets/chondro/indices')
    assert r.status_code == 200 and r.json()['indices'] == ACTIVE

    r = c.post('/api/cell_subsets', json={'name': 'chondro', 'cell_indices': [1]})
    assert r.status_code == 400

    r = c.get('/api/cell_subsets/nope/indices')
    assert r.status_code == 404

    r = c.delete('/api/cell_subsets/chondro', params={'drop_derived': 'true'})
    assert r.status_code == 200
    assert 'subset_chondro' not in a.adata.obs


def test_the_scanpy_routes_take_a_subset_name():
    a, c = _install()
    _global_pipeline(a)
    c.post('/api/cell_subsets', json={'name': 'chondro', 'cell_indices': ACTIVE})

    assert c.post('/api/scanpy/highly_variable_genes',
                  json={'n_top_genes': 10, 'cell_subset': 'chondro'}).status_code == 200
    assert c.post('/api/scanpy/pca', json={'n_comps': 4, 'cell_subset': 'chondro'}).status_code == 200
    assert c.post('/api/scanpy/neighbors', json={'n_neighbors': 5, 'cell_subset': 'chondro'}).status_code == 200
    assert c.post('/api/scanpy/umap', json={'cell_subset': 'chondro'}).status_code == 200
    r = c.post('/api/scanpy/leiden', json={'resolution': 1.0, 'cell_subset': 'chondro'})
    assert r.status_code == 200 and r.json()['key_added'] == 'leiden_chondro'

    assert 'X_pca_chondro' in a.adata.obsm and 'X_umap_chondro' in a.adata.obsm
    assert not np.isnan(a.adata.obsm['X_umap']).any()

    r = c.get('/api/scanpy/prerequisites/leiden', params={'cell_subset': 'chondro'})
    assert r.status_code == 200 and r.json()['satisfied'] is True

    r = c.post('/api/scanpy/pca', json={'n_comps': 4, 'cell_subset': 'nope'})
    assert r.status_code == 404


# --- the barplot honours the mask ---------------------------------------------

def test_crosstab_counts_only_the_active_cells_when_given_some():
    a = _adaptor()
    a.adata.obs['grp'] = pd.Categorical(['x'] * 30 + ['y'] * 30)
    full = a.crosstab('cell_type', 'grp')
    assert sum(sum(row) for row in full['counts']) == 60

    part = a.crosstab('cell_type', 'grp', active_cell_indices=list(range(30)))
    assert sum(sum(row) for row in part['counts']) == 30
    assert part['n_cells'] == 30
    # Every cell in the first 30 is 'x', so the y column is empty but still there.
    y = part['b_categories'].index('y')
    assert all(row[y] == 0 for row in part['counts'])


def test_the_crosstab_route_accepts_the_indices_in_a_post_body():
    a, c = _install()
    a.adata.obs['grp'] = pd.Categorical(['x'] * 30 + ['y'] * 30)
    r = c.post('/api/obs/crosstab', json={'a': 'cell_type', 'b': 'grp',
                                          'active_cell_indices': list(range(30))})
    assert r.status_code == 200, r.text
    assert sum(sum(row) for row in r.json()['counts']) == 30
    r = c.get('/api/obs/crosstab', params={'a': 'cell_type', 'b': 'grp'})
    assert r.status_code == 200 and sum(sum(row) for row in r.json()['counts']) == 60


# --- a pre-existing PCA bug this work exposed ----------------------------------

def test_an_explicit_gene_subset_is_used_whole_not_intersected_with_hvg():
    """scanpy's mask_var defaults to 'highly_variable' whenever the column
    exists, so PCA on 'other' silently used only the genes that were *also*
    pooled HVGs — and crashed when none were."""
    a = _adaptor()
    a.adata.var['highly_variable'] = np.array([True] * 10 + [False] * 70)
    a.adata.var['other'] = np.array([False] * 40 + [True] * 40)  # disjoint from the HVGs
    out = a.run_pca(n_comps=4, gene_subset='other')
    assert out['n_genes_used'] == 40
    loaded = ~np.isnan(a.adata.varm['PCs'][:, 0])
    assert loaded.sum() == 40
    assert np.abs(a.adata.varm['PCs'][loaded]).sum(axis=1).min() > 0


def test_the_component_count_is_clamped_to_the_hvgs_scanpy_will_actually_use():
    """The clamp looked at every gene while scanpy masked to the HVG column,
    so a small dataset asked arpack for more components than it had genes."""
    a = _adaptor(n_genes=30)
    a.adata.var['highly_variable'] = np.array([True] * 12 + [False] * 18)
    out = a.run_pca(n_comps=25)
    assert out['n_comps'] == 11 and out['n_genes_used'] == 12
    a.create_cell_subset('chondro', ACTIVE)
    out = a.run_pca(n_comps=25, cell_subset='chondro')
    assert out['n_comps'] == 11 and out['embedding_name'] == 'X_pca_chondro'


def test_a_subset_run_never_lands_on_the_datasets_own_key_whatever_graph_is_named():
    """The browser sends the default graph explicitly as 'connectivities'. That
    contributes no suffix, and the first version of the naming rule then wrote
    the subset's Leiden over the dataset's own leiden column."""
    a = _adaptor()
    _global_pipeline(a)
    before = a.adata.obs['leiden'].copy()
    a.create_cell_subset('chondro', ACTIVE)

    out = a.run_leiden(resolution=1.0, graph_key='connectivities', cell_subset='chondro')
    assert out['key_added'] == 'leiden_chondro'
    pd.testing.assert_series_equal(a.adata.obs['leiden'], before)

    out = a.run_umap(graph_key='connectivities', cell_subset='chondro')
    assert out['embedding_name'] == 'X_umap_chondro'
    assert not np.isnan(a.adata.obsm['X_umap']).any()

    # Another named graph is appended, so the dataset's leiden_other is safe too.
    a.adata.obsp['other_connectivities'] = a.adata.obsp['connectivities'].copy()
    out = a.run_leiden(resolution=1.0, graph_key='other_connectivities', cell_subset='chondro')
    assert out['key_added'] == 'leiden_chondro_other'
    # The subset's own graph adds nothing more.
    a.run_neighbors(n_neighbors=5, cell_subset='chondro')
    out = a.run_leiden(resolution=1.0, graph_key='chondro_connectivities', cell_subset='chondro')
    assert out['key_added'] == 'leiden_chondro'
