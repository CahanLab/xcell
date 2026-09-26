"""Adaptor tests for the figure registry: create/update/delete, provenance, data, attach."""
import json

import anndata
import numpy as np
import pandas as pd
import pytest
from scipy.sparse import csr_matrix

from xcell import enrichment_figures as ef
from xcell import gene_set_sources as gss
from xcell.adaptor import DataAdaptor, FigureInputMissing

GENES = ['Col1a1', 'Col1a2', 'Col3a1', 'Ptprc', 'Cd3e', 'Cd19', 'Hoxd13', 'Meis1'] + [f'G{i}' for i in range(24)]
LIB = [{'source': 'msigdb', 'id': 'toy', 'species': 'mouse'}]


@pytest.fixture(autouse=True)
def _cache(tmp_path, monkeypatch):
    gss.set_cache_dir(tmp_path / 'cache')
    monkeypatch.setattr(gss, 'fetch_text', lambda url, **kw: (_ for _ in ()).throw(AssertionError(url)))
    monkeypatch.setattr(gss, 'fetch_json', lambda url, **kw: (_ for _ in ()).throw(AssertionError(url)))
    gss.save_library({
        'source': 'msigdb', 'id': 'toy', 'name': 'Toy', 'species': 'mouse', 'version': '1', 'n_sets': 3,
        'sets': [{'name': 'COLLAGEN', 'genes': ['Col1a1', 'Col1a2', 'Col3a1', 'G0', 'G1']},
                 {'name': 'IMMUNE', 'genes': ['Ptprc', 'Cd3e', 'Cd19', 'G2', 'G3']},
                 {'name': 'HOX', 'genes': ['Hoxd13', 'Meis1', 'G4', 'G5', 'G6']}]})
    yield
    gss.set_cache_dir(None)


def _adata():
    rng = np.random.default_rng(0)
    labels = np.array(['a'] * 30 + ['b'] * 30 + ['c'] * 30)
    lam = np.full((labels.size, len(GENES)), 1.0)
    for g, hi in (('a', ('Col1a1', 'Col1a2', 'Col3a1')), ('b', ('Ptprc', 'Cd3e', 'Cd19')), ('c', ('Hoxd13', 'Meis1'))):
        for gene in hi:
            lam[labels == g, GENES.index(gene)] = 8.0
    ad = anndata.AnnData(X=csr_matrix(rng.poisson(lam).astype(np.float32)))
    ad.var_names = GENES
    ad.obs['grp'] = pd.Categorical(labels)
    ad.obs['batch'] = pd.Categorical(['x', 'y'] * 45)
    return ad


def _with_batch():
    a = DataAdaptor('x.h5ad', adata=_adata())
    compute_fn, apply_fn = a.prepare_gsea_batch('grp', libraries=LIB, min_set_size=2, n_perm=60)
    col = apply_fn(compute_fn(lambda f, m: None))
    return a, col


def test_create_validates_kind_and_inputs_and_allocates_ids():
    a, col = _with_batch()
    with pytest.raises(ValueError, match='enrichment_heatmap'):
        a.create_figure('pie', inputs={})
    with pytest.raises(ValueError, match='enrichment_keys'):
        a.create_figure('enrichment_heatmap', inputs={})
    with pytest.raises(ValueError, match='ghost'):
        a.create_figure('enrichment_heatmap', inputs={'enrichment_keys': ['ghost']})
    with pytest.raises(ValueError, match='column_a'):
        a.create_figure('composition_barplot', inputs={'column_a': 'nope', 'column_b': 'grp'})
    f1 = a.create_figure('enrichment_heatmap', inputs={'enrichment_keys': [col['key']]})
    f2 = a.create_figure('enrichment_network', title='Net', inputs={'enrichment_keys': [col['key']]}, params={'top_n': 2})
    assert (f1['id'], f2['id']) == ('fig_1', 'fig_2') and f2['title'] == 'Net'
    assert f1['params']['padj_max'] == 0.05 and f1['params']['value'] == 'nes' and f1['params']['top_n'] == 5
    assert f2['params']['top_n'] == 2 and f2['params']['layout'] == 'force'
    assert f1['title']                                   # a default title was made
    a.delete_figure('fig_2')
    assert a.create_figure('enrichment_heatmap', inputs={'enrichment_keys': [col['key']]})['id'] == 'fig_3'
    assert [f['id'] for f in a.list_figures()] == ['fig_3', 'fig_1']
    assert json.loads(a.adata.uns['xcell_figures']['fig_1'])['kind'] == 'enrichment_heatmap'
    assert 'fig_2' not in a.adata.uns['xcell_figures']
    with pytest.raises(KeyError):
        a.get_figure('fig_2')
    acts = [h['action'] for h in a._action_history]
    assert acts[-4:] == ['figure_create', 'figure_create', 'figure_delete', 'figure_create']


def test_provenance_links_the_batch_step_and_the_create_step():
    a, col = _with_batch()
    batch_step = [s for s in a.analysis_record.steps if s.action == 'enrichment_gsea_batch'][0].index
    f = a.create_figure('enrichment_heatmap', inputs={'enrichment_keys': [col['key']]})
    assert batch_step in f['provenance']['steps']
    create_step = a.analysis_record.steps[-1]
    assert create_step.action == 'figure_create' and f['provenance']['created_step'] == create_step.index
    assert a.get_figure(f['id'])['provenance']['created_step'] == create_step.index
    # a figure on one member key also links the batch step (it lists members)
    f2 = a.create_figure('enrichment_heatmap', inputs={'enrichment_keys': [col['members']['a']]})
    assert batch_step in f2['provenance']['steps']


def test_update_merges_params_and_logs_only_changes():
    a, col = _with_batch()
    f = a.create_figure('enrichment_heatmap', inputs={'enrichment_keys': [col['key']]})
    u = a.update_figure(f['id'], params={'padj_max': 0.5}, title='Programs by cluster')
    assert u['params']['padj_max'] == 0.5 and u['params']['top_n'] == 5 and u['title'] == 'Programs by cluster'
    assert u['updated_at'] >= u['created_at']
    last = a._action_history[-1]
    assert last['action'] == 'figure_update' and last['params'] == {'id': f['id'], 'title': 'Programs by cluster', 'params': {'padj_max': 0.5}}
    with pytest.raises(ValueError, match='unknown'):
        a.update_figure(f['id'], params={'nonsense': 1})
    with pytest.raises(KeyError):
        a.update_figure('fig_99', title='x')


def test_figure_data_matches_pure_assembly_and_override_does_not_persist():
    a, col = _with_batch()
    f = a.create_figure('enrichment_heatmap', inputs={'enrichment_keys': [col['key']]},
                        params={'padj_max': 1.0, 'top_n': 3, 'collapse_jaccard': None})
    n_before = len(a._action_history)
    data = a.figure_data(f['id'])
    assert [c['label'] for c in data['cols']] == ['a', 'b', 'c']
    expected = ef.assemble_matrix({g: col['member_results'][g] for g in ['a', 'b', 'c']}, value='nes', padj_max=1.0,
                                  top_n=3, direction='both', collapse_jaccard=None, row_order='peak',
                                  col_order='input', members=None)
    assert [r['name'] for r in data['rows']] == [r['name'] for r in expected['rows']]
    assert data['values'] == expected['values']
    over = a.figure_data(f['id'], params_override={'top_n': 1})
    assert len(over['rows']) <= len(data['rows'])
    assert a.get_figure(f['id'])['params']['top_n'] == 3 and len(a._action_history) == n_before
    net = a.create_figure('enrichment_network', inputs={'enrichment_keys': [col['key']]}, params={'padj_max': 1.0})
    nd = a.figure_data(net['id'])
    assert {n['kind'] for n in nd['nodes']} == {'group', 'set'} and nd['edges']
    # single results as columns, labelled by their contrast
    f3 = a.create_figure('enrichment_heatmap', inputs={'enrichment_keys': [col['members']['a'], col['members']['b']]}, params={'padj_max': 1.0})
    assert [c['label'] for c in a.figure_data(f3['id'])['cols']] == ['grp: a vs rest', 'grp: b vs rest']


def test_figure_data_after_input_deleted_raises_but_figure_stays_listed():
    a, col = _with_batch()
    f = a.create_figure('enrichment_heatmap', inputs={'enrichment_keys': [col['key']]})
    a.delete_enrichment_result(col['key'])
    with pytest.raises(FigureInputMissing, match=col['key']):
        a.figure_data(f['id'])
    assert [x['id'] for x in a.list_figures()] == [f['id']]
    # the other kinds draw from live .obs, so deleting an enrichment result never touches them
    assert 'counts' in a.figure_data(a.create_figure('composition_barplot', inputs={'column_a': 'grp', 'column_b': 'batch'})['id'])


def test_attach_links_record_figure_to_the_figure():
    a, col = _with_batch()
    f = a.create_figure('enrichment_heatmap', inputs={'enrichment_keys': [col['key']]})
    out = a.attach_figure_to_record(f['id'], 'iVBORw0KGgo=', caption='Fig 1')
    fig = a.analysis_record.figures[out['record_figure_id']]
    assert fig.figure_id == f['id'] and fig.caption == 'Fig 1' and out['step_index'] == f['provenance']['created_step']
    assert fig.to_dict()['figure_id'] == f['id']
    with pytest.raises(KeyError):
        a.attach_figure_to_record('fig_99', 'iVBORw0KGgo=')


# --- Part C: barplot and expression heatmap kinds -------------------------------

def _adata_c():
    ad = _adata()
    ad.obs['subset_half'] = [True] * 40 + [False] * 50
    ad.uns['xcell_cell_subsets'] = {'half': {'n_cells': 40, 'created_at': 't', 'origin': 'test'}}
    return ad


def test_barplot_figure_data_matches_crosstab_and_honours_cells():
    a = DataAdaptor('x.h5ad', adata=_adata_c())
    f = a.create_figure('composition_barplot', inputs={'column_a': 'grp', 'column_b': 'batch'})
    d = a.figure_data(f['id'])
    assert d['a_categories'] == a.crosstab('grp', 'batch')['a_categories'] and d['counts'] == a.crosstab('grp', 'batch')['counts']
    assert f['params'] == {'order': 'category', 'share_of': None, 'normalize': True, 'min_cells': 0, 'show_values': False}
    g = a.create_figure('composition_barplot', inputs={'column_a': 'grp', 'column_b': 'batch', 'cell_subset': 'half'})
    assert a.figure_data(g['id'])['n_cells'] == 40 and g['inputs']['cell_subset'] == 'half'
    h = a.create_figure('composition_barplot', inputs={'column_a': 'grp', 'column_b': 'batch', 'cell_indices': list(range(10))})
    assert a.figure_data(h['id'])['n_cells'] == 10 and h['inputs']['cell_indices'] == list(range(10))
    with pytest.raises(KeyError):
        a.create_figure('composition_barplot', inputs={'column_a': 'grp', 'column_b': 'batch', 'cell_subset': 'ghost'})


def test_expression_heatmap_figure_data_and_validation():
    a = DataAdaptor('x.h5ad', adata=_adata_c())
    sets = [{'name': 'collagen', 'genes': ['Col1a1', 'Col1a2', 'Col3a1']}, {'name': 'immune', 'genes': ['Ptprc', 'Cd3e', 'Cd19']}]
    f = a.create_figure('expression_heatmap', inputs={'gene_sets': sets, 'obs_column': 'grp'})
    d = a.figure_data(f['id'])
    assert len(d['row_labels']) == 6 and len(d['matrix']) == 6 and [c['name'] for c in d['column_groups']] == ['a', 'b', 'c']
    assert f['params']['cell_ordering'] == 'category' and f['params']['n_bins'] == 50
    agg = a.figure_data(f['id'], params_override={'aggregate_gene_sets': True})
    assert len(agg['row_labels']) == 2
    sub = a.create_figure('expression_heatmap', inputs={'gene_sets': sets, 'obs_column': 'grp', 'cell_subset': 'half'})
    assert a.figure_data(sub['id'])['n_cells'] == 40
    with pytest.raises(ValueError, match='gene_sets'):
        a.create_figure('expression_heatmap', inputs={'gene_sets': []})
    with pytest.raises(ValueError, match='nope'):
        a.create_figure('expression_heatmap', inputs={'gene_sets': sets, 'line_name': 'nope'})


# --- review fix pass -----------------------------------------------------------

def test_codegen_replays_figure_create_update_delete_against_a_real_adaptor():
    from xcell import codegen
    a, col = _with_batch()
    f = a.create_figure('enrichment_heatmap', inputs={'enrichment_keys': [col['key']]})
    a.update_figure(f['id'], title='T', params={'top_n': 2})
    a.delete_figure(f['id'])
    steps = [s for s in a.analysis_record.steps if s.action.startswith('figure_')]
    assert [s.action for s in steps] == ['figure_create', 'figure_update', 'figure_delete']
    b, col2 = _with_batch()                      # a fresh session replays the notebook lines
    ns = {'xa': b, 'adata': b.adata}
    exec('\n'.join(codegen.notebook_preamble()), ns)
    for s in steps:
        t = codegen.translate(s)
        assert t.fidelity == 'xcell'
        for line in t.code:
            exec(line, ns)                       # a wrong kwarg name raises TypeError here
    assert b.get_enrichment_results() and 'fig_1' not in b._figure_store()


def test_figure_data_rejects_an_input_key_reused_by_another_run():
    a, col = _with_batch()
    f = a.create_figure('enrichment_heatmap', inputs={'enrichment_keys': [col['key']]})
    assert f['inputs']['enrichment_keys'] == [col['key']] and f['input_ids'][col['key']]
    a.delete_enrichment_result(col['key'])
    compute_fn, apply_fn = a.prepare_gsea_batch('grp', groups=['a', 'c'], libraries=LIB, min_set_size=2, n_perm=60)
    again = apply_fn(compute_fn(lambda f, m: None))
    assert again['key'] == col['key']            # the key came back around
    with pytest.raises(FigureInputMissing, match='replaced'):
        a.figure_data(f['id'])
    assert [x['id'] for x in a.list_figures()] == [f['id']]


def test_columns_are_keyed_by_input_and_group_so_labels_never_collide():
    a, col = _with_batch()
    compute_fn, apply_fn = a.prepare_gsea_batch('grp', libraries=LIB, min_set_size=2, n_perm=60, seed=1)
    col2 = apply_fn(compute_fn(lambda f, m: None))
    f = a.create_figure('enrichment_heatmap', inputs={'enrichment_keys': [col['key'], col2['key']]}, params={'padj_max': 1.0})
    d = a.figure_data(f['id'])
    assert len(d['cols']) == 6
    assert [c['key'] for c in d['cols']] == [f"{col['key']}:{g}" for g in 'abc'] + [f"{col2['key']}:{g}" for g in 'abc']
    assert len({c['label'] for c in d['cols']}) == 6         # disambiguated for display


def test_attach_finds_its_create_step_or_stays_standalone():
    a, col = _with_batch()
    f = a.create_figure('enrichment_heatmap', inputs={'enrichment_keys': [col['key']]})
    a._log_action('noop', {}, {})
    out = a.attach_figure_to_record(f['id'], 'iVBORw0KGgo=')
    assert a.analysis_record.steps[out['step_index']].action == 'figure_create'
    a.analysis_record.clear()
    a._log_action('noop', {}, {})
    out2 = a.attach_figure_to_record(f['id'], 'iVBORw0KGgo=')
    assert out2['step_index'] is None
    assert a.analysis_record.figures[out2['record_figure_id']].step_index is None


def test_failed_batch_leaves_no_reserved_record():
    a = DataAdaptor('x.h5ad', adata=_adata())
    with pytest.raises(ValueError):
        a.run_overlap_enrichment_batch('grp', top_n=6, libraries=LIB, min_set_size=50, min_overlap=1)
    assert a.get_enrichment_results() == []
    assert 'ora_grp_batch' not in a._enrichment_store()


def test_barplot_provenance_does_not_link_every_analysis_on_its_column():
    a, col = _with_batch()                        # an enrichment step with obs_column 'grp'
    f = a.create_figure('composition_barplot', inputs={'column_a': 'grp', 'column_b': 'batch'})
    assert f['provenance']['steps'] == []


# --- second review fix pass ----------------------------------------------------

def test_heatmap_figure_refuses_to_draw_when_its_line_column_or_subset_is_gone():
    ad = _adata_c()
    a = DataAdaptor('x.h5ad', adata=ad)
    a.set_lines([{'name': 'L1', 'embedding': 'X', 'points': [[0, 0], [1, 1]], 'draw_type': 'line', 'closed': False}])
    sets = [{'name': 'collagen', 'genes': ['Col1a1', 'Col1a2', 'Col3a1']}]
    f = a.create_figure('expression_heatmap', inputs={'gene_sets': sets, 'obs_column': 'grp', 'line_name': 'L1'},
                        params={'cell_ordering': 'category'})
    a.set_lines([])
    with pytest.raises(FigureInputMissing, match='L1'):
        a.figure_data(f['id'])
    g = a.create_figure('expression_heatmap', inputs={'gene_sets': sets, 'obs_column': 'batch'})
    del a.adata.obs['batch']
    with pytest.raises(FigureInputMissing, match='batch'):
        a.figure_data(g['id'])
    h = a.create_figure('expression_heatmap', inputs={'gene_sets': sets, 'cell_subset': 'half'})
    a.adata.uns['xcell_cell_subsets'] = {}
    with pytest.raises(FigureInputMissing, match='half'):
        a.figure_data(h['id'])
    # an ordering that needs a line the inputs do not carry is a 400, on create and on preview
    with pytest.raises(ValueError, match='line'):
        a.create_figure('expression_heatmap', inputs={'gene_sets': sets}, params={'cell_ordering': 'line_position'})
    k = a.create_figure('expression_heatmap', inputs={'gene_sets': sets})
    with pytest.raises(ValueError, match='line'):
        a.figure_data(k['id'], params_override={'cell_ordering': 'line_distance'})
    with pytest.raises(ValueError, match='column'):
        a.figure_data(k['id'], params_override={'cell_ordering': 'category'})


def test_heatmap_figure_records_the_transform_the_tab_drew_with():
    a = DataAdaptor('x.h5ad', adata=_adata_c())
    sets = [{'name': 'collagen', 'genes': ['Col1a1', 'Col1a2', 'Col3a1']}]
    f = a.create_figure('expression_heatmap', inputs={'gene_sets': sets, 'transform': 'log1p'})
    assert f['inputs']['transform'] == 'log1p'
    raw = a.figure_data(a.create_figure('expression_heatmap', inputs={'gene_sets': sets})['id'])
    logged = a.figure_data(f['id'])
    assert raw['matrix'] != logged['matrix']
    with pytest.raises(ValueError, match='transform'):
        a.create_figure('expression_heatmap', inputs={'gene_sets': sets, 'transform': 'sqrt'})


def test_cell_indices_are_deduplicated_integers_and_capped_in_the_record():
    from xcell.analysis_record import SELECTION_CAP
    a = DataAdaptor('x.h5ad', adata=_adata_c())
    f = a.create_figure('composition_barplot', inputs={'column_a': 'grp', 'column_b': 'batch', 'cell_indices': [3, 3, 4, 3]})
    assert f['inputs']['cell_indices'] == [3, 4] and a.figure_data(f['id'])['n_cells'] == 2
    with pytest.raises(ValueError, match='integer'):
        a.create_figure('composition_barplot', inputs={'column_a': 'grp', 'column_b': 'batch', 'cell_indices': [1.7, 2]})
    summary = [s for s in a.list_figures() if s['id'] == f['id']][0]
    assert 'cell_indices' not in summary['inputs'] and summary['inputs']['n_cell_indices'] == 2
    big = list(range(a.n_cells)) * 1                                # small dataset: simulate the cap
    a.SELECTION_CAP_FOR_TESTS = 5
    import xcell.adaptor as mod
    orig = mod.SELECTION_CAP
    mod.SELECTION_CAP = 5
    try:
        g = a.create_figure('composition_barplot', inputs={'column_a': 'grp', 'column_b': 'batch', 'cell_indices': list(range(10))})
    finally:
        mod.SELECTION_CAP = orig
    step = a.analysis_record.steps[-1]
    assert step.action == 'figure_create' and 'cell_indices' not in step.params['inputs']
    assert step.params['inputs']['n_cell_indices'] == 10 and step.params['inputs']['cell_indices_omitted'] is True
    assert a.get_figure(g['id'])['inputs']['cell_indices'] == list(range(10))   # the figure itself keeps them
    assert SELECTION_CAP >= 10


def test_codegen_replay_binds_figures_by_id_even_when_ids_differ():
    from xcell import codegen
    a, col = _with_batch()
    f = a.create_figure('enrichment_heatmap', inputs={'enrichment_keys': [col['key']]})
    a.update_figure(f['id'], title='T')
    a.delete_figure(f['id'])
    steps = [s for s in a.analysis_record.steps if s.action.startswith('figure_')]
    b, col2 = _with_batch()
    pre = b.create_figure('enrichment_network', inputs={'enrichment_keys': [col2['key']]})   # already holds fig_1
    ns = {'xa': b, 'adata': b.adata}
    exec('\n'.join(codegen.notebook_preamble()), ns)
    for s in steps:
        for line in codegen.translate(s).code:
            exec(line, ns)
    assert b.get_figure(pre['id'])['title'] != 'T'            # the pre-existing figure was left alone
    assert [x['id'] for x in b.list_figures()] == [pre['id']]  # the replayed one was created, edited, deleted
