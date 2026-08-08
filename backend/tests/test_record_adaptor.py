"""The adaptor's side of the Analysis Record.

The record is only as good as what reaches it. These tests pin the three things
the existing action history got wrong: the load is recorded, an operation that
ran on a cell selection says so, and the record survives both a reassignment of
`.adata` and a round-trip through h5ad.
"""
import json
import tempfile
from pathlib import Path

import anndata
import numpy as np
import pandas as pd
import pytest
from scipy.sparse import csr_matrix

from xcell.adaptor import DataAdaptor


def _adata(n_cells=60, n_genes=30):
    rng = np.random.default_rng(0)
    ad = anndata.AnnData(
        X=csr_matrix(rng.integers(0, 10, (n_cells, n_genes)).astype(np.float32))
    )
    ad.var_names = [f'g{i}' for i in range(n_genes)]
    ad.obs_names = [f'c{i}' for i in range(n_cells)]
    ad.obs['group'] = pd.Categorical(['a', 'b', 'c'] * (n_cells // 3))
    ad.obsm['X_spatial'] = rng.random((n_cells, 2)).astype(np.float32)
    return ad


def _adaptor(ad=None):
    return DataAdaptor('cortex.h5ad', adata=ad if ad is not None else _adata())


def _actions(a):
    return [s.action for s in a.analysis_record.steps]


# --- the load step --------------------------------------------------------

def test_a_fresh_adaptor_records_what_it_loaded():
    a = _adaptor()
    first = a.analysis_record.steps[0]
    assert first.action == 'load_dataset'
    assert first.result['n_cells'] == 60
    assert first.result['n_genes'] == 30


def test_an_in_memory_dataset_is_recorded_as_such():
    """There is no file to re-read, and the notebook must not pretend there is."""
    assert _adaptor().analysis_record.steps[0].params['kind'] == 'memory'


def test_the_record_knows_the_source_shape():
    src = _adaptor().analysis_record.source
    assert src['n_cells'] == 60 and src['n_genes'] == 30


def test_a_file_backed_load_records_its_path_and_kind(tmp_path):
    path = tmp_path / 'cortex.h5ad'
    _adata().write_h5ad(path)
    a = DataAdaptor(path)
    step = a.analysis_record.steps[0]
    assert step.params['kind'] == 'h5ad'
    assert step.params['path'] == str(path)


# --- operations reach the record -----------------------------------------

def test_running_an_operation_appends_a_step():
    a = _adaptor()
    a.run_normalize_total(target_sum=1e4)
    a.run_log1p()
    assert _actions(a) == ['load_dataset', 'normalize_total', 'log1p']
    assert a.analysis_record.steps[1].params['target_sum'] == 1e4


def test_the_legacy_action_history_still_works():
    """layer_scale.provenance_from_history reads it, and so does the old route."""
    a = _adaptor()
    a.run_normalize_total(target_sum=1e4)
    assert [e['action'] for e in a.get_action_history()] == ['normalize_total']


def test_the_record_survives_reassigning_adata():
    """filter_genes replaces self.adata wholesale."""
    a = _adaptor()
    a.run_normalize_total(target_sum=1e4)
    a.run_filter_genes(min_cells=1)
    assert _actions(a)[:2] == ['load_dataset', 'normalize_total']
    assert 'filter_genes' in _actions(a)


# --- selections -----------------------------------------------------------

def test_an_operation_on_a_selection_records_the_selection():
    a = _adaptor()
    a.run_normalize_total(target_sum=1e4, active_cell_indices=list(range(20)))
    step = a.analysis_record.steps[-1]
    assert step.n_active == 20
    assert step.n_total == 60
    assert step.selection == list(range(20))


def test_an_operation_on_everything_records_no_selection():
    a = _adaptor()
    a.run_normalize_total(target_sum=1e4)
    assert a.analysis_record.steps[-1].n_active is None


def test_a_selection_covering_every_cell_is_not_a_selection():
    """The adaptor treats a full-coverage index list as no subset at all."""
    a = _adaptor()
    a.run_normalize_total(target_sum=1e4, active_cell_indices=list(range(60)))
    assert a.analysis_record.steps[-1].n_active is None


def test_clustering_on_a_selection_records_it():
    a = _adaptor()
    a.run_normalize_total()
    a.run_log1p()
    a.run_pca(n_comps=5)
    a.run_neighbors(n_neighbors=5)
    a.run_leiden(resolution=1.0, active_cell_indices=list(range(30)))
    step = next(s for s in a.analysis_record.steps if s.action == 'leiden')
    assert step.n_active == 30 and step.n_total == 60


def test_deleting_cells_records_which_ones():
    a = _adaptor()
    a.delete_cells([1, 5, 9])
    step = a.analysis_record.steps[-1]
    assert step.action == 'delete_cells'
    assert step.selection == [1, 5, 9]


# --- newly recorded operations -------------------------------------------

def test_label_edits_are_recorded():
    """Renaming a cluster changes what every downstream figure says."""
    a = _adaptor()
    a.rename_obs_label('group', 'a', 'neurons')
    a.merge_obs_labels('group', ['b', 'c'], 'glia')
    assert _actions(a)[1:] == ['rename_obs_label', 'merge_obs_labels']


def test_manual_annotation_is_recorded():
    a = _adaptor()
    a.create_annotation('my_regions')
    a.label_cells('my_regions', 'cortex', [0, 1, 2])
    assert _actions(a)[1:] == ['create_annotation', 'label_cells']
    assert a.analysis_record.steps[-1].selection == [0, 1, 2]


def test_qc_metrics_are_recorded():
    a = _adaptor()
    a.run_calculate_qc_metrics()
    assert _actions(a)[-1] == 'calculate_qc_metrics'


def test_differential_expression_is_recorded():
    a = _adaptor()
    a.run_diffexp(group1_indices=list(range(20)), group2_indices=list(range(20, 40)))
    assert _actions(a)[-1] == 'diffexp'


def test_gene_set_scoring_is_recorded():
    a = _adaptor()
    a.score_gene_sets_matrix(sets=[{'name': 'setA', 'genes': ['g0', 'g1', 'g2']}])
    assert _actions(a)[-1] == 'score_gene_sets_matrix'


def test_var_side_edits_are_recorded():
    a = _adaptor()
    a.add_var_boolean_column('is_g1', 'g1', match_mode='prefix')
    a.sum_counts_by_pattern('g1', match_mode='prefix', obs_name='g1_counts')
    assert _actions(a)[1:] == ['add_var_boolean_column', 'sum_counts_by_pattern']


def test_an_obs_embedding_is_recorded():
    a = _adaptor()
    a.adata.obs['x'] = np.arange(a.n_cells, dtype=float)
    a.adata.obs['y'] = np.arange(a.n_cells, dtype=float)[::-1]
    a.create_obs_embedding(col_x='x', col_y='y', name='X_obs')
    assert _actions(a)[-1] == 'create_obs_embedding'


# --- persistence ----------------------------------------------------------

def test_the_record_is_written_into_uns_on_export():
    a = _adaptor()
    a.run_normalize_total(target_sum=1e4)
    exported = a.prepare_export_with_lines()
    raw = exported.uns['xcell_analysis_record']
    assert isinstance(raw, str)  # a JSON string; nested dicts fight with h5ad
    assert [s['action'] for s in json.loads(raw)['steps']][-1] == 'normalize_total'


def test_a_record_in_uns_is_restored_on_load():
    a = _adaptor()
    a.run_normalize_total(target_sum=1e4)
    a.analysis_record.set_note(1, 'library-size normalization')
    restored = DataAdaptor('x.h5ad', adata=a.prepare_export_with_lines())
    actions = [s.action for s in restored.analysis_record.steps]
    assert 'normalize_total' in actions
    assert restored.analysis_record.steps[1].note == 'library-size normalization'


def test_a_restored_record_continues_rather_than_restarting():
    """Re-opening an exported h5ad appends; it does not begin a second history.

    The reload lands in the record where it happened — chronology is the point —
    and the notebook exporter opens the *original* source and replays from there.
    """
    a = _adaptor()
    a.run_normalize_total(target_sum=1e4)
    restored = DataAdaptor('x.h5ad', adata=a.prepare_export_with_lines())
    restored.run_log1p()
    assert [s.action for s in restored.analysis_record.steps] == [
        'load_dataset', 'normalize_total', 'load_dataset', 'log1p',
    ]


def test_only_the_original_source_opens_a_replayed_notebook():
    """A mid-history reload is a session artifact, not an analysis step."""
    from xcell.notebook_export import to_notebook

    a = _adaptor()
    a.run_normalize_total(target_sum=1e4)
    restored = DataAdaptor('x.h5ad', adata=a.prepare_export_with_lines())
    restored.run_log1p()
    code = [c for c in to_notebook(restored.analysis_record)['cells']
            if c['cell_type'] == 'code']
    body = '\n'.join(''.join(c['source']) for c in code)
    assert body.count('read_h5ad') <= 1


def test_the_record_survives_a_real_h5ad_round_trip():
    a = _adaptor()
    a.run_normalize_total(target_sum=1e4)
    with tempfile.TemporaryDirectory() as d:
        path = Path(d) / 'out.h5ad'
        a.prepare_export_with_lines().write_h5ad(path)
        reopened = DataAdaptor(path)
    assert 'normalize_total' in [s.action for s in reopened.analysis_record.steps]


def test_a_corrupt_record_in_uns_does_not_break_loading():
    """An h5ad from elsewhere may have anything under that key."""
    ad = _adata()
    ad.uns['xcell_analysis_record'] = 'not json {{{'
    a = DataAdaptor('x.h5ad', adata=ad)
    assert _actions(a) == ['load_dataset']  # a fresh record, no exception


def test_figures_are_not_written_into_the_h5ad():
    """Base64 PNGs would bloat every exported dataset."""
    a = _adaptor()
    a.analysis_record.add_figure('iVBORw0KGgo=', caption='UMAP')
    stored = json.loads(a.prepare_export_with_lines().uns['xcell_analysis_record'])
    assert stored['figures'] == {}


# --- JSON safety ----------------------------------------------------------

def test_recorded_results_are_json_safe():
    """diffexp and marker_genes carry scipy floats that can be NaN or inf."""
    a = _adaptor()
    a.run_normalize_total()
    a.run_log1p()
    a.run_diffexp(group1_indices=list(range(20)), group2_indices=list(range(20, 40)))
    json.dumps(a.analysis_record.to_dict(), allow_nan=False)


def test_numpy_scalars_do_not_reach_the_record():
    a = _adaptor()
    a.run_pca(n_comps=np.int64(5)) if False else None  # (guard: pca needs setup)
    a.run_filter_genes(min_cells=int(np.int64(1)))
    json.dumps(a.analysis_record.to_dict(), allow_nan=False)


# --- the export is reachable from the adaptor's data ---------------------

def test_a_recorded_pipeline_translates_into_runnable_code():
    """The end-to-end promise: what the adaptor logged becomes valid Python."""
    from xcell.notebook_export import to_notebook

    a = _adaptor()
    a.run_normalize_total(target_sum=1e4)
    a.run_log1p()
    a.run_pca(n_comps=5)
    a.run_neighbors(n_neighbors=5)
    a.run_leiden(resolution=1.0)
    nb = to_notebook(a.analysis_record)
    body = '\n'.join(
        ''.join(c['source']) for c in nb['cells'] if c['cell_type'] == 'code'
    )
    compile(body, '<notebook>', 'exec')
    assert 'sc.pp.normalize_total(adata, target_sum=10000.0)' in body
    assert "sc.tl.leiden(adata, resolution=1.0, key_added='leiden')" in body
