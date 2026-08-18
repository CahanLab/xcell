"""Translating a recorded step into prose and Python.

This is where reproducibility actually lives, so the tests are about honesty as
much as correctness: code that claims `exact` must be the call the adaptor
really makes, a step that ran on a selection must say so, and an action nobody
taught the registry about must degrade to prose rather than vanish.
"""
import re
from pathlib import Path

from xcell.analysis_record import AnalysisRecord
from xcell.codegen import (
    EXACT,
    MANUAL,
    XCELL,
    registered_actions,
    translate,
)

ADAPTOR_SRC = Path(__file__).resolve().parents[1] / 'xcell' / 'adaptor.py'


def _step(action, params=None, result=None, **kw):
    """One step, in isolation, exactly as the recorder would build it."""
    r = AnalysisRecord()
    return r.add_step(action, params or {}, result or {}, **kw)


# --- coverage -------------------------------------------------------------

def test_every_logged_action_has_a_registry_entry():
    """Adding a _log_action without a spec silently degrades exports to prose.

    Scanning the source is blunt, but it is the only check that fails when
    someone adds a new operation and forgets this module exists.
    """
    src = ADAPTOR_SRC.read_text()
    logged = set(re.findall(r"_log_action\(\s*\n?\s*['\"](\w+)['\"]", src))
    assert logged, 'the scan found nothing — the regex has drifted'
    missing = logged - registered_actions()
    assert not missing, f"actions logged but not translatable: {sorted(missing)}"


def test_the_registry_does_not_invent_actions():
    """Entries for actions nobody logs are dead weight and mislead readers."""
    src = ADAPTOR_SRC.read_text()
    logged = set(re.findall(r"_log_action\(\s*\n?\s*['\"](\w+)['\"]", src))
    extra = registered_actions() - logged - {'load_dataset'}
    assert not extra, f"registry entries for actions never logged: {sorted(extra)}"


# --- the exact tier -------------------------------------------------------

def test_normalize_total_emits_the_scanpy_call_the_adaptor_makes():
    t = translate(_step('normalize_total', {'target_sum': 10000.0}, {'status': 'completed'}))
    assert t.code == ['sc.pp.normalize_total(adata, target_sum=10000.0)']
    assert t.fidelity == EXACT
    assert 'import scanpy as sc' in t.imports


def test_normalize_total_without_a_target_sum_omits_the_argument():
    """scanpy's default (the median) is not the same as target_sum=None."""
    t = translate(_step('normalize_total', {}, {}))
    assert t.code == ['sc.pp.normalize_total(adata)']


def test_log1p_emits_the_scanpy_call():
    t = translate(_step('log1p', {}, {}))
    assert t.code == ['sc.pp.log1p(adata)']
    assert t.fidelity == EXACT


def test_leiden_emits_resolution_and_key():
    t = translate(_step('leiden', {'resolution': 0.5, 'key_added': 'clusters'},
                        {'n_clusters': 7}))
    assert t.code == ["sc.tl.leiden(adata, resolution=0.5, key_added='clusters')"]
    assert t.fidelity == EXACT


def test_umap_emits_its_three_parameters():
    t = translate(_step('umap', {'min_dist': 0.3, 'spread': 1.0, 'n_components': 2}, {}))
    assert t.code == ['sc.tl.umap(adata, min_dist=0.3, spread=1.0, n_components=2)']


def test_neighbors_passes_through_only_the_logged_kwargs():
    t = translate(_step('neighbors', {'n_neighbors': 15, 'metric': 'euclidean'}, {}))
    assert t.code == ["sc.pp.neighbors(adata, n_neighbors=15, metric='euclidean')"]


def test_pca_without_a_gene_subset_is_a_one_liner():
    t = translate(_step('pca', {'n_comps': 50, 'svd_solver': 'arpack', 'gene_subset': None}, {}))
    assert t.code == ["sc.tl.pca(adata, n_comps=50, svd_solver='arpack')"]
    assert t.fidelity == EXACT


def test_pca_on_a_var_column_subsets_the_genes_first():
    """xcell copies out the gene subset, runs PCA there, and copies X_pca back."""
    t = translate(_step('pca', {'n_comps': 30, 'svd_solver': 'arpack',
                                'gene_subset': 'highly_variable'}, {}))
    body = '\n'.join(t.code)
    assert "adata.var['highly_variable']" in body
    assert 'sc.tl.pca(' in body
    assert "adata.obsm['X_pca']" in body
    assert t.fidelity == EXACT


def test_pca_on_an_explicit_gene_list_falls_back_to_the_xcell_api():
    t = translate(_step('pca', {'n_comps': 30, 'svd_solver': 'arpack',
                                'gene_subset': ['A', 'B']}, {}))
    assert t.fidelity == XCELL


def test_filter_cells_emits_one_scanpy_call_per_threshold():
    # scanpy accepts exactly one threshold per call; a single splatted call
    # would raise in the exported notebook. Sequential calls compute the same
    # AND the adaptor applies.
    t = translate(_step('filter_cells', {'min_genes': 200, 'max_counts': 50000}, {}))
    assert t.code == ['sc.pp.filter_cells(adata, max_counts=50000)',
                      'sc.pp.filter_cells(adata, min_genes=200)']
    assert t.fidelity == EXACT


def test_filter_cells_with_one_threshold_is_a_single_call():
    t = translate(_step('filter_cells', {'min_genes': 200}, {}))
    assert t.code == ['sc.pp.filter_cells(adata, min_genes=200)']


def test_highly_variable_genes_emits_every_logged_parameter():
    t = translate(_step('highly_variable_genes', {
        'n_top_genes': 2000, 'min_mean': 0.0125, 'max_mean': 3.0,
        'min_disp': 0.5, 'flavor': 'seurat', 'subset': False,
    }, {'n_highly_variable': 2000}))
    line = t.code[0]
    assert line.startswith('sc.pp.highly_variable_genes(adata, ')
    assert 'n_top_genes=2000' in line and "flavor='seurat'" in line


def test_qc_metrics_keeps_an_explicit_percent_top_none():
    """Dropping this None silently changes the call.

    xcell passes percent_top=None, meaning "no top-N columns". scanpy's own
    default is (50, 100, 200, 500), which raises IndexError on any dataset with
    fewer than 500 genes — so omitting the argument produces a notebook that
    crashes where xcell did not.
    """
    t = translate(_step('calculate_qc_metrics',
                        {'qc_vars': [], 'percent_top': None, 'log1p': True}, {}))
    assert 'percent_top=None' in t.code[0]


def test_qc_metrics_still_forwards_a_real_percent_top():
    t = translate(_step('calculate_qc_metrics',
                        {'qc_vars': [], 'percent_top': [50], 'log1p': True}, {}))
    assert 'percent_top=[50]' in t.code[0]


def test_spatial_neighbors_emits_the_squidpy_call():
    t = translate(_step('spatial_neighbors', {
        'n_neighs': 6, 'coord_type': 'generic', 'spatial_key': 'spatial', 'delaunay': False,
    }, {}))
    assert t.code[0].startswith('sq.gr.spatial_neighbors(adata, ')
    assert 'import squidpy as sq' in t.imports


def test_a_section_aware_spatial_graph_warns_that_the_call_is_incomplete():
    """xcell drops cross-section edges after squidpy returns; the call alone
    would not reproduce that."""
    t = translate(_step('spatial_neighbors', {
        'n_neighs': 6, 'coord_type': 'generic', 'spatial_key': 'spatial',
        'delaunay': False, 'section_col': 'sample',
    }, {}))
    assert any('section' in w.lower() for w in t.warnings)


# --- the xcell tier -------------------------------------------------------

def test_an_xcell_only_operation_uses_the_xcell_api():
    t = translate(_step('ligrec', {'interactions': ['a|b'], 'write_significance': True}, {}))
    assert t.fidelity == XCELL
    assert any('xa.finalize_ligrec(' in line for line in t.code)
    assert 'from xcell.adaptor import DataAdaptor' in t.imports


def test_two_phase_operations_are_driven_to_completion():
    """prepare_* returns (compute, apply); calling prepare alone does nothing."""
    t = translate(_step('contourize', {
        'genes': ['Col1a1'], 'contour_levels': 6, 'log_transform': True,
        'smooth_sigma': 2.0, 'grid_res': 200, 'annotation_key': 'contour_1',
    }, {}))
    assert t.fidelity == XCELL
    assert any('_xcell_run(' in line and 'prepare_contourize' in line for line in t.code)


# --- the manual tier ------------------------------------------------------

def test_an_unknown_action_becomes_prose_not_silence():
    t = translate(_step('some_future_tool', {'alpha': 3}, {'n': 1}))
    assert t.fidelity == MANUAL
    assert t.code == []
    assert 'some_future_tool' in t.title
    assert 'alpha' in t.summary  # the parameters are still shown


def test_deleting_cells_is_reproducible_when_the_selection_was_kept():
    t = translate(_step('delete_cells', {'n_indices': 3}, {'n_cells_deleted': 3},
                        selection=[1, 4, 9], n_total=100))
    assert t.fidelity == EXACT
    assert any('SELECTIONS[' in line for line in t.code)


def test_deleting_cells_is_not_reproducible_without_the_selection():
    """Over the cap the indices are gone, so there is nothing honest to emit."""
    t = translate(_step('delete_cells', {'n_indices': 90000}, {'n_cells_deleted': 90000}))
    assert t.fidelity == MANUAL
    assert t.code == []


# --- selections -----------------------------------------------------------

def test_a_step_that_ran_on_a_selection_says_so():
    t = translate(_step('leiden', {'resolution': 1.0, 'key_added': 'leiden'}, {},
                        selection=list(range(40)), n_total=100))
    assert any('40' in w and '100' in w for w in t.warnings)


def test_a_whole_dataset_step_carries_no_selection_warning():
    t = translate(_step('leiden', {'resolution': 1.0, 'key_added': 'leiden'}, {}))
    assert t.warnings == []


# --- summaries ------------------------------------------------------------

def test_a_whole_float_parameter_reads_as_a_whole_number():
    """target_sum arrives as 10000.0; "10,000.0 counts" reads like a typo."""
    t = translate(_step('normalize_total', {'target_sum': 10000.0}, {}))
    assert '10,000 counts' in t.summary
    assert '10,000.0' not in t.summary


def test_the_summary_reports_the_recorded_outcome():
    """The boiler-plate annotation uses real numbers, not the parameters alone."""
    t = translate(_step('leiden', {'resolution': 1.0, 'key_added': 'leiden'},
                        {'n_clusters': 12}))
    assert '12' in t.summary and 'leiden' in t.summary


def test_the_summary_survives_a_result_missing_its_usual_keys():
    """Records restored from an older h5ad may not carry today's result shape."""
    t = translate(_step('leiden', {'resolution': 1.0, 'key_added': 'leiden'}, {}))
    assert t.summary  # no KeyError, no empty string


def test_filtering_reports_what_it_removed():
    t = translate(_step('filter_cells', {'min_genes': 200},
                        {'n_cells_before': 100, 'n_cells_after': 80, 'n_cells_removed': 20}))
    assert '20' in t.summary


# --- literal rendering ----------------------------------------------------

def test_string_arguments_are_quoted():
    t = translate(_step('leiden', {'resolution': 1.0, 'key_added': "o'brien"}, {}))
    assert '"o\'brien"' in t.code[0] or "'o\\'brien'" in t.code[0]


def test_a_path_with_a_quote_cannot_break_out_of_the_string():
    """Everything here ends up inside a notebook the user will execute."""
    nasty = '/tmp/a"; import os; os.system("x")#.h5ad'
    t = translate(_step('load_dataset', {'path': nasty, 'kind': 'h5ad'}, {}))
    body = '\n'.join(t.code)
    # The path survives only as a single quoted literal — nothing escapes it.
    assert 'os.system' not in body.replace(repr(nasty), '')
    compile(body, '<load>', 'exec')


def test_load_emits_the_reader_matching_the_source_kind():
    def body(kind, path):
        return '\n'.join(translate(_step('load_dataset', {'path': path, 'kind': kind}, {})).code)
    assert 'read_h5ad' in body('h5ad', '/d/a.h5ad')
    assert 'read_10x_mtx' in body('10x_mtx', '/d/mtx')
    assert 'read_10x_h5' in body('10x_h5', '/d/a.h5')


def test_a_load_of_unknown_provenance_is_not_reproducible():
    """An in-memory adata (combine_spatial, tests) has no file to re-read."""
    t = translate(_step('load_dataset', {'path': 'x.h5ad', 'kind': 'memory'}, {}))
    assert t.fidelity == MANUAL


def test_booleans_and_none_render_as_python_literals():
    t = translate(_step('highly_variable_genes', {
        'n_top_genes': None, 'min_mean': 0.0125, 'max_mean': 3.0,
        'min_disp': 0.5, 'flavor': 'seurat', 'subset': True,
    }, {}))
    assert 'subset=True' in t.code[0]
    assert 'n_top_genes' not in t.code[0]  # None means "not requested"


def test_generated_code_is_syntactically_valid_python():
    """Representative entries, compiled with realistic parameters."""
    cases = [
        ('normalize_total', {'target_sum': 1e4}), ('log1p', {}),
        ('leiden', {'resolution': 1.0, 'key_added': 'leiden'}),
        ('umap', {'min_dist': 0.5, 'spread': 1.0, 'n_components': 2}),
        ('pca', {'n_comps': 50, 'svd_solver': 'arpack', 'gene_subset': 'highly_variable'}),
        ('neighbors', {'n_neighbors': 15, 'metric': 'euclidean'}),
        ('contourize', {'genes': ["it's"], 'contour_levels': 6}),
        ('marker_genes', {'obs_column': 'leiden', 'top_n': 25}),
        ('rename_obs_label', {'column': 'leiden', 'old_label': '0', 'new_label': "Muller's"}),
        ('merge_obs_labels', {'column': 'leiden', 'labels': ['0', '1'], 'new_label': 'glia'}),
        ('create_annotation', {'name': 'regions', 'default_value': 'unassigned'}),
        ('calculate_qc_metrics', {'qc_vars': ['mito'], 'percent_top': None, 'log1p': True}),
        ('exclude_genes', {'gene_names': ['A'], 'patterns': ['^MT-', "^it's"]}),
    ]
    for action, params in cases:
        t = translate(_step(action, params, {}))
        compile('\n'.join(t.code), f'<{action}>', 'exec')


def test_every_registry_entry_compiles_with_no_parameters_recorded():
    """A template with a stray quote or brace is a broken notebook cell.

    Empty params is the degenerate case every builder must survive — a record
    restored from someone else's h5ad can be missing anything.
    """
    for action in sorted(registered_actions()):
        t = translate(_step(action, {}, {}))
        if t.code:
            compile('\n'.join(t.code), f'<{action}>', 'exec')


def test_every_registry_entry_produces_a_summary_with_no_parameters():
    for action in sorted(registered_actions()):
        assert translate(_step(action, {}, {})).summary.strip(), action


def test_label_cells_needs_its_selection_to_be_reproducible():
    """The selection *is* the operation — without it there is nothing to emit."""
    params = {'annotation': 'regions', 'label': 'cortex'}
    assert translate(_step('label_cells', params, {})).fidelity == MANUAL
    with_sel = translate(_step('label_cells', params, {}, selection=[1, 2], n_total=10))
    assert with_sel.fidelity == EXACT
    compile('\n'.join(with_sel.code), '<label_cells>', 'exec')


# --- graph choice ---------------------------------------------------------

def test_codegen_and_adaptor_agree_on_the_uns_key():
    """codegen must not import the adaptor (it promises no AnnData, no adaptor),
    so the key is spelled twice. This is what keeps the two spellings equal."""
    from xcell.adaptor import _GRAPH_META_KEY as adaptor_key
    from xcell.codegen import _GRAPH_META_KEY as codegen_key

    assert adaptor_key == codegen_key


def test_umap_on_the_default_graph_emits_one_unchanged_line():
    step = _step('umap', {'min_dist': 0.5, 'spread': 1.0, 'n_components': 2})
    t = translate(step)
    assert t.fidelity == EXACT
    assert t.code == [
        'sc.tl.umap(adata, min_dist=0.5, spread=1.0, n_components=2)'
    ]


def test_umap_on_a_named_graph_emits_the_neighbors_entry_it_needs():
    step = _step('umap', {
        'min_dist': 0.5, 'spread': 1.0, 'n_components': 2,
        'graph_key': 'spatial_connectivities', 'key_added': 'X_umap_spatial',
        'neighbors_meta': {
            'connectivities_key': 'spatial_connectivities',
            'distances_key': 'spatial_distances',
            'params': {'n_neighbors': 6, 'method': 'umap'},
        },
    })
    t = translate(step)
    assert t.fidelity == EXACT
    code = '\n'.join(t.code)
    # Emitting only the call would fail in the notebook: the entry it names
    # does not exist there.
    assert "adata.uns['_xcell_graph']" in code
    assert "'connectivities_key': 'spatial_connectivities'" in code
    assert "'method': 'umap'" in code
    assert "'distances_key'" in code
    assert "neighbors_key='_xcell_graph'" in code
    assert "key_added='X_umap_spatial'" in code
    # graph_key is xcell's own parameter name; scanpy has no such argument.
    assert 'graph_key=' not in code


def test_umap_export_carries_use_rep_so_the_notebook_makes_no_phantom_pca():
    """Without use_rep, scanpy computes and stores a PCA when X_pca is absent.

    The adaptor avoids that; the notebook has to avoid it too, or executing the
    export produces an object the recording never described.
    """
    step = _step('umap', {
        'min_dist': 0.5, 'spread': 1.0, 'n_components': 2,
        'graph_key': 'spatial_connectivities', 'key_added': 'X_umap_spatial',
        'neighbors_meta': {
            'connectivities_key': 'spatial_connectivities',
            'distances_key': 'spatial_distances',
            'params': {'n_neighbors': 6, 'method': 'umap', 'use_rep': 'X'},
        },
    })
    assert "'use_rep': 'X'" in '\n'.join(translate(step).code)


def test_umap_with_only_a_custom_name_stays_a_one_liner():
    step = _step('umap', {'min_dist': 0.5, 'spread': 1.0,
                          'n_components': 2, 'key_added': 'X_umap_v2'})
    t = translate(step)
    assert t.code == [
        "sc.tl.umap(adata, min_dist=0.5, spread=1.0, n_components=2, "
        "key_added='X_umap_v2')"
    ]


def test_leiden_on_a_named_graph_passes_obsp():
    step = _step('leiden', {'resolution': 0.5, 'key_added': 'leiden_spatial',
                            'graph_key': 'spatial_connectivities'})
    t = translate(step)
    assert t.fidelity == EXACT
    assert t.code == [
        "sc.tl.leiden(adata, resolution=0.5, key_added='leiden_spatial', "
        "obsp='spatial_connectivities')"
    ]


def test_leiden_on_the_default_graph_is_unchanged():
    step = _step('leiden', {'resolution': 0.5, 'key_added': 'leiden'})
    assert translate(step).code == [
        "sc.tl.leiden(adata, resolution=0.5, key_added='leiden')"
    ]
