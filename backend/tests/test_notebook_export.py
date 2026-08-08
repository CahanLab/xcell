"""Rendering a record as a notebook or markdown.

The notebook is the deliverable, so these tests are about the things that make
it usable by someone who wasn't there: it opens the right file, it says how much
of itself actually re-runs, it doesn't quietly drop the steps it can't
reproduce, and — above all — it is valid, executable Python.
"""
import base64
import json

import pytest

from xcell.analysis_record import AnalysisRecord
from xcell.notebook_export import (
    figure_filename,
    figure_payloads,
    selections_payload,
    to_markdown,
    to_notebook,
)

PNG = base64.b64encode(b'\x89PNG\r\n\x1a\n fake').decode()


def _loaded(path='/data/cortex.h5ad'):
    r = AnalysisRecord(source={'path': path, 'n_cells': 100, 'n_genes': 20})
    r.add_step('load_dataset', {'path': path, 'kind': 'h5ad'},
               {'n_cells': 100, 'n_genes': 20})
    return r


def _pipeline():
    r = _loaded()
    r.add_step('normalize_total', {'target_sum': 10000.0}, {'status': 'completed'})
    r.add_step('log1p', {}, {'status': 'completed'})
    r.add_step('leiden', {'resolution': 1.0, 'key_added': 'leiden'}, {'n_clusters': 12})
    return r


def _code_cells(nb):
    return [c for c in nb['cells'] if c['cell_type'] == 'code']


def _md(nb):
    return '\n'.join(''.join(c['source']) for c in nb['cells'] if c['cell_type'] == 'markdown')


def _src(cell):
    return ''.join(cell['source'])


# --- notebook structure ---------------------------------------------------

def test_the_notebook_is_structurally_valid_nbformat_4():
    nb = to_notebook(_pipeline())
    assert nb['nbformat'] == 4
    assert isinstance(nb['nbformat_minor'], int)
    assert isinstance(nb['metadata'], dict)
    for cell in nb['cells']:
        assert cell['cell_type'] in ('code', 'markdown')
        assert isinstance(cell['source'], list)
        assert isinstance(cell['metadata'], dict)
        if cell['cell_type'] == 'code':
            assert cell['execution_count'] is None
            assert isinstance(cell['outputs'], list)


def test_the_notebook_declares_a_python_kernel():
    """Without kernelspec, Jupyter prompts the user to pick one."""
    meta = to_notebook(_pipeline())['metadata']
    assert meta['kernelspec']['language'] == 'python'
    assert meta['language_info']['name'] == 'python'


def test_the_notebook_is_json_serializable():
    json.dumps(to_notebook(_pipeline()), allow_nan=False)


def test_nbformat_accepts_it():
    """The real validator, when it happens to be installed."""
    nbformat = pytest.importorskip('nbformat')
    nbformat.validate(nbformat.from_dict(to_notebook(_pipeline())))


# --- what the reader sees first -------------------------------------------

def test_the_title_and_abstract_open_the_document():
    r = _pipeline()
    r.title = 'Cortex atlas'
    r.abstract = 'Clustering of 100 cortical cells.'
    first = _src(to_notebook(r)['cells'][0])
    assert first.startswith('# Cortex atlas')
    assert 'Clustering of 100 cortical cells.' in first


def test_an_untitled_record_still_gets_a_heading():
    assert _src(to_notebook(_pipeline())['cells'][0]).startswith('# ')


def test_the_header_states_how_much_of_the_notebook_re_runs():
    r = _pipeline()                              # normalize, log1p, leiden
    r.add_step('some_future_tool', {}, {})       # manual
    r.add_step('smooth', {'graph_key': 'connectivities', 'n_steps': 1,
                          'source_layer': 'X', 'output_layer': 'smoothed',
                          'self_loop_weight': 1.0}, {})  # xcell
    header = _src(to_notebook(r)['cells'][0])
    assert '**5 steps.**' in header
    assert '3 re-run as written' in header
    assert '1 need the xcell Python API' in header
    assert '1 are manual' in header


def test_the_header_count_matches_the_numbered_steps():
    """A reader who counts the headings must get the number the header claims.

    The load is the setup cell, not a step — counting it would put the header
    one ahead of what is visible.
    """
    r = _pipeline()
    nb = to_notebook(r)
    headings = [c for c in nb['cells']
                if c['cell_type'] == 'markdown' and _src(c).startswith('## ')]
    assert '**3 steps.**' in _src(nb['cells'][0])
    assert len(headings) == 3
    assert _src(headings[0]).startswith('## 1. ')


def test_the_header_counts_steps_that_ran_on_a_selection():
    r = _pipeline()
    r.add_step('leiden', {'resolution': 2.0, 'key_added': 'sub'}, {},
               selection=[1, 2, 3], n_total=100)
    assert 'selection' in _src(to_notebook(r)['cells'][0]).lower()


# --- setup ----------------------------------------------------------------

def test_the_setup_cell_opens_the_source_file():
    nb = to_notebook(_pipeline())
    setup = _src(_code_cells(nb)[0])
    assert 'import scanpy as sc' in setup
    assert "SOURCE_PATH = '/data/cortex.h5ad'" in setup


def test_the_reader_goes_through_the_path_constant():
    """One line to edit when the notebook moves, not two that can disagree."""
    setup = _src(_code_cells(to_notebook(_pipeline()))[0])
    assert 'adata = sc.read_h5ad(SOURCE_PATH)' in setup
    assert setup.count('/data/cortex.h5ad') == 1


def test_the_adaptor_is_only_set_up_when_a_step_needs_it():
    plain = _src(_code_cells(to_notebook(_pipeline()))[0])
    assert 'DataAdaptor' not in plain

    r = _pipeline()
    r.add_step('cluster_genes', {'resolution': 0.5, 'key_added': 'gene_cluster'}, {})
    withx = '\n'.join(_src(c) for c in _code_cells(to_notebook(r)))
    assert 'DataAdaptor' in withx
    assert '_xcell_run' in withx  # the two-phase helper comes with it


def test_the_selections_sidecar_is_loaded_only_when_there_are_selections():
    plain = _src(_code_cells(to_notebook(_pipeline()))[0])
    assert 'SELECTIONS' not in plain

    r = _pipeline()
    r.add_step('delete_cells', {'n_indices': 2}, {'n_cells_deleted': 2},
               selection=[3, 4], n_total=100)
    setup = _src(_code_cells(to_notebook(r))[0])
    assert 'SELECTIONS' in setup
    assert 'selections.json' in setup


def test_the_selections_payload_is_keyed_by_step():
    r = _pipeline()
    r.add_step('delete_cells', {'n_indices': 2}, {}, selection=[3, 4], n_total=100)
    assert selections_payload(r) == {'step_4': [3, 4]}


def test_a_record_with_no_selections_has_an_empty_payload():
    assert selections_payload(_pipeline()) == {}


# --- steps ----------------------------------------------------------------

def test_each_step_gets_prose_then_code_in_that_order():
    nb = to_notebook(_pipeline())
    kinds = [c['cell_type'] for c in nb['cells']]
    # header(md), setup(code), then (md, code) per step after the load
    assert kinds[:2] == ['markdown', 'code']
    assert kinds[2] == 'markdown' and kinds[3] == 'code'


def test_a_step_renders_its_generated_code():
    body = '\n'.join(_src(c) for c in _code_cells(to_notebook(_pipeline())))
    assert 'sc.pp.normalize_total(adata, target_sum=10000.0)' in body
    assert "sc.tl.leiden(adata, resolution=1.0, key_added='leiden')" in body


def test_a_user_note_appears_with_its_step():
    r = _pipeline()
    r.set_note(3, 'resolution picked from a silhouette sweep')
    assert 'resolution picked from a silhouette sweep' in _md(to_notebook(r))


def test_a_subset_warning_appears_with_its_step():
    r = _pipeline()
    r.add_step('leiden', {'resolution': 2.0, 'key_added': 'sub'}, {},
               selection=[1, 2, 3], n_total=100)
    assert '3' in _md(to_notebook(r)) and 'selection' in _md(to_notebook(r)).lower()


def test_only_the_marked_report_span_is_exported():
    r = _pipeline()
    r.mark_start()
    r.add_step('umap', {'min_dist': 0.5, 'spread': 1.0, 'n_components': 2}, {})
    body = '\n'.join(_src(c) for c in _code_cells(to_notebook(r)))
    assert 'sc.tl.umap' in body
    assert 'sc.tl.leiden' not in body


def test_the_setup_cell_survives_a_marker_past_the_load():
    """The load step is outside the report span, but the notebook still has to
    open the file."""
    r = _pipeline()
    r.mark_start()
    assert 'sc.read_h5ad' in _src(_code_cells(to_notebook(r))[0])


def test_manual_steps_are_listed_where_they_cannot_be_missed():
    r = _pipeline()
    r.add_step('some_future_tool', {'alpha': 3}, {})
    text = _md(to_notebook(r))
    assert 'some_future_tool' in text
    assert 'alpha' in text


def test_include_code_false_produces_a_narrative_only_document():
    nb = to_notebook(_pipeline(), include_code=False)
    assert _code_cells(nb) == []
    assert 'Leiden clustering' in _md(nb)


# --- figures --------------------------------------------------------------

def test_a_figure_is_embedded_as_a_display_data_output():
    r = _pipeline()
    r.add_figure(PNG, caption='UMAP by cluster')
    outputs = [o for c in _code_cells(to_notebook(r)) for o in c['outputs']]
    assert any(o['output_type'] == 'display_data' and 'image/png' in o['data']
               for o in outputs)
    assert 'UMAP by cluster' in _md(to_notebook(r))


def test_a_figure_on_a_step_with_no_code_still_renders():
    """A manual step has no code cell to hang an output on."""
    r = _pipeline()
    r.add_step('some_future_tool', {}, {})
    r.add_figure(PNG, caption='what it looked like')
    nb = to_notebook(r)
    outputs = [o for c in _code_cells(nb) for o in c['outputs']]
    assert any('image/png' in o.get('data', {}) for o in outputs)


def test_figures_can_be_left_out():
    r = _pipeline()
    r.add_figure(PNG, caption='UMAP')
    outputs = [o for c in _code_cells(to_notebook(r, include_figures=False))
               for o in c['outputs']]
    assert outputs == []


def test_a_standalone_figure_lands_at_the_end():
    r = AnalysisRecord()
    r.add_figure(PNG, caption='captured before anything ran')
    r.add_step('log1p', {}, {})
    assert 'captured before anything ran' in _md(to_notebook(r))


def test_figure_payloads_are_decoded_bytes_keyed_by_filename():
    r = _pipeline()
    fig = r.add_figure(PNG, caption='UMAP')
    payloads = figure_payloads(r)
    assert payloads[figure_filename(fig)] == base64.b64decode(PNG)


def test_a_corrupt_figure_is_skipped_rather_than_crashing_the_export():
    r = _pipeline()
    r.add_figure('not base64 !!!', caption='broken')
    assert figure_payloads(r) == {}
    to_notebook(r)  # and the notebook still renders


# --- markdown -------------------------------------------------------------

def test_markdown_leads_with_the_title():
    r = _pipeline()
    r.title = 'Cortex atlas'
    assert to_markdown(r).startswith('# Cortex atlas')


def test_markdown_fences_the_generated_code():
    text = to_markdown(_pipeline())
    assert '```python' in text
    assert 'sc.pp.normalize_total(adata, target_sum=10000.0)' in text


def test_markdown_links_figures_into_a_sibling_directory():
    r = _pipeline()
    fig = r.add_figure(PNG, caption='UMAP')
    text = to_markdown(r, figure_dir='analysis_figures')
    assert f'![UMAP](analysis_figures/{figure_filename(fig)})' in text


def test_markdown_omits_figures_when_there_is_nowhere_to_put_them():
    r = _pipeline()
    r.add_figure(PNG, caption='UMAP')
    assert '![' not in to_markdown(r, include_figures=False)


def test_markdown_carries_the_same_reproducibility_header():
    assert 'step' in to_markdown(_pipeline()).split('\n\n')[1].lower()


# --- executability --------------------------------------------------------

def test_every_generated_code_cell_compiles():
    r = _pipeline()
    r.add_step('pca', {'n_comps': 50, 'svd_solver': 'arpack', 'gene_subset': None}, {})
    r.add_step('neighbors', {'n_neighbors': 15, 'metric': 'euclidean'}, {})
    r.add_step('umap', {'min_dist': 0.5, 'spread': 1.0, 'n_components': 2}, {})
    r.add_step('delete_cells', {'n_indices': 1}, {}, selection=[2], n_total=100)
    r.add_step('cluster_genes', {'resolution': 0.5, 'key_added': 'gene_cluster'}, {})
    for cell in _code_cells(to_notebook(r)):
        compile(_src(cell), '<cell>', 'exec')


def test_the_whole_notebook_compiles_as_one_script():
    """Cells share a namespace, so concatenation has to parse too."""
    body = '\n'.join(_src(c) for c in _code_cells(to_notebook(_pipeline())))
    compile(body, '<notebook>', 'exec')


# --- edges ----------------------------------------------------------------

def test_an_empty_record_still_produces_a_valid_notebook():
    nb = to_notebook(AnalysisRecord())
    assert nb['nbformat'] == 4 and nb['cells']
    json.dumps(nb)


def test_a_record_whose_source_was_never_a_file_says_so():
    r = AnalysisRecord()
    r.add_step('load_dataset', {'path': 'in-memory', 'kind': 'memory'}, {})
    r.add_step('log1p', {}, {})
    text = _md(to_notebook(r))
    assert 'log1p' in text.lower() or 'Log-transform' in text
