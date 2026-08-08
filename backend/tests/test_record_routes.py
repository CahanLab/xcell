"""Route tests for /api/record — reading, annotating, and exporting the record."""
import base64
import json

import anndata
import numpy as np
import pandas as pd
from fastapi.testclient import TestClient
from scipy.sparse import csr_matrix

from xcell.adaptor import DataAdaptor
from xcell.api import routes
from xcell.main import app

PNG = base64.b64encode(b'\x89PNG\r\n\x1a\n fake').decode()


def _adata(n_cells=40, n_genes=12):
    rng = np.random.default_rng(0)
    ad = anndata.AnnData(
        X=csr_matrix(rng.integers(0, 10, (n_cells, n_genes)).astype(np.float32))
    )
    ad.var_names = [f'g{i}' for i in range(n_genes)]
    ad.obs_names = [f'c{i}' for i in range(n_cells)]
    ad.obs['group'] = pd.Categorical(['a', 'b'] * (n_cells // 2))
    return ad


def _install(slot='primary'):
    a = DataAdaptor('cortex.h5ad', adata=_adata())
    routes.set_adaptor(a, slot=slot)
    return a


def _client():
    return TestClient(app)


# --- reading --------------------------------------------------------------

def test_the_record_is_readable():
    a = _install()
    a.run_normalize_total(target_sum=1e4)
    body = _client().get('/api/record').json()
    assert [s['action'] for s in body['steps']] == ['load_dataset', 'normalize_total']
    assert body['source']['n_cells'] == 40


def test_each_step_arrives_already_translated():
    """The panel shows the summary and fidelity; it shouldn't reimplement them."""
    a = _install()
    a.run_normalize_total(target_sum=1e4)
    step = _client().get('/api/record').json()['steps'][-1]
    assert step['title'] == 'Normalize total counts'
    assert step['fidelity'] == 'exact'
    assert '10,000' in step['summary']


def test_the_response_carries_the_reproducibility_counts():
    a = _install()
    a.run_normalize_total(target_sum=1e4)
    counts = _client().get('/api/record').json()['counts']
    assert counts['total'] == 1  # the load is setup, not a reported step
    assert counts['exact'] == 1


def test_figure_bytes_are_not_in_the_listing():
    """A dozen PNGs would make every poll megabytes."""
    a = _install()
    a.analysis_record.add_figure(PNG, caption='UMAP')
    figures = _client().get('/api/record').json()['figures']
    assert figures[0]['caption'] == 'UMAP'
    assert 'png_b64' not in figures[0]


def test_reading_without_data_is_a_503():
    routes.remove_adaptor('secondary')
    assert _client().get('/api/record?dataset=secondary').status_code == 503


def test_the_secondary_slot_has_its_own_record():
    _install('primary')
    b = _install('secondary')
    b.run_log1p()
    primary = _client().get('/api/record').json()
    secondary = _client().get('/api/record?dataset=secondary').json()
    assert [s['action'] for s in primary['steps']] == ['load_dataset']
    assert [s['action'] for s in secondary['steps']][-1] == 'log1p'
    routes.remove_adaptor('secondary')


# --- annotating -----------------------------------------------------------

def test_the_title_and_abstract_can_be_set():
    _install()
    r = _client().put('/api/record/meta',
                      json={'title': 'Cortex atlas', 'abstract': 'A study.'})
    assert r.status_code == 200
    body = _client().get('/api/record').json()
    assert body['title'] == 'Cortex atlas' and body['abstract'] == 'A study.'


def test_a_step_note_can_be_set_and_cleared():
    a = _install()
    a.run_log1p()
    c = _client()
    assert c.post('/api/record/step/1/note', json={'note': 'why'}).status_code == 200
    assert c.get('/api/record').json()['steps'][1]['note'] == 'why'
    c.post('/api/record/step/1/note', json={'note': ''})
    assert c.get('/api/record').json()['steps'][1]['note'] is None


def test_a_note_on_a_step_that_does_not_exist_is_a_404():
    _install()
    assert _client().post('/api/record/step/99/note',
                          json={'note': 'x'}).status_code == 404


def test_marking_the_start_trims_the_report_without_losing_history():
    a = _install()
    a.run_normalize_total(target_sum=1e4)
    c = _client()
    assert c.post('/api/record/mark').json()['report_start'] == 2
    a.run_log1p()
    body = c.get('/api/record').json()
    assert len(body['steps']) == 3          # everything is still listed
    assert body['counts']['total'] == 1     # but only log1p is in the report


def test_clearing_keeps_the_load_step():
    a = _install()
    a.run_log1p()
    c = _client()
    assert c.delete('/api/record').status_code == 200
    assert [s['action'] for s in c.get('/api/record').json()['steps']] == ['load_dataset']


# --- figures --------------------------------------------------------------

def test_a_captured_figure_attaches_to_the_latest_step():
    a = _install()
    a.run_log1p()
    r = _client().post('/api/record/figure',
                       json={'png_b64': PNG, 'caption': 'UMAP by cluster'})
    assert r.status_code == 200
    fig_id = r.json()['id']
    assert a.analysis_record.figures[fig_id].step_index == 1


def test_a_data_url_prefix_is_accepted():
    """canvas.toDataURL returns 'data:image/png;base64,...'."""
    a = _install()
    r = _client().post('/api/record/figure',
                       json={'png_b64': f'data:image/png;base64,{PNG}'})
    assert r.status_code == 200
    assert a.analysis_record.figures[r.json()['id']].png_b64 == PNG


def test_something_that_is_not_a_png_is_rejected():
    _install()
    r = _client().post('/api/record/figure', json={'png_b64': 'zzz not base64 zzz'})
    assert r.status_code == 400


def test_a_figure_can_be_removed():
    a = _install()
    fig_id = _client().post('/api/record/figure', json={'png_b64': PNG}).json()['id']
    assert _client().delete(f'/api/record/figure/{fig_id}').status_code == 200
    assert a.analysis_record.figures == {}


def test_removing_a_figure_that_is_gone_is_a_404():
    _install()
    assert _client().delete('/api/record/figure/nope').status_code == 404


# --- export ---------------------------------------------------------------

def _pipeline(a):
    a.run_normalize_total(target_sum=1e4)
    a.run_log1p()


def test_exporting_a_notebook_writes_a_valid_ipynb(tmp_path):
    a = _install()
    _pipeline(a)
    r = _client().post('/api/record/export', json={
        'format': 'ipynb', 'output_dir': str(tmp_path), 'filename': 'analysis',
    })
    assert r.status_code == 200, r.text
    written = [p['path'] for p in r.json()['files']]
    assert str(tmp_path / 'analysis.ipynb') in written
    nb = json.loads((tmp_path / 'analysis.ipynb').read_text())
    assert nb['nbformat'] == 4
    body = '\n'.join(''.join(c['source']) for c in nb['cells']
                     if c['cell_type'] == 'code')
    assert 'sc.pp.normalize_total(adata, target_sum=10000.0)' in body


def test_exporting_markdown_writes_a_md_file(tmp_path):
    a = _install()
    _pipeline(a)
    r = _client().post('/api/record/export', json={
        'format': 'md', 'output_dir': str(tmp_path), 'filename': 'analysis',
    })
    assert r.status_code == 200, r.text
    assert (tmp_path / 'analysis.md').exists()
    assert '```python' in (tmp_path / 'analysis.md').read_text()


def test_both_formats_can_be_written_at_once(tmp_path):
    a = _install()
    _pipeline(a)
    _client().post('/api/record/export', json={
        'format': 'both', 'output_dir': str(tmp_path), 'filename': 'analysis',
    })
    assert (tmp_path / 'analysis.ipynb').exists()
    assert (tmp_path / 'analysis.md').exists()


def test_the_response_previews_the_document(tmp_path):
    """So the panel can show what was written without reading it back."""
    a = _install()
    _pipeline(a)
    body = _client().post('/api/record/export', json={
        'format': 'ipynb', 'output_dir': str(tmp_path), 'filename': 'analysis',
    }).json()
    assert body['markdown'].startswith('# ')
    assert body['counts']['exact'] == 2


def test_figures_are_written_alongside_a_markdown_export(tmp_path):
    a = _install()
    _pipeline(a)
    _client().post('/api/record/figure', json={'png_b64': PNG, 'caption': 'UMAP'})
    r = _client().post('/api/record/export', json={
        'format': 'md', 'output_dir': str(tmp_path), 'filename': 'analysis',
    })
    figures = tmp_path / 'analysis_figures'
    assert figures.is_dir() and list(figures.glob('*.png'))
    assert 'analysis_figures/' in (tmp_path / 'analysis.md').read_text()
    assert any(p['path'].endswith('.png') for p in r.json()['files'])


def test_a_notebook_export_needs_no_figure_directory(tmp_path):
    """Base64 lives inside the .ipynb, so it stays a single file."""
    a = _install()
    _pipeline(a)
    _client().post('/api/record/figure', json={'png_b64': PNG})
    _client().post('/api/record/export', json={
        'format': 'ipynb', 'output_dir': str(tmp_path), 'filename': 'analysis',
    })
    assert not (tmp_path / 'analysis_figures').exists()


def test_recorded_selections_are_written_beside_the_notebook(tmp_path):
    a = _install()
    a.run_normalize_total(target_sum=1e4, active_cell_indices=[0, 1, 2])
    _client().post('/api/record/export', json={
        'format': 'ipynb', 'output_dir': str(tmp_path), 'filename': 'analysis',
    })
    sidecar = tmp_path / 'analysis_selections.json'
    assert sidecar.exists()
    assert json.loads(sidecar.read_text()) == {'step_1': [0, 1, 2]}


def test_no_sidecar_when_nothing_ran_on_a_selection(tmp_path):
    a = _install()
    _pipeline(a)
    _client().post('/api/record/export', json={
        'format': 'ipynb', 'output_dir': str(tmp_path), 'filename': 'analysis',
    })
    assert not (tmp_path / 'analysis_selections.json').exists()


# --- export validation ----------------------------------------------------

def test_a_missing_output_directory_is_a_400(tmp_path):
    _install()
    r = _client().post('/api/record/export', json={
        'format': 'ipynb', 'output_dir': str(tmp_path / 'nope'), 'filename': 'a',
    })
    assert r.status_code == 400


def test_a_file_where_a_directory_should_be_is_a_400(tmp_path):
    _install()
    target = tmp_path / 'afile'
    target.write_text('x')
    r = _client().post('/api/record/export', json={
        'format': 'ipynb', 'output_dir': str(target), 'filename': 'a',
    })
    assert r.status_code == 400


def test_a_filename_cannot_escape_the_output_directory(tmp_path):
    """The filename comes from a text box; it must not be a path."""
    _install()
    for bad in ('../outside', 'sub/dir', '/etc/passwd', '..'):
        r = _client().post('/api/record/export', json={
            'format': 'ipynb', 'output_dir': str(tmp_path), 'filename': bad,
        })
        assert r.status_code == 400, bad
    assert not list(tmp_path.parent.glob('outside*'))


def test_an_unknown_format_is_a_400(tmp_path):
    _install()
    r = _client().post('/api/record/export', json={
        'format': 'pdf', 'output_dir': str(tmp_path), 'filename': 'a',
    })
    assert r.status_code == 400


def test_an_extension_in_the_filename_is_not_doubled(tmp_path):
    _install()
    _client().post('/api/record/export', json={
        'format': 'ipynb', 'output_dir': str(tmp_path), 'filename': 'analysis.ipynb',
    })
    assert (tmp_path / 'analysis.ipynb').exists()
    assert not (tmp_path / 'analysis.ipynb.ipynb').exists()


def test_code_and_figures_can_be_switched_off(tmp_path):
    a = _install()
    _pipeline(a)
    _client().post('/api/record/figure', json={'png_b64': PNG})
    _client().post('/api/record/export', json={
        'format': 'ipynb', 'output_dir': str(tmp_path), 'filename': 'narrative',
        'include_code': False, 'include_figures': False,
    })
    nb = json.loads((tmp_path / 'narrative.ipynb').read_text())
    assert [c for c in nb['cells'] if c['cell_type'] == 'code'] == []
