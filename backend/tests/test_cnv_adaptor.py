"""The adaptor side of CNV inference: validation, snapshot, write-back, record,
the cnv_heatmap figure kind, and the routes.

Same synthetic tumor as test_cnv.py, with positions carried in .var so no
shipped table is needed. ``normal`` cells are the reference; ``A``, ``B``
and ``C`` carry distinct chromosome events.
"""
import json
import time

import anndata
import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient
from scipy.sparse import csr_matrix

from xcell import codegen
from xcell import cnv
from xcell.adaptor import DataAdaptor, FigureInputMissing
from xcell.api import routes
from xcell.main import app

N_CHROM, GENES_PER_CHROM = 6, 120
GROUPS = {'normal': 100, 'A': 60, 'B': 60, 'C': 60}
EVENTS = {'A': {'chr2': 2.0}, 'B': {'chr4': 0.5}, 'C': {'chr2': 2.0, 'chr6': 0.5}}
N_CELLS = sum(GROUPS.values())


def _adata(seed=0, log=False):
    rng = np.random.default_rng(seed)
    n_genes = N_CHROM * GENES_PER_CHROM
    names = [f'g{c}_{g:03d}' for c in range(1, N_CHROM + 1) for g in range(GENES_PER_CHROM)]
    chroms = np.array([f'chr{c}' for c in range(1, N_CHROM + 1) for _ in range(GENES_PER_CHROM)])
    starts = np.array([1000 * (g + 1) for _ in range(N_CHROM) for g in range(GENES_PER_CHROM)])
    lam = rng.lognormal(0.3, 0.8, size=n_genes)
    labels = np.concatenate([[g] * n for g, n in GROUPS.items()])
    X = np.empty((labels.size, n_genes), dtype=np.float32)
    for i, lab in enumerate(labels):
        scale = np.ones(n_genes)
        for chrom, f in EVENTS.get(lab, {}).items():
            scale[chroms == chrom] = f
        X[i] = rng.poisson(lam * scale * rng.uniform(0.6, 1.8))
    if log:
        X = np.log1p(X / X.sum(axis=1, keepdims=True) * 1e4).astype(np.float32)
    ad = anndata.AnnData(X=csr_matrix(X))
    ad.var_names = names
    ad.var['chromosome'] = chroms
    ad.var['start'] = starts
    ad.obs['group'] = pd.Categorical(labels)
    ad.obs['batch'] = pd.Categorical(['x', 'y'] * (labels.size // 2))
    return ad


def _adaptor(**kw):
    return DataAdaptor('x.h5ad', adata=_adata(**kw))


DEFAULTS = dict(chromosome_column='chromosome', start_column='start', reference_column='group',
                reference_categories=['normal'], window=30, step=5, n_pcs=10, resolution=0.5)


def _run(a, **kw):
    params = {**DEFAULTS, **kw}
    compute_fn, apply_fn = a.prepare_cnv(**params)
    return apply_fn(compute_fn(lambda f, m=None: None))


# ---------- status ----------

def test_status_lists_position_candidates_and_reference_columns():
    s = _adaptor().cnv_status()
    assert s['n_cells'] == N_CELLS
    assert s['species'] is None                      # synthetic names vote for neither
    assert {c['name'] for c in s['var_columns']} >= {'chromosome', 'start'}
    ref = {c['name']: c['categories'] for c in s['obs_categorical']}
    assert ref['group'] == ['A', 'B', 'C', 'normal']
    assert s['x_scale'] == 'raw_counts'
    assert s['keys'] == []
    assert isinstance(s['table_species'], list)


# ---------- validation ----------

def test_z_scored_input_is_refused():
    a = _adaptor(log=True)
    X = a.adata.X.toarray()
    a.adata.X = csr_matrix(((X - X.mean(axis=0)) / (X.std(axis=0) + 1e-9)).astype(np.float32))
    with pytest.raises(ValueError, match='scaled'):
        a.prepare_cnv(**DEFAULTS)


def test_bad_columns_and_categories():
    a = _adaptor()
    with pytest.raises(ValueError, match='chromosome'):
        a.prepare_cnv(**{**DEFAULTS, 'chromosome_column': 'nope'})
    with pytest.raises(ValueError, match='start'):
        a.prepare_cnv(**{**DEFAULTS, 'start_column': None})
    with pytest.raises(ValueError, match='reference_column'):
        a.prepare_cnv(**{**DEFAULTS, 'reference_column': 'nope'})
    with pytest.raises(ValueError, match="'ghost'"):
        a.prepare_cnv(**{**DEFAULTS, 'reference_categories': ['normal', 'ghost']})
    with pytest.raises(ValueError, match='cluster'):
        a.prepare_cnv(**{**DEFAULTS, 'cluster': 'kmeans'})


def test_reference_with_no_cells_inside_the_mask_is_refused():
    a = _adaptor()
    tumor_only = np.where(a.adata.obs['group'] != 'normal')[0].tolist()
    with pytest.raises(ValueError, match='no cells'):
        a.prepare_cnv(**DEFAULTS, active_cell_indices=tumor_only)


def test_no_species_and_no_columns_is_refused():
    a = _adaptor()
    with pytest.raises(ValueError, match='species'):
        a.prepare_cnv(reference_column='group', reference_categories=['normal'])


def test_too_few_placed_genes_is_refused():
    a = _adaptor()
    a.adata.var['chromosome'] = np.where(np.arange(a.n_genes) < 20, 'chr1', None)
    with pytest.raises(ValueError, match='positions'):
        a.prepare_cnv(**DEFAULTS)


def test_key_may_not_take_over_an_existing_obsm():
    a = _adaptor()
    a.adata.obsm['X_umap'] = np.zeros((N_CELLS, 2), dtype=np.float32)
    with pytest.raises(ValueError, match='X_umap'):
        a.prepare_cnv(**{**DEFAULTS, 'key': 'umap'})
    _run(a, key='tumor')
    _run(a, key='tumor')     # a CNV run's own key can be replaced


def test_shipped_table_path_needs_the_species_table():
    a = _adaptor()
    with pytest.raises(ValueError, match='zebrafish'):
        a.prepare_cnv(species='zebrafish', reference_column='group', reference_categories=['normal'])


# ---------- run ----------

def test_run_writes_matrix_scores_clones_and_record():
    a = _adaptor()
    r = _run(a)
    ad = a.adata
    assert r['key'] == 'cnv'
    assert 'X_cnv' in ad.obsm and ad.obsm['X_cnv'].shape == (N_CELLS, r['n_windows'])
    assert ad.obsm['X_cnv'].dtype == np.float32 and np.isfinite(ad.obsm['X_cnv']).all()
    assert ad.obsm['X_cnv_chrom'].shape == (N_CELLS, N_CHROM)
    assert ad.uns['xcell_score_matrices']['X_cnv_chrom']['columns'] == [f'chr{i}' for i in range(1, 7)]
    assert r['score_column'] == 'cnv_score' and 'cnv_score' in ad.obs
    assert r['clone_column'] == 'cnv_clone'
    clones = ad.obs['cnv_clone']
    assert isinstance(clones.dtype, pd.CategoricalDtype)
    assert list(clones.cat.categories) == [f'clone_{i + 1}' for i in range(len(clones.cat.categories))]
    assert len(ad.uns['cnv_clone_colors']) == len(clones.cat.categories)
    # The quietest clone is the normal cells; clones recover the planted groups.
    from sklearn.metrics import adjusted_rand_score
    assert adjusted_rand_score(ad.obs['group'], clones) > 0.9
    assert (ad.obs['group'][clones == 'clone_1'] == 'normal').mean() > 0.9
    rec = json.loads(ad.uns['xcell_cnv']['cnv'])
    assert rec['key'] == 'cnv' and len(rec['windows']) == r['n_windows']
    assert [c['name'] for c in rec['chromosomes']] == [f'chr{i}' for i in range(1, 7)]
    assert rec['params']['window'] == 30 and rec['reference'] == {'column': 'group', 'categories': ['normal']}
    assert rec['clones'][0]['reference_fraction'] > 0.9
    assert r['clones'] == rec['clones']
    assert r['n_genes_used'] == N_CHROM * GENES_PER_CHROM - r['n_genes_filtered']
    assert r['positions_source'] == 'var_columns'
    assert a.analysis_record.steps[-1].action == 'cnv'
    json.dumps(r)                                     # JSON-clean


def test_log_normalized_input_gives_the_same_clones_as_counts():
    from sklearn.metrics import adjusted_rand_score
    a, b = _adaptor(), _adaptor(log=True)
    _run(a); _run(b)
    assert adjusted_rand_score(a.adata.obs['cnv_clone'], b.adata.obs['cnv_clone']) > 0.95


def test_mask_leaves_nan_and_unassigned_outside():
    a = _adaptor()
    keep = np.where(a.adata.obs['group'].isin(['normal', 'A']))[0]
    r = _run(a, active_cell_indices=keep.tolist())
    M = a.adata.obsm['X_cnv']
    assert np.isfinite(M[keep]).all()
    outside = np.setdiff1d(np.arange(N_CELLS), keep)
    assert np.isnan(M[outside]).all()
    assert np.isnan(a.adata.obs['cnv_score'].to_numpy()[outside]).all()
    assert (a.adata.obs['cnv_clone'].to_numpy()[outside] == 'unassigned').all()
    assert 'unassigned' in a.adata.obs['cnv_clone'].cat.categories
    assert r['n_cells'] == keep.size
    assert a.analysis_record.steps[-1].n_active == keep.size


def test_no_reference_uses_all_cells_and_still_separates():
    from sklearn.metrics import adjusted_rand_score
    a = _adaptor()
    r = _run(a, reference_column=None, reference_categories=None)
    assert r['reference'] is None
    assert any('reference' in w.lower() for w in r['warnings'])
    # Weaker than with a reference: the chr2 gain shared by A and C is partly absorbed by the all-cell mean.
    assert adjusted_rand_score(a.adata.obs['group'], a.adata.obs['cnv_clone']) > 0.6


def test_two_reference_categories_apply_the_bounds_rule():
    a = _adaptor()
    r = _run(a, reference_categories=['normal', 'B'])     # B's chr4 loss is now "normal range"
    rec = json.loads(a.adata.uns['xcell_cnv']['cnv'])
    assert rec['reference']['categories'] == ['normal', 'B']
    chrom = a.adata.obsm['X_cnv_chrom']
    j = [c['name'] for c in rec['chromosomes']].index('chr4')
    b = (a.adata.obs['group'] == 'B').to_numpy()
    assert abs(chrom[b, j].mean()) < 0.1


def test_cluster_none_and_hierarchical_and_custom_key():
    a = _adaptor()
    r = _run(a, cluster='none', key='tumor')
    assert r['clone_column'] is None and 'tumor_clone' not in a.adata.obs
    assert 'X_tumor' in a.adata.obsm and 'tumor_score' in a.adata.obs
    r2 = _run(a, cluster='hierarchical', n_clusters=4, key='tumor')
    assert r2['clone_column'] == 'tumor_clone'
    assert len(a.adata.obs['tumor_clone'].cat.categories) == 4
    assert [k['key'] for k in a.cnv_results()] == ['tumor']
    assert a.cnv_status()['keys'] == ['tumor']


def test_include_xy_and_layer():
    a = _adaptor()
    a.adata.var.loc[a.adata.var_names[:GENES_PER_CHROM], 'chromosome'] = 'chrX'
    a.adata.layers['copy'] = a.adata.X.copy()
    r = _run(a, species='human', layer='copy')
    assert 'chrX' not in [c['name'] for c in r['chromosomes']]
    r = _run(a, species='human', include_xy=True)
    assert [c['name'] for c in r['chromosomes']][-1] == 'chrX'


def test_the_record_step_and_its_notebook_code():
    a = _adaptor()
    _run(a, active_cell_indices=list(range(200)))
    step = a.analysis_record.steps[-1]
    assert step.action == 'cnv' and step.n_active == 200
    assert 'active_cell_indices' not in step.params
    t = codegen.translate(step)
    code = '\n'.join(t.code)
    assert 'xa.prepare_cnv(' in code and "active_cell_indices=SELECTIONS['step_" in code
    compile(code, '<cnv>', 'exec')


# ---------- figure kind ----------

def test_cnv_heatmap_figure_create_and_data():
    a = _adaptor()
    _run(a)
    rec = a.create_figure('cnv_heatmap', inputs={'key': 'cnv', 'obs_column': 'cnv_clone'})
    assert rec['kind'] == 'cnv_heatmap'
    assert rec['params'] == {'row_mode': 'cells', 'max_rows': 400, 'group_order': 'cnv_score', 'vmax': None,
                             'colormap': 'rdbu'}
    data = a.figure_data(rec['id'])
    assert data['kind'] == 'cnv_heatmap' and data['figure_id'] == rec['id']
    assert len(data['matrix']) <= 400 and data['n_cells'] == N_CELLS
    assert [c['name'] for c in data['column_groups']] == [f'chr{i}' for i in range(1, 7)]
    assert data['row_groups'][0] == 'clone_1' and data['vmax'] > 0
    grouped = a.figure_data(rec['id'], params_override={'row_mode': 'groups', 'vmax': 0.5})
    assert len(grouped['matrix']) == len(a.adata.obs['cnv_clone'].cat.categories) and grouped['vmax'] == 0.5
    # Ungrouped, and scoped to a selection.
    rec2 = a.create_figure('cnv_heatmap', inputs={'key': 'cnv', 'obs_column': None, 'cell_indices': list(range(50))})
    d2 = a.figure_data(rec2['id'])
    assert d2['n_cells'] == 50 and all(g is None for g in d2['row_groups'])
    json.dumps(data)


def test_cnv_heatmap_figure_validation_and_missing_inputs():
    a = _adaptor()
    with pytest.raises(ValueError, match='key'):
        a.create_figure('cnv_heatmap', inputs={'key': 'cnv'})
    _run(a)
    with pytest.raises(ValueError, match='obs_column'):
        a.create_figure('cnv_heatmap', inputs={'key': 'cnv', 'obs_column': 'nope'})
    with pytest.raises(ValueError, match='row_mode'):
        a.create_figure('cnv_heatmap', inputs={'key': 'cnv'}, params={'row_mode': 'sideways'})
    rec = a.create_figure('cnv_heatmap', inputs={'key': 'cnv', 'obs_column': 'group'})
    del a.adata.obs['group']
    with pytest.raises(FigureInputMissing):
        a.figure_data(rec['id'])
    rec2 = a.create_figure('cnv_heatmap', inputs={'key': 'cnv'})
    del a.adata.obsm['X_cnv']
    with pytest.raises(FigureInputMissing):
        a.figure_data(rec2['id'])


# ---------- routes ----------

def _install():
    a = _adaptor()
    routes.set_adaptor(a, slot='primary')
    return a


def test_status_and_results_routes():
    _install()
    c = TestClient(app)
    res = c.get('/api/cnv/status')
    assert res.status_code == 200 and res.json()['n_cells'] == N_CELLS
    assert c.get('/api/cnv/results').json() == {'results': []}


def test_run_route_is_a_task_and_bad_input_is_400():
    _install()
    c = TestClient(app)
    bad = c.post('/api/cnv', json={**DEFAULTS, 'reference_column': 'nope'})
    assert bad.status_code == 400 and 'reference_column' in bad.json()['detail']
    res = c.post('/api/cnv', json=DEFAULTS)
    assert res.status_code == 202
    task_id = res.json()['task_id']
    for _ in range(300):
        st = c.get(f'/api/tasks/{task_id}').json()
        if st['status'] in ('completed', 'failed'):
            break
        time.sleep(0.05)
    assert st['status'] == 'completed', st
    assert st['result']['clone_column'] == 'cnv_clone'
    out = c.get('/api/cnv/results').json()['results']
    assert [r['key'] for r in out] == ['cnv'] and 'windows' not in out[0]   # summary, not the whole record
    assert out[0]['n_windows'] == st['result']['n_windows']


# ---------- persistence ----------

def test_results_survive_an_h5ad_round_trip(tmp_path):
    a = _adaptor()
    _run(a)
    a.create_figure('cnv_heatmap', inputs={'key': 'cnv', 'obs_column': 'cnv_clone'})
    path = tmp_path / 'out.h5ad'
    a.adata.write_h5ad(path)
    b = DataAdaptor('x.h5ad', adata=anndata.read_h5ad(path))
    assert [r['key'] for r in b.cnv_results()] == ['cnv']
    assert b.adata.obsm['X_cnv'].shape == a.adata.obsm['X_cnv'].shape
    assert list(b.adata.obs['cnv_clone'].cat.categories) == list(a.adata.obs['cnv_clone'].cat.categories)
    fig = b.list_figures()[0]
    data = b.figure_data(fig['id'])
    assert data['kind'] == 'cnv_heatmap' and len(data['column_groups']) == N_CHROM
