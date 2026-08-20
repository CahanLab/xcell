"""Choosing which .obs / .var columns survive a combine, and how.

The dangerous case is a name collision that is not a value collision: two
datasets each carrying `seurat_clusters` from *independent* clusterings, where
cluster 3 in one has nothing to do with cluster 3 in the other. Merging those
into one column invents a comparison. So the default for anything label-shaped
is to keep them apart, and per-cell measurements — which do mean the same
thing everywhere — merge.
"""
import anndata
import numpy as np
import pandas as pd
import pytest
from scipy.sparse import csr_matrix

from xcell.adaptor import combine_datasets, describe_combine_columns


GENES = [f'g{i}' for i in range(20)]


def _base(n: int, seed: int, genes=None) -> anndata.AnnData:
    rng = np.random.default_rng(seed)
    genes = genes or GENES
    return anndata.AnnData(
        X=csr_matrix(rng.poisson(1.0, size=(n, len(genes))).astype(np.float32)),
        obs=pd.DataFrame(index=[f'c{i}' for i in range(n)]),
        var=pd.DataFrame(index=list(genes)),
    )


def _write(tmp_path, name: str, a: anndata.AnnData):
    p = tmp_path / name
    a.write_h5ad(p)
    return p


def _pair(tmp_path):
    """Two datasets sharing an obs measurement, an obs label, and a var flag."""
    rng = np.random.default_rng(0)
    a, b = _base(30, 1), _base(40, 2)
    for ad, n, seed in ((a, 30, 3), (b, 40, 4)):
        r = np.random.default_rng(seed)
        ad.obs['nCount_RNA'] = r.uniform(500, 5000, n)          # measurement
        ad.obs['seurat_clusters'] = pd.Categorical(
            r.choice(['0', '1', '2'], n)                         # independent labels
        )
        ad.var['highly_variable'] = r.random(len(GENES)) > 0.5   # per-dataset flag
        ad.var['gene_symbol'] = list(GENES)                      # identical everywhere
    a.obs['only_in_a'] = rng.random(30)
    return _write(tmp_path, 'a.h5ad', a), _write(tmp_path, 'b.h5ad', b)


# ---------------------------------------------------------------------------
# describe_combine_columns — what the picker shows before you commit
# ---------------------------------------------------------------------------
def _by_name(entries):
    return {e['name']: e for e in entries}


def test_describe_reports_which_datasets_hold_each_column(tmp_path):
    pa, pb = _pair(tmp_path)
    got = describe_combine_columns([pa, pb], ['A', 'B'])
    obs = _by_name(got['obs'])
    assert obs['nCount_RNA']['datasets'] == ['A', 'B']
    assert obs['only_in_a']['datasets'] == ['A']


def test_a_measurement_in_several_datasets_defaults_to_merge(tmp_path):
    pa, pb = _pair(tmp_path)
    obs = _by_name(describe_combine_columns([pa, pb], ['A', 'B'])['obs'])
    assert obs['nCount_RNA']['suggested'] == 'merge'


def test_a_label_in_several_datasets_defaults_to_separate(tmp_path):
    """Independent clusterings must not be silently pooled."""
    pa, pb = _pair(tmp_path)
    obs = _by_name(describe_combine_columns([pa, pb], ['A', 'B'])['obs'])
    assert obs['seurat_clusters']['suggested'] == 'separate'
    assert 'reason' in obs['seurat_clusters']


def test_a_low_cardinality_integer_column_is_treated_as_a_label(tmp_path):
    """Clusters stored as ints are still clusters."""
    a, b = _base(30, 1), _base(40, 2)
    a.obs['cluster'] = np.random.default_rng(1).integers(0, 5, 30)
    b.obs['cluster'] = np.random.default_rng(2).integers(0, 5, 40)
    pa, pb = _write(tmp_path, 'a.h5ad', a), _write(tmp_path, 'b.h5ad', b)
    obs = _by_name(describe_combine_columns([pa, pb], ['A', 'B'])['obs'])
    assert obs['cluster']['suggested'] == 'separate'


def test_a_column_only_one_dataset_has_defaults_to_merge(tmp_path):
    """No collision is possible, so there is nothing to keep apart."""
    pa, pb = _pair(tmp_path)
    obs = _by_name(describe_combine_columns([pa, pb], ['A', 'B'])['obs'])
    assert obs['only_in_a']['suggested'] == 'merge'


def test_a_var_column_that_is_identical_everywhere_defaults_to_first(tmp_path):
    """gene_symbol says the same thing in every input — one copy is enough."""
    pa, pb = _pair(tmp_path)
    var = _by_name(describe_combine_columns([pa, pb], ['A', 'B'])['var'])
    assert var['gene_symbol']['suggested'] == 'first'
    assert var['gene_symbol']['identical'] is True


def test_a_var_column_that_disagrees_defaults_to_separate(tmp_path):
    """Each dataset's own highly_variable is its own answer."""
    pa, pb = _pair(tmp_path)
    var = _by_name(describe_combine_columns([pa, pb], ['A', 'B'])['var'])
    assert var['highly_variable']['suggested'] == 'separate'
    assert var['highly_variable']['identical'] is False


def test_describe_compares_var_columns_only_over_shared_genes(tmp_path):
    """Genes outside the intersection are dropped by the combine anyway, so
    disagreement there must not force a column apart."""
    a = _base(30, 1)
    b = _base(40, 2, genes=GENES[:10] + ['extra1', 'extra2'])
    a.var['note'] = ['x'] * len(GENES)
    b.var['note'] = ['x'] * 10 + ['DIFFERENT', 'DIFFERENT']
    pa, pb = _write(tmp_path, 'a.h5ad', a), _write(tmp_path, 'b.h5ad', b)
    var = _by_name(describe_combine_columns([pa, pb], ['A', 'B'])['var'])
    assert var['note']['identical'] is True


# ---------------------------------------------------------------------------
# combine_datasets — applying the policy
# ---------------------------------------------------------------------------
def test_merge_keeps_one_column_with_every_cell_s_own_value(tmp_path):
    pa, pb = _pair(tmp_path)
    out = combine_datasets([pa, pb], ['A', 'B'], obs_policy={'nCount_RNA': 'merge'})
    assert 'nCount_RNA' in out.obs.columns
    assert out.obs['nCount_RNA'].notna().all()


def test_separate_splits_a_column_per_dataset(tmp_path):
    pa, pb = _pair(tmp_path)
    out = combine_datasets(
        [pa, pb], ['A', 'B'], obs_policy={'seurat_clusters': 'separate'},
    )
    assert 'seurat_clusters' not in out.obs.columns
    assert 'seurat_clusters__A' in out.obs.columns
    assert 'seurat_clusters__B' in out.obs.columns


def test_a_separated_column_is_empty_for_the_other_dataset_s_cells(tmp_path):
    """The whole point: B's cells have no value in A's clustering."""
    pa, pb = _pair(tmp_path)
    out = combine_datasets(
        [pa, pb], ['A', 'B'], obs_policy={'seurat_clusters': 'separate'},
    )
    is_a = out.obs['sample'] == 'A'
    assert out.obs.loc[is_a, 'seurat_clusters__A'].notna().all()
    assert out.obs.loc[~is_a, 'seurat_clusters__A'].isna().all()
    assert out.obs.loc[~is_a, 'seurat_clusters__B'].notna().all()


def test_drop_removes_a_column_entirely(tmp_path):
    pa, pb = _pair(tmp_path)
    out = combine_datasets([pa, pb], ['A', 'B'], obs_policy={'nCount_RNA': 'drop'})
    assert not [c for c in out.obs.columns if c.startswith('nCount_RNA')]


def test_columns_absent_from_the_policy_follow_the_suggested_default(tmp_path):
    pa, pb = _pair(tmp_path)
    out = combine_datasets([pa, pb], ['A', 'B'], obs_policy={})
    assert 'nCount_RNA' in out.obs.columns                # measurement → merged
    assert 'seurat_clusters__A' in out.obs.columns        # labels → separated


def test_no_policy_at_all_keeps_the_previous_behavior_for_shared_columns(tmp_path):
    """Callers that never asked for column control still get their columns."""
    pa, pb = _pair(tmp_path)
    out = combine_datasets([pa, pb], ['A', 'B'])
    assert 'nCount_RNA' in out.obs.columns
    assert 'sample' in out.obs.columns


def test_var_separate_keeps_each_dataset_s_annotation(tmp_path):
    pa, pb = _pair(tmp_path)
    out = combine_datasets(
        [pa, pb], ['A', 'B'], var_policy={'highly_variable': 'separate'},
    )
    assert 'highly_variable__A' in out.var.columns
    assert 'highly_variable__B' in out.var.columns
    assert out.var['highly_variable__A'].dtype == bool


def test_var_first_takes_one_copy(tmp_path):
    pa, pb = _pair(tmp_path)
    out = combine_datasets([pa, pb], ['A', 'B'], var_policy={'gene_symbol': 'first'})
    assert list(out.var['gene_symbol']) == list(out.var_names)


def test_var_drop_removes_the_column(tmp_path):
    pa, pb = _pair(tmp_path)
    out = combine_datasets([pa, pb], ['A', 'B'], var_policy={'gene_symbol': 'drop'})
    assert 'gene_symbol' not in out.var.columns


def test_separated_names_do_not_collide_with_an_existing_column(tmp_path):
    """A dataset that already has `x__A` must not be overwritten by splitting `x`."""
    a, b = _base(30, 1), _base(40, 2)
    r = np.random.default_rng(5)
    a.obs['x'] = pd.Categorical(r.choice(['p', 'q'], 30))
    b.obs['x'] = pd.Categorical(r.choice(['p', 'q'], 40))
    a.obs['x__A'] = np.arange(30, dtype=float)
    b.obs['x__A'] = np.arange(40, dtype=float)
    pa, pb = _write(tmp_path, 'a.h5ad', a), _write(tmp_path, 'b.h5ad', b)
    out = combine_datasets(
        [pa, pb], ['A', 'B'],
        obs_policy={'x': 'separate', 'x__A': 'merge'},
    )
    assert 'x__A' in out.obs.columns
    assert out.obs['x__A'].notna().all()          # the pre-existing column, intact
    assert 'x__A_1' in out.obs.columns or 'x__A__A' in out.obs.columns


def test_an_unknown_policy_value_is_rejected(tmp_path):
    pa, pb = _pair(tmp_path)
    with pytest.raises(ValueError, match="policy"):
        combine_datasets([pa, pb], ['A', 'B'], obs_policy={'nCount_RNA': 'nope'})


def test_a_policy_for_a_column_no_dataset_has_is_rejected(tmp_path):
    """A typo silently doing nothing is how you lose a column without noticing."""
    pa, pb = _pair(tmp_path)
    with pytest.raises(ValueError, match="ncount_rna"):
        combine_datasets([pa, pb], ['A', 'B'], obs_policy={'ncount_rna': 'merge'})


def test_the_sample_column_is_always_written(tmp_path):
    pa, pb = _pair(tmp_path)
    out = combine_datasets([pa, pb], ['A', 'B'], obs_policy={'sample': 'drop'})
    assert 'sample' in out.obs.columns
    assert set(out.obs['sample']) == {'A', 'B'}


def test_the_policy_is_recorded_in_uns(tmp_path):
    pa, pb = _pair(tmp_path)
    out = combine_datasets(
        [pa, pb], ['A', 'B'], obs_policy={'seurat_clusters': 'separate'},
    )
    rec = out.uns['xcell_combine']
    assert rec['obs_policy']['seurat_clusters'] == 'separate'
    assert rec['obs_policy']['nCount_RNA'] == 'merge'   # resolved default, recorded


def test_spatial_mode_applies_the_policy_too(tmp_path):
    a, b = _base(30, 1), _base(40, 2)
    a.obsm['X_spatial'] = np.random.default_rng(1).uniform(0, 100, (30, 2))
    b.obsm['X_spatial'] = np.random.default_rng(2).uniform(0, 100, (40, 2))
    r = np.random.default_rng(9)
    a.obs['cl'] = pd.Categorical(r.choice(['x', 'y'], 30))
    b.obs['cl'] = pd.Categorical(r.choice(['x', 'y'], 40))
    pa, pb = _write(tmp_path, 'a.h5ad', a), _write(tmp_path, 'b.h5ad', b)
    out = combine_datasets([pa, pb], ['A', 'B'], obs_policy={'cl': 'separate'})
    assert out.uns['xcell_combine']['mode'] == 'spatial'
    assert 'cl__A' in out.obs.columns


def test_the_combined_result_still_writes_to_h5ad(tmp_path):
    pa, pb = _pair(tmp_path)
    out = combine_datasets([pa, pb], ['A', 'B'])
    out.write_h5ad(tmp_path / 'combined.h5ad')
    back = anndata.read_h5ad(tmp_path / 'combined.h5ad')
    assert back.n_obs == out.n_obs


# ---------------------------------------------------------------------------
# routes
# ---------------------------------------------------------------------------
def _client():
    from fastapi.testclient import TestClient
    from xcell.main import app
    return TestClient(app)


def test_columns_route_describes_both_axes(tmp_path):
    pa, pb = _pair(tmp_path)
    r = _client().post('/api/combine/columns', json={
        'files': [{'file_path': str(pa), 'label': 'A'},
                  {'file_path': str(pb), 'label': 'B'}],
    })
    assert r.status_code == 200, r.text
    body = r.json()
    obs = _by_name(body['obs'])
    assert obs['seurat_clusters']['suggested'] == 'separate'
    assert obs['nCount_RNA']['suggested'] == 'merge'
    assert _by_name(body['var'])['gene_symbol']['suggested'] == 'first'


def test_columns_route_defaults_labels_from_filenames(tmp_path):
    pa, pb = _pair(tmp_path)
    body = _client().post('/api/combine/columns', json={
        'files': [{'file_path': str(pa)}, {'file_path': str(pb)}],
    }).json()
    assert _by_name(body['obs'])['only_in_a']['datasets'] == ['a']


def test_columns_route_404s_on_a_missing_file(tmp_path):
    pa, _ = _pair(tmp_path)
    r = _client().post('/api/combine/columns', json={
        'files': [{'file_path': str(pa)}, {'file_path': str(tmp_path / 'nope.h5ad')}],
    })
    assert r.status_code == 404


def test_columns_route_needs_two_files(tmp_path):
    pa, _ = _pair(tmp_path)
    r = _client().post('/api/combine/columns',
                       json={'files': [{'file_path': str(pa)}]})
    assert r.status_code == 400


def test_combine_route_applies_the_column_policy(tmp_path):
    pa, pb = _pair(tmp_path)
    r = _client().post('/api/combine', json={
        'files': [{'file_path': str(pa), 'label': 'A'},
                  {'file_path': str(pb), 'label': 'B'}],
        'slot': 'primary',
        'obs_policy': {'seurat_clusters': 'separate', 'nCount_RNA': 'drop'},
    })
    assert r.status_code == 200, r.text
    from xcell.api import routes as _routes
    obs_cols = set(_routes.get_adaptor('primary').adata.obs.columns)
    assert 'seurat_clusters__A' in obs_cols
    assert not [c for c in obs_cols if c.startswith('nCount_RNA')]


def test_combine_route_rejects_a_bad_policy_value(tmp_path):
    pa, pb = _pair(tmp_path)
    r = _client().post('/api/combine', json={
        'files': [{'file_path': str(pa)}, {'file_path': str(pb)}],
        'obs_policy': {'nCount_RNA': 'nope'},
    })
    assert r.status_code == 400
    assert 'policy' in r.json()['detail']
