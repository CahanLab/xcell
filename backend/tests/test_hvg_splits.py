"""Highly variable genes computed per data split, not just pooled.

Pooled HVG on a multi-sample dataset finds genes that vary *between* samples
as readily as genes that vary within them, so a batch effect scores as
biology. Splitting by sample answers a different, more useful question — what
varies inside each one — and the union / intersection of those answers is what
downstream multi-sample work (meta-programs, integration) actually wants.

The planted data has four gene blocks: variable in both groups, variable only
in A, variable only in B, and flat everywhere.
"""
import anndata
import numpy as np
import pandas as pd
import pytest
from scipy.sparse import csr_matrix

from xcell.adaptor import DataAdaptor


N_PER_GROUP = 400
# 25 genes per block, 100 in total: scanpy's dispersion binning needs enough
# genes per bin to rank anything, and asking for exactly the number planted as
# variable in a group (SHARED + that group's own block) makes the expected
# answer exact rather than approximate.
BLOCK = 25
TOP = 2 * BLOCK
SHARED, A_ONLY, B_ONLY, FLAT = range(4)


def _genes(block: int) -> list[str]:
    return [f'g{i:02d}' for i in range(block * BLOCK, (block + 1) * BLOCK)]


def _adata(tiny_group: bool = False, continuous: bool = False):
    """Two groups, with a different gene block variable in each.

    Two details make this exercise scanpy's HVG rather than fooling it:
    ``.X`` is log1p-normalized, because ``flavor='seurat'`` applies ``expm1``
    and reads raw counts as astronomically large; and every gene shares the
    same mean, differing only in variance, because the dispersion is
    normalized within mean bins precisely to cancel a mean-variance trend.
    """
    rng = np.random.default_rng(0)
    n_genes = 4 * BLOCK
    rows, labels = [], []
    for group in ('A', 'B'):
        mu = np.full((N_PER_GROUP, n_genes), 20.0)
        own = slice(BLOCK, 2 * BLOCK) if group == 'A' else slice(2 * BLOCK, 3 * BLOCK)
        for block in (slice(0, BLOCK), own):
            width = block.stop - block.start
            hi = rng.random((N_PER_GROUP, width)) < 0.5
            mu[:, block] = np.where(hi, 39.0, 1.0)   # mean 20, wildly overdispersed
        rows.append(np.log1p(rng.poisson(mu)).astype(np.float32))
        labels += [group] * N_PER_GROUP

    ad = anndata.AnnData(X=csr_matrix(np.vstack(rows)))
    ad.var_names = [f'g{i:02d}' for i in range(n_genes)]
    labels = np.array(labels, dtype=object)
    if tiny_group:
        labels[:3] = 'tiny'
    ad.obs['sample'] = pd.Categorical(labels)
    if continuous:
        ad.obs['score'] = rng.random(len(labels))
    return ad


def _adaptor(**kw) -> DataAdaptor:
    return DataAdaptor('x.h5ad', adata=_adata(**kw))


def _hvg(a: DataAdaptor, col: str) -> set[str]:
    return set(a.adata.var_names[a.adata.var[col].astype(bool).values])


# ---------------------------------------------------------------------------
# validation
# ---------------------------------------------------------------------------
def test_unknown_split_column_is_rejected():
    a = _adaptor()
    with pytest.raises(ValueError, match="nope"):
        a.run_highly_variable_genes(split_by='nope', n_top_genes=TOP)


def test_a_continuous_split_column_is_rejected():
    a = _adaptor(continuous=True)
    with pytest.raises(ValueError, match="categor"):
        a.run_highly_variable_genes(split_by='score', n_top_genes=TOP)


def test_subsetting_the_matrix_while_splitting_is_rejected():
    """`subset` would have to pick one group's answer to keep — there is no
    honest choice, so the caller has to make it explicitly."""
    a = _adaptor()
    with pytest.raises(ValueError, match="subset"):
        a.run_highly_variable_genes(split_by='sample', subset=True, n_top_genes=TOP)


def test_a_column_with_one_usable_group_is_rejected():
    a = _adaptor()
    a.adata.obs['one'] = pd.Categorical(['only'] * a.n_cells)
    with pytest.raises(ValueError, match="at least 2"):
        a.run_highly_variable_genes(split_by='one', n_top_genes=TOP)


# ---------------------------------------------------------------------------
# per-group columns
# ---------------------------------------------------------------------------
def test_one_boolean_column_per_group():
    a = _adaptor()
    res = a.run_highly_variable_genes(split_by='sample', n_top_genes=TOP)
    assert 'highly_variable__A' in a.adata.var.columns
    assert 'highly_variable__B' in a.adata.var.columns
    assert a.adata.var['highly_variable__A'].dtype == bool
    assert res['groups']['A']['n_highly_variable'] == TOP


def test_each_group_finds_its_own_variable_block():
    a = _adaptor()
    a.run_highly_variable_genes(split_by='sample', n_top_genes=TOP)
    hv_a, hv_b = _hvg(a, 'highly_variable__A'), _hvg(a, 'highly_variable__B')
    assert len(hv_a & set(_genes(A_ONLY))) >= 0.8 * BLOCK
    assert len(hv_a & set(_genes(B_ONLY))) <= 0.3 * BLOCK
    assert len(hv_b & set(_genes(B_ONLY))) >= 0.8 * BLOCK
    assert len(hv_b & set(_genes(A_ONLY))) <= 0.3 * BLOCK


def test_a_split_run_leaves_the_pooled_column_alone():
    """`highly_variable` means the pooled answer; a split run must not quietly
    redefine what every gene_subset dropdown already points at."""
    a = _adaptor()
    a.run_highly_variable_genes(n_top_genes=30)
    pooled = _hvg(a, 'highly_variable')
    a.run_highly_variable_genes(split_by='sample', n_top_genes=TOP)
    assert _hvg(a, 'highly_variable') == pooled


def test_no_pooled_column_is_invented_by_a_split_run():
    a = _adaptor()
    a.run_highly_variable_genes(split_by='sample', n_top_genes=TOP)
    assert 'highly_variable' not in a.adata.var.columns


def test_groups_below_the_cell_floor_are_skipped_and_reported():
    a = _adaptor(tiny_group=True)
    res = a.run_highly_variable_genes(
        split_by='sample', n_top_genes=TOP, min_cells_per_group=10,
    )
    assert 'highly_variable__tiny' not in a.adata.var.columns
    assert [s['group'] for s in res['skipped']] == ['tiny']
    assert res['skipped'][0]['n_cells'] == 3


def test_genes_unexpressed_in_a_group_are_not_variable_there():
    """Silencing genes that are otherwise the group's strongest HVGs: with no
    counts in A they cannot be variable in A, however they behave elsewhere.
    An all-zero gene also has an undefined dispersion, which is why they are
    dropped before scanpy bins anything rather than left to sort arbitrarily.
    """
    a = _adaptor()
    silent = _genes(SHARED)[:5]
    idx = [a.adata.var_names.get_loc(g) for g in silent]
    dense = a.adata.X.toarray()
    is_a = (a.adata.obs['sample'] == 'A').values
    dense[np.ix_(is_a, idx)] = 0.0
    a.adata.X = csr_matrix(dense)
    a.run_highly_variable_genes(split_by='sample', n_top_genes=TOP)
    assert not (_hvg(a, 'highly_variable__A') & set(silent))
    # Still expressed and still variable in B, so still found there.
    assert len(_hvg(a, 'highly_variable__B') & set(silent)) >= 3


# ---------------------------------------------------------------------------
# union / intersection
# ---------------------------------------------------------------------------
def test_union_and_intersection_are_only_written_when_asked():
    a = _adaptor()
    a.run_highly_variable_genes(split_by='sample', n_top_genes=TOP)
    assert 'highly_variable__union' not in a.adata.var.columns
    assert 'highly_variable__intersection' not in a.adata.var.columns


def test_union_is_the_or_of_every_group():
    a = _adaptor()
    a.run_highly_variable_genes(split_by='sample', n_top_genes=TOP, add_union=True)
    assert _hvg(a, 'highly_variable__union') == (
        _hvg(a, 'highly_variable__A') | _hvg(a, 'highly_variable__B')
    )


def test_intersection_is_the_and_of_every_group():
    a = _adaptor()
    a.run_highly_variable_genes(
        split_by='sample', n_top_genes=TOP, add_intersection=True,
    )
    assert _hvg(a, 'highly_variable__intersection') == (
        _hvg(a, 'highly_variable__A') & _hvg(a, 'highly_variable__B')
    )


def test_intersection_keeps_what_varies_everywhere_and_union_what_varies_anywhere():
    """The planted answer: shared genes survive the intersection, and the
    group-specific blocks only make the union."""
    a = _adaptor()
    a.run_highly_variable_genes(
        split_by='sample', n_top_genes=TOP, add_union=True, add_intersection=True,
    )
    inter, union = _hvg(a, 'highly_variable__intersection'), _hvg(a, 'highly_variable__union')
    assert len(inter & set(_genes(SHARED))) >= 0.8 * BLOCK
    assert len(inter & set(_genes(A_ONLY))) <= 0.3 * BLOCK
    assert len(union & set(_genes(A_ONLY))) >= 0.8 * BLOCK
    assert len(union & set(_genes(B_ONLY))) >= 0.8 * BLOCK
    assert inter <= union
    assert len(union & set(_genes(FLAT))) <= 0.3 * BLOCK


def test_the_result_reports_the_counts_for_every_column_written():
    a = _adaptor()
    res = a.run_highly_variable_genes(
        split_by='sample', n_top_genes=TOP, add_union=True, add_intersection=True,
    )
    assert res['columns'] == [
        'highly_variable__A', 'highly_variable__B',
        'highly_variable__union', 'highly_variable__intersection',
    ]
    assert res['n_union'] >= res['n_intersection']
    import json
    json.dumps(res)


def test_a_split_run_is_logged():
    a = _adaptor()
    a.run_highly_variable_genes(split_by='sample', n_top_genes=TOP)
    logged = [e for e in a._action_history if e['action'] == 'highly_variable_genes']
    assert logged and logged[-1]['params']['split_by'] == 'sample'


def test_pooled_behavior_is_unchanged_without_split_by():
    a = _adaptor()
    res = a.run_highly_variable_genes(n_top_genes=30)
    assert res['n_highly_variable'] == 30
    assert 'highly_variable' in a.adata.var.columns
    assert not [c for c in a.adata.var.columns if c.startswith('highly_variable__')]


# ---------------------------------------------------------------------------
# route
# ---------------------------------------------------------------------------
def test_route_passes_the_split_options_through():
    from fastapi.testclient import TestClient
    from xcell.api import routes
    from xcell.main import app

    routes.set_adaptor(_adaptor(), slot='primary')
    r = TestClient(app).post('/api/scanpy/highly_variable_genes', json={
        'n_top_genes': TOP, 'split_by': 'sample',
        'add_union': True, 'add_intersection': True,
    })
    assert r.status_code == 200, r.text
    body = r.json()
    assert 'highly_variable__union' in body['columns']
    adata = routes.get_adaptor('primary').adata
    assert 'highly_variable__intersection' in adata.var.columns


def test_route_reports_a_bad_split_column_as_400():
    from fastapi.testclient import TestClient
    from xcell.api import routes
    from xcell.main import app

    routes.set_adaptor(_adaptor(), slot='primary')
    r = TestClient(app).post('/api/scanpy/highly_variable_genes', json={
        'n_top_genes': TOP, 'split_by': 'nope',
    })
    assert r.status_code == 400


def test_route_treats_an_empty_split_column_as_no_split():
    """The generic Scanpy panel always sends the field, empty when unset — an
    empty string must mean "pooled", not a column named ''."""
    from fastapi.testclient import TestClient
    from xcell.api import routes
    from xcell.main import app

    routes.set_adaptor(_adaptor(), slot='primary')
    r = TestClient(app).post('/api/scanpy/highly_variable_genes', json={
        'n_top_genes': 30, 'split_by': '',
    })
    assert r.status_code == 200, r.text
    adata = routes.get_adaptor('primary').adata
    assert 'highly_variable' in adata.var.columns
    assert not [c for c in adata.var.columns if c.startswith('highly_variable__')]
