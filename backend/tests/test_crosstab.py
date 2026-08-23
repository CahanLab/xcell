"""Counting cells two ways at once, for the stacked barplot.

A stacked bar is a claim about composition: "38% of cluster 3 is proximal".
That only holds if every cell in the bar is accounted for, so the awkward
cases — a cell with no value for the stacking column, a category no cell
uses — have to be counted rather than dropped. A bar that silently sums to
0.9 looks exactly like one that sums to 1.
"""
import anndata
import numpy as np
import pandas as pd
import pytest
from scipy.sparse import csr_matrix

from xcell.adaptor import DataAdaptor


def _adata(obs: pd.DataFrame) -> anndata.AnnData:
    n = len(obs)
    rng = np.random.default_rng(0)
    return anndata.AnnData(
        X=csr_matrix(rng.poisson(1.0, size=(n, 5)).astype(np.float32)),
        obs=obs,
        var=pd.DataFrame(index=[f'g{i}' for i in range(5)]),
    )


def _adaptor(adata: anndata.AnnData) -> DataAdaptor:
    return DataAdaptor("x.h5ad", adata=adata)


def _simple() -> DataAdaptor:
    obs = pd.DataFrame({
        'leiden': pd.Categorical(['0', '0', '0', '1', '1', '2']),
        'territory': pd.Categorical(
            ['prox', 'prox', 'dist', 'dist', 'dist', 'prox']),
    }, index=[f'c{i}' for i in range(6)])
    return _adaptor(_adata(obs))


def test_counts_every_pair():
    out = _simple().crosstab('leiden', 'territory')
    assert out['a_categories'] == ['0', '1', '2']
    assert out['b_categories'] == ['dist', 'prox']
    # rows follow a_categories, columns follow b_categories
    assert out['counts'] == [[1, 2], [2, 0], [0, 1]]


def test_every_cell_lands_in_exactly_one_box():
    out = _simple().crosstab('leiden', 'territory')
    assert sum(sum(row) for row in out['counts']) == out['n_cells'] == 6


def test_keeps_a_category_no_cell_uses():
    # A cluster emptied by a filter still belongs on the axis — dropping it
    # silently renumbers everything to its right.
    obs = pd.DataFrame({
        'leiden': pd.Categorical(['0', '0', '2'], categories=['0', '1', '2']),
        'territory': pd.Categorical(['prox', 'dist', 'prox']),
    })
    out = _adaptor(_adata(obs)).crosstab('leiden', 'territory')
    assert out['a_categories'] == ['0', '1', '2']
    assert out['counts'][1] == [0, 0]


def test_counts_cells_with_no_value_rather_than_dropping_them():
    # Territories leave a cell blank when it has no coordinate. Dropping those
    # would make the bar sum to less than the cluster.
    obs = pd.DataFrame({
        'leiden': pd.Categorical(['0', '0', '0']),
        'territory': pd.Categorical(['prox', None, None]),
    })
    out = _adaptor(_adata(obs)).crosstab('leiden', 'territory')
    assert out['b_categories'] == ['prox', '(none)']
    assert out['counts'] == [[1, 2]]
    assert sum(out['counts'][0]) == out['n_cells'] == 3


def test_leaves_the_missing_bucket_out_when_nothing_is_missing():
    out = _simple().crosstab('leiden', 'territory')
    assert '(none)' not in out['b_categories']


def test_carries_the_colors_the_dataset_already_has():
    obs = pd.DataFrame({
        'leiden': pd.Categorical(['0', '1']),
        'territory': pd.Categorical(['prox', 'dist']),
    })
    ad = _adata(obs)
    ad.uns['territory_colors'] = ['#111111', '#222222']
    out = _adaptor(ad).crosstab('leiden', 'territory')
    # Aligned to b_categories, which are sorted: dist, prox
    assert out['b_colors'] == ['#111111', '#222222']


def test_has_no_color_for_the_missing_bucket():
    obs = pd.DataFrame({
        'leiden': pd.Categorical(['0', '1']),
        'territory': pd.Categorical(['prox', None]),
    })
    ad = _adata(obs)
    ad.uns['territory_colors'] = ['#111111']
    out = _adaptor(ad).crosstab('leiden', 'territory')
    assert out['b_categories'] == ['prox', '(none)']
    assert out['b_colors'] == ['#111111', None]


def test_works_on_plain_string_columns():
    obs = pd.DataFrame({'a': ['x', 'x', 'y'], 'b': ['p', 'q', 'q']})
    out = _adaptor(_adata(obs)).crosstab('a', 'b')
    assert out['a_categories'] == ['x', 'y']
    assert out['counts'] == [[1, 1], [0, 1]]


def test_refuses_a_continuous_column():
    # Binning a continuous variable is a decision the caller has to make; doing
    # it here would invent categories nobody chose.
    obs = pd.DataFrame({
        'leiden': pd.Categorical(['0', '1']),
        'n_counts': [1.5, 2.5],
    })
    with pytest.raises(ValueError, match='continuous'):
        _adaptor(_adata(obs)).crosstab('leiden', 'n_counts')


def test_says_which_column_is_missing():
    with pytest.raises(KeyError, match='nope'):
        _simple().crosstab('leiden', 'nope')


def test_refuses_to_cross_a_column_with_itself():
    with pytest.raises(ValueError, match='same column'):
        _simple().crosstab('leiden', 'leiden')
