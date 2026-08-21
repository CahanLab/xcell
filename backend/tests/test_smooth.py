"""kNN smoothing of expression, and the transform it should run on.

Smoothing is a local average, and averaging does not commute with the log.
Averaging log values is not the log of the average — on real ST data the gap is
large (see docs/measurements/2026-08-21-smoothing-transform.md), so *which
matrix* is smoothed changes the answer, not just its scale. These tests pin
down the pipeline the measurement recommends: average counts, then normalize,
then log.
"""
import anndata
import numpy as np
import pytest
from scipy.sparse import csr_matrix, issparse

from xcell.adaptor import DataAdaptor


def _adata(n=12, n_genes=6, seed=0):
    """A chain graph, so each cell's neighbours are exactly its two flanks —
    the smoothed value is then something you can work out by hand."""
    rng = np.random.default_rng(seed)
    X = rng.poisson(4.0, (n, n_genes)).astype(np.float32)
    # Deliberately uneven depth: half the cells are sequenced twice as deep,
    # which is what makes "normalize before or after" a real question.
    X[::2] *= 2
    ad = anndata.AnnData(X=csr_matrix(X))
    ad.var_names = [f'g{i}' for i in range(n_genes)]
    ad.obs_names = [f'c{i}' for i in range(n)]

    A = np.zeros((n, n), dtype=np.float32)
    for i in range(n - 1):
        A[i, i + 1] = A[i + 1, i] = 1.0
    ad.obsp['connectivities'] = csr_matrix(A)
    return ad


def _adaptor(**kw):
    return DataAdaptor('x.h5ad', adata=_adata(**kw))


def _dense(a, key):
    m = a.adata.layers[key]
    return np.asarray(m.todense()) if issparse(m) else np.asarray(m)


# --- the operator ---------------------------------------------------------

def test_a_cell_is_averaged_with_its_neighbours_and_itself():
    a = _adaptor()
    X = np.asarray(a.adata.X.todense())
    a.run_smooth(graph_key='connectivities', output_layer='s')
    # Cell 1's neighbours on the chain are 0 and 2; with self_loop_weight=1 the
    # average is over the three of them equally.
    np.testing.assert_allclose(
        _dense(a, 's')[1], (X[0] + X[1] + X[2]) / 3.0, rtol=1e-5,
    )


def test_the_result_lands_in_a_layer_and_leaves_x_alone():
    a = _adaptor()
    before = np.asarray(a.adata.X.todense()).copy()
    a.run_smooth(graph_key='connectivities', output_layer='s')
    assert 's' in a.adata.layers
    np.testing.assert_array_equal(np.asarray(a.adata.X.todense()), before)


def test_an_unknown_graph_is_refused_by_name():
    a = _adaptor()
    with pytest.raises(ValueError, match='nope'):
        a.run_smooth(graph_key='nope')


def test_an_unknown_source_layer_is_refused():
    # Not 'counts': the adaptor snapshots integer .X into layers['counts'] on
    # load, so that name exists on this fixture.
    a = _adaptor()
    with pytest.raises(ValueError, match='velocity'):
        a.run_smooth(graph_key='connectivities', source_layer='velocity')


# --- the transform it runs on --------------------------------------------

def test_by_default_nothing_is_transformed():
    """The existing contract: smoothing alone, so a counts matrix stays on a
    count scale."""
    a = _adaptor()
    X = np.asarray(a.adata.X.todense())
    a.run_smooth(graph_key='connectivities', output_layer='s', post_transform='none')
    np.testing.assert_allclose(
        _dense(a, 's')[1], (X[0] + X[1] + X[2]) / 3.0, rtol=1e-5,
    )


def test_cp10k_puts_every_cell_on_the_same_total():
    a = _adaptor()
    a.run_smooth(graph_key='connectivities', output_layer='s', post_transform='cp10k')
    sums = _dense(a, 's').sum(axis=1)
    np.testing.assert_allclose(sums, 1e4, rtol=1e-4)


def test_lognorm_is_log1p_of_the_normalized_smoothed_counts():
    a = _adaptor()
    a.run_smooth(graph_key='connectivities', output_layer='norm', post_transform='cp10k')
    a.run_smooth(graph_key='connectivities', output_layer='log', post_transform='lognorm')
    np.testing.assert_allclose(_dense(a, 'log'), np.log1p(_dense(a, 'norm')), rtol=1e-5)


def test_averaging_counts_then_normalizing_is_not_the_same_as_normalizing_first():
    """The reason the option exists. Averaging raw counts weights each
    neighbour by its depth — the Poisson-optimal thing to do — and that is a
    different estimate from averaging already-normalized profiles."""
    a = _adaptor()
    X = np.asarray(a.adata.X.todense())
    a.adata.layers['cp10k'] = csr_matrix(X / X.sum(axis=1, keepdims=True) * 1e4)

    a.run_smooth(graph_key='connectivities', output_layer='after', post_transform='cp10k')
    a.run_smooth(graph_key='connectivities', source_layer='cp10k', output_layer='before',
                 post_transform='cp10k')

    assert not np.allclose(_dense(a, 'after'), _dense(a, 'before'), rtol=1e-3)


def test_an_unknown_post_transform_is_refused_by_name():
    a = _adaptor()
    with pytest.raises(ValueError, match='sqrt'):
        a.run_smooth(graph_key='connectivities', post_transform='sqrt')


def test_the_transform_is_recorded_so_the_layer_can_be_identified_later():
    """layer_scale reads xcell's own action history to say what a matrix
    holds; a lognorm layer that claimed to be counts would mislead every
    downstream picker."""
    a = _adaptor()
    result = a.run_smooth(graph_key='connectivities', output_layer='s',
                          post_transform='lognorm')
    assert result['post_transform'] == 'lognorm'


def test_a_cell_with_no_counts_survives_normalization():
    """An empty row has nothing to rescale; dividing by its zero total would
    put NaN into the layer and poison everything downstream."""
    ad = _adata()
    X = np.asarray(ad.X.todense())
    X[0] = 0.0
    X[1] = 0.0  # and its only neighbours, so the smoothed row is empty too
    ad.X = csr_matrix(X)
    a = DataAdaptor('x.h5ad', adata=ad)
    a.adata.obsp['iso'] = csr_matrix(np.zeros((ad.n_obs, ad.n_obs), dtype=np.float32))
    a.run_smooth(graph_key='iso', output_layer='s', post_transform='lognorm')
    assert np.isfinite(_dense(a, 's')).all()


def test_the_layer_reports_how_it_was_made_including_the_transform():
    """Provenance is what tells a later reader whether a smoothed layer holds
    counts or log-normalized values — the classifier infers it from the values,
    but the history knows."""
    from xcell import layer_scale as ls

    a = _adaptor()
    a.run_smooth(graph_key='connectivities', output_layer='s', post_transform='lognorm')
    lines = ls.provenance_from_history(a._action_history, 's')
    assert any('smoothing' in line and 'log1p' in line for line in lines), lines
