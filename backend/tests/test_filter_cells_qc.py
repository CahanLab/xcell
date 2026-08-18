"""Filter Cells: QC distributions endpoint + multi-criteria filtering.

The UI shows per-cell counts/genes histograms with min/max cutoffs on each,
which means one request can carry several thresholds at once. scanpy's
``sc.pp.filter_cells`` accepts exactly one threshold per call, so the adaptor
combines criteria itself (keep = AND of all bounds, inclusive, matching what
sequential scanpy calls would leave).
"""
import numpy as np
import pandas as pd
import anndata
import pytest
from scipy.sparse import csr_matrix

from xcell.adaptor import DataAdaptor
from xcell.api import routes


def _adata() -> anndata.AnnData:
    # 5 cells with known counts and gene counts:
    #   c0: counts  3, genes 3
    #   c1: counts 10, genes 2
    #   c2: counts  1, genes 1
    #   c3: counts 30, genes 3
    #   c4: counts  6, genes 2
    X = np.array([
        [1, 1, 1, 0],
        [5, 5, 0, 0],
        [1, 0, 0, 0],
        [10, 10, 10, 0],
        [3, 3, 0, 0],
    ], dtype=np.float32)
    return anndata.AnnData(
        X=csr_matrix(X),
        obs=pd.DataFrame(index=[f'c{i}' for i in range(5)]),
        var=pd.DataFrame(index=[f'g{i}' for i in range(4)]),
    )


def _adaptor() -> DataAdaptor:
    return DataAdaptor('dummy.h5ad', adata=_adata())


def test_qc_distributions_report_per_cell_counts_and_genes():
    a = _adaptor()
    qc = a.filter_cells_qc()
    assert qc['counts'] == [3.0, 10.0, 1.0, 30.0, 6.0]
    assert qc['genes'] == [3, 2, 1, 3, 2]
    # JSON-serializable plain python numbers, not numpy scalars
    assert all(type(v) is float for v in qc['counts'])
    assert all(type(v) is int for v in qc['genes'])


def test_multiple_criteria_combine_as_and():
    a = _adaptor()
    result = a.run_filter_cells(min_counts=3, max_counts=20, min_genes=2)
    # c0 (3 counts, 3 genes), c1 (10, 2), c4 (6, 2) survive;
    # c2 fails min_counts and min_genes, c3 fails max_counts.
    assert result['n_cells_after'] == 3
    assert result['n_cells_removed'] == 2
    assert list(a.adata.obs_names) == ['c0', 'c1', 'c4']


def test_bounds_are_inclusive_like_scanpy():
    a = _adaptor()
    result = a.run_filter_cells(min_counts=3, max_counts=30)
    # equality on both bounds keeps the cell: c0 (==3) and c3 (==30) stay
    assert list(a.adata.obs_names) == ['c0', 'c1', 'c3', 'c4']
    assert result['n_cells_removed'] == 1


def test_single_criterion_matches_scanpy_exactly():
    import scanpy as sc
    ours = _adaptor()
    ours.run_filter_cells(min_genes=2)

    reference = _adata()
    sc.pp.filter_cells(reference, min_genes=2)
    assert list(ours.adata.obs_names) == list(reference.obs_names)


def test_subset_only_evaluates_active_cells():
    a = _adaptor()
    # Only c0..c2 are evaluated; c3 (30 counts) would fail max_counts=20 but
    # is outside the active set, so it survives.
    result = a.run_filter_cells(min_counts=3, max_counts=20,
                                active_cell_indices=[0, 1, 2])
    assert list(a.adata.obs_names) == ['c0', 'c1', 'c3', 'c4']
    assert result['n_cells_removed'] == 1


def test_no_thresholds_is_a_noop():
    a = _adaptor()
    result = a.run_filter_cells()
    assert result['n_cells_removed'] == 0
    assert a.adata.n_obs == 5


def test_qc_route_returns_distributions(monkeypatch):
    a = _adaptor()
    monkeypatch.setattr(routes, 'get_adaptor', lambda dataset=None: a)
    out = routes.filter_cells_qc(dataset=None)
    assert out['counts'][3] == 30.0
    assert out['genes'][0] == 3
