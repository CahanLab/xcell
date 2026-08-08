"""Building a neighbor graph on an active cell selection.

`run_neighbors` builds a throwaway AnnData holding only the selected cells'
`.obsm` representation — deliberately with no `.X`, since the matrix is not
needed. But scanpy's `use_rep=None` default resolves the representation from
`adata.n_vars`, which on that throwaway object is 0, so it chose `.X` and hit
None. The result was that Neighbors on a selection failed for *every* dataset,
not just small ones.

Naming the representation explicitly is the fix; these tests keep the default
from drifting back.
"""
import anndata
import numpy as np
import pytest
from scipy.sparse import csr_matrix

from xcell.adaptor import DataAdaptor


def _adaptor(n_cells=60, n_genes=80, seed=0):
    rng = np.random.default_rng(seed)
    ad = anndata.AnnData(
        X=csr_matrix(rng.random((n_cells, n_genes)).astype(np.float32))
    )
    ad.var_names = [f'g{i}' for i in range(n_genes)]
    ad.obs_names = [f'c{i}' for i in range(n_cells)]
    return DataAdaptor('x.h5ad', adata=ad)


@pytest.mark.parametrize('n_genes', [8, 80])
def test_neighbors_runs_on_a_selection(n_genes):
    """Both sides of scanpy's n_vars > 50 branch — the bug hit either way."""
    a = _adaptor(n_genes=n_genes)
    active = list(range(30))
    a.run_pca(n_comps=5, active_cell_indices=active)
    result = a.run_neighbors(n_neighbors=5, active_cell_indices=active)
    assert result['status'] == 'completed'


def test_the_graph_is_full_size_with_edges_only_among_active_cells():
    a = _adaptor()
    active = list(range(30))
    a.run_pca(n_comps=5, active_cell_indices=active)
    a.run_neighbors(n_neighbors=5, active_cell_indices=active)

    conn = a.adata.obsp['connectivities'].tocoo()
    assert conn.shape == (60, 60)
    assert conn.nnz > 0
    inactive = set(range(30, 60))
    assert not (set(conn.row.tolist()) & inactive)
    assert not (set(conn.col.tolist()) & inactive)


def test_a_pc_subset_representation_is_honoured_on_a_selection():
    """use_rep names an obsm key; the subset path has to carry it through."""
    a = _adaptor()
    a.run_pca(n_comps=6)
    a.create_pca_subset(drop_pc_indices=[1])
    key = next(k for k in a.adata.obsm if k.startswith('X_pca_'))
    result = a.run_neighbors(n_neighbors=5, use_rep=key,
                             active_cell_indices=list(range(30)))
    assert result['status'] == 'completed'


def test_the_whole_dataset_path_is_unaffected():
    a = _adaptor()
    a.run_pca(n_comps=5)
    assert a.run_neighbors(n_neighbors=5)['status'] == 'completed'
    assert a.adata.obsp['connectivities'].shape == (60, 60)


def test_clustering_a_selection_end_to_end():
    """The sequence a user actually performs: lasso, PCA, neighbors, Leiden."""
    a = _adaptor()
    active = list(range(30))
    a.run_pca(n_comps=5, active_cell_indices=active)
    a.run_neighbors(n_neighbors=5, active_cell_indices=active)
    result = a.run_leiden(resolution=1.0, active_cell_indices=active)
    assert result['n_clusters'] >= 1
    labels = a.adata.obs['leiden'].astype(str)
    assert (labels.iloc[30:] == 'unassigned').all()
    assert not (labels.iloc[:30] == 'unassigned').any()
