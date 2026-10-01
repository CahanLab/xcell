"""The pure diffusion module: diffusion maps and DPT on any connectivity graph.

On a symmetric, connected kNN the results must be scanpy's, bit for bit — the
module only wraps ``sc.tl.diffmap`` / ``sc.tl.dpt``. The other tests pin what
the wrapper adds for graphs that are not a scanpy kNN: isolated cells get NaN
(scanpy silently places them at 0), an asymmetric graph is symmetrised (eigsh
assumes symmetry), and cells DPT cannot reach get NaN rather than inf.
"""

import warnings

import anndata
import numpy as np
import pytest
import scanpy as sc
from scipy.sparse import csr_matrix

from xcell import diffusion as dm


def _knn(n=120, two_blobs=False, seed=0):
    """A scanpy kNN graph over a noisy curve (one component) or two far blobs."""
    rng = np.random.default_rng(seed)
    t = np.linspace(0, 1, n)
    X = np.column_stack([t * 10, np.sin(t * 6), rng.normal(0, 0.05, n)])
    X = np.hstack([X, rng.normal(0, 0.05, (n, 5))]).astype(np.float32)
    if two_blobs:
        X[n // 2:, 0] += 1000.0
    ad = anndata.AnnData(X=X)
    ad.obsm["X_pca"] = X
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        sc.pp.neighbors(ad, n_neighbors=10, use_rep="X_pca")
    return ad


def _no_warn():
    return warnings.catch_warnings()


# --- prepare_graph ---------------------------------------------------------

def test_prepare_graph_passes_a_symmetric_graph_through():
    ad = _knn()
    g = dm.prepare_graph(ad.obsp["connectivities"])
    assert not g.symmetrized
    assert g.n_isolated == 0
    assert list(g.kept) == list(range(ad.n_obs))
    assert (g.matrix != ad.obsp["connectivities"]).nnz == 0


def test_prepare_graph_symmetrises_by_averaging():
    W = csr_matrix(np.array([[0, 1, 0], [0, 0, 1], [1, 1, 0]], dtype=np.float32))
    g = dm.prepare_graph(W)
    assert g.symmetrized
    M = g.matrix.toarray()
    np.testing.assert_allclose(M, M.T)
    assert M[0, 1] == pytest.approx(0.5)   # one-way edge halved
    assert M[1, 2] == pytest.approx(1.0)   # mutual edge kept


def test_prepare_graph_drops_self_loops_and_finds_isolated_rows():
    W = np.zeros((4, 4), dtype=np.float32)
    W[0, 1] = W[1, 0] = 1
    W[1, 2] = W[2, 1] = 1
    W[3, 3] = 1  # only a self-loop: isolated
    g = dm.prepare_graph(csr_matrix(W))
    assert g.n_isolated == 1
    assert list(g.kept) == [0, 1, 2]
    assert g.matrix.shape == (3, 3)
    assert g.matrix.diagonal().sum() == 0


def test_prepare_graph_reports_components_largest_first():
    ad = _knn(n=100, two_blobs=True)
    g = dm.prepare_graph(ad.obsp["connectivities"])
    assert g.n_components == 2
    assert g.component_sizes == [50, 50]


def test_prepare_graph_rejects_negative_weights():
    W = csr_matrix(np.array([[0, -1], [-1, 0]], dtype=np.float32))
    with pytest.raises(ValueError, match="negative"):
        dm.prepare_graph(W)


def test_prepare_graph_rejects_a_non_square_matrix():
    with pytest.raises(ValueError, match="square"):
        dm.prepare_graph(csr_matrix(np.ones((3, 4), dtype=np.float32)))


# --- diffusion_map ---------------------------------------------------------

def test_diffusion_map_equals_scanpy_on_a_connected_knn():
    ad = _knn()
    sc.tl.diffmap(ad, n_comps=8)
    res = dm.diffusion_map(ad.obsp["connectivities"], n_comps=8)
    assert np.array_equal(res["X"], ad.obsm["X_diffmap"])
    assert np.array_equal(res["evals"], ad.uns["diffmap_evals"])
    assert res["n_isolated"] == 0
    assert res["symmetrized"] is False
    assert res["n_components"] == 1
    assert res["n_stationary"] == 1
    assert res["view_dims"] == [1, 2]
    assert res["warnings"] == []


def test_isolated_cells_get_nan_not_a_fake_origin():
    ad = _knn()
    W = ad.obsp["connectivities"].tolil()
    W[7, :] = 0
    W[:, 7] = 0
    W = W.tocsr()
    W.eliminate_zeros()
    res = dm.diffusion_map(W, n_comps=6)
    assert np.isnan(res["X"][7]).all()
    assert np.isfinite(np.delete(res["X"], 7, axis=0)).all()
    assert res["n_isolated"] == 1
    assert any("isolated" in w for w in res["warnings"])


def test_asymmetric_graph_is_symmetrised_and_flagged():
    ad = _knn()
    W = ad.obsp["connectivities"].tolil()
    W[0, 50] = 1.0  # a one-way edge
    res = dm.diffusion_map(W.tocsr(), n_comps=6)
    assert res["symmetrized"] is True
    assert np.isfinite(res["X"]).all()
    assert any("symmetri" in w for w in res["warnings"])


def test_disconnected_graph_reports_stationary_components():
    ad = _knn(n=100, two_blobs=True)
    res = dm.diffusion_map(ad.obsp["connectivities"], n_comps=6)
    assert res["n_components"] == 2
    assert res["n_stationary"] == 2
    assert res["view_dims"] == [2, 3]
    assert any("2 disconnected" in w for w in res["warnings"])


def test_diffusion_map_rejects_too_few_components():
    ad = _knn()
    with pytest.raises(ValueError, match="at least 3"):
        dm.diffusion_map(ad.obsp["connectivities"], n_comps=2)


def test_diffusion_map_rejects_more_components_than_cells_allow():
    W = csr_matrix(np.array([[0, 1, 1], [1, 0, 1], [1, 1, 0]], dtype=np.float32))
    with pytest.raises(ValueError, match="cells"):
        dm.diffusion_map(W, n_comps=5)


def test_diffusion_map_output_is_json_safe_shape():
    ad = _knn()
    res = dm.diffusion_map(ad.obsp["connectivities"], n_comps=5)
    assert res["X"].shape == (ad.n_obs, 5)
    assert res["X"].dtype == np.float32
    assert len(res["evals"]) == 5


# --- dpt_pseudotime ---------------------------------------------------------

def test_dpt_equals_scanpy_on_a_connected_knn():
    ad = _knn()
    sc.tl.diffmap(ad, n_comps=10)
    ad.uns["iroot"] = 3
    sc.tl.dpt(ad, n_dcs=10)
    res = dm.dpt_pseudotime(ad.obsp["connectivities"], ad.obsm["X_diffmap"],
                            ad.uns["diffmap_evals"], root=3, n_dcs=10)
    assert np.array_equal(res["pseudotime"], ad.obs["dpt_pseudotime"].to_numpy())
    assert res["pseudotime"][3] == 0.0
    assert res["n_unreachable"] == 0


def test_dpt_runs_along_the_curve():
    """The toy cells are ordered along a curve: pseudotime from the first end
    must rise with the index."""
    ad = _knn()
    res_map = dm.diffusion_map(ad.obsp["connectivities"], n_comps=10)
    res = dm.dpt_pseudotime(ad.obsp["connectivities"], res_map["X"], res_map["evals"],
                            root=0, n_dcs=10)
    from scipy.stats import spearmanr
    assert spearmanr(np.arange(ad.n_obs), res["pseudotime"]).statistic > 0.95


def test_dpt_gives_nan_to_cells_the_root_cannot_reach():
    ad = _knn(n=100, two_blobs=True)
    res_map = dm.diffusion_map(ad.obsp["connectivities"], n_comps=6)
    res = dm.dpt_pseudotime(ad.obsp["connectivities"], res_map["X"], res_map["evals"],
                            root=0, n_dcs=6)
    pt = res["pseudotime"]
    assert np.isnan(pt[50:]).all()
    assert np.isfinite(pt[:50]).all()
    assert res["n_unreachable"] == 50
    assert not np.isinf(pt).any()


def test_dpt_leaves_isolated_cells_nan():
    ad = _knn()
    W = ad.obsp["connectivities"].tolil()
    W[7, :] = 0
    W[:, 7] = 0
    W = W.tocsr()
    W.eliminate_zeros()
    res_map = dm.diffusion_map(W, n_comps=6)
    res = dm.dpt_pseudotime(W, res_map["X"], res_map["evals"], root=0, n_dcs=6)
    assert np.isnan(res["pseudotime"][7])
    assert np.isfinite(np.delete(res["pseudotime"], 7)).all()


def test_dpt_rejects_a_root_without_coordinates():
    ad = _knn()
    W = ad.obsp["connectivities"].tolil()
    W[7, :] = 0
    W[:, 7] = 0
    W = W.tocsr()
    W.eliminate_zeros()
    res_map = dm.diffusion_map(W, n_comps=6)
    with pytest.raises(ValueError, match="root"):
        dm.dpt_pseudotime(W, res_map["X"], res_map["evals"], root=7, n_dcs=6)


def test_dpt_clamps_n_dcs_to_the_map():
    ad = _knn()
    res_map = dm.diffusion_map(ad.obsp["connectivities"], n_comps=5)
    res = dm.dpt_pseudotime(ad.obsp["connectivities"], res_map["X"], res_map["evals"],
                            root=0, n_dcs=50)
    assert res["n_dcs"] == 5


# --- root_from_cells --------------------------------------------------------

def test_root_from_one_cell_is_that_cell():
    ad = _knn()
    res_map = dm.diffusion_map(ad.obsp["connectivities"], n_comps=6)
    assert dm.root_from_cells(res_map["X"], res_map["evals"], [17], n_dcs=6) == 17


def test_root_from_a_group_is_the_cell_nearest_its_centroid():
    ad = _knn()
    res_map = dm.diffusion_map(ad.obsp["connectivities"], n_comps=6)
    # Cells 10..30 lie along the curve; the middle one is nearest their centroid.
    root = dm.root_from_cells(res_map["X"], res_map["evals"], list(range(10, 31)), n_dcs=6)
    assert 17 <= root <= 23


def test_root_from_cells_ignores_cells_without_coordinates():
    X = np.array([[1, 0, 0], [1, 1, 1], [np.nan, np.nan, np.nan]], dtype=np.float32)
    evals = np.array([1.0, 0.9, 0.8], dtype=np.float32)
    assert dm.root_from_cells(X, evals, [1, 2], n_dcs=3) == 1


def test_root_from_cells_rejects_cells_all_without_coordinates():
    X = np.array([[1, 0, 0], [np.nan, np.nan, np.nan]], dtype=np.float32)
    evals = np.array([1.0, 0.9, 0.8], dtype=np.float32)
    with pytest.raises(ValueError, match="coordinates"):
        dm.root_from_cells(X, evals, [1], n_dcs=3)
