"""Tests for the pure multicontour logic (xcell.multicontour)."""

import numpy as np
import anndata
from scipy.sparse import csr_matrix

import xcell.multicontour as mc


def make_adata():
    # 5x5 grid of spots; 2 gene sets high in disjoint corners.
    coords = np.array([[float(i), float(j)] for i in range(5) for j in range(5)])
    n = coords.shape[0]
    X = np.zeros((n, 4), dtype=np.float32)
    for k, (i, j) in enumerate([(i, j) for i in range(5) for j in range(5)]):
        if i <= 1:
            X[k, 0] = X[k, 1] = 10.0  # setA genes
        if i >= 3:
            X[k, 2] = X[k, 3] = 10.0  # setB genes
    ad = anndata.AnnData(X=csr_matrix(X))
    ad.var_names = ["a0", "a1", "b0", "b1"]
    ad.obsm["spatial"] = coords
    ad.obsm["X_pca"] = coords.copy()  # simple 2D profile
    return ad


# --- Task 2: score_module / binarize / auto_cutoff ---

def test_score_module_returns_bands_and_cutoff():
    ad = make_adata()
    res = mc.score_module(ad, ["a0", "a1"], contour_levels=3,
                          log_transform=True, clip_percentiles=(1, 99),
                          grid_res=40, smooth_sigma=2.0)
    assert res["score"].shape == (ad.n_obs,)
    assert res["bands"].shape == (ad.n_obs,)
    assert res["auto_cutoff"] == res["thresholds"][-1]
    assert len(res["histogram"]) == len(res["band_values"])


def test_binarize_uses_cutoff():
    ad = make_adata()
    res = mc.score_module(ad, ["a0", "a1"], contour_levels=3, log_transform=True,
                          clip_percentiles=(1, 99), grid_res=40, smooth_sigma=2.0)
    high = mc.binarize(res["bands"], res["auto_cutoff"])
    assert high.dtype == bool
    assert high.sum() > 0
    assert high.sum() < ad.n_obs


# --- Task 3: assign_tissue / conflict resolution ---

def test_assign_single_high_and_unassigned():
    highs = {"A": np.array([True, False, False]),
             "B": np.array([False, True, False])}
    labels, status = mc.assign_tissue(
        highs, adata=None, profile_k=5, spatial_conn=None, pca=None, coords=None)
    assert list(labels) == ["A", "B", "unassigned"]
    assert list(status) == ["single", "single", "unassigned"]


def test_conflict_resolved_by_neighbor_majority():
    # 4 spots in a line: 0=A,1=A,2=conflict(A&B),3=B. spot2's graph neighbors are
    # 1 (A) and 3 (B); in PCA space spot2 sits much closer to 1, so the nearest
    # neighbor vote resolves it to A.
    highs = {"A": np.array([True, True, True, False]),
             "B": np.array([False, False, True, True])}
    coords = np.array([[0., 0.], [1., 0.], [2., 0.], [3., 0.]])
    pca = np.array([[0., 0.], [1., 0.], [1.2, 0.], [3., 0.]])  # spot2 ~ spot1
    conn = np.array([
        [0, 1, 0, 0], [1, 0, 1, 0], [0, 1, 0, 1], [0, 0, 1, 0]], dtype=float)
    labels, status = mc.assign_tissue(
        highs, adata=None, profile_k=1, spatial_conn=conn, pca=pca, coords=coords)
    assert labels[2] == "A"
    assert status[2] == "resolved"


def test_conflict_unresolved_when_no_unambiguous_neighbors():
    highs = {"A": np.array([True, True]), "B": np.array([True, True])}
    coords = np.array([[0., 0.], [1., 0.]])
    conn = np.array([[0, 1], [1, 0]], dtype=float)
    labels, status = mc.assign_tissue(
        highs, adata=None, profile_k=2, spatial_conn=conn, pca=coords.copy(), coords=coords)
    assert list(labels) == ["unassigned", "unassigned"]
    assert list(status) == ["unassigned", "unassigned"]


def test_conflict_coordinate_knn_fallback_when_no_graph():
    # No spatial graph -> coordinate-kNN pool. spot2 conflict; nearest unambiguous is spot1 (A).
    highs = {"A": np.array([True, True, True, False]),
             "B": np.array([False, False, True, True])}
    coords = np.array([[0., 0.], [1., 0.], [1.2, 0.], [3., 0.]])
    labels, status = mc.assign_tissue(
        highs, adata=None, profile_k=3, spatial_conn=None, pca=coords.copy(), coords=coords)
    assert labels[2] == "A"
    assert status[2] == "resolved"


# --- Task 4: data-aware defaults ---

def test_suggest_grid_res_scales_with_spots():
    small = mc.suggest_grid_res(n_spots=500)
    big = mc.suggest_grid_res(n_spots=344021)
    assert 50 <= small <= 600 and 50 <= big <= 600
    assert big >= small


def test_suggest_smooth_sigma_positive():
    coords = np.array([[float(i), float(j)] for i in range(10) for j in range(10)])
    sigma = mc.suggest_smooth_sigma(coords, grid_res=100)
    assert sigma > 0
