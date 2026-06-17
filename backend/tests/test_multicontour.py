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


# --- section-aware scoring ---

def _two_section_adata():
    """Two adjacent blobs (sections A and B). A gene 'mA' is high across A and
    zero across B; coords place A at x<5, B at x>=5 (same y range)."""
    cells = []
    secs = []
    for i in range(10):
        for j in range(10):
            cells.append((float(i), float(j)))
            secs.append("A" if i < 5 else "B")
    coords = np.array(cells)
    n = coords.shape[0]
    sections = np.array(secs)
    X = np.zeros((n, 2), dtype=np.float32)
    X[sections == "A", 0] = 10.0  # gene mA high in A, 0 in B
    X[sections == "B", 1] = 10.0  # gene mB high in B, 0 in A
    ad = anndata.AnnData(X=csr_matrix(X))
    ad.var_names = ["mA", "mB"]
    ad.obsm["spatial"] = coords
    ad.obsm["X_pca"] = coords.copy()
    ad.obs["section"] = sections
    return ad, sections


def test_section_aware_scoring_no_bleed_into_other_section():
    ad, sections = _two_section_adata()
    # Module on gene mA (high in A only). Without sections, smoothing bleeds
    # mA's signal across the A/B boundary into B cells.
    glob = mc.score_module(ad, ["mA"], contour_levels=3, log_transform=False,
                           grid_res=40, smooth_sigma=2.0)
    aware = mc.score_module(ad, ["mA"], contour_levels=3, log_transform=False,
                            grid_res=40, smooth_sigma=2.0, sections=sections)
    b = sections == "B"
    assert glob["score"][b].max() > 1e-3          # global bleeds into B
    assert aware["score"][b].max() < 1e-6          # section-aware: no bleed


def test_section_aware_A_independent_of_B_position():
    # Section A's section-aware scores must not depend on where section B sits in
    # space (B's expression is unchanged, so global normalization is unchanged).
    ad1, sections = _two_section_adata()
    ad2 = ad1.copy()
    coords2 = np.asarray(ad2.obsm["spatial"]).copy()
    coords2[sections == "B", 0] += 1000.0  # move B far away
    ad2.obsm["spatial"] = coords2

    a = sections == "A"
    s1 = mc.score_module(ad1, ["mA"], contour_levels=3, log_transform=False,
                         grid_res=40, smooth_sigma=2.0, sections=sections)
    s2 = mc.score_module(ad2, ["mA"], contour_levels=3, log_transform=False,
                         grid_res=40, smooth_sigma=2.0, sections=sections)
    np.testing.assert_allclose(s1["score"][a], s2["score"][a], rtol=1e-6, atol=1e-6)


def test_section_aware_small_section_no_crash():
    # A 2-cell section can't be cubic-interpolated; must not crash.
    coords = np.array([[0., 0.], [1., 0.], [2., 0.], [3., 0.],
                       [100., 0.], [101., 0.]])
    sections = np.array(["A", "A", "A", "A", "B", "B"])
    X = np.ones((6, 1), dtype=np.float32)
    ad = anndata.AnnData(X=csr_matrix(X))
    ad.var_names = ["g"]
    ad.obsm["spatial"] = coords
    res = mc.score_module(ad, ["g"], contour_levels=2, log_transform=False,
                          grid_res=20, smooth_sigma=1.0, sections=sections)
    assert np.isfinite(res["score"]).all()
    assert res["score"].shape == (6,)


# --- section-aware conflict resolution ---

def test_assign_tissue_drops_cross_section_neighbors():
    # spot2 conflicts (A&B). Its only unambiguous neighbor in its OWN section is
    # spot1 (B-section, label B). spot3 (label A) is in a different section and
    # must be ignored even though it's a graph neighbor and PCA-closer.
    highs = {"A": np.array([False, False, True, True]),
             "B": np.array([True, True, True, False])}
    coords = np.array([[0., 0.], [1., 0.], [2., 0.], [3., 0.]])
    pca = np.array([[0., 0.], [1., 0.], [2.0, 0.], [2.1, 0.]])  # spot2 closest to spot3
    conn = np.array([
        [0, 1, 0, 0], [1, 0, 1, 0], [0, 1, 0, 1], [0, 0, 1, 0]], dtype=float)
    sections = np.array(["B", "B", "B", "A"])
    labels, status = mc.assign_tissue(
        highs, adata=None, profile_k=5, spatial_conn=conn, pca=pca, coords=coords,
        sections=sections)
    # spot2 is in section B; only same-section unambiguous neighbor is spot1 (B).
    assert labels[2] == "B"
    assert status[2] == "resolved"
