"""Tests for the ligand-receptor spatial scoring core (CytoSignal-style)."""

import numpy as np
from scipy import sparse

from xcell import ligrec


def _grid(n=8):
    """n x n grid of unit-spaced points; returns (coords, n_cells)."""
    xs, ys = np.meshgrid(np.arange(n), np.arange(n))
    coords = np.column_stack([xs.ravel(), ys.ravel()]).astype(float)
    return coords, coords.shape[0]


# --- neighbor graphs --------------------------------------------------------

def test_median_nn_distance_unit_grid():
    coords, _ = _grid(6)
    assert abs(ligrec.median_nn_distance(coords) - 1.0) < 1e-9


def test_gaussian_graph_rows_sum_to_one_and_self_included():
    coords, n = _grid(6)
    A = ligrec.gaussian_graph(coords, radius=2.0, sigma=1.0)
    assert A.shape == (n, n)
    rs = np.asarray(A.sum(axis=1)).ravel()
    assert np.allclose(rs, 1.0, atol=1e-9)
    # Self weight present on the diagonal.
    assert (A.diagonal() > 0).all()


def test_gaussian_graph_respects_radius():
    coords, n = _grid(6)
    A = ligrec.gaussian_graph(coords, radius=1.5, sigma=1.0)
    # With radius 1.5 (unit grid), a cell connects to itself + 4-neighbors +
    # diagonal neighbors (dist sqrt2 ~1.41), but not cells 2 apart.
    # Pick the center-ish cell index 14 (row2,col2 in 6x6 -> 2*6+2=14).
    i = 14
    nbrs = A[i].toarray().ravel()
    # cell two to the right is index 16 (row2,col4); distance 2 > radius -> 0
    assert nbrs[16] == 0


def test_delaunay_graph_connects_adjacent_points():
    coords, n = _grid(5)
    A, A_bin = ligrec.delaunay_graph(coords)
    assert A.shape == (n, n) and A_bin.shape == (n, n)
    # binary adjacency symmetric
    assert (A_bin != A_bin.T).nnz == 0
    # every interior point has at least 4 neighbors
    deg = np.asarray((A_bin > 0).sum(axis=1)).ravel()
    assert deg.max() >= 4


def test_to_mean_graph_rows_sum_to_one():
    coords, n = _grid(5)
    _, A_bin = ligrec.delaunay_graph(coords)
    G = ligrec.to_mean_graph(A_bin)
    rs = np.asarray(G.sum(axis=1)).ravel()
    # rows with neighbors sum to 1
    assert np.allclose(rs[rs > 0], 1.0, atol=1e-9)


def test_delaunay_max_radius_drops_long_edges():
    # Two clusters far apart; with a small max_radius no cross-cluster edges.
    left = np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]])
    right = left + np.array([100.0, 0.0])
    coords = np.vstack([left, right])
    _, A_bin = ligrec.delaunay_graph(coords, max_radius=5.0)
    # no edge between {0,1,2} and {3,4,5}
    block = A_bin.toarray()[:3, 3:]
    assert block.sum() == 0


# --- score formula ----------------------------------------------------------

def test_normalize_log1p_zero_libsize_safe():
    M = np.array([[1.0, 2.0], [0.0, 0.0]])
    lib = np.array([3.0, 0.0])
    out = ligrec.normalize_log1p(M, lib)
    assert np.isfinite(out).all()
    assert (out[1] == 0).all()  # zero-lib cell -> zeros, not nan/inf


def test_score_block_is_product_of_imputed_ligand_and_receptor():
    # 3 cells in a row; ligand high at cell 0, receptor high at cell 1.
    coords = np.array([[0.0, 0.0], [1.0, 0.0], [2.0, 0.0]])
    X = np.array([
        [10.0, 0.0],   # cell0: ligand gene 0 high
        [0.0, 10.0],   # cell1: receptor gene 1 high
        [0.0, 0.0],    # cell2: nothing
    ])
    lib = X.sum(axis=1)
    A = ligrec.gaussian_graph(coords, radius=1.5, sigma=1.0)
    _, A_bin = ligrec.delaunay_graph(coords)
    G = ligrec.to_mean_graph(A_bin)
    scores = ligrec.score_block(
        X, lib, A, G, lig_idx=[np.array([0])], rec_idx=[np.array([1])]
    )
    assert scores.shape == (3, 1)
    # Cell 1 receives ligand from neighbor cell 0 and expresses the receptor, so
    # it should have the highest LR score of the three cells.
    assert scores[1, 0] == scores[:, 0].max()
    assert scores[1, 0] > 0


# --- orchestrator: permutation significance ---------------------------------

def _signaling_field():
    """40x40 grid with a localized secreted-ligand source (realistic geometry).

    Ligand L is secreted from a small, dense central source (globally rare).
    Receptor Rnear sits in a thin shell immediately around the source, within
    the ligand's diffusion reach, so L->Rnear signaling fires there. Receptor
    Rfar is expressed only far away, out of reach -- an anti-colocalized negative
    control. N is noise. Because the ligand is globally rare but locally dense,
    a permuted (scattered) ligand essentially never reconstructs the local
    density, giving the true signal decisive significance. Genes: L, Rnear,
    Rfar, N."""
    w = 40
    xs, ys = np.meshgrid(np.arange(w), np.arange(w))
    coords = np.column_stack([xs.ravel(), ys.ravel()]).astype(float)
    rng = np.random.default_rng(0)
    center = np.array([20.0, 20.0])
    d = np.linalg.norm(coords - center, axis=1)
    L = np.where(d <= 2.0, 20.0, 0.0)                     # dense, rare source
    Rnear = np.where((d > 2.0) & (d <= 3.7), 20.0, 0.0)   # shell within reach
    Rfar = np.where(d >= 10.0, 20.0, 0.0)                 # out of ligand's reach
    N = rng.poisson(1.0, coords.shape[0]).astype(float)
    X = np.column_stack([L, Rnear, Rfar, N])
    var_names = ["L", "Rnear", "Rfar", "N"]
    return X, var_names, coords


def test_compute_ligrec_detects_colocalized_signaling():
    X, var_names, coords = _signaling_field()
    pairs = [
        {"interaction": "L->Rnear", "ligand": ["L"], "receptor": ["Rnear"], "type": "diffusion"},
        {"interaction": "L->Rfar", "ligand": ["L"], "receptor": ["Rfar"], "type": "diffusion"},
        {"interaction": "N->Rnear", "ligand": ["N"], "receptor": ["Rnear"], "type": "diffusion"},
    ]
    res = ligrec.compute_ligrec(
        X, var_names, coords, pairs, radius=4.0, n_perm=200, p_thresh=0.05, seed=1
    )
    by = {s["interaction"]: s for s in res["summary"]}
    # The real, co-localized ligand->receptor pair fires on many cells...
    assert by["L->Rnear"]["n_signif"] > 0
    # ...far more than an anti-colocalized receptor or a noise ligand.
    assert by["L->Rnear"]["n_signif"] > by["L->Rfar"]["n_signif"]
    assert by["L->Rnear"]["n_signif"] > by["N->Rnear"]["n_signif"]


def test_compute_ligrec_signal_localized_near_source():
    X, var_names, coords = _signaling_field()
    pairs = [{"interaction": "L->Rnear", "ligand": ["L"], "receptor": ["Rnear"], "type": "diffusion"}]
    res = ligrec.compute_ligrec(
        X, var_names, coords, pairs, radius=4.0, n_perm=200, p_thresh=0.05, seed=1
    )
    sig = res["significant"][:, 0]
    assert sig.any()
    center = np.array([20.0, 20.0])
    mean_dist = np.linalg.norm(coords[sig] - center, axis=1).mean()
    # significant cells cluster within the diffusion reach of the source
    assert mean_dist <= 5.0


def test_compute_ligrec_summary_sorted_and_shaped():
    X, var_names, coords = _signaling_field()
    pairs = [
        {"interaction": "L->Rnear", "ligand": ["L"], "receptor": ["Rnear"], "type": "diffusion"},
        {"interaction": "L->Rfar", "ligand": ["L"], "receptor": ["Rfar"], "type": "diffusion"},
    ]
    res = ligrec.compute_ligrec(X, var_names, coords, pairs, radius=4.0, n_perm=100, seed=2)
    assert res["scores"].shape == (X.shape[0], 2)
    assert res["pvalues"].shape == (X.shape[0], 2)
    # summary sorted by n_signif descending
    ns = [s["n_signif"] for s in res["summary"]]
    assert ns == sorted(ns, reverse=True)
    for s in res["summary"]:
        assert {"interaction", "type", "ligand", "receptor", "n_signif", "frac_signif", "mean_score"} <= set(s)


def test_compute_ligrec_contact_pairs_use_delaunay():
    X, var_names, coords = _signaling_field()
    pairs = [{"interaction": "L->Rnear", "ligand": ["L"], "receptor": ["Rnear"], "type": "contact"}]
    res = ligrec.compute_ligrec(X, var_names, coords, pairs, radius=4.0, n_perm=100, seed=3)
    # contact mode still computes a finite score field
    assert np.isfinite(res["scores"]).all()
    assert res["scores"].shape == (X.shape[0], 1)


# --- within-section permutation -------------------------------------------

def test_section_permutation_stays_within_section():
    sections = np.array([0, 0, 0, 1, 1, 1, 1])
    rng = np.random.default_rng(0)
    perm = ligrec.section_permutation(len(sections), sections, rng)
    # a valid permutation
    assert sorted(perm.tolist()) == list(range(len(sections)))
    # every cell maps to a cell in the same section
    assert (sections[perm] == sections).all()


def test_section_permutation_none_is_global():
    rng = np.random.default_rng(0)
    perm = ligrec.section_permutation(5, None, rng)
    assert sorted(perm.tolist()) == [0, 1, 2, 3, 4]


def test_compute_ligrec_section_aware_no_cross_section_signal():
    # Two well-separated sections. Ligand source + matching receptor live in
    # section A; a decoy receptor sits in section B (no ligand reaches across the
    # gap). With section awareness, only the within-section pair should fire.
    rng = np.random.default_rng(0)
    # Section A: the proven 40x40 field (sparse source -> decisive signal).
    wA = 40
    xs, ys = np.meshgrid(np.arange(wA), np.arange(wA))
    secA = np.column_stack([xs.ravel(), ys.ravel()]).astype(float)
    # Section B: a small grid placed far away to host the decoy receptor.
    wB = 10
    xb, yb = np.meshgrid(np.arange(wB), np.arange(wB))
    secB = np.column_stack([xb.ravel(), yb.ravel()]).astype(float) + np.array([300.0, 0.0])
    coords = np.vstack([secA, secB])
    nA, nB = secA.shape[0], secB.shape[0]
    sections = np.array(["A"] * nA + ["B"] * nB)

    cA = np.array([20.0, 20.0])
    dA = np.linalg.norm(secA - cA, axis=1)
    # Section A: a dense ligand source + a broadly-expressed receptor (broad
    # receptors + housekeeping keep library sizes stable, like real data).
    L = np.concatenate([np.where(dA <= 2.0, 20.0, 0.0), np.zeros(nB)])
    Rin = np.concatenate([12.0 + rng.poisson(2.0, nA), np.zeros(nB)])
    # Decoy receptor only in section B, where the ligand never reaches.
    Rdecoy = np.concatenate([np.zeros(nA), 12.0 + rng.poisson(2.0, nB)])
    HK = 20.0 + rng.poisson(5.0, nA + nB)  # housekeeping -> stable lib size
    X = np.column_stack([L, Rin, Rdecoy, HK])
    var_names = ["L", "Rin", "Rdecoy", "HK"]
    pairs = [
        {"interaction": "L->Rin", "ligand": ["L"], "receptor": ["Rin"], "type": "diffusion"},
        {"interaction": "L->Rdecoy", "ligand": ["L"], "receptor": ["Rdecoy"], "type": "diffusion"},
    ]
    res = ligrec.compute_ligrec(
        X, var_names, coords, pairs, radius=4.0, n_perm=150, p_thresh=0.05,
        sections=sections, seed=1,
    )
    by = {s["interaction"]: s for s in res["summary"]}
    assert by["L->Rin"]["n_signif"] > 0
    assert by["L->Rdecoy"]["n_signif"] == 0
