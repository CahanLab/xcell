"""Tests for xcell.neighborhood — cell-type neighborhood composition + enrichment.

The planted geometry gives a known answer: types A and B alternate on one
checkerboard grid (strong attraction), while type C sits on a distant grid of
its own (homotypic enrichment, avoidance of A and B).
"""
import json

import numpy as np
import pytest

from xcell import neighborhood as nb


def _checkerboard(width: int = 10, offset: float = 0.0) -> np.ndarray:
    xs, ys = np.meshgrid(np.arange(width, dtype=float), np.arange(width, dtype=float))
    coords = np.column_stack([xs.ravel() + offset, ys.ravel()])
    return coords


def _planted():
    """A/B interleaved at origin; C on its own distant grid."""
    ab = _checkerboard(10)
    c = _checkerboard(10, offset=1000.0)
    coords = np.vstack([ab, c])
    parity = (ab[:, 0] + ab[:, 1]) % 2
    labels = np.array(["A" if p == 0 else "B" for p in parity] + ["C"] * len(c))
    return coords, labels


# ---------------------------------------------------------------------------
# graph construction
# ---------------------------------------------------------------------------
def test_knn_edges_exactly_k_per_cell_no_self():
    coords, _ = _planted()
    src, dst = nb.knn_edges(coords, n_neighs=6)
    assert len(src) == len(dst) == coords.shape[0] * 6
    assert not np.any(src == dst)
    counts = np.bincount(src, minlength=coords.shape[0])
    assert np.all(counts == 6)


def test_knn_edges_clamp_k_to_available_cells():
    coords = _checkerboard(2)  # 4 cells; k=10 impossible
    src, dst = nb.knn_edges(coords, n_neighs=10)
    counts = np.bincount(src, minlength=4)
    assert np.all(counts == 3)


def test_knn_edges_never_cross_sections():
    coords, _ = _planted()
    sections = np.array(["s1"] * 100 + ["s2"] * 100)
    src, dst = nb.knn_edges(coords, n_neighs=6, sections=sections)
    assert np.all(sections[src] == sections[dst])


def test_radius_edges_isolated_cell_has_none():
    coords = np.array([[0.0, 0.0], [1.0, 0.0], [500.0, 500.0]])
    src, dst = nb.radius_edges(coords, radius=2.0)
    assert set(src) == {0, 1}
    assert 2 not in set(src) and 2 not in set(dst)


# ---------------------------------------------------------------------------
# composition + enrichment
# ---------------------------------------------------------------------------
def test_composition_rows_sum_to_one():
    coords, labels = _planted()
    res = nb.compute_neighborhood(coords, labels, n_neighs=6, n_perms=50, seed=0)
    cc = res["cell_composition"]
    assert cc.shape == (200, 3)
    np.testing.assert_allclose(cc.sum(axis=1), 1.0, atol=1e-9)
    comp = np.array(res["composition"])
    np.testing.assert_allclose(comp.sum(axis=1), 1.0, atol=1e-9)


def test_planted_attraction_avoidance_and_homotypic():
    coords, labels = _planted()
    res = nb.compute_neighborhood(coords, labels, n_neighs=6, n_perms=200, seed=0)
    cats = res["categories"]
    assert cats == ["A", "B", "C"]
    z = np.array(res["zscores"])
    q = np.array(res["qvals"])
    a, b, c = 0, 1, 2
    assert z[a, b] > 2.0          # A and B interleaved -> attraction
    assert z[a, c] < -2.0         # A never near C -> avoidance
    assert z[c, c] > 2.0          # C only neighbors C -> homotypic enrichment
    assert q[a, b] < 0.05 and q[a, c] < 0.05 and q[c, c] < 0.05
    comp = np.array(res["composition"])
    assert comp[a, b] > 0.5       # checkerboard: most A neighbors are B
    assert comp[a, c] == 0.0
    assert comp[c, c] == 1.0
    lfc = np.array(res["log2fc"])
    assert lfc[a, b] > 0 and lfc[a, c] < 0


def test_categories_param_fixes_axis_order():
    coords, labels = _planted()
    res = nb.compute_neighborhood(
        coords, labels, categories=["C", "B", "A"], n_neighs=6, n_perms=20, seed=0
    )
    assert res["categories"] == ["C", "B", "A"]
    comp = np.array(res["composition"])
    assert comp[0, 0] == 1.0      # C row now first, still all-C neighborhoods


def test_result_is_json_safe():
    coords, labels = _planted()
    res = nb.compute_neighborhood(coords, labels, n_neighs=6, n_perms=50, seed=0)
    payload = {k: v for k, v in res.items() if k != "cell_composition"}
    json.dumps(payload, allow_nan=False)  # raises on NaN/inf


def test_deterministic_for_same_seed():
    coords, labels = _planted()
    r1 = nb.compute_neighborhood(coords, labels, n_neighs=6, n_perms=100, seed=7)
    r2 = nb.compute_neighborhood(coords, labels, n_neighs=6, n_perms=100, seed=7)
    assert r1["zscores"] == r2["zscores"]
    assert r1["pvals"] == r2["pvals"]


def test_sections_confine_null_and_edges():
    # Two identical AB checkerboards far apart, as two sections: within-section
    # permutation must keep per-section label balance, and edges stay inside.
    ab1 = _checkerboard(10)
    ab2 = _checkerboard(10, offset=1000.0)
    coords = np.vstack([ab1, ab2])
    parity1 = (ab1[:, 0] + ab1[:, 1]) % 2
    labels = np.array(
        ["A" if p == 0 else "B" for p in parity1] * 2
    )
    sections = np.array(["s1"] * 100 + ["s2"] * 100)
    res = nb.compute_neighborhood(
        coords, labels, n_neighs=6, n_perms=100, seed=0, sections=sections
    )
    z = np.array(res["zscores"])
    assert z[0, 1] > 2.0          # attraction still detected within sections


def test_radius_mode_zero_neighbor_cells_get_nan_composition():
    coords = np.array([[0.0, 0.0], [1.0, 0.0], [0.5, 0.5], [500.0, 500.0]])
    labels = np.array(["A", "B", "A", "B"])
    res = nb.compute_neighborhood(
        coords, labels, mode="radius", radius=2.0, n_perms=20, seed=0
    )
    cc = res["cell_composition"]
    assert np.all(np.isnan(cc[3]))
    np.testing.assert_allclose(cc[:3].sum(axis=1), 1.0)
    # ...but the JSON side must stay finite.
    payload = {k: v for k, v in res.items() if k != "cell_composition"}
    json.dumps(payload, allow_nan=False)


def test_progress_callback_monotone_and_completes():
    coords, labels = _planted()
    seen: list[float] = []

    def report(frac, message=None):
        seen.append(frac)

    nb.compute_neighborhood(
        coords, labels, n_neighs=6, n_perms=100, seed=0, progress_callback=report
    )
    assert seen, "progress_callback was never called"
    assert all(b >= a for a, b in zip(seen, seen[1:]))
    assert seen[-1] == 1.0


def test_validation_errors():
    coords, labels = _planted()
    with pytest.raises(ValueError):
        nb.compute_neighborhood(coords, np.array(["A"] * 200), n_perms=10)  # 1 category
    with pytest.raises(ValueError):
        nb.compute_neighborhood(coords, labels, mode="radius", radius=None, n_perms=10)
    with pytest.raises(ValueError):
        nb.compute_neighborhood(coords, labels, mode="voronoi", n_perms=10)
    with pytest.raises(ValueError):
        nb.compute_neighborhood(coords, labels, n_neighs=0, n_perms=10)
    with pytest.raises(ValueError):
        # labels outside the explicit category list
        nb.compute_neighborhood(coords, labels, categories=["A", "B"], n_perms=10)
