"""Merging ST spots into single large cells: the pure algorithm.

A spot is a patch of tissue, not a cell. These tests pin down the grouping —
that it never exceeds one cell's footprint, that it refuses to cross a planted
boundary, and above all that it is deterministic, which is the property that
ruled out letting noisy low-count correlations rank the merges.
"""

import numpy as np
import pytest

from xcell.spot_merge import estimate_pitch_um, max_spots_for


def _grid(nx: int, ny: int, pitch: float = 15.0) -> np.ndarray:
    """A regular lattice, the shape a real ST slide has."""
    xs, ys = np.meshgrid(np.arange(nx) * pitch, np.arange(ny) * pitch)
    return np.column_stack([xs.ravel(), ys.ravel()]).astype(float)


def test_pitch_is_the_lattice_spacing():
    assert estimate_pitch_um(_grid(6, 6, pitch=15.0)) == pytest.approx(15.0)


def test_pitch_survives_jitter_and_a_few_gaps():
    rng = np.random.default_rng(0)
    coords = _grid(8, 8, pitch=20.0)
    coords += rng.normal(0, 0.5, coords.shape)       # imaging jitter
    coords = np.delete(coords, [3, 17, 40], axis=0)  # missing spots
    assert estimate_pitch_um(coords) == pytest.approx(20.0, abs=1.5)


def test_max_spots_counts_lattice_points_in_the_disc():
    # A 40 um disc over a 15 um lattice: pi*20^2 / 15^2 = 5.58 -> 6
    assert max_spots_for(40.0, 15.0) == 6
    # Diameter equal to pitch admits exactly one spot
    assert max_spots_for(15.0, 15.0) == 1


def test_max_spots_is_never_below_one():
    assert max_spots_for(1.0, 15.0) == 1


def test_pitch_needs_two_spots():
    with pytest.raises(ValueError, match="at least two"):
        estimate_pitch_um(np.array([[0.0, 0.0]]))


# --- the grouping itself -----------------------------------------------------

from xcell.spot_merge import MergeParams, merge_spots  # noqa: E402


def _profiles(n: int, g: int = 8, kind: int = 0, rng=None) -> np.ndarray:
    """Counts whose gene pattern depends on `kind`, so types differ."""
    rng = rng or np.random.default_rng(0)
    base = np.full((n, g), 1.0)
    base[:, kind * 2:(kind + 1) * 2] += 20.0     # a type-specific block
    return base + rng.poisson(0.5, (n, g))


def _norm(counts: np.ndarray) -> np.ndarray:
    """CPM + log1p, the matrix the veto sees."""
    depth = counts.sum(axis=1, keepdims=True)
    depth[depth == 0] = 1.0
    return np.log1p(counts / depth * 1e4)


def _run(coords, counts, **kw):
    return merge_spots(coords, counts, _norm(counts), MergeParams(**kw))


def test_groups_respect_the_diameter_cap():
    coords = _grid(6, 6, pitch=15.0)
    r = _run(coords, _profiles(36), max_diameter_um=40.0)
    for gid in range(r.n_groups):
        members = coords[r.labels == gid]
        if len(members) < 2:
            continue
        pairwise = np.linalg.norm(members[:, None, :] - members[None, :, :], axis=-1)
        assert pairwise.max() <= 40.0 + 1e-9


def test_every_spot_lands_in_exactly_one_group():
    coords = _grid(5, 5, pitch=15.0)
    r = _run(coords, _profiles(25), max_diameter_um=40.0)
    assert (r.labels >= 0).all()
    assert sorted(set(r.labels.tolist())) == list(range(r.n_groups))


def test_merging_actually_reduces_the_spot_count():
    r = _run(_grid(6, 6, pitch=15.0), _profiles(36), max_diameter_um=40.0)
    assert r.n_groups < 36


def _two_types():
    """Two cell types meeting at x=30.

    The boundary sits *inside* what the 40 um cap would otherwise merge into
    one group, so only the veto can keep the types apart. Putting it on a cap
    boundary instead would let the geometry pass the test on its own.
    """
    coords = _grid(6, 4, pitch=15.0)
    left = coords[:, 0] < 30.0
    counts = np.zeros((len(coords), 8))
    counts[left] = _profiles(int(left.sum()), kind=0)
    counts[~left] = _profiles(int((~left).sum()), kind=1)
    return coords, counts, left


def test_the_veto_blocks_merging_across_a_planted_boundary():
    coords, counts, left = _two_types()
    r = _run(coords, counts, max_diameter_um=40.0, min_correlation=0.5)
    for gid in range(r.n_groups):
        members = r.labels == gid
        assert len(set(left[members].tolist())) == 1, "group spans the boundary"
    assert r.n_vetoed > 0


def test_a_permissive_veto_does_not_block_within_one_type():
    r = _run(_grid(4, 4, pitch=15.0), _profiles(16),
             max_diameter_um=40.0, min_correlation=0.15)
    assert r.n_vetoed == 0
    assert r.n_groups < 16


def test_max_spots_caps_group_size():
    r = _run(_grid(6, 6, pitch=15.0), _profiles(36),
             max_diameter_um=1000.0, max_spots=3)
    assert np.bincount(r.labels).max() <= 3


def test_the_same_input_gives_the_same_labels_every_time():
    coords, counts = _grid(6, 6, pitch=15.0), _profiles(36)
    a = _run(coords, counts, max_diameter_um=40.0)
    b = _run(coords, counts, max_diameter_um=40.0)
    np.testing.assert_array_equal(a.labels, b.labels)


def test_reports_what_the_veto_did():
    coords, counts, _ = _two_types()
    r = _run(coords, counts, max_diameter_um=40.0, min_correlation=0.5)
    q = r.correlation_quantiles
    assert set(q) == {"p10", "p50", "p90"}
    assert -1.0 <= q["p10"] <= q["p50"] <= q["p90"] <= 1.0


def test_a_single_spot_is_left_alone():
    r = _run(np.array([[0.0, 0.0]]), _profiles(1), max_diameter_um=40.0)
    assert r.n_groups == 1
    assert r.labels.tolist() == [0]


def test_collinear_spots_do_not_defeat_the_graph():
    # Delaunay raises on degenerate input; a row of spots is a real slide edge.
    coords = np.column_stack([np.arange(5) * 15.0, np.zeros(5)])
    r = _run(coords, _profiles(5), max_diameter_um=40.0)
    assert r.n_groups < 5


def test_size_histogram_accounts_for_every_spot():
    r = _run(_grid(5, 5, pitch=15.0), _profiles(25), max_diameter_um=40.0)
    assert sum(size * n for size, n in r.size_histogram.items()) == 25


# --- eligibility -------------------------------------------------------------

from xcell.spot_merge import eligible_mask  # noqa: E402


def test_all_is_the_default_and_takes_everything():
    assert eligible_mask(_profiles(9), MergeParams()).all()


def test_min_counts_takes_only_the_shallow_spots():
    counts = np.ones((4, 3))
    counts[0] *= 10.0   # depth 30
    counts[1] *= 1.0    # depth 3
    counts[2] *= 100.0  # depth 300
    counts[3] *= 2.0    # depth 6
    mask = eligible_mask(counts, MergeParams(eligibility="min_counts", min_counts=10))
    assert mask.tolist() == [False, True, False, True]


def test_quantile_takes_the_shallow_fraction():
    counts = np.ones((10, 2)) * np.arange(1, 11)[:, None]
    mask = eligible_mask(counts, MergeParams(eligibility="quantile", quantile=0.3))
    assert mask.sum() == 3
    assert mask[:3].all()


def test_an_unknown_eligibility_mode_is_refused():
    with pytest.raises(ValueError, match="eligibility"):
        eligible_mask(_profiles(4), MergeParams(eligibility="vibes"))


def test_min_counts_mode_needs_a_threshold():
    with pytest.raises(ValueError, match="min_counts"):
        eligible_mask(_profiles(4), MergeParams(eligibility="min_counts"))


def test_ineligible_spots_pass_through_unmerged():
    coords = _grid(4, 4, pitch=15.0)
    counts = _profiles(16)
    counts[5] *= 1000.0        # far too deep to be eligible
    r = _run(coords, counts, max_diameter_um=40.0,
             eligibility="min_counts", min_counts=1000.0)
    assert (r.labels == r.labels[5]).sum() == 1, "a deep spot was absorbed"
