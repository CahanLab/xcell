"""Measuring whether a predicted spatial map is any good.

The metrics exist because 'the map looks wrong' was not actionable: eight saved
parameter variants existed on one dataset with no way to rank them. Each test
here builds geometry whose right answer is known by construction.
"""

import numpy as np
import pytest

from xcell.localize_metrics import dispersion, occupancy


def _disc(n=2000, r=1.0, seed=0):
    """A filled disc of radius r, centred at the origin."""
    rng = np.random.default_rng(seed)
    rad = r * np.sqrt(rng.random(n))
    th = rng.random(n) * 2 * np.pi
    return np.column_stack([rad * np.cos(th), rad * np.sin(th)])


def test_dispersion_is_one_when_the_prediction_matches_the_reference():
    ref = _disc(seed=0)
    out = dispersion(ref, _disc(seed=1))
    assert out['area_ratio'] == pytest.approx(1.0, abs=0.15)
    assert out['std_ratio_x'] == pytest.approx(1.0, abs=0.15)
    assert out['frac_outside'] == pytest.approx(0.0, abs=0.05)


def test_dispersion_detects_a_collapsed_map():
    """The failure this exists to catch: weighted_mean shrank the limb map to
    0.15 of the tissue area while every axis correlation stayed correct."""
    ref = _disc(seed=0)
    out = dispersion(ref, _disc(seed=1) * 0.5)      # half the radius
    assert out['area_ratio'] == pytest.approx(0.25, abs=0.06)   # area goes as r^2
    assert out['std_ratio_x'] == pytest.approx(0.5, abs=0.06)


def test_dispersion_detects_an_overshoot_and_counts_cells_outside():
    ref = _disc(seed=0)
    out = dispersion(ref, _disc(seed=1) * 2.0)
    assert out['area_ratio'] > 3.0
    # Rescaling to match variance pushed 14.8% of real cells out of the tissue;
    # that has to be visible rather than hidden inside an area ratio.
    assert out['frac_outside'] > 0.5


def test_dispersion_ignores_a_handful_of_outliers():
    """inner=0.95 trims the hull, so one stray point cannot dominate."""
    ref = _disc(seed=0)
    pred = np.vstack([_disc(seed=1), np.array([[50.0, 50.0]])])
    assert dispersion(ref, pred)['area_ratio'] == pytest.approx(1.0, abs=0.2)


def test_occupancy_is_clean_when_predictions_spread():
    ref = _disc(n=500, seed=0)
    out = occupancy(ref, _disc(n=400, seed=1))
    assert out['n_pred'] == 400
    assert out['n_distinct'] > 250          # most cells find their own spot
    # Random points nearest-neighbouring onto random spots clump a little by
    # chance — the bound only has to separate this from real pile-up, which is
    # 40 on one spot in the test below.
    assert out['max_per_spot'] <= 12


def test_occupancy_detects_pile_up():
    """greedy best_match put 2,683 cells on 865 spots, 15 on a single one."""
    ref = _disc(n=500, seed=0)
    pred = np.repeat(ref[:10], 40, axis=0)   # 400 cells crammed onto 10 spots
    out = occupancy(ref, pred)
    assert out['n_distinct'] == 10
    assert out['max_per_spot'] == 40
    assert out['effective_n'] == pytest.approx(10.0, abs=0.5)


def test_occupancy_effective_n_penalises_unevenness():
    """Two maps can use the same spots and still differ: effective_n is the
    exponential of the assignment entropy, so it drops when a few spots absorb
    most cells even though n_distinct does not."""
    ref = _disc(n=500, seed=0)
    even = np.repeat(ref[:20], 10, axis=0)                       # 10 each
    skewed = np.vstack([np.repeat(ref[:1], 181, axis=0),
                        np.repeat(ref[1:20], 1, axis=0)])        # 181 then 1s
    assert occupancy(ref, even)['n_distinct'] == 20
    assert occupancy(ref, skewed)['n_distinct'] == 20
    assert occupancy(ref, skewed)['effective_n'] < occupancy(ref, even)['effective_n']


def test_metrics_are_json_safe_on_degenerate_input():
    import json
    ref = _disc(n=10, seed=0)
    single = np.tile(ref[:1], (5, 1))        # no spread at all -> no hull
    json.dumps(dispersion(ref, single))      # raises on NaN/inf
    json.dumps(occupancy(ref, single))
