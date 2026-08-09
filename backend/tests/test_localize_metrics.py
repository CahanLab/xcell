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


# --- spatial pattern fidelity ---------------------------------------------

from xcell.localize_metrics import spatial_pattern_fidelity  # noqa: E402


def _annulus_scores(coords, inner=0.75):
    """High on the rim, low inside — the epidermis pattern."""
    r = np.linalg.norm(coords, axis=1)
    return (r >= inner).astype(float)


def test_pattern_fidelity_is_high_when_the_pattern_is_reproduced():
    ref, pred = _disc(seed=0), _disc(seed=1)
    out = spatial_pattern_fidelity(ref, _annulus_scores(ref), pred, _annulus_scores(pred))
    assert out['correlation'] > 0.7


def test_pattern_fidelity_is_negative_when_the_pattern_is_inverted():
    """The limb failure exactly: epidermis belongs on the rim and was predicted
    into the centre. This must come out clearly negative, not merely low."""
    ref, pred = _disc(seed=0), _disc(seed=1)
    out = spatial_pattern_fidelity(
        ref, _annulus_scores(ref), pred, 1.0 - _annulus_scores(pred),
    )
    assert out['correlation'] < -0.3


def test_pattern_fidelity_is_near_zero_for_an_unrelated_pattern():
    ref, pred = _disc(seed=0), _disc(seed=1)
    rng = np.random.default_rng(3)
    out = spatial_pattern_fidelity(ref, _annulus_scores(ref), pred, rng.random(len(pred)))
    assert abs(out['correlation']) < 0.3


def test_pattern_fidelity_ignores_the_scale_of_the_scores():
    """ST spots and dissociated cells are on different expression scales, so a
    metric sensitive to that would compare the platforms, not the biology."""
    ref, pred = _disc(seed=0), _disc(seed=1)
    a = spatial_pattern_fidelity(ref, _annulus_scores(ref), pred, _annulus_scores(pred))
    b = spatial_pattern_fidelity(
        ref, _annulus_scores(ref), pred, 100.0 * _annulus_scores(pred) + 7.0,
    )
    assert a['correlation'] == pytest.approx(b['correlation'], abs=1e-9)


def test_pattern_fidelity_handles_an_empty_overlap():
    import json
    ref = _disc(seed=0)
    far = _disc(seed=1) + 1000.0            # nowhere near the reference
    out = spatial_pattern_fidelity(ref, _annulus_scores(ref), far, _annulus_scores(far))
    json.dumps(out)
    assert out['correlation'] is None or np.isfinite(out['correlation'])


# --- axis fidelity ---------------------------------------------------------

from xcell.localize_metrics import axis_direction, axis_fidelity  # noqa: E402


def test_axis_direction_finds_the_gradient():
    """A score that increases with x must yield a direction along +x, whatever
    the coordinate units are."""
    coords = _disc(seed=0)
    u = axis_direction(coords, coords[:, 0])
    assert abs(u[0]) == pytest.approx(1.0, abs=0.05)
    assert u[0] > 0
    assert abs(u[1]) < 0.2


def test_axis_direction_finds_a_diagonal_gradient():
    coords = _disc(seed=0)
    u = axis_direction(coords, coords[:, 0] + coords[:, 1])
    assert u[0] == pytest.approx(0.707, abs=0.1)
    assert u[1] == pytest.approx(0.707, abs=0.1)


def test_axis_fidelity_reports_the_reference_as_the_ceiling():
    """The prediction is never reported alone: -0.17 means nothing until you
    know the reference itself only reaches -0.33."""
    ref, pred = _disc(seed=0), _disc(seed=1)
    out = axis_fidelity(ref, ref[:, 0], pred, pred[:, 0])
    assert out['reference'] > 0.9
    assert out['prediction'] > 0.9


def test_axis_fidelity_falls_when_the_gradient_is_lost():
    """best_match kept full tissue area and drove every axis correlation to
    roughly zero; that has to be visible."""
    ref, pred = _disc(seed=0), _disc(seed=1)
    rng = np.random.default_rng(5)
    out = axis_fidelity(ref, ref[:, 0], pred, rng.random(len(pred)))
    assert out['reference'] > 0.9
    assert abs(out['prediction']) < 0.2


def test_axis_fidelity_uses_the_references_direction_for_both():
    """Deriving the direction from the reference is what makes this
    orientation-free — the user never declares which way is distal."""
    ref, pred = _disc(seed=0), _disc(seed=1)
    out = axis_fidelity(ref, ref[:, 1], pred, pred[:, 1])
    assert abs(out['direction'][1]) > 0.9      # the y axis, discovered
    assert out['prediction'] > 0.9


def test_axis_fidelity_survives_a_constant_score():
    import json
    ref, pred = _disc(seed=0), _disc(seed=1)
    out = axis_fidelity(ref, np.ones(len(ref)), pred, np.ones(len(pred)))
    json.dumps(out)
