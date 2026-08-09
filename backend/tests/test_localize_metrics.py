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


# --- the whole panel -------------------------------------------------------

from xcell.localize_metrics import evaluate_map  # noqa: E402


def test_evaluate_map_bundles_every_metric_per_marker_set():
    ref, pred = _disc(seed=0), _disc(seed=1)
    out = evaluate_map(ref, pred, [
        {'name': 'skin', 'ref_scores': _annulus_scores(ref),
         'pred_scores': _annulus_scores(pred)},
        {'name': 'distal', 'ref_scores': ref[:, 0], 'pred_scores': pred[:, 0]},
    ])
    assert set(out) == {'dispersion', 'occupancy', 'markers'}
    assert [m['name'] for m in out['markers']] == ['skin', 'distal']
    assert out['markers'][0]['pattern']['correlation'] > 0.7
    assert out['markers'][1]['axis']['prediction'] > 0.9
    assert out['dispersion']['area_ratio'] == pytest.approx(1.0, abs=0.15)


def test_evaluate_map_is_json_safe_with_no_marker_sets():
    import json
    ref, pred = _disc(seed=0), _disc(seed=1)
    json.dumps(evaluate_map(ref, pred, []))


# --- would a mean estimator lose this population? -------------------------

from xcell.localize_metrics import mean_collapse_risk  # noqa: E402


def test_collapse_risk_flags_an_annular_population():
    """The limb failure, reduced to geometry. The epidermis is a ring; the mean
    of a ring is its hole; weighted_mean predicts every epidermal cell into it."""
    ref = _disc(seed=0)
    out = mean_collapse_risk(ref, np.linalg.norm(ref, axis=1))   # high on the rim
    assert out['risk'] > 0.8
    # The hole is eight ring-spacings across, which is the number the UI quotes.
    assert out['distance_ratio'] > 4.0


def test_collapse_risk_flags_a_population_in_two_distant_patches():
    """The other shape weighted_mean cannot represent: the mean lands in the
    gap between the patches, which is a location the population never occupies.

    A gap and a hole are the same failure to this metric, and score alike (~0.88
    each): in both, the mean sits many population-spacings away from the nearest
    member.
    """
    ref = _disc(seed=0)
    out = mean_collapse_risk(ref, np.abs(ref[:, 0]))             # both left and right
    assert out['risk'] > 0.8


def test_collapse_risk_flags_a_one_cell_wide_perimeter():
    """The epidermis's actual shape, and the case that pins ``neighbors``.

    A thin population has only its two neighbours along the band, so averaging
    over many of them walks around the ring and reports a "spacing" far larger
    than the real one — which hides the hole rather than measuring it. At the
    shipped ``neighbors=5`` this scores ~0.79; at 25 it falls to ~0.38 and the
    warning would never fire on the shape it exists for.
    """
    grid = np.array([[float(i), float(j)] for i in range(20) for j in range(20)])
    on_edge = ((grid[:, 0] == 0) | (grid[:, 0] == 19)
               | (grid[:, 1] == 0) | (grid[:, 1] == 19))
    assert mean_collapse_risk(grid, on_edge.astype(float))['risk'] > 0.7


def test_collapse_risk_clears_a_compact_population():
    ref = _disc(seed=0)
    blob = -np.linalg.norm(ref - np.array([0.4, 0.3]), axis=1)
    out = mean_collapse_risk(ref, blob)
    assert out['risk'] < 0.2
    # The mean is an ordinary place inside the blob, so it sits about as far
    # from its members as they sit from each other.
    assert out['distance_ratio'] == pytest.approx(1.0, abs=0.4)


def test_collapse_risk_does_not_cry_wolf_on_a_scattered_population():
    """A type spread through the tissue is not what this catches — the mean
    lands in the middle, which is as good a guess as any. Advice that fires on
    an ordinary population is noise, and would train the user to ignore it.

    Swept across seeds rather than fixed at one, because a single draw is how a
    rare false positive hides: an earlier formulation counted how many of the
    local neighbours were population *members*, which at 8 expected out of 40
    strayed past the warning threshold on about 1 seed in 60. The shipped
    version peaks at 0.06 here, so the margin is the point of the sweep.

    Seed 0 is excluded, and the reason is a trap worth naming: ``_disc(seed=0)``
    spends ``rng.random(n)`` on its radii first, so ``default_rng(0).random(n)``
    reproduces that same draw and the "random" score is exactly the squared
    radius — a perfect annulus, which this metric should and does flag.
    """
    ref = _disc(seed=0)
    for seed in range(1, 25):
        scores = np.random.default_rng(seed).random(len(ref))
        assert mean_collapse_risk(ref, scores)['risk'] < 0.5, seed


def test_collapse_risk_reports_where_the_mean_would_land():
    """The centroid is returned because the warning is about a *place*, and a
    user who wants to check it needs the coordinate."""
    ref = _disc(seed=0)
    out = mean_collapse_risk(ref, np.linalg.norm(ref, axis=1))
    assert out['centroid'] == pytest.approx([0.0, 0.0], abs=0.1)
    assert out['n_population'] == pytest.approx(0.2 * len(ref), rel=0.02)


def test_collapse_risk_is_json_safe_on_degenerate_input():
    import json
    ref = _disc(n=200, seed=0)
    json.dumps(mean_collapse_risk(ref, np.ones(len(ref))))       # no variation
    json.dumps(mean_collapse_risk(_disc(n=5, seed=0), np.arange(5.0)))
    assert mean_collapse_risk(ref, np.ones(len(ref)))['risk'] is None
