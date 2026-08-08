"""kNN projection of dissociated cells onto a spatial reference.

The fixtures here are built to break the method, not to flatter it: a
uniformly-dispersed population whose neighbours scatter, a bimodal population
present in two distant patches, and a query population with no counterpart in
the reference. Those three are where a coordinate prediction looks perfectly
convincing and is wrong, so they are what the confidence scores exist for and
what these tests hold them to.
"""
import json
import math

import numpy as np
import pytest

from xcell.localize import (
    AGGREGATIONS,
    METRICS,
    TRANSFORMS,
    cross_validate_localization,
    evaluate_localization,
    project_knn,
    transform_expression,
)


# --- fixtures -------------------------------------------------------------

def _gradient_genes(coords, n_genes=30, seed=0, depth=1.0):
    """Expression as smooth spatial gradients — each gene a random plane over
    the tissue, so position is recoverable from expression."""
    rng = np.random.default_rng(seed)
    xy = np.column_stack([coords, np.ones(len(coords))])
    # Normalize coords so gene scales don't depend on tissue units.
    xy[:, 0] = (xy[:, 0] - xy[:, 0].mean()) / (xy[:, 0].std() or 1)
    xy[:, 1] = (xy[:, 1] - xy[:, 1].mean()) / (xy[:, 1].std() or 1)
    W = rng.normal(0, 1, (3, n_genes))
    expr = xy @ W
    expr = np.maximum(expr + 3.0, 0.0) * depth
    expr = expr + rng.normal(0, 0.15, expr.shape)
    return np.maximum(expr, 0.0).astype(np.float32)


def _tissue(n=400, seed=0, span=100.0):
    rng = np.random.default_rng(seed)
    coords = rng.uniform(0, span, (n, 2))
    return coords, _gradient_genes(coords, seed=seed)


def _matched(n_ref=400, n_query=120, seed=0):
    """A reference and a query drawn from the same generative tissue, so the
    query's true positions are known."""
    ref_coords, ref_expr = _tissue(n_ref, seed=seed)
    rng = np.random.default_rng(seed + 100)
    q_coords = rng.uniform(0, 100.0, (n_query, 2))
    # Same planes (same seed) so the two "platforms" describe the same biology.
    q_expr = _gradient_genes(q_coords, seed=seed)
    return ref_expr, ref_coords, q_expr, q_coords


def _median_error(pred, truth):
    ok = ~np.isnan(pred[:, 0])
    return float(np.median(np.linalg.norm(pred[ok] - truth[ok], axis=1)))


# --- transforms -----------------------------------------------------------

def test_every_named_transform_runs():
    X = np.abs(np.random.default_rng(0).normal(5, 2, (20, 8))).astype(np.float32)
    for name in TRANSFORMS:
        out = transform_expression(X, name)
        assert out.shape == X.shape
        assert np.isfinite(out).all(), name


def test_zscore_standardizes_each_gene():
    X = np.array([[1.0, 10.0], [3.0, 30.0], [5.0, 50.0]], dtype=np.float32)
    out = transform_expression(X, 'zscore')
    assert np.allclose(out.mean(axis=0), 0, atol=1e-5)
    assert np.allclose(out.std(axis=0), 1, atol=1e-5)


def test_zscore_survives_a_gene_with_no_variance():
    """A gene absent from one platform is constant there; it must not become
    NaN and poison every distance."""
    X = np.array([[1.0, 7.0], [2.0, 7.0], [3.0, 7.0]], dtype=np.float32)
    out = transform_expression(X, 'zscore')
    assert np.isfinite(out).all()
    assert np.allclose(out[:, 1], 0)


def test_rank_is_within_cell_and_monotone_invariant():
    X = np.array([[1.0, 5.0, 3.0]], dtype=np.float32)
    a = transform_expression(X, 'rank')
    b = transform_expression(np.exp(X), 'rank')  # any monotone rescaling
    assert np.allclose(a, b)


def test_an_unknown_transform_is_refused():
    with pytest.raises(ValueError):
        transform_expression(np.ones((2, 2), dtype=np.float32), 'wavelet')


# --- the easy case: does it work at all -----------------------------------

def test_a_recoverable_tissue_is_recovered():
    ref_expr, ref_coords, q_expr, q_coords = _matched()
    p = project_knn(q_expr, ref_expr, ref_coords, k=10)
    # Tissue spans 100 units; a working method lands well inside a tenth of it.
    assert _median_error(p.coords, q_coords) < 10.0


def test_it_beats_predicting_the_tissue_centroid():
    """The baseline that always looks good on a compact tissue. Failing to beat
    it means the method is doing nothing."""
    ref_expr, ref_coords, q_expr, q_coords = _matched()
    p = project_knn(q_expr, ref_expr, ref_coords, k=10)
    centroid_error = float(np.median(
        np.linalg.norm(q_coords - ref_coords.mean(axis=0), axis=1)
    ))
    assert _median_error(p.coords, q_coords) < centroid_error / 2


def test_every_query_cell_gets_a_prediction_and_two_scores():
    ref_expr, ref_coords, q_expr, _ = _matched(n_query=37)
    p = project_knn(q_expr, ref_expr, ref_coords, k=8)
    assert p.coords.shape == (37, 2)
    assert p.confidence.shape == (37,)
    assert p.similarity.shape == (37,)
    assert p.neighbor_indices.shape == (37, 8)


def test_scores_stay_in_their_declared_range():
    ref_expr, ref_coords, q_expr, _ = _matched()
    p = project_knn(q_expr, ref_expr, ref_coords, k=10)
    assert np.all((p.confidence >= 0) & (p.confidence <= 1))
    assert np.isfinite(p.similarity).all()


def test_k_larger_than_the_reference_is_clamped():
    ref_expr, ref_coords, q_expr, _ = _matched(n_ref=12, n_query=5)
    p = project_knn(q_expr, ref_expr, ref_coords, k=500)
    assert p.neighbor_indices.shape[1] == 12


def test_every_metric_and_aggregation_produces_a_usable_map():
    ref_expr, ref_coords, q_expr, q_coords = _matched()
    centroid_error = float(np.median(
        np.linalg.norm(q_coords - ref_coords.mean(axis=0), axis=1)
    ))
    for metric in METRICS:
        for aggregation in AGGREGATIONS:
            p = project_knn(q_expr, ref_expr, ref_coords, k=10,
                            metric=metric, aggregation=aggregation)
            err = _median_error(p.coords, q_coords)
            assert err < centroid_error, f'{metric}/{aggregation} -> {err:.1f}'


# --- the dispersed population: confidence must catch it -------------------

N_LOC, N_DISP = 100, 60


def _with_dispersed_population(n_ref=400, seed=0):
    """A population with one flat expression profile scattered over the whole
    tissue. Its neighbours cannot agree on a location, and a centroid of them
    is meaningless.

    The query is a realistic *mixture* of both populations, not a handful of
    near-identical cells: a per-gene z-score is only meaningful across cells
    that actually vary, and every real query is heterogeneous.

    Returns (ref_expr, ref_coords, query_expr, query_truth, groups) with the
    localized cells first (``N_LOC``) and the dispersed ones after (``N_DISP``).
    """
    ref_coords, ref_expr = _tissue(n_ref, seed=seed)
    rng = np.random.default_rng(seed + 7)
    n_genes = ref_expr.shape[1]
    flat = np.full(n_genes, 4.0, dtype=np.float32)

    def _flat_cells(n):
        return (np.tile(flat, (n, 1))
                + rng.normal(0, 0.05, (n, n_genes))).astype(np.float32)

    disp_coords = rng.uniform(0, 100.0, (120, 2))
    full_expr = np.vstack([ref_expr, _flat_cells(120)])
    full_coords = np.vstack([ref_coords, disp_coords])

    q_loc_coords = rng.uniform(0, 100.0, (N_LOC, 2))
    q_disp_coords = rng.uniform(0, 100.0, (N_DISP, 2))
    query = np.vstack([_gradient_genes(q_loc_coords, seed=seed), _flat_cells(N_DISP)])
    truth = np.vstack([q_loc_coords, q_disp_coords])
    groups = np.array(['localized'] * N_LOC + ['dispersed'] * N_DISP)
    return full_expr, full_coords, query, truth, groups


def test_a_dispersed_population_scores_low_confidence():
    ref_expr, ref_coords, query, _, _ = _with_dispersed_population()
    p = project_knn(query, ref_expr, ref_coords, k=15)
    assert p.confidence[N_LOC:].mean() < 0.35, p.confidence[N_LOC:].mean()


def test_a_localized_population_scores_high_confidence():
    ref_expr, ref_coords, query, _, _ = _with_dispersed_population()
    p = project_knn(query, ref_expr, ref_coords, k=15)
    assert p.confidence[:N_LOC].mean() > 0.6, p.confidence[:N_LOC].mean()


def test_confidence_separates_the_two_populations():
    """The whole point: the score has to rank the untrustworthy predictions
    below the trustworthy ones on the same run."""
    ref_expr, ref_coords, query, _, _ = _with_dispersed_population()
    p = project_knn(query, ref_expr, ref_coords, k=15)
    assert p.confidence[N_LOC:].max() < p.confidence[:N_LOC].min()


def test_a_dispersed_population_really_is_placed_badly():
    """The premise behind the confidence score, stated as a fact about error:
    it is not that these cells are merely uncertain, it is that the coordinate
    they get is wrong."""
    ref_expr, ref_coords, query, truth, _ = _with_dispersed_population()
    p = project_knn(query, ref_expr, ref_coords, k=15)
    localized = _median_error(p.coords[:N_LOC], truth[:N_LOC])
    dispersed = _median_error(p.coords[N_LOC:], truth[N_LOC:])
    assert dispersed > 3 * localized


def test_low_confidence_cells_can_be_left_unplaced():
    ref_expr, ref_coords, query, _, _ = _with_dispersed_population()
    p = project_knn(query, ref_expr, ref_coords, k=15, min_confidence=0.5)
    assert np.isnan(p.coords[N_LOC:]).all()
    assert not np.isnan(p.coords[:N_LOC]).any()
    assert p.n_unplaced == N_DISP


def test_nothing_is_dropped_by_default():
    """Silently discarding cells is its own kind of lie."""
    ref_expr, ref_coords, query, _, _ = _with_dispersed_population()
    p = project_knn(query, ref_expr, ref_coords, k=15)
    assert p.n_unplaced == 0
    assert np.isfinite(p.coords).all()


# --- the bimodal population: why `densest` exists -------------------------

N_BG, N_PATCH = 120, 40


def _with_bimodal_population(seed=0):
    """One population in two distant patches sharing an identical profile. The
    mean of its neighbours lands in the empty gap between them.

    As above, the query is a realistic mixture; the patch cells are the last
    ``N_PATCH`` rows.
    """
    ref_coords, ref_expr = _tissue(400, seed=seed)
    rng = np.random.default_rng(seed + 11)
    n_genes = ref_expr.shape[1]
    n_each = 60
    a = rng.normal([10, 10], 3.0, (n_each, 2))
    b = rng.normal([90, 90], 3.0, (n_each, 2))

    profile = np.zeros(n_genes, dtype=np.float32)
    profile[0] = 20.0  # a distinctive marker, flat across both patches

    def _patch_cells(n):
        return (np.tile(profile, (n, 1))
                + rng.normal(0, 0.05, (n, n_genes))).astype(np.float32)

    full_expr = np.vstack([ref_expr, _patch_cells(2 * n_each)])
    full_coords = np.vstack([ref_coords, a, b])

    bg_coords = rng.uniform(0, 100.0, (N_BG, 2))
    query = np.vstack([_gradient_genes(bg_coords, seed=seed), _patch_cells(N_PATCH)])
    return full_expr, full_coords, query, np.array([[10.0, 10.0], [90.0, 90.0]])


def _distance_to_nearest_patch(coords, patches):
    return np.min(
        np.linalg.norm(coords[:, None, :] - patches[None, :, :], axis=2), axis=1
    )


def test_the_mean_of_a_bimodal_neighbourhood_lands_between_the_patches():
    """Documenting the artifact, so the fix below has something to beat."""
    ref_expr, ref_coords, query, patches = _with_bimodal_population()
    p = project_knn(query, ref_expr, ref_coords, k=30, aggregation='weighted_mean')
    to_nearest = _distance_to_nearest_patch(p.coords[N_BG:], patches)
    assert np.median(to_nearest) > 20.0


def test_densest_picks_one_patch_instead_of_the_gap():
    ref_expr, ref_coords, query, patches = _with_bimodal_population()
    mean = project_knn(query, ref_expr, ref_coords, k=30, aggregation='weighted_mean')
    dense = project_knn(query, ref_expr, ref_coords, k=30, aggregation='densest')
    assert (np.median(_distance_to_nearest_patch(dense.coords[N_BG:], patches))
            < np.median(_distance_to_nearest_patch(mean.coords[N_BG:], patches)) / 2)


def test_densest_lands_the_bimodal_cells_in_an_actual_patch():
    ref_expr, ref_coords, query, patches = _with_bimodal_population()
    p = project_knn(query, ref_expr, ref_coords, k=30, aggregation='densest')
    assert np.median(_distance_to_nearest_patch(p.coords[N_BG:], patches)) < 12.0


def test_densest_does_not_hurt_the_unimodal_cells():
    """It has to be safe to leave on — a fix for one population that damages
    the rest is not a fix."""
    ref_expr, ref_coords, q_expr, q_coords = _matched()
    mean = project_knn(q_expr, ref_expr, ref_coords, k=15, aggregation='weighted_mean')
    dense = project_knn(q_expr, ref_expr, ref_coords, k=15, aggregation='densest')
    assert (_median_error(dense.coords, q_coords)
            < _median_error(mean.coords, q_coords) * 1.5)


def test_best_match_always_lands_on_a_real_reference_cell():
    ref_expr, ref_coords, query, _ = _with_bimodal_population()
    p = project_knn(query, ref_expr, ref_coords, k=30, aggregation='best_match')
    for row in p.coords:
        assert np.isclose(np.linalg.norm(ref_coords - row, axis=1).min(), 0)


# --- a query population with no counterpart -------------------------------

def test_a_query_type_absent_from_the_reference_scores_low_similarity():
    """It still gets a coordinate — placement and plausibility are separate
    claims — but the similarity says not to believe it."""
    ref_expr, ref_coords, q_expr, _ = _matched()
    rng = np.random.default_rng(3)
    alien = rng.normal(50, 5, (20, ref_expr.shape[1])).astype(np.float32)
    alien[:, ::2] = 0.0  # a profile unlike anything in the reference
    p = project_knn(np.vstack([q_expr[:20], alien]), ref_expr, ref_coords, k=10)
    assert p.similarity[20:].mean() < p.similarity[:20].mean()
    assert np.isfinite(p.coords).all()


# --- cross-platform scaling ----------------------------------------------

def test_zscore_absorbs_a_per_gene_platform_scale():
    """The reason zscore is the default: a platform that captures some genes
    more efficiently must not move every cell."""
    ref_expr, ref_coords, q_expr, q_coords = _matched()
    rng = np.random.default_rng(5)
    scale = rng.uniform(0.2, 5.0, ref_expr.shape[1]).astype(np.float32)
    skewed = (q_expr * scale).astype(np.float32)

    clean = _median_error(project_knn(q_expr, ref_expr, ref_coords, k=10,
                                      transform='zscore').coords, q_coords)
    scaled = _median_error(project_knn(skewed, ref_expr, ref_coords, k=10,
                                       transform='zscore').coords, q_coords)
    assert scaled < clean * 2.5


def test_without_a_transform_the_same_scaling_does_real_damage():
    ref_expr, ref_coords, q_expr, q_coords = _matched()
    rng = np.random.default_rng(5)
    scale = rng.uniform(0.2, 5.0, ref_expr.shape[1]).astype(np.float32)
    skewed = (q_expr * scale).astype(np.float32)
    raw = _median_error(project_knn(skewed, ref_expr, ref_coords, k=10,
                                    transform='none').coords, q_coords)
    zsc = _median_error(project_knn(skewed, ref_expr, ref_coords, k=10,
                                    transform='zscore').coords, q_coords)
    assert zsc < raw


# --- sections -------------------------------------------------------------

def _two_sections(seed=0):
    """The same tissue cut twice and laid out side by side. A coordinate
    averaged across the two lands in the gap."""
    coords, expr = _tissue(300, seed=seed)
    shifted = coords + np.array([500.0, 0.0])
    return (
        np.vstack([expr, expr]),
        np.vstack([coords, shifted]),
        np.array(['s1'] * len(coords) + ['s2'] * len(coords)),
        expr[:25].copy(),
    )


def test_a_prediction_never_averages_across_sections():
    ref_expr, ref_coords, sections, query = _two_sections()
    p = project_knn(query, ref_expr, ref_coords, k=20, ref_sections=sections)
    # The gap runs from x=100 to x=500; nothing may land inside it.
    assert not np.any((p.coords[:, 0] > 120) & (p.coords[:, 0] < 480))


def test_each_cell_is_told_which_section_it_was_assigned():
    ref_expr, ref_coords, sections, query = _two_sections()
    p = project_knn(query, ref_expr, ref_coords, k=20, ref_sections=sections)
    assert p.sections is not None
    assert set(p.sections) <= {'s1', 's2'}


def test_ignoring_sections_produces_the_artifact_they_prevent():
    """Without the section column the mean lands in empty space between cuts."""
    ref_expr, ref_coords, _, query = _two_sections()
    p = project_knn(query, ref_expr, ref_coords, k=20)
    assert np.any((p.coords[:, 0] > 120) & (p.coords[:, 0] < 480))


def test_no_sections_means_no_section_assignment():
    ref_expr, ref_coords, q_expr, _ = _matched()
    assert project_knn(q_expr, ref_expr, ref_coords, k=10).sections is None


# --- input validation -----------------------------------------------------

def test_mismatched_gene_counts_are_refused():
    ref_expr, ref_coords, q_expr, _ = _matched()
    with pytest.raises(ValueError, match='gene'):
        project_knn(q_expr[:, :5], ref_expr, ref_coords, k=5)


def test_coordinates_must_match_the_reference_cells():
    ref_expr, ref_coords, q_expr, _ = _matched()
    with pytest.raises(ValueError):
        project_knn(q_expr, ref_expr, ref_coords[:10], k=5)


def test_an_empty_reference_is_refused():
    with pytest.raises(ValueError):
        project_knn(np.ones((3, 4), dtype=np.float32),
                    np.zeros((0, 4), dtype=np.float32),
                    np.zeros((0, 2)), k=5)


def test_unknown_metric_and_aggregation_are_refused():
    ref_expr, ref_coords, q_expr, _ = _matched()
    with pytest.raises(ValueError):
        project_knn(q_expr, ref_expr, ref_coords, metric='mahalanobis')
    with pytest.raises(ValueError):
        project_knn(q_expr, ref_expr, ref_coords, aggregation='vote')


def test_progress_is_reported_when_asked():
    ref_expr, ref_coords, q_expr, _ = _matched(n_query=300)
    seen = []
    project_knn(q_expr, ref_expr, ref_coords, k=10,
                progress=lambda frac, msg: seen.append(frac))
    assert seen and seen[-1] == pytest.approx(1.0)
    assert all(0 <= f <= 1 for f in seen)


# --- evaluation -----------------------------------------------------------

def test_evaluation_reports_error_against_both_baselines():
    ref_expr, ref_coords, q_expr, q_coords = _matched()
    p = project_knn(q_expr, ref_expr, ref_coords, k=10)
    m = evaluate_localization(p.coords, q_coords, ref_coords=ref_coords)
    assert m['median_error'] < m['baseline_centroid']
    assert m['median_error'] < m['baseline_random']
    assert 0 <= m['median_error_normalized'] <= 1


def test_evaluation_reports_how_many_cells_land_close():
    ref_expr, ref_coords, q_expr, q_coords = _matched()
    p = project_knn(q_expr, ref_expr, ref_coords, k=10)
    m = evaluate_localization(p.coords, q_coords, ref_coords=ref_coords)
    within = m['fraction_within']
    assert set(within) == {'5%', '10%', '25%'}
    assert 0 <= within['5%'] <= within['10%'] <= within['25%'] <= 1


def test_evaluation_measures_preserved_structure():
    """Per-cell error and preserved global structure are different claims."""
    ref_expr, ref_coords, q_expr, q_coords = _matched()
    p = project_knn(q_expr, ref_expr, ref_coords, k=10)
    m = evaluate_localization(p.coords, q_coords, ref_coords=ref_coords)
    assert m['structure_correlation'] > 0.5


def test_evaluation_breaks_error_down_by_group():
    """A single median hides that one cell type is ten times worse."""
    ref_expr, ref_coords, query, truth, groups = _with_dispersed_population()
    p = project_knn(query, ref_expr, ref_coords, k=15)
    m = evaluate_localization(p.coords, truth, ref_coords=ref_coords, groups=groups)
    per = m['per_group']
    assert set(per) == {'dispersed', 'localized'}
    assert per['dispersed']['median_error'] > per['localized']['median_error']
    assert per['dispersed']['n'] == N_DISP and per['localized']['n'] == N_LOC


def test_evaluation_checks_that_confidence_predicts_error():
    """If it doesn't, the score is decoration and the report must say so."""
    ref_expr, ref_coords, query, truth, _ = _with_dispersed_population()
    p = project_knn(query, ref_expr, ref_coords, k=15)
    m = evaluate_localization(p.coords, truth, ref_coords=ref_coords,
                              confidence=p.confidence)
    assert m['confidence_error_correlation'] < -0.3  # higher confidence, less error


def test_unplaced_cells_are_excluded_and_counted():
    ref_expr, ref_coords, q_expr, q_coords = _matched()
    p = project_knn(q_expr, ref_expr, ref_coords, k=10)
    coords = p.coords.copy()
    coords[:10] = np.nan
    m = evaluate_localization(coords, q_coords, ref_coords=ref_coords)
    assert m['n_evaluated'] == len(q_coords) - 10
    assert m['n_unplaced'] == 10
    assert math.isfinite(m['median_error'])


def test_evaluating_nothing_reports_none_rather_than_nan():
    """No cell placed is a real outcome; NaN would not survive the API."""
    truth = np.zeros((5, 2))
    m = evaluate_localization(np.full((5, 2), np.nan), truth)
    assert m['n_evaluated'] == 0
    assert m['median_error'] is None
    json.dumps(m, allow_nan=False)


def test_metrics_are_json_safe():
    ref_expr, ref_coords, q_expr, q_coords = _matched()
    p = project_knn(q_expr, ref_expr, ref_coords, k=10)
    m = evaluate_localization(p.coords, q_coords, ref_coords=ref_coords,
                              confidence=p.confidence,
                              groups=np.array(['a'] * len(q_coords)))
    json.dumps(m, allow_nan=False)


# --- cross-validation -----------------------------------------------------

def test_cross_validation_recovers_held_out_positions():
    ref_coords, ref_expr = _tissue(400, seed=1)
    m = cross_validate_localization(ref_expr, ref_coords, k=10, holdout_fraction=0.25)
    assert m['median_error'] < m['baseline_centroid']
    assert m['n_evaluated'] == 100


def test_cross_validation_is_deterministic():
    ref_coords, ref_expr = _tissue(300, seed=1)
    a = cross_validate_localization(ref_expr, ref_coords, k=10, seed=42)
    b = cross_validate_localization(ref_expr, ref_coords, k=10, seed=42)
    assert a['median_error'] == b['median_error']


def test_cross_validation_says_it_is_an_upper_bound():
    """Within one dataset there is no platform gap, so the real number is
    worse. The caller must not be able to miss that."""
    ref_coords, ref_expr = _tissue(200, seed=1)
    m = cross_validate_localization(ref_expr, ref_coords, k=10)
    assert m['same_platform'] is True


def test_cross_validation_holds_out_from_the_reference_it_predicts_with():
    """A held-out cell must not be its own nearest neighbour, which would make
    the error zero and the whole check meaningless."""
    ref_coords, ref_expr = _tissue(200, seed=1)
    m = cross_validate_localization(ref_expr, ref_coords, k=5, holdout_fraction=0.5)
    assert m['median_error'] > 0


def test_cross_validation_places_cells_correctly_within_their_section():
    """Two cuts of the same tissue are transcriptionally identical, so no method
    can tell which one a cell came from — and demanding that would be demanding
    the impossible. What the section handling must deliver is a coordinate drawn
    from one coherent section and correct *within* it; being off by exactly the
    section offset is a section mix-up, not a position error, while anything in
    between means the prediction landed in the empty gap."""
    ref_expr, ref_coords, sections, _ = _two_sections()
    offset = 500.0
    m = cross_validate_localization(ref_expr, ref_coords, k=10,
                                    ref_sections=sections, holdout_fraction=0.2)
    err = m['median_error']
    assert min(err, abs(err - offset)) < 20.0, err

def test_cross_validation_is_json_safe():
    ref_coords, ref_expr = _tissue(200, seed=1)
    groups = np.array(['a', 'b'] * 100)
    json.dumps(cross_validate_localization(ref_expr, ref_coords, k=10, groups=groups),
               allow_nan=False)
