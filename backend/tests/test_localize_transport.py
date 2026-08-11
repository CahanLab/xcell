import numpy as np
import pytest

from xcell.localize import project_knn


def _pair(n_query=60, n_ref=120, n_genes=40, seed=0):
    """A query whose cells each resemble one reference spot, on a 2-D gradient."""
    rng = np.random.default_rng(seed)
    coords = rng.uniform(0, 10, size=(n_ref, 2))
    loadings = rng.normal(size=(2, n_genes))
    ref = coords @ loadings + rng.normal(scale=0.1, size=(n_ref, n_genes))
    pick = rng.choice(n_ref, size=n_query, replace=False)
    query = ref[pick] + rng.normal(scale=0.1, size=(n_query, n_genes))
    return query.astype(np.float32), ref.astype(np.float32), coords, pick


def test_transport_is_an_accepted_aggregation():
    q, r, c, _ = _pair()
    proj = project_knn(q, r, c, aggregation='transport')
    assert proj.coords.shape == (q.shape[0], 2)
    assert proj.transport == 'full'
    assert proj.n_ref_used == r.shape[0]


def test_transport_recovers_the_true_locations_on_a_clean_pair():
    q, r, c, pick = _pair()
    proj = project_knn(q, r, c, aggregation='transport', epsilon=0.3)
    err = np.linalg.norm(proj.coords - c[pick], axis=1)
    assert np.median(err) < 2.0, f'median error {np.median(err):.2f} on a clean pair'


def test_epsilon_moves_the_map_along_the_frontier():
    """Smaller epsilon fills more of the tissue. This is the whole dial.

    Deliberately not a comparison against ``weighted_mean``. On a clean
    synthetic pair weighted_mean does not collapse -- measured here it is the
    most accurate estimator of the lot (median error 0.90 against transport's
    0.98 at its sharpest) -- because each cell's k neighbours really are its
    spatial neighbours. The collapse transport exists to fix comes from
    *ambiguous* matching on real cross-platform data, where it was measured at
    hull area 0.15. A synthetic fixture cannot demonstrate that, and a test
    that pretended to would be measuring nothing.
    """
    q, r, c, _ = _pair()
    spreads = [
        project_knn(q, r, c, aggregation='transport',
                    epsilon=eps, max_iterations=500).coords.std(axis=0).mean()
        for eps in (0.05, 0.2, 1.0, 2.0)
    ]
    assert spreads == sorted(spreads, reverse=True), spreads
    # The sharpest setting must stay near the reference's own spread, and the
    # smoothest must visibly collapse toward the centroid.
    reference_spread = c.std(axis=0).mean()
    assert spreads[0] > 0.85 * reference_spread
    assert spreads[-1] < 0.5 * reference_spread


def test_transport_reports_its_convergence():
    q, r, c, _ = _pair()
    proj = project_knn(q, r, c, aggregation='transport', max_iterations=3)
    assert proj.sinkhorn_iterations == 3
    assert proj.marginal_error is not None and np.isfinite(proj.marginal_error)


def test_transport_rejects_a_non_positive_epsilon():
    q, r, c, _ = _pair()
    with pytest.raises(ValueError, match='epsilon'):
        project_knn(q, r, c, aggregation='transport', epsilon=0.0)


def test_transport_confidence_is_finite_and_bounded():
    q, r, c, _ = _pair()
    proj = project_knn(q, r, c, aggregation='transport')
    assert np.all(np.isfinite(proj.confidence))
    assert proj.confidence.min() >= 0.0 and proj.confidence.max() <= 1.0


def test_transport_keeps_a_cell_within_one_section():
    # Two cuts far apart on the x axis. A prediction landing between them is
    # in empty space, which is the failure the section handling exists for.
    rng = np.random.default_rng(3)
    n_genes = 30
    coords_a = np.column_stack([rng.uniform(0, 5, 60), rng.uniform(0, 5, 60)])
    coords_b = coords_a + np.array([500.0, 0.0])
    coords = np.vstack([coords_a, coords_b])
    sections = np.array(['a'] * 60 + ['b'] * 60)
    ref = rng.normal(size=(120, n_genes)).astype(np.float32)
    # Query cells copy section-a spots, so every prediction belongs in a.
    query = (ref[:30] + rng.normal(scale=0.05, size=(30, n_genes))).astype(np.float32)

    proj = project_knn(
        query, ref, coords, aggregation='transport',
        ref_sections=sections, epsilon=0.3,
    )
    # Nothing may land in the 5..500 gap.
    assert not ((proj.coords[:, 0] > 10) & (proj.coords[:, 0] < 495)).any()
    assert proj.sections is not None


def test_transport_subsamples_a_reference_too_large_to_solve_whole(monkeypatch):
    from xcell import localize as lz

    q, r, c, _ = _pair(n_query=40, n_ref=200)
    # Force the ceiling below this pair rather than building a huge one.
    monkeypatch.setattr(lz, '_TRANSPORT_MAX_ENTRIES', 40 * 50)
    proj = project_knn(q, r, c, aggregation='transport')
    assert proj.transport == 'subsampled'
    assert proj.n_ref_used < 200
    assert np.all(np.isfinite(proj.coords))


def test_other_aggregations_report_no_transport_fields():
    q, r, c, _ = _pair()
    proj = project_knn(q, r, c, aggregation='weighted_mean')
    assert proj.transport is None
    assert proj.n_ref_used is None
    assert proj.sinkhorn_iterations is None
    assert proj.marginal_error is None
