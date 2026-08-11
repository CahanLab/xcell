import numpy as np
import pytest

from xcell.transport import (
    SinkhornResult,
    barycentric_projection,
    normalize_cost,
    posterior_spread,
    sinkhorn_log,
    stratified_subsample,
)


def test_normalize_cost_is_zero_based_and_unit_spread():
    rng = np.random.default_rng(0)
    cost = rng.normal(5.0, 2.0, size=(50, 80))
    out, sigma = normalize_cost(cost)
    assert out.min() == pytest.approx(0.0, abs=1e-12)
    assert out.std() == pytest.approx(1.0, rel=1e-9)
    # sigma is the spread of the double-centred matrix, not the raw one: the
    # row and column effects cannot move the coupling, so they must not set
    # the scale epsilon is measured against.
    centred = (cost - cost.mean(axis=1, keepdims=True)
               - cost.mean(axis=0, keepdims=True) + cost.mean())
    assert sigma == pytest.approx(centred.std(), rel=1e-9)
    assert sigma < cost.std()


def test_normalize_cost_is_invariant_to_scale_and_shift():
    rng = np.random.default_rng(1)
    cost = rng.normal(0.0, 1.0, size=(30, 40))
    base, _ = normalize_cost(cost)
    scaled, _ = normalize_cost(cost * 7.0)
    shifted, _ = normalize_cost(cost + 3.0)
    np.testing.assert_allclose(scaled, base, rtol=1e-9, atol=1e-9)
    np.testing.assert_allclose(shifted, base, rtol=1e-9, atol=1e-9)


def test_normalize_cost_survives_a_constant_matrix():
    # Every pair equally similar carries no information; dividing by a zero
    # spread would poison the whole matrix with inf.
    out, sigma = normalize_cost(np.full((10, 12), 4.0))
    assert np.all(np.isfinite(out))
    assert sigma == 1.0


def _toy_cost(n=40, m=60, seed=0):
    rng = np.random.default_rng(seed)
    qx = rng.uniform(0, 1, size=n)
    rx = rng.uniform(0, 1, size=m)
    return np.abs(qx[:, None] - rx[None, :])


def test_sinkhorn_satisfies_both_marginals():
    cost, _ = normalize_cost(_toy_cost())
    n, m = cost.shape
    res = sinkhorn_log(cost, epsilon=0.5, max_iterations=500, tol=1e-9)
    assert isinstance(res, SinkhornResult)
    np.testing.assert_allclose(res.coupling.sum(axis=1), 1.0 / n, rtol=1e-6)
    np.testing.assert_allclose(res.coupling.sum(axis=0), 1.0 / m, rtol=1e-6)


def test_sinkhorn_is_invariant_to_cost_scale_after_normalization():
    # The whole justification for normalize_cost: the same epsilon must mean
    # the same thing whatever units the similarity came in.
    raw = _toy_cost()
    a, _ = normalize_cost(raw)
    b, _ = normalize_cost(raw * 13.0)
    pa = sinkhorn_log(a, epsilon=0.5, max_iterations=500, tol=1e-9).coupling
    pb = sinkhorn_log(b, epsilon=0.5, max_iterations=500, tol=1e-9).coupling
    np.testing.assert_allclose(pa, pb, rtol=1e-6, atol=1e-12)


def test_larger_epsilon_gives_a_more_diffuse_coupling():
    cost, _ = normalize_cost(_toy_cost())
    sharp = sinkhorn_log(cost, epsilon=0.1, max_iterations=500).coupling
    smooth = sinkhorn_log(cost, epsilon=2.0, max_iterations=500).coupling

    def entropy(p):
        q = p / p.sum()
        return float(-(q * np.log(q + 1e-300)).sum())

    assert entropy(smooth) > entropy(sharp)


def test_sinkhorn_is_finite_at_an_epsilon_that_underflows_the_naive_form():
    cost, _ = normalize_cost(_toy_cost())
    # The kernel form builds exp(-C/eps) directly. At eps=0.002 the largest
    # costs reach exp(-1200), which is not small -- it is *gone*, flushed to
    # exactly zero in float64, and every ratio computed from it is 0/0.
    assert np.exp(-cost / 0.002).min() == 0.0

    # No threshold on the error: at an epsilon this small Sinkhorn converges
    # slowly by nature (3,000 iterations still leaves it near 3e-2), and
    # picking a number that happened to pass would assert nothing. What must
    # hold is that the iterates stay finite and keep improving, where the
    # kernel form has no iterates at all.
    short = sinkhorn_log(cost, epsilon=0.002, max_iterations=300, tol=0.0)
    long = sinkhorn_log(cost, epsilon=0.002, max_iterations=3000, tol=0.0)
    assert np.all(np.isfinite(long.coupling))
    assert not np.isnan(long.coupling).any()
    assert long.marginal_error < short.marginal_error


def test_sinkhorn_reports_budget_exhaustion_honestly():
    cost, _ = normalize_cost(_toy_cost())
    res = sinkhorn_log(cost, epsilon=0.05, max_iterations=2, tol=1e-12)
    assert res.iterations == 2
    assert res.marginal_error > 1e-12   # it did not converge, and says so


def test_sinkhorn_reports_progress():
    cost, _ = normalize_cost(_toy_cost())
    seen = []
    sinkhorn_log(cost, epsilon=0.5, max_iterations=50, tol=1e-12,
                 progress=lambda f, msg: seen.append((f, msg)))
    assert seen
    assert all(0.0 <= f <= 1.0 for f, _ in seen)


def test_normalization_ignores_row_and_column_offsets():
    """The coupling must not move when the cost gains per-cell or per-spot offsets.

    Entropic OT is invariant to C_ij -> C_ij + a_i + b_j, because the potentials
    absorb both. A normalizer that counts those offsets therefore makes the same
    epsilon mean different things on two matrices with *identical* couplings.
    Measured on the E11.5 limb pair, the column effect alone was 98% of the
    global spread -- some reference spots are similar to everything -- so
    dividing by it inflated the scale 6.2x and every run collapsed.
    """
    rng = np.random.default_rng(5)
    base = _toy_cost(30, 40)
    offsets = (rng.normal(scale=5.0, size=30)[:, None]
               + rng.normal(scale=5.0, size=40)[None, :])
    pa = sinkhorn_log(normalize_cost(base)[0], 0.5,
                      max_iterations=800, tol=1e-10).coupling
    pb = sinkhorn_log(normalize_cost(base + offsets)[0], 0.5,
                      max_iterations=800, tol=1e-10).coupling
    np.testing.assert_allclose(pa, pb, rtol=1e-5, atol=1e-12)


def test_sinkhorn_rejects_a_non_positive_epsilon():
    cost, _ = normalize_cost(_toy_cost())
    with pytest.raises(ValueError, match='epsilon'):
        sinkhorn_log(cost, epsilon=0.0)


def test_barycentric_is_the_row_weighted_mean_of_coordinates():
    coupling = np.array([[0.5, 0.5, 0.0],
                         [0.0, 0.0, 1.0]]) / 2.0   # rows sum to 1/n
    coords = np.array([[0.0, 0.0], [2.0, 0.0], [10.0, 4.0]])
    pred, sections = barycentric_projection(coupling, coords)
    assert sections is None
    np.testing.assert_allclose(pred, [[1.0, 0.0], [10.0, 4.0]])


def test_a_row_split_across_sections_does_not_land_in_the_gap():
    # Two cuts 100 apart. A row with mass in both must resolve to one of them,
    # not to the empty space between -- the failure project_knn already guards
    # against for the k-NN path.
    coupling = np.array([[0.4, 0.2, 0.3, 0.1]])
    coords = np.array([[0.0, 0.0], [1.0, 0.0], [100.0, 0.0], [101.0, 0.0]])
    sections = np.array(['a', 'a', 'b', 'b'])
    pred, chosen = barycentric_projection(coupling, coords, sections)
    assert chosen is not None and chosen[0] == 'a'          # 0.6 of the mass
    assert pred[0][0] < 1.0                                  # inside section a
    np.testing.assert_allclose(pred[0], [1.0 / 3.0, 0.0])    # 0.4/0.6, 0.2/0.6


def test_posterior_spread_is_zero_for_a_certain_row():
    coupling = np.array([[0.0, 1.0]])
    coords = np.array([[0.0, 0.0], [3.0, 4.0]])
    pred, _ = barycentric_projection(coupling, coords)
    assert posterior_spread(coupling, coords, pred)[0] == pytest.approx(0.0)


def test_posterior_spread_grows_as_the_posterior_spreads():
    coords = np.array([[0.0, 0.0], [10.0, 0.0]])
    tight = np.array([[0.99, 0.01]])
    loose = np.array([[0.5, 0.5]])
    st = posterior_spread(tight, coords, barycentric_projection(tight, coords)[0])
    sl = posterior_spread(loose, coords, barycentric_projection(loose, coords)[0])
    assert sl[0] > st[0]


def test_subsample_returns_the_requested_size():
    rng = np.random.default_rng(0)
    coords = rng.uniform(0, 100, size=(5000, 2))
    idx = stratified_subsample(coords, 500)
    assert len(idx) == 500
    assert len(np.unique(idx)) == 500


def test_subsample_keeps_every_occupied_region():
    # A dense blob plus a sparse arm. Uniform sampling would thin the arm in
    # proportion to its density, and the column marginal would then be
    # enforcing occupancy of a tissue that is missing a limb.
    rng = np.random.default_rng(0)
    blob = rng.normal(10, 1.0, size=(4000, 2))
    arm = np.stack([np.linspace(40, 60, 40), np.full(40, 50.0)], axis=1)
    coords = np.vstack([blob, arm])
    idx = stratified_subsample(coords, 300)
    kept_arm = (idx >= 4000).sum()
    assert kept_arm >= 5, f'the sparse arm all but vanished: {kept_arm} of 40'


def test_subsample_returns_everything_when_the_target_is_larger():
    coords = np.random.default_rng(0).uniform(0, 1, size=(100, 2))
    idx = stratified_subsample(coords, 500)
    np.testing.assert_array_equal(idx, np.arange(100))


def test_subsample_is_deterministic_for_a_seed():
    coords = np.random.default_rng(0).uniform(0, 1, size=(2000, 2))
    np.testing.assert_array_equal(
        stratified_subsample(coords, 200, seed=7),
        stratified_subsample(coords, 200, seed=7),
    )


def test_subsample_indices_are_valid_and_sorted():
    coords = np.random.default_rng(0).uniform(0, 1, size=(3000, 2))
    idx = stratified_subsample(coords, 400)
    assert idx.min() >= 0 and idx.max() < 3000
    np.testing.assert_array_equal(idx, np.sort(idx))
