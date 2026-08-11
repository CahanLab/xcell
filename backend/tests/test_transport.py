import numpy as np
import pytest

from xcell.transport import SinkhornResult, normalize_cost, sinkhorn_log


def test_normalize_cost_is_zero_based_and_unit_spread():
    rng = np.random.default_rng(0)
    cost = rng.normal(5.0, 2.0, size=(50, 80))
    out, sigma = normalize_cost(cost)
    assert out.min() == pytest.approx(0.0, abs=1e-12)
    assert out.std() == pytest.approx(1.0, rel=1e-9)
    assert sigma == pytest.approx(cost.std(), rel=1e-9)


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
    res = sinkhorn_log(cost, epsilon=0.002, max_iterations=3000, tol=1e-6)
    assert np.all(np.isfinite(res.coupling))
    assert res.marginal_error < 1e-2


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


def test_sinkhorn_rejects_a_non_positive_epsilon():
    cost, _ = normalize_cost(_toy_cost())
    with pytest.raises(ValueError, match='epsilon'):
        sinkhorn_log(cost, epsilon=0.0)
