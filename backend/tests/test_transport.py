import numpy as np
import pytest

from xcell.transport import normalize_cost


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
