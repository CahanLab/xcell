"""Tests for robust gene co-expression module detection."""
import numpy as np
import pytest

from xcell import gene_coexpression as gc


# ---- shared synthetic helpers (reused by later tasks) ---------------------

def _profiles_from_factors(rng, factors, noise=0.15):
    """Stack genes generated from a list of (factor_vector, n_genes) specs.

    Returns (X_genes, labels): X_genes is (total_genes, n_cells), labels is an
    int ground-truth module id per gene row (noise factor -> its own ids).
    """
    rows, labels = [], []
    for mod_id, (f, n) in enumerate(factors):
        f = (f - f.mean()) / (f.std() + 1e-9)
        for _ in range(n):
            g = f + rng.standard_normal(f.shape[0]) * noise
            rows.append(g)
            labels.append(mod_id)
    return np.asarray(rows, dtype=float), np.asarray(labels)


def test_standardize_rows_are_unit_norm_pearson():
    rng = np.random.default_rng(0)
    X = rng.standard_normal((6, 200))
    Z = gc._standardize_profiles(X, metric="pearson")
    norms = np.linalg.norm(Z, axis=1)
    assert np.allclose(norms, 1.0, atol=1e-8)


def test_corr_matrix_recovers_perfect_correlation():
    rng = np.random.default_rng(1)
    base = rng.standard_normal(200)
    X = np.vstack([base, 2 * base + 3, -base])  # corr: 1, 1, -1 with base
    C = gc.corr_matrix(X, metric="pearson")
    assert C.shape == (3, 3)
    assert np.allclose(np.diag(C), 1.0, atol=1e-8)
    assert C[0, 1] == pytest.approx(1.0, abs=1e-6)
    assert C[0, 2] == pytest.approx(-1.0, abs=1e-6)
    assert np.allclose(C, C.T)


def test_distance_matrix_is_one_minus_corr_clipped():
    rng = np.random.default_rng(2)
    base = rng.standard_normal(200)
    X = np.vstack([base, base, -base])
    D = gc.distance_matrix(X, metric="pearson")
    assert np.all(D >= -1e-9) and np.all(D <= 2 + 1e-9)
    assert np.allclose(np.diag(D), 0.0, atol=1e-8)
    assert D[0, 1] == pytest.approx(0.0, abs=1e-6)
    assert D[0, 2] == pytest.approx(2.0, abs=1e-6)


def test_bicor_is_robust_to_outliers_where_pearson_breaks():
    rng = np.random.default_rng(3)
    base = rng.standard_normal(300)
    a = base + rng.standard_normal(300) * 0.1
    b = base + rng.standard_normal(300) * 0.1
    # Corrupt a few cells with large opposite-sign spikes.
    for i in (5, 50, 150, 250):
        a[i] += 30
        b[i] -= 30
    X = np.vstack([a, b])
    pear = gc.corr_matrix(X, metric="pearson")[0, 1]
    bic = gc.corr_matrix(X, metric="bicor")[0, 1]
    assert bic > pear            # bicor less damaged by the outliers
    assert bic > 0.7             # and still recognizes the co-expression


def test_spearman_handles_monotone_nonlinear():
    rng = np.random.default_rng(4)
    base = rng.standard_normal(300)
    X = np.vstack([base, np.exp(base)])  # monotone but nonlinear
    sp = gc.corr_matrix(X, metric="spearman")[0, 1]
    assert sp > 0.95


def test_coherence_high_for_single_factor_module():
    rng = np.random.default_rng(10)
    X, _ = _profiles_from_factors(rng, [(rng.standard_normal(200), 10)], noise=0.1)
    Z = gc._standardize_profiles(X, "pearson")
    assert gc._module_coherence(Z) > 0.85


def test_coherence_low_for_two_factor_glued_module():
    rng = np.random.default_rng(11)
    f1 = rng.standard_normal(200)
    f2 = rng.standard_normal(200)  # independent of f1
    X, _ = _profiles_from_factors(rng, [(f1, 6), (f2, 6)], noise=0.1)
    Z = gc._standardize_profiles(X, "pearson")
    assert gc._module_coherence(Z) < 0.7


def test_single_gene_module_is_perfectly_coherent():
    rng = np.random.default_rng(12)
    Z = gc._standardize_profiles(rng.standard_normal((1, 200)), "pearson")
    assert gc._module_coherence(Z) == 1.0


def test_eigengene_tracks_underlying_factor():
    rng = np.random.default_rng(13)
    f = rng.standard_normal(200)
    X, _ = _profiles_from_factors(rng, [(f, 8)], noise=0.1)
    Z = gc._standardize_profiles(X, "pearson")
    eg = gc._module_eigengene(Z)
    assert eg.shape == (200,)
    fc = (f - f.mean()) / np.linalg.norm(f - f.mean())
    assert abs(float(eg @ fc)) > 0.9
    assert np.linalg.norm(eg) == pytest.approx(1.0, abs=1e-8)


from sklearn.metrics import adjusted_rand_score


def test_base_cut_recovers_separated_modules():
    rng = np.random.default_rng(20)
    X, labels = _profiles_from_factors(
        rng,
        [(rng.standard_normal(200), 12),
         (rng.standard_normal(200), 10),
         (rng.standard_normal(200), 8)],
        noise=0.12,
    )
    D = gc.distance_matrix(X, metric="pearson")
    found = gc._auto_cut_hierarchical(D)
    assert len(set(found)) >= 2
    assert adjusted_rand_score(labels, found) > 0.7


def test_base_cut_trivial_for_two_genes():
    rng = np.random.default_rng(21)
    D = gc.distance_matrix(rng.standard_normal((2, 50)), metric="pearson")
    found = gc._auto_cut_hierarchical(D)
    assert found.shape == (2,)
