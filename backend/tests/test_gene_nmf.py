"""Tests for xcell.gene_nmf — NMF gene-program discovery (GeneNMF port).

The planted factorization gives a known answer: three disjoint blocks of
genes, each driven by its own cell population, so a rank-3 factorization has
to recover the blocks exactly. Everything about the scaling convention
(L1-diagonalized, factors sorted by decreasing weight) is pinned down here
because the downstream specificity weighting is *not* invariant to it.
"""
import json

import numpy as np
import pytest
import scipy.sparse as sp

from xcell import gene_nmf


N_CELLS, N_GENES, K = 240, 45, 3
BLOCK = N_GENES // K  # 15 genes per planted program


def _planted(seed: int = 0):
    """X = H @ W.T with three disjoint gene blocks. Returns (X, W, H, names)."""
    rng = np.random.default_rng(seed)
    W = np.zeros((N_GENES, K))
    for j in range(K):
        W[j * BLOCK:(j + 1) * BLOCK, j] = rng.uniform(0.5, 1.5, BLOCK)
    # Each cell uses mostly one program, so the blocks are separable.
    H = rng.uniform(0.0, 0.2, size=(N_CELLS, K))
    dominant = rng.integers(0, K, N_CELLS)
    H[np.arange(N_CELLS), dominant] += rng.uniform(2.0, 6.0, N_CELLS)
    X = H @ W.T
    names = [f"G{i:03d}" for i in range(N_GENES)]
    return X, W, H, names


def _planted_sparse(seed: int = 0):
    X, _, _, names = _planted(seed)
    X[X < 0.5] = 0.0
    return sp.csr_matrix(X), names


def _blocks(names):
    """The planted gene names, per program."""
    return [set(names[j * BLOCK:(j + 1) * BLOCK]) for j in range(K)]


# ---------------------------------------------------------------------------
# nmf() — the factorization kernel
# ---------------------------------------------------------------------------
def test_nmf_returns_nonnegative_factors_of_the_right_shape():
    X, _, _, _ = _planted()
    res = gene_nmf.nmf(X, k=K, seed=0)
    assert res["W"].shape == (N_GENES, K)
    assert res["H"].shape == (N_CELLS, K)
    assert res["d"].shape == (K,)
    assert res["W"].min() >= 0.0
    assert res["H"].min() >= 0.0


def test_nmf_recovers_a_planted_low_rank_matrix():
    X, _, _, _ = _planted()
    res = gene_nmf.nmf(X, k=K, seed=0, max_iter=500, tol=1e-9)
    recon = res["H"] @ res["W"].T
    rel = np.linalg.norm(X - recon) / np.linalg.norm(X)
    assert rel < 0.01, f"relative reconstruction error {rel:.4f}"


def test_nmf_l1_diagonalization_normalizes_gene_loading_columns():
    """RcppML's default: columns of W sum to 1, scale lives in d."""
    X, _, _, _ = _planted()
    res = gene_nmf.nmf(X, k=K, seed=0)
    np.testing.assert_allclose(res["W"].sum(axis=0), np.ones(K), rtol=1e-6)


def test_nmf_cell_scores_are_in_data_units_not_renormalized():
    """H must reconstruct X against the L1-normalized W, so programs stay
    comparable to each other and across cells."""
    X, _, _, _ = _planted()
    res = gene_nmf.nmf(X, k=K, seed=0, max_iter=500, tol=1e-9)
    recon = res["H"] @ res["W"].T
    assert np.allclose(recon, X, atol=0.05 * X.max())


def test_nmf_sorts_factors_by_decreasing_weight():
    X, _, _, _ = _planted()
    res = gene_nmf.nmf(X, k=K, seed=0)
    d = res["d"]
    assert np.all(np.diff(d) <= 1e-9), f"d not descending: {d}"


def test_nmf_is_deterministic_for_a_fixed_seed():
    X, _, _, _ = _planted()
    a = gene_nmf.nmf(X, k=K, seed=7)
    b = gene_nmf.nmf(X, k=K, seed=7)
    np.testing.assert_allclose(a["W"], b["W"])
    np.testing.assert_allclose(a["H"], b["H"])


def test_nmf_result_does_not_depend_on_thread_count():
    """Row-chunked sparse matmuls write disjoint output rows, so threading is
    bitwise-neutral; otherwise runs stop being reproducible across machines."""
    X, _ = _planted_sparse()
    one = gene_nmf.nmf(X, k=K, seed=3, n_threads=1)
    many = gene_nmf.nmf(X, k=K, seed=3, n_threads=4)
    np.testing.assert_allclose(one["W"], many["W"], rtol=1e-12, atol=1e-14)


def test_nmf_accepts_sparse_input_and_matches_the_dense_result():
    X, _ = _planted_sparse()
    dense = gene_nmf.nmf(X.toarray(), k=K, seed=1, max_iter=60)
    sparse = gene_nmf.nmf(X, k=K, seed=1, max_iter=60)
    np.testing.assert_allclose(dense["error"], sparse["error"], rtol=1e-6)
    np.testing.assert_allclose(dense["W"], sparse["W"], rtol=1e-4, atol=1e-6)


def test_nmf_reports_iterations_and_convergence():
    """Noisy data has a real residual floor, so the run genuinely plateaus.
    (A noiseless exact-rank matrix never does — it just keeps halving.)"""
    X, _, _, _ = _planted()
    rng = np.random.default_rng(11)
    X = X + rng.uniform(0.0, 0.3 * X.mean(), size=X.shape)
    res = gene_nmf.nmf(X, k=K, seed=0, max_iter=500, tol=1e-6)
    assert 0 < res["n_iter"] < 500
    assert res["converged"] is True
    assert res["error"] >= 0.0


def test_default_tolerance_fits_the_data_not_just_the_first_few_iterations():
    """A stopping rule scaled to the *data* norm rather than to the error still
    on the table quits while the fit is improving fast, and silently returns a
    barely-started factorization. Compared against the data scale, the default
    has to land essentially where an exhaustive run does.
    """
    X, _, _, _ = _planted()
    scale = np.linalg.norm(X)
    quick = gene_nmf.nmf(X, k=K, seed=0)  # defaults
    thorough = gene_nmf.nmf(X, k=K, seed=0, tol=1e-12, max_iter=3000)
    excess = (quick["error"] - thorough["error"]) / scale
    assert excess < 1e-4, (
        f"default stop is premature: relative error {quick['error']/scale:.3g} "
        f"after {quick['n_iter']} iters vs {thorough['error']/scale:.3g} after "
        f"{thorough['n_iter']}"
    )


def test_nmf_reports_non_convergence_when_the_iteration_budget_runs_out():
    X, _, _, _ = _planted()
    res = gene_nmf.nmf(X, k=K, seed=0, max_iter=2, tol=1e-12)
    assert res["n_iter"] == 2
    assert res["converged"] is False


def test_nmf_rejects_negative_input():
    X, _, _, _ = _planted()
    X[0, 0] = -1.0
    with pytest.raises(ValueError, match="non-negative"):
        gene_nmf.nmf(X, k=K)


def test_nmf_rejects_k_larger_than_the_matrix():
    X, _, _, _ = _planted()
    with pytest.raises(ValueError, match="k"):
        gene_nmf.nmf(X, k=N_GENES + 5)


def test_nmf_rejects_k_below_two():
    X, _, _, _ = _planted()
    with pytest.raises(ValueError, match="k"):
        gene_nmf.nmf(X, k=1)


def test_l1_regularization_sparsifies_the_gene_loadings():
    X, _, _, _ = _planted(seed=2)
    plain = gene_nmf.nmf(X, k=K, seed=0, l1_w=0.0)
    reg = gene_nmf.nmf(X, k=K, seed=0, l1_w=0.5)
    zero_frac = lambda W: float((W <= 1e-12).mean())
    assert zero_frac(reg["W"]) > zero_frac(plain["W"])


def test_nmf_rejects_l1_outside_the_unit_interval():
    X, _, _, _ = _planted()
    with pytest.raises(ValueError, match="l1"):
        gene_nmf.nmf(X, k=K, l1_w=1.5)


# ---------------------------------------------------------------------------
# program_genes() — GeneNMF's getNMFgenes()
# ---------------------------------------------------------------------------
def _uniform_loadings():
    """Program 0 spreads evenly over 10 genes; program 1 over 2."""
    W = np.zeros((12, 2))
    W[0:10, 0] = 0.1          # 10 genes at 10% each
    W[10:12, 1] = 0.5         # 2 genes at 50% each
    names = [f"g{i}" for i in range(12)]
    return W, names


def test_program_genes_returns_one_entry_per_program():
    W, names = _uniform_loadings()
    progs = gene_nmf.program_genes(
        W, names, specificity_weight=0.0, weight_explained=0.9
    )
    assert len(progs) == 2
    assert [p["name"] for p in progs] == ["NMF_1", "NMF_2"]


def test_program_genes_cuts_at_the_cumulative_weight_threshold():
    """Ten genes at 10% each, cut at 0.55, keeps the first five."""
    W, names = _uniform_loadings()
    progs = gene_nmf.program_genes(
        W, names, specificity_weight=0.0, weight_explained=0.55
    )
    assert len(progs[0]["genes"]) == 5


def test_program_genes_sorts_genes_by_descending_weight():
    W = np.zeros((5, 1))
    W[:, 0] = [0.1, 0.5, 0.05, 0.3, 0.05]
    names = ["a", "b", "c", "d", "e"]
    progs = gene_nmf.program_genes(
        W, names, specificity_weight=0.0, weight_explained=0.85
    )
    assert progs[0]["genes"][:2] == ["b", "d"]
    assert progs[0]["weights"] == sorted(progs[0]["weights"], reverse=True)


def test_program_genes_caps_at_max_genes():
    W = np.zeros((100, 1))
    W[:, 0] = 0.01
    names = [f"g{i}" for i in range(100)]
    progs = gene_nmf.program_genes(
        W, names, specificity_weight=0.0, weight_explained=0.99, max_genes=7
    )
    assert len(progs[0]["genes"]) == 7


def test_program_genes_drops_a_program_whose_top_gene_exceeds_the_threshold():
    """GeneNMF's strict `cumsum <` cutoff empties a program dominated by one
    gene; those programs are dropped, not returned with an empty gene list."""
    W, names = _uniform_loadings()
    progs = gene_nmf.program_genes(
        W, names, specificity_weight=0.0, weight_explained=0.45
    )
    assert [p["name"] for p in progs] == ["NMF_1"]


def test_program_genes_excludes_the_gene_that_lands_exactly_on_the_threshold():
    """GeneNMF's cutoff is a strict ``cumsum < weight_explained``. Shares of
    3:1 give a cumulative sum of exactly 0.75 after the first gene — binary-
    exact, so this pins the boundary rather than sampling near it."""
    W = np.array([[3.0], [1.0]])
    progs = gene_nmf.program_genes(
        W, ["big", "small"], specificity_weight=0.0, weight_explained=0.75
    )
    assert progs == [], "a gene sitting exactly on the threshold must be excluded"


def _shared_vs_specific():
    """One gene split evenly across both programs, one exclusive to program 0."""
    W = np.zeros((5, 2))
    W[0] = [0.30, 0.30]   # "shared"    — half its mass in the other program
    W[1] = [0.29, 0.00]   # "specific"  — exclusive to program 0
    W[2] = [0.41, 0.00]   # "filler"    — exclusive, and the strongest
    W[3] = [0.10, 0.00]   # "tail"      — keeps the cumulative cutoff off the edge
    W[4] = [0.00, 0.70]   # "other"     — program 1 only
    return W, ["shared", "specific", "filler", "tail", "other"]


def test_specificity_weighting_demotes_a_gene_shared_across_programs():
    W, names = _shared_vs_specific()
    flat = gene_nmf.program_genes(
        W, names, specificity_weight=0.0, weight_explained=0.95
    )
    spec = gene_nmf.program_genes(
        W, names, specificity_weight=5.0, weight_explained=0.95
    )
    # Unweighted, the shared gene outranks the exclusive one on raw loading.
    assert flat[0]["genes"].index("shared") < flat[0]["genes"].index("specific")
    # Weighted, it loses so much mass it falls out of the program entirely.
    assert "shared" not in spec[0]["genes"]


def test_specificity_weighting_promotes_an_exclusive_gene():
    W, names = _shared_vs_specific()
    flat = gene_nmf.program_genes(
        W, names, specificity_weight=0.0, weight_explained=0.95
    )
    spec = gene_nmf.program_genes(
        W, names, specificity_weight=5.0, weight_explained=0.95
    )
    assert spec[0]["genes"].index("specific") < flat[0]["genes"].index("specific")


def _realistic_loadings(n_genes: int = 1500, k: int = 12, seed: int = 0):
    """Gene loadings shaped like a real single-cell factorization.

    Calibrated against an actual 24k-cell E11.5 limb run (k=12, 5001 HVGs) on
    the two properties that drive gene selection:

    * per-gene specificity — the largest share any one program takes of a
      gene's loading — percentiles [0.16, 0.20, 0.24, 0.30, 0.43] here against
      [0.16, 0.21, 0.27, 0.36, 0.57] measured. Nearly every gene is shared
      across several programs; nothing is exclusive.
    * heavy-tailed columns, so a few percent of genes carry half a program.

    Toy fixtures with disjoint gene blocks have specificity 1.0 for every gene
    and so cannot exercise this path at all.
    """
    rng = np.random.default_rng(seed)
    gene_scale = rng.normal(0.0, 1.5, size=(n_genes, 1))     # heavy-tailed columns
    W = np.exp(gene_scale + rng.normal(0.0, 0.9, size=(n_genes, k)))  # sets specificity
    W /= W.sum(axis=0, keepdims=True)
    return W, [f"g{i:04d}" for i in range(n_genes)]


def test_default_specificity_weight_keeps_programs_usable():
    """Specificity weighting is exponential: too large an exponent piles a
    program's whole mass onto its few most exclusive genes, and the cumulative
    cutoff then keeps almost nothing. GeneNMF's default of 5 gets away with it
    because it always feeds the *averaged* consensus of many programs; on one
    factorization it guts them. See
    docs/measurements/2026-08-19-gene-nmf-specificity-weight.md.
    """
    W, names = _realistic_loadings()
    progs = gene_nmf.program_genes(W, names)  # shipped defaults
    assert len(progs) == 12, f"defaults kept only {len(progs)}/12 programs"
    sizes = sorted(len(p["genes"]) for p in progs)
    assert sizes[0] >= 20, f"smallest program is not a program: {sizes}"
    assert sizes[len(sizes) // 2] >= 40, f"programs are stunted: {sizes}"


def test_specificity_weight_trades_program_size_for_exclusivity():
    """Raising it monotonically narrows programs — the knob works, and 5 (the
    meta-program default) is past the point where a program survives."""
    W, names = _realistic_loadings()
    sizes = []
    for sw in (0.0, 1.0, 3.0, 5.0):
        progs = gene_nmf.program_genes(W, names, specificity_weight=sw)
        sizes.append(int(np.median([len(p["genes"]) for p in progs])))
    assert sizes == sorted(sizes, reverse=True), sizes
    assert sizes[-1] < 20, f"specificity weighting is not biting: {sizes}"


def test_program_gene_weights_are_shares_of_the_whole_program():
    """Weights are normalized over the full program before truncation, so the
    kept ones sum to just under weight_explained (GeneNMF's convention)."""
    W, names = _uniform_loadings()
    progs = gene_nmf.program_genes(
        W, names, specificity_weight=0.0, weight_explained=0.75
    )
    total = sum(progs[0]["weights"])
    assert 0.65 < total < 0.75


def test_program_genes_keeps_the_original_factor_index_after_a_drop():
    W, names = _uniform_loadings()
    progs = gene_nmf.program_genes(
        W, names, specificity_weight=0.0, weight_explained=0.45
    )
    assert progs[0]["index"] == 0


def test_program_genes_rejects_a_name_count_that_does_not_match_w():
    W, names = _uniform_loadings()
    with pytest.raises(ValueError, match="gene_names"):
        gene_nmf.program_genes(W, names[:-1], specificity_weight=0.0)


# ---------------------------------------------------------------------------
# run_gene_programs() — the end-to-end entry point
# ---------------------------------------------------------------------------
def test_run_gene_programs_recovers_the_planted_blocks():
    X, _, _, names = _planted()
    out = gene_nmf.run_gene_programs(
        X, names, k=K, seed=0, weight_explained=0.8, max_iter=500
    )
    recovered = [set(p["genes"]) for p in out["programs"]]
    planted = _blocks(names)
    assert len(recovered) == K
    for got in recovered:
        assert any(got <= block for block in planted), (
            f"program {sorted(got)} is not contained in any planted block"
        )
    matched = {
        i for got in recovered for i, block in enumerate(planted) if got <= block
    }
    assert matched == {0, 1, 2}


def test_run_gene_programs_returns_cell_scores_aligned_to_programs():
    X, _, _, names = _planted()
    out = gene_nmf.run_gene_programs(X, names, k=K, seed=0, weight_explained=0.8)
    scores = out["cell_scores"]
    assert scores.shape == (N_CELLS, len(out["programs"]))
    assert scores.min() >= 0.0


def test_run_gene_programs_returns_loadings_aligned_to_programs():
    """The adaptor stores these in .varm, so they have to be the surviving
    programs' columns only, still on the unit simplex."""
    X, _, _, names = _planted()
    out = gene_nmf.run_gene_programs(X, names, k=K, seed=0, weight_explained=0.8)
    loadings = out["loadings"]
    assert loadings.shape == (N_GENES, len(out["programs"]))
    np.testing.assert_allclose(loadings.sum(axis=0), 1.0, rtol=1e-6)
    assert loadings.min() >= 0.0


def test_run_gene_programs_reports_json_safe_metrics():
    X, _, _, names = _planted()
    out = gene_nmf.run_gene_programs(X, names, k=K, seed=0)
    arrays = {"cell_scores", "loadings"}
    payload = {k: v for k, v in out.items() if k not in arrays}
    json.dumps(payload)  # must not raise: no numpy scalars, no NaN/inf
    for prog in out["programs"]:
        assert np.isfinite(prog["weights"]).all()
    assert np.isfinite(out["reconstruction_error"])


def test_run_gene_programs_reports_which_programs_were_dropped():
    X, _, _, names = _planted()
    out = gene_nmf.run_gene_programs(
        X, names, k=K, seed=0, weight_explained=0.01  # cuts every program
    )
    assert out["n_dropped"] == K
    assert out["programs"] == []


def test_run_gene_programs_progress_callback_reaches_one():
    X, _, _, names = _planted()
    seen = []
    gene_nmf.run_gene_programs(
        X, names, k=K, seed=0, max_iter=60,
        progress_callback=lambda f, m: seen.append(f),
    )
    assert seen, "progress callback was never called"
    assert seen == sorted(seen)
    assert seen[0] >= 0.0
    assert seen[-1] == 1.0
