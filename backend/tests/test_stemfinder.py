"""PyStemFinder differentiation scores (Analyze → Cells → Differentiation).

The pure module prepares expression and neighbourhoods and calls PyStemFinder;
the adaptor scopes, snapshots and writes back. PyStemFinder is an optional
dependency: everything that needs it skips when it is absent, and the tests
that pin the "absent" behaviour run either way.

The toy data has two populations. In ``hi`` the five cell-cycle markers are
expressed in about half the cells, in ``lo`` in about one in ten — so ``hi``
is the heterogeneous, "less differentiated" population and must score so.
"""

import importlib.util

import anndata
import numpy as np
import pandas as pd
import pytest
from scipy.sparse import csr_matrix, issparse

from xcell import stemfinder as sf

HAVE_PSF = importlib.util.find_spec("pystemfinder") is not None
needs_psf = pytest.mark.skipif(not HAVE_PSF, reason="pystemfinder not installed")

MARKERS = ["Mcm5", "Pcna", "Top2a", "Mki67", "Cdk1"]      # all in psf's mouse lists
TFS = ["Sox2", "Myc", "Gata1"]
N_HI, N_LO = 90, 90
N_CELLS = N_HI + N_LO


def _counts():
    rng = np.random.default_rng(0)
    n_bg = 40
    pop = np.array(["hi"] * N_HI + ["lo"] * N_LO)
    # Background genes separate the populations, so PCA and the kNN graph do.
    bg = rng.poisson(3.0, size=(N_CELLS, n_bg)).astype(np.float32)
    bg[pop == "hi", :10] += rng.poisson(6.0, size=(N_HI, 10))
    bg[pop == "lo", 10:20] += rng.poisson(6.0, size=(N_LO, 10))
    on = np.where(pop == "hi", 0.5, 0.1)[:, None]
    mk = (rng.random((N_CELLS, len(MARKERS))) < on) * rng.poisson(5.0, size=(N_CELLS, len(MARKERS)))
    tf = (rng.random((N_CELLS, len(TFS))) < np.where(pop == "hi", 0.8, 0.2)[:, None]) * 2
    X = np.hstack([bg, mk, tf]).astype(np.float32)
    names = [f"bg{i:02d}" for i in range(n_bg)] + MARKERS + TFS
    return X, names, pop


def _adata(log=False):
    X, names, pop = _counts()
    if log:
        X = np.log1p(X / X.sum(axis=1, keepdims=True) * 1e4).astype(np.float32)
    ad = anndata.AnnData(X=csr_matrix(X))
    ad.var_names = names
    ad.obs["pop"] = pd.Categorical(pop)
    return ad


# ---------- markers ----------

def test_match_genes_is_case_insensitive_and_keeps_the_dataset_spelling():
    present, missing = sf.match_genes(["MCM5", "pcna", "Nope", "Mcm5"], ["Mcm5", "Pcna", "Sox2"])
    assert present == ["Mcm5", "Pcna"]
    assert missing == ["Nope"]


def test_match_genes_prefers_an_exact_match_over_a_case_folded_one():
    present, _ = sf.match_genes(["Abc"], ["ABC", "Abc"])
    assert present == ["Abc"]


# ---------- expression ----------

def test_counts_are_normalized_over_every_gene_then_logged():
    X, names, _ = _counts()
    cols = [names.index(g) for g in MARKERS]
    got = sf.log_normalized_columns(csr_matrix(X), cols, "raw_counts")
    want = np.log1p(X / X.sum(axis=1, keepdims=True) * 1e4)[:, cols]
    np.testing.assert_allclose(got, want, rtol=1e-5)


def test_log_scale_input_is_read_as_is():
    ad = _adata(log=True)
    cols = [list(ad.var_names).index(g) for g in MARKERS]
    got = sf.log_normalized_columns(ad.X, cols, "log_normalized")
    np.testing.assert_allclose(got, ad.X[:, cols].toarray())


def test_scaling_is_per_gene():
    M = np.array([[0.0, 1.0], [2.0, 1.0], [4.0, 1.0]])
    S = sf.scale_columns(M)
    np.testing.assert_allclose(S[:, 0].mean(), 0.0, atol=1e-12)
    assert S[0, 0] < 0 < S[2, 0]
    np.testing.assert_allclose(S[:, 1], 0.0)          # constant gene: no NaN


def test_count_expressed_counts_nonzero_columns_without_densifying():
    X = csr_matrix(np.array([[0, 1, 2], [0, 0, 0], [3, 0, 1]], dtype=float))
    np.testing.assert_array_equal(sf.count_expressed(X, [0, 2]), [1, 0, 2])


# ---------- neighbourhoods ----------

def test_knn_from_embedding_gives_k_minus_one_neighbours_per_cell():
    rng = np.random.default_rng(0)
    rep = rng.normal(size=(50, 5))
    g = sf.knn_from_embedding(rep, n_neighbors=7, n_pcs=3)
    assert issparse(g) and g.shape == (50, 50)
    assert set(np.diff(g.indptr)) == {6}


def test_knn_clamps_k_below_the_cell_count():
    rep = np.random.default_rng(0).normal(size=(6, 3))
    g = sf.knn_from_embedding(rep, n_neighbors=40, n_pcs=None)
    assert np.diff(g.indptr).max() <= 5


def test_subgraph_keeps_only_edges_inside_the_scope():
    g = csr_matrix(np.array([[0, 1, 1, 0], [1, 0, 0, 1], [1, 0, 0, 0], [0, 1, 0, 0]], dtype=float))
    sub = sf.subgraph(g, np.array([0, 2, 3]))
    np.testing.assert_array_equal(sub.toarray(), [[0, 1, 0], [1, 0, 0], [0, 0, 0]])


# ---------- scores ----------

def _scores(**kw):
    ad = _adata()
    cols = [list(ad.var_names).index(g) for g in MARKERS]
    log = sf.log_normalized_columns(ad.X, cols, "raw_counts")
    graph = sf.knn_from_embedding(np.log1p(ad.X.toarray()), n_neighbors=14, n_pcs=None)
    args = dict(metrics=["stemfinder", "diffometer"], method="gini", threshold=0.0,
                binarize_on="scaled", weight_by="equal", include_self=True)
    args.update(kw)
    return sf.compute_scores(log, MARKERS, graph, **args), ad


@needs_psf
def test_the_heterogeneous_population_scores_as_less_differentiated():
    s, ad = _scores()
    hi = (ad.obs["pop"] == "hi").to_numpy()
    # stemfinder is oriented like pseudotime: lower = less differentiated.
    assert np.median(s["stemfinder"][hi]) < np.median(s["stemfinder"][~hi])
    assert np.median(s["stemfinder_raw"][hi]) > np.median(s["stemfinder_raw"][~hi])
    assert np.median(s["diffometer"][hi]) > np.median(s["diffometer"][~hi])


@needs_psf
def test_diffometer_without_self_is_twice_stemfinder_raw_over_the_markers():
    # PyStemFinder's documented identity; holds only if both saw the same
    # binarized matrix and graph.
    s, _ = _scores(include_self=False)
    np.testing.assert_allclose(s["diffometer"], 2 * s["stemfinder_raw"] / len(MARKERS))


@needs_psf
def test_stdev_and_variance_read_log_normalized_expression():
    s, _ = _scores(metrics=["stemfinder"], method="variance")
    sd, _ = _scores(metrics=["stemfinder"], method="stdev")
    assert np.all(s["stemfinder_raw"] >= 0)
    assert not np.allclose(s["stemfinder_raw"], sd["stemfinder_raw"])


@needs_psf
def test_a_cell_with_no_neighbours_gets_nan_and_does_not_poison_the_rest():
    ad = _adata()
    cols = [list(ad.var_names).index(g) for g in MARKERS]
    log = sf.log_normalized_columns(ad.X, cols, "raw_counts")
    graph = sf.knn_from_embedding(np.log1p(ad.X.toarray()), n_neighbors=14, n_pcs=None).tolil()
    graph[0, :] = 0
    s = sf.compute_scores(log, MARKERS, graph.tocsr(), metrics=["stemfinder", "diffometer"],
                          method="gini", threshold=0.0, binarize_on="scaled",
                          weight_by="equal", include_self=False)
    assert np.isnan(s["stemfinder"][0]) and np.isnan(s["diffometer"][0])
    assert np.isfinite(s["stemfinder"][1:]).all()
    assert np.nanmax(s["stemfinder"]) <= 1.0


@needs_psf
def test_an_isolated_cell_is_nan_for_diffometer_even_counting_itself():
    # With include_self the lone cell's neighbourhood is itself: impurity 0,
    # which reads as "fully differentiated". It has no neighbourhood at all.
    ad = _adata()
    cols = [list(ad.var_names).index(g) for g in MARKERS]
    log = sf.log_normalized_columns(ad.X, cols, "raw_counts")
    graph = sf.knn_from_embedding(np.log1p(ad.X.toarray()), n_neighbors=14, n_pcs=None).tolil()
    graph[0, :] = 0
    s = sf.compute_scores(log, MARKERS, graph.tocsr(), metrics=["stemfinder", "diffometer"],
                          method="gini", threshold=0.0, binarize_on="scaled",
                          weight_by="equal", include_self=True)
    assert np.isnan(s["diffometer"][0]) and np.isnan(s["stemfinder_raw"][0])
    assert np.isfinite(s["diffometer"][1:]).all()


def test_neighbour_counts_exclude_the_cell_itself():
    g = csr_matrix(np.array([[1, 1, 0], [0, 0, 0], [1, 1, 1]], dtype=float))
    np.testing.assert_array_equal(sf.neighbour_counts(g), [1, 0, 2])


def test_the_knn_graph_matches_scanpys_neighbours():
    # stemFinder reads only which cells are neighbours. Built directly, it
    # skips the UMAP connectivities sc.pp.neighbors spends most of its time on.
    import anndata as _ad
    import scanpy as sc
    rng = np.random.default_rng(1)
    rep = rng.normal(size=(300, 8)).astype(np.float32)
    ours = sf.knn_from_embedding(rep, n_neighbors=17, n_pcs=5)
    ref = _ad.AnnData(obs={"i": np.arange(300)})
    ref.obsm["X_rep"] = rep[:, :5]
    sc.pp.neighbors(ref, n_neighbors=17, use_rep="X_rep")
    theirs = ref.obsp["distances"]
    for i in range(300):
        assert set(ours[i].indices) == set(theirs[i].indices)


@needs_psf
def test_cell_cycle_mean_is_the_mean_log_expression_of_the_markers():
    ad = _adata()
    cols = [list(ad.var_names).index(g) for g in MARKERS]
    log = sf.log_normalized_columns(ad.X, cols, "raw_counts")
    s, _ = _scores(metrics=["cc_mean"])
    np.testing.assert_allclose(s["stemfinder_cc_mean"], log.mean(axis=1))


def test_expression_weighting_needs_log_normalized_binarizing():
    with pytest.raises(ValueError, match="expression"):
        sf.validate_options(metrics=["diffometer"], method="gini", binarize_on="scaled",
                            weight_by="expression", scale_verdict="raw_counts")


def test_z_scored_input_refuses_the_log_normalized_metrics():
    with pytest.raises(ValueError, match="scaled"):
        sf.validate_options(metrics=["stemfinder"], method="stdev", binarize_on="scaled",
                            weight_by="equal", scale_verdict="z_scored")
    # gini on scaled input is exactly what stemFinder wants.
    sf.validate_options(metrics=["stemfinder"], method="gini", binarize_on="scaled",
                        weight_by="equal", scale_verdict="z_scored")


def test_unknown_metric_is_rejected():
    with pytest.raises(ValueError, match="metric"):
        sf.validate_options(metrics=["potency"], method="gini", binarize_on="scaled",
                            weight_by="equal", scale_verdict="raw_counts")


# ---------- summary ----------

def test_group_summary_sorts_by_median_and_is_json_safe():
    scores = {"stemfinder": np.array([0.1, 0.2, 0.9, 0.8, np.nan])}
    labels = np.array(["a", "a", "b", "b", "c"], dtype=object)
    out = sf.group_summary(scores, labels, sort_by="stemfinder")
    assert [g["group"] for g in out] == ["a", "b", "c"]
    assert out[0]["n"] == 2
    assert out[0]["medians"]["stemfinder"] == pytest.approx(0.15)
    assert out[2]["medians"]["stemfinder"] is None


# ---------- availability ----------

def test_availability_reports_absence_with_install_instructions(monkeypatch):
    monkeypatch.setattr(sf, "_import_module", lambda: None)
    a = sf.availability()
    assert a["available"] is False
    assert "pip install pystemfinder" in a["install_hint"]
    with pytest.raises(ValueError, match="pip install pystemfinder"):
        sf.import_psf()


@needs_psf
def test_availability_reports_the_version():
    a = sf.availability()
    assert a["available"] is True and a["version"]


# ---------- column names ----------
# A suffix is appended to each base name, and 'stemfinder' is a prefix of the
# other bases: suffix 'raw' would write stemFinder's score to 'stemfinder_raw',
# the raw score's own name — overwriting an unsuffixed run's raw score and
# reading back as "higher = less differentiated", the wrong way round.

def test_column_names_suffix_every_base_a_metric_writes():
    assert sf.column_names(["stemfinder", "diffometer"], "k50") == {
        "stemfinder": "stemfinder_k50", "stemfinder_raw": "stemfinder_raw_k50",
        "diffometer": "diffometer_k50",
    }
    assert sf.column_names(["cc_mean"], "") == {"stemfinder_cc_mean": "stemfinder_cc_mean"}


@pytest.mark.parametrize("suffix", ["raw", "raw_counts", "cc_mean", "n_TFs_v2"])
def test_a_suffix_that_reads_back_as_another_score_is_refused(suffix):
    with pytest.raises(ValueError, match=f"stemfinder_{suffix}"):
        sf.column_names(["stemfinder"], suffix)


def test_a_suffix_is_refused_only_where_it_can_be_misread():
    # diffOmeter and the baselines have no shorter base to collide with.
    assert sf.column_names(["diffometer", "cc_mean"], "raw") == {
        "diffometer": "diffometer_raw", "stemfinder_cc_mean": "stemfinder_cc_mean_raw",
    }
    # A word that merely starts with the letters is a different token.
    assert sf.column_names(["stemfinder"], "rawcounts")["stemfinder"] == "stemfinder_rawcounts"


def test_potency_reads_every_written_name_back_to_its_metric():
    names = sf.column_names(["stemfinder", "diffometer", "cc_mean", "n_tfs"], "v2")
    potency = {p["column"]: p["direction"] for p in sf.potency_columns(list(names.values()))}
    assert potency == {"stemfinder_v2": "min", "stemfinder_raw_v2": "max", "diffometer_v2": "max"}
