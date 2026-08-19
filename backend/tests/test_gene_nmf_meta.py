"""Tests for the multi-sample half of xcell.gene_nmf — meta-programs.

The planted setup is the question meta-programs exist to answer: three gene
blocks are used by every sample, and a fourth by only one. A correct run
recovers all four, and tells them apart by sample coverage — the shared ones
at 1.0, the private one at 1/3.
"""
import json

import numpy as np
import pytest
import scipy.sparse as sp

from xcell import gene_nmf


N_SAMPLES, N_CELLS, N_GENES = 3, 400, 60
BLOCK = 15          # genes per planted program
SHARED = ("A", "B", "C")
PRIVATE_SAMPLE = "s1"   # only this sample uses block D


def _blocks():
    """Gene names per planted block, in order A, B, C, D."""
    return [
        [f"G{i:03d}" for i in range(j * BLOCK, (j + 1) * BLOCK)]
        for j in range(4)
    ]


def _planted(seed: int = 0):
    """Three samples sharing blocks A/B/C; s1 also drives block D."""
    rng = np.random.default_rng(seed)
    names = [f"G{i:03d}" for i in range(N_GENES)]
    Xs, labels = [], []
    for s in range(N_SAMPLES):
        sample = f"s{s}"
        n_progs = 4 if sample == PRIVATE_SAMPLE else 3
        W = np.zeros((N_GENES, n_progs))
        for j in range(n_progs):
            W[j * BLOCK:(j + 1) * BLOCK, j] = rng.uniform(4.0, 9.0, BLOCK)
        H = rng.uniform(0.0, 0.2, size=(N_CELLS, n_progs))
        dominant = rng.integers(0, n_progs, N_CELLS)
        H[np.arange(N_CELLS), dominant] += rng.uniform(3.0, 7.0, N_CELLS)
        Xs.append(rng.poisson(H @ W.T).astype(np.float32))
        labels += [sample] * N_CELLS
    X = sp.csr_matrix(np.log1p(np.vstack(Xs)))
    return X, names, np.array(labels)


def _fit(**kw):
    X, names, labels = _planted()
    kw.setdefault("ks", (3, 4))
    kw.setdefault("seed", 0)
    return gene_nmf.multi_nmf(X, names, labels, **kw), names


def _which_block(genes, blocks):
    """Index of the planted block a gene set best matches, or None."""
    got = set(genes)
    for i, block in enumerate(blocks):
        if len(got & set(block)) >= 0.6 * len(got):
            return i
    return None


# ---------------------------------------------------------------------------
# multi_nmf() — one factorization per sample per k
# ---------------------------------------------------------------------------
def test_multi_nmf_runs_every_sample_at_every_k():
    fit, _ = _fit(ks=(3, 4))
    assert len(fit["labels"]) == N_SAMPLES * (3 + 4)
    assert fit["loadings"].shape == (N_GENES, N_SAMPLES * (3 + 4))


def test_program_labels_carry_the_sample_and_k_they_came_from():
    fit, _ = _fit(ks=(3,))
    assert fit["labels"][0] == "s0.k3.1"
    assert set(fit["samples"]) == {"s0", "s1", "s2"}
    assert fit["samples"].count("s0") == 3


def test_each_program_column_is_l1_normalized():
    """Meta-programs compare loadings across factorizations; unnormalized
    columns would make whichever sample had more counts dominate."""
    fit, _ = _fit()
    np.testing.assert_allclose(
        fit["loadings"].sum(axis=0), np.ones(len(fit["labels"])), rtol=1e-6
    )


def test_multi_nmf_skips_samples_with_too_few_cells():
    X, names, labels = _planted()
    # astype(object) first: the label array is <U2, so assigning "tiny" into
    # it would silently truncate to "ti".
    labels = labels.astype(object)
    labels[:5] = "tiny"          # 5 cells is below the floor
    fit = gene_nmf.multi_nmf(X, names, labels, ks=(3,), min_cells=10, seed=0)
    assert "tiny" not in fit["samples"]
    assert [s["sample"] for s in fit["skipped"]] == ["tiny"]
    assert fit["skipped"][0]["n_cells"] == 5


def test_multi_nmf_ignores_a_sample_category_with_no_cells():
    """Real .obs categoricals keep categories that no cell uses any more."""
    X, names, labels = _planted()
    fit = gene_nmf.multi_nmf(
        X, names, labels, ks=(3,), seed=0, samples=["s0", "s1", "s2", "unassigned"],
    )
    assert "unassigned" not in fit["samples"]


def test_multi_nmf_drops_k_that_exceeds_a_sample_size():
    X, names, labels = _planted()
    fit = gene_nmf.multi_nmf(X, names, labels, ks=(3, N_GENES + 10), seed=0)
    assert set(fit["ks_used"]) == {3}


def test_multi_nmf_is_deterministic():
    a, _ = _fit()
    b, _ = _fit()
    np.testing.assert_allclose(a["loadings"], b["loadings"])


def test_multi_nmf_rejects_a_label_array_of_the_wrong_length():
    X, names, _ = _planted()
    with pytest.raises(ValueError, match="sample_labels"):
        gene_nmf.multi_nmf(X, names, np.array(["a", "b"]), ks=(3,))


def test_multi_nmf_reports_progress_across_all_runs():
    seen = []
    _fit(progress_callback=lambda f, m: seen.append(f))
    assert seen == sorted(seen)
    assert seen[-1] <= 1.0


# ---------------------------------------------------------------------------
# meta_programs() — cluster programs across samples and k
# ---------------------------------------------------------------------------
def _meta(n_mp=4, **kw):
    fit, names = _fit()
    kw.setdefault("weight_explained", 0.8)
    return gene_nmf.meta_programs(fit, names, n_mp=n_mp, **kw), names


def test_meta_programs_recovers_every_planted_block():
    out, _ = _meta(n_mp=4)
    blocks = _blocks()
    matched = {_which_block(mp["genes"], blocks) for mp in out["metaprograms"]}
    assert {0, 1, 2, 3} <= matched


def test_shared_blocks_score_full_sample_coverage():
    out, _ = _meta(n_mp=4)
    blocks = _blocks()
    by_block = {_which_block(mp["genes"], blocks): mp for mp in out["metaprograms"]}
    for shared in (0, 1, 2):
        assert by_block[shared]["sample_coverage"] == pytest.approx(1.0)


def test_a_single_sample_block_is_flagged_by_low_coverage():
    """The whole point: a program only one sample shows must be separable from
    one every sample shows."""
    out, _ = _meta(n_mp=4)
    blocks = _blocks()
    by_block = {_which_block(mp["genes"], blocks): mp for mp in out["metaprograms"]}
    assert by_block[3]["sample_coverage"] == pytest.approx(1 / 3)


def test_metaprograms_are_ranked_by_coverage_then_silhouette():
    out, _ = _meta(n_mp=4)
    keys = [(mp["sample_coverage"], mp["silhouette"]) for mp in out["metaprograms"]]
    assert keys == sorted(keys, reverse=True)
    assert [mp["name"] for mp in out["metaprograms"]][:2] == ["MP1", "MP2"]


def test_similarity_matrix_is_symmetric_with_a_unit_diagonal():
    out, _ = _meta()
    S = np.asarray(out["similarity"])
    n = len(out["program_labels"])
    assert S.shape == (n, n)
    np.testing.assert_allclose(S, S.T, atol=1e-9)
    np.testing.assert_allclose(np.diag(S), np.ones(n), atol=1e-6)


def test_every_program_lands_in_exactly_one_metaprogram():
    out, _ = _meta(n_mp=4)
    clusters = np.asarray(out["clusters"])
    assert len(clusters) == len(out["program_labels"])
    assert set(np.unique(clusters)) <= set(range(len(out["metaprograms"])))


def test_composition_counts_programs_per_sample():
    out, _ = _meta(n_mp=4)
    blocks = _blocks()
    by_block = {_which_block(mp["genes"], blocks): mp for mp in out["metaprograms"]}
    private = by_block[3]
    comp = out["composition"][private["name"]]
    assert comp[PRIVATE_SAMPLE] > 0
    assert sum(v for s, v in comp.items() if s != PRIVATE_SAMPLE) == 0


def test_min_confidence_drops_genes_seen_in_only_a_few_programs():
    strict, _ = _meta(n_mp=4, min_confidence=0.99)
    loose, _ = _meta(n_mp=4, min_confidence=0.0)
    n_strict = sum(len(mp["genes"]) for mp in strict["metaprograms"])
    n_loose = sum(len(mp["genes"]) for mp in loose["metaprograms"])
    assert n_strict < n_loose


def test_jaccard_metric_also_recovers_the_blocks():
    out, _ = _meta(n_mp=4, metric="jaccard")
    blocks = _blocks()
    matched = {_which_block(mp["genes"], blocks) for mp in out["metaprograms"]}
    assert {0, 1, 2} <= matched


def test_meta_programs_rejects_an_unknown_metric():
    fit, names = _fit()
    with pytest.raises(ValueError, match="metric"):
        gene_nmf.meta_programs(fit, names, n_mp=3, metric="nope")


def test_meta_programs_rejects_more_metaprograms_than_programs():
    fit, names = _fit()
    with pytest.raises(ValueError, match="n_mp"):
        gene_nmf.meta_programs(fit, names, n_mp=len(fit["labels"]) + 1)


def test_metaprogram_weights_sum_to_one():
    """GeneNMF renormalizes the consensus so a weight reads as a share of the
    meta-program, independent of how many programs went into it."""
    out, _ = _meta(n_mp=4)
    for mp in out["metaprograms"]:
        assert sum(mp["weights"]) == pytest.approx(1.0)


def test_meta_program_result_is_json_safe():
    out, _ = _meta(n_mp=4)
    payload = {k: v for k, v in out.items() if k not in ("similarity", "linkage")}
    json.dumps(payload)


def test_similarity_rows_are_ordered_for_a_readable_heatmap():
    """The dendrogram leaf order groups a meta-program's programs together."""
    out, _ = _meta(n_mp=4)
    order = list(out["order"])
    assert sorted(order) == list(range(len(out["program_labels"])))
    clusters = np.asarray(out["clusters"])[order]
    # walking the ordered leaves, each cluster appears as one contiguous run
    runs = 1 + int(np.sum(clusters[1:] != clusters[:-1]))
    assert runs == len(set(clusters.tolist()))


# ---------------------------------------------------------------------------
# run_meta_programs() — end to end, including per-cell scores
# ---------------------------------------------------------------------------
def test_run_meta_programs_scores_every_cell_for_every_metaprogram():
    X, names, labels = _planted()
    out = gene_nmf.run_meta_programs(
        X, names, labels, ks=(3, 4), n_mp=4, seed=0, weight_explained=0.8,
    )
    scores = out["cell_scores"]
    assert scores.shape == (N_SAMPLES * N_CELLS, len(out["metaprograms"]))
    assert np.isfinite(scores).all()


def test_cell_scores_peak_on_the_cells_that_drive_the_program():
    X, names, labels = _planted()
    out = gene_nmf.run_meta_programs(
        X, names, labels, ks=(3, 4), n_mp=4, seed=0, weight_explained=0.8,
    )
    blocks = _blocks()
    dense = np.asarray(X.todense())
    for i, mp in enumerate(out["metaprograms"]):
        b = _which_block(mp["genes"], blocks)
        if b is None:
            continue
        idx = [names.index(g) for g in blocks[b]]
        planted = dense[:, idx].sum(axis=1)
        assert np.corrcoef(out["cell_scores"][:, i], planted)[0, 1] > 0.7


# ---------------------------------------------------------------------------
# the two consensus rules, tested where the planted fixture can't reach them
# ---------------------------------------------------------------------------
def test_specificity_is_scored_within_a_factorization_not_across_the_study():
    """A gene exclusive to one factor of *its own* model is specific, even if
    every sample's model has such a factor. Scoring across all programs at once
    would penalize exactly the recurrent programs meta-programs exist to find,
    and the penalty would deepen with every sample added.
    """
    # Two models of two factors. Gene 0 is exclusive within each model, but
    # loads on 2 of the 4 columns overall.
    W = np.zeros((3, 4))
    W[0, 0] = W[0, 2] = 1.0     # exclusive within model 0 and model 1
    W[1, 1] = W[1, 3] = 1.0
    W[2, :] = 0.25              # spread across every factor of both models
    fit = {"loadings": W, "model_index": np.array([0, 0, 1, 1])}

    per_model = gene_nmf.weighted_loadings(fit, 5.0)
    pooled = gene_nmf._specificity_weighted(W, 5.0)

    # Within its model gene 0 owns its factor outright, so it keeps its weight.
    assert per_model[0, 0] > 0.9
    # Pooled over all four columns it looks half-shared and is crushed.
    assert pooled[0, 0] < 0.5 * per_model[0, 0]


def test_trimmed_mean_drops_a_lone_outlier_program():
    """One runaway program should not carry a gene into the consensus."""
    n = 15                                   # 3-sigma can only fire when
    sub = np.zeros((2, n))                   # (n-1)/sqrt(n) > 3, i.e. n >= 12
    sub[0, :] = 1.0                          # gene 0: consistent across programs
    sub[1, 0] = 100.0                        # gene 1: one program, enormous
    avg = gene_nmf.trimmed_mean(sub)
    assert avg[0] == pytest.approx(1.0)
    assert avg[1] == pytest.approx(0.0), "the outlier program was not trimmed"


def test_trimmed_mean_is_a_plain_mean_below_three_programs():
    sub = np.array([[1.0, 3.0]])
    assert gene_nmf.trimmed_mean(sub)[0] == pytest.approx(2.0)


def _realistic_fit(n_models: int = 3, n_factors: int = 5, n_genes: int = 800, seed: int = 0):
    """Programs shaped like real factorizations, recurring across models.

    Each model sees the same ``n_factors`` templates with noise, so they
    cluster; within a model the loadings carry the heavy-tailed, low-specificity
    shape measured on real data (see _realistic_loadings in test_gene_nmf.py).
    Disjoint-block fixtures give every gene specificity 1.0, which makes the
    specificity weighting a no-op and hides what these defaults are tuned for.
    """
    rng = np.random.default_rng(seed)
    gene_scale = rng.normal(0.0, 1.5, size=(n_genes, 1))
    templates = np.exp(gene_scale + rng.normal(0.0, 0.9, size=(n_genes, n_factors)))
    cols, labels, samples, model_index = [], [], [], []
    for m in range(n_models):
        block = templates * np.exp(rng.normal(0.0, 0.25, size=templates.shape))
        block /= block.sum(axis=0, keepdims=True)
        for j in range(n_factors):
            cols.append(block[:, j])
            labels.append(f"s{m}.k{n_factors}.{j + 1}")
            samples.append(f"s{m}")
            model_index.append(m)
    return {
        "loadings": np.column_stack(cols),
        "labels": labels,
        "samples": samples,
        "ks": [n_factors] * len(labels),
        "model_index": np.asarray(model_index),
    }, [f"g{i:04d}" for i in range(n_genes)]


def test_default_meta_program_settings_produce_usable_gene_sets():
    """The consensus average smooths the specificity weighting enough that
    GeneNMF's exponent of 5 works here — but only paired with a higher
    weight_explained, because after weighting the top genes carry most of the
    mass and a 0.5 cutoff would keep a handful. Measured on 3 real limb
    sections x k=4..9: at (5, 0.5) two of eight meta-programs die and the rest
    hold ~7 genes; at (5, 0.8) all eight survive with the cleanest marker
    purity of any setting tried. See
    docs/measurements/2026-08-19-gene-nmf-metaprogram-defaults.md.
    """
    fit, names = _realistic_fit()
    out = gene_nmf.meta_programs(fit, names, n_mp=5)  # shipped defaults
    sizes = sorted(mp["n_genes"] for mp in out["metaprograms"])
    assert len(out["metaprograms"]) == 5, f"only {len(out['metaprograms'])} survived"
    assert sizes[len(sizes) // 2] >= 20, f"meta-programs are stunted: {sizes}"


def test_the_single_sample_weight_cutoff_would_stunt_meta_programs():
    """Why the two paths ship different weight_explained defaults."""
    fit, names = _realistic_fit()
    tight = gene_nmf.meta_programs(fit, names, n_mp=5, weight_explained=0.5)
    sizes = sorted(mp["n_genes"] for mp in tight["metaprograms"])
    assert sizes[len(sizes) // 2] < 20, sizes
