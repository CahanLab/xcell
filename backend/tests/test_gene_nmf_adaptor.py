"""Adaptor tests for NMF gene-program discovery.

The AnnData carries three planted gene blocks driven by three cell
populations, so a rank-3 run has a known right answer and the assertions can
check what landed in .obsm / .varm / .uns rather than just the return value.
"""
import json

import anndata
import numpy as np
import pandas as pd
import pytest
from scipy.sparse import csr_matrix

from xcell.adaptor import DataAdaptor


N_CELLS, N_GENES, K = 150, 30, 3
BLOCK = N_GENES // K


def _adata(n_extra_genes: int = 0):
    """Three disjoint co-expressed gene blocks + optional flat filler genes."""
    rng = np.random.default_rng(0)
    W = np.zeros((N_GENES, K))
    for j in range(K):
        W[j * BLOCK:(j + 1) * BLOCK, j] = rng.uniform(4.0, 9.0, BLOCK)
    H = rng.uniform(0.0, 0.2, size=(N_CELLS, K))
    dominant = rng.integers(0, K, N_CELLS)
    H[np.arange(N_CELLS), dominant] += rng.uniform(2.0, 6.0, N_CELLS)
    counts = rng.poisson(H @ W.T).astype(np.float32)

    names = [f"G{i:03d}" for i in range(N_GENES)]
    if n_extra_genes:
        filler = rng.poisson(0.5, size=(N_CELLS, n_extra_genes)).astype(np.float32)
        counts = np.hstack([counts, filler])
        names += [f"F{i:03d}" for i in range(n_extra_genes)]

    ad = anndata.AnnData(X=csr_matrix(counts))
    ad.var_names = names
    ad.obs["group"] = pd.Categorical(
        ["a" if d == 0 else "b" for d in dominant]
    )
    ad.var["highly_variable"] = [True] * N_GENES + [False] * n_extra_genes
    ad.obsm["spatial"] = rng.random((N_CELLS, 2)) * 100
    return ad


def _adaptor(**kw) -> DataAdaptor:
    return DataAdaptor("x.h5ad", adata=_adata(**kw))


def _run(a: DataAdaptor, **kw):
    kw.setdefault("k", K)
    kw.setdefault("weight_explained", 0.8)
    compute_fn, apply_fn = a.prepare_gene_nmf(**kw)
    return apply_fn(compute_fn(lambda frac, message=None: None))


def _blocks():
    return [
        set(f"G{i:03d}" for i in range(j * BLOCK, (j + 1) * BLOCK))
        for j in range(K)
    ]


# ---------------------------------------------------------------------------
# validation happens synchronously, in prepare_*
# ---------------------------------------------------------------------------
def test_prepare_rejects_k_below_two():
    a = _adaptor()
    with pytest.raises(ValueError, match="k"):
        a.prepare_gene_nmf(k=1)


def test_prepare_rejects_k_above_the_gene_count():
    a = _adaptor()
    with pytest.raises(ValueError, match="k"):
        a.prepare_gene_nmf(k=N_GENES + 1)


def test_prepare_rejects_an_unknown_gene_subset_column():
    a = _adaptor()
    with pytest.raises(ValueError, match="nope"):
        a.prepare_gene_nmf(k=K, gene_subset="nope")


def test_prepare_rejects_an_unknown_layer():
    a = _adaptor()
    with pytest.raises(ValueError, match="[Ll]ayer"):
        a.prepare_gene_nmf(k=K, layer="missing")


def test_prepare_rejects_negative_expression():
    """Scaled/centered data is the common foot-gun; it must fail up front with
    an explanation rather than deep inside a background task."""
    ad = _adata()
    ad.layers["scaled"] = ad.X.toarray() - 5.0
    a = DataAdaptor("x.h5ad", adata=ad)
    with pytest.raises(ValueError, match="non-negative"):
        a.prepare_gene_nmf(k=K, layer="scaled", transform="none")


def test_prepare_rejects_an_existing_key_without_overwrite():
    a = _adaptor()
    _run(a, key="NMF")
    with pytest.raises(ValueError, match="overwrite"):
        a.prepare_gene_nmf(k=K, key="NMF")


def test_prepare_allows_replacing_a_key_with_overwrite():
    a = _adaptor()
    _run(a, key="NMF")
    out = _run(a, key="NMF", overwrite=True)
    assert out["obsm_key"] == "NMF"


def test_prepare_rejects_an_empty_cell_selection():
    a = _adaptor()
    with pytest.raises(ValueError, match="[Cc]ell"):
        a.prepare_gene_nmf(k=K, cell_indices=[])


def test_prepare_rejects_a_subset_smaller_than_k():
    a = _adaptor()
    with pytest.raises(ValueError, match="k"):
        a.prepare_gene_nmf(k=K, cell_indices=[0, 1])


# ---------------------------------------------------------------------------
# what the run writes into AnnData
# ---------------------------------------------------------------------------
def test_run_writes_a_registered_score_matrix():
    a = _adaptor()
    out = _run(a, key="NMF")
    assert a.adata.obsm["NMF"].shape == (N_CELLS, out["n_programs"])
    reg = a.adata.uns["xcell_score_matrices"]["NMF"]
    assert reg["columns"] == [p["name"] for p in out["programs"]]
    assert reg["source"] == "gene_nmf"


def test_score_matrix_columns_appear_in_the_schema():
    """Without the registry entry the programs are invisible to the UI."""
    a = _adaptor()
    out = _run(a, key="NMF")
    schema = a.get_schema()
    assert schema["score_matrices"]["NMF"] == [p["name"] for p in out["programs"]]
    assert "NMF" in schema["embeddings"]


def test_run_writes_full_length_gene_loadings_to_varm():
    a = _adaptor(n_extra_genes=8)
    out = _run(a, key="NMF", gene_subset="highly_variable")
    loadings = a.adata.varm["NMF_loadings"]
    assert loadings.shape == (N_GENES + 8, out["n_programs"])
    # genes outside the fitted subset carry no loading at all
    assert np.all(loadings[N_GENES:] == 0.0)
    assert np.all(loadings >= 0.0)


def test_run_records_the_programs_and_params_in_uns():
    a = _adaptor()
    out = _run(a, key="NMF", seed=4)
    rec = a.adata.uns["xcell_gene_nmf"]["NMF"]
    assert rec["params"]["k"] == K
    assert rec["params"]["seed"] == 4
    assert list(rec["program_names"]) == [p["name"] for p in out["programs"]]
    first = out["programs"][0]
    assert list(rec["programs"][first["name"]]["genes"]) == first["genes"]


def test_stored_record_survives_an_h5ad_round_trip(tmp_path):
    """uns has to hold plain dicts/lists — a list of dicts will not write."""
    a = _adaptor()
    out = _run(a, key="NMF")
    path = tmp_path / "out.h5ad"
    a.adata.write_h5ad(path)
    back = anndata.read_h5ad(path)
    rec = back.uns["xcell_gene_nmf"]["NMF"]
    assert list(rec["program_names"]) == [p["name"] for p in out["programs"]]
    assert "NMF_loadings" in back.varm
    assert back.obsm["NMF"].shape == (N_CELLS, out["n_programs"])


def test_run_recovers_the_planted_gene_blocks():
    a = _adaptor()
    out = _run(a, key="NMF")
    planted = _blocks()
    assert out["n_programs"] == K
    matched = set()
    for prog in out["programs"]:
        got = set(prog["genes"])
        hits = [i for i, block in enumerate(planted) if len(got & block) >= 0.6 * len(got)]
        assert hits, f"program {sorted(got)} matches no planted block"
        matched.update(hits)
    assert matched == {0, 1, 2}


def test_cell_scores_are_highest_for_the_population_driving_the_program():
    a = _adaptor()
    out = _run(a, key="NMF")
    scores = np.asarray(a.adata.obsm["NMF"])
    # The program whose genes are block 0 should peak on the cells that use it.
    block0 = _blocks()[0]
    which = [
        i for i, p in enumerate(out["programs"])
        if len(set(p["genes"]) & block0) >= 0.6 * len(p["genes"])
    ]
    assert which, "no program recovered block 0"
    col = scores[:, which[0]]
    # Compare against what the model actually saw (log-normalized), not raw
    # counts — library-size differences would blur the correlation.
    norm = a.normalized_adata[:, sorted(block0)].X
    expr = np.asarray(norm.sum(axis=1)).ravel()
    assert np.corrcoef(col, expr)[0, 1] > 0.8


# ---------------------------------------------------------------------------
# gene and cell subsetting
# ---------------------------------------------------------------------------
def test_gene_subset_column_restricts_the_fit_to_those_genes():
    a = _adaptor(n_extra_genes=12)
    out = _run(a, key="NMF", gene_subset="highly_variable")
    assert out["n_genes_used"] == N_GENES
    used = {g for p in out["programs"] for g in p["genes"]}
    assert not any(g.startswith("F") for g in used)


def test_gene_subset_accepts_an_explicit_gene_list():
    a = _adaptor()
    genes = [f"G{i:03d}" for i in range(2 * BLOCK)]
    out = _run(a, key="NMF", k=2, gene_subset=genes)
    assert out["n_genes_used"] == len(genes)


def test_cell_subset_leaves_excluded_cells_unscored():
    """Cells the model never saw get NaN, not a zero that reads as 'program
    absent here'."""
    a = _adaptor()
    idx = list(range(0, N_CELLS, 2))
    out = _run(a, key="NMF", cell_indices=idx)
    assert out["n_cells_used"] == len(idx)
    scores = np.asarray(a.adata.obsm["NMF"])
    assert not np.isnan(scores[idx]).any()
    excluded = [i for i in range(N_CELLS) if i not in set(idx)]
    assert np.isnan(scores[excluded]).all()


# ---------------------------------------------------------------------------
# result shape and bookkeeping
# ---------------------------------------------------------------------------
def test_result_is_json_serializable():
    a = _adaptor()
    out = _run(a, key="NMF")
    json.dumps(out)


def test_stored_record_round_trips_through_get_gene_nmf_result():
    a = _adaptor()
    _run(a, key="NMF")
    got = a.get_gene_nmf_result("NMF")
    assert got is not None
    json.dumps(got)
    assert [p["name"] for p in got["programs"]]


def test_get_gene_nmf_result_returns_none_for_an_unknown_key():
    a = _adaptor()
    assert a.get_gene_nmf_result("nope") is None


def test_run_logs_an_action():
    a = _adaptor()
    _run(a, key="NMF")
    actions = [entry["action"] for entry in a._action_history]
    assert "gene_nmf" in actions


def test_two_keys_coexist_without_clobbering_each_other():
    a = _adaptor()
    _run(a, key="NMF")
    _run(a, key="NMF_alt", seed=9)
    assert set(a.adata.uns["xcell_gene_nmf"]) == {"NMF", "NMF_alt"}
    assert "NMF" in a.adata.obsm and "NMF_alt" in a.adata.obsm


def test_progress_is_reported_during_compute():
    a = _adaptor()
    seen = []
    compute_fn, _ = a.prepare_gene_nmf(k=K, weight_explained=0.8)
    compute_fn(lambda frac, message=None: seen.append(frac))
    assert seen and seen[-1] == 1.0


def test_compute_does_not_mutate_anndata():
    """compute_fn must be side-effect free — only apply_fn writes."""
    a = _adaptor()
    compute_fn, _ = a.prepare_gene_nmf(k=K, key="NMF", weight_explained=0.8)
    before_obsm = set(a.adata.obsm)
    before_uns = set(a.adata.uns)
    compute_fn(lambda frac, message=None: None)
    assert set(a.adata.obsm) == before_obsm
    assert set(a.adata.uns) == before_uns
