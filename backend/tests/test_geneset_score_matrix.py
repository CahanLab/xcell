"""Batch gene-set folder scoring into an .obsm matrix, plus obsm column read
and embedding-from-two-columns."""

import numpy as np
import anndata
import pytest
from scipy.sparse import csr_matrix

from xcell.adaptor import DataAdaptor


def _adata(n_cells=50, n_genes=12):
    rng = np.random.default_rng(1)
    X = csr_matrix(rng.integers(0, 10, size=(n_cells, n_genes)).astype(np.float32))
    ad = anndata.AnnData(X=X)
    ad.var_names = [f"g{i}" for i in range(n_genes)]
    return ad


SETS = [
    {"name": "progenitorState", "genes": ["g0", "g1", "g2"]},
    {"name": "differentiatedState", "genes": ["g3", "g4", "g5"]},
    {"name": "cellCycle", "genes": ["g6", "g7"]},
]


def _adaptor():
    return DataAdaptor("x.h5ad", adata=_adata())


# --- score_gene_sets_matrix -------------------------------------------------

def test_matrix_shape_and_registry():
    a = _adaptor()
    res = a.score_gene_sets_matrix(SETS, per_gene_norm="none", aggregation="mean",
                                   obsm_name="scores")
    assert res["obsm_name"] == "scores"
    assert res["columns"] == ["progenitorState", "differentiatedState", "cellCycle"]
    assert a.adata.obsm["scores"].shape == (a.n_cells, 3)
    # column names recorded in .uns registry so unnamed obsm cols resolve back
    reg = a.adata.uns["xcell_score_matrices"]["scores"]
    assert reg["columns"] == ["progenitorState", "differentiatedState", "cellCycle"]
    assert reg["aggregation"] == "mean"


def test_column_values_match_single_set_scoring():
    a = _adaptor()
    a.score_gene_sets_matrix(SETS, per_gene_norm="zscore_mad", aggregation="mean",
                             obsm_name="scores")
    mat = a.adata.obsm["scores"]
    for j, s in enumerate(SETS):
        expected = a._aggregate_gene_set_scores(
            s["genes"], a.adata.X, per_gene_norm="zscore_mad", aggregation="mean")
        np.testing.assert_allclose(mat[:, j], expected, rtol=1e-6, atol=1e-6)


def test_gene_mask_excludes_masked_genes():
    a = _adaptor()
    # Only g0..g4 visible → 'cellCycle' (g6,g7) is fully masked and dropped.
    a._visible_gene_mask = np.array([True] * 5 + [False] * 7)
    res = a.score_gene_sets_matrix(SETS, per_gene_norm="none", aggregation="mean",
                                   obsm_name="scores")
    assert res["columns"] == ["progenitorState", "differentiatedState"]
    assert a.adata.obsm["scores"].shape == (a.n_cells, 2)
    assert any(s["name"] == "cellCycle" for s in res["skipped"])
    # differentiatedState kept g3,g4 (g5 masked) → excluded count reported
    diff = res["per_column"]["differentiatedState"]
    assert diff["n_masked_excluded"] == 1
    # its values should match scoring with only the visible genes
    expected = a._aggregate_gene_set_scores(["g3", "g4"], a.adata.X,
                                            per_gene_norm="none", aggregation="mean")
    np.testing.assert_allclose(a.adata.obsm["scores"][:, 1], expected, rtol=1e-6, atol=1e-6)


def test_collision_requires_overwrite():
    a = _adaptor()
    a.score_gene_sets_matrix(SETS, obsm_name="scores")
    with pytest.raises(ValueError):
        a.score_gene_sets_matrix(SETS, obsm_name="scores")
    # overwrite replaces (e.g. fewer sets → fewer columns)
    res = a.score_gene_sets_matrix(SETS[:1], obsm_name="scores", overwrite=True)
    assert res["columns"] == ["progenitorState"]
    assert a.adata.obsm["scores"].shape == (a.n_cells, 1)


def test_all_sets_empty_raises():
    a = _adaptor()
    with pytest.raises(ValueError):
        a.score_gene_sets_matrix([{"name": "none", "genes": ["nope1", "nope2"]}],
                                 obsm_name="scores")


# --- get_obsm_column --------------------------------------------------------

def test_get_obsm_column_by_name():
    a = _adaptor()
    a.score_gene_sets_matrix(SETS, per_gene_norm="none", obsm_name="scores")
    out = a.get_obsm_column("scores", "differentiatedState")
    np.testing.assert_allclose(out["values"], a.adata.obsm["scores"][:, 1], rtol=1e-6, atol=1e-6)
    assert out["min"] <= out["max"]


def test_get_obsm_column_unknown_raises():
    a = _adaptor()
    a.score_gene_sets_matrix(SETS, obsm_name="scores")
    with pytest.raises((KeyError, ValueError)):
        a.get_obsm_column("scores", "nonexistent")


# --- create_obsm_embedding --------------------------------------------------

def test_embedding_from_two_columns():
    a = _adaptor()
    a.score_gene_sets_matrix(SETS, per_gene_norm="none", obsm_name="scores")
    res = a.create_obsm_embedding("scores", "progenitorState", "differentiatedState",
                                  name="prog_vs_diff")
    key = res["embedding_name"]
    assert key in a.adata.obsm
    emb = a.adata.obsm[key]
    assert emb.shape == (a.n_cells, 2)
    np.testing.assert_allclose(emb[:, 0], a.adata.obsm["scores"][:, 0], rtol=1e-6, atol=1e-6)
    np.testing.assert_allclose(emb[:, 1], a.adata.obsm["scores"][:, 1], rtol=1e-6, atol=1e-6)
    # shows up as a selectable embedding in the schema
    assert key in a.get_schema()["embeddings"]


def test_schema_lists_score_matrix_columns():
    a = _adaptor()
    a.score_gene_sets_matrix(SETS, obsm_name="scores")
    schema = a.get_schema()
    assert schema["score_matrices"]["scores"] == [
        "progenitorState", "differentiatedState", "cellCycle"]
