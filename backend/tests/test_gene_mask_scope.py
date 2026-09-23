"""What the active `.var` gene mask applies to.

The mask started as a Gene-Panel view filter — browse, search, gene-set score
aggregation — and every analysis ignored it. That made it a lie: a heatmap of
a gene set drew rows the panel said were hidden, and marker genes returned
genes the user had deliberately put out of scope.

The rule these tests pin down: **the mask is the visible gene universe for
anything that reports or displays genes, and is ignored by operations that
build the dataset's cell-space structure or hand genes to another dataset** —
PCA/neighbours/UMAP, spot merging, and the Localize reference bundle. Those
keep their own explicit gene-subset controls, and a session-only view must not
silently change an embedding that gets written into the file.
"""

import anndata
import numpy as np
import pandas as pd
import pytest
from scipy.sparse import csr_matrix

from xcell.adaptor import DataAdaptor
from xcell.heatmap import compute_heatmap_data


N_CELLS, N_GENES = 60, 10
GENES = [f"g{i:02d}" for i in range(N_GENES)]
VISIBLE = GENES[:6]      # keep_columns=['panel'] leaves these
HIDDEN = GENES[6:]


def _adata():
    rng = np.random.default_rng(0)
    t = np.linspace(0.0, 1.0, N_CELLS)
    X = 3.0 + rng.normal(0, 0.5, size=(N_CELLS, N_GENES))
    for j in range(N_GENES):
        X[:, j] += (1.0 + 0.4 * j) * t        # every gene tracks position
    X = np.maximum(X, 0.0).astype(np.float32)

    ad = anndata.AnnData(X=csr_matrix(X))
    ad.var_names = GENES
    ad.var["panel"] = [g in VISIBLE for g in GENES]
    ad.var["highly_variable"] = [i % 2 == 0 for i in range(N_GENES)]
    ad.obs["grp"] = pd.Categorical(["a" if i % 2 else "b" for i in range(N_CELLS)])
    ad.obsm["X_pca"] = np.column_stack([t * 4 - 2, t * 4 - 2])
    return ad


def _adaptor(masked=True):
    a = DataAdaptor("x.h5ad", adata=_adata())
    if masked:
        a.set_gene_mask(keep_columns=["panel"], hide_columns=[])
    return a


# ---------- the resolver is where the universe narrows ----------

def test_no_subset_means_every_visible_gene():
    a = _adaptor()
    mask, kind, meta = a._resolve_gene_mask(None)

    assert sorted(a.adata.var_names[mask]) == sorted(VISIBLE)
    assert meta["n_genes"] == len(VISIBLE)
    assert meta["n_hidden_by_gene_mask"] == len(HIDDEN)
    assert kind == "all"


def test_a_column_subset_is_intersected_with_the_mask():
    a = _adaptor()
    mask, _, meta = a._resolve_gene_mask("highly_variable")

    expected = [g for g in VISIBLE if GENES.index(g) % 2 == 0]
    assert sorted(a.adata.var_names[mask]) == sorted(expected)
    assert meta["n_genes"] == len(expected)


def test_an_explicit_gene_list_is_intersected_with_the_mask():
    a = _adaptor()
    asked = [VISIBLE[0], HIDDEN[0], "not_a_gene"]
    mask, _, meta = a._resolve_gene_mask(asked)

    assert list(a.adata.var_names[mask]) == [VISIBLE[0]]
    assert meta["n_hidden_by_gene_mask"] == 1
    # Genes the dataset never had stay a separate report from masked ones.
    assert meta["genes_missing"] == ["not_a_gene"]


def test_a_fully_hidden_subset_is_a_clear_error():
    a = _adaptor()
    with pytest.raises(ValueError, match="gene mask"):
        a._resolve_gene_mask(HIDDEN)


def test_no_mask_leaves_the_resolver_untouched():
    a = _adaptor(masked=False)
    mask, _, meta = a._resolve_gene_mask(None)

    assert mask.sum() == N_GENES
    assert "n_hidden_by_gene_mask" not in meta


def test_structure_building_callers_opt_out():
    a = _adaptor()
    mask, _, meta = a._resolve_gene_mask(None, apply_visible_mask=False)

    assert mask.sum() == N_GENES
    assert "n_hidden_by_gene_mask" not in meta


# ---------- the operations that report genes ----------

def test_marker_genes_never_name_a_hidden_gene():
    a = _adaptor()
    r = a.run_marker_genes(obs_column="grp", top_n=N_GENES)

    named = {g["gene"] for grp in r["results"] for g in grp["genes"]}
    assert named
    assert named <= set(VISIBLE)
    assert r["n_genes_tested"] == len(VISIBLE)


def test_diffexp_never_names_a_hidden_gene():
    a = _adaptor()
    g1 = list(range(0, 30))
    g2 = list(range(30, N_CELLS))
    r = a.run_diffexp(group1_indices=g1, group2_indices=g2, top_n=N_GENES)

    named = {g["gene"] for side in ("positive", "negative") for g in r[side]}
    assert named
    assert named <= set(VISIBLE)
    assert r["n_genes_tested"] == len(VISIBLE)


def test_line_association_tests_only_visible_genes():
    a = _adaptor()
    a.set_lines([{"name": "L", "embeddingName": "X_pca",
                  "points": [[-2.0, -2.0], [2.0, 2.0]]}])
    r = a.test_line_association("L")

    assert r["diagnostics"]["n_genes_tested"] == len(VISIBLE)
    assert {g["gene"] for g in r["all_genes"]} <= set(VISIBLE)
    assert r["gene_subset"]["n_hidden_by_gene_mask"] == len(HIDDEN)


# ---------- the operations that build structure ----------

def test_pca_ignores_the_mask():
    """A session-only view must not quietly change an embedding in the file."""
    masked = _adaptor()
    masked.run_pca(n_comps=3)
    plain = _adaptor(masked=False)
    plain.run_pca(n_comps=3)

    assert np.allclose(
        np.abs(masked.adata.obsm["X_pca"]), np.abs(plain.adata.obsm["X_pca"])
    )


# ---------- the heatmap ----------

def _row(result, gene):
    return np.array(result["matrix"][result["row_labels"].index(gene)])


def test_heatmap_drops_hidden_genes_and_says_how_many():
    a = _adaptor()
    r = compute_heatmap_data(a, genes=GENES)

    assert r["row_labels"] == VISIBLE
    assert r["n_genes_hidden"] == len(HIDDEN)


def test_heatmap_keeps_row_groups_aligned_after_hiding():
    a = _adaptor()
    groups = [
        {"name": "keep", "genes": VISIBLE[:3]},
        {"name": "mixed", "genes": [VISIBLE[3], HIDDEN[0]]},
    ]
    r = compute_heatmap_data(a, genes=VISIBLE[:4] + [HIDDEN[0]],
                             gene_set_groups=groups)

    assert len(r["row_groups"]) == len(r["row_labels"])
    assert r["row_labels"] == VISIBLE[:4]
    assert r["row_groups"] == ["keep", "keep", "keep", "mixed"]


def test_heatmap_aggregate_uses_only_visible_genes():
    a = _adaptor()
    groups = [{"name": "mixed", "genes": [VISIBLE[0], HIDDEN[0]]}]
    masked = compute_heatmap_data(a, genes=[VISIBLE[0], HIDDEN[0]],
                                  gene_set_groups=groups, aggregate_gene_sets=True)
    only_visible = compute_heatmap_data(
        _adaptor(masked=False), genes=[VISIBLE[0]],
        gene_set_groups=[{"name": "mixed", "genes": [VISIBLE[0]]}],
        aggregate_gene_sets=True)

    assert masked["row_labels"] == ["mixed"]
    assert np.allclose(masked["matrix"], only_visible["matrix"])


def test_heatmap_with_every_gene_hidden_returns_empty_rather_than_raising():
    a = _adaptor()
    r = compute_heatmap_data(a, genes=HIDDEN)

    assert r["row_labels"] == []
    assert r["matrix"] == []
    assert r["n_genes_hidden"] == len(HIDDEN)


def test_heatmap_reports_zero_hidden_when_no_mask_is_active():
    r = compute_heatmap_data(_adaptor(masked=False), genes=GENES)

    assert r["row_labels"] == GENES
    assert r["n_genes_hidden"] == 0
