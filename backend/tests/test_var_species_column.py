"""Species annotation in .var: prefix detection, stripping, per-species counts."""
import numpy as np
import anndata
import pytest
from scipy.sparse import csr_matrix
from fastapi.testclient import TestClient

from xcell.adaptor import DataAdaptor
from xcell.main import app
from xcell.api import routes


def _adata():
    # 4 human (GRCh38_) + 3 mouse (mm10___) genes; integer-ish counts in .X.
    genes = ["GRCh38_A1BG", "GRCh38_TP53", "GRCh38_EGFR", "GRCh38_MYC",
             "mm10___Xkr4", "mm10___Sox2", "mm10___Actb"]
    rng = np.random.default_rng(1)
    X = rng.integers(0, 20, size=(10, 7)).astype(np.float32)
    ad = anndata.AnnData(X=csr_matrix(X))
    ad.var_names = genes
    return ad, X


def _big_adata(n_human=500, n_mouse=495, n_underscore_genes=5):
    """Realistically sized: two genomes plus a handful of real gene symbols
    that happen to contain an underscore (e.g. MT_ND1) — those must NOT be
    mistaken for a genome prefix."""
    genes = (
        [f"GRCh38_H{i}" for i in range(n_human)]
        + [f"mm10___M{i}" for i in range(n_mouse)]
        + [f"MT_ND{i}" for i in range(n_underscore_genes)]
    )
    rng = np.random.default_rng(0)
    X = rng.integers(0, 5, size=(6, len(genes))).astype(np.float32)
    ad = anndata.AnnData(X=csr_matrix(X))
    ad.var_names = genes
    return ad


# --------------------------------------------------------------- detection


def test_detect_finds_both_prefixes():
    ad, _ = _adata()
    a = DataAdaptor("x.h5ad", adata=ad)
    r = a.detect_species_prefixes()
    by_label = {p["label"]: p for p in r["prefixes"]}
    assert set(by_label) == {"GRCh38", "mm10"}
    assert by_label["GRCh38"]["prefix"] == "GRCh38_"
    assert by_label["GRCh38"]["n_genes"] == 4
    assert by_label["mm10"]["prefix"] == "mm10___"
    assert by_label["mm10"]["n_genes"] == 3
    assert r["n_unprefixed"] == 0


def test_detect_orders_by_gene_count_descending():
    ad, _ = _adata()
    a = DataAdaptor("x.h5ad", adata=ad)
    r = a.detect_species_prefixes()
    counts = [p["n_genes"] for p in r["prefixes"]]
    assert counts == sorted(counts, reverse=True)


def test_detect_ignores_rare_underscore_gene_symbols():
    a = DataAdaptor("x.h5ad", adata=_big_adata())
    r = a.detect_species_prefixes(min_fraction=0.01)
    labels = {p["label"] for p in r["prefixes"]}
    assert labels == {"GRCh38", "mm10"}, "MT_ND* must not look like a genome"
    assert r["n_unprefixed"] == 5


def test_detect_on_unprefixed_data_finds_nothing():
    ad = anndata.AnnData(X=csr_matrix(np.ones((4, 3), dtype=np.float32)))
    ad.var_names = ["Actb", "Sox2", "Xkr4"]
    a = DataAdaptor("x.h5ad", adata=ad)
    r = a.detect_species_prefixes()
    assert r["prefixes"] == []
    assert r["n_unprefixed"] == 3


# ------------------------------------------------------- .var species column


def test_add_species_column_labels_genes():
    ad, _ = _adata()
    a = DataAdaptor("x.h5ad", adata=ad)
    r = a.add_var_species_column()
    sp = a.adata.var["species"].astype(str).tolist()
    assert sp == ["GRCh38"] * 4 + ["mm10"] * 3
    assert r["species_column"] == "species"
    assert r["counts"] == {"GRCh38": 4, "mm10": 3}


def test_add_species_column_marks_unprefixed_unknown():
    a = DataAdaptor("x.h5ad", adata=_big_adata())
    a.add_var_species_column()
    sp = a.adata.var["species"].astype(str)
    assert (sp == "unknown").sum() == 5


def test_add_species_column_explicit_prefixes_and_labels():
    ad, _ = _adata()
    a = DataAdaptor("x.h5ad", adata=ad)
    r = a.add_var_species_column(prefixes="GRCh38_, mm10___", labels="human, mouse")
    assert r["counts"] == {"human": 4, "mouse": 3}
    assert a.adata.var["species"].astype(str).tolist()[0] == "human"


def test_add_species_column_is_idempotent_without_overwrite():
    ad, _ = _adata()
    a = DataAdaptor("x.h5ad", adata=ad)
    a.add_var_species_column(labels="human, mouse")
    # Second call must not silently relabel from detection.
    r = a.add_var_species_column()
    assert r["derived"] is False
    assert a.adata.var["species"].astype(str).tolist()[0] == "human"


def test_add_species_column_overwrite_rederives():
    ad, _ = _adata()
    a = DataAdaptor("x.h5ad", adata=ad)
    a.add_var_species_column(labels="human, mouse")
    r = a.add_var_species_column(overwrite=True)
    assert r["derived"] is True
    assert a.adata.var["species"].astype(str).tolist()[0] == "GRCh38"


def test_add_species_column_no_prefixes_raises_when_deriving():
    ad = anndata.AnnData(X=csr_matrix(np.ones((4, 3), dtype=np.float32)))
    ad.var_names = ["Actb", "Sox2", "Xkr4"]
    a = DataAdaptor("x.h5ad", adata=ad)
    with pytest.raises(ValueError, match="no species prefix"):
        a.add_var_species_column()


def test_add_species_column_rejects_non_matching_labels():
    ad, _ = _adata()
    a = DataAdaptor("x.h5ad", adata=ad)
    with pytest.raises(ValueError, match="labels"):
        a.add_var_species_column(prefixes="GRCh38_, mm10___", labels="human")


# --------------------------------------------------- counts from .var column


def test_sum_counts_by_species_uses_var_column():
    ad, X = _adata()
    a = DataAdaptor("x.h5ad", adata=ad)
    a.add_var_species_column()
    r = a.sum_counts_by_species()
    assert set(r["obs_columns"]) == {"GRCh38_counts", "mm10_counts"}
    assert np.allclose(a.adata.obs["GRCh38_counts"].to_numpy(), X[:, :4].sum(axis=1))
    assert np.allclose(a.adata.obs["mm10_counts"].to_numpy(), X[:, 4:].sum(axis=1))
    assert r["n_genes"] == {"GRCh38": 4, "mm10": 3}


def test_sum_counts_by_species_survives_prefix_strip():
    """The whole point: counting must not depend on gene-name prefixes."""
    ad, X = _adata()
    a = DataAdaptor("x.h5ad", adata=ad)
    a.add_var_species_column()
    a.rename_genes(pattern="^(GRCh38|mm10)_+")
    r = a.sum_counts_by_species()
    assert np.allclose(a.adata.obs["GRCh38_counts"].to_numpy(), X[:, :4].sum(axis=1))
    assert np.allclose(a.adata.obs["mm10_counts"].to_numpy(), X[:, 4:].sum(axis=1))
    assert r["n_genes"] == {"GRCh38": 4, "mm10": 3}


def test_sum_counts_by_species_skips_unknown_by_default():
    a = DataAdaptor("x.h5ad", adata=_big_adata())
    a.add_var_species_column()
    r = a.sum_counts_by_species()
    assert "unknown_counts" not in r["obs_columns"]
    r2 = a.sum_counts_by_species(include_unknown=True)
    assert "unknown_counts" in r2["obs_columns"]


def test_sum_counts_by_species_feeds_assign_species():
    ad, _ = _adata()
    a = DataAdaptor("x.h5ad", adata=ad)
    a.add_var_species_column()
    a.rename_genes(pattern="^(GRCh38|mm10)_+")
    r = a.sum_counts_by_species()
    out = a.assign_species(r["obs_columns"], obs_name="species_call", threshold=0.9)
    assert out["obs_name"] == "species_call"
    assert set(out["counts"]) <= {"GRCh38", "mm10", "mixed", "unassigned"}


def test_sum_counts_by_species_missing_column_raises():
    ad, _ = _adata()
    a = DataAdaptor("x.h5ad", adata=ad)
    with pytest.raises(ValueError, match="not found"):
        a.sum_counts_by_species()


def test_sum_counts_by_species_needs_two_species_to_be_useful():
    ad = anndata.AnnData(X=csr_matrix(np.ones((4, 3), dtype=np.float32)))
    ad.var_names = ["GRCh38_A", "GRCh38_B", "GRCh38_C"]
    a = DataAdaptor("x.h5ad", adata=ad)
    a.add_var_species_column()
    r = a.sum_counts_by_species()
    assert r["obs_columns"] == ["GRCh38_counts"]


# ------------------------------------------------------------------- routes


def test_route_detect_species_prefixes(monkeypatch):
    ad, _ = _adata()
    a = DataAdaptor("x.h5ad", adata=ad)
    monkeypatch.setattr(routes, "get_adaptor", lambda dataset=None: a)
    client = TestClient(app)
    resp = client.post("/api/scanpy/detect_species_prefixes", json={})
    assert resp.status_code == 200, resp.text
    assert len(resp.json()["prefixes"]) == 2


def test_route_add_var_species_column(monkeypatch):
    ad, _ = _adata()
    a = DataAdaptor("x.h5ad", adata=ad)
    monkeypatch.setattr(routes, "get_adaptor", lambda dataset=None: a)
    client = TestClient(app)
    resp = client.post("/api/scanpy/add_var_species_column", json={})
    assert resp.status_code == 200, resp.text
    assert resp.json()["counts"] == {"GRCh38": 4, "mm10": 3}


def test_route_sum_counts_by_species(monkeypatch):
    ad, _ = _adata()
    a = DataAdaptor("x.h5ad", adata=ad)
    a.add_var_species_column()
    monkeypatch.setattr(routes, "get_adaptor", lambda dataset=None: a)
    client = TestClient(app)
    resp = client.post("/api/scanpy/sum_counts_by_species", json={})
    assert resp.status_code == 200, resp.text
    assert set(resp.json()["obs_columns"]) == {"GRCh38_counts", "mm10_counts"}
