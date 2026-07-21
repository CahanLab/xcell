"""Gene-symbol find/replace: strip prefixes, rewrite names, guard duplicates."""
import numpy as np
import anndata
import pytest
from scipy.sparse import csr_matrix
from fastapi.testclient import TestClient

from xcell.adaptor import DataAdaptor
from xcell.main import app
from xcell.api import routes


def _adata(genes=None):
    genes = genes or ["GRCh38_A1BG", "GRCh38_TP53", "mm10___Xkr4", "mm10___Sox2"]
    X = csr_matrix(np.ones((5, len(genes)), dtype=np.float32))
    ad = anndata.AnnData(X=X)
    ad.var_names = genes
    return ad


# ------------------------------------------------------------------ stripping


def test_empty_replacement_strips_the_match():
    a = DataAdaptor("x.h5ad", adata=_adata())
    r = a.rename_genes(pattern="^mm10___")
    assert a.adata.var_names.tolist() == ["GRCh38_A1BG", "GRCh38_TP53", "Xkr4", "Sox2"]
    assert r["n_renamed"] == 2


def test_strips_either_species_prefix_in_one_pass():
    a = DataAdaptor("x.h5ad", adata=_adata())
    r = a.rename_genes(pattern="^(GRCh38|mm10)_+")
    assert a.adata.var_names.tolist() == ["A1BG", "TP53", "Xkr4", "Sox2"]
    assert r["n_renamed"] == 4


def test_replacement_string_is_substituted():
    a = DataAdaptor("x.h5ad", adata=_adata())
    r = a.rename_genes(pattern="^mm10___", replacement="mouse-")
    assert a.adata.var_names.tolist()[2:] == ["mouse-Xkr4", "mouse-Sox2"]
    assert r["n_renamed"] == 2


def test_regex_backreference_in_replacement():
    a = DataAdaptor("x.h5ad", adata=_adata())
    a.rename_genes(pattern=r"^(GRCh38|mm10)_+(.+)$", replacement=r"\2_\1")
    assert a.adata.var_names.tolist()[0] == "A1BG_GRCh38"


def test_literal_mode_does_not_interpret_metacharacters():
    a = DataAdaptor("x.h5ad", adata=_adata(["A.B", "AXB", "CD"]))
    r = a.rename_genes(pattern="A.B", replacement="Z", match_mode="literal")
    assert a.adata.var_names.tolist() == ["Z", "AXB", "CD"]
    assert r["n_renamed"] == 1


def test_literal_mode_replaces_every_occurrence():
    a = DataAdaptor("x.h5ad", adata=_adata(["x_x_x", "y"]))
    a.rename_genes(pattern="x", replacement="q", match_mode="literal")
    assert a.adata.var_names.tolist() == ["q_q_q", "y"]


def test_only_matching_genes_are_touched():
    a = DataAdaptor("x.h5ad", adata=_adata())
    a.rename_genes(pattern="^mm10___")
    assert a.adata.var_names.tolist()[:2] == ["GRCh38_A1BG", "GRCh38_TP53"]


# ------------------------------------------------------------------- guards


def test_no_match_raises():
    a = DataAdaptor("x.h5ad", adata=_adata())
    with pytest.raises(ValueError, match="no gene names match"):
        a.rename_genes(pattern="^ZZZ_")


def test_empty_pattern_raises():
    a = DataAdaptor("x.h5ad", adata=_adata())
    with pytest.raises(ValueError, match="pattern"):
        a.rename_genes(pattern="")


def test_invalid_regex_raises():
    a = DataAdaptor("x.h5ad", adata=_adata())
    with pytest.raises(ValueError, match="[Ii]nvalid regex"):
        a.rename_genes(pattern="^(unclosed")


def test_bad_match_mode_raises():
    a = DataAdaptor("x.h5ad", adata=_adata())
    with pytest.raises(ValueError, match="match_mode"):
        a.rename_genes(pattern="mm10", match_mode="fuzzy")


def test_refuses_to_create_duplicates():
    a = DataAdaptor("x.h5ad", adata=_adata(
        ["GRCh38_ACTB", "mm10___ACTB", "mm10___SOX2"]))
    with pytest.raises(ValueError, match="duplicate"):
        a.rename_genes(pattern="^(GRCh38|mm10)_+")
    # Refusal must leave the names untouched.
    assert a.adata.var_names.tolist() == [
        "GRCh38_ACTB", "mm10___ACTB", "mm10___SOX2"]


def test_make_unique_allows_duplicates():
    a = DataAdaptor("x.h5ad", adata=_adata(
        ["GRCh38_ACTB", "mm10___ACTB", "mm10___SOX2"]))
    r = a.rename_genes(pattern="^(GRCh38|mm10)_+", make_unique=True)
    assert len(set(a.adata.var_names)) == 3
    assert r["n_renamed"] == 3


# ------------------------------------------------------------ bookkeeping


def test_originals_are_preserved():
    a = DataAdaptor("x.h5ad", adata=_adata())
    a.rename_genes(pattern="^(GRCh38|mm10)_+")
    assert a.adata.var["gene_symbol_original"].tolist()[0] == "GRCh38_A1BG"


def test_repeat_rename_keeps_the_true_original():
    a = DataAdaptor("x.h5ad", adata=_adata())
    a.rename_genes(pattern="^(GRCh38|mm10)_+")
    a.rename_genes(pattern="^A1BG$", replacement="AAA")
    assert a.adata.var_names.tolist()[0] == "AAA"
    assert a.adata.var["gene_symbol_original"].tolist()[0] == "GRCh38_A1BG"


def test_rename_clears_stale_normalized_cache():
    a = DataAdaptor("x.h5ad", adata=_adata())
    a._normalized_adata = a.adata.copy()
    a.rename_genes(pattern="^mm10___")
    assert a._normalized_adata is None


def test_reports_examples_of_what_changed():
    a = DataAdaptor("x.h5ad", adata=_adata())
    r = a.rename_genes(pattern="^mm10___")
    assert {"before": "mm10___Xkr4", "after": "Xkr4"} in r["examples"]


def test_species_column_survives_rename():
    """Renaming must not disturb .var annotations — that is the whole point
    of keeping species in a column rather than in the gene name."""
    a = DataAdaptor("x.h5ad", adata=_adata())
    a.add_var_species_column()
    a.rename_genes(pattern="^(GRCh38|mm10)_+")
    assert a.adata.var["species"].astype(str).tolist() == [
        "GRCh38", "GRCh38", "mm10", "mm10"]
    assert a.sum_counts_by_species()["n_genes"] == {"GRCh38": 2, "mm10": 2}


# ------------------------------------------- index-name / export collisions


def _writes_cleanly(a):
    """True if the adaptor's export actually round-trips to disk."""
    import tempfile, os
    path = tempfile.mktemp(suffix=".h5ad")
    try:
        a.prepare_export_with_lines().write_h5ad(path)
        return True
    finally:
        if os.path.exists(path):
            os.remove(path)


def test_rename_after_index_swap_still_exports():
    """swap_var_index names the index after a column and drops that column;
    a later rename must not re-create the column under the index's name, or
    h5ad export dies on 'index.name is also used by a column'."""
    a = DataAdaptor("x.h5ad", adata=_adata())
    a.rename_genes(pattern="^(GRCh38|mm10)_+")
    a.swap_var_index("gene_symbol_original")
    a.rename_genes(pattern="^(GRCh38|mm10)_+")
    assert a.adata.var.index.name != "gene_symbol_original"
    assert _writes_cleanly(a)


def test_rename_keeps_index_name_when_it_does_not_collide():
    a = DataAdaptor("x.h5ad", adata=_adata())
    a.adata.var.index.name = "gene_ids"
    a.rename_genes(pattern="^mm10___")
    assert a.adata.var.index.name == "gene_ids"
    assert _writes_cleanly(a)


def test_export_survives_preexisting_index_name_collision():
    """Sessions that already hit this bug must still be able to export."""
    a = DataAdaptor("x.h5ad", adata=_adata())
    a.adata.var["gene_symbol_original"] = [f"orig_{n}" for n in a.adata.var_names]
    a.adata.var.index.name = "gene_symbol_original"
    assert _writes_cleanly(a)
    # The column's values must survive the fix untouched.
    assert a.adata.var["gene_symbol_original"].tolist()[0] == "orig_GRCh38_A1BG"


def test_plain_rename_then_export_round_trips():
    a = DataAdaptor("x.h5ad", adata=_adata())
    a.rename_genes(pattern="^(GRCh38|mm10)_+")
    assert _writes_cleanly(a)


# ------------------------------------------------------------------- routes


def test_route_rename_genes(monkeypatch):
    a = DataAdaptor("x.h5ad", adata=_adata())
    monkeypatch.setattr(routes, "get_adaptor", lambda dataset=None: a)
    client = TestClient(app)
    resp = client.post("/api/scanpy/rename_genes",
                       json={"pattern": "^(GRCh38|mm10)_+", "replacement": ""})
    assert resp.status_code == 200, resp.text
    assert resp.json()["n_renamed"] == 4
    assert a.adata.var_names.tolist()[0] == "A1BG"


def test_route_rename_genes_duplicate_is_400(monkeypatch):
    a = DataAdaptor("x.h5ad", adata=_adata(["GRCh38_ACTB", "mm10___ACTB"]))
    monkeypatch.setattr(routes, "get_adaptor", lambda dataset=None: a)
    client = TestClient(app)
    resp = client.post("/api/scanpy/rename_genes",
                       json={"pattern": "^(GRCh38|mm10)_+"})
    assert resp.status_code == 400
    assert "duplicate" in resp.json()["detail"].lower()


def test_route_rename_genes_no_match_is_400(monkeypatch):
    a = DataAdaptor("x.h5ad", adata=_adata())
    monkeypatch.setattr(routes, "get_adaptor", lambda dataset=None: a)
    client = TestClient(app)
    resp = client.post("/api/scanpy/rename_genes", json={"pattern": "^ZZZ"})
    assert resp.status_code == 400
