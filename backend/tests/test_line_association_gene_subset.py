"""Restricting line-association testing to a chosen set of genes.

The engine has always accepted an explicit gene list; these pin down what the
UI needs on top of it — that the subset actually narrows the multiple-testing
burden, and that genes the set asks for but the dataset doesn't have are
*reported* rather than silently dropped.
"""

import time

import anndata
import numpy as np
import pytest
from fastapi.testclient import TestClient
from scipy.sparse import csr_matrix

from xcell.main import app
from xcell.api import routes
from xcell.adaptor import DataAdaptor


N_CELLS, N_GENES = 60, 12
UP_GENES = ["g00", "g01", "g02", "g03"]
DOWN_GENES = ["g04", "g05", "g06", "g07"]
FLAT_GENES = ["g08", "g09", "g10", "g11"]
LINE = [[-2.0, -2.0], [2.0, 2.0]]


def _adata():
    """Cells strung along the y=x diagonal, with genes that track position.

    Effect sizes are graded (R^2 ~0.25 to ~0.5) rather than overwhelming, so
    p-values land in a measurable range instead of underflowing to exactly 0 —
    otherwise "correcting over 8 genes beats correcting over 12" is untestable.
    """
    rng = np.random.default_rng(0)
    t = np.linspace(0.0, 1.0, N_CELLS)
    coords = np.column_stack([
        t * 4.0 - 2.0 + rng.normal(0, 0.02, N_CELLS),
        t * 4.0 - 2.0 + rng.normal(0, 0.02, N_CELLS),
    ])

    amplitudes = [2.0, 2.5, 3.0, 3.5]
    X = 3.0 + rng.normal(0.0, 1.0, size=(N_CELLS, N_GENES))
    for j, amp in enumerate(amplitudes):
        X[:, j] += amp * t                 # g00..g03 increase along the line
        X[:, j + 4] += amp * (1.0 - t)     # g04..g07 decrease
    # g08..g11 stay noise-only
    X = np.maximum(X, 0.0)

    ad = anndata.AnnData(X=csr_matrix(X.astype(np.float32)))
    ad.var_names = [f"g{i:02d}" for i in range(N_GENES)]
    ad.var["highly_variable"] = [i < 6 for i in range(N_GENES)]
    ad.var["expressed"] = [i % 2 == 0 for i in range(N_GENES)]
    ad.obsm["X_pca"] = coords
    return ad


def _adaptor():
    a = DataAdaptor("x.h5ad", adata=_adata())
    a.set_lines([{"name": "L", "embeddingName": "X_pca", "points": LINE}])
    return a


def _run(a, **kwargs):
    return a.test_line_association("L", **kwargs)


# ---------- gene list restricts what is tested ----------

def test_gene_list_restricts_genes_tested():
    a = _adaptor()
    subset = ["g00", "g04", "g09"]
    r = _run(a, gene_subset=subset)

    assert r["diagnostics"]["n_genes_tested"] == 3
    tested = {g["gene"] for g in r["all_genes"]}
    assert tested == set(subset)


def test_gene_list_narrows_fdr_burden():
    """The point of a focused set: fewer tests, so borderline genes survive FDR."""
    a = _adaptor()
    all_genes = _run(a)
    subset = _run(a, gene_subset=UP_GENES + DOWN_GENES)

    fdr_all = {g["gene"]: g["fdr"] for g in all_genes["all_genes"]}
    fdr_sub = {g["gene"]: g["fdr"] for g in subset["all_genes"]}
    # Same raw p-values, corrected over 8 genes instead of 12.
    assert all(fdr_sub[g] <= fdr_all[g] + 1e-12 for g in fdr_sub)
    assert any(fdr_sub[g] < fdr_all[g] for g in fdr_sub)


def test_gene_list_results_never_leak_genes_outside_the_set():
    a = _adaptor()
    r = _run(a, gene_subset=UP_GENES, cluster_genes=True)
    inside = set(UP_GENES)
    assert {g["gene"] for g in r["positive"]} <= inside
    assert {g["gene"] for g in r["negative"]} <= inside
    for mod in r["modules"]:
        assert {g["gene"] for g in mod["genes"]} <= inside


# ---------- what the caller is told about the subset ----------

def test_result_reports_the_resolved_subset():
    a = _adaptor()
    r = _run(a, gene_subset=["g00", "g01"])
    assert r["gene_subset"] == {
        "type": "gene_list",
        "n_genes": 2,
        "n_requested": 2,
        "genes_missing": [],
    }


def test_missing_genes_are_reported_not_silently_dropped():
    a = _adaptor()
    r = _run(a, gene_subset=["g00", "NOPE1", "g01", "NOPE2"])
    gs = r["gene_subset"]
    assert gs["type"] == "gene_list"
    assert gs["n_genes"] == 2
    assert gs["n_requested"] == 4
    assert gs["genes_missing"] == ["NOPE1", "NOPE2"]
    assert r["diagnostics"]["n_genes_tested"] == 2


def test_result_reports_subset_when_no_subset_given():
    a = _adaptor()
    r = _run(a)
    assert r["gene_subset"] == {
        "type": "all",
        "n_genes": N_GENES,
        "n_requested": None,
        "genes_missing": [],
    }


def test_result_reports_subset_for_a_var_column():
    a = _adaptor()
    r = _run(a, gene_subset="highly_variable")
    assert r["gene_subset"]["type"] == "column:highly_variable"
    assert r["gene_subset"]["n_genes"] == 6


def test_result_reports_subset_for_combined_columns():
    a = _adaptor()
    r = _run(a, gene_subset={"columns": ["highly_variable", "expressed"], "operation": "intersection"})
    # g00, g02, g04 — even-indexed among the first six
    assert r["gene_subset"]["n_genes"] == 3
    assert r["gene_subset"]["type"] == "intersection:highly_variable+expressed"


# ---------- failure modes the UI has to be able to explain ----------

def test_gene_list_with_no_overlap_raises():
    a = _adaptor()
    with pytest.raises(ValueError, match="None of the specified genes"):
        _run(a, gene_subset=["NOPE1", "NOPE2"])


def test_empty_gene_list_raises_a_clear_error():
    a = _adaptor()
    with pytest.raises(ValueError, match="[Ee]mpty gene list"):
        _run(a, gene_subset=[])


# ---------- multi-line pooling takes the same subset ----------

def test_multi_line_association_accepts_a_gene_list():
    a = _adaptor()
    a.set_lines([
        {"name": "L", "embeddingName": "X_pca", "points": LINE},
        {"name": "M", "embeddingName": "X_pca", "points": LINE},
    ])
    half = list(range(N_CELLS // 2))
    other = list(range(N_CELLS // 2, N_CELLS))
    r = a.test_multi_line_association(
        lines=[
            {"name": "L", "cell_indices": half},
            {"name": "M", "cell_indices": other},
        ],
        gene_subset=["g00", "g04", "NOPE"],
    )
    assert r["diagnostics"]["n_genes_tested"] == 2
    assert r["gene_subset"] == {
        "type": "gene_list",
        "n_genes": 2,
        "n_requested": 3,
        "genes_missing": ["NOPE"],
    }


# ---------- routes: the request models must accept every subset shape ----------

def _install():
    a = DataAdaptor("x.h5ad", adata=_adata())
    a.set_lines([{"name": "L", "embeddingName": "X_pca", "points": LINE}])
    routes.set_adaptor(a, slot="primary")
    return a


def _poll(client, task_id, timeout=30.0):
    deadline = time.time() + timeout
    while time.time() < deadline:
        r = client.get(f"/api/tasks/{task_id}").json()
        if r["status"] in ("completed", "error", "cancelled"):
            return r
        time.sleep(0.05)
    raise AssertionError("task did not finish")


def _post(client, path, body):
    r = client.post(path, json=body)
    assert r.status_code == 202, r.text
    done = _poll(client, r.json()["task_id"])
    assert done["status"] == "completed", done
    return done["result"]


def test_route_accepts_a_gene_list_subset():
    _install()
    c = TestClient(app)
    result = _post(c, "/api/lines/association", {
        "line_name": "L",
        "gene_subset": ["g00", "g04"],
    })
    assert result["diagnostics"]["n_genes_tested"] == 2
    assert result["gene_subset"]["type"] == "gene_list"


def test_route_accepts_combined_columns_subset():
    """Regression: the request model rejected the dict the UI sends for 2+ columns."""
    _install()
    c = TestClient(app)
    result = _post(c, "/api/lines/association", {
        "line_name": "L",
        "gene_subset": {"columns": ["highly_variable", "expressed"], "operation": "union"},
    })
    assert result["diagnostics"]["n_genes_tested"] == 9  # 6 HVG ∪ 6 even-indexed


def test_multi_line_route_accepts_combined_columns_subset():
    _install()
    c = TestClient(app)
    half = list(range(N_CELLS // 2))
    other = list(range(N_CELLS // 2, N_CELLS))
    result = _post(c, "/api/lines/multi-association", {
        "lines": [
            {"name": "L", "cell_indices": half},
            {"name": "L", "cell_indices": other},
        ],
        "gene_subset": {"columns": ["highly_variable", "expressed"], "operation": "intersection"},
    })
    assert result["diagnostics"]["n_genes_tested"] == 3


def test_route_reports_missing_genes_in_the_response():
    _install()
    c = TestClient(app)
    result = _post(c, "/api/lines/association", {
        "line_name": "L",
        "gene_subset": ["g00", "NOPE"],
    })
    assert result["gene_subset"]["genes_missing"] == ["NOPE"]
    assert result["gene_subset"]["n_requested"] == 2


def test_route_empty_gene_list_is_a_400_not_a_failed_task():
    _install()
    c = TestClient(app)
    r = c.post("/api/lines/association", json={"line_name": "L", "gene_subset": []})
    assert r.status_code == 400
    assert "empty gene list" in r.json()["detail"].lower()
