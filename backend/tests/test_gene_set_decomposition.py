"""Decomposing a gene set into expression programs, and the coherence diagnostic.

Two planted patterns: block A (12 genes) follows pattern 1, block B (6 genes)
follows an independent pattern 2, plus flat filler genes. The set A ∪ B is
exactly the collagen case — one name, two expression configurations — so a
rank-2 decomposition has a known right answer, and A alone is one pattern.
"""
import time
import types

import anndata
import numpy as np
import pandas as pd
import pytest
from scipy.sparse import csr_matrix
from fastapi.testclient import TestClient

from xcell import gene_coexpression as gc
from xcell import gene_set_decomposition as gsd
from xcell.adaptor import DataAdaptor


N_CELLS = 240
A = [f"A{i:02d}" for i in range(12)]
B = [f"B{i:02d}" for i in range(6)]
FILLER = [f"F{i:02d}" for i in range(10)]


def _planted(seed=0):
    rng = np.random.default_rng(seed)
    p1 = rng.gamma(2.0, 2.0, N_CELLS)
    p2 = rng.gamma(2.0, 2.0, N_CELLS)
    cols = []
    for _ in A:
        cols.append(p1 * rng.uniform(1.0, 2.0) + rng.normal(0, 0.3, N_CELLS))
    for _ in B:
        cols.append(p2 * rng.uniform(1.0, 2.0) + rng.normal(0, 0.3, N_CELLS))
    for _ in FILLER:
        cols.append(rng.uniform(0.5, 1.5, N_CELLS))
    X = np.clip(np.vstack(cols).T, 0, None)
    return X.astype(np.float32)


def _adata():
    rng = np.random.default_rng(1)
    X = _planted()
    counts = rng.poisson(X * 3).astype(np.float32)
    ad = anndata.AnnData(X=csr_matrix(counts))
    ad.var_names = A + B + FILLER
    ad.obs["group"] = pd.Categorical(["x" if i % 2 else "y" for i in range(N_CELLS)])
    # On a 28-gene matrix normalize_total couples every gene into one
    # compositional pattern that real data does not have (the co-expression
    # tests dodge it the same way), so coherence is measured on raw counts.
    ad.layers["raw"] = ad.X.copy()
    return ad


def _adaptor():
    return DataAdaptor("x.h5ad", adata=_adata())


# --- pure: coherence ------------------------------------------------------------

def test_set_coherence_separates_one_pattern_from_two():
    X = _planted()
    genes = A + B + FILLER
    gi = {g: i for i, g in enumerate(genes)}
    one = gc.set_coherence(X[:, [gi[g] for g in A]].T, metric="pearson")
    two = gc.set_coherence(X[:, [gi[g] for g in A + B]].T, metric="pearson")
    assert one["n_genes"] == 12 and one["eigengene_pve"] > 0.7
    assert two["eigengene_pve"] < 0.7 and two["eigen_pve"][1] > 0.2
    assert one["suggested_k"] == 1 and two["suggested_k"] == 2
    assert one["one_pattern"] is True and one["n_significant"] == 1
    assert two["one_pattern"] is False and two["n_significant"] == 2
    assert 0 < two["noise_edge_pve"] < two["eigen_pve"][1]
    assert 0.5 < two["signal_share_top"] < 0.8
    assert 0 <= two["mean_abs_corr"] <= 1 and len(two["eigen_pve"]) <= 10
    assert abs(sum(two["eigen_pve"]) - 1.0) < 1e-6 or sum(two["eigen_pve"]) < 1.0
    import json
    json.dumps(two)


def test_set_coherence_single_gene_and_zero_variance():
    X = _planted()
    out = gc.set_coherence(X[:, :1].T, metric="pearson")
    assert out["eigengene_pve"] == 1.0 and out["mean_abs_corr"] is None and out["one_pattern"] is True
    flat = np.ones((3, 50))
    out = gc.set_coherence(flat, metric="pearson")
    assert out["eigengene_pve"] == 0.0


# --- pure: programs -----------------------------------------------------------

def test_pca_programs_recover_the_two_blocks_with_signed_lists():
    X = np.log1p(_planted())
    genes = A + B + FILLER
    out = gsd.pca_programs(X, genes, k=2, loading_threshold=0.3, seed=0)
    assert out["scores"].shape == (N_CELLS, 2) and out["loadings"].shape == (len(genes), 2)
    assert len(out["variance_ratio"]) == 2 and 0 < sum(out["variance_ratio"]) <= 1.0
    assert [p["name"] for p in out["programs"]] == ["PC1", "PC2"] and out["programs"][0]["index"] == 0
    # Which block lands on which component depends on noise; each block must own one.
    pa = max(out["programs"], key=lambda p: len(set(p["genes"]) & set(A)))
    pb = max(out["programs"], key=lambda p: len(set(p["genes"]) & set(B)))
    assert pa is not pb
    assert len(set(pa["genes"]) & set(A)) >= 10 and not (set(pa["genes"]) & set(B))
    assert len(set(pb["genes"]) & set(B)) >= 5 and not (set(pb["genes"]) & set(A))
    p1, p2 = out["programs"]
    # the leading gene of every component loads positively (sign convention)
    for p in out["programs"]:
        assert p["genes"] and p["weights"][0] > 0
        assert len(p["genes"]) == len(p["weights"]) and len(p["genes_down"]) == len(p["weights_down"])
    # filler genes carry no program
    assert not (set(p1["genes"]) | set(p2["genes"])) & set(FILLER)


def test_nmf_programs_recover_the_two_blocks():
    X = np.log1p(_planted())
    genes = A + B + FILLER
    out = gsd.nmf_programs(X, genes, k=2, seed=0, max_genes=len(genes))
    assert out["scores"].shape[0] == N_CELLS and out["loadings"].shape[0] == len(genes)
    assert [p["name"] for p in out["programs"]] == ["F1", "F2"]
    top = [set(p["genes"][:6]) for p in out["programs"]]
    assert any(len(t & set(A)) >= 5 for t in top) and any(len(t & set(B)) >= 4 for t in top)
    assert all(p["genes_down"] == [] for p in out["programs"])


def test_pca_programs_rejects_bad_k():
    X = np.log1p(_planted())
    with pytest.raises(ValueError):
        gsd.pca_programs(X, A + B + FILLER, k=0)


# --- adaptor -------------------------------------------------------------------

def _run(a, genes=A + B, **kw):
    kw.setdefault("key", "matrix")
    compute_fn, apply_fn = a.prepare_gene_set_decomposition(genes, **kw)
    return apply_fn(compute_fn(lambda frac, message=None: None))


def test_adaptor_pca_writes_score_matrix_loadings_and_registry():
    a = _adaptor()
    out = _run(a, method="pca", k=2)
    ad = a.adata
    assert out["key"] == "matrix" and out["method"] == "pca" and out["obsm_key"] == "matrix"
    assert out["program_names"] == ["PC1", "PC2"]
    assert out["n_genes_used"] == 18 and out["genes_missing"] == [] and out["n_cells"] == N_CELLS
    assert ad.obsm["matrix"].shape == (N_CELLS, 2) and not np.isnan(ad.obsm["matrix"]).any()
    assert ad.uns["xcell_score_matrices"]["matrix"]["columns"] == ["PC1", "PC2"]
    assert ad.uns["xcell_score_matrices"]["matrix"]["source"] == "gene_set_decomposition"
    L = ad.varm["matrix_loadings"]
    assert L.shape == (ad.n_vars, 2)
    filler_rows = [list(ad.var_names).index(g) for g in FILLER]
    assert np.all(L[filler_rows] == 0)
    reg = ad.uns["xcell_gene_set_decomposition"]["matrix"]
    assert reg["method"] == "pca" and reg["genes_used"] == A + B
    assert set(reg["programs"]) == {"PC1", "PC2"}
    assert "genes" in reg["programs"]["PC1"] and "genes_down" in reg["programs"]["PC1"]
    assert reg["coherence"]["eigengene_pve"] < 0.9
    assert a.get_schema()["score_matrices"]["matrix"] == ["PC1", "PC2"]
    assert a._action_history[-1]["action"] == "gene_set_decomposition"
    pa = max(out["programs"], key=lambda p: len(set(p["genes"]) & set(A)))
    assert len(set(pa["genes"]) & set(A)) >= 10
    import json
    json.dumps(out)


def test_adaptor_nmf_branch_and_scope_and_overwrite():
    a = _adaptor()
    idx = list(range(150))
    out = _run(a, method="nmf", k=2, cell_indices=idx)
    assert out["program_names"] == ["F1", "F2"] and out["n_cells"] == 150
    m = a.adata.obsm["matrix"]
    assert np.isnan(m[150:]).all() and not np.isnan(m[:150]).any()
    with pytest.raises(ValueError, match="already exists"):
        a.prepare_gene_set_decomposition(A + B, key="matrix", method="pca", k=2)
    out = _run(a, method="pca", k=2, overwrite=True)
    assert out["program_names"] == ["PC1", "PC2"]
    assert a.adata.uns["xcell_gene_set_decomposition"]["matrix"]["method"] == "pca"


def test_adaptor_validation_and_clamping():
    a = _adaptor()
    with pytest.raises(ValueError, match="at least 2"):
        a.prepare_gene_set_decomposition(["A00"], key="k", method="pca")
    with pytest.raises(ValueError, match="method"):
        a.prepare_gene_set_decomposition(A, key="k", method="ica")
    out = _run(a, genes=A[:3] + ["NOPE"], key="tiny", method="pca", k=10)
    assert out["k"] == 2 and out["genes_missing"] == ["NOPE"]


def test_adaptor_coherence():
    a = _adaptor()
    one = a.gene_set_coherence(A, layer="raw")
    two = a.gene_set_coherence(A + B + ["NOPE"], cell_indices=list(range(200)), layer="raw")
    assert one["eigengene_pve"] > two["eigengene_pve"]
    assert two["n_genes_used"] == 18 and two["genes_missing"] == ["NOPE"] and two["n_cells"] == 200
    assert two["suggested_k"] == 2 and two["one_pattern"] is False and one["one_pattern"] is True


def test_readers():
    a = _adaptor()
    assert a.list_gene_set_decompositions() == []
    _run(a, method="pca", k=2)
    runs = a.list_gene_set_decompositions()
    assert runs[0]["key"] == "matrix" and runs[0]["n_programs"] == 2
    full = a.get_gene_set_decomposition("matrix")
    assert full["programs"][0]["name"] == "PC1" and full["coherence"]["n_genes"] == 18
    with pytest.raises(KeyError):
        a.get_gene_set_decomposition("nope")


# --- routes ----------------------------------------------------------------------

def _poll(client, task_id, timeout=60.0):
    t0 = time.time()
    while time.time() - t0 < timeout:
        st = client.get(f"/api/tasks/{task_id}").json()
        if st["status"] in ("completed", "error", "cancelled"):
            return st
        time.sleep(0.05)
    raise AssertionError("task did not finish")


def test_routes(monkeypatch):
    from xcell.main import app
    from xcell.api import routes
    a = _adaptor()
    monkeypatch.setattr(routes, "get_adaptor", lambda dataset=None: a)
    client = TestClient(app)
    r = client.post("/api/gene_sets/coherence", json={"genes": A + B, "cell_context": "all", "layer": "raw"})
    assert r.status_code == 200, r.text
    assert r.json()["suggested_k"] >= 2 and "eigen_pve" in r.json()
    r = client.post("/api/gene_sets/decompose", json={
        "genes": A + B, "key": "matrix", "method": "pca", "k": 2,
        "cell_context": "selection", "cell_indices": list(range(120)),
    })
    assert r.status_code == 202, r.text
    st = _poll(client, r.json()["task_id"])
    assert st["status"] == "completed", st
    assert st["result"]["n_cells"] == 120 and st["result"]["program_names"] == ["PC1", "PC2"]
    r = client.get("/api/gene_sets/decompositions")
    assert r.status_code == 200 and r.json()["runs"][0]["key"] == "matrix"
    r = client.get("/api/gene_sets/decompositions/matrix")
    assert r.status_code == 200 and r.json()["method"] == "pca"
    assert client.get("/api/gene_sets/decompositions/nope").status_code == 404
    r = client.post("/api/gene_sets/decompose", json={"genes": ["A00"], "key": "x", "method": "pca"})
    assert r.status_code == 400


def test_codegen_entry():
    from xcell import codegen
    spec = codegen.REGISTRY["gene_set_decomposition"]
    step = types.SimpleNamespace(action="gene_set_decomposition", params={
        "genes": A, "key": "matrix", "method": "pca", "k": 2, "loading_threshold": 0.2,
        "layer": None, "transform": "log1p", "seed": 0,
    }, result={"program_names": ["PC1", "PC2"], "obsm_key": "matrix"}, selection=None, n_active=None, n_total=None)
    text = "\n".join(spec.code(step))
    assert "prepare_gene_set_decomposition" in text and "method='pca'" in text
    assert "2 programs" in spec.summary(step.params, step.result)
