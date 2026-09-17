"""Gene maps: similarity channels, their combination, embedding, modules, and storage.

Three planted gene blocks (as in the NMF tests) give expression similarity a
known answer; hand-built memberships and edges give the annotation and STRING
channels exact values.
"""
import time
import types

import anndata
import numpy as np
import pandas as pd
import pytest
from scipy.sparse import csr_matrix
from fastapi.testclient import TestClient

from xcell import gene_similarity as gs
from xcell.adaptor import DataAdaptor


N_CELLS, N_GENES, K = 150, 30, 3
BLOCK = N_GENES // K
GENES = [f"G{i:03d}" for i in range(N_GENES)]
BLOCK_OF = [i // BLOCK for i in range(N_GENES)]


def _adata():
    rng = np.random.default_rng(0)
    W = np.zeros((N_GENES, K))
    for j in range(K):
        W[j * BLOCK:(j + 1) * BLOCK, j] = rng.uniform(4.0, 9.0, BLOCK)
    H = rng.uniform(0.0, 0.2, size=(N_CELLS, K))
    dominant = rng.integers(0, K, N_CELLS)
    H[np.arange(N_CELLS), dominant] += rng.uniform(2.0, 6.0, N_CELLS)
    counts = rng.poisson(H @ W.T).astype(np.float32)
    ad = anndata.AnnData(X=csr_matrix(counts))
    ad.var_names = GENES
    ad.var["highly_variable"] = [True] * 20 + [False] * 10
    ad.layers["raw"] = ad.X.copy()
    return ad


def _adaptor():
    return DataAdaptor("x.h5ad", adata=_adata())


def _X_genes():
    return np.asarray(_adata().layers["raw"].todense()).T  # genes × cells


# --- channels --------------------------------------------------------------------

def test_annotation_similarity_is_cosine_over_memberships_case_insensitive():
    S, n_terms = gs.annotation_similarity(["A", "B", "C", "D"], {"s1": ["a", "b"], "s2": ["B", "C"], "s3": ["c"]})
    assert S.shape == (4, 4) and np.allclose(np.diag(S), 1.0)
    assert S[0, 1] == pytest.approx(1 / np.sqrt(2)) and S[1, 0] == S[0, 1]
    assert S[0, 2] == 0.0
    assert S[1, 2] == pytest.approx(0.5)
    assert n_terms == [1, 2, 2, 0]
    assert S[3].sum() == pytest.approx(1.0)  # unannotated gene: self only


def test_string_similarity_places_edge_scores_symmetrically():
    S, n_edges = gs.string_similarity(["A", "B", "C"], [{"a": "a", "b": "B", "score": 0.9}, {"a": "X", "b": "C", "score": 0.5}])
    assert S[0, 1] == pytest.approx(0.9) and S[1, 0] == pytest.approx(0.9)
    assert S[0, 2] == 0.0 and np.allclose(np.diag(S), 1.0)
    assert n_edges == 1


def test_expression_similarity_is_unit_interval_and_block_structured():
    S = gs.expression_similarity(_X_genes(), metric="pearson")
    assert S.shape == (N_GENES, N_GENES) and S.min() >= 0.0 and S.max() <= 1.0
    within = np.mean([S[i, j] for i in range(N_GENES) for j in range(N_GENES) if i != j and BLOCK_OF[i] == BLOCK_OF[j]])
    between = np.mean([S[i, j] for i in range(N_GENES) for j in range(N_GENES) if BLOCK_OF[i] != BLOCK_OF[j]])
    assert within > between + 0.2


def test_combine_is_a_weighted_mean_over_present_channels():
    a = np.array([[1.0, 0.2], [0.2, 1.0]])
    b = np.array([[1.0, 0.8], [0.8, 1.0]])
    S, used = gs.combine({"expression": (a, 1.0), "annotation": (b, 3.0), "string": (None, 1.0), "zero": (b, 0.0)})
    assert S[0, 1] == pytest.approx((0.2 * 1 + 0.8 * 3) / 4)
    assert used == {"expression": 0.25, "annotation": 0.75}
    with pytest.raises(ValueError, match="No similarity channel"):
        gs.combine({"expression": (None, 1.0)})


# --- embedding, modules, order ------------------------------------------------

def test_modules_recover_planted_blocks_and_embedding_has_shape():
    from sklearn.metrics import adjusted_rand_score
    S = gs.expression_similarity(_X_genes(), metric="pearson")
    labels = gs.modules(S, n_neighbors=8, resolution=1.0, seed=0)
    assert len(labels) == N_GENES and adjusted_rand_score(BLOCK_OF, labels) >= 0.9
    coords = gs.embed(S, method="umap", n_neighbors=8, seed=0)
    assert coords.shape == (N_GENES, 2) and np.isfinite(coords).all()
    coords = gs.embed(S, method="mds", seed=0)
    assert coords.shape == (N_GENES, 2) and np.isfinite(coords).all()
    order = gs.leaf_order(S)
    assert sorted(order) == list(range(N_GENES))
    # the heatmap order keeps each module contiguous, largest module first
    mo = gs.module_order(S, labels)
    assert sorted(mo) == list(range(N_GENES))
    runs = [labels[i] for i in mo]
    seen: list[int] = []
    for m in runs:
        if not seen or seen[-1] != m:
            assert m not in seen, "module split into non-contiguous runs"
            seen.append(m)
    sizes = [labels.count(m) for m in seen]
    assert sizes == sorted(sizes, reverse=True)
    with pytest.raises(ValueError, match="embedding"):
        gs.embed(S, method="tsne")


def test_build_gene_map_returns_serialisable_summary():
    import json
    S_expr = gs.expression_similarity(_X_genes(), metric="pearson")
    out = gs.build_gene_map(GENES, channels={"expression": (S_expr, 1.0)}, n_neighbors=8, resolution=1.0, embedding="mds", seed=0)
    assert out["genes"] == GENES and out["coords"].shape == (N_GENES, 2)
    assert len(out["modules"]) == N_GENES and out["n_modules"] >= 2
    assert out["similarity"].dtype == np.float32
    summary = {k: v for k, v in out.items() if k not in ("coords", "similarity", "modules", "order")}
    json.dumps(summary)
    assert summary["channel_weights"] == {"expression": 1.0}


# --- adaptor ---------------------------------------------------------------------

def _run(a, **kw):
    kw.setdefault("key", "map")
    compute_fn, apply_fn = a.prepare_gene_map(**kw)
    return apply_fn(compute_fn(lambda frac, message=None: None))


def test_adaptor_expression_only_map_is_stored_and_readable():
    from sklearn.metrics import adjusted_rand_score
    a = _adaptor()
    out = _run(a, genes=GENES, layer="raw", expression_metric="pearson", annotation_weight=0.0, n_neighbors=8, embedding="mds")
    assert out["key"] == "map" and out["n_genes"] == N_GENES and out["genes_missing"] == []
    assert out["channel_weights"] == {"expression": 1.0}
    assert adjusted_rand_score(BLOCK_OF, a.adata.uns["xcell_gene_maps"]["map"]["modules"]) >= 0.9
    rec = a.adata.uns["xcell_gene_maps"]["map"]
    assert rec["coords"].shape == (N_GENES, 2) and rec["similarity"].shape == (N_GENES, N_GENES)
    assert a._action_history[-1]["action"] == "gene_map"
    runs = a.list_gene_maps()
    assert runs[0]["key"] == "map" and runs[0]["n_genes"] == N_GENES
    full = a.get_gene_map("map")
    assert full["genes"] == GENES and len(full["coords"]) == N_GENES and "similarity" not in full
    full = a.get_gene_map("map", include_similarity=True)
    assert len(full["similarity"]) == N_GENES and len(full["similarity"][0]) == N_GENES
    import json
    json.dumps(full)
    with pytest.raises(KeyError):
        a.get_gene_map("nope")


def test_adaptor_gene_subset_and_annotation_channel(monkeypatch):
    from xcell import gene_set_sources as gss
    fake_lib = {"source": "msigdb", "id": "m2.cgp", "species": "mouse", "name": "Curated",
                "sets": [{"name": "block0", "genes": GENES[:10]}, {"name": "block1", "genes": GENES[10:20]}]}
    monkeypatch.setattr(gss, "find_library", lambda source, library_id, species=None: fake_lib if library_id == "m2.cgp" else None)
    a = _adaptor()
    out = _run(a, gene_subset="highly_variable", layer="raw", expression_metric="pearson",
               annotation_libraries=[{"source": "msigdb", "id": "m2.cgp", "species": "mouse"}],
               annotation_weight=1.0, n_neighbors=6, embedding="mds")
    assert out["n_genes"] == 20 and set(out["channel_weights"]) == {"expression", "annotation"}
    assert out["channels"]["annotation"]["n_genes_annotated"] == 20
    with pytest.raises(ValueError, match="not been fetched"):
        a.prepare_gene_map(genes=GENES, key="x", annotation_libraries=[{"source": "msigdb", "id": "nope"}])
    with pytest.raises(ValueError, match="at least 3"):
        a.prepare_gene_map(genes=GENES[:2], key="x")
    with pytest.raises(ValueError, match="already exists"):
        a.prepare_gene_map(genes=GENES, key="map")


def test_adaptor_string_channel_uses_fetched_edges(monkeypatch):
    from xcell import gene_set_sources as gss
    calls = {}

    def fake_network(genes, species, *, required_score=400):
        calls["n"] = len(genes); calls["species"] = species
        return {"edges": [{"a": GENES[0], "b": GENES[1], "score": 0.95}], "species": species, "n_genes": len(genes)}
    monkeypatch.setattr(gss, "string_network", fake_network)
    a = _adaptor()
    out = _run(a, genes=GENES, layer="raw", expression_metric="pearson", annotation_weight=0.0,
               string_weight=0.5, string_species="mouse", n_neighbors=8, embedding="mds")
    assert calls == {"n": N_GENES, "species": "mouse"}
    assert set(out["channel_weights"]) == {"expression", "string"}
    assert out["channels"]["string"]["n_edges"] == 1


# --- routes ------------------------------------------------------------------------

def _poll(client, task_id, timeout=90.0):
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
    r = client.post("/api/gene_map/run", json={
        "genes": GENES, "key": "map", "layer": "raw", "expression_metric": "pearson",
        "annotation_weight": 0.0, "n_neighbors": 8, "embedding": "mds", "cell_context": "all",
    })
    assert r.status_code == 202, r.text
    st = _poll(client, r.json()["task_id"])
    assert st["status"] == "completed", st
    assert st["result"]["n_genes"] == N_GENES
    r = client.get("/api/gene_map")
    assert r.status_code == 200 and r.json()["runs"][0]["key"] == "map"
    r = client.get("/api/gene_map/map")
    assert r.status_code == 200 and len(r.json()["coords"]) == N_GENES and "similarity" not in r.json()
    r = client.get("/api/gene_map/map?similarity=true")
    assert len(r.json()["similarity"]) == N_GENES
    assert client.get("/api/gene_map/nope").status_code == 404
    r = client.post("/api/gene_map/run", json={"genes": GENES[:2], "key": "x"})
    assert r.status_code == 400
    # gene_subset as a column-combination spec must not 422
    r = client.post("/api/gene_map/run", json={"gene_subset": {"columns": ["highly_variable"], "operation": "union"},
                                               "key": "map2", "layer": "raw", "annotation_weight": 0.0, "n_neighbors": 6, "embedding": "mds"})
    assert r.status_code == 202, r.text
    assert _poll(client, r.json()["task_id"])["result"]["n_genes"] == 20


def test_codegen_entry():
    from xcell import codegen
    spec = codegen.REGISTRY["gene_map"]
    step = types.SimpleNamespace(action="gene_map", params={"genes": GENES[:5], "key": "map", "expression_weight": 1.0,
                                                             "annotation_weight": 1.0, "string_weight": 0.0, "n_neighbors": 15,
                                                             "resolution": 1.0, "embedding": "umap", "seed": 0},
                                 result={"n_genes": 5, "n_modules": 2, "key": "map"}, selection=None, n_active=None, n_total=None)
    text = "\n".join(spec.code(step))
    assert "prepare_gene_map" in text and "key='map'" in text
    assert "2 modules" in spec.summary(step.params, step.result)
