"""Clustering cells on the genes of one gene set.

Three planted cell populations drive three gene blocks; clustering on the
block genes must recover the populations, and everything the run writes must
land under suffixed keys so the dataset's own X_pca / leiden are untouched.
"""
import time
import types

import anndata
import numpy as np
import pandas as pd
import pytest
from scipy.sparse import csr_matrix
from fastapi.testclient import TestClient

from xcell.adaptor import DataAdaptor


N_CELLS, N_BLOCK_GENES, K = 150, 30, 3
BLOCK = N_BLOCK_GENES // K
GENES = [f"G{i:03d}" for i in range(N_BLOCK_GENES)]


def _adata(n_filler: int = 20):
    rng = np.random.default_rng(0)
    W = np.zeros((N_BLOCK_GENES, K))
    for j in range(K):
        W[j * BLOCK:(j + 1) * BLOCK, j] = rng.uniform(4.0, 9.0, BLOCK)
    H = rng.uniform(0.0, 0.2, size=(N_CELLS, K))
    dominant = rng.integers(0, K, N_CELLS)
    H[np.arange(N_CELLS), dominant] += rng.uniform(2.0, 6.0, N_CELLS)
    counts = rng.poisson(H @ W.T).astype(np.float32)
    filler = rng.poisson(0.5, size=(N_CELLS, n_filler)).astype(np.float32)
    ad = anndata.AnnData(X=csr_matrix(np.hstack([counts, filler])))
    ad.var_names = GENES + [f"F{i:03d}" for i in range(n_filler)]
    ad.obs["planted"] = pd.Categorical([str(d) for d in dominant])
    ad.obsm["X_pca"] = rng.random((N_CELLS, 5)).astype(np.float32)  # must survive untouched
    return ad


def _adaptor():
    return DataAdaptor("x.h5ad", adata=_adata())


def _run(a, genes=GENES, **kw):
    kw.setdefault("key", "blocks")
    compute_fn, apply_fn = a.prepare_cluster_cells_by_gene_set(genes, **kw)
    return apply_fn(compute_fn(lambda frac, message=None: None))


def test_recovers_planted_populations_and_writes_suffixed_keys():
    from sklearn.metrics import adjusted_rand_score
    a = _adaptor()
    before_pca = a.adata.obsm["X_pca"].copy()
    out = _run(a, n_comps=10, n_neighbors=15, resolution=0.5)

    assert out["key"] == "blocks"
    assert out["obs_column"] == "leiden_blocks"
    assert out["embedding"] == "X_umap_blocks"
    assert out["pca_key"] == "X_pca_blocks"
    assert out["graph_key"] == "blocks_connectivities"
    assert out["n_genes_used"] == N_BLOCK_GENES and out["genes_missing"] == []
    assert out["n_cells"] == N_CELLS
    assert 2 <= out["n_clusters"] <= 6
    assert sum(out["cluster_sizes"].values()) == N_CELLS

    ad = a.adata
    labels = ad.obs["leiden_blocks"]
    assert isinstance(labels.dtype, pd.CategoricalDtype)
    assert adjusted_rand_score(ad.obs["planted"], labels) >= 0.8
    assert ad.obsm["X_pca_blocks"].shape == (N_CELLS, 10) and not np.isnan(ad.obsm["X_pca_blocks"]).any()
    assert ad.obsm["X_umap_blocks"].shape == (N_CELLS, 2)
    assert ad.obsp["blocks_connectivities"].shape == (N_CELLS, N_CELLS)
    assert ad.obsp["blocks_distances"].shape == (N_CELLS, N_CELLS)
    assert ad.uns["blocks"]["connectivities_key"] == "blocks_connectivities"
    reg = ad.uns["xcell_gene_set_clusterings"]["blocks"]
    assert reg["genes_used"] == GENES and reg["n_cells"] == N_CELLS
    assert reg["params"]["resolution"] == 0.5
    # The dataset's own embedding and clustering were not touched.
    np.testing.assert_array_equal(ad.obsm["X_pca"], before_pca)
    assert "leiden" not in ad.obs
    schema = a.get_schema()
    assert "X_umap_blocks" in schema["embeddings"] and "leiden_blocks" in schema["obs_columns"]
    assert a._action_history[-1]["action"] == "cluster_cells_by_gene_set"
    import json
    json.dumps(out)


def test_cell_subset_labels_the_rest_unassigned_and_nans_their_embedding():
    a = _adaptor()
    idx = list(range(100))
    out = _run(a, cell_indices=idx, n_comps=8, run_umap=False)
    assert out["n_cells"] == 100 and out["embedding"] is None
    labels = a.adata.obs["leiden_blocks"]
    assert (labels.iloc[100:] == "unassigned").all()
    assert not (labels.iloc[:100] == "unassigned").any()
    pca = a.adata.obsm["X_pca_blocks"]
    assert np.isnan(pca[100:]).all() and not np.isnan(pca[:100]).any()
    assert "X_umap_blocks" not in a.adata.obsm
    # graph rows for excluded cells are empty
    conn = a.adata.obsp["blocks_connectivities"].tocsr()
    assert conn[100:].nnz == 0 and conn[:100].nnz > 0


def test_existing_leiden_route_can_recluster_the_stored_graph():
    a = _adaptor()
    _run(a, n_comps=8, run_umap=False)
    out = a.run_leiden(resolution=0.2, graph_key="blocks_connectivities")
    assert out["key_added"] == "leiden_blocks"
    assert out["n_clusters"] >= 1


def test_validation_is_synchronous():
    a = _adaptor()
    with pytest.raises(ValueError, match="at least 2"):
        a.prepare_cluster_cells_by_gene_set(["G000"], key="x")
    with pytest.raises(ValueError, match="None of the"):
        a.prepare_cluster_cells_by_gene_set(["nope1", "nope2"], key="x")
    with pytest.raises(ValueError, match="key"):
        a.prepare_cluster_cells_by_gene_set(GENES, key="   ")
    out = _run(a, genes=GENES + ["NOPE"], n_comps=5, run_umap=False)
    assert out["genes_missing"] == ["NOPE"] and out["n_genes_used"] == N_BLOCK_GENES
    with pytest.raises(ValueError, match="already exists"):
        a.prepare_cluster_cells_by_gene_set(GENES, key="blocks")
    _run(a, n_comps=5, run_umap=False, overwrite=True)
    # n_comps is clamped to what a 6-gene set allows rather than failing
    out = _run(a, genes=GENES[:6], key="tiny", n_comps=50, run_umap=False)
    assert a.adata.obsm["X_pca_tiny"].shape[1] == 5
    assert out["n_comps"] == 5


def _poll(client, task_id, timeout=60.0):
    t0 = time.time()
    while time.time() - t0 < timeout:
        st = client.get(f"/api/tasks/{task_id}").json()
        if st["status"] in ("completed", "error", "cancelled"):
            return st
        time.sleep(0.05)
    raise AssertionError("task did not finish")


def test_route_runs_as_task_with_a_selection(monkeypatch):
    from xcell.main import app
    from xcell.api import routes
    a = _adaptor()
    monkeypatch.setattr(routes, "get_adaptor", lambda dataset=None: a)
    client = TestClient(app)
    r = client.post("/api/gene_sets/cluster_cells", json={
        "genes": GENES, "key": "blocks", "n_comps": 8, "n_neighbors": 10, "resolution": 0.5,
        "run_umap": False, "cell_context": "selection", "cell_indices": list(range(80)),
    })
    assert r.status_code == 202, r.text
    st = _poll(client, r.json()["task_id"])
    assert st["status"] == "completed", st
    assert st["result"]["n_cells"] == 80 and st["result"]["obs_column"] == "leiden_blocks"
    r = client.post("/api/gene_sets/cluster_cells", json={"genes": ["G000"], "key": "x", "cell_context": "all"})
    assert r.status_code == 400


def test_codegen_emits_the_two_phase_call():
    from xcell import codegen
    spec = codegen.REGISTRY["cluster_cells_by_gene_set"]
    step = types.SimpleNamespace(action="cluster_cells_by_gene_set", params={
        "genes": ["G000", "G001"], "key": "blocks", "n_comps": 10, "n_neighbors": 15,
        "resolution": 0.5, "run_umap": True, "scale": True, "transform": "log1p", "layer": None, "seed": 0,
    }, result={"n_clusters": 3}, selection=None, n_active=None, n_total=None)
    lines = spec.code(step)
    text = "\n".join(lines if isinstance(lines, list) else lines.lines)
    assert "prepare_cluster_cells_by_gene_set" in text and "genes=" in text and "key='blocks'" in text
    assert "3 clusters" in spec.summary(step.params, step.result)
