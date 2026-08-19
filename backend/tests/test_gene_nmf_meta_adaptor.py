"""Adaptor + route tests for meta-programs (multi-sample NMF).

Three samples share two gene blocks; a third block belongs to one sample only.
A correct run separates them by sample coverage and persists the whole thing
in a form that survives an h5ad round trip.
"""
import json
import time

import anndata
import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient
from scipy.sparse import csr_matrix

from xcell.adaptor import DataAdaptor
from xcell.api import routes
from xcell.main import app


N_SAMPLES, N_CELLS, N_GENES = 3, 200, 45
BLOCK = 15
PRIVATE = "s1"


def _adata(empty_category: bool = False, tiny_sample: bool = False):
    rng = np.random.default_rng(0)
    rows, labels = [], []
    for s in range(N_SAMPLES):
        sample = f"s{s}"
        n_progs = 3 if sample == PRIVATE else 2
        W = np.zeros((N_GENES, n_progs))
        for j in range(n_progs):
            W[j * BLOCK:(j + 1) * BLOCK, j] = rng.uniform(4.0, 9.0, BLOCK)
        H = rng.uniform(0.0, 0.2, size=(N_CELLS, n_progs))
        dom = rng.integers(0, n_progs, N_CELLS)
        H[np.arange(N_CELLS), dom] += rng.uniform(3.0, 7.0, N_CELLS)
        rows.append(rng.poisson(H @ W.T).astype(np.float32))
        labels += [sample] * N_CELLS
    counts = np.vstack(rows)
    labels = np.array(labels, dtype=object)
    if tiny_sample:
        labels[:4] = "tiny"

    ad = anndata.AnnData(X=csr_matrix(counts))
    ad.var_names = [f"G{i:03d}" for i in range(N_GENES)]
    ad.var["highly_variable"] = [True] * N_GENES
    cats = sorted(set(labels)) + (["unassigned"] if empty_category else [])
    ad.obs["sample"] = pd.Categorical(labels, categories=cats)
    ad.obs["score"] = rng.random(len(labels))
    return ad


def _adaptor(**kw) -> DataAdaptor:
    return DataAdaptor("x.h5ad", adata=_adata(**kw))


def _run(a: DataAdaptor, **kw):
    kw.setdefault("sample_column", "sample")
    kw.setdefault("ks", [2, 3])
    kw.setdefault("n_mp", 3)
    kw.setdefault("min_cells", 10)
    compute_fn, apply_fn = a.prepare_meta_programs(**kw)
    return apply_fn(compute_fn(lambda frac, message=None: None))


# ---------------------------------------------------------------------------
# validation
# ---------------------------------------------------------------------------
def test_prepare_rejects_an_unknown_sample_column():
    a = _adaptor()
    with pytest.raises(ValueError, match="nope"):
        a.prepare_meta_programs(sample_column="nope", ks=[2], n_mp=2)


def test_prepare_rejects_a_continuous_sample_column():
    """A float column is a measurement, not a grouping — 24k 'samples'."""
    a = _adaptor()
    with pytest.raises(ValueError, match="categor"):
        a.prepare_meta_programs(sample_column="score", ks=[2], n_mp=2)


def test_prepare_rejects_a_single_sample():
    a = _adaptor()
    a.adata.obs["one"] = pd.Categorical(["only"] * a.n_cells)
    with pytest.raises(ValueError, match="at least 2"):
        a.prepare_meta_programs(sample_column="one", ks=[2], n_mp=2)


def test_prepare_rejects_empty_or_invalid_ks():
    a = _adaptor()
    with pytest.raises(ValueError, match="k"):
        a.prepare_meta_programs(sample_column="sample", ks=[], n_mp=2)
    with pytest.raises(ValueError, match="k"):
        a.prepare_meta_programs(sample_column="sample", ks=[1], n_mp=2)


def test_prepare_rejects_an_existing_key_without_overwrite():
    a = _adaptor()
    _run(a, key="MP")
    with pytest.raises(ValueError, match="overwrite"):
        a.prepare_meta_programs(sample_column="sample", ks=[2], n_mp=2, key="MP")


def test_prepare_rejects_negative_expression():
    ad = _adata()
    ad.layers["scaled"] = ad.X.toarray() - 5.0
    a = DataAdaptor("x.h5ad", adata=ad)
    with pytest.raises(ValueError, match="non-negative"):
        a.prepare_meta_programs(
            sample_column="sample", ks=[2], n_mp=2,
            layer="scaled", transform="none",
        )


# ---------------------------------------------------------------------------
# what a run persists
# ---------------------------------------------------------------------------
def test_run_writes_a_registered_score_matrix():
    a = _adaptor()
    out = _run(a, key="MP")
    names = [mp["name"] for mp in out["metaprograms"]]
    assert a.adata.obsm["MP"].shape == (N_SAMPLES * N_CELLS, len(names))
    reg = a.adata.uns["xcell_score_matrices"]["MP"]
    assert reg["columns"] == names
    assert reg["source"] == "gene_nmf_meta"


def test_metaprogram_columns_appear_in_the_schema():
    a = _adaptor()
    out = _run(a, key="MP")
    schema = a.get_schema()
    assert schema["score_matrices"]["MP"] == [
        mp["name"] for mp in out["metaprograms"]
    ]


def test_run_separates_a_private_program_by_sample_coverage():
    a = _adaptor()
    out = _run(a, key="MP", n_mp=3)
    coverage = sorted(mp["sample_coverage"] for mp in out["metaprograms"])
    assert coverage[0] == pytest.approx(1 / 3)
    assert coverage[-1] == pytest.approx(1.0)


def test_run_records_everything_needed_to_redraw_the_heatmap():
    a = _adaptor()
    out = _run(a, key="MP")
    rec = a.adata.uns["xcell_gene_nmf_meta"]["MP"]
    n = len(rec["program_labels"])
    assert np.asarray(rec["similarity"]).shape == (n, n)
    assert len(rec["clusters"]) == n
    assert len(rec["order"]) == n
    assert rec["params"]["sample_column"] == "sample"


def test_stored_record_survives_an_h5ad_round_trip(tmp_path):
    a = _adaptor()
    out = _run(a, key="MP")
    path = tmp_path / "mp.h5ad"
    a.adata.write_h5ad(path)
    back = anndata.read_h5ad(path)
    rec = back.uns["xcell_gene_nmf_meta"]["MP"]
    assert list(rec["metaprogram_names"]) == [
        mp["name"] for mp in out["metaprograms"]
    ]
    assert back.obsm["MP"].shape[1] == len(out["metaprograms"])


def test_result_round_trips_through_get_meta_programs_result():
    a = _adaptor()
    _run(a, key="MP")
    got = a.get_meta_programs_result("MP")
    assert got is not None
    json.dumps(got)
    assert [mp["name"] for mp in got["metaprograms"]]
    assert got["composition"]


def test_get_meta_programs_result_is_none_for_an_unknown_key():
    assert _adaptor().get_meta_programs_result("nope") is None


def test_result_is_json_serializable():
    a = _adaptor()
    json.dumps(_run(a, key="MP"))


def test_run_logs_an_action():
    a = _adaptor()
    _run(a, key="MP")
    assert "gene_nmf_meta" in [e["action"] for e in a._action_history]


def test_empty_sample_categories_are_ignored():
    a = _adaptor(empty_category=True)
    out = _run(a, key="MP")
    assert "unassigned" not in out["samples"]


def test_small_samples_are_skipped_and_reported():
    a = _adaptor(tiny_sample=True)
    out = _run(a, key="MP", min_cells=10)
    assert [s["sample"] for s in out["skipped"]] == ["tiny"]


def test_compute_does_not_mutate_anndata():
    a = _adaptor()
    compute_fn, _ = a.prepare_meta_programs(
        sample_column="sample", ks=[2, 3], n_mp=3, key="MP",
    )
    before_obsm, before_uns = set(a.adata.obsm), set(a.adata.uns)
    compute_fn(lambda frac, message=None: None)
    assert set(a.adata.obsm) == before_obsm
    assert set(a.adata.uns) == before_uns


def test_progress_reaches_one():
    a = _adaptor()
    seen = []
    compute_fn, _ = a.prepare_meta_programs(
        sample_column="sample", ks=[2, 3], n_mp=3,
    )
    compute_fn(lambda frac, message=None: seen.append(frac))
    assert seen and seen[-1] == 1.0


# ---------------------------------------------------------------------------
# routes
# ---------------------------------------------------------------------------
def _install(**kw):
    routes.set_adaptor(DataAdaptor("x.h5ad", adata=_adata(**kw)), slot="primary")


def _poll(client, task_id, timeout=60.0):
    deadline = time.time() + timeout
    while time.time() < deadline:
        r = client.get(f"/api/tasks/{task_id}").json()
        if r["status"] in ("completed", "error", "cancelled"):
            return r
        time.sleep(0.05)
    raise AssertionError("task did not finish")


def test_route_result_is_null_before_any_run():
    _install()
    assert TestClient(app).get("/api/gene_nmf/meta/result").json() is None


def test_route_run_and_result_roundtrip():
    _install()
    c = TestClient(app)
    r = c.post("/api/gene_nmf/meta/run", json={
        "sample_column": "sample", "ks": [2, 3], "n_mp": 3, "key": "MP",
    })
    assert r.status_code == 202, r.text
    done = _poll(c, r.json()["task_id"])
    assert done["status"] == "completed", done
    result = done["result"]
    assert result["metaprograms"]

    stored = c.get("/api/gene_nmf/meta/result", params={"key": "MP"}).json()
    assert [mp["name"] for mp in stored["metaprograms"]] == [
        mp["name"] for mp in result["metaprograms"]
    ]


def test_route_bad_sample_column_is_a_400():
    _install()
    r = TestClient(app).post("/api/gene_nmf/meta/run", json={
        "sample_column": "nope", "ks": [2], "n_mp": 2,
    })
    assert r.status_code == 400


def test_route_lists_candidate_sample_columns():
    """The UI needs to offer columns that could plausibly be samples."""
    _install()
    body = TestClient(app).get("/api/gene_nmf/meta/columns").json()
    names = [c["name"] for c in body["columns"]]
    assert "sample" in names
    assert "score" not in names          # continuous
    entry = next(c for c in body["columns"] if c["name"] == "sample")
    assert entry["n_samples"] == N_SAMPLES
