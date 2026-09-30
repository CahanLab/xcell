"""The adaptor side of PyStemFinder scoring: scope, snapshot, write back, record.

See test_stemfinder.py for the pure module and the toy data's design: ``hi``
cells express the five cell-cycle markers heterogeneously and must score as
less differentiated than ``lo`` cells.
"""

import importlib.util

import anndata
import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient
from scipy.sparse import csr_matrix

from xcell import codegen
from xcell import stemfinder as sf
from xcell.adaptor import DataAdaptor
from xcell.api import routes
from xcell.main import app

HAVE_PSF = importlib.util.find_spec("pystemfinder") is not None
needs_psf = pytest.mark.skipif(not HAVE_PSF, reason="pystemfinder not installed")

MARKERS = ["Mcm5", "Pcna", "Top2a", "Mki67", "Cdk1"]
TFS = ["Sox2", "Myc", "Gata1"]
N_HI, N_LO = 90, 90
N_CELLS = N_HI + N_LO


def _counts():
    rng = np.random.default_rng(0)
    n_bg = 40
    pop = np.array(["hi"] * N_HI + ["lo"] * N_LO)
    bg = rng.poisson(3.0, size=(N_CELLS, n_bg)).astype(np.float32)
    bg[pop == "hi", :10] += rng.poisson(6.0, size=(N_HI, 10))
    bg[pop == "lo", 10:20] += rng.poisson(6.0, size=(N_LO, 10))
    on = np.where(pop == "hi", 0.5, 0.1)[:, None]
    mk = (rng.random((N_CELLS, len(MARKERS))) < on) * rng.poisson(5.0, size=(N_CELLS, len(MARKERS)))
    tf = (rng.random((N_CELLS, len(TFS))) < np.where(pop == "hi", 0.8, 0.2)[:, None]) * 2
    X = np.hstack([bg, mk, tf]).astype(np.float32)
    names = [f"bg{i:02d}" for i in range(n_bg)] + MARKERS + TFS
    return X, names, pop


def _adata(log=False):
    X, names, pop = _counts()
    lognorm = np.log1p(X / X.sum(axis=1, keepdims=True) * 1e4)
    ad = anndata.AnnData(X=csr_matrix((lognorm if log else X).astype(np.float32)))
    ad.var_names = names
    ad.obs["pop"] = pd.Categorical(pop)
    # A deterministic PCA of the background genes, like the dataset's own X_pca.
    Z = lognorm[:, :40] - lognorm[:, :40].mean(axis=0)
    U, S, _ = np.linalg.svd(Z, full_matrices=False)
    ad.obsm["X_pca"] = (U[:, :10] * S[:10]).astype(np.float32)
    return ad


def _adaptor(log=False):
    return DataAdaptor("x.h5ad", adata=_adata(log))


def _run(a, **kw):
    kw.setdefault("markers", MARKERS)
    compute_fn, apply_fn = a.prepare_stemfinder(**kw)
    return apply_fn(compute_fn(lambda f, m=None: None))


# ---------- status ----------

def test_status_describes_the_dataset_for_the_pickers():
    s = _adaptor().stemfinder_status()
    assert s["n_cells"] == N_CELLS
    assert s["default_n_neighbors"] == round(np.sqrt(N_CELLS))
    assert "X_pca" in s["pc_embeddings"]
    assert "species" in s              # too few symbols here for a confident guess
    assert s["x_scale"] == "raw_counts"
    assert "available" in s and "install_hint" in s


def test_status_and_prepare_without_the_package(monkeypatch):
    monkeypatch.setattr(sf, "_import_module", lambda: None)
    a = _adaptor()
    assert a.stemfinder_status()["available"] is False
    with pytest.raises(ValueError, match="pip install pystemfinder"):
        a.prepare_stemfinder(markers=MARKERS)


# ---------- validation is synchronous ----------

@needs_psf
def test_no_marker_in_the_dataset_is_a_400_before_any_work():
    with pytest.raises(ValueError, match="None of the"):
        _adaptor().prepare_stemfinder(markers=["NotAGene", "Another"])


@needs_psf
def test_an_unknown_embedding_or_graph_is_rejected():
    a = _adaptor()
    with pytest.raises(ValueError, match="X_nope"):
        a.prepare_stemfinder(markers=MARKERS, use_rep="X_nope")
    with pytest.raises(ValueError, match="nope_connectivities"):
        a.prepare_stemfinder(markers=MARKERS, graph="existing", graph_key="nope_connectivities")


# ---------- the run ----------

@needs_psf
def test_run_writes_scores_that_rank_the_populations():
    a = _adaptor()
    r = _run(a)
    assert r["columns"] == ["stemfinder", "stemfinder_raw", "diffometer"]
    for col in r["columns"]:
        assert a.adata.obs[col].dtype == np.float64
        assert np.isfinite(a.adata.obs[col]).all()
    hi = (a.adata.obs["pop"] == "hi").to_numpy()
    sfv = a.adata.obs["stemfinder"].to_numpy()
    assert np.median(sfv[hi]) < np.median(sfv[~hi])
    assert r["n_markers_used"] == len(MARKERS)
    assert r["n_neighbors"] == round(np.sqrt(N_CELLS))
    assert r["n_cells_scored"] == N_CELLS
    assert r["stats"]["stemfinder"]["max"] == pytest.approx(1.0)


@needs_psf
def test_default_markers_are_the_species_cell_cycle_list():
    r = _run(_adaptor(), markers=None, species="mouse")
    assert r["n_markers_used"] == len(MARKERS)            # the five present
    assert r["n_markers_missing"] > 50                     # the rest of S + G2M


@needs_psf
def test_counts_and_log_normalized_x_give_the_same_scores():
    a, b = _adaptor(log=False), _adaptor(log=True)
    _run(a, metrics=["stemfinder", "diffometer", "cc_mean"])
    _run(b, metrics=["stemfinder", "diffometer", "cc_mean"])
    for col in ["stemfinder_raw", "diffometer", "stemfinder_cc_mean"]:
        np.testing.assert_allclose(a.adata.obs[col], b.adata.obs[col], rtol=1e-4, atol=1e-6)


@needs_psf
def test_expressed_tfs_count_the_species_tf_list():
    a = _adaptor()
    r = _run(a, metrics=["n_tfs"], species="mouse")
    X, names, _ = _counts()
    want = (X[:, [names.index(t) for t in TFS]] > 0).sum(axis=1)
    np.testing.assert_array_equal(a.adata.obs["stemfinder_n_TFs"].to_numpy(), want)
    assert r["n_tfs_used"] == len(TFS)


@needs_psf
def test_the_mask_scores_only_its_cells_as_a_run_on_them_alone_would():
    idx = list(range(0, N_CELLS, 2))
    a = _adaptor()
    r = _run(a, active_cell_indices=idx)
    out = a.adata.obs["stemfinder"].to_numpy()
    outside = np.setdiff1d(np.arange(N_CELLS), idx)
    assert np.isnan(out[outside]).all()
    assert r["n_cells_scored"] == len(idx)
    assert r["n_neighbors"] == round(np.sqrt(len(idx)))

    alone = DataAdaptor("x.h5ad", adata=_adata()[idx].copy())
    _run(alone)
    np.testing.assert_allclose(out[idx], alone.adata.obs["stemfinder"].to_numpy())


@needs_psf
def test_scores_equal_a_direct_pystemfinder_run():
    # The contract the feature rests on: xcell only prepares what PyStemFinder
    # expects, so the same preparation done by hand gives the same numbers.
    import pystemfinder as psf
    import scanpy as sc
    a = _adaptor()
    _run(a)
    ref = _adata()
    sc.pp.normalize_total(ref, target_sum=1e4)
    sc.pp.log1p(ref)
    ref = ref[:, MARKERS].copy()
    sc.pp.scale(ref, max_value=10)
    sc.pp.neighbors(ref, n_neighbors=round(np.sqrt(N_CELLS)), n_pcs=10, use_rep="X_pca")
    psf.stemfinder(ref, MARKERS)
    psf.diffometer(ref, MARKERS)
    np.testing.assert_allclose(a.adata.obs["stemfinder_raw"], ref.obs["stemfinder_raw"], atol=1e-12)
    np.testing.assert_allclose(a.adata.obs["diffometer"], ref.obs["diffometer"], atol=1e-12)


@needs_psf
def test_a_masked_existing_graph_reports_and_blanks_isolated_cells():
    import scanpy as sc
    a = _adaptor()
    sc.pp.neighbors(a.adata, n_neighbors=6, use_rep="X_pca")
    idx = list(range(0, N_CELLS, 7))                       # sparse mask: many lose every neighbour
    r = _run(a, graph="existing", graph_key="connectivities", active_cell_indices=idx)
    d = a.adata.obs["diffometer"].to_numpy()[idx]
    assert r["n_isolated"] > 0
    assert np.isnan(d).sum() == r["n_isolated"]
    assert r["min_neighbors"] == 0
    assert any("neighbour" in w for w in r["warnings"])


@needs_psf
def test_summary_by_must_be_categorical():
    a = _adaptor()
    a.adata.obs["depth"] = np.arange(N_CELLS, dtype=float)
    with pytest.raises(ValueError, match="categorical"):
        a.prepare_stemfinder(markers=MARKERS, summary_by="depth")


@needs_psf
def test_summary_sorts_the_other_metrics_high_first():
    r = _run(_adaptor(), metrics=["diffometer"], summary_by="pop")
    assert [g["group"] for g in r["summary"]] == ["hi", "lo"]      # higher diffOmeter = less differentiated


@needs_psf
def test_markers_among_the_hvgs_are_flagged():
    a = _adaptor()
    a.adata.var["highly_variable"] = [g in MARKERS[:3] for g in a.adata.var_names]
    r = _run(a)
    assert any("highly variable" in w for w in r["warnings"])


@needs_psf
def test_an_existing_graph_is_used_through_its_distances():
    import scanpy as sc
    a = _adaptor()
    sc.pp.neighbors(a.adata, n_neighbors=12, use_rep="X_pca")
    r = _run(a, graph="existing", graph_key="connectivities")
    assert r["graph_source"] == "distances"
    assert r["n_neighbors"] is None
    assert np.isfinite(a.adata.obs["stemfinder"]).all()
    # The dataset's graph is read, never replaced.
    assert a.adata.uns["neighbors"]["params"]["n_neighbors"] == 12


@needs_psf
def test_suffix_names_every_column():
    a = _adaptor()
    r = _run(a, suffix="k50 run")
    assert r["columns"] == ["stemfinder_k50_run", "stemfinder_raw_k50_run", "diffometer_k50_run"]
    assert "stemfinder" not in a.adata.obs


@needs_psf
def test_summary_by_ranks_groups_least_differentiated_first():
    r = _run(_adaptor(), summary_by="pop")
    assert [g["group"] for g in r["summary"]] == ["hi", "lo"]
    assert r["summary"][0]["medians"]["stemfinder"] < r["summary"][1]["medians"]["stemfinder"]


@needs_psf
def test_the_record_step_and_its_notebook_code():
    a = _adaptor()
    _run(a, active_cell_indices=list(range(100)))
    step = a.analysis_record.steps[-1]
    assert step.action == "stemfinder"
    assert step.n_active == 100
    assert "active_cell_indices" not in step.params
    t = codegen.translate(step)
    code = "\n".join(t.code)
    assert "xa.prepare_stemfinder(" in code
    # The notebook scores the same cells, so it needs no whole-dataset caveat.
    assert "active_cell_indices=SELECTIONS['step_" in code
    assert not any("whole dataset" in w for w in t.warnings)
    compile(code, "<stemfinder>", "exec")


# ---------- routes ----------

def test_status_route(monkeypatch):
    a = _adaptor()
    monkeypatch.setattr(routes, "get_adaptor", lambda dataset=None: a)
    res = TestClient(app).get("/api/stemfinder/status")
    assert res.status_code == 200
    assert res.json()["n_cells"] == N_CELLS


def test_run_route_without_the_package_is_a_400_with_the_hint(monkeypatch):
    monkeypatch.setattr(sf, "_import_module", lambda: None)
    a = _adaptor()
    monkeypatch.setattr(routes, "get_adaptor", lambda dataset=None: a)
    res = TestClient(app).post("/api/stemfinder", json={"markers": MARKERS})
    assert res.status_code == 400
    assert "pip install pystemfinder" in res.json()["detail"]


@needs_psf
def test_run_route_submits_a_task(monkeypatch):
    a = _adaptor()
    monkeypatch.setattr(routes, "get_adaptor", lambda dataset=None: a)
    captured = {}

    def submit(compute_fn, apply_fn):
        captured["result"] = apply_fn(compute_fn(lambda f, m=None: None))
        return "t1"

    monkeypatch.setattr(routes.task_manager, "submit", submit)
    res = TestClient(app).post("/api/stemfinder", json={
        "metrics": ["stemfinder"], "markers": MARKERS, "active_cell_indices": list(range(120)),
    })
    assert res.status_code == 202, res.text
    assert res.json()["task_id"] == "t1"
    assert captured["result"]["n_cells_scored"] == 120
