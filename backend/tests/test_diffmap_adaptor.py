"""Diffusion maps and DPT through the adaptor, the routes and the notebook export.

The toy cells sit along a one-dimensional trajectory (``t`` from 0 to 1):
expression drifts smoothly with ``t`` and the cells lie along a spatial strip
in the same order, so a pseudotime rooted at ``t = 0`` must rise with ``t`` on either
graph. ``potency`` equals ``t`` — like stemFinder's ``stemfinder``, lower means
less differentiated.

See test_diffusion.py for the pure module; these tests pin what the adaptor
adds: scope, names, the registry, roots, the subset tree and the record.
"""

import json

import anndata
import numpy as np
import pandas as pd
import pytest
import scanpy as sc
from fastapi.testclient import TestClient
from scipy.sparse import csr_matrix
from scipy.stats import spearmanr

from xcell import codegen
from xcell.adaptor import DataAdaptor
from xcell.api import routes
from xcell.main import app

N = 80


def _adata():
    rng = np.random.default_rng(0)
    t = np.linspace(0.0, 1.0, N)
    rising = np.outer(t, np.linspace(1, 3, 10))
    falling = np.outer(1 - t, np.linspace(1, 3, 10))
    X = np.hstack([rising, falling]) * 3 + rng.normal(0, 0.05, (N, 20))
    X = np.clip(X, 0, None).astype(np.float32)
    ad = anndata.AnnData(X=csr_matrix(X))
    ad.var_names = [f"g{i}" for i in range(20)]
    ad.obs["t"] = t
    ad.obs["potency"] = t
    ad.obs["stage"] = pd.Categorical(np.where(t < 1 / 3, "early", np.where(t < 2 / 3, "mid", "late")))
    # A strip in trajectory order, so space follows t too (a raster grid would
    # not: distance from its corner is row + column).
    ad.obsm["spatial"] = np.column_stack([np.arange(N, dtype=float), rng.normal(0, 0.1, N)])
    Z = X - X.mean(axis=0)
    U, S, _ = np.linalg.svd(Z, full_matrices=False)
    ad.obsm["X_pca"] = (U[:, :10] * S[:10]).astype(np.float32)
    return ad


def _adaptor(graph=True):
    a = DataAdaptor("x.h5ad", adata=_adata())
    if graph:
        a.run_neighbors(n_neighbors=10)
    return a


def _rho(a, column):
    pt = a.adata.obs[column].to_numpy()
    ok = np.isfinite(pt)
    return spearmanr(a.adata.obs["t"].to_numpy()[ok], pt[ok]).statistic


# --- diffusion map ----------------------------------------------------------

def test_diffmap_on_the_default_graph_writes_X_diffmap_and_scanpys_evals():
    a = _adaptor()
    r = a.run_diffmap(n_comps=10)
    assert r["embedding_name"] == "X_diffmap"
    X = a.adata.obsm["X_diffmap"]
    assert X.shape == (N, 10)
    assert np.isfinite(X).all()
    np.testing.assert_array_equal(a.adata.uns["diffmap_evals"], np.asarray(r["eigenvalues"], dtype=np.float32))
    entry = a.adata.uns["xcell_diffmaps"]["X_diffmap"]
    assert entry["graph_key"] == "connectivities"
    assert int(entry["n_comps"]) == 10
    assert r["view_dims"] == [1, 2]
    assert r["n_isolated"] == 0 and r["symmetrized"] is False
    json.dumps(r, allow_nan=False)


def test_diffmap_first_component_follows_the_trajectory():
    a = _adaptor()
    a.run_diffmap(n_comps=10)
    dc1 = a.adata.obsm["X_diffmap"][:, 1]
    assert abs(spearmanr(a.adata.obs["t"], dc1).statistic) > 0.95


def test_diffmap_on_the_spatial_graph_gets_its_own_key():
    a = _adaptor()
    a.run_spatial_neighbors(n_neighs=6, coord_type="generic")
    r = a.run_diffmap(n_comps=8, graph_key="spatial_connectivities")
    assert r["embedding_name"] == "X_diffmap_spatial"
    assert "X_diffmap" not in a.adata.obsm
    assert a.adata.uns["xcell_diffmaps"]["X_diffmap_spatial"]["graph_key"] == "spatial_connectivities"
    # squidpy's kNN spatial graph is not mutual.
    assert r["symmetrized"] is True


def test_diffmap_on_an_asymmetric_spatial_graph_with_isolated_cells():
    a = _adaptor(graph=False)
    rng = np.random.default_rng(1)
    W = (rng.random((N, N)) < 0.08).astype(np.float32)
    np.fill_diagonal(W, 0)
    W[5, :] = 0
    W[:, 5] = 0          # cell 5 isolated
    W[0, 40] = 1.0       # one-way edge
    W[40, 0] = 0.0
    a.adata.obsp["rad_connectivities"] = csr_matrix(W)
    r = a.run_diffmap(n_comps=6, graph_key="rad_connectivities")
    X = a.adata.obsm[r["embedding_name"]]
    assert np.isnan(X[5]).all()
    assert np.isfinite(np.delete(X, 5, axis=0)).all()
    assert r["n_isolated"] == 1 and r["symmetrized"] is True
    assert len(r["warnings"]) >= 2
    json.dumps(r, allow_nan=False)


def test_diffmap_on_a_selection_writes_nan_outside():
    a = _adaptor()
    r = a.run_diffmap(n_comps=6, active_cell_indices=list(range(40)))
    X = a.adata.obsm[r["embedding_name"]]
    assert r["embedding_name"] == "X_diffmap"
    assert np.isfinite(X[:40]).all()
    assert np.isnan(X[40:]).all()
    assert a.analysis_record.steps[-1].n_active == 40


def test_diffmap_on_a_subset_never_lands_on_the_datasets_key():
    a = _adaptor()
    a.create_cell_subset("early", list(range(30)))
    # The browser sends the default graph explicitly; it must still suffix.
    r = a.run_diffmap(n_comps=6, graph_key="connectivities", cell_subset="early")
    assert r["embedding_name"] == "X_diffmap_early"
    assert "X_diffmap" not in a.adata.obsm
    X = a.adata.obsm["X_diffmap_early"]
    assert np.isfinite(X[:30]).all() and np.isnan(X[30:]).all()
    assert a.adata.uns["xcell_diffmaps"]["X_diffmap_early"]["cell_subset"] == "early"


def test_diffmap_on_a_subset_is_recorded_in_the_tree():
    a = _adaptor()
    a.create_cell_subset("early", list(range(30)))
    a.run_diffmap(n_comps=6, cell_subset="early")
    summary = next(s for s in a.list_cell_subsets() if s["name"] == "early")
    assert summary["derived"]["diffmap"] == ["X_diffmap_early"]
    assert "X_diffmap_early" in summary["embeddings"]


def test_diffmap_refuses_to_overwrite_another_embedding():
    a = _adaptor()
    before = a.adata.obsm["X_pca"].copy()
    with pytest.raises(ValueError, match="X_pca"):
        a.run_diffmap(n_comps=6, key_added="X_pca")
    np.testing.assert_array_equal(a.adata.obsm["X_pca"], before)


def test_diffmap_rejects_an_unknown_graph():
    a = _adaptor()
    with pytest.raises(ValueError, match="not found"):
        a.run_diffmap(graph_key="nope_connectivities")


def test_diffmap_without_a_graph_asks_for_neighbors():
    a = _adaptor(graph=False)
    with pytest.raises(ValueError, match="neighbors"):
        a.run_diffmap()


# --- DPT ----------------------------------------------------------------------

def _with_map(**kw):
    a = _adaptor()
    a.run_diffmap(n_comps=10, **kw)
    return a


def test_dpt_from_the_least_differentiated_cell_runs_forward():
    a = _with_map()
    r = a.run_dpt("X_diffmap", root_mode="obs_min", root_column="potency")
    assert r["key_added"] == "dpt_pseudotime"
    assert r["root_index"] == 0
    assert "potency" in r["root_rule"]
    assert a.adata.obs["dpt_pseudotime"].iloc[0] == 0.0
    assert _rho(a, "dpt_pseudotime") > 0.95
    json.dumps(r, allow_nan=False)


def test_dpt_root_modes():
    a = _with_map()
    assert a.run_dpt(root_mode="obs_max", root_column="potency")["root_index"] == N - 1
    assert a.run_dpt(root_mode="cells", root_cells=[17])["root_index"] == 17
    group = a.run_dpt(root_mode="group", root_column="stage", root_value="early")["root_index"]
    assert a.adata.obs["stage"].iloc[group] == "early"
    assert 5 <= group <= 21
    centre = a.run_dpt(root_mode="cells", root_cells=list(range(40, 61)))["root_index"]
    assert 45 <= centre <= 55
    # DC1 runs along the trajectory, so its two tips are its two ends (kNN
    # noise can move the extreme a cell or two in).
    tips = sorted(a.run_dpt(root_mode=m, root_component=1)["root_index"] for m in ("dc_min", "dc_max"))
    assert tips[0] <= 4 and tips[1] >= N - 5


def test_dpt_root_stays_inside_the_map():
    a = _adaptor()
    a.create_cell_subset("late", list(range(40, N)))
    a.run_diffmap(n_comps=6, cell_subset="late")
    r = a.run_dpt("X_diffmap_late", root_mode="obs_min", root_column="potency")
    # potency's global minimum is cell 0, which the subset's map does not cover.
    assert r["root_index"] == 40
    assert r["key_added"] == "dpt_pseudotime_late"
    pt = a.adata.obs["dpt_pseudotime_late"].to_numpy()
    assert np.isnan(pt[:40]).all() and np.isfinite(pt[40:]).all()


def test_dpt_default_name_follows_the_maps_graph():
    a = _adaptor()
    a.run_spatial_neighbors(n_neighs=6, coord_type="generic")
    a.run_diffmap(n_comps=6, graph_key="spatial_connectivities")
    r = a.run_dpt("X_diffmap_spatial", root_mode="obs_min", root_column="potency")
    assert r["key_added"] == "dpt_pseudotime_spatial"
    assert _rho(a, "dpt_pseudotime_spatial") > 0.9


@pytest.mark.parametrize("kw, match", [
    ({"root_mode": "sideways"}, "root_mode"),
    ({"root_mode": "obs_min", "root_column": "missing"}, "missing"),
    ({"root_mode": "obs_min", "root_column": "stage"}, "numeric"),
    ({"root_mode": "group", "root_column": "stage", "root_value": "nope"}, "nope"),
    ({"root_mode": "cells", "root_cells": []}, "root"),
    ({"root_mode": "dc_max", "root_component": 99}, "component"),
])
def test_dpt_bad_roots_are_value_errors(kw, match):
    a = _with_map()
    with pytest.raises(ValueError, match=match):
        a.run_dpt("X_diffmap", **kw)


def test_dpt_on_an_unknown_map_is_a_key_error():
    with pytest.raises(KeyError):
        _adaptor().run_dpt("X_diffmap_nothing", root_mode="cells", root_cells=[0])


def test_dpt_on_an_embedding_that_is_not_a_diffusion_map():
    with pytest.raises(ValueError, match="diffusion map"):
        _adaptor().run_dpt("X_pca", root_mode="cells", root_cells=[0])


def test_dpt_rerun_overwrites():
    a = _with_map()
    a.run_dpt(root_mode="cells", root_cells=[0])
    first = a.adata.obs["dpt_pseudotime"].copy()
    r = a.run_dpt(root_mode="cells", root_cells=[N - 1])
    assert r["key_added"] == "dpt_pseudotime"
    assert a.adata.obs["dpt_pseudotime"].iloc[N - 1] == 0.0
    assert not np.allclose(first, a.adata.obs["dpt_pseudotime"])
    assert list(a.adata.obs.columns).count("dpt_pseudotime") == 1


def test_dpt_refuses_to_overwrite_another_column():
    a = _with_map()
    with pytest.raises(ValueError, match="stage"):
        a.run_dpt(root_mode="cells", root_cells=[0], key_added="stage")
    assert a.adata.obs["stage"].dtype.name == "category"


def test_dpt_response_is_json_safe_on_a_disconnected_subset():
    a = _adaptor()
    ends = list(range(12)) + list(range(N - 12, N))
    a.create_cell_subset("ends", ends)
    rm = a.run_diffmap(n_comps=6, cell_subset="ends")
    assert rm["n_components"] == 2
    r = a.run_dpt("X_diffmap_ends", root_mode="cells", root_cells=[0])
    assert r["n_unreachable"] == 12
    assert r["warnings"]
    json.dumps(r, allow_nan=False)
    pt = a.adata.obs["dpt_pseudotime_ends"].to_numpy()
    assert np.isfinite(pt[:12]).all() and np.isnan(pt[12:]).all()


def test_dpt_on_a_diffusion_map_scanpy_wrote():
    a = _adaptor()
    sc.tl.diffmap(a.adata, n_comps=8)
    r = a.run_dpt("X_diffmap", root_mode="cells", root_cells=[0])
    assert r["key_added"] == "dpt_pseudotime"
    assert _rho(a, "dpt_pseudotime") > 0.95


def test_dpt_after_the_maps_graph_is_gone_says_rerun():
    a = _adaptor()
    a.run_spatial_neighbors(n_neighs=6, coord_type="generic")
    a.run_diffmap(n_comps=6, graph_key="spatial_connectivities")
    del a.adata.obsp["spatial_connectivities"]
    with pytest.raises(ValueError, match="re-run"):
        a.run_dpt("X_diffmap_spatial", root_mode="cells", root_cells=[0])


def test_dpt_records_the_resolved_root_and_writes_the_registry():
    a = _with_map()
    r = a.run_dpt(root_mode="obs_min", root_column="potency")
    step = a.analysis_record.steps[-1]
    assert step.action == "dpt"
    assert step.params["root_mode"] == "cells"
    assert step.params["root_cells"] == [r["root_index"]]
    entry = a.adata.uns["xcell_dpt"]["dpt_pseudotime"]
    assert entry["diffmap_key"] == "X_diffmap"
    assert int(entry["root_index"]) == r["root_index"]
    assert a.adata.uns["iroot"] == r["root_index"]


def test_subset_cascade_delete_drops_diffmaps_and_pseudotime():
    a = _adaptor()
    a.create_cell_subset("early", list(range(30)))
    a.run_diffmap(n_comps=6, cell_subset="early")
    a.run_dpt("X_diffmap_early", root_mode="cells", root_cells=[0])
    summary = next(s for s in a.list_cell_subsets() if s["name"] == "early")
    assert summary["derived"]["dpt"] == ["dpt_pseudotime_early"]
    out = a.delete_cell_subset("early", drop_derived=True)
    assert "X_diffmap_early" in out["dropped"] and "dpt_pseudotime_early" in out["dropped"]
    assert "X_diffmap_early" not in a.adata.obsm
    assert "dpt_pseudotime_early" not in a.adata.obs
    assert "X_diffmap_early" not in a.adata.uns.get("xcell_diffmaps", {})
    assert "dpt_pseudotime_early" not in a.adata.uns.get("xcell_dpt", {})


def test_list_diffmaps_and_potency_columns():
    a = _with_map()
    a.run_spatial_neighbors(n_neighs=6, coord_type="generic")
    a.run_diffmap(n_comps=6, graph_key="spatial_connectivities")
    a.adata.obs["stemfinder"] = a.adata.obs["t"]
    a.adata.obs["diffometer_sub1"] = 1 - a.adata.obs["t"]
    a.adata.obs["stemfinder_cc_mean"] = a.adata.obs["t"]
    out = a.list_diffmaps()
    keys = {d["key"]: d for d in out["diffmaps"]}
    assert set(keys) == {"X_diffmap", "X_diffmap_spatial"}
    assert keys["X_diffmap"]["n_comps"] == 10 and keys["X_diffmap"]["n_cells"] == N
    assert keys["X_diffmap_spatial"]["graph_key"] == "spatial_connectivities"
    assert keys["X_diffmap"]["view_dims"] == [1, 2]
    potency = {p["column"]: p["direction"] for p in out["potency"]}
    assert potency == {"stemfinder": "min", "diffometer_sub1": "max"}
    json.dumps(out, allow_nan=False)


def test_potency_preset_on_a_real_stemfinder_orientation():
    a = _with_map()
    a.adata.obs["stemfinder"] = a.adata.obs["t"]  # lower = less differentiated
    r = a.run_dpt(root_mode="obs_min", root_column="stemfinder")
    assert r["root_index"] == 0


# --- routes -------------------------------------------------------------------

def _client(monkeypatch, a):
    monkeypatch.setattr(routes, "get_adaptor", lambda dataset=None: a)
    return TestClient(app)


def test_routes(monkeypatch):
    a = _adaptor()
    c = _client(monkeypatch, a)
    res = c.post("/api/scanpy/diffmap", json={"n_comps": 8})
    assert res.status_code == 200, res.text
    assert res.json()["embedding_name"] == "X_diffmap"
    res = c.post("/api/scanpy/diffmap", json={"n_comps": 2})
    assert res.status_code == 400
    res = c.get("/api/scanpy/diffmaps")
    assert res.status_code == 200 and res.json()["diffmaps"][0]["key"] == "X_diffmap"
    res = c.post("/api/scanpy/dpt", json={"diffmap_key": "X_diffmap", "root_mode": "obs_min",
                                          "root_column": "potency"})
    assert res.status_code == 200, res.text
    assert res.json()["root_index"] == 0
    res = c.post("/api/scanpy/dpt", json={"diffmap_key": "X_diffmap_nope", "root_cells": [0]})
    assert res.status_code == 404
    res = c.post("/api/scanpy/dpt", json={"diffmap_key": "X_diffmap", "root_mode": "nope"})
    assert res.status_code == 400


def test_prerequisites_route_knows_diffmap(monkeypatch):
    c = _client(monkeypatch, _adaptor(graph=False))
    res = c.get("/api/scanpy/prerequisites/diffmap")
    assert res.status_code == 200
    assert res.json()["satisfied"] is False


# --- notebook export ------------------------------------------------------------

def test_diffmap_and_dpt_steps_translate_to_runnable_calls():
    a = _adaptor()
    a.create_cell_subset("early", list(range(30)))
    a.run_diffmap(n_comps=6, cell_subset="early")
    a.run_dpt("X_diffmap_early", root_mode="obs_min", root_column="potency")
    dm_step, dpt_step = a.analysis_record.steps[-2:]
    t1 = codegen.translate(dm_step)
    t2 = codegen.translate(dpt_step)
    code1, code2 = "\n".join(t1.code), "\n".join(t2.code)
    assert "xa.run_diffmap(" in code1 and "cell_subset='early'" in code1
    assert "xa.run_dpt(" in code2 and "root_cells=[0]" in code2
    assert "potency" in t2.summary
    compile(code1 + "\n" + code2, "<diffusion>", "exec")


def test_a_diffmap_on_a_selection_replays_it():
    a = _adaptor()
    a.run_diffmap(n_comps=6, active_cell_indices=list(range(40)))
    t = codegen.translate(a.analysis_record.steps[-1])
    code = "\n".join(t.code)
    assert "active_cell_indices=SELECTIONS['step_" in code
    assert not any("whole dataset" in w for w in t.warnings)


# --- fragmented graphs and name collisions (from review) ------------------------

def _islands(n_islands=20, per=15, gap=100.0):
    """Cells in tight, far-apart islands: a radius graph falls into one piece each."""
    rng = np.random.default_rng(2)
    centres = np.column_stack([np.arange(n_islands) * gap, np.zeros(n_islands)])
    coords = np.vstack([c + rng.normal(0, 1.0, (per, 2)) for c in centres])
    ad = anndata.AnnData(X=csr_matrix(rng.random((coords.shape[0], 5)).astype(np.float32)))
    ad.obsm["spatial"] = coords
    return DataAdaptor("x.h5ad", adata=ad)


def test_a_radius_graph_in_many_pieces_is_a_400_with_advice(monkeypatch):
    import time
    a = _islands()
    a.run_spatial_neighbors(coord_type="generic", radius=10.0)
    c = _client(monkeypatch, a)
    t = time.time()
    res = c.post("/api/scanpy/diffmap", json={"graph_key": "spatial_connectivities", "n_comps": 15})
    assert res.status_code == 400
    assert "20 disconnected pieces" in res.json()["detail"]
    assert "subset" in res.json()["detail"]
    assert time.time() - t < 5
    assert "X_diffmap_spatial" not in a.adata.obsm


def test_a_radius_graph_in_few_pieces_maps_each():
    a = _islands(n_islands=3, per=40)
    a.run_spatial_neighbors(coord_type="generic", radius=10.0)
    r = a.run_diffmap(n_comps=8, graph_key="spatial_connectivities")
    assert r["n_components"] == 3 and r["n_stationary"] == 3
    assert r["view_dims"] == [3, 4]
    assert np.isfinite(a.adata.obsm["X_diffmap_spatial"]).all()
    json.dumps(r, allow_nan=False)


def test_a_subset_named_like_a_graph_does_not_claim_its_maps():
    a = _adaptor()
    a.run_spatial_neighbors(n_neighs=6, coord_type="generic")
    a.run_diffmap(n_comps=6, graph_key="spatial_connectivities")      # dataset-level
    a.run_dpt("X_diffmap_spatial", root_mode="cells", root_cells=[0])
    with pytest.raises(ValueError, match="spatial"):
        a.create_cell_subset("spatial", list(range(30)))
    # A file saved before the name was refused can still hold one.
    a.adata.obs["subset_spatial"] = np.arange(N) < 30
    a.adata.uns["xcell_cell_subsets"] = {"spatial": {"obs_key": "subset_spatial"}}
    summary = next(s for s in a.list_cell_subsets() if s["name"] == "spatial")
    assert summary["derived"]["diffmap"] == []
    assert summary["derived"]["dpt"] == []
    out = a.delete_cell_subset("spatial", drop_derived=True)
    assert "X_diffmap_spatial" in a.adata.obsm
    assert "dpt_pseudotime_spatial" in a.adata.obs
    assert "X_diffmap_spatial" not in out["dropped"]
