"""Marker genes honour the browser's cell mask.

The mask is the cell universe for the other comparisons (two-group diffexp
filters both groups by it); marker genes ran one-vs-rest over every cell
whatever the mask said, so a user who masked down to one tissue got markers
computed against cells they had put out of view.

The dataset is built so the answer changes with scope: in the first half of
the cells group ``a`` is marked by ``g00``, in the second half by ``g01``.
"""

import anndata
import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient
from scipy.sparse import csr_matrix

from xcell.adaptor import DataAdaptor
from xcell.api import routes
from xcell.main import app


N_CELLS, N_GENES = 120, 8
GENES = [f"g{i:02d}" for i in range(N_GENES)]
HALF = N_CELLS // 2
FIRST = list(range(HALF))
SECOND = list(range(HALF, N_CELLS))


def _adata():
    rng = np.random.default_rng(0)
    X = rng.poisson(1.0, size=(N_CELLS, N_GENES)).astype(np.float32)
    grp = np.array(["a", "b", "c"] * (N_CELLS // 3))
    is_a = grp == "a"
    X[:HALF][is_a[:HALF], 0] += 8.0      # g00 marks a in the first half
    X[HALF:][is_a[HALF:], 1] += 8.0      # g01 marks a in the second half
    ad = anndata.AnnData(X=csr_matrix(X))
    ad.var_names = GENES
    ad.obs["grp"] = pd.Categorical(grp)
    return ad


def _adaptor():
    return DataAdaptor("x.h5ad", adata=_adata())


def _top(result, group):
    return next(r for r in result["results"] if r["group"] == group)["genes"][0]["gene"]


def test_without_a_mask_every_cell_is_tested():
    r = _adaptor().run_marker_genes(obs_column="grp", top_n=3)
    assert r["n_cells_tested"] == N_CELLS


def test_the_mask_decides_which_cells_are_compared():
    a = _adaptor()
    first = a.run_marker_genes(obs_column="grp", top_n=3, active_cell_indices=FIRST)
    second = a.run_marker_genes(obs_column="grp", top_n=3, active_cell_indices=SECOND)

    assert _top(first, "a") == "g00"
    assert _top(second, "a") == "g01"
    assert first["n_cells_tested"] == HALF
    assert second["n_cells_tested"] == HALF


def test_the_mask_leaves_the_live_obs_untouched():
    a = _adaptor()
    before = a.adata.obs["grp"].copy()
    a.run_marker_genes(obs_column="grp", top_n=3, active_cell_indices=FIRST)
    assert a.adata.n_obs == N_CELLS
    pd.testing.assert_series_equal(a.adata.obs["grp"], before)


def test_a_group_with_no_cells_in_the_mask_is_left_out():
    a = _adaptor()
    no_c = [i for i in range(N_CELLS) if a.adata.obs["grp"].iloc[i] != "c"]
    r = a.run_marker_genes(obs_column="grp", top_n=3, active_cell_indices=no_c)
    assert [g["group"] for g in r["results"]] == ["a", "b"]


def test_a_checked_group_with_no_cells_in_the_mask_is_not_an_error():
    # The modal lists every category; masking one out must not turn the
    # user's checked groups into "Groups not found".
    a = _adaptor()
    no_c = [i for i in range(N_CELLS) if a.adata.obs["grp"].iloc[i] != "c"]
    r = a.run_marker_genes(obs_column="grp", groups=["a", "b", "c"], top_n=3, active_cell_indices=no_c)
    assert [g["group"] for g in r["results"]] == ["a", "b"]


def test_a_mask_leaving_one_group_is_a_clear_error():
    a = _adaptor()
    only_a = [i for i in range(N_CELLS) if a.adata.obs["grp"].iloc[i] == "a"]
    with pytest.raises(ValueError, match="active cells"):
        a.run_marker_genes(obs_column="grp", top_n=3, active_cell_indices=only_a)


def test_a_group_of_one_cell_gets_no_markers_instead_of_failing_the_run():
    # scanpy refuses a group with a single sample, which a tight mask produces
    # easily; the other groups still deserve their answer.
    a = _adaptor()
    grp = a.adata.obs["grp"].to_numpy()
    one_c = [i for i in range(N_CELLS) if grp[i] != "c"] + [int(np.flatnonzero(grp == "c")[0])]
    r = a.run_marker_genes(obs_column="grp", top_n=3, active_cell_indices=sorted(one_c))
    by_group = {g["group"]: g["genes"] for g in r["results"]}
    assert by_group["c"] == []
    assert by_group["a"] and by_group["b"]


def test_out_of_range_indices_are_rejected():
    with pytest.raises(ValueError, match="out of range"):
        _adaptor().run_marker_genes(obs_column="grp", active_cell_indices=[0, N_CELLS])


def test_the_record_notes_the_run_was_on_a_selection():
    a = _adaptor()
    a.run_marker_genes(obs_column="grp", top_n=3, active_cell_indices=FIRST)
    step = a.analysis_record.steps[-1]
    assert step.action == "marker_genes"
    assert step.n_active == HALF
    # The indices live on the step, not in the params the notebook splats.
    assert "active_cell_indices" not in step.params


def test_the_route_forwards_the_mask(monkeypatch):
    a = _adaptor()
    monkeypatch.setattr(routes, "get_adaptor", lambda dataset=None: a)
    client = TestClient(app)
    res = client.post("/api/marker-genes", json={
        "obs_column": "grp", "top_n": 3, "active_cell_indices": SECOND,
    })
    assert res.status_code == 200, res.text
    body = res.json()
    assert body["n_cells_tested"] == HALF
    assert next(r for r in body["results"] if r["group"] == "a")["genes"][0]["gene"] == "g01"
