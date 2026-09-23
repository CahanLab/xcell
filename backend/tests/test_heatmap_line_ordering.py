"""Ordering heatmap columns by a drawn line.

The three line-aware orderings ("line_position", "line_distance" and
"category_then_position") all go through the adaptor's projection helper, which
takes *coordinates* and returns a ``(positions, distances)`` pair. These tests
pin that contract down from the heatmap side, and check the ordering actually
sorts cells rather than merely returning without raising.
"""

import anndata
import numpy as np
import pandas as pd
from fastapi.testclient import TestClient
from scipy.sparse import csr_matrix

from xcell.adaptor import DataAdaptor
from xcell.api import routes
from xcell.heatmap import compute_heatmap_data
from xcell.main import app


N_CELLS = 40
LINE = [[-2.0, -2.0], [2.0, 2.0]]


def _adata():
    """Cells along y=x, shuffled so an unsorted result is distinguishable.

    ``up`` rises with position along the line, ``down`` falls, and ``far``
    rises with perpendicular distance from it — one gene per ordering so each
    assertion has a signal that only the right sort produces.
    """
    rng = np.random.default_rng(0)
    t = np.linspace(0.0, 1.0, N_CELLS)
    offset = np.linspace(-1.0, 1.0, N_CELLS)
    rng.shuffle(offset)
    coords = np.column_stack([
        t * 4.0 - 2.0 - offset,
        t * 4.0 - 2.0 + offset,
    ])

    X = np.column_stack([
        5.0 * t,                       # up
        5.0 * (1.0 - t),               # down
        5.0 * np.abs(offset),          # far (distance from the diagonal)
        rng.normal(3.0, 0.1, N_CELLS),  # flat
    ])
    ad = anndata.AnnData(X=csr_matrix(np.maximum(X, 0.0).astype(np.float32)))
    ad.var_names = ["up", "down", "far", "flat"]
    # Two halves that interleave along the line, so category ordering and
    # position ordering disagree and "category then position" is testable.
    half = np.array(["a" if i % 2 == 0 else "b" for i in range(N_CELLS)])
    ad.obs["grp"] = pd.Categorical(half)
    ad.obsm["X_pca"] = coords

    order = rng.permutation(N_CELLS)
    return ad[order].copy()


def _adaptor():
    a = DataAdaptor("x.h5ad", adata=_adata())
    a.set_lines([{"name": "L", "embeddingName": "X_pca", "points": LINE}])
    return a


def _row(result, gene):
    return np.array(result["matrix"][result["row_labels"].index(gene)])


# ---------- ordering by position along the line ----------

def test_line_position_orders_cells_along_the_line():
    r = compute_heatmap_data(_adaptor(), genes=["up", "down", "flat"],
                             cell_ordering="line_position", line_name="L")

    assert r["n_cells"] == N_CELLS
    up = _row(r, "up")
    assert np.all(np.diff(up) >= -1e-9), "up should rise monotonically once sorted"
    assert up[0] < up[-1]
    down = _row(r, "down")
    assert np.all(np.diff(down) <= 1e-9)


def test_line_distance_orders_cells_by_distance_from_the_line():
    r = compute_heatmap_data(_adaptor(), genes=["far", "up"],
                             cell_ordering="line_distance", line_name="L")

    far = _row(r, "far")
    assert np.all(np.diff(far) >= -1e-9), "far should rise with distance once sorted"
    # Position-linked genes must NOT come out sorted under a distance ordering.
    assert not np.all(np.diff(_row(r, "up")) >= -1e-9)


def test_category_then_position_sorts_within_each_category():
    r = compute_heatmap_data(_adaptor(), genes=["up"], obs_column="grp",
                             cell_ordering="category_then_position", line_name="L")

    groups = {g["name"]: g for g in r["column_groups"]}
    assert set(groups) == {"a", "b"}
    up = _row(r, "up")
    for g in groups.values():
        block = up[g["start"]:g["start"] + g["size"]]
        assert np.all(np.diff(block) >= -1e-9), f"{g['name']} not sorted by position"


# ---------- interaction with the rest of the pipeline ----------

def test_line_ordering_respects_a_cell_subset():
    """cell_indices must select cells, not index into a reordered projection."""
    a = _adaptor()
    coords = a.adata.obsm["X_pca"]
    keep = np.where(coords[:, 0] + coords[:, 1] < 0)[0]  # first half of the line
    r = compute_heatmap_data(a, genes=["up"], cell_ordering="line_position",
                             line_name="L", cell_indices=keep.tolist())

    assert r["n_cells"] == len(keep)
    up = _row(r, "up")
    assert np.all(np.diff(up) >= -1e-9)


def test_line_ordering_uses_the_columns_the_line_was_drawn_on():
    """A line drawn on columns 2/3 must not be projected against columns 0/1."""
    ad = _adata()
    n = ad.n_obs
    # Pad the embedding so columns 2/3 hold the real coordinates and columns
    # 0/1 hold noise that would produce a different ordering.
    rng = np.random.default_rng(1)
    ad.obsm["X_wide"] = np.column_stack([
        rng.normal(0, 1, n), rng.normal(0, 1, n), ad.obsm["X_pca"],
    ])
    a = DataAdaptor("x.h5ad", adata=ad)
    a.set_lines([{"name": "L", "embeddingName": "X_wide", "points": LINE,
                  "dimX": 2, "dimY": 3}])

    r = compute_heatmap_data(a, genes=["up"], cell_ordering="line_position",
                             line_name="L")
    assert np.all(np.diff(_row(r, "up")) >= -1e-9)


def test_degenerate_line_does_not_raise():
    """A line whose points coincide has no direction; it must still return."""
    a = DataAdaptor("x.h5ad", adata=_adata())
    a.set_lines([{"name": "L", "embeddingName": "X_pca",
                  "points": [[1.0, 1.0], [1.0, 1.0]]}])

    r = compute_heatmap_data(a, genes=["up"], cell_ordering="line_position",
                             line_name="L")
    assert r["n_cells"] == N_CELLS


def test_unknown_line_falls_back_to_unordered():
    a = _adaptor()
    r = compute_heatmap_data(a, genes=["up"], cell_ordering="line_position",
                             line_name="nope")
    assert r["n_cells"] == N_CELLS


# ---------- through the route ----------

def _install(monkeypatch, a):
    monkeypatch.setattr(routes, "get_adaptor", lambda dataset=None: a)
    return TestClient(app)


def test_route_returns_a_matrix_for_line_position(monkeypatch):
    client = _install(monkeypatch, _adaptor())
    resp = client.post("/api/heatmap/data", json={
        "genes": ["up", "down"], "cell_ordering": "line_position",
        "line_name": "L", "gene_ordering": "peak_position", "n_bins": 10,
    })

    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body["n_bins"] == 10
    assert len(body["matrix"]) == 2
    assert body["row_labels"] == ["down", "up"]  # peak-position ordering
