"""Adaptor tests for the neighborhood composition/enrichment feature."""
import json

import anndata
import numpy as np
import pandas as pd
import pytest
from scipy.sparse import csr_matrix

from xcell.adaptor import DataAdaptor


def _adata(with_section: bool = False):
    """AB checkerboard at origin + distant all-C block (planted attraction)."""
    xs, ys = np.meshgrid(np.arange(8, dtype=float), np.arange(8, dtype=float))
    ab = np.column_stack([xs.ravel(), ys.ravel()])
    c = ab + np.array([1000.0, 0.0])
    coords = np.vstack([ab, c])
    parity = (ab[:, 0] + ab[:, 1]) % 2
    labels = ["A" if p == 0 else "B" for p in parity] + ["C"] * len(c)

    rng = np.random.default_rng(0)
    X = csr_matrix(rng.poisson(1.0, size=(len(coords), 5)).astype(np.float32))
    ad = anndata.AnnData(X=X)
    ad.obs["cell_type"] = pd.Categorical(labels)
    ad.obs["score"] = rng.random(len(coords))
    ad.obsm["spatial"] = coords
    if with_section:
        half = ["s1"] * len(ab) + ["s2"] * len(c)
        ad.obs["section"] = pd.Categorical(half)
    return ad


def _adaptor(**kw) -> DataAdaptor:
    return DataAdaptor("x.h5ad", adata=_adata(**kw))


def _run(a: DataAdaptor, **kw):
    compute_fn, apply_fn = a.prepare_neighborhood(
        column="cell_type", n_neighs=6, n_perms=100, seed=0, **kw
    )
    result = compute_fn(lambda frac, message=None: None)
    return apply_fn(result)


def test_prepare_requires_spatial():
    ad = _adata()
    del ad.obsm["spatial"]
    a = DataAdaptor("x.h5ad", adata=ad)
    with pytest.raises(ValueError):
        a.prepare_neighborhood(column="cell_type")


def test_prepare_rejects_bad_columns():
    a = _adaptor()
    with pytest.raises(ValueError):
        a.prepare_neighborhood(column="nope")
    with pytest.raises(ValueError):
        a.prepare_neighborhood(column="score")  # continuous
    ad = _adata()
    ad.obs["flat"] = pd.Categorical(["only"] * ad.n_obs)
    with pytest.raises(ValueError):
        DataAdaptor("x.h5ad", adata=ad).prepare_neighborhood(column="flat")


def test_prepare_rejects_missing_labels():
    ad = _adata()
    col = ad.obs["cell_type"].copy()
    col[col.index[0]] = np.nan
    ad.obs["cell_type"] = col
    with pytest.raises(ValueError):
        DataAdaptor("x.h5ad", adata=ad).prepare_neighborhood(column="cell_type")


def test_prepare_rejects_bad_section_col():
    a = _adaptor()
    with pytest.raises(ValueError):
        a.prepare_neighborhood(column="cell_type", section_col="nope")


def test_compute_apply_writes_state_and_returns_json():
    a = _adaptor()
    payload = _run(a)

    # Returned payload: planted A-B attraction, JSON-safe throughout.
    assert payload["categories"] == ["A", "B", "C"]
    z = np.array(payload["zscores"])
    assert z[0, 1] > 2.0 and z[0, 2] < -2.0
    json.dumps(payload, allow_nan=False)
    assert "cell_composition" not in payload

    # AnnData state: obsm matrix + score-matrix registry + uns result.
    comp = a.adata.obsm["neighborhood_composition"]
    assert comp.shape == (a.adata.n_obs, 3)
    reg = a.adata.uns["xcell_score_matrices"]["neighborhood_composition"]
    assert list(reg["columns"]) == ["A", "B", "C"]
    stored = a.adata.uns["xcell_neighborhood"]
    assert stored["categories"] == ["A", "B", "C"]
    assert stored["params"]["column"] == "cell_type"
    assert stored["params"]["n_neighs"] == 6


def test_axis_order_follows_categorical_order():
    ad = _adata()
    ad.obs["cell_type"] = ad.obs["cell_type"].cat.reorder_categories(["C", "B", "A"])
    a = DataAdaptor("x.h5ad", adata=ad)
    payload = _run(a)
    assert payload["categories"] == ["C", "B", "A"]


def test_section_col_threads_through():
    a = _adaptor(with_section=True)
    payload = _run(a, section_col="section")
    assert payload["params"]["section_col"] == "section"
    z = np.array(payload["zscores"])
    assert z[0, 1] > 2.0  # attraction still found within section s1


def test_get_neighborhood_result_roundtrip():
    a = _adaptor()
    assert a.get_neighborhood_result() is None
    _run(a)
    stored = a.get_neighborhood_result()
    assert stored is not None
    assert stored["categories"] == ["A", "B", "C"]
    json.dumps(stored, allow_nan=False)
