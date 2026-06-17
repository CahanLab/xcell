"""Tests for the two-phase multicontour adaptor methods."""

import numpy as np
import anndata
from scipy.sparse import csr_matrix

from xcell.adaptor import DataAdaptor


def _adata(with_pca=True):
    coords = np.array([[float(i), float(j)] for i in range(5) for j in range(5)])
    n = coords.shape[0]
    X = np.zeros((n, 4), dtype=np.float32)
    for k, (i, j) in enumerate([(i, j) for i in range(5) for j in range(5)]):
        if i <= 1:
            X[k, 0] = X[k, 1] = 10.0
        if i >= 3:
            X[k, 2] = X[k, 3] = 10.0
    ad = anndata.AnnData(X=csr_matrix(X))
    ad.var_names = ["a0", "a1", "b0", "b1"]
    ad.obsm["spatial"] = coords
    if with_pca:
        ad.obsm["X_pca"] = coords.copy()
    return ad


def test_prepare_requires_pca():
    a = DataAdaptor("x.h5ad", adata=_adata(with_pca=False))
    try:
        a.prepare_multicontour({"A": ["a0", "a1"], "B": ["b0", "b1"]}, contour_levels=3)
        assert False, "expected ValueError"
    except ValueError as e:
        assert "PCA" in str(e)


def test_prepare_requires_two_sets():
    a = DataAdaptor("x.h5ad", adata=_adata())
    try:
        a.prepare_multicontour({"A": ["a0", "a1"]}, contour_levels=3)
        assert False, "expected ValueError"
    except ValueError as e:
        assert "2" in str(e)


def test_prepare_then_finalize_writes_column():
    a = DataAdaptor("x.h5ad", adata=_adata())
    prep = a.prepare_multicontour(
        {"A": ["a0", "a1"], "B": ["b0", "b1"]}, contour_levels=3,
        grid_res=40, smooth_sigma=2.0)
    token = prep["token"]
    assert len(prep["modules"]) == 2
    cutoffs = {m["name"]: m["auto_cutoff"] for m in prep["modules"]}
    res = a.finalize_multicontour(token=token, cutoffs=cutoffs, profile_k=5,
                                  out_name="tissue", save_qc=True, params=prep["params"])
    assert res["annotation_key"] == "tissue"
    assert "tissue" in a.adata.obs
    cats = set(a.adata.obs["tissue"].cat.categories)
    assert {"A", "B", "unassigned"}.issubset(cats)
    assert "tissue_status" in a.adata.obs
    assert "A_high" in a.adata.obs and "B_high" in a.adata.obs


def _two_section_adata():
    cells, secs = [], []
    for i in range(10):
        for j in range(10):
            cells.append((float(i), float(j)))
            secs.append("A" if i < 5 else "B")
    coords = np.array(cells)
    n = coords.shape[0]
    sections = np.array(secs)
    X = np.zeros((n, 2), dtype=np.float32)
    X[sections == "A", 0] = 10.0
    X[sections == "B", 1] = 10.0
    ad = anndata.AnnData(X=csr_matrix(X))
    ad.var_names = ["mA", "mB"]
    ad.obsm["spatial"] = coords
    ad.obsm["X_pca"] = coords.copy()
    ad.obs["section"] = sections
    return ad


# smooth_sigma chosen so global processing bleeds gene mA across the A/B
# boundary far enough to cross a contour band in section B; section-awareness
# must prevent that.
_BLEED_SIGMA = 6.0


def test_run_contourize_section_col_no_bleed():
    import numpy as _np
    a = DataAdaptor("x.h5ad", adata=_two_section_adata())
    # gene mA high in section A only
    a.run_contourize(["mA"], contour_levels=2, log_transform=False, grid_res=40,
                     smooth_sigma=_BLEED_SIGMA, annotation_key="c_sec", section_col="section")
    bands = _np.asarray(a.adata.obs["c_sec"].astype(float))
    sec = a.adata.obs["section"].values
    # section B should stay at the zero band (no bleed from A)
    assert bands[sec == "B"].max() == 0.0


def test_run_contourize_bleeds_without_section_col():
    import numpy as _np
    a = DataAdaptor("x.h5ad", adata=_two_section_adata())
    a.run_contourize(["mA"], contour_levels=2, log_transform=False, grid_res=40,
                     smooth_sigma=_BLEED_SIGMA, annotation_key="c_glob")
    bands = _np.asarray(a.adata.obs["c_glob"].astype(float))
    sec = a.adata.obs["section"].values
    assert bands[sec == "B"].max() > 0.0  # global processing bleeds into B


def test_prepare_multicontour_section_col_in_params():
    a = DataAdaptor("x.h5ad", adata=_two_section_adata())
    prep = a.prepare_multicontour(
        {"A": ["mA"], "B": ["mB"]}, contour_levels=2, grid_res=40,
        smooth_sigma=2.0, log_transform=False, section_col="section")
    assert prep["params"]["section_col"] == "section"
    cutoffs = {m["name"]: m["auto_cutoff"] for m in prep["modules"]}
    res = a.finalize_multicontour(token=prep["token"], cutoffs=cutoffs,
                                  out_name="tissue", params=prep["params"])
    assert res["annotation_key"] == "tissue"
    assert "tissue" in a.adata.obs


def test_suggest_contour_params():
    a = DataAdaptor("x.h5ad", adata=_adata())
    s = a.suggest_contour_params()
    assert s["grid_res"] >= 50 and s["grid_res"] <= 600
    assert s["smooth_sigma"] > 0


def test_suggest_contour_params_requires_spatial():
    ad = _adata()
    del ad.obsm["spatial"]
    a = DataAdaptor("x.h5ad", adata=ad)
    try:
        a.suggest_contour_params()
        assert False, "expected ValueError"
    except ValueError as e:
        assert "spatial" in str(e).lower()


def test_finalize_recomputes_on_cache_miss():
    a = DataAdaptor("x.h5ad", adata=_adata())
    prep = a.prepare_multicontour(
        {"A": ["a0", "a1"], "B": ["b0", "b1"]}, contour_levels=3,
        grid_res=40, smooth_sigma=2.0)
    cutoffs = {m["name"]: m["auto_cutoff"] for m in prep["modules"]}
    # Simulate cache loss (e.g. server restart).
    a._multicontour_cache.clear()
    res = a.finalize_multicontour(token="missing", cutoffs=cutoffs, profile_k=5,
                                  out_name="tissue2", save_qc=False, params=prep["params"])
    assert res["annotation_key"] == "tissue2"
    assert "tissue2" in a.adata.obs
