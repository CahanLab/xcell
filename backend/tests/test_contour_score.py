"""Characterization test for the extracted contour scoring core."""

import numpy as np
import anndata
from scipy.sparse import csr_matrix

from xcell.adaptor import _contour_score_field


def _toy_adata():
    rng = np.random.default_rng(0)
    coords = np.array([[float(i), float(j)] for i in range(6) for j in range(6)])
    n = coords.shape[0]
    X = csr_matrix(rng.integers(0, 5, size=(n, 3)).astype(np.float32))
    ad = anndata.AnnData(X=X)
    ad.var_names = ["g0", "g1", "g2"]
    ad.obsm["spatial"] = coords
    return ad


def test_score_field_shape_and_range():
    ad = _toy_adata()
    expr = {g: np.asarray(ad[:, g].X.todense()).ravel() for g in ["g0", "g1"]}
    score, vmax = _contour_score_field(
        coords=ad.obsm["spatial"],
        gene_expr=expr,
        log_transform=True,
        clip_percentiles=(1, 99),
        grid_res=50,
        smooth_sigma=2.0,
    )
    assert score.shape == (ad.n_obs,)
    assert np.isfinite(score).all()
    assert vmax >= score.max() - 1e-9
