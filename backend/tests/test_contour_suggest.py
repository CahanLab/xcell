"""The data-aware facts the Contour modal reasons from.

Sigma is measured in grid pixels, so the modal cannot tell whether a setting is
sane without knowing the spot spacing and the tissue extent. These are the
numbers that make that possible.
"""

import numpy as np
import anndata
import pytest
from scipy.sparse import csr_matrix

from xcell.adaptor import DataAdaptor
from xcell.multicontour import spot_geometry


def _adata(step=2.0, n_side=10):
    """A regular grid, so the nearest-neighbour spacing is exactly `step`."""
    coords = np.array([[i * step, j * step]
                       for i in range(n_side) for j in range(n_side)])
    n = coords.shape[0]
    rng = np.random.default_rng(0)
    ad = anndata.AnnData(X=csr_matrix(rng.poisson(1.0, (n, 4)).astype(np.float32)))
    ad.var_names = ["a", "b", "c", "d"]
    ad.obsm["spatial"] = coords
    return ad


def test_spot_geometry_on_a_regular_grid():
    coords = _adata(step=2.0, n_side=10).obsm["spatial"]
    spacing, extent = spot_geometry(coords)
    assert spacing == pytest.approx(2.0)
    # 10 points at stride 2 spans 0..18 on both axes.
    assert extent == pytest.approx(18.0)


def test_spot_geometry_survives_a_degenerate_input():
    """One spot has no spacing and no extent. None means 'not computable' —
    NaN would not survive the JSON round trip."""
    spacing, extent = spot_geometry(np.array([[1.0, 1.0]]))
    assert spacing is None
    assert extent is None


def test_suggest_contour_params_reports_the_geometry():
    a = DataAdaptor("x.h5ad", adata=_adata(step=2.0, n_side=10))
    out = a.suggest_contour_params()

    assert out["n_spots"] == 100
    assert out["median_spacing"] == pytest.approx(2.0)
    assert out["extent"] == pytest.approx(18.0)
    # Unchanged, so existing callers keep working. suggest_grid_res clamps to
    # [50, 600], so 100 spots gives the floor rather than sqrt(100).
    assert out["grid_res"] == 50
    assert out["smooth_sigma"] > 0


def test_suggest_contour_params_is_json_safe():
    import json
    a = DataAdaptor("x.h5ad", adata=_adata())
    json.dumps(a.suggest_contour_params())   # raises on NaN/inf
