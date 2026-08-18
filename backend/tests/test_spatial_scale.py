"""Physical scale of spatial coordinates: uns['xcell_spatial_scale'].

The h5ad carries no units — Curio Seeker writes µm, plain Visium writes
image pixels, Localize writes whatever the reference used — so the scale is
an explicit per-dataset setting, with one auto-detection: the Visium HD
loader builds its coordinates in µm and says so in uns['spatial'].
"""
import numpy as np
import pandas as pd
import anndata
import pytest
from scipy.sparse import csr_matrix

from xcell.adaptor import DataAdaptor
from xcell.api import routes


def _adata(uns_spatial: dict | None = None) -> anndata.AnnData:
    rng = np.random.default_rng(0)
    a = anndata.AnnData(
        X=csr_matrix(rng.poisson(1.0, size=(10, 5)).astype(np.float32)),
        obs=pd.DataFrame(index=[f'c{i}' for i in range(10)]),
        var=pd.DataFrame(index=[f'g{i}' for i in range(5)]),
    )
    a.obsm['spatial'] = rng.uniform(0, 100, size=(10, 2))
    if uns_spatial is not None:
        a.uns['spatial'] = uns_spatial
    return a


def _adaptor(**kw) -> DataAdaptor:
    return DataAdaptor('dummy.h5ad', adata=_adata(**kw))


def test_scale_is_unset_by_default():
    a = _adaptor()
    assert a.spatial_scale() == {'um_per_unit': None, 'source': None}


def test_set_scale_persists_to_uns_and_reads_back():
    a = _adaptor()
    out = a.spatial_scale_set(2.5)
    assert out == {'um_per_unit': 2.5, 'source': 'user'}
    assert a.adata.uns['xcell_spatial_scale'] == {'um_per_unit': 2.5}
    assert a.spatial_scale() == {'um_per_unit': 2.5, 'source': 'user'}


def test_setting_none_clears_back_to_unset():
    a = _adaptor()
    a.spatial_scale_set(1.0)
    out = a.spatial_scale_set(None)
    assert out == {'um_per_unit': None, 'source': None}
    assert 'xcell_spatial_scale' not in a.adata.uns


@pytest.mark.parametrize('bad', [0, -1.5, float('nan'), float('inf')])
def test_nonpositive_or_nonfinite_scale_is_rejected(bad):
    a = _adaptor()
    with pytest.raises(ValueError):
        a.spatial_scale_set(bad)


def test_visium_hd_coordinates_are_recognized_as_microns():
    # visium_hd.py builds obsm['spatial'] in µm and records bin_size_um.
    a = _adaptor(uns_spatial={'bin_size_um': 8, 'sample_id': 's1'})
    assert a.spatial_scale() == {'um_per_unit': 1.0, 'source': 'visium_hd'}


def test_user_setting_overrides_visium_hd_autodetect():
    a = _adaptor(uns_spatial={'bin_size_um': 8})
    a.spatial_scale_set(0.5)
    assert a.spatial_scale() == {'um_per_unit': 0.5, 'source': 'user'}


def test_routes_get_and_put(monkeypatch):
    a = _adaptor()
    monkeypatch.setattr(routes, 'get_adaptor', lambda dataset=None: a)
    assert routes.get_spatial_scale(dataset=None)['um_per_unit'] is None
    out = routes.put_spatial_scale(
        routes.SpatialScaleRequest(um_per_unit=1.0), dataset=None)
    assert out == {'um_per_unit': 1.0, 'source': 'user'}
