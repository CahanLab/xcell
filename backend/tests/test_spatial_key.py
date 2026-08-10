"""Which .obsm holds this dataset's spatial coordinates.

Auto-detection looked for ``spatial`` or ``X_spatial`` and nothing else, so a map
produced by Localize — written to ``X_spatial_pred`` — was invisible to every
spatial tool: Spatial Neighbors, Contour, Define Sections and Ligand-Receptor
all refused to run on the coordinates the user had just created. Naming the key
explicitly did not help either, because the prerequisite gate ran first.

The choice has to be *explicit* rather than a wider auto-detect: a query that has
been localized several ways carries several predicted maps, and picking one for
the user would be a guess about which analysis they meant.
"""
import anndata
import numpy as np
import pytest
from scipy.sparse import csr_matrix

from xcell.adaptor import DataAdaptor


def _adata(n=60, seed=0):
    rng = np.random.default_rng(seed)
    ad = anndata.AnnData(X=csr_matrix(rng.poisson(1.0, (n, 6)).astype(np.float32)))
    ad.var_names = [f'g{i}' for i in range(6)]
    ad.obs_names = [f'c{i}' for i in range(n)]
    return ad


def _localized(n=60, seed=0):
    """What Localize leaves behind: coordinates plus its two companion scores."""
    rng = np.random.default_rng(seed)
    ad = _adata(n, seed)
    ad.obsm['X_spatial_pred'] = rng.uniform(0, 100, (n, 2))
    ad.obsm['X_spatial_pred_inj'] = rng.uniform(0, 100, (n, 2))
    ad.obs['X_spatial_pred_confidence'] = rng.random(n)
    ad.obs['X_spatial_pred_similarity'] = rng.random(n)
    ad.obs['X_spatial_pred_inj_confidence'] = rng.random(n)
    ad.obs['X_spatial_pred_inj_similarity'] = rng.random(n)
    ad.obsm['X_umap'] = rng.normal(0, 1, (n, 2))
    return ad


def _measured(n=60, seed=0):
    ad = _adata(n, seed)
    ad.obsm['spatial'] = np.random.default_rng(seed).uniform(0, 100, (n, 2))
    return ad


# --- the bug ---------------------------------------------------------------

def test_a_predicted_map_is_not_spatial_until_it_is_chosen():
    """Auto-detection must not start guessing: this dataset has two predicted
    maps and nothing on disk says which one the user means."""
    a = DataAdaptor('q.h5ad', adata=_localized())
    assert a._get_spatial_key() is None
    assert a.check_prerequisites('spatial_neighbors')['missing'] == ['has_spatial']


def test_choosing_a_predicted_map_unblocks_the_spatial_tools():
    a = DataAdaptor('q.h5ad', adata=_localized())
    a.set_spatial_key('X_spatial_pred_inj')
    assert a._get_spatial_key() == 'X_spatial_pred_inj'
    assert a._has_spatial_coordinates() is True
    assert a.check_prerequisites('spatial_neighbors')['satisfied'] is True
    assert a.check_prerequisites('contourize')['satisfied'] is True


def test_the_choice_reaches_the_tool_that_reads_the_coordinates():
    """The gate opening is not enough — the run has to use the chosen array."""
    ad = _localized()
    a = DataAdaptor('q.h5ad', adata=ad)
    a.set_spatial_key('X_spatial_pred_inj')
    compute_fn, apply_fn = a.prepare_spatial_neighbors(n_neighs=4)
    apply_fn(compute_fn())
    assert 'spatial_connectivities' in a.adata.obsp
    assert a.adata.obsp['spatial_connectivities'].shape == (ad.n_obs, ad.n_obs)


def test_switching_between_two_predicted_maps_changes_the_answer():
    """Comparing maps is the whole point of having several; the graph built from
    one must not silently persist as the graph for the other."""
    a = DataAdaptor('q.h5ad', adata=_localized())
    a.set_spatial_key('X_spatial_pred')
    first = a.prepare_spatial_neighbors(n_neighs=4)[0]()
    a.set_spatial_key('X_spatial_pred_inj')
    second = a.prepare_spatial_neighbors(n_neighs=4)[0]()
    assert first['spatial_key'] == 'X_spatial_pred'
    assert second['spatial_key'] == 'X_spatial_pred_inj'
    assert not np.array_equal(
        first['spatial_connectivities'].toarray(),
        second['spatial_connectivities'].toarray(),
    )


# --- it must not break datasets that really have coordinates ---------------

def test_a_measured_dataset_needs_no_choice():
    a = DataAdaptor('ref.h5ad', adata=_measured())
    assert a._get_spatial_key() == 'spatial'
    assert a._has_spatial_coordinates() is True


def test_an_explicit_choice_overrides_auto_detection():
    ad = _measured()
    ad.obsm['X_spatial_pred'] = np.random.default_rng(1).uniform(0, 5, (ad.n_obs, 2))
    a = DataAdaptor('ref.h5ad', adata=ad)
    assert a._get_spatial_key() == 'spatial'
    a.set_spatial_key('X_spatial_pred')
    assert a._get_spatial_key() == 'X_spatial_pred'


def test_clearing_the_choice_returns_to_auto_detection():
    ad = _measured()
    ad.obsm['X_spatial_pred'] = np.random.default_rng(1).uniform(0, 5, (ad.n_obs, 2))
    a = DataAdaptor('ref.h5ad', adata=ad)
    a.set_spatial_key('X_spatial_pred')
    a.set_spatial_key(None)
    assert a._get_spatial_key() == 'spatial'


def test_a_stale_choice_does_not_strand_the_dataset():
    """The key is stored in .uns, so it survives a save and can name an .obsm
    that a later session no longer has. That must degrade to auto-detection
    rather than making every spatial tool unusable."""
    ad = _measured()
    ad.uns['xcell_spatial_key'] = 'X_spatial_pred_from_a_previous_life'
    a = DataAdaptor('ref.h5ad', adata=ad)
    assert a._get_spatial_key() == 'spatial'


# --- validation ------------------------------------------------------------

def test_choosing_something_that_is_not_there_is_refused():
    a = DataAdaptor('q.h5ad', adata=_localized())
    with pytest.raises(ValueError, match='not found'):
        a.set_spatial_key('X_nope')


def test_a_one_dimensional_obsm_cannot_be_coordinates():
    ad = _localized()
    ad.obsm['X_one'] = np.arange(ad.n_obs, dtype=float).reshape(-1, 1)
    a = DataAdaptor('q.h5ad', adata=ad)
    with pytest.raises(ValueError, match='at least two'):
        a.set_spatial_key('X_one')


# --- what the picker needs to show -----------------------------------------

def test_the_options_name_every_usable_embedding():
    a = DataAdaptor('q.h5ad', adata=_localized())
    out = a.spatial_key_options()
    assert {o['key'] for o in out['options']} == {
        'X_spatial_pred', 'X_spatial_pred_inj', 'X_umap'}
    assert out['current'] is None and out['explicit'] is None


def test_the_options_say_which_maps_are_predicted():
    """A predicted map is a different claim from a measured one, so the picker
    has to be able to say so rather than listing them all alike."""
    a = DataAdaptor('q.h5ad', adata=_localized())
    by_key = {o['key']: o for o in a.spatial_key_options()['options']}
    assert by_key['X_spatial_pred']['predicted'] is True
    assert by_key['X_spatial_pred_inj']['predicted'] is True
    assert by_key['X_umap']['predicted'] is False


def test_the_options_count_cells_with_no_coordinate():
    """min_confidence leaves NaN coordinates on purpose. A spatial graph built
    over those is meaningless, so the count has to be visible *before* the
    choice rather than discovered as a strange result afterwards."""
    ad = _localized()
    ad.obsm['X_spatial_pred'][:7] = np.nan
    a = DataAdaptor('q.h5ad', adata=ad)
    by_key = {o['key']: o for o in a.spatial_key_options()['options']}
    assert by_key['X_spatial_pred']['n_missing'] == 7
    assert by_key['X_spatial_pred_inj']['n_missing'] == 0


def test_the_options_report_the_current_choice():
    a = DataAdaptor('q.h5ad', adata=_localized())
    a.set_spatial_key('X_spatial_pred')
    out = a.spatial_key_options()
    assert out['current'] == 'X_spatial_pred'
    assert out['explicit'] == 'X_spatial_pred'


def test_the_options_are_json_safe():
    import json
    a = DataAdaptor('q.h5ad', adata=_localized())
    json.dumps(a.spatial_key_options())


# --- the route contract ----------------------------------------------------

from fastapi.testclient import TestClient          # noqa: E402

from xcell.api import routes                        # noqa: E402
from xcell.main import app                          # noqa: E402

client = TestClient(app)


def _install(ad=None):
    routes.set_adaptor(DataAdaptor('q.h5ad', adata=ad or _localized()), slot='primary')


def test_the_route_lists_the_candidates():
    _install()
    r = client.get('/api/spatial_key')
    assert r.status_code == 200, r.text
    body = r.json()
    assert body['current'] is None
    assert {o['key'] for o in body['options']} == {
        'X_spatial_pred', 'X_spatial_pred_inj', 'X_umap'}


def test_the_route_sets_the_key_and_unblocks_the_prerequisite():
    _install()
    r = client.put('/api/spatial_key', json={'key': 'X_spatial_pred_inj'})
    assert r.status_code == 200, r.text
    assert r.json()['current'] == 'X_spatial_pred_inj'
    prereq = client.get('/api/scanpy/prerequisites/spatial_neighbors').json()
    assert prereq['satisfied'] is True
    assert client.get('/api/scanpy/has_spatial').json() == {
        'has_spatial': True, 'spatial_key': 'X_spatial_pred_inj'}


def test_the_route_clears_the_key():
    _install()
    client.put('/api/spatial_key', json={'key': 'X_spatial_pred'})
    assert client.put('/api/spatial_key', json={'key': None}).json()['current'] is None


def test_an_unusable_key_is_a_400():
    _install()
    r = client.put('/api/spatial_key', json={'key': 'X_nope'})
    assert r.status_code == 400
    assert 'X_nope' in r.json()['detail']


def test_the_route_response_is_json_safe():
    import json
    _install()
    json.dumps(client.get('/api/spatial_key').json())
