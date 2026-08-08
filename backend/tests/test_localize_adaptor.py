"""The adaptor's side of Localize — the first feature spanning two datasets.

Genes are matched by *name* across the two AnnDatas, which is the whole job the
adaptor does here that the pure module cannot: the pure module takes two aligned
matrices and has no way to know a panel was reordered or only partly shared.
"""
import json

import anndata
import numpy as np
import pandas as pd
import pytest
from scipy.sparse import csr_matrix

from xcell.adaptor import DataAdaptor


def _spatial(n=180, n_genes=24, seed=0):
    """A reference whose expression genuinely encodes position."""
    rng = np.random.default_rng(seed)
    coords = rng.uniform(0, 100, (n, 2))
    u = (coords[:, 0] - 50) / 50
    v = (coords[:, 1] - 50) / 50
    planes = rng.normal(0, 1, (3, n_genes))
    X = np.column_stack([u, v, np.ones(n)]) @ planes
    X = np.maximum(X + 3.0, 0.0) + rng.normal(0, 0.1, (n, n_genes))
    ad = anndata.AnnData(X=csr_matrix(np.maximum(X, 0).astype(np.float32)))
    ad.var_names = [f'g{i}' for i in range(n_genes)]
    ad.obs_names = [f'spot{i}' for i in range(n)]
    ad.obsm['spatial'] = coords
    ad.obs['region'] = pd.Categorical(np.where(coords[:, 0] < 50, 'left', 'right'))
    ad.var['panel'] = True
    return ad, coords


def _query_from(ref_ad, coords, n=60, seed=1):
    """A query drawn from the same generative tissue, at a different depth."""
    rng = np.random.default_rng(seed)
    picks = rng.integers(0, ref_ad.n_obs, n)
    X = np.asarray(ref_ad.X.todense())[picks] * 3.0
    X = X + rng.normal(0, 0.2, X.shape)
    ad = anndata.AnnData(X=csr_matrix(np.maximum(X, 0).astype(np.float32)))
    ad.var_names = list(ref_ad.var_names)
    ad.obs_names = [f'cell{i}' for i in range(n)]
    ad.obs['cell_type'] = pd.Categorical(['a', 'b'] * (n // 2))
    return ad, coords[picks]


def _pair(seed=0):
    ref_ad, coords = _spatial(seed=seed)
    q_ad, truth = _query_from(ref_ad, coords)
    return (
        DataAdaptor('spatial.h5ad', adata=ref_ad),
        DataAdaptor('scrna.h5ad', adata=q_ad),
        truth,
    )


def _run(query, bundle, **kw):
    compute_fn, apply_fn = query.prepare_localize(bundle, **kw)
    return apply_fn(compute_fn())


# --- the reference bundle -------------------------------------------------

def test_the_reference_hands_over_plain_arrays():
    """The route wires two adaptors together; neither reaches into the other's
    AnnData."""
    ref, _, _ = _pair()
    bundle = ref.spatial_reference_bundle()
    assert bundle['expr'].shape == (180, 24)
    assert bundle['coords'].shape == (180, 2)
    assert bundle['genes'] == list(ref.adata.var_names)
    assert bundle['sections'] is None
    json.dumps({k: v for k, v in bundle.items() if k not in ('expr', 'coords', 'sections')})


def test_a_reference_without_coordinates_is_refused():
    ad, _ = _spatial()
    del ad.obsm['spatial']
    with pytest.raises(ValueError, match='spatial'):
        DataAdaptor('x.h5ad', adata=ad).spatial_reference_bundle()


def test_the_bundle_can_be_restricted_to_a_gene_subset():
    ref, _, _ = _pair()
    ref.adata.var['panel'] = [i < 10 for i in range(ref.n_genes)]
    bundle = ref.spatial_reference_bundle(gene_subset='panel')
    assert bundle['expr'].shape[1] == 10
    assert len(bundle['genes']) == 10


def test_the_bundle_carries_section_labels_when_asked():
    ref, _, _ = _pair()
    bundle = ref.spatial_reference_bundle(section_col='region')
    assert bundle['sections'] is not None
    assert set(bundle['sections']) == {'left', 'right'}


def test_an_unknown_section_column_is_refused():
    ref, _, _ = _pair()
    with pytest.raises(ValueError):
        ref.spatial_reference_bundle(section_col='nope')


# --- gene matching --------------------------------------------------------

def test_genes_are_matched_by_name_not_position():
    """A reordered panel is the case that silently produces nonsense if the two
    matrices are simply lined up column-by-column."""
    ref, query, truth = _pair()
    shuffled = query.adata[:, list(reversed(list(query.adata.var_names)))].copy()
    query = DataAdaptor('scrna.h5ad', adata=shuffled)
    result = _run(query, ref.spatial_reference_bundle(), k=10)
    err = np.median(np.linalg.norm(
        query.adata.obsm['X_spatial_pred'] - truth, axis=1))
    assert err < 20


def test_the_overlap_is_reported_before_anything_runs():
    ref, query, _ = _pair()
    report = query.localize_gene_overlap(ref.spatial_reference_bundle())
    assert report['n_shared'] == 24
    assert report['frac_of_reference'] == 1.0
    assert report['missing_from_query'] == []


def test_genes_the_query_lacks_are_named():
    ref, query, _ = _pair()
    query = DataAdaptor('scrna.h5ad', adata=query.adata[:, :20].copy())
    report = query.localize_gene_overlap(ref.spatial_reference_bundle())
    assert report['n_shared'] == 20
    assert set(report['missing_from_query']) == {'g20', 'g21', 'g22', 'g23'}


def test_too_little_overlap_is_a_hard_stop():
    """Twelve shared genes produce a confident-looking map of nothing. Failing
    loudly beats a caveat nobody reads."""
    ref, query, _ = _pair()
    query = DataAdaptor('scrna.h5ad', adata=query.adata[:, :3].copy())
    with pytest.raises(ValueError, match='overlap'):
        _run(query, ref.spatial_reference_bundle(), k=5)


def test_no_shared_genes_at_all_is_refused():
    ref, query, _ = _pair()
    renamed = query.adata.copy()
    renamed.var_names = [f'other{i}' for i in range(renamed.n_vars)]
    with pytest.raises(ValueError):
        _run(DataAdaptor('q.h5ad', adata=renamed), ref.spatial_reference_bundle())


# --- running it -----------------------------------------------------------

def test_the_prediction_lands_in_obsm_and_is_recoverable():
    ref, query, truth = _pair()
    result = _run(query, ref.spatial_reference_bundle(), k=10)
    coords = query.adata.obsm['X_spatial_pred']
    assert coords.shape == (60, 2)
    assert np.median(np.linalg.norm(coords - truth, axis=1)) < 20
    assert result['embedding_name'] == 'X_spatial_pred'
    assert result['n_cells'] == 60


def test_the_new_embedding_shows_up_in_the_schema():
    """Otherwise the user cannot select it and the whole feature is invisible."""
    ref, query, _ = _pair()
    _run(query, ref.spatial_reference_bundle(), k=10)
    assert 'X_spatial_pred' in query.get_schema()['embeddings']


def test_both_confidence_scores_are_written_to_obs():
    ref, query, _ = _pair()
    _run(query, ref.spatial_reference_bundle(), k=10)
    assert 'X_spatial_pred_confidence' in query.adata.obs.columns
    assert 'X_spatial_pred_similarity' in query.adata.obs.columns
    conf = query.adata.obs['X_spatial_pred_confidence']
    assert conf.between(0, 1).all()


def test_the_key_can_be_renamed_so_two_runs_coexist():
    ref, query, _ = _pair()
    bundle = ref.spatial_reference_bundle()
    _run(query, bundle, k=5, key_added='X_k5')
    _run(query, bundle, k=25, key_added='X_k25')
    assert {'X_k5', 'X_k25'} <= set(query.get_schema()['embeddings'])
    assert 'X_k5_confidence' in query.adata.obs.columns


def test_a_section_assignment_is_written_when_sections_are_used():
    ref, query, _ = _pair()
    _run(query, ref.spatial_reference_bundle(section_col='region'), k=10)
    assigned = query.adata.obs['X_spatial_pred_section']
    assert set(assigned.astype(str)) <= {'left', 'right'}


def test_unplaced_cells_are_nan_and_counted():
    ref, query, _ = _pair()
    result = _run(query, ref.spatial_reference_bundle(), k=10, min_confidence=1.01)
    assert result['n_unplaced'] == 60
    assert np.isnan(query.adata.obsm['X_spatial_pred']).all()


def test_a_nan_embedding_still_serves_over_the_api():
    """NaN coordinates are the honest answer for an unplaceable cell, so every
    downstream reader has to cope with them."""
    ref, query, _ = _pair()
    _run(query, ref.spatial_reference_bundle(), k=10, min_confidence=1.01)
    json.dumps(query.get_embedding('X_spatial_pred'), allow_nan=False)


def test_the_result_summarizes_the_confidence_distribution():
    """So the panel can say what a threshold would cost before it is applied."""
    ref, query, _ = _pair()
    result = _run(query, ref.spatial_reference_bundle(), k=10)
    dist = result['confidence_distribution']
    assert set(dist) >= {'median', 'q25', 'q75', 'n_below_0.2', 'n_below_0.5'}
    json.dumps(result, allow_nan=False)


# --- validation -----------------------------------------------------------

def test_prepare_validates_before_the_background_task_starts():
    """A bad parameter must be a 400, not a task that fails a minute later."""
    ref, query, _ = _pair()
    with pytest.raises(ValueError):
        query.prepare_localize(ref.spatial_reference_bundle(), metric='mahalanobis')
    with pytest.raises(ValueError):
        query.prepare_localize(ref.spatial_reference_bundle(), aggregation='vote')
    with pytest.raises(ValueError):
        query.prepare_localize(ref.spatial_reference_bundle(), transform='wavelet')


def test_compute_reports_progress():
    ref, query, _ = _pair()
    compute_fn, apply_fn = query.prepare_localize(ref.spatial_reference_bundle(), k=10)
    seen = []
    apply_fn(compute_fn(lambda frac, msg: seen.append(frac)))
    assert seen and seen[-1] == 1.0


def test_the_run_is_recorded_in_the_analysis_record():
    ref, query, _ = _pair()
    _run(query, ref.spatial_reference_bundle(), k=10)
    step = query.analysis_record.steps[-1]
    assert step.action == 'localize'
    assert step.params['k'] == 10
    json.dumps(query.analysis_record.to_dict(), allow_nan=False)


# --- evaluation through the adaptor --------------------------------------

def test_cross_validation_runs_on_the_reference_alone():
    ref, _, _ = _pair()
    compute_fn, apply_fn = ref.prepare_localize_cross_validation(k=10)
    metrics = apply_fn(compute_fn())
    assert metrics['same_platform'] is True
    assert metrics['median_error'] < metrics['baseline_centroid']
    json.dumps(metrics, allow_nan=False)


def test_cross_validation_can_break_error_down_by_a_column():
    ref, _, _ = _pair()
    compute_fn, apply_fn = ref.prepare_localize_cross_validation(k=10, groupby='region')
    metrics = apply_fn(compute_fn())
    assert set(metrics['per_group']) == {'left', 'right'}


def test_a_prediction_can_be_scored_against_known_truth():
    ref, query, truth = _pair()
    _run(query, ref.spatial_reference_bundle(), k=10)
    query.adata.obsm['spatial_true'] = truth
    metrics = query.evaluate_localization('X_spatial_pred', 'spatial_true',
                                          groupby='cell_type')
    assert metrics['median_error'] < metrics['baseline_centroid']
    assert set(metrics['per_group']) == {'a', 'b'}
    json.dumps(metrics, allow_nan=False)


def test_scoring_against_a_missing_truth_key_is_an_error():
    ref, query, _ = _pair()
    _run(query, ref.spatial_reference_bundle(), k=10)
    with pytest.raises(KeyError):
        query.evaluate_localization('X_spatial_pred', 'not_there')
