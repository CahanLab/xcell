"""Batch enrichment across a column's groups: GSEA and marker-gene ORA."""
import json

import anndata
import numpy as np
import pandas as pd
import pytest
from scipy.sparse import csr_matrix

from xcell import gene_set_sources as gss
from xcell.adaptor import DataAdaptor

GENES = ['Col1a1', 'Col1a2', 'Col3a1', 'Ptprc', 'Cd3e', 'Cd19', 'Hoxd13', 'Meis1'] + [f'G{i}' for i in range(24)]
LIB = [{'source': 'msigdb', 'id': 'toy', 'species': 'mouse'}]


@pytest.fixture(autouse=True)
def _cache(tmp_path, monkeypatch):
    gss.set_cache_dir(tmp_path / 'cache')
    monkeypatch.setattr(gss, 'fetch_text', lambda url, **kw: (_ for _ in ()).throw(AssertionError(url)))
    monkeypatch.setattr(gss, 'fetch_json', lambda url, **kw: (_ for _ in ()).throw(AssertionError(url)))
    gss.save_library({
        'source': 'msigdb', 'id': 'toy', 'name': 'Toy', 'species': 'mouse', 'version': '1', 'n_sets': 3,
        'sets': [{'name': 'COLLAGEN', 'genes': ['Col1a1', 'Col1a2', 'Col3a1', 'G0', 'G1']},
                 {'name': 'IMMUNE', 'genes': ['Ptprc', 'Cd3e', 'Cd19', 'G2', 'G3']},
                 {'name': 'HOX', 'genes': ['Hoxd13', 'Meis1', 'G4', 'G5', 'G6']}]})
    yield
    gss.set_cache_dir(None)


def _adata(n_per=(30, 30, 30, 1)):
    """Groups a (Col*), b (immune), c (Hox) and a one-cell group d."""
    rng = np.random.default_rng(0)
    labels = np.concatenate([[g] * n for g, n in zip('abcd', n_per)])
    n = labels.size
    lam = np.full((n, len(GENES)), 1.0)
    for g, hi in (('a', ('Col1a1', 'Col1a2', 'Col3a1')), ('b', ('Ptprc', 'Cd3e', 'Cd19')), ('c', ('Hoxd13', 'Meis1'))):
        for gene in hi:
            lam[labels == g, GENES.index(gene)] = 8.0
    ad = anndata.AnnData(X=csr_matrix(rng.poisson(lam).astype(np.float32)))
    ad.var_names = GENES
    ad.obs['grp'] = pd.Categorical(labels)
    return ad


def _run(a, **kw):
    compute_fn, apply_fn = a.prepare_gsea_batch('grp', libraries=LIB, min_set_size=2, n_perm=60, **kw)
    return apply_fn(compute_fn(lambda f, m: None))


def test_gsea_batch_stores_members_and_collection_and_skips_tiny_group():
    a = DataAdaptor('x.h5ad', adata=_adata())
    col = _run(a)
    assert col['kind'] == 'gsea_batch' and col['key'] == 'gsea_grp_batch'
    assert col['groups'] == ['a', 'b', 'c'] and col['skipped'] == {'d': 'fewer than 2 cells in group'}
    assert col['members'] == {'a': 'gsea_grp_a_vs_rest', 'b': 'gsea_grp_b_vs_rest', 'c': 'gsea_grp_c_vs_rest'}
    store = a.adata.uns['xcell_enrichment']
    assert set(col['members'].values()) <= set(store) and 'gsea_grp_batch' in store
    member = json.loads(store['gsea_grp_a_vs_rest'])
    assert member['kind'] == 'gsea' and {r['name']: r for r in member['results']}['COLLAGEN']['nes'] > 0
    assert a._action_history[-1]['action'] == 'enrichment_gsea_batch'
    assert a._action_history[-1]['result']['key'] == 'gsea_grp_batch'
    summaries = {s['key']: s for s in a.get_enrichment_results()}
    assert summaries['gsea_grp_batch']['n_groups'] == 3 and summaries['gsea_grp_a_vs_rest']['n_groups'] is None
    full = a.get_enrichment_result('gsea_grp_batch')
    assert set(full['member_results']) == {'a', 'b', 'c'} and full['member_results']['b']['kind'] == 'gsea'


def test_gsea_batch_two_groups_are_mirrors_and_groups_subset():
    a = DataAdaptor('x.h5ad', adata=_adata((30, 30, 30, 5)))
    col = _run(a, groups=['a', 'b'])
    assert col['groups'] == ['a', 'b'] and col['skipped'] == {}
    ra = {r['name']: r for r in a.get_enrichment_result(col['members']['a'])['results']}
    rb = {r['name']: r for r in a.get_enrichment_result(col['members']['b'])['results']}
    assert ra['COLLAGEN']['nes'] > 0 and rb['COLLAGEN']['nes'] < 0
    col2 = _run(a, groups=['a', 'b'], reference='b')
    assert col2['members'] == {'a': 'gsea_grp_a_vs_b'} and col2['groups'] == ['a']


def test_gsea_batch_validation():
    a = DataAdaptor('x.h5ad', adata=_adata((2, 1, 1, 1)))
    with pytest.raises(ValueError, match='every group'):
        a.prepare_gsea_batch('grp', libraries=LIB, min_set_size=2, groups=['b', 'c'])
    with pytest.raises(ValueError, match='nope'):
        a.prepare_gsea_batch('nope', libraries=LIB, min_set_size=2)
    with pytest.raises(ValueError, match='Unknown group'):
        a.prepare_gsea_batch('grp', libraries=LIB, min_set_size=2, groups=['zzz'])
