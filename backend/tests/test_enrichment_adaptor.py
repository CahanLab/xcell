"""Adaptor tests for enrichment: universe, resolution, storage, GSEA rankings."""
import json

import anndata
import numpy as np
import pandas as pd
import pytest
from scipy.sparse import csr_matrix

from xcell import gene_set_sources as gss
from xcell.adaptor import DataAdaptor

GENES = ['Col1a1', 'Col1a2', 'Col3a1', 'Sox9', 'Acan', 'Actb', 'Gapdh', 'Ptprc', 'Cd3e', 'Cd19',
         'Hoxd13', 'Meis1'] + [f'G{i}' for i in range(20)]


@pytest.fixture(autouse=True)
def _cache(tmp_path, monkeypatch):
    gss.set_cache_dir(tmp_path / 'cache')
    monkeypatch.setattr(gss, 'fetch_text', lambda url, **kw: (_ for _ in ()).throw(AssertionError(url)))
    monkeypatch.setattr(gss, 'fetch_json', lambda url, **kw: (_ for _ in ()).throw(AssertionError(url)))
    gss.save_library({
        'source': 'msigdb', 'id': 'toy', 'name': 'Toy library', 'species': 'mouse', 'version': '1',
        'n_sets': 3, 'sets': [
            {'name': 'COLLAGEN', 'genes': ['COL1A1', 'COL1A2', 'COL3A1', 'G0', 'G1'], 'url': 'u'},
            {'name': 'IMMUNE', 'genes': ['Ptprc', 'Cd3e', 'Cd19', 'G2', 'G3']},
            {'name': 'TINY', 'genes': ['Sox9']},
        ]})
    yield
    gss.set_cache_dir(None)


def _adata():
    rng = np.random.default_rng(0)
    X = csr_matrix(rng.poisson(1.0, size=(30, len(GENES))).astype(np.float32))
    ad = anndata.AnnData(X=X)
    ad.var_names = GENES
    ad.var['panel'] = [g.startswith('Col') or g.startswith('G') for g in GENES]
    ad.obs['grp'] = pd.Categorical(['a'] * 15 + ['b'] * 15)
    return ad


def _adaptor():
    return DataAdaptor('x.h5ad', adata=_adata())


LIB = [{'source': 'msigdb', 'id': 'toy', 'species': 'mouse'}]


def test_overlap_run_resolves_case_and_stores_json():
    a = _adaptor()
    res = a.run_overlap_enrichment(['col1a1', 'Col1a2', 'Col3a1', 'Nope'], name='my set',
                                   libraries=LIB, min_set_size=2, min_overlap=1)
    assert res['kind'] == 'ora' and res['key'] == 'ora_my_set'
    assert res['query'] == {'name': 'my set', 'n_input': 4, 'n_in_universe': 3, 'genes_missing': ['Nope']}
    assert res['universe_size'] == len(GENES) and res['n_sets_input'] == 3 and res['n_sets_tested'] == 2
    top = res['results'][0]
    assert top['name'] == 'COLLAGEN' and top['library'] == 'Toy library'
    assert top['genes'] == ['Col1a1', 'Col1a2', 'Col3a1'] and top['n_overlap'] == 3
    stored = json.loads(a.adata.uns['xcell_enrichment']['ora_my_set'])
    assert stored['results'][0]['name'] == 'COLLAGEN'
    assert a.get_enrichment_result('ora_my_set')['key'] == 'ora_my_set'
    assert [s['key'] for s in a.get_enrichment_results()] == ['ora_my_set']
    last = a._action_history[-1]
    assert last['action'] == 'enrichment_ora' and last['result']['key'] == 'ora_my_set'


def test_overlap_universe_honours_gene_mask_and_subset_column():
    a = _adaptor()
    a.set_gene_mask(keep_columns=['panel'], hide_columns=[])
    res = a.run_overlap_enrichment(['Col1a1', 'Col1a2', 'Ptprc'], libraries=LIB, min_set_size=2, min_overlap=1)
    assert res['universe_size'] == 24            # 3 Col + Gapdh + 20 G
    assert res['query']['n_in_universe'] == 2 and res['query']['genes_missing'] == ['Ptprc']
    assert [r['name'] for r in res['results']] == ['COLLAGEN', 'IMMUNE']   # IMMUNE keeps G2,G3
    b = _adaptor()
    res2 = b.run_overlap_enrichment(['Col1a1', 'Col1a2'], gene_subset='panel', libraries=LIB,
                                    min_set_size=2, min_overlap=1)
    assert res2['universe_size'] == 24 and res2['gene_subset_type'] == 'column:panel'


def test_overlap_keys_never_overwrite_and_delete():
    a = _adaptor()
    k1 = a.run_overlap_enrichment(['Col1a1', 'Col1a2'], name='x', libraries=LIB, min_set_size=2, min_overlap=1)['key']
    k2 = a.run_overlap_enrichment(['Col1a1', 'Col1a2'], name='x', libraries=LIB, min_set_size=2, min_overlap=1)['key']
    assert (k1, k2) == ('ora_x', 'ora_x_2')
    assert a.delete_enrichment_result('ora_x') == {'deleted': 'ora_x'}
    assert [s['key'] for s in a.get_enrichment_results()] == ['ora_x_2']
    with pytest.raises(KeyError):
        a.get_enrichment_result('ora_x')


def test_overlap_inline_sets_and_errors():
    a = _adaptor()
    res = a.run_overlap_enrichment(['Col1a1', 'Col1a2'],
                                   sets=[{'name': 'mine', 'genes': ['Col1a1', 'Col1a2', 'Sox9']}],
                                   min_set_size=2, min_overlap=1)
    assert res['results'][0]['library'] == 'My gene sets'
    with pytest.raises(ValueError, match='not cached'):
        a.run_overlap_enrichment(['Col1a1', 'Col1a2'], libraries=[{'source': 'msigdb', 'id': 'missing'}])
    with pytest.raises(ValueError, match='at least 2'):
        a.run_overlap_enrichment(['Col1a1', 'Nope'], sets=[{'name': 'mine', 'genes': ['Col1a1', 'Col1a2']}],
                                 min_set_size=1)
    with pytest.raises(ValueError, match='No gene sets'):
        a.run_overlap_enrichment(['Col1a1', 'Col1a2'])


# --- GSEA rankings -------------------------------------------------------------

def _adata_de():
    """Group a over-expresses Col*, group b over-expresses Ptprc/Cd3e/Cd19."""
    rng = np.random.default_rng(0)
    n = 60
    lam = np.full((n, len(GENES)), 1.0)
    grp = np.array(['a'] * 30 + ['b'] * 30)
    for g in ('Col1a1', 'Col1a2', 'Col3a1'):
        lam[grp == 'a', GENES.index(g)] = 8.0
    for g in ('Ptprc', 'Cd3e', 'Cd19'):
        lam[grp == 'b', GENES.index(g)] = 8.0
    X = csr_matrix(rng.poisson(lam).astype(np.float32))
    ad = anndata.AnnData(X=X)
    ad.var_names = GENES
    ad.obs['grp'] = pd.Categorical(grp)
    ad.obs['subset_half'] = [True] * 40 + [False] * 20
    ad.uns['xcell_cell_subsets'] = {'half': {'n_cells': 40, 'created_at': 't', 'origin': 'test'}}
    pcs = rng.uniform(-0.1, 0.1, size=(len(GENES), 2))
    pcs[GENES.index('Hoxd13'), 0] = 0.9
    pcs[GENES.index('Meis1'), 0] = -0.9
    pcs[GENES.index('Sox9'), 0] = 0.0          # outside the PCA's gene mask -> unranked
    ad.varm['PCs'] = pcs
    ad.obsm['X_pca'] = np.zeros((n, 2))
    return ad


def _run(a, ranking, **kw):
    compute_fn, apply_fn = a.prepare_gsea(ranking, libraries=LIB, min_set_size=2, n_perm=100, **kw)
    return apply_fn(compute_fn(lambda f, m: None))


def test_gsea_diffexp_ranking_vs_rest_and_vs_group():
    a = DataAdaptor('x.h5ad', adata=_adata_de())
    res = _run(a, {'kind': 'diffexp', 'obs_column': 'grp', 'group': 'a', 'reference': 'rest'})
    assert res['kind'] == 'gsea' and res['key'] == 'gsea_grp_a_vs_rest'
    assert set(res['ranking']['genes'][:3]) == {'Col1a1', 'Col1a2', 'Col3a1'}
    assert len(res['ranking']['scores']) == len(GENES)
    by = {r['name']: r for r in res['results']}
    assert by['COLLAGEN']['nes'] > 0 and by['IMMUNE']['nes'] < 0
    assert set(by['COLLAGEN']['leading_edge']) >= {'Col1a1', 'Col1a2', 'Col3a1'}
    assert json.loads(a.adata.uns['xcell_enrichment']['gsea_grp_a_vs_rest'])['kind'] == 'gsea'
    assert a._action_history[-1]['action'] == 'enrichment_gsea'
    res2 = _run(a, {'kind': 'diffexp', 'obs_column': 'grp', 'group': 'b', 'reference': 'a', 'metric': 'log2fc'})
    assert res2['key'] == 'gsea_grp_b_vs_a'
    assert {r['name']: r for r in res2['results']}['IMMUNE']['nes'] > 0


def test_gsea_diffexp_within_named_subset_and_validation():
    a = DataAdaptor('x.h5ad', adata=_adata_de())
    res = _run(a, {'kind': 'diffexp', 'obs_column': 'grp', 'group': 'a', 'reference': 'rest', 'cell_subset': 'half'})
    assert res['params']['ranking']['cell_subset'] == 'half' and res['ranking']['n_ranked'] == len(GENES)
    assert res['key'] == 'gsea_grp_a_vs_rest' and 'half' in res['label']
    for bad, msg in [
        ({'kind': 'diffexp', 'obs_column': 'nope', 'group': 'a'}, 'nope'),
        ({'kind': 'diffexp', 'obs_column': 'grp', 'group': 'zzz'}, 'zzz'),
        ({'kind': 'diffexp', 'obs_column': 'grp', 'group': 'a', 'reference': 'a'}, 'reference'),
        ({'kind': 'diffexp', 'obs_column': 'grp', 'group': 'a', 'method': 'logreg'}, 'method'),
        ({'kind': 'pca', 'component': 5}, 'component'),
        ({'kind': 'scores', 'genes': ['Col1a1'], 'scores': [1.0, 2.0]}, 'same length'),
        ({'kind': 'nope'}, 'kind'),
    ]:
        with pytest.raises(ValueError, match=msg):
            a.prepare_gsea(bad, libraries=LIB, min_set_size=2)
    with pytest.raises(KeyError):
        a.prepare_gsea({'kind': 'diffexp', 'obs_column': 'grp', 'group': 'a', 'cell_subset': 'ghost'},
                       libraries=LIB, min_set_size=2)


def test_gsea_pca_and_scores_rankings():
    a = DataAdaptor('x.h5ad', adata=_adata_de())
    res = _run(a, {'kind': 'pca', 'component': 0})
    assert res['key'] == 'gsea_pca1'
    assert res['ranking']['genes'][0] == 'Hoxd13' and res['ranking']['genes'][-1] == 'Meis1'
    assert res['ranking']['n_ranked'] == len(GENES) - 1         # zero loadings are unranked
    res2 = _run(a, {'kind': 'scores', 'genes': ['COL1A1', 'col1a2', 'Col3a1', 'Ptprc'], 'scores': [3, 2, 1, -1]},
                key='custom')
    assert res2['key'] == 'custom' and res2['ranking']['n_ranked'] == 4 and res2['ranking']['genes'][0] == 'Col1a1'
    assert res2['ranking']['label'] == 'custom scores'


def test_gsea_size_filter_uses_ranked_members_not_universe_size():
    """A set larger than max_set_size in the universe but within it after
    unranked (zero-loading) genes drop out must still be tested."""
    ad = _adata_de()
    pcs = np.zeros((len(GENES), 2))
    for g in ('Col1a1', 'Col1a2', 'Col3a1'):
        pcs[GENES.index(g), 0] = 0.9
    pcs[GENES.index('Ptprc'), 0] = -0.5
    ad.varm['PCs'] = pcs
    a = DataAdaptor('x.h5ad', adata=ad)
    compute_fn, apply_fn = a.prepare_gsea({'kind': 'pca', 'component': 0}, libraries=LIB,
                                         min_set_size=2, max_set_size=4, n_perm=50)
    res = apply_fn(compute_fn(lambda f, m: None))
    names = {r['name'] for r in res['results']}
    assert 'COLLAGEN' in names                  # 5 in universe, 3 ranked
    assert res['ranking']['n_ranked'] == 4


def test_gsea_validates_weight_seed_n_perm_and_source_and_logs_scores():
    a = DataAdaptor('x.h5ad', adata=_adata_de())
    rk = {'kind': 'diffexp', 'obs_column': 'grp', 'group': 'a'}
    with pytest.raises(ValueError, match='weight'):
        a.prepare_gsea(rk, libraries=LIB, min_set_size=2, weight=-1)
    with pytest.raises(ValueError, match='seed'):
        a.prepare_gsea(rk, libraries=LIB, min_set_size=2, seed=-3)
    with pytest.raises(ValueError, match='n_perm'):
        a.prepare_gsea(rk, libraries=LIB, min_set_size=2, n_perm=10 ** 7)
    with pytest.raises(ValueError, match='Unknown gene-set source'):
        a.run_overlap_enrichment(['Col1a1', 'Col1a2'], libraries=[{'source': '../../etc', 'id': 'toy'}])
    with pytest.raises(ValueError, match='No gene set has'):
        a.prepare_gsea(rk, libraries=LIB, min_set_size=50)
    res = _run(a, {'kind': 'scores', 'genes': ['Col1a1', 'Col1a2', 'Ptprc'], 'scores': [2, 1, -1]})
    assert res['params']['ranking']['genes'] == ['Col1a1', 'Col1a2', 'Ptprc']
    assert res['params']['ranking']['scores'] == [2.0, 1.0, -1.0]
