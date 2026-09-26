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
