import json

import anndata as ad
import numpy as np
import pandas as pd
import pytest
from scipy.sparse import csr_matrix

from xcell.adaptor import DataAdaptor

MOUSE = ['ENSMUSG00000002603', 'ENSMUSG00000051951', 'ENSMUSG00000000001']


def _adata(names, n=12, seed=0):
    rng = np.random.default_rng(seed)
    X = csr_matrix(rng.poisson(1.0, size=(n, len(names))).astype(np.float32))
    obs = pd.DataFrame(index=[f'c{i}' for i in range(n)])
    var = pd.DataFrame(index=pd.Index(list(names)))
    return ad.AnnData(X=X, obs=obs, var=var)


def _adaptor(a):
    return DataAdaptor('x.h5ad', adata=a)


def test_preview_detects_mouse_and_mutates_nothing():
    a = _adata(MOUSE)
    d = _adaptor(a)
    before = list(a.var_names)
    out = d.preview_gene_symbol_mapping()
    assert out['species'] == 'mouse'
    assert out['applicable'] is True
    assert out['n_mapped'] >= 2
    assert list(a.var_names) == before
    assert 'gene_symbol' not in a.var.columns
    json.dumps(out)


def test_preview_says_it_is_not_needed_when_names_are_already_symbols():
    d = _adaptor(_adata(['Tgfb1', 'Xkr4', 'Shh']))
    out = d.preview_gene_symbol_mapping()
    assert out['applicable'] is False
    assert 'ensembl' in out['reason'].lower()


def test_preview_names_an_unsupported_species():
    d = _adaptor(_adata(['ENSDARG00000000001', 'ENSDARG00000000002']))
    out = d.preview_gene_symbol_mapping()
    assert out['applicable'] is False
    assert 'zebrafish' in out['reason']


def test_preview_points_at_an_existing_symbol_column():
    a = _adata(MOUSE)
    a.var['gene_name'] = ['Tgfb1', 'Xkr4', 'Gnai3']
    out = _adaptor(a).preview_gene_symbol_mapping()
    assert out['existing_symbol_column'] == 'gene_name'
    assert 'gene_name' in out['reason']


def test_map_writes_the_column_and_leaves_the_index_alone():
    a = _adata(MOUSE)
    d = _adaptor(a)
    out = d.map_gene_symbols()
    assert 'gene_symbol' in a.var.columns
    assert list(a.var_names) == MOUSE
    assert a.var['gene_symbol'].iloc[0] == 'Tgfb1'
    assert out['n_mapped'] >= 2
    json.dumps(out)


def test_map_with_set_as_index_promotes_symbols_and_keeps_them_unique():
    a = _adata(MOUSE)
    d = _adaptor(a)
    d.map_gene_symbols(set_as_index=True)
    assert 'Tgfb1' in list(a.var_names)
    assert len(set(a.var_names)) == len(a.var_names)


def test_the_promoted_index_is_strings_not_categorical(recwarn):
    # A categorical .var index is what AnnData warns about on assignment and is
    # a hazard on save, so the column is written as plain strings.
    a = _adata(MOUSE)
    _adaptor(a).map_gene_symbols(set_as_index=True)
    assert a.var.index.dtype == object
    assert not any('categorical' in str(w.message).lower() for w in recwarn)


def test_map_refuses_an_unsupported_species():
    d = _adaptor(_adata(['ENSDARG00000000001', 'ENSDARG00000000002']))
    with pytest.raises(ValueError, match='zebrafish'):
        d.map_gene_symbols()


def test_unmapped_genes_keep_their_id_rather_than_going_blank():
    a = _adata(['ENSMUSG00000002603', 'ENSMUSG99999999999'])
    d = _adaptor(a)
    d.map_gene_symbols()
    vals = list(a.var['gene_symbol'])
    assert vals[1] == 'ENSMUSG99999999999'
    assert all(v and not pd.isna(v) for v in vals)


def test_the_new_column_is_offered_as_a_gene_identifier():
    # The whole design rests on the existing Gene IDs picker seeing it.
    a = _adata(MOUSE)
    d = _adaptor(a)
    d.map_gene_symbols()
    assert 'gene_symbol' in d.get_var_identifier_columns()['columns']
