"""Overlap of library gene sets with the loaded dataset, and a species guess from symbol case."""
import numpy as np
import pandas as pd
import anndata
import pytest
from scipy.sparse import csr_matrix

from xcell.adaptor import DataAdaptor
from xcell import gene_symbols as gs


def _adata():
    rng = np.random.default_rng(0)
    genes = ['Col1a1', 'Col1a2', 'Col2a1', 'Acan', 'Sox9', 'Runx2', 'mt-Nd1', 'Gm1992']
    ad = anndata.AnnData(X=csr_matrix(rng.poisson(1.0, size=(20, len(genes))).astype(np.float32)))
    ad.var_names = genes
    ad.var['highly_variable'] = [True, True, False, True, False, False, False, False]
    ad.var['spatially_variable'] = [1, 0, 1, 0, 0, 0, 0, 0]
    return ad


def _adaptor():
    return DataAdaptor('x.h5ad', adata=_adata())


def test_overlap_counts_exact_and_case_insensitive_matches_and_columns():
    a = _adaptor()
    out = a.gene_set_overlap(
        [{'name': 'collagens', 'genes': ['COL1A1', 'COL1A2', 'COL2A1', 'COL9A9']},
         {'name': 'exact', 'genes': ['Acan', 'Sox9']}],
        columns=['highly_variable', 'spatially_variable'],
    )
    s0, s1 = out['sets']
    assert s0['n_genes'] == 4 and s0['n_present'] == 3
    assert s0['n_exact'] == 0 and s0['n_case_insensitive'] == 3
    assert s0['genes_resolved'] == ['Col1a1', 'Col1a2', 'Col2a1']  # dataset spelling
    assert s0['genes_missing'] == ['COL9A9']
    assert s0['columns'] == {'highly_variable': 2, 'spatially_variable': 2}
    assert s1['n_exact'] == 2 and s1['n_case_insensitive'] == 0
    assert s1['columns'] == {'highly_variable': 1, 'spatially_variable': 0}
    assert out['n_genes_dataset'] == 8
    assert out['columns'] == ['highly_variable', 'spatially_variable']


def test_overlap_keeps_genes_down_separately():
    a = _adaptor()
    out = a.gene_set_overlap([{'name': 'dir', 'genes': ['Col1a1'], 'genesDown': ['SOX9', 'Nope']}])
    s = out['sets'][0]
    assert s['genes_resolved'] == ['Col1a1']
    assert s['genes_down_resolved'] == ['Sox9']
    assert s['genes_missing'] == ['Nope']
    assert s['n_genes'] == 3 and s['n_present'] == 2


def test_overlap_rejects_unknown_column_and_truncates_missing():
    a = _adaptor()
    with pytest.raises(ValueError, match='boolean .var column'):
        a.gene_set_overlap([{'name': 'x', 'genes': ['Col1a1']}], columns=['nope'])
    out = a.gene_set_overlap([{'name': 'big', 'genes': [f'ZZ{i}' for i in range(150)]}])
    assert out['sets'][0]['n_missing'] == 150 and len(out['sets'][0]['genes_missing']) == 100


def test_overlap_does_not_mutate_and_is_json_clean():
    a = _adaptor()
    before = list(a.adata.var_names)
    out = a.gene_set_overlap([{'name': 'e', 'genes': []}])
    assert out['sets'][0]['n_present'] == 0 and out['sets'][0]['genes_resolved'] == []
    assert list(a.adata.var_names) == before
    import json
    json.dumps(out)


# --- species guess from symbol case -----------------------------------------

def test_guess_species_prefers_ensembl_prefixes():
    assert gs.guess_species(['ENSMUSG00000051951', 'ENSMUSG00000089699', 'Xkr4'])['species'] == 'mouse'
    assert gs.guess_species(['ENSG00000141510', 'ENSG00000012048'])['species'] == 'human'


def test_guess_species_from_symbol_case():
    out = gs.guess_species(['Col1a1', 'Col1a2', 'Sox9', 'mt-Nd1', 'H2-Ab1'])
    assert out['species'] == 'mouse' and out['method'] == 'symbol_case'
    out = gs.guess_species(['COL1A1', 'COL1A2', 'SOX9', 'MT-ND1', 'HLA-A'])
    assert out['species'] == 'human' and out['method'] == 'symbol_case'


def test_guess_species_is_none_when_ambiguous():
    assert gs.guess_species(['g1', 'g2', 'g3'])['species'] is None
    assert gs.guess_species([])['species'] is None


def test_adaptor_species_guess_route_shape():
    a = _adaptor()
    out = a.guess_species()
    assert out['species'] == 'mouse' and 'method' in out and 'n_genes' in out
