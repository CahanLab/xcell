"""Loading 10x MEX (matrix.mtx + barcodes.tsv + genes/features.tsv) trios.

scanpy's ``read_10x_mtx`` infers the layout from file names and accepts
exactly two: uncompressed ``matrix.mtx`` + ``genes.tsv`` (Cell Ranger v2)
or gzipped ``matrix.mtx.gz`` + ``features.tsv.gz`` (v3+). GEO submissions
of v2 output gzip everything — ``GSM…_genes.tsv.gz`` — which fits neither,
so xcell reads the three files itself and lets the feature file's column
count say which vintage it is.
"""
import gzip

import numpy as np
import pytest
from scipy import io as spio
from scipy.sparse import csr_matrix, issparse

from xcell.adaptor import load_dataset_file


IDS = ['ENSMUSG01', 'ENSMUSG02', 'ENSMUSG03', 'ENSMUSG04', 'ENSMUSG05']
SYMBOLS = ['Xkr4', 'Gm1992', 'Rp1', 'Rp1', 'Sox17']  # duplicate symbol on purpose


def _counts(n_cells: int, n_genes: int) -> np.ndarray:
    rng = np.random.default_rng(0)
    return rng.poisson(1.0, size=(n_genes, n_cells)).astype(np.int32)


def _write_mex(directory, *, prefix='', gz=True, feature_file='genes',
               n_cells=8, gz_features=None, gz_barcodes=None,
               feature_types=None):
    """Write a MEX trio and return the matrix path.

    ``feature_file`` picks the v2 ``genes`` (2 columns) or v3 ``features``
    (3 columns) name. ``gz`` compresses every file unless ``gz_features`` /
    ``gz_barcodes`` override it for one of the companions.
    """
    directory.mkdir(exist_ok=True)
    gz_features = gz if gz_features is None else gz_features
    gz_barcodes = gz if gz_barcodes is None else gz_barcodes
    A = _counts(n_cells, len(IDS))

    def _put(name, text, compress):
        p = directory / (name + ('.gz' if compress else ''))
        if compress:
            with gzip.open(p, 'wt') as fh:
                fh.write(text)
        else:
            p.write_text(text)
        return p

    plain = directory / f'{prefix}matrix.mtx'
    spio.mmwrite(str(plain), csr_matrix(A).tocoo(), field='integer')
    matrix = plain
    if gz:
        matrix = directory / f'{prefix}matrix.mtx.gz'
        with open(plain, 'rb') as src, gzip.open(matrix, 'wb') as dst:
            dst.write(src.read())
        plain.unlink()

    if feature_file == 'genes':
        rows = [f'{i}\t{s}' for i, s in zip(IDS, SYMBOLS)]
    else:
        types = feature_types or ['Gene Expression'] * len(IDS)
        rows = [f'{i}\t{s}\t{t}' for i, s, t in zip(IDS, SYMBOLS, types)]
    _put(f'{prefix}{feature_file}.tsv', '\n'.join(rows) + '\n', gz_features)
    _put(f'{prefix}barcodes.tsv',
         '\n'.join(f'BC{i}-1' for i in range(n_cells)) + '\n', gz_barcodes)
    return matrix, A


def _check(a, A, *, ids=IDS, symbols=SYMBOLS):
    """The loaded AnnData must be cells × genes with scanpy's var layout."""
    assert a.shape == (A.shape[1], len(ids))
    assert issparse(a.X)
    np.testing.assert_array_equal(a.X.toarray(), A.T)
    # var_names are symbols made unique; the original ids sit in .var['gene_ids']
    assert list(a.var['gene_ids']) == list(ids)
    assert a.var_names.is_unique
    assert [v.split('-')[0] for v in a.var_names] == list(symbols)
    assert list(a.obs_names) == [f'BC{i}-1' for i in range(A.shape[1])]


# --- the bug: Cell Ranger v2 output gzipped by GEO --------------------------

def test_prefixed_trio_with_gzipped_genes_tsv_loads(tmp_path):
    matrix, A = _write_mex(tmp_path, prefix='GSM5059814_E11pt5whole1_rna_',
                           gz=True, feature_file='genes')
    assert matrix.name == 'GSM5059814_E11pt5whole1_rna_matrix.mtx.gz'
    a, kind = load_dataset_file(matrix)
    assert kind == '10x_mtx'
    _check(a, A)


def test_folder_with_gzipped_genes_tsv_loads(tmp_path):
    d = tmp_path / 'filtered_gene_bc_matrices'
    _, A = _write_mex(d, gz=True, feature_file='genes')
    a, kind = load_dataset_file(d)
    assert kind == '10x_mtx'
    _check(a, A)


def test_mixed_compression_trio_loads(tmp_path):
    # gzipped matrix, plain companions — GEO uploads are not always consistent
    matrix, A = _write_mex(tmp_path, prefix='GSM1_', gz=True,
                           feature_file='genes', gz_features=False,
                           gz_barcodes=False)
    a, kind = load_dataset_file(matrix)
    assert kind == '10x_mtx'
    _check(a, A)


# --- layouts scanpy already handled; must keep working -----------------------

def test_legacy_uncompressed_folder_loads(tmp_path):
    d = tmp_path / 'v2'
    _, A = _write_mex(d, gz=False, feature_file='genes')
    a, kind = load_dataset_file(d)
    assert kind == '10x_mtx'
    _check(a, A)


def test_v3_features_trio_keeps_gene_expression_rows_only(tmp_path):
    types = ['Gene Expression'] * 4 + ['Antibody Capture']
    matrix, A = _write_mex(tmp_path, prefix='GSM2_', gz=True,
                           feature_file='features', feature_types=types)
    a, kind = load_dataset_file(matrix)
    assert kind == '10x_mtx'
    _check(a, A[:4], ids=IDS[:4], symbols=SYMBOLS[:4])
    assert list(a.var['feature_types']) == ['Gene Expression'] * 4
