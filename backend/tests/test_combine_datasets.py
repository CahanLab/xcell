"""Combining datasets: spatial sections side-by-side, anything else by rows.

Spatial inputs keep the existing behavior — sections offset left-to-right
with a gap, one ``X_spatial`` in the result. Non-spatial inputs (or a mixed
set) concatenate instead: intersection of genes, a ``sample`` column, and
whatever .obsm arrays every input shares are kept, because for dissociated
data those embeddings are the only geometry there is.
"""
import numpy as np
import pandas as pd
import anndata
import pytest
from scipy.sparse import csr_matrix

from xcell.adaptor import combine_datasets


GENES = [f'g{i}' for i in range(20)]


def _base(n: int, seed: int, genes=None) -> anndata.AnnData:
    rng = np.random.default_rng(seed)
    genes = genes or GENES
    return anndata.AnnData(
        X=csr_matrix(rng.poisson(1.0, size=(n, len(genes))).astype(np.float32)),
        obs=pd.DataFrame(index=[f'c{i}' for i in range(n)]),
        var=pd.DataFrame(index=list(genes)),
    )


def _write(tmp_path, name: str, a: anndata.AnnData):
    p = tmp_path / name
    a.write_h5ad(p)
    return p


def test_two_spatial_files_lay_out_side_by_side(tmp_path):
    a = _base(30, 1)
    a.obsm['X_spatial'] = np.random.default_rng(1).uniform(0, 100, (30, 2))
    b = _base(40, 2)
    b.obsm['X_spatial'] = np.random.default_rng(2).uniform(0, 100, (40, 2))
    out = combine_datasets(
        [_write(tmp_path, 'a.h5ad', a), _write(tmp_path, 'b.h5ad', b)],
        ['s1', 's2'],
    )
    assert out.uns['xcell_combine']['mode'] == 'spatial'
    assert out.n_obs == 70
    assert list(out.obs['sample'].cat.categories) == ['s1', 's2']
    sp = out.obsm['X_spatial']
    x1 = sp[np.asarray(out.obs['sample'] == 's1'), 0]
    x2 = sp[np.asarray(out.obs['sample'] == 's2'), 0]
    assert x1.max() < x2.min()  # disjoint, left-to-right


def test_the_spatial_convention_key_is_accepted_too(tmp_path):
    # squidpy/scanpy convention is obsm['spatial']; only X_spatial was
    # accepted before, which refused perfectly good Visium files.
    a = _base(10, 1)
    a.obsm['spatial'] = np.random.default_rng(1).uniform(0, 50, (10, 2))
    b = _base(12, 2)
    b.obsm['X_spatial'] = np.random.default_rng(2).uniform(0, 50, (12, 2))
    out = combine_datasets(
        [_write(tmp_path, 'a.h5ad', a), _write(tmp_path, 'b.h5ad', b)],
        ['s1', 's2'],
    )
    assert out.uns['xcell_combine']['mode'] == 'spatial'
    assert 'X_spatial' in out.obsm


def test_non_spatial_files_concatenate_and_keep_shared_embeddings(tmp_path):
    a = _base(15, 1)
    a.obsm['X_pca'] = np.random.default_rng(1).normal(size=(15, 10))
    b = _base(25, 2)
    b.obsm['X_pca'] = np.random.default_rng(2).normal(size=(25, 10))
    out = combine_datasets(
        [_write(tmp_path, 'a.h5ad', a), _write(tmp_path, 'b.h5ad', b)],
        ['d1', 'd2'],
    )
    assert out.uns['xcell_combine']['mode'] == 'concat'
    assert out.n_obs == 40
    assert out.obsm['X_pca'].shape == (40, 10)
    assert 'X_spatial' not in out.obsm
    assert list(out.obs['sample'].cat.categories) == ['d1', 'd2']
    # obs_names collision-safe
    assert out.obs_names.is_unique


def test_gene_intersection_is_taken_in_concat_mode(tmp_path):
    a = _base(10, 1, genes=[f'g{i}' for i in range(20)])
    b = _base(10, 2, genes=[f'g{i}' for i in range(10, 30)])
    out = combine_datasets(
        [_write(tmp_path, 'a.h5ad', a), _write(tmp_path, 'b.h5ad', b)],
        ['d1', 'd2'],
    )
    assert list(out.var_names) == [f'g{i}' for i in range(10, 20)]


def test_a_mixed_set_concatenates_rather_than_pretending_geometry(tmp_path):
    spatial = _base(10, 1)
    spatial.obsm['X_spatial'] = np.random.default_rng(1).uniform(0, 50, (10, 2))
    plain = _base(10, 2)
    out = combine_datasets(
        [_write(tmp_path, 'a.h5ad', spatial), _write(tmp_path, 'b.h5ad', plain)],
        ['s', 'p'],
    )
    assert out.uns['xcell_combine']['mode'] == 'concat'
    # The one-sided coordinates cannot cover the result and are dropped.
    assert 'X_spatial' not in out.obsm


def test_no_shared_genes_raises(tmp_path):
    a = _base(5, 1, genes=['a1', 'a2'])
    b = _base(5, 2, genes=['b1', 'b2'])
    with pytest.raises(ValueError, match='No shared genes'):
        combine_datasets(
            [_write(tmp_path, 'a.h5ad', a), _write(tmp_path, 'b.h5ad', b)],
            ['x', 'y'],
        )


def test_fewer_than_two_files_raises(tmp_path):
    a = _base(5, 1)
    with pytest.raises(ValueError, match='At least 2'):
        combine_datasets([_write(tmp_path, 'a.h5ad', a)], ['x'])


# ---------------------------------------------------------------------------
# non-h5ad inputs: combine goes through the same loader as File -> Load
# ---------------------------------------------------------------------------
def _write_10x_h5(tmp_path, name: str, n: int, genes=None):
    """A minimal CellRanger v3 feature-barcode .h5 (genes x cells CSC)."""
    import h5py
    genes = genes or GENES
    rng = np.random.default_rng(0)
    A = rng.poisson(1.0, size=(n, len(genes))).astype(np.int32)
    M = csr_matrix(A).T.tocsc()
    p = tmp_path / name
    with h5py.File(p, 'w') as f:
        g = f.create_group('matrix')
        g.create_dataset('data', data=M.data.astype(np.int32))
        g.create_dataset('indices', data=M.indices.astype(np.int64))
        g.create_dataset('indptr', data=M.indptr.astype(np.int64))
        g.create_dataset('shape', data=np.array([len(genes), n], dtype=np.int64))
        g.create_dataset('barcodes', data=np.array([f'BC{i}'.encode() for i in range(n)]))
        ft = g.create_group('features')
        ft.create_dataset('id', data=np.array([s.encode() for s in genes]))
        ft.create_dataset('name', data=np.array([s.encode() for s in genes]))
        ft.create_dataset('feature_type', data=np.array([b'Gene Expression'] * len(genes)))
        ft.create_dataset('genome', data=np.array([b'toy'] * len(genes)))
    return p


def _write_10x_mtx_dir(tmp_path, name: str, n: int, genes=None):
    """A legacy 10x matrix folder: matrix.mtx + genes.tsv + barcodes.tsv."""
    from scipy import io as spio
    genes = genes or GENES
    rng = np.random.default_rng(1)
    A = rng.poisson(1.0, size=(len(genes), n)).astype(np.int32)
    d = tmp_path / name
    d.mkdir()
    spio.mmwrite(str(d / 'matrix.mtx'), csr_matrix(A).tocoo())
    (d / 'genes.tsv').write_text('\n'.join(f'{g}\t{g}' for g in genes) + '\n')
    (d / 'barcodes.tsv').write_text('\n'.join(f'BC{i}' for i in range(n)) + '\n')
    return d


def test_load_dataset_file_identifies_each_format(tmp_path):
    from xcell.adaptor import load_dataset_file
    a, kind = load_dataset_file(_write(tmp_path, 'a.h5ad', _base(10, 1)))
    assert kind == 'h5ad' and a.n_obs == 10

    a, kind = load_dataset_file(_write_10x_h5(tmp_path, 'b.h5', 12))
    assert kind == '10x_h5' and a.n_obs == 12
    assert list(a.var_names) == GENES

    a, kind = load_dataset_file(_write_10x_mtx_dir(tmp_path, 'tenx', 8))
    assert kind == '10x_mtx' and a.n_obs == 8
    assert list(a.var_names) == GENES


def test_combine_h5ad_with_10x_h5(tmp_path):
    from xcell.adaptor import combine_datasets
    out = combine_datasets(
        [_write(tmp_path, 'a.h5ad', _base(10, 1)), _write_10x_h5(tmp_path, 'b.h5', 12)],
        ['plain', 'tenx'],
    )
    assert out.n_obs == 22
    assert out.uns['xcell_combine']['mode'] == 'concat'
    assert list(out.obs['sample'].cat.categories) == ['plain', 'tenx']
    assert list(out.var_names) == GENES


def test_combine_with_10x_mtx_dir(tmp_path):
    from xcell.adaptor import combine_datasets
    out = combine_datasets(
        [_write(tmp_path, 'a.h5ad', _base(10, 1)), _write_10x_mtx_dir(tmp_path, 'tenx', 8)],
        ['plain', 'folder'],
    )
    assert out.n_obs == 18


def test_combine_unloadable_file_names_the_culprit(tmp_path):
    from xcell.adaptor import combine_datasets
    junk = tmp_path / 'notes.txt'
    junk.write_text('not a dataset')
    with pytest.raises(ValueError, match='notes.txt'):
        combine_datasets(
            [_write(tmp_path, 'a.h5ad', _base(10, 1)), junk],
            ['a', 'junk'],
        )


# ---------------------------------------------------------------------------
# /api/combine route: same accepted kinds as /api/load
# ---------------------------------------------------------------------------
def _client():
    from fastapi.testclient import TestClient
    from xcell.main import app
    return TestClient(app)


def test_combine_route_accepts_h5(tmp_path):
    c = _client()
    a = str(_write(tmp_path, 'a.h5ad', _base(10, 1)))
    b = str(_write_10x_h5(tmp_path, 'b.h5', 12))
    r = c.post('/api/combine', json={
        'files': [{'file_path': a}, {'file_path': b}], 'slot': 'primary',
    })
    assert r.status_code == 200, r.text
    body = r.json()
    assert body['n_cells'] == 22
    assert body['labels'] == ['a', 'b']  # default labels drop the extension


def test_combine_route_accepts_10x_folder(tmp_path):
    c = _client()
    a = str(_write(tmp_path, 'a.h5ad', _base(10, 1)))
    d = str(_write_10x_mtx_dir(tmp_path, 'tenx', 8))
    r = c.post('/api/combine', json={
        'files': [{'file_path': a}, {'file_path': d}], 'slot': 'primary',
    })
    assert r.status_code == 200, r.text
    assert r.json()['labels'] == ['a', 'tenx']


def test_combine_route_rejects_unsupported_suffix(tmp_path):
    c = _client()
    a = str(_write(tmp_path, 'a.h5ad', _base(10, 1)))
    junk = tmp_path / 'notes.txt'
    junk.write_text('nope')
    r = c.post('/api/combine', json={
        'files': [{'file_path': a}, {'file_path': str(junk)}],
    })
    assert r.status_code == 400
    assert 'notes.txt' in r.json()['detail']


def test_combine_route_rds_without_r_is_a_400(tmp_path, monkeypatch):
    from xcell.api import routes as routes_mod
    monkeypatch.setattr(routes_mod.shutil, 'which', lambda _: None)
    c = _client()
    a = str(_write(tmp_path, 'a.h5ad', _base(10, 1)))
    rds = tmp_path / 'b.rds'
    rds.write_bytes(b'\x1f\x8b')  # existence is all that's checked before R
    r = c.post('/api/combine', json={
        'files': [{'file_path': a}, {'file_path': str(rds)}],
    })
    assert r.status_code == 400
    assert 'R is not installed' in r.json()['detail']
