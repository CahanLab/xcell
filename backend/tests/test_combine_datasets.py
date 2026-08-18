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

from xcell.adaptor import combine_h5ads


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
    out = combine_h5ads(
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
    out = combine_h5ads(
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
    out = combine_h5ads(
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
    out = combine_h5ads(
        [_write(tmp_path, 'a.h5ad', a), _write(tmp_path, 'b.h5ad', b)],
        ['d1', 'd2'],
    )
    assert list(out.var_names) == [f'g{i}' for i in range(10, 20)]


def test_a_mixed_set_concatenates_rather_than_pretending_geometry(tmp_path):
    spatial = _base(10, 1)
    spatial.obsm['X_spatial'] = np.random.default_rng(1).uniform(0, 50, (10, 2))
    plain = _base(10, 2)
    out = combine_h5ads(
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
        combine_h5ads(
            [_write(tmp_path, 'a.h5ad', a), _write(tmp_path, 'b.h5ad', b)],
            ['x', 'y'],
        )


def test_fewer_than_two_files_raises(tmp_path):
    a = _base(5, 1)
    with pytest.raises(ValueError, match='At least 2'):
        combine_h5ads([_write(tmp_path, 'a.h5ad', a)], ['x'])
