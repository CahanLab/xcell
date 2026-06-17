"""cluster_gene_set should optionally honor the active .var gene mask."""

import numpy as np
import anndata
from scipy.sparse import csr_matrix

from xcell.adaptor import DataAdaptor


def _adata():
    rng = np.random.default_rng(0)
    n_cells, n_genes = 60, 10
    X = csr_matrix(rng.integers(0, 8, size=(n_cells, n_genes)).astype(np.float32))
    ad = anndata.AnnData(X=X)
    ad.var_names = [f"g{i}" for i in range(n_genes)]
    return ad


GENES = [f"g{i}" for i in range(10)]


def test_use_gene_mask_excludes_masked_genes():
    a = DataAdaptor("x.h5ad", adata=_adata())
    # Active mask: only g0..g4 visible.
    a._visible_gene_mask = np.array([True] * 5 + [False] * 5)
    clusters = a.cluster_gene_set(GENES, method="kmeans", k=2, use_gene_mask=True)
    flat = {g for c in clusters for g in c}
    assert flat <= {"g0", "g1", "g2", "g3", "g4"}
    assert flat  # non-empty


def test_without_mask_all_genes_used():
    a = DataAdaptor("x.h5ad", adata=_adata())
    a._visible_gene_mask = np.array([True] * 5 + [False] * 5)
    clusters = a.cluster_gene_set(GENES, method="kmeans", k=2, use_gene_mask=False)
    flat = {g for c in clusters for g in c}
    assert flat == set(GENES)


def test_use_gene_mask_noop_when_no_mask_active():
    a = DataAdaptor("x.h5ad", adata=_adata())
    # No active mask (_visible_gene_mask is None) -> use_gene_mask is a no-op.
    clusters = a.cluster_gene_set(GENES, method="kmeans", k=2, use_gene_mask=True)
    flat = {g for c in clusters for g in c}
    assert flat == set(GENES)
