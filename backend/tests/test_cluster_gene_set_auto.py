"""cluster_gene_set(method='auto') co-expression module path."""
import numpy as np
import anndata
from scipy.sparse import csr_matrix

from xcell.adaptor import DataAdaptor


def _adata_with_modules(seed=0):
    rng = np.random.default_rng(seed)
    n_cells = 200
    f1 = rng.standard_normal(n_cells)
    f2 = rng.standard_normal(n_cells)
    cols, names = [], []
    for i in range(8):
        cols.append(f1 + rng.standard_normal(n_cells) * 0.15); names.append(f"A{i}")
    for i in range(8):
        cols.append(f2 + rng.standard_normal(n_cells) * 0.15); names.append(f"B{i}")
    for i in range(4):
        cols.append(rng.standard_normal(n_cells)); names.append(f"N{i}")
    X = np.vstack(cols).T            # (cells, genes)
    X = X - X.min()                  # keep non-negative, count-like
    ad = anndata.AnnData(X=csr_matrix(X.astype(np.float32)))
    ad.var_names = names
    # A 'raw' layer so purity tests read expression directly, bypassing the
    # default normalize_total+log1p path. On a 20-gene synthetic matrix
    # normalize_total injects a compositional confound that does not exist on
    # real (thousands-of-genes) data; the raw layer isolates the algorithm.
    ad.layers["raw"] = ad.X.copy()
    return ad, names


def test_auto_returns_valid_partition():
    ad, names = _adata_with_modules()
    a = DataAdaptor("x.h5ad", adata=ad)
    # default (normalized) path: partition property is normalization-agnostic.
    clusters = a.cluster_gene_set(names, method="auto", metric="pearson", min_genes=4)
    flat = [g for c in clusters for g in c]
    assert sorted(flat) == sorted(names)         # every gene exactly once
    assert len(flat) == len(set(flat))


def test_auto_groups_coexpressed_genes_together():
    ad, names = _adata_with_modules()
    a = DataAdaptor("x.h5ad", adata=ad)
    clusters = a.cluster_gene_set(
        names, method="auto", metric="pearson", min_genes=4, layer="raw"
    )
    # find the cluster containing A0; it should hold most of the A-genes and
    # exclude the B-module genes.
    a_cluster = next(c for c in clusters if "A0" in c)
    a_hits = sum(1 for g in a_cluster if g.startswith("A"))
    b_in_a = sum(1 for g in a_cluster if g.startswith("B"))
    assert a_hits >= 6                            # >=6 of 8 A-genes co-cluster
    assert b_in_a == 0                            # no B-genes leak into A


def test_auto_does_not_require_k():
    ad, names = _adata_with_modules()
    a = DataAdaptor("x.h5ad", adata=ad)
    # no k passed at all
    clusters = a.cluster_gene_set(names, method="auto", layer="raw")
    assert len(clusters) >= 1


def test_auto_report_returns_modules_unassigned_diagnostics():
    ad, names = _adata_with_modules()
    a = DataAdaptor("x.h5ad", adata=ad)
    rep = a.auto_coexpression_report(
        names, metric="pearson", min_genes=4, layer="raw", min_module_corr=0.3
    )
    assert set(rep) == {"modules", "unassigned", "diagnostics"}
    # partition: modules + unassigned == all known genes, each once
    flat = [g for m in rep["modules"] for g in m] + list(rep["unassigned"])
    assert sorted(flat) == sorted(names)
    # noise genes are not forced into a module
    assert all(not g.startswith("N") for m in rep["modules"] for g in m)
    diag = rep["diagnostics"]
    assert len(diag["module_coherence"]) == len(rep["modules"])
    assert diag["n_unassigned"] == len(rep["unassigned"])


def test_auto_min_module_corr_gates_uncorrelated_genes():
    """A high min_module_corr should route the uncorrelated noise genes
    (N0..N3) to the trailing unassigned group, not into a module."""
    ad, names = _adata_with_modules()
    a = DataAdaptor("x.h5ad", adata=ad)
    clusters = a.cluster_gene_set(
        names, method="auto", metric="pearson", min_genes=4,
        layer="raw", min_module_corr=0.3,
    )
    # the noise genes correlate with nothing -> none should sit in an A/B module
    a_cluster = next((c for c in clusters if "A0" in c), [])
    b_cluster = next((c for c in clusters if "B0" in c), [])
    assert not any(g.startswith("N") for g in a_cluster)
    assert not any(g.startswith("N") for g in b_cluster)
    # partition preserved
    flat = [g for c in clusters for g in c]
    assert sorted(flat) == sorted(names)
