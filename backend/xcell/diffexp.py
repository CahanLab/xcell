"""Differential expression analysis using scanpy.

This module implements differential expression analysis between two cell groups
using scanpy's rank_genes_groups function.
"""

import numpy as np
import pandas as pd
import scanpy as sc
from anndata import AnnData
from typing import Any


def compute_diffexp(
    adata: AnnData,
    group1_indices: list[int],
    group2_indices: list[int],
    top_n: int = 10,
    method: str = "wilcoxon",
    corr_method: str = "benjamini-hochberg",
    min_fold_change: float | None = None,
    min_in_group_fraction: float | None = None,
    max_out_group_fraction: float | None = None,
    max_pval_adj: float | None = None,
) -> dict[str, Any]:
    """Compute differential expression between two cell groups using scanpy.

    Args:
        adata: AnnData object containing expression data
        group1_indices: Cell indices for group 1
        group2_indices: Cell indices for group 2
        top_n: Number of top genes to return for each direction
        method: Statistical method ('wilcoxon', 't-test', 't-test_overestim_var', 'logreg')
        corr_method: P-value correction method ('benjamini-hochberg', 'bonferroni')
        min_fold_change: Minimum fold change for filtering (sc.tl.filter_rank_genes_groups)
        min_in_group_fraction: Minimum fraction of cells in group expressing gene
        max_out_group_fraction: Maximum fraction of cells outside group expressing gene
        max_pval_adj: Maximum adjusted p-value for filtering

    Returns:
        Dictionary containing:
        - positive: Top N genes upregulated in group1 (list of dicts with gene, log2fc, pval, pval_adj)
        - negative: Top N genes upregulated in group2 (list of dicts)
        - group1_count: Number of cells in group 1
        - group2_count: Number of cells in group 2
    """
    n1 = len(group1_indices)
    n2 = len(group2_indices)

    if n1 < 2 or n2 < 2:
        raise ValueError("Each group must have at least 2 cells for statistical testing")

    # Create a temporary grouping column
    temp_col = "_diffexp_group_"
    group_labels = np.array(["other"] * adata.n_obs, dtype=object)
    group_labels[group1_indices] = "group1"
    group_labels[group2_indices] = "group2"
    adata.obs[temp_col] = pd.Categorical(group_labels, categories=["group1", "group2", "other"])

    try:
        # Run differential expression: group1 vs group2
        sc.tl.rank_genes_groups(
            adata,
            groupby=temp_col,
            groups=["group1"],
            reference="group2",
            method=method,
            corr_method=corr_method,
            use_raw=False,
            key_added="_diffexp_result_",
        )

        # Apply filter_rank_genes_groups if any filter params are set
        use_filter = any(v is not None for v in [min_fold_change, min_in_group_fraction, max_out_group_fraction, max_pval_adj])
        if use_filter:
            filter_kwargs: dict[str, Any] = {
                "key": "_diffexp_result_",
                "key_added": "_diffexp_filtered_",
            }
            if min_fold_change is not None:
                filter_kwargs["min_fold_change"] = min_fold_change
            if min_in_group_fraction is not None:
                filter_kwargs["min_in_group_fraction"] = min_in_group_fraction
            if max_out_group_fraction is not None:
                filter_kwargs["max_out_group_fraction"] = max_out_group_fraction
            # Note: scanpy filter_rank_genes_groups doesn't have a max_pval_adj param directly,
            # so we apply it ourselves after extraction
            sc.tl.filter_rank_genes_groups(adata, **filter_kwargs)
            result_key = "_diffexp_filtered_"
        else:
            result_key = "_diffexp_result_"

        # Extract results for group1 (upregulated in group1)
        result = adata.uns[result_key]

        # Get gene names, scores, pvals, logfoldchanges
        genes = result["names"]["group1"]
        pvals = result["pvals"]["group1"]
        pvals_adj = result["pvals_adj"]["group1"]
        logfoldchanges = result["logfoldchanges"]["group1"]

        # Build positive list (top N upregulated in group1, positive logfc)
        positive = []
        for i in range(len(genes)):
            if len(positive) >= top_n:
                break
            gene_name = str(genes[i]) if not isinstance(genes[i], float) else ""
            if use_filter and (not gene_name or gene_name == "nan"):
                continue
            if logfoldchanges[i] > 0:
                adj_p = float(pvals_adj[i])
                if max_pval_adj is not None and adj_p > max_pval_adj:
                    continue
                positive.append({
                    "gene": gene_name,
                    "log2fc": float(logfoldchanges[i]),
                    "pval": float(pvals[i]),
                    "pval_adj": adj_p,
                })

        # Build negative list (top N upregulated in group2, negative logfc)
        # These are at the end of the sorted list (most negative logfc)
        negative = []
        for i in range(len(genes) - 1, -1, -1):
            if len(negative) >= top_n:
                break
            gene_name = str(genes[i]) if not isinstance(genes[i], float) else ""
            if use_filter and (not gene_name or gene_name == "nan"):
                continue
            if logfoldchanges[i] < 0:
                adj_p = float(pvals_adj[i])
                if max_pval_adj is not None and adj_p > max_pval_adj:
                    continue
                negative.append({
                    "gene": gene_name,
                    "log2fc": float(logfoldchanges[i]),
                    "pval": float(pvals[i]),
                    "pval_adj": adj_p,
                })

    finally:
        # Clean up temporary column and results
        if temp_col in adata.obs.columns:
            del adata.obs[temp_col]
        if "_diffexp_result_" in adata.uns:
            del adata.uns["_diffexp_result_"]
        if "_diffexp_filtered_" in adata.uns:
            del adata.uns["_diffexp_filtered_"]

    return {
        "positive": positive,
        "negative": negative,
        "group1_count": n1,
        "group2_count": n2,
    }
