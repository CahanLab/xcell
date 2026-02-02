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
) -> dict[str, Any]:
    """Compute differential expression between two cell groups using scanpy.

    Args:
        adata: AnnData object containing expression data
        group1_indices: Cell indices for group 1
        group2_indices: Cell indices for group 2
        top_n: Number of top genes to return for each direction
        method: Statistical method ('wilcoxon', 't-test', 't-test_overestim_var', 'logreg')

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
            use_raw=False,
            key_added="_diffexp_result_",
        )

        # Extract results for group1 (upregulated in group1)
        result = adata.uns["_diffexp_result_"]

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
            if logfoldchanges[i] > 0:
                positive.append({
                    "gene": str(genes[i]),
                    "log2fc": float(logfoldchanges[i]),
                    "pval": float(pvals[i]),
                    "pval_adj": float(pvals_adj[i]),
                })

        # Build negative list (top N upregulated in group2, negative logfc)
        # These are at the end of the sorted list (most negative logfc)
        negative = []
        for i in range(len(genes) - 1, -1, -1):
            if len(negative) >= top_n:
                break
            if logfoldchanges[i] < 0:
                negative.append({
                    "gene": str(genes[i]),
                    "log2fc": float(logfoldchanges[i]),
                    "pval": float(pvals[i]),
                    "pval_adj": float(pvals_adj[i]),
                })

    finally:
        # Clean up temporary column and results
        if temp_col in adata.obs.columns:
            del adata.obs[temp_col]
        if "_diffexp_result_" in adata.uns:
            del adata.uns["_diffexp_result_"]

    return {
        "positive": positive,
        "negative": negative,
        "group1_count": n1,
        "group2_count": n2,
    }
