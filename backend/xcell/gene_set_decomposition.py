"""Decompose one gene set into expression programs — pure numerics.

A gene set whose members do not co-vary (the collagens: fibrillar, basement
membrane, cartilage) is several expression configurations under one name. A
single score (mean, UCell) flattens them into one number; the informative
object is the low-rank structure of the cells × set-genes submatrix.

Two methods, one output shape:

* ``pca_programs`` — PCA on per-gene z-scores. Each component yields a signed
  pair of gene lists (loadings above / below a fraction of the component's
  largest |loading|), which is exactly the directional format UCell scores.
  Signs are fixed so the leading gene of every component loads positively.
* ``nmf_programs`` — :mod:`xcell.gene_nmf` on the (non-negative) submatrix,
  with ``max_genes`` capped at the set size. Programs are non-negative, so
  the down list is always empty.

Takes arrays, returns dicts; never touches AnnData.
"""
from __future__ import annotations

from typing import Any, Callable

import numpy as np


def _programs_shape(programs: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Every program carries the same keys, so the UI never branches on method."""
    for p in programs:
        p.setdefault('genes_down', [])
        p.setdefault('weights_down', [])
    return programs


def pca_programs(
    X,
    gene_names: list[str],
    *,
    k: int = 3,
    loading_threshold: float = 0.2,
    seed: int = 0,
) -> dict[str, Any]:
    """PCA of a cells × genes matrix into ``k`` signed programs.

    ``loading_threshold`` is relative: a gene joins a component's up (down)
    list when its loading is at least that fraction of the component's
    largest |loading|, so the cut adapts to how concentrated each component
    is instead of using one absolute number for 6-gene and 300-gene sets.
    """
    from sklearn.decomposition import PCA  # noqa: PLC0415

    X = np.asarray(X, dtype=np.float64)
    if X.ndim != 2:
        raise ValueError('X must be cells × genes')
    n, g = X.shape
    if len(gene_names) != g:
        raise ValueError(f'{len(gene_names)} gene names for {g} columns')
    k_max = min(g - 1, n - 1)
    if k < 1 or k > k_max:
        raise ValueError(f'k must be between 1 and {k_max} for {g} genes × {n} cells')
    thr_frac = float(loading_threshold)
    if not (0.0 <= thr_frac <= 1.0):
        raise ValueError('loading_threshold must be within [0, 1]')

    # z-score per gene; a zero-variance gene stays 0 and loads on nothing.
    mu = X.mean(axis=0)
    sd = X.std(axis=0)
    sd[sd == 0] = 1.0
    Z = np.clip((X - mu) / sd, -10.0, 10.0)

    pca = PCA(n_components=k, svd_solver='full', random_state=seed)
    scores = pca.fit_transform(Z)
    loadings = pca.components_.T.copy()  # genes × k
    for j in range(k):
        lead = int(np.argmax(np.abs(loadings[:, j])))
        if loadings[lead, j] < 0:
            loadings[:, j] *= -1.0
            scores[:, j] *= -1.0

    programs: list[dict[str, Any]] = []
    for j in range(k):
        col = loadings[:, j]
        top = float(np.abs(col).max())
        thr = thr_frac * top if top > 0 else 0.0
        up = [int(i) for i in np.argsort(-col) if col[i] > 0 and col[i] >= thr]
        down = [int(i) for i in np.argsort(col) if col[i] < 0 and -col[i] >= thr]
        programs.append({
            'name': f'PC{j + 1}', 'index': j,
            'genes': [gene_names[i] for i in up],
            'weights': [round(float(col[i]), 4) for i in up],
            'genes_down': [gene_names[i] for i in down],
            'weights_down': [round(float(col[i]), 4) for i in down],
            'variance_ratio': float(pca.explained_variance_ratio_[j]),
        })
    return {
        'method': 'pca', 'k': int(k),
        'scores': scores.astype(np.float32),
        'loadings': loadings.astype(np.float32),
        'variance_ratio': [float(v) for v in pca.explained_variance_ratio_],
        'programs': _programs_shape(programs),
    }


def nmf_programs(
    X,
    gene_names: list[str],
    *,
    k: int = 3,
    seed: int = 0,
    max_genes: int | None = None,
    specificity_weight: float = 1.0,
    weight_explained: float = 0.5,
    progress_callback: Callable[[float, str], None] | None = None,
) -> dict[str, Any]:
    """NMF of a non-negative cells × genes matrix into up to ``k`` programs.

    ``gene_nmf`` may drop a factor that one gene carries alone; the surviving
    programs are renumbered ``F1..`` in order and ``n_dropped`` says how many
    went.
    """
    from scipy import sparse as sp  # noqa: PLC0415

    from xcell import gene_nmf as gnmf  # noqa: PLC0415

    X = np.asarray(X, dtype=np.float32)
    if X.ndim != 2:
        raise ValueError('X must be cells × genes')
    n, g = X.shape
    if len(gene_names) != g:
        raise ValueError(f'{len(gene_names)} gene names for {g} columns')
    if k < 1 or k > min(g, n):
        raise ValueError(f'k must be between 1 and {min(g, n)} for {g} genes × {n} cells')
    if float(X.min()) < 0:
        raise ValueError('NMF needs non-negative values; pick a non-negative layer or transform')

    out = gnmf.run_gene_programs(
        sp.csr_matrix(X), list(gene_names), k=int(k), seed=int(seed),
        max_genes=int(max_genes or g), specificity_weight=float(specificity_weight),
        weight_explained=float(weight_explained), progress_callback=progress_callback,
    )
    programs = [
        {
            'name': f'F{i + 1}', 'index': i,
            'genes': list(p['genes']),
            'weights': [round(float(w), 4) for w in p['weights']],
            'genes_down': [], 'weights_down': [],
            'factor_weight': float(out['factor_weights'][i]),
        }
        for i, p in enumerate(out['programs'])
    ]
    return {
        'method': 'nmf', 'k': int(k),
        'scores': np.asarray(out['cell_scores'], dtype=np.float32),
        'loadings': np.asarray(out['loadings'], dtype=np.float32),
        'variance_ratio': None,
        'factor_weights': [float(w) for w in out['factor_weights']],
        'programs': _programs_shape(programs),
        'n_dropped': int(out['n_dropped']),
        'converged': bool(out['converged']),
        'n_iter': int(out['n_iter']),
    }
