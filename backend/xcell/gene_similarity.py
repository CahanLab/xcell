"""Gene–gene similarity from expression, annotation and STRING; embedding and modules.

Three channels, each a genes × genes matrix in [0, 1] with a unit diagonal:

* **expression** — a correlation across cells (:func:`gene_coexpression.corr_matrix`,
  bicor by default) mapped to ``(r + 1) / 2``, so anti-correlated genes sit
  far apart rather than at zero;
* **annotation** — cosine similarity of gene × term membership vectors over
  whatever cached libraries the caller chose (a gene is a row, every set it
  belongs to is a column). A gene no library mentions is similar to nothing
  but itself, and the caller is told how many such genes there are;
* **STRING** — the combined score of each edge STRING returns for the list,
  zero where it has no edge.

:func:`combine` takes a weighted mean over the channels that are present;
:func:`modules` runs Leiden on the kNN graph of the combined similarity;
:func:`embed` lays the genes out in 2-D with UMAP on the precomputed distance
``1 - S`` (classical MDS as the fallback and for tiny sets); and
:func:`leaf_order` gives the heatmap its dendrogram order.

Pure: arrays and dicts in, dicts out; no AnnData, no network.
"""
from __future__ import annotations

import random
import warnings
from typing import Any

import numpy as np


def _norm(name: Any) -> str:
    return str(name).strip().upper()


def expression_similarity(X_genes, metric: str = 'bicor') -> np.ndarray:
    """``(corr + 1) / 2`` over a (n_genes, n_cells) matrix."""
    from xcell import gene_coexpression as gc  # noqa: PLC0415

    C = gc.corr_matrix(np.asarray(X_genes, dtype=float), metric=metric)
    S = (C + 1.0) / 2.0
    np.clip(S, 0.0, 1.0, out=S)
    np.fill_diagonal(S, 1.0)
    return S


def annotation_similarity(genes: list[str], memberships: dict[str, list[str]]) -> tuple[np.ndarray, list[int]]:
    """Cosine similarity of membership vectors; also each gene's term count.

    Names match case-insensitively, because a human-symbol library annotates
    ``COL1A1`` and the dataset spells it ``Col1a1``.
    """
    g = len(genes)
    idx: dict[str, int] = {}
    for i, name in enumerate(genes):
        idx.setdefault(_norm(name), i)
    columns: list[list[int]] = []
    for members in memberships.values():
        rows = sorted({idx[_norm(m)] for m in members if _norm(m) in idx})
        if rows:
            columns.append(rows)
    M = np.zeros((g, max(1, len(columns))), dtype=float)
    for j, rows in enumerate(columns):
        M[rows, j] = 1.0
    n_terms = M.sum(axis=1)
    norms = np.sqrt(n_terms)
    norms[norms == 0] = 1.0
    Mn = M / norms[:, None]
    S = Mn @ Mn.T
    np.clip(S, 0.0, 1.0, out=S)
    np.fill_diagonal(S, 1.0)
    return S, [int(x) for x in n_terms]


def string_similarity(genes: list[str], edges: list[dict[str, Any]]) -> tuple[np.ndarray, int]:
    """Edge scores placed symmetrically; unit diagonal; returns the matrix and the edge count used."""
    g = len(genes)
    idx: dict[str, int] = {}
    for i, name in enumerate(genes):
        idx.setdefault(_norm(name), i)
    S = np.eye(g, dtype=float)
    used: set[tuple[int, int]] = set()
    for e in edges:
        if not isinstance(e, dict):
            continue
        i = idx.get(_norm(e.get('a', '')))
        j = idx.get(_norm(e.get('b', '')))
        if i is None or j is None or i == j:
            continue
        s = min(1.0, max(0.0, float(e.get('score') or 0.0)))
        if s > S[i, j]:
            S[i, j] = S[j, i] = s
        used.add((min(i, j), max(i, j)))
    return S, len(used)


def combine(channels: dict[str, tuple[np.ndarray | None, float]]) -> tuple[np.ndarray, dict[str, float]]:
    """Weighted mean over the channels that are present with a positive weight."""
    present = {name: (S, float(w)) for name, (S, w) in channels.items() if S is not None and float(w) > 0.0}
    if not present:
        raise ValueError('No similarity channel is available — enable expression, annotation or STRING')
    total = sum(w for _, w in present.values())
    S = None
    for mat, w in present.values():
        term = np.asarray(mat, dtype=float) * (w / total)
        S = term if S is None else S + term
    assert S is not None
    np.clip(S, 0.0, 1.0, out=S)
    np.fill_diagonal(S, 1.0)
    return S, {name: w / total for name, (_, w) in present.items()}


def _knn_edges(S: np.ndarray, n_neighbors: int) -> tuple[list[tuple[int, int]], list[float]]:
    g = S.shape[0]
    k = max(1, min(int(n_neighbors), g - 1))
    weights: dict[tuple[int, int], float] = {}
    for i in range(g):
        taken = 0
        for j in np.argsort(-S[i]):
            j = int(j)
            if j == i:
                continue
            w = float(S[i, j])
            if w <= 0.0:
                break
            key = (min(i, j), max(i, j))
            weights[key] = max(weights.get(key, 0.0), w)
            taken += 1
            if taken >= k:
                break
    edges = list(weights)
    return edges, [weights[e] for e in edges]


def modules(S: np.ndarray, n_neighbors: int = 15, resolution: float = 1.0, seed: int = 0) -> list[int]:
    """Leiden modules on the kNN graph of ``S``, relabelled so module 0 is the largest."""
    import igraph as ig  # noqa: PLC0415

    g = S.shape[0]
    edges, weights = _knn_edges(S, n_neighbors)
    graph = ig.Graph(n=g, edges=edges)
    graph.es['weight'] = weights
    ig.set_random_number_generator(random.Random(int(seed)))
    kwargs: dict[str, Any] = {'objective_function': 'modularity', 'weights': 'weight', 'n_iterations': -1}
    try:
        part = graph.community_leiden(resolution=float(resolution), **kwargs)
    except TypeError:  # older python-igraph spells it resolution_parameter
        part = graph.community_leiden(resolution_parameter=float(resolution), **kwargs)
    raw = list(part.membership)
    counts: dict[int, int] = {}
    for m in raw:
        counts[m] = counts.get(m, 0) + 1
    relabel = {old: new for new, old in enumerate(sorted(counts, key=lambda m: (-counts[m], m)))}
    return [int(relabel[m]) for m in raw]


def embed(S: np.ndarray, method: str = 'umap', n_neighbors: int = 15, seed: int = 0) -> np.ndarray:
    """2-D layout from the precomputed distance ``1 - S``."""
    g = S.shape[0]
    D = 1.0 - np.asarray(S, dtype=float)
    D = 0.5 * (D + D.T)
    np.clip(D, 0.0, None, out=D)
    np.fill_diagonal(D, 0.0)
    method = str(method or 'umap').lower()
    if method not in ('umap', 'mds'):
        raise ValueError(f"Unknown embedding '{method}'; expected 'umap' or 'mds'")
    if method == 'umap' and g >= 10:
        try:
            import umap  # noqa: PLC0415
        except ImportError:
            method = 'mds'
        else:
            k = max(2, min(int(n_neighbors), g - 1))
            with warnings.catch_warnings():
                warnings.simplefilter('ignore')
                reducer = umap.UMAP(n_components=2, metric='precomputed', n_neighbors=k,
                                    min_dist=0.3, random_state=int(seed))
                coords = reducer.fit_transform(D)
            return np.asarray(coords, dtype=np.float32)
    from sklearn.manifold import MDS  # noqa: PLC0415

    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        coords = MDS(n_components=2, dissimilarity='precomputed', random_state=int(seed),
                     normalized_stress='auto').fit_transform(D)
    return np.asarray(coords, dtype=np.float32)


def leaf_order(S: np.ndarray) -> list[int]:
    """Average-linkage leaf order over ``1 - S`` for the similarity heatmap."""
    from scipy.cluster.hierarchy import leaves_list, linkage  # noqa: PLC0415
    from scipy.spatial.distance import squareform  # noqa: PLC0415

    g = S.shape[0]
    if g < 3:
        return list(range(g))
    D = 1.0 - np.asarray(S, dtype=float)
    D = 0.5 * (D + D.T)
    np.clip(D, 0.0, None, out=D)
    np.fill_diagonal(D, 0.0)
    Z = linkage(squareform(D, checks=False), method='average')
    return [int(i) for i in leaves_list(Z)]


def module_order(S: np.ndarray, labels: list[int]) -> list[int]:
    """Heatmap order: modules as contiguous blocks (largest first), dendrogram order within each.

    A plain dendrogram over all genes interleaves modules, so the module
    boxes on the heatmap fragment into slivers; ordering within modules keeps
    every box one block while still putting similar genes side by side.
    """
    order: list[int] = []
    for m in sorted(set(labels), key=lambda m: (-labels.count(m), m)):
        idx = [i for i, lab in enumerate(labels) if lab == m]
        sub = S[np.ix_(idx, idx)]
        order.extend(idx[j] for j in leaf_order(sub))
    return order


def build_gene_map(
    genes: list[str],
    *,
    channels: dict[str, tuple[np.ndarray | None, float]],
    n_neighbors: int = 15,
    resolution: float = 1.0,
    embedding: str = 'umap',
    seed: int = 0,
) -> dict[str, Any]:
    """Combine channels, cluster, embed and order. Arrays stay arrays; the rest is JSON-safe."""
    S, used = combine(channels)
    labels = modules(S, n_neighbors=n_neighbors, resolution=resolution, seed=seed)
    coords = embed(S, method=embedding, n_neighbors=n_neighbors, seed=seed)
    order = module_order(S, labels)
    n_modules = (max(labels) + 1) if labels else 0
    sizes = [0] * n_modules
    for m in labels:
        sizes[m] += 1
    return {
        'genes': list(genes),
        'similarity': S.astype(np.float32),
        'coords': coords,
        'modules': labels,
        'order': order,
        'n_modules': n_modules,
        'module_sizes': sizes,
        'channel_weights': used,
        'embedding': embedding,
        'n_neighbors': int(n_neighbors),
        'resolution': float(resolution),
    }
