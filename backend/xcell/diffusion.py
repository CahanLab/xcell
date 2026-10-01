"""Diffusion maps and diffusion pseudotime on any connectivity graph.

Pure: a graph and arrays in, plain dicts out; no adaptor. The computation is
scanpy's — ``sc.tl.diffmap`` / ``sc.tl.dpt`` run on a throwaway AnnData that
holds only the graph — so on a symmetric, connected kNN the results are
scanpy's bit for bit. What this module adds is for the graphs xcell has that
are not a scanpy kNN (spatial, combined, a subset's slice):

- An isolated cell makes scanpy's density normalisation divide by zero, and
  sparse algebra then skips the empty row, which places the cell at 0 on every
  component: a position, silently. Isolated cells are left out and get NaN.
- ``eigsh`` assumes a symmetric matrix; squidpy's kNN spatial graph is not
  (k nearest is not a mutual relation), and an asymmetric matrix gives wrong
  eigenvectors without an error.
- Across disconnected components DPT returns inf, which JSON cannot carry.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import sparse

#: Eigenvalues at or above this are stationary — one per connected component.
#: scanpy's own cut in ``Neighbors._get_dpt_row``, set for float32 precision.
STATIONARY_EVAL = 0.9994

# The neighbors entry the throwaway AnnData carries for scanpy to read.
_NEIGHBORS_KEY = '_xcell_diffusion'


@dataclass(frozen=True)
class PreparedGraph:
    """A graph made fit for diffusion, and what had to be done to it."""

    matrix: sparse.csr_matrix   # kept × kept, symmetric, no self-loops
    kept: np.ndarray            # rows of the input that have an edge
    n_isolated: int
    symmetrized: bool
    component_labels: np.ndarray
    component_sizes: list[int]  # largest first

    @property
    def n_components(self) -> int:
        return len(self.component_sizes)


def prepare_graph(conn) -> PreparedGraph:
    """Symmetrise, drop self-loops and isolated rows, find the components.

    A graph that needs none of that is passed through as the very same matrix,
    so the scanpy path stays exact — even reordering the stored entries could
    change float32 sums.
    """
    from scipy.sparse.csgraph import connected_components

    if conn.ndim != 2 or conn.shape[0] != conn.shape[1]:
        raise ValueError(f"A connectivity graph must be square; got shape {tuple(conn.shape)}.")
    W = conn if sparse.isspmatrix_csr(conn) else sparse.csr_matrix(conn)
    if W.nnz and W.data.min() < 0:
        raise ValueError("The graph has negative edge weights; diffusion needs non-negative ones.")

    if W.diagonal().any():
        W = W.copy()
        W.setdiag(0)
        W.eliminate_zeros()

    diff = W - W.T
    symmetrized = False
    if diff.nnz:
        scale = float(np.abs(W.data).max()) if W.nnz else 1.0
        if float(np.abs(diff.data).max()) > 1e-6 * scale:
            # sklearn's spectral_embedding convention. Averaging, rather than
            # taking the union, keeps a weighted graph's weights meaningful.
            W = ((W + W.T) * 0.5).tocsr()
            symmetrized = True

    degree = np.asarray(W.sum(axis=1)).ravel()
    kept = np.flatnonzero(degree > 0)
    if kept.size == 0:
        raise ValueError("The graph has no edges.")
    if kept.size < W.shape[0]:
        W = W[kept][:, kept]

    _, labels = connected_components(W, directed=False)
    sizes = sorted((int(s) for s in np.bincount(labels)), reverse=True)
    return PreparedGraph(
        matrix=W, kept=kept, n_isolated=int(conn.shape[0] - kept.size),
        symmetrized=symmetrized, component_labels=labels, component_sizes=sizes,
    )


def _scanpy_frame(W: sparse.csr_matrix):
    """An AnnData carrying only the graph, the way scanpy's tools read it."""
    import anndata
    import pandas as pd

    ad = anndata.AnnData(obs=pd.DataFrame(index=[str(i) for i in range(W.shape[0])]))
    ad.obsp['connectivities'] = W
    # NeighborsView insists on a distances key; diffusion never reads the
    # matrix, so it may name nothing. No 'params': scanpy would require
    # n_neighbors in it, and estimates it from the graph when it is absent.
    ad.uns[_NEIGHBORS_KEY] = {'connectivities_key': 'connectivities',
                              'distances_key': '_xcell_no_distances'}
    return ad


def diffusion_map(conn, n_comps: int = 15, random_state: int = 0) -> dict:
    """The diffusion components of a graph (scanpy's ``X_diffmap`` layout).

    Column 0 is the stationary state, as in scanpy; with k disconnected
    components the first k columns are stationary. ``view_dims`` names the
    first informative pair.
    """
    import scanpy as sc

    n_comps = int(n_comps)
    if n_comps < 3:
        raise ValueError("A diffusion map needs at least 3 components "
                         "(the first is the stationary state).")
    n = int(conn.shape[0])
    g = prepare_graph(conn)
    m = int(g.matrix.shape[0])
    if n_comps > m - 1:
        raise ValueError(f"{n_comps} components need at least {n_comps + 1} connected cells; "
                         f"this graph has {m}.")

    ad = _scanpy_frame(g.matrix)
    sc.tl.diffmap(ad, n_comps=n_comps, neighbors_key=_NEIGHBORS_KEY, random_state=random_state)
    evals = np.asarray(ad.uns['diffmap_evals'], dtype=np.float32)
    X = np.full((n, n_comps), np.nan, dtype=np.float32)
    X[g.kept] = np.asarray(ad.obsm['X_diffmap'], dtype=np.float32)

    n_stationary = int((evals >= STATIONARY_EVAL).sum())
    first = min(max(n_stationary, 1), n_comps - 2)
    view_dims = [first, first + 1]

    warnings: list[str] = []
    if g.n_isolated:
        plural = 's' if g.n_isolated != 1 else ''
        warnings.append(f"{g.n_isolated} isolated cell{plural} (no edges in this graph) "
                        "left out; they have no coordinates.")
    if g.symmetrized:
        warnings.append("The graph is not symmetric; it was symmetrised as (W + Wᵀ)/2 first.")
    if g.n_components > 1:
        shown = ', '.join(str(s) for s in g.component_sizes[:5])
        more = ', …' if g.n_components > 5 else ''
        warnings.append(
            f"The graph has {g.n_components} disconnected components (sizes {shown}{more}); "
            f"the first {n_stationary} diffusion components only tell them apart. "
            f"View DC{view_dims[0]} × DC{view_dims[1]}, or run on a subset that is one component.")

    return {
        'X': X,
        'evals': evals,
        'kept': g.kept,
        'n_cells_used': m,
        'n_isolated': g.n_isolated,
        'symmetrized': g.symmetrized,
        'n_components': g.n_components,
        'component_sizes': g.component_sizes,
        'n_stationary': n_stationary,
        'view_dims': view_dims,
        'warnings': warnings,
    }


def dpt_pseudotime(conn, X, evals, root: int, n_dcs: int = 10) -> dict:
    """Diffusion pseudotime from ``root`` over a map :func:`diffusion_map` made.

    ``conn`` and ``X`` cover the same cells (rows); the cells with coordinates
    are the ones the map was computed on, and the graph is re-sliced to them.
    Cells the root cannot reach, and cells without coordinates, get NaN.
    """
    import scanpy as sc

    X = np.asarray(X, dtype=np.float32)
    evals = np.asarray(evals, dtype=np.float32)
    n = int(conn.shape[0])
    if X.shape[0] != n:
        raise ValueError(f"The diffusion map has {X.shape[0]} rows but the graph {n}.")
    if X.shape[1] != evals.size:
        raise ValueError(f"The diffusion map has {X.shape[1]} components but {evals.size} eigenvalues.")
    covered = ~np.isnan(X).any(axis=1)
    root = int(root)
    if not (0 <= root < n) or not covered[root]:
        raise ValueError(f"The root cell ({root}) has no diffusion coordinates; "
                         "pick a cell the diffusion map covers.")
    n_dcs = int(min(max(int(n_dcs), 2), X.shape[1]))

    cells = np.flatnonzero(covered)
    sub = conn if cells.size == n else conn.tocsr()[cells][:, cells]
    g = prepare_graph(sub)
    if g.n_isolated:
        raise ValueError("The graph has changed since this diffusion map was computed "
                         "(cells it covered have lost their edges); re-run the diffusion map.")

    ad = _scanpy_frame(g.matrix)
    ad.obsm['X_diffmap'] = X[cells]
    ad.uns['diffmap_evals'] = evals
    ad.uns['iroot'] = int(np.searchsorted(cells, root))
    sc.tl.dpt(ad, n_dcs=n_dcs, neighbors_key=_NEIGHBORS_KEY)
    local = ad.obs['dpt_pseudotime'].to_numpy(dtype=np.float64)
    local[~np.isfinite(local)] = np.nan

    pseudotime = np.full(n, np.nan, dtype=np.float64)
    pseudotime[cells] = local
    return {
        'pseudotime': pseudotime,
        'n_unreachable': int(np.isnan(local).sum()),
        'n_dcs': n_dcs,
        'root': root,
    }


def dpt_space(X, evals, n_dcs: int) -> np.ndarray:
    """Coordinates in which Euclidean distance is DPT's distance.

    Non-stationary components are scaled by λ/(1−λ); stationary ones enter
    unscaled — scanpy's ``_get_dpt_row``, written as a vector per cell.
    """
    X = np.asarray(X, dtype=np.float64)
    ev = np.asarray(evals, dtype=np.float64)[:n_dcs]
    weight = np.where(ev < STATIONARY_EVAL, ev / np.where(ev < STATIONARY_EVAL, 1 - ev, 1.0), 1.0)
    return X[:, :n_dcs] * weight


def root_from_cells(X, evals, cells, n_dcs: int) -> int:
    """One cell to root DPT at, standing for a group: the one nearest the
    group's centroid in DPT space. A tip cell would hang the whole ordering
    on one noisy outlier; a medoid is the same idea at O(g²)."""
    cells = np.asarray(list(cells), dtype=np.int64)
    if cells.size == 0:
        raise ValueError("No root cells were given.")
    if cells.min() < 0 or cells.max() >= np.asarray(X).shape[0]:
        raise ValueError("A root cell index is out of range.")
    Y = dpt_space(X, evals, n_dcs)
    ok = cells[~np.isnan(Y[cells]).any(axis=1)]
    if ok.size == 0:
        raise ValueError("None of the root cells has diffusion coordinates.")
    if ok.size == 1:
        return int(ok[0])
    centre = Y[ok].mean(axis=0)
    return int(ok[np.argmin(((Y[ok] - centre) ** 2).sum(axis=1))])
