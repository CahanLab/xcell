"""NMF gene-program discovery — a port of GeneNMF's single-sample core.

GeneNMF (Andreatta & Carmona, https://github.com/carmonalab/GeneNMF) factorizes
a single-cell expression matrix into a handful of non-negative "gene programs":
``X ≈ H · Wᵀ`` where ``W`` holds per-gene loadings and ``H`` holds per-cell
usage. Its R implementation delegates the factorization to ``RcppML::nmf`` and
then turns the loadings into interpretable gene sets. This module reimplements
both halves in NumPy/SciPy:

* :func:`nmf` — the factorization, by hierarchical alternating least squares
  (fast-HALS, Cichocki & Phan 2009). Output follows RcppML's default **L1
  diagonalization**: ``A ≈ W·diag(d)·H`` with the columns of ``W`` summing to 1
  and factors sorted by decreasing ``d``. That convention is not cosmetic —
  :func:`program_genes` compares each gene's loading *across* programs, so an
  unnormalized factor would look artificially specific.
* :func:`program_genes` — GeneNMF's ``getNMFgenes()``: specificity re-weighting
  followed by a cumulative-weight cutoff.
* :func:`run_gene_programs` — both, plus the metrics the UI reports.

Pure NumPy / SciPy: the module takes arrays and returns dicts, and never
touches AnnData.

**Performance.** The two sparse products per iteration are ~95% of the runtime,
and SciPy releases the GIL inside them, so they are split across a thread pool
over disjoint row blocks (~4x on 8 threads). Row blocking is exact, not
approximate — the same rows are reduced in the same order whatever the thread
count — so results stay bit-for-bit reproducible.
"""
from __future__ import annotations

import os
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Callable

import numpy as np
import scipy.sparse as sp

# Guards a division by a factor that L1 regularization drove to zero.
_EPS = 1e-12
_MAX_THREADS = 8


# ---------------------------------------------------------------------------
# threaded sparse products
# ---------------------------------------------------------------------------
def _default_threads() -> int:
    return max(1, min(_MAX_THREADS, os.cpu_count() or 1))


class _RowBlocks:
    """A sparse matrix pre-split into row blocks for threaded ``A @ B``.

    The blocks are built once and reused every iteration; each is a view onto
    the parent's ``data``/``indices`` (only the short ``indptr`` is copied), so
    holding them costs nothing beyond the parent matrix.
    """

    def __init__(self, A: sp.csr_matrix, n_threads: int):
        self.A = A
        self.shape = A.shape
        self.dtype = A.dtype
        n_rows = A.shape[0]
        # Below a few rows per thread the pool costs more than it saves.
        self.n_threads = n_threads if n_rows >= 8 * n_threads else 1
        if self.n_threads == 1:
            self.blocks: list[tuple[int, int, sp.csr_matrix]] = []
            return
        bounds = np.linspace(0, n_rows, self.n_threads + 1).astype(np.int64)
        indptr = A.indptr
        self.blocks = []
        for lo, hi in zip(bounds[:-1], bounds[1:]):
            start, stop = int(indptr[lo]), int(indptr[hi])
            block = sp.csr_matrix(
                (
                    A.data[start:stop],
                    A.indices[start:stop],
                    indptr[lo:hi + 1] - start,
                ),
                shape=(int(hi - lo), A.shape[1]),
                copy=False,
            )
            self.blocks.append((int(lo), int(hi), block))

    def dot(self, B: np.ndarray, pool: ThreadPoolExecutor | None) -> np.ndarray:
        if self.n_threads == 1 or pool is None:
            return self.A @ B
        out = np.empty((self.shape[0], B.shape[1]), dtype=B.dtype)

        def _work(item):
            lo, hi, block = item
            out[lo:hi] = block @ B

        list(pool.map(_work, self.blocks))
        return out


class _DenseOperand:
    """Dense stand-in for :class:`_RowBlocks` — BLAS already threads GEMM."""

    def __init__(self, A: np.ndarray):
        self.A = A
        self.shape = A.shape
        self.dtype = A.dtype

    def dot(self, B: np.ndarray, pool: ThreadPoolExecutor | None) -> np.ndarray:
        return self.A @ B


# ---------------------------------------------------------------------------
# factorization
# ---------------------------------------------------------------------------
def _validate(X, k: int, l1_w: float, l1_h: float) -> None:
    if X.ndim != 2:
        raise ValueError(f"X must be 2-D (n_cells, n_genes), got shape {X.shape}")
    n_cells, n_genes = X.shape
    if not isinstance(k, (int, np.integer)) or k < 2:
        raise ValueError(f"k must be an integer >= 2, got {k!r}")
    if k > min(n_cells, n_genes):
        raise ValueError(
            f"k={k} exceeds the matrix ({n_cells} cells x {n_genes} genes); "
            f"k must be <= {min(n_cells, n_genes)}"
        )
    for name, val in (("l1_w", l1_w), ("l1_h", l1_h)):
        if not (0.0 <= val < 1.0):
            raise ValueError(f"{name} must be in [0, 1), got {val!r}")
    data = X.data if sp.issparse(X) else X
    if data.size and float(np.min(data)) < 0.0:
        raise ValueError(
            "X must be non-negative for NMF; found a negative value. Use "
            "log-normalized counts, not scaled or centered expression."
        )


def _sweep(F: np.ndarray, G: np.ndarray, XF: np.ndarray, l1: float) -> None:
    """One HALS pass: update every column of ``G`` in place.

    ``F`` is the fixed factor, ``XF = X·F`` (or ``Xᵀ·F``) and ``G`` the factor
    being solved for. Each column is the exact non-negative least-squares
    solution given the others, which is what makes HALS converge in far fewer
    passes than multiplicative updates.
    """
    FtF = F.T @ F
    k = G.shape[1]
    for j in range(k):
        num = XF[:, j] - G @ FtF[:, j] + G[:, j] * FtF[j, j]
        if l1:
            # RcppML applies L1 after rescaling the factor, so the threshold is
            # a fraction of the column's own scale rather than a raw magnitude
            # — otherwise the same setting means different things per dataset.
            peak = float(num.max())
            if peak > 0.0:
                num = num - l1 * peak
        np.maximum(num, 0.0, out=num)
        num /= FtF[j, j] + _EPS
        G[:, j] = num


def _l1_diagonalize(W: np.ndarray, H: np.ndarray):
    """Rescale to RcppML's convention and sort factors by decreasing weight.

    Columns of ``W`` are put on the unit simplex; the scale moves into ``H`` so
    ``H @ W.T`` still reconstructs the data. ``d`` is each factor's total cell
    usage, which is what orders the programs.
    """
    col_sums = W.sum(axis=0)
    dead = col_sums <= 0.0
    safe = np.where(dead, 1.0, col_sums)
    W = W / safe
    H = H * safe
    if dead.any():
        # A factor L1 drove to zero contributes nothing; drop its usage too so
        # it sorts to the end instead of carrying a phantom weight.
        W[:, dead] = 0.0
        H[:, dead] = 0.0
    d = H.sum(axis=0)
    order = np.argsort(-d, kind="stable")
    return (
        np.ascontiguousarray(W[:, order]),
        np.ascontiguousarray(H[:, order]),
        d[order],
    )


def nmf(
    X,
    k: int,
    *,
    l1_w: float = 0.0,
    l1_h: float = 0.0,
    max_iter: int = 500,
    tol: float = 1e-4,
    seed: int = 0,
    n_threads: int | None = None,
    progress_callback: Callable[[float, str], None] | None = None,
) -> dict[str, Any]:
    """Factorize a non-negative ``(n_cells, n_genes)`` matrix.

    Args:
        X: cells x genes, non-negative. Dense or any SciPy sparse format;
            sparse input stays sparse (a 100k-cell matrix is never densified).
        k: number of programs (>= 2 — a rank-1 "program" is just the mean
            profile, and specificity weighting is undefined for it).
        l1_w/l1_h: sparsity on gene loadings / cell usage, in ``[0, 1)`` as a
            fraction of each factor's own scale. 0 disables (GeneNMF's default).
        max_iter/tol: stop when a pass improves the squared residual by less
            than ``tol`` of the data's total energy, or the budget runs out.
        seed: fixes the random initialization, so runs are reproducible.
        n_threads: row-block parallelism for the sparse products. Defaults to
            ``min(8, cpu_count)``. Ignored for dense input (BLAS threads it).

    Returns:
        ``{'W': (n_genes, k), 'H': (n_cells, k), 'd': (k,), 'n_iter',
        'converged', 'error'}`` where ``W`` columns sum to 1, ``H @ W.T``
        reconstructs ``X``, ``d`` is each factor's total usage (descending),
        and ``error`` is the Frobenius residual norm.
    """
    _validate(X, k, l1_w, l1_h)
    n_cells, n_genes = X.shape

    if sp.issparse(X):
        Xc = X.tocsr()
        dtype = Xc.dtype if Xc.dtype in (np.float32, np.float64) else np.float64
        Xc = Xc.astype(dtype, copy=False)
        x_norm_sq = float(np.dot(Xc.data.astype(np.float64), Xc.data.astype(np.float64)))
        n_threads = _default_threads() if n_threads is None else max(1, int(n_threads))
        left = _RowBlocks(Xc, n_threads)
        right = _RowBlocks(Xc.T.tocsr(), n_threads)
        want_pool = max(left.n_threads, right.n_threads) > 1
    else:
        Xd = np.asarray(X)
        dtype = Xd.dtype if Xd.dtype in (np.float32, np.float64) else np.float64
        Xd = Xd.astype(dtype, copy=False)
        flat = Xd.reshape(-1).astype(np.float64, copy=False)
        x_norm_sq = float(np.dot(flat, flat))
        left = _DenseOperand(Xd)
        right = _DenseOperand(np.ascontiguousarray(Xd.T))
        want_pool = False

    rng = np.random.default_rng(seed)
    scale = np.sqrt(max(x_norm_sq / (n_cells * n_genes), _EPS) / k)
    H = np.abs(rng.normal(0.0, scale, size=(n_cells, k))).astype(dtype)
    W = np.abs(rng.normal(0.0, scale, size=(n_genes, k))).astype(dtype)

    report_every = max(1, max_iter // 20)
    prev = None
    err = float("nan")
    n_iter = 0
    converged = False

    pool = ThreadPoolExecutor(max(left.n_threads, right.n_threads)) if want_pool else None
    try:
        for it in range(max_iter):
            n_iter = it + 1
            _sweep(W, H, left.dot(W, pool), l1_h)
            XtH = right.dot(H, pool)
            _sweep(H, W, XtH, l1_w)

            # Squared residual for the *current* (H, W): XtH and HtH depend on
            # H alone, so the W we just wrote is the only thing that moved.
            HtH = (H.T @ H).astype(np.float64)
            WtW = (W.T @ W).astype(np.float64)
            cross = float(np.sum(XtH.astype(np.float64) * W.astype(np.float64)))
            err = max(x_norm_sq - 2.0 * cross + float(np.sum(WtW * HtH)), 0.0)

            # Relative to the error still on the table, not to the data norm:
            # scaling by ||X||^2 makes the threshold enormous next to a
            # late-stage residual and quits after a handful of passes.
            if prev is not None and (prev - err) <= tol * max(prev, _EPS):
                converged = True
                if progress_callback is not None:
                    progress_callback(1.0, f"converged after {n_iter} iterations")
                break
            prev = err
            if progress_callback is not None and it % report_every == 0:
                progress_callback(
                    min(0.99, it / max_iter), f"NMF iteration {n_iter}/{max_iter}"
                )
    finally:
        if pool is not None:
            pool.shutdown(wait=True)

    if progress_callback is not None and not converged:
        progress_callback(1.0, f"stopped after {n_iter} iterations")

    W, H, d = _l1_diagonalize(W.astype(np.float64), H.astype(np.float64))
    return {
        "W": W,
        "H": H,
        "d": d,
        "n_iter": int(n_iter),
        "converged": bool(converged),
        "error": float(np.sqrt(err)),
    }


# ---------------------------------------------------------------------------
# programs — GeneNMF's getNMFgenes()
# ---------------------------------------------------------------------------
def cumulative_cutoff(
    values: np.ndarray, weight_explained: float, max_genes: int | float = np.inf
):
    """GeneNMF's ``weightCumul``: the top entries that together account for
    ``weight_explained`` of the total.

    Returns ``(indices, shares)`` ordered by descending weight, where ``shares``
    are fractions of the *whole* vector, so they sum to just under
    ``weight_explained``.

    The cutoff is a strict ``<``, exactly as in GeneNMF — so a vector whose top
    entry already carries ``weight_explained`` on its own yields nothing at all,
    and the caller drops it rather than reporting a one-gene program.
    """
    values = np.asarray(values, dtype=float)
    total = float(values.sum())
    if total <= 0.0:
        return np.empty(0, dtype=int), np.empty(0, dtype=float)
    order = np.argsort(-values, kind="stable")
    shares = values[order] / total
    n_keep = int(np.count_nonzero(np.cumsum(shares) < weight_explained))
    n_keep = int(min(n_keep, max_genes))
    return order[:n_keep], shares[:n_keep]


def _specificity_weighted(W: np.ndarray, specificity_weight: float) -> np.ndarray:
    """GeneNMF's ``wgtLoad``: damp genes whose loading is spread over programs.

    A gene's specificity is the largest share any single program takes of its
    total loading — 1 for an exclusive gene, 1/k for one split evenly — raised
    to ``specificity_weight``. Housekeeping genes load on everything, so
    without this they dominate every program's top of list.

    The exponent bites hard, because on real data almost no gene is exclusive:
    in a 24k-cell E11.5 limb run the median gene's best program took only 27%
    of its loading, so at GeneNMF's default of 5 the most exclusive gene ends
    up ~600x ahead of the median and takes the whole program — after which the
    cumulative cutoff keeps nothing and the program is dropped (4 of 12
    survived, median 4 genes). GeneNMF gets away with 5 because it only ever
    applies this to the *averaged* consensus of many programs, which smooths
    the distribution first; there is no such averaging in the single-sample
    path, so the default here is 1.0 (11-12 of 12 survive, and the programs
    are visibly crisper than with no weighting at all). Measurements:
    docs/measurements/2026-08-19-gene-nmf-specificity-weight.md.
    """
    row_sums = W.sum(axis=1, keepdims=True)
    share = W / np.where(row_sums > 0.0, row_sums, 1.0)
    spec = share.max(axis=1, keepdims=True)
    return W * (spec ** specificity_weight)


def program_genes(
    W: np.ndarray,
    gene_names: list[str],
    *,
    specificity_weight: float = 1.0,
    weight_explained: float = 0.5,
    max_genes: int = 200,
    name_prefix: str = "NMF",
) -> list[dict[str, Any]]:
    """Turn gene loadings into one ranked gene set per program.

    Each program keeps its highest-loading genes up to the point where they
    cumulatively account for ``weight_explained`` of its total weight, capped
    at ``max_genes``.

    Returns one dict per surviving program: ``{'name', 'index', 'genes',
    'weights'}``. ``index`` is the factor's position in ``W``, kept so cell
    scores stay aligned when a program is dropped. ``weights`` are shares of
    the *whole* program, so they sum to just under ``weight_explained``.

    A program whose single top gene already carries ``weight_explained`` of the
    mass yields no genes at all — GeneNMF's cutoff is a strict ``<`` — and is
    dropped rather than returned empty, which is what GeneNMF does too.
    """
    W = np.asarray(W, dtype=float)
    if W.ndim != 2:
        raise ValueError(f"W must be 2-D (n_genes, k), got shape {W.shape}")
    if len(gene_names) != W.shape[0]:
        raise ValueError(
            f"gene_names has {len(gene_names)} entries but W has {W.shape[0]} rows"
        )
    if not (0.0 < weight_explained <= 1.0):
        raise ValueError(
            f"weight_explained must be in (0, 1], got {weight_explained!r}"
        )
    if specificity_weight < 0.0:
        raise ValueError(
            f"specificity_weight must be >= 0, got {specificity_weight!r}"
        )
    if specificity_weight:
        W = _specificity_weighted(W, specificity_weight)

    programs: list[dict[str, Any]] = []
    for j in range(W.shape[1]):
        kept, shares = cumulative_cutoff(W[:, j], weight_explained, max_genes)
        if kept.size == 0:
            continue
        programs.append({
            "name": f"{name_prefix}_{j + 1}",
            "index": j,
            "genes": [str(gene_names[i]) for i in kept],
            "weights": [float(v) for v in shares],
        })
    return programs


# ---------------------------------------------------------------------------
# end-to-end
# ---------------------------------------------------------------------------
def run_gene_programs(
    X,
    gene_names: list[str],
    *,
    k: int = 10,
    l1_w: float = 0.0,
    l1_h: float = 0.0,
    max_iter: int = 500,
    tol: float = 1e-4,
    seed: int = 0,
    specificity_weight: float = 1.0,
    weight_explained: float = 0.5,
    max_genes: int = 200,
    n_threads: int | None = None,
    progress_callback: Callable[[float, str], None] | None = None,
) -> dict[str, Any]:
    """Factorize, then extract one gene set per program.

    Returns the programs, the per-cell usage and per-gene loadings of the
    surviving ones (``cell_scores`` / ``loadings``, columns aligned to
    ``programs``), and JSON-safe run metrics. Everything but those two arrays
    is directly serializable.
    """
    inner = None
    if progress_callback is not None:
        # Reserve the last slice of the bar for gene extraction.
        def inner(frac: float, msg: str) -> None:
            progress_callback(min(0.95, 0.95 * float(frac)), msg)

    fit = nmf(
        X, k,
        l1_w=l1_w, l1_h=l1_h, max_iter=max_iter, tol=tol, seed=seed,
        n_threads=n_threads, progress_callback=inner,
    )
    programs = program_genes(
        fit["W"], gene_names,
        specificity_weight=specificity_weight,
        weight_explained=weight_explained,
        max_genes=max_genes,
    )
    kept = [p["index"] for p in programs]
    H, W = fit["H"], fit["W"]
    cell_scores = H[:, kept] if kept else np.zeros((H.shape[0], 0), dtype=H.dtype)
    loadings = W[:, kept] if kept else np.zeros((W.shape[0], 0), dtype=W.dtype)

    x_norm = float(np.linalg.norm(X.data if sp.issparse(X) else np.asarray(X)))
    if progress_callback is not None:
        progress_callback(1.0, f"{len(programs)} programs")

    return {
        "programs": programs,
        "cell_scores": cell_scores,
        "loadings": loadings,
        "n_programs": len(programs),
        "n_dropped": int(k - len(programs)),
        "k": int(k),
        "n_cells": int(X.shape[0]),
        "n_genes": int(X.shape[1]),
        "n_iter": fit["n_iter"],
        "converged": fit["converged"],
        "reconstruction_error": float(fit["error"]),
        "relative_error": (
            float(fit["error"] / x_norm) if x_norm > 0.0 else None
        ),
        "factor_weights": [float(fit["d"][i]) for i in kept],
        "params": {
            "k": int(k), "l1_w": float(l1_w), "l1_h": float(l1_h),
            "max_iter": int(max_iter), "tol": float(tol), "seed": int(seed),
            "specificity_weight": float(specificity_weight),
            "weight_explained": float(weight_explained),
            "max_genes": int(max_genes),
        },
    }


# ---------------------------------------------------------------------------
# multi-sample: meta-programs across samples and ranks
# ---------------------------------------------------------------------------
def multi_nmf(
    X,
    gene_names: list[str],
    sample_labels,
    *,
    ks=(4, 5, 6),
    samples: list[str] | None = None,
    min_cells: int = 10,
    l1_w: float = 0.0,
    l1_h: float = 0.0,
    max_iter: int = 500,
    tol: float = 1e-4,
    seed: int = 0,
    n_threads: int | None = None,
    progress_callback: Callable[[float, str], None] | None = None,
) -> dict[str, Any]:
    """Factorize each sample on its own, at each rank in ``ks``.

    GeneNMF's ``multiNMF``. Running samples separately is the point: a program
    fitted on pooled data can be a batch effect, whereas one that turns up
    independently in several samples cannot. Sweeping ``k`` as well means a
    program does not have to be an artifact of one arbitrary rank.

    Args:
        X: cells x genes, non-negative (dense or sparse).
        sample_labels: one label per cell.
        ks: ranks to fit per sample. A rank larger than a sample's cell or gene
            count is dropped rather than failing the whole run.
        samples: sample order to use; categories with no cells are ignored.
            Defaults to first-appearance order.
        min_cells: samples smaller than this are skipped and reported.

    Returns ``{'loadings': (n_genes, n_programs), 'labels', 'samples', 'ks',
    'model_index', 'ks_used', 'skipped'}``. Each loading column is L1
    normalized, and ``model_index`` says which (sample, k) factorization a
    column came from — specificity weighting is computed *within* a
    factorization, so that grouping has to survive.
    """
    labels_arr = np.asarray(sample_labels).astype(str)
    n_cells, n_genes = X.shape
    if labels_arr.shape[0] != n_cells:
        raise ValueError(
            f"sample_labels has {labels_arr.shape[0]} entries but X has "
            f"{n_cells} cells"
        )
    if len(gene_names) != n_genes:
        raise ValueError(
            f"gene_names has {len(gene_names)} entries but X has {n_genes} genes"
        )
    ks_req = sorted({int(k) for k in ks})
    if not ks_req or any(k < 2 for k in ks_req):
        raise ValueError(f"ks must be integers >= 2, got {sorted(ks)!r}")

    if samples is None:
        _, first = np.unique(labels_arr, return_index=True)
        order = [str(labels_arr[i]) for i in sorted(first)]
    else:
        order = [str(s) for s in samples]

    sparse = sp.issparse(X)
    Xc = X.tocsr() if sparse else np.asarray(X)

    runs: list[tuple[str, int, np.ndarray]] = []
    skipped: list[dict[str, Any]] = []
    for sample in order:
        rows = np.flatnonzero(labels_arr == sample)
        if rows.size == 0:
            # A leftover .obs category no cell uses any more; not a skip.
            continue
        if rows.size < min_cells:
            skipped.append({
                "sample": sample, "n_cells": int(rows.size),
                "reason": f"fewer than {min_cells} cells",
            })
            continue
        runs.append((sample, rows.size, rows))

    plan = [
        (sample, k, rows)
        for sample, n, rows in runs
        for k in ks_req
        if k <= min(n, n_genes)
    ]
    if not plan:
        raise ValueError(
            "Nothing to factorize: every sample was too small, or every k "
            "exceeded the data. Lower k or min_cells."
        )

    cols: list[np.ndarray] = []
    out_labels: list[str] = []
    out_samples: list[str] = []
    out_ks: list[int] = []
    model_index: list[int] = []
    for m, (sample, k, rows) in enumerate(plan):
        sub = Xc[rows] if sparse else Xc[rows, :]
        fit = nmf(
            sub, k, l1_w=l1_w, l1_h=l1_h, max_iter=max_iter, tol=tol,
            seed=seed, n_threads=n_threads,
        )
        W = fit["W"]
        for j in range(W.shape[1]):
            cols.append(W[:, j])
            out_labels.append(f"{sample}.k{k}.{j + 1}")
            out_samples.append(sample)
            out_ks.append(k)
            model_index.append(m)
        if progress_callback is not None:
            progress_callback(
                (m + 1) / len(plan), f"{sample} k={k} ({m + 1}/{len(plan)})",
            )

    return {
        "loadings": np.column_stack(cols),
        "labels": out_labels,
        "samples": out_samples,
        "ks": out_ks,
        "model_index": np.asarray(model_index, dtype=int),
        "ks_used": sorted({k for _, k, _ in plan}),
        "skipped": skipped,
    }


def weighted_loadings(fit: dict[str, Any], specificity_weight: float) -> np.ndarray:
    """Specificity-weight each factorization's loadings *within* that model.

    GeneNMF applies ``wgtLoad`` per model, so "how exclusive is this gene"
    always means "among this factorization's own k factors" — comparing across
    every program in the study would make a gene's score depend on how many
    samples happened to be included.
    """
    W = np.asarray(fit["loadings"], dtype=float)
    if not specificity_weight:
        return W
    out = np.empty_like(W)
    model_index = np.asarray(fit["model_index"])
    for m in np.unique(model_index):
        cols = np.flatnonzero(model_index == m)
        out[:, cols] = _specificity_weighted(W[:, cols], specificity_weight)
    return out


def _program_similarity(
    Wg: np.ndarray, metric: str, weight_explained: float, max_genes: int
) -> np.ndarray:
    """Program x program similarity — cosine on weights, or Jaccard on sets."""
    if metric == "cosine":
        norms = np.linalg.norm(Wg, axis=0)
        Z = Wg / np.where(norms > 0, norms, 1.0)
        S = Z.T @ Z
    elif metric == "jaccard":
        n = Wg.shape[1]
        member = np.zeros((Wg.shape[0], n), dtype=np.float64)
        for j in range(n):
            kept, _ = cumulative_cutoff(Wg[:, j], weight_explained, max_genes)
            member[kept, j] = 1.0
        inter = member.T @ member
        sizes = member.sum(axis=0)
        union = sizes[:, None] + sizes[None, :] - inter
        S = np.divide(inter, union, out=np.zeros_like(inter), where=union > 0)
    else:
        raise ValueError(f"Unknown metric {metric!r}; expected 'cosine' or 'jaccard'")
    S = 0.5 * (S + S.T)
    np.clip(S, -1.0, 1.0, out=S)
    np.fill_diagonal(S, 1.0)
    return S


def trimmed_mean(sub: np.ndarray) -> np.ndarray:
    """Per-gene mean across a cluster's programs, dropping >3 SD outliers.

    With fewer than three programs there is no usable SD, so it is a plain mean
    — which is what GeneNMF does too.
    """
    mean = sub.mean(axis=1)
    if sub.shape[1] < 3:
        return mean
    sd = sub.std(axis=1, ddof=1)
    lo, hi = mean - 3.0 * sd, mean + 3.0 * sd
    keep = (sub > lo[:, None]) & (sub < hi[:, None])
    counts = keep.sum(axis=1)
    trimmed = np.where(counts > 0, (sub * keep).sum(axis=1) / np.maximum(counts, 1), mean)
    return trimmed


def meta_programs(
    fit: dict[str, Any],
    gene_names: list[str],
    *,
    n_mp: int = 10,
    # Deliberately not the single-sample defaults. The consensus average
    # smooths the specificity weighting, so GeneNMF's exponent of 5 works here
    # — but only paired with a higher cutoff, because after weighting the top
    # genes carry most of a program's mass. Measured on three real limb
    # sections: docs/measurements/2026-08-19-gene-nmf-metaprogram-defaults.md
    specificity_weight: float = 5.0,
    weight_explained: float = 0.8,
    max_genes: int = 200,
    metric: str = "cosine",
    min_confidence: float = 0.5,
    linkage_method: str = "ward",
) -> dict[str, Any]:
    """Cluster every program from :func:`multi_nmf` into consensus programs.

    GeneNMF's ``getMetaPrograms``. Programs are compared by their gene weights,
    clustered, and each cluster condensed into one gene set: the outlier-trimmed
    mean weight across its programs, cut at ``weight_explained``, then filtered
    to genes that actually recur (``min_confidence`` — the fraction of the
    cluster's programs whose own gene set contains the gene).

    Returns the meta-programs ranked by sample coverage then silhouette, plus
    the program similarity matrix, cluster assignment and dendrogram leaf order
    for the heatmap, and per-sample composition counts.
    """
    from scipy.cluster.hierarchy import fcluster, leaves_list, linkage
    from scipy.spatial.distance import squareform

    labels = list(fit["labels"])
    prog_samples = [str(s) for s in fit["samples"]]
    n_programs = len(labels)
    if not isinstance(n_mp, (int, np.integer)) or n_mp < 1:
        raise ValueError(f"n_mp must be an integer >= 1, got {n_mp!r}")
    if n_mp > n_programs:
        raise ValueError(
            f"n_mp={n_mp} exceeds the {n_programs} programs found; ask for "
            f"fewer meta-programs or sweep more values of k"
        )
    if not (0.0 <= min_confidence <= 1.0):
        raise ValueError(f"min_confidence must be in [0, 1], got {min_confidence!r}")

    Wg = weighted_loadings(fit, specificity_weight)
    S = _program_similarity(Wg, metric, weight_explained, max_genes)

    D = np.clip(1.0 - S, 0.0, 2.0)
    np.fill_diagonal(D, 0.0)
    Z = linkage(squareform(D, checks=False), method=linkage_method)
    raw = fcluster(Z, t=n_mp, criterion="maxclust") - 1
    order = list(int(i) for i in leaves_list(Z))

    # Each program's own gene set, for the recurrence (confidence) filter.
    per_program = [
        set(cumulative_cutoff(Wg[:, j], 0.9, 1000)[0].tolist())
        for j in range(n_programs)
    ]

    all_samples = sorted(set(prog_samples))
    sil = None
    if 2 <= len(set(raw.tolist())) <= n_programs - 1:
        from sklearn.metrics import silhouette_samples
        sil = silhouette_samples(D, raw, metric="precomputed")

    built: list[dict[str, Any]] = []
    for c in sorted(set(raw.tolist())):
        cols = np.flatnonzero(raw == c)
        avg = trimmed_mean(Wg[:, cols])
        kept, _ = cumulative_cutoff(avg, weight_explained)

        if min_confidence > 0.0 and kept.size:
            seen = np.zeros(len(gene_names), dtype=float)
            for j in cols:
                seen[list(per_program[j])] += 1.0
            confidence = seen / len(cols)
            kept = kept[confidence[kept] > min_confidence]
        kept = kept[:max_genes]

        members = [prog_samples[j] for j in cols]
        in_cluster = S[np.ix_(cols, cols)]
        iu = np.triu_indices(len(cols), k=1)
        built.append({
            "cluster": int(c),
            "genes": [str(gene_names[i]) for i in kept],
            "raw_weights": avg[kept],
            "n_programs": int(len(cols)),
            "sample_coverage": float(len(set(members)) / len(all_samples)),
            "silhouette": float(sil[cols].mean()) if sil is not None else 0.0,
            "mean_similarity": (
                float(in_cluster[iu].mean()) if len(cols) > 1 else 0.0
            ),
            "samples": members,
        })

    kept_mps = [mp for mp in built if mp["genes"]]
    n_dropped = len(built) - len(kept_mps)
    kept_mps.sort(key=lambda m: (m["sample_coverage"], m["silhouette"]), reverse=True)

    remap = {mp["cluster"]: i for i, mp in enumerate(kept_mps)}
    clusters = [int(remap.get(int(c), -1)) for c in raw]

    metaprograms: list[dict[str, Any]] = []
    composition: dict[str, dict[str, int]] = {}
    for i, mp in enumerate(kept_mps):
        name = f"MP{i + 1}"
        w = np.asarray(mp["raw_weights"], dtype=float)
        total = float(w.sum())
        metaprograms.append({
            "name": name,
            "genes": mp["genes"],
            # Renormalized within the meta-program, so a weight reads as a
            # share of it whatever the cutoff kept.
            "weights": [float(v) for v in (w / total if total > 0 else w)],
            "n_programs": mp["n_programs"],
            "n_genes": len(mp["genes"]),
            "sample_coverage": mp["sample_coverage"],
            "silhouette": mp["silhouette"],
            "mean_similarity": mp["mean_similarity"],
        })
        counts = {s: 0 for s in all_samples}
        for s in mp["samples"]:
            counts[s] += 1
        composition[name] = counts

    return {
        "metaprograms": metaprograms,
        "composition": composition,
        "similarity": S,
        "linkage": Z,
        "order": order,
        "clusters": clusters,
        "program_labels": labels,
        "program_samples": prog_samples,
        "program_ks": [int(k) for k in fit["ks"]],
        "samples": all_samples,
        "n_dropped": int(n_dropped),
        "n_programs": n_programs,
    }


def score_metaprograms(X, gene_names: list[str], metaprograms) -> np.ndarray:
    """Per-cell score for each meta-program: its weighted mean expression.

    GeneNMF leaves scoring to the user (its demos reach for UCell on the MP
    gene sets); this is the same idea with the consensus weights already in
    hand, so a meta-program is colorable the moment it exists.
    """
    index = {str(g): i for i, g in enumerate(gene_names)}
    n_cells = X.shape[0]
    out = np.zeros((n_cells, len(metaprograms)), dtype=np.float64)
    for j, mp in enumerate(metaprograms):
        idx = [index[g] for g in mp["genes"] if g in index]
        if not idx:
            continue
        w = np.asarray(
            [mp["weights"][i] for i, g in enumerate(mp["genes"]) if g in index],
            dtype=float,
        )
        total = float(w.sum())
        if total <= 0:
            continue
        sub = X[:, idx]
        vals = sub.toarray() if sp.issparse(sub) else np.asarray(sub)
        out[:, j] = (vals @ w) / total
    return out


def run_meta_programs(
    X,
    gene_names: list[str],
    sample_labels,
    *,
    ks=(4, 5, 6),
    n_mp: int = 10,
    samples: list[str] | None = None,
    min_cells: int = 10,
    l1_w: float = 0.0,
    l1_h: float = 0.0,
    max_iter: int = 500,
    tol: float = 1e-4,
    seed: int = 0,
    specificity_weight: float = 5.0,   # see meta_programs()
    weight_explained: float = 0.8,
    max_genes: int = 200,
    metric: str = "cosine",
    min_confidence: float = 0.5,
    linkage_method: str = "ward",
    n_threads: int | None = None,
    progress_callback: Callable[[float, str], None] | None = None,
) -> dict[str, Any]:
    """Factorize every sample at every k, then consolidate into meta-programs.

    Adds ``cell_scores`` (cells x meta-programs) to what :func:`meta_programs`
    returns. Everything except ``cell_scores``, ``similarity`` and ``linkage``
    is JSON-serializable.
    """
    inner = None
    if progress_callback is not None:
        def inner(frac: float, msg: str) -> None:
            progress_callback(min(0.9, 0.9 * float(frac)), msg)

    fit = multi_nmf(
        X, gene_names, sample_labels, ks=ks, samples=samples, min_cells=min_cells,
        l1_w=l1_w, l1_h=l1_h, max_iter=max_iter, tol=tol, seed=seed,
        n_threads=n_threads, progress_callback=inner,
    )
    if progress_callback is not None:
        progress_callback(0.92, f"clustering {len(fit['labels'])} programs")

    out = meta_programs(
        fit, gene_names, n_mp=n_mp, specificity_weight=specificity_weight,
        weight_explained=weight_explained, max_genes=max_genes, metric=metric,
        min_confidence=min_confidence, linkage_method=linkage_method,
    )
    out["cell_scores"] = score_metaprograms(X, gene_names, out["metaprograms"])
    out["skipped"] = fit["skipped"]
    out["ks_used"] = fit["ks_used"]
    out["params"] = {
        "ks": [int(k) for k in fit["ks_used"]], "n_mp": int(n_mp),
        "min_cells": int(min_cells), "seed": int(seed),
        "specificity_weight": float(specificity_weight),
        "weight_explained": float(weight_explained),
        "max_genes": int(max_genes), "metric": metric,
        "min_confidence": float(min_confidence),
        "linkage_method": linkage_method,
        "l1_w": float(l1_w), "l1_h": float(l1_h),
        "max_iter": int(max_iter), "tol": float(tol),
    }
    if progress_callback is not None:
        progress_callback(1.0, f"{len(out['metaprograms'])} meta-programs")
    return out
