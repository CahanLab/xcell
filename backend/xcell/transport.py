"""Entropic optimal transport for spatial localization.

Every other aggregation in ``localize`` sits at one end of a single trade-off:
averaging keeps the gradients and collapses the map toward the tissue's centre,
while taking one location per cell fills the tissue and destroys the gradients.
Transport is the only estimator that moves *along* that frontier, because a
marginal constraint on the reference side forces the predicted population to
occupy the tissue while the entropy term still lets each cell hedge across
several spots.

Pure: takes arrays, returns arrays. Never imports the adaptor and never sees
AnnData, so it is testable without one.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np


def normalize_cost(cost: np.ndarray) -> tuple[np.ndarray, float]:
    """Make the cost matrix dimensionless, so epsilon means the same thing.

    Sinkhorn forms exp(-C/eps), so what epsilon trades against is the *spread*
    of C, not its absolute scale. Measured on the E11.5 limb pair, the
    transform/metric choice moves the effective epsilon across 3.5x (0.60
    spreads under none+cosine, 2.09 under rank+cosine) while the useful tuning
    range spans only 2.5x -- the preprocessing moves the dial further than the
    dial travels. Dividing by the spread is what lets one default survive every
    setting.

    The shift is free: entropic OT is invariant to a constant added to C,
    because the resulting exp(-a/eps) is absorbed into the scaling vectors.
    """
    c = np.asarray(cost, dtype=np.float64)
    c = c - c.min()
    sigma = float(c.std())
    if sigma <= 1e-12:
        # Every pair equally similar. There is nothing to scale, and dividing
        # would fill the matrix with inf.
        return c, 1.0
    return c / sigma, sigma


@dataclass
class SinkhornResult:
    """A coupling, plus what it cost to get and how well it satisfies itself."""

    coupling: np.ndarray      # (n_query, n_ref), rows sum to 1/n, columns to 1/m
    iterations: int
    # Max relative deviation of a *row* sum from 1/n. Rows, not columns,
    # because the column update runs last: right after it the column marginal
    # holds to machine precision no matter how far from converged the coupling
    # is, so measuring it would report ~1e-15 on iteration two and stop there.
    # The un-enforced marginal is the only one carrying information.
    marginal_error: float


def sinkhorn_log(
    cost: np.ndarray,
    epsilon: float,
    *,
    max_iterations: int = 200,
    tol: float = 1e-4,
    check_every: int = 10,
    progress: Callable[[float, str], None] | None = None,
) -> SinkhornResult:
    """Entropic OT in the log domain. ``cost`` must already be normalized.

    Log-domain always, rather than the faster kernel form. The kernel form does
    not fail on the kernel entries -- it fails on the scaling vectors, which
    grow and shrink without bound over iterations, and it does so at exactly
    the small epsilon where this estimator is most interesting. One stable path
    is worth more than the 2-3x per-iteration saving of a form that breaks
    where it is most needed.
    """
    from scipy.special import logsumexp

    if epsilon <= 0:
        raise ValueError(f'epsilon must be positive, got {epsilon}')
    if max_iterations < 1:
        raise ValueError(f'max_iterations must be at least 1, got {max_iterations}')

    c = np.asarray(cost, dtype=np.float64)
    n, m = c.shape
    log_a = -np.log(n)
    log_b = -np.log(m)
    f = np.zeros(n, dtype=np.float64)
    g = np.zeros(m, dtype=np.float64)

    target_row = 1.0 / n
    used = 0
    for it in range(1, max_iterations + 1):
        used = it
        M = (-c + f[:, None] + g[None, :]) / epsilon
        f += epsilon * (log_a - logsumexp(M, axis=1))
        M = (-c + f[:, None] + g[None, :]) / epsilon
        g += epsilon * (log_b - logsumexp(M, axis=0))

        # Checking every iteration doubles the work in the loop for a number
        # that moves slowly; never checking means we can only ever spend the
        # whole budget.
        if it % check_every == 0 or it == max_iterations:
            P = np.exp((-c + f[:, None] + g[None, :]) / epsilon)
            err = float(np.abs(P.sum(axis=1) - target_row).max() / target_row)
            if progress is not None:
                progress(it / max_iterations,
                         f'Sinkhorn iteration {it} of {max_iterations}, '
                         f'marginal error {err:.2e}')
            if err < tol:
                break

    P = np.exp((-c + f[:, None] + g[None, :]) / epsilon)
    err = float(np.abs(P.sum(axis=1) - target_row).max() / target_row)
    return SinkhornResult(coupling=P, iterations=used, marginal_error=err)


# --- reading a coupling -----------------------------------------------------

def _row_normalize(coupling: np.ndarray) -> np.ndarray:
    """Rows of P sum to 1/n; a posterior over locations must sum to 1."""
    P = np.asarray(coupling, dtype=np.float64)
    total = P.sum(axis=1, keepdims=True)
    return P / np.where(total > 0, total, 1.0)


def barycentric_projection(
    coupling: np.ndarray,
    coords: np.ndarray,
    sections: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray | None]:
    """Predict each cell's coordinate as the mean of its posterior.

    Taking the row argmax instead returns to the single-location regime and
    loses the axes (measured: 0.98 area, ~0 correlation), which is the failure
    this estimator exists to avoid.
    """
    W = _row_normalize(coupling)
    coords = np.asarray(coords, dtype=np.float64)
    if sections is None:
        return W @ coords, None

    # Averaging across sections lands in the gap between cuts, so resolve each
    # row to the section holding most of its mass and renormalize within it.
    sections = np.asarray(sections).astype(str)
    labels = np.unique(sections)
    mass = np.stack([W[:, sections == s].sum(axis=1) for s in labels], axis=1)
    chosen = labels[np.argmax(mass, axis=1)]

    out = np.zeros((W.shape[0], coords.shape[1]), dtype=np.float64)
    for s in labels:
        rows = np.flatnonzero(chosen == s)
        if rows.size == 0:
            continue
        cols = np.flatnonzero(sections == s)
        sub = W[np.ix_(rows, cols)]
        total = sub.sum(axis=1, keepdims=True)
        sub = sub / np.where(total > 0, total, 1.0)
        out[rows] = sub @ coords[cols]
    return out, chosen


def posterior_spread(
    coupling: np.ndarray,
    coords: np.ndarray,
    predictions: np.ndarray,
) -> np.ndarray:
    """Weighted RMS distance of the posterior from its own mean, per cell.

    The same quantity ``project_knn`` already computes over the k neighbours,
    evaluated over the coupling instead. Keeping the formula identical is what
    lets confidence stay comparable across aggregations -- row entropy was the
    other candidate and is not in spatial units, so a transport run's
    confidence could not be read against any other run's.
    """
    W = _row_normalize(coupling)
    coords = np.asarray(coords, dtype=np.float64)
    predictions = np.asarray(predictions, dtype=np.float64)
    offsets = coords[None, :, :] - predictions[:, None, :]
    return np.sqrt((W * (offsets ** 2).sum(axis=2)).sum(axis=1))
