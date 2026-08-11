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
