"""Merging neighbouring ST spots into single large cells.

A spot is a patch of tissue, not a cell. Where the cell is much larger than the
spot — a hypertrophic chondrocyte against a Visium HD bin — one cell is spread
across several spots, each holding a fraction of its transcripts, so every spot
in the region looks shallow and none is a usable profile of the cell.

This module puts the pieces back: it groups neighbouring spots up to the size of
one cell so their counts can be summed. Geometry drives the grouping, because
the low-count profiles that motivate merging are exactly the ones too noisy to
decide it — a correlation between two 40-count profiles is largely sampling
noise, and ranking by it gives a different answer on every run. Similarity
enters only as a veto.

Pure: arrays in, arrays out. Knows nothing of AnnData.
"""
from __future__ import annotations

import math

import numpy as np


def estimate_pitch_um(coords: np.ndarray) -> float:
    """Spot spacing, as the median nearest-neighbour distance.

    Measured rather than assumed: it costs one KD-tree query, and it turns a
    wrong µm-per-unit into an obviously absurd number the user can see instead
    of a silently wrong size cap on every merge.
    """
    if len(coords) < 2:
        raise ValueError("Estimating spot pitch needs at least two spots.")
    from scipy.spatial import cKDTree

    # k=2 because a point's own nearest neighbour is itself, at distance 0.
    distances, _ = cKDTree(coords).query(coords, k=2)
    return float(np.median(distances[:, 1]))


def max_spots_for(diameter_um: float, pitch_um: float) -> int:
    """How many lattice points a disc of this diameter covers."""
    area_ratio = math.pi * (diameter_um / 2.0) ** 2 / (pitch_um ** 2)
    return max(1, int(math.ceil(area_ratio)))
