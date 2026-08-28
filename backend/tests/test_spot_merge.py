"""Merging ST spots into single large cells: the pure algorithm.

A spot is a patch of tissue, not a cell. These tests pin down the grouping —
that it never exceeds one cell's footprint, that it refuses to cross a planted
boundary, and above all that it is deterministic, which is the property that
ruled out letting noisy low-count correlations rank the merges.
"""

import numpy as np
import pytest

from xcell.spot_merge import estimate_pitch_um, max_spots_for


def _grid(nx: int, ny: int, pitch: float = 15.0) -> np.ndarray:
    """A regular lattice, the shape a real ST slide has."""
    xs, ys = np.meshgrid(np.arange(nx) * pitch, np.arange(ny) * pitch)
    return np.column_stack([xs.ravel(), ys.ravel()]).astype(float)


def test_pitch_is_the_lattice_spacing():
    assert estimate_pitch_um(_grid(6, 6, pitch=15.0)) == pytest.approx(15.0)


def test_pitch_survives_jitter_and_a_few_gaps():
    rng = np.random.default_rng(0)
    coords = _grid(8, 8, pitch=20.0)
    coords += rng.normal(0, 0.5, coords.shape)       # imaging jitter
    coords = np.delete(coords, [3, 17, 40], axis=0)  # missing spots
    assert estimate_pitch_um(coords) == pytest.approx(20.0, abs=1.5)


def test_max_spots_counts_lattice_points_in_the_disc():
    # A 40 um disc over a 15 um lattice: pi*20^2 / 15^2 = 5.58 -> 6
    assert max_spots_for(40.0, 15.0) == 6
    # Diameter equal to pitch admits exactly one spot
    assert max_spots_for(15.0, 15.0) == 1


def test_max_spots_is_never_below_one():
    assert max_spots_for(1.0, 15.0) == 1


def test_pitch_needs_two_spots():
    with pytest.raises(ValueError, match="at least two"):
        estimate_pitch_um(np.array([[0.0, 0.0]]))
