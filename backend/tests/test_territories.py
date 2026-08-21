"""Territory geometry — cuts in, faces out.

The invariant every test here defends: faces derived from a ring plus its cuts
tile the ring exhaustively and disjointly. Overlaps and gaps are not errors to
be detected, they are states the representation cannot reach.
"""
import numpy as np
import pytest

from xcell import territories as terr


SQUARE = [[0.0, 0.0], [10.0, 0.0], [10.0, 10.0], [0.0, 10.0]]


def _cut(points, closed=False, cid="c1"):
    return {"id": cid, "points": points, "closed": closed}


def test_a_cut_stopping_short_is_extended_to_the_ring():
    """Freehand cuts never land on the edge; a cut that divides nothing would
    be a silently missing boundary."""
    short = _cut([[2.0, 5.0], [8.0, 5.0]])
    (geom,) = terr.extend_cuts(SQUARE, [short])
    xs = [p[0] for p in geom.coords]
    assert min(xs) == pytest.approx(0.0)
    assert max(xs) == pytest.approx(10.0)


def test_overshoot_is_trimmed_at_the_ring():
    over = _cut([[-50.0, 5.0], [60.0, 5.0]])
    (geom,) = terr.extend_cuts(SQUARE, [over])
    xs = [p[0] for p in geom.coords]
    assert min(xs) == pytest.approx(0.0)
    assert max(xs) == pytest.approx(10.0)


def test_a_closed_cut_is_kept_as_a_loop_not_extended():
    loop = _cut([[4.0, 4.0], [6.0, 4.0], [6.0, 6.0], [4.0, 6.0]], closed=True)
    (geom,) = terr.extend_cuts(SQUARE, [loop])
    assert geom.is_closed


def test_a_cut_entirely_outside_the_ring_is_dropped():
    outside = _cut([[20.0, 20.0], [30.0, 20.0]])
    assert terr.extend_cuts(SQUARE, [outside]) == []


def test_a_degenerate_cut_of_one_point_is_dropped():
    assert terr.extend_cuts(SQUARE, [_cut([[5.0, 5.0]])]) == []
