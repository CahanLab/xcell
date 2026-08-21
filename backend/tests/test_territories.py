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


def test_one_cut_splits_the_ring_into_two_faces_that_tile_it():
    faces = terr.derive_faces(SQUARE, [_cut([[2.0, 5.0], [8.0, 5.0]])])
    assert len(faces) == 2
    assert sum(f.area for f in faces) == pytest.approx(100.0)
    assert faces[0].intersection(faces[1]).area == pytest.approx(0.0)


def test_two_crossing_cuts_make_four_faces():
    cuts = [_cut([[1.0, 5.0], [9.0, 5.0]], cid="h"),
            _cut([[5.0, 1.0], [5.0, 9.0]], cid="v")]
    faces = terr.derive_faces(SQUARE, cuts)
    assert len(faces) == 4
    assert sum(f.area for f in faces) == pytest.approx(100.0)


def test_a_closed_cut_makes_an_island_and_its_surround():
    loop = _cut([[4.0, 4.0], [6.0, 4.0], [6.0, 6.0], [4.0, 6.0]], closed=True)
    faces = terr.derive_faces(SQUARE, [loop])
    assert len(faces) == 2
    areas = sorted(f.area for f in faces)
    assert areas[0] == pytest.approx(4.0)     # the island
    assert areas[1] == pytest.approx(96.0)    # everything else


def test_no_cuts_leaves_the_ring_as_a_single_face():
    faces = terr.derive_faces(SQUARE, [])
    assert len(faces) == 1
    assert faces[0].area == pytest.approx(100.0)


def test_face_order_is_deterministic_so_the_ui_does_not_reshuffle():
    cuts = [_cut([[1.0, 5.0], [9.0, 5.0]])]
    first = terr.derive_faces(SQUARE, cuts)
    second = terr.derive_faces(SQUARE, list(reversed(cuts)))
    assert [f.centroid.y for f in first] == [f.centroid.y for f in second]
    # top face first: reading order, not shapely's internal order
    assert first[0].centroid.y > first[1].centroid.y


def test_a_self_intersecting_freehand_cut_still_yields_valid_faces():
    """Noding in unary_union handles the crossing; the extra loop becomes its
    own face rather than an invalid geometry."""
    squiggle = _cut([[2.0, 4.0], [8.0, 6.0], [8.0, 4.0], [2.0, 6.0]])
    faces = terr.derive_faces(SQUARE, [squiggle])
    assert all(f.is_valid for f in faces)
    assert sum(f.area for f in faces) == pytest.approx(100.0)
