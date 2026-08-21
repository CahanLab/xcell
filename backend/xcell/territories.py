"""Hand-drawn spatial territories: dividing cuts in, regions out.

A territory type stores the curves that *divide* a space, not the regions
between them. The regions are derived, so a boundary is one object shared by
both of its neighbours: move it and both follow. Overlaps and gaps are not
detected and repaired here because the representation cannot express them.

Pure: plain lists and arrays in, plain lists and arrays out. Never imports the
adaptor, never touches AnnData.
"""
from __future__ import annotations

from typing import Any

import numpy as np
from shapely.geometry import LineString, Point, Polygon
from shapely.ops import polygonize, unary_union


def extend_cuts(ring: list[list[float]], cuts: list[dict[str, Any]]) -> list[LineString]:
    """Project each open cut's ends outward, then clip every cut to the ring.

    A freehand cut rarely lands on the tissue edge, and a cut that stops short
    divides nothing. Each end is extended along its own terminal direction by
    more than the ring's diagonal and the result intersected with the ring, so
    the extension stops exactly at the boundary and overshoot is trimmed the
    same way. The caller's drawn points are never modified — extension is
    derived every time, so repeated editing cannot compound.
    """
    ring_poly = Polygon(ring)
    minx, miny, maxx, maxy = ring_poly.bounds
    reach = float(np.hypot(maxx - minx, maxy - miny)) or 1.0

    out: list[LineString] = []
    for cut in cuts:
        pts = [(float(x), float(y)) for x, y in cut.get("points", [])]
        if len(pts) < 2:
            continue

        if cut.get("closed"):
            geom = LineString(pts + [pts[0]])
        else:
            geom = LineString([_project(pts[0], pts[1], reach)] + pts
                              + [_project(pts[-1], pts[-2], reach)])

        clipped = geom.intersection(ring_poly)
        if clipped.is_empty:
            continue
        out.append(clipped)
    return out


def _project(end: tuple[float, float], inward: tuple[float, float],
             reach: float) -> tuple[float, float]:
    """`end` pushed away from `inward` by `reach`."""
    dx, dy = end[0] - inward[0], end[1] - inward[1]
    norm = float(np.hypot(dx, dy))
    if norm == 0:
        return end
    return (end[0] + dx / norm * reach, end[1] + dy / norm * reach)


def derive_faces(ring: list[list[float]], cuts: list[dict[str, Any]]) -> list[Polygon]:
    """The regions the cuts divide the ring into.

    ``unary_union`` nodes every crossing (including a freehand cut crossing
    itself), and ``polygonize`` then builds the faces bounded by that noded
    network. Faces come out exhaustive and disjoint by construction, which is
    the whole reason boundaries rather than regions are the stored object.

    Ordered top-to-bottom then left-to-right so the UI's face list is stable
    across re-derivations.
    """
    ring_poly = Polygon(ring)
    network = unary_union([ring_poly.exterior] + extend_cuts(ring, cuts))
    faces = [f for f in polygonize(network)
             if f.representative_point().within(ring_poly)]
    return sorted(faces, key=lambda f: (-round(f.centroid.y, 9),
                                        round(f.centroid.x, 9)))


def name_faces(faces: list[Polygon],
               anchors: list[dict[str, Any]]) -> list[str | None]:
    """Each face takes the name of an anchor point it contains.

    Names live on points rather than on faces so that editing a boundary
    cannot orphan a label: re-derive the faces, and every anchor is still
    inside whichever face now surrounds it.
    """
    out: list[str | None] = []
    for face in faces:
        hit = None
        for anchor in anchors:
            if face.contains(Point(float(anchor["x"]), float(anchor["y"]))):
                hit = str(anchor["name"])
                break
        out.append(hit)
    return out


def assign(coords, faces: list[Polygon], names: list[str | None],
           *, unassigned: str = "unassigned") -> np.ndarray:
    """Label every coordinate by the face containing it.

    A coordinate outside every face, or inside an unnamed one, gets
    ``unassigned`` — never the nearest face. Snapping to the nearest region
    would invent an annotation the user never drew.
    """
    import shapely
    from shapely import STRtree

    coords = np.asarray(coords, dtype=float)
    labels = np.full(len(coords), unassigned, dtype=object)
    if not faces or len(coords) == 0:
        return labels

    tree = STRtree(faces)
    # The predicate is evaluated as input.predicate(tree) — so the point must be
    # *within* the face. "contains" reads the right way round in English and
    # silently matches nothing, which is why this is spelled out.
    input_idx, tree_idx = tree.query(shapely.points(coords), predicate="within")

    claimed = np.zeros(len(coords), dtype=bool)
    for i, t in zip(input_idx, tree_idx):
        # A point landing exactly on a shared edge matches both neighbours;
        # first match wins, deterministically, because faces are ordered.
        if claimed[i]:
            continue
        claimed[i] = True
        if names[t] is not None:
            labels[i] = names[t]
    return labels
