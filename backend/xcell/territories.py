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
