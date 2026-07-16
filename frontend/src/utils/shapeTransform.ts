// Geometry helpers for keeping drawn shapes/lines aligned with the cells they
// overlay when those cells are transformed (rotated, reflected, shifted).
//
// The affine here mirrors the backend `transform_embedding` (adaptor.py) EXACTLY
// so a shape lands on the same cells after a transform as before it:
//   1. translate to origin (subtract centroid)
//   2. reflect   — reflectY negates x (backend reflect_y), reflectX negates y
//                  (backend reflect_x)
//   3. rotate    — counter-clockwise, matrix [[cos,-sin],[sin,cos]]
//   4. translate back (add centroid)
//   5. apply the (translateX, translateY) offset
//
// Rotations and reflections are performed around `centroid`, which the caller
// computes from the cells' PRE-transform positions (the same centroid the
// backend uses: all cells for a whole-embedding "adjust", or the selected
// subset for a "quilt" transform).

export type Pt = [number, number]

export interface ShapeAffine {
  centroid: Pt
  rotationDegrees?: number
  reflectX?: boolean // reflect about x-axis (negate y) — backend reflect_x
  reflectY?: boolean // reflect about y-axis (negate x) — backend reflect_y
  translateX?: number
  translateY?: number
}

export function transformPoint([x, y]: Pt, t: ShapeAffine): Pt {
  const [cx, cy] = t.centroid
  let dx = x - cx
  let dy = y - cy
  if (t.reflectY) dx = -dx
  if (t.reflectX) dy = -dy
  const deg = t.rotationDegrees || 0
  if (deg !== 0) {
    const th = (deg * Math.PI) / 180
    const c = Math.cos(th)
    const s = Math.sin(th)
    const rx = dx * c - dy * s
    const ry = dx * s + dy * c
    dx = rx
    dy = ry
  }
  return [cx + dx + (t.translateX || 0), cy + dy + (t.translateY || 0)]
}

export function transformPoints(pts: Pt[], t: ShapeAffine): Pt[] {
  return pts.map((p) => transformPoint(p, t))
}

export function meanOf(pts: Pt[]): Pt {
  let sx = 0
  let sy = 0
  for (const [x, y] of pts) {
    sx += x
    sy += y
  }
  const n = pts.length || 1
  return [sx / n, sy / n]
}

// Andrew's monotone-chain convex hull. Returns hull vertices (CCW), dropping
// collinear points. Fewer than 3 unique points → returns the input (degenerate,
// treated as "no region" by shapeOverlapsHull).
export function convexHull(points: Pt[]): Pt[] {
  const pts = points.slice().sort((a, b) => a[0] - b[0] || a[1] - b[1])
  const n = pts.length
  if (n <= 2) return pts
  const cross = (o: Pt, a: Pt, b: Pt) =>
    (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])
  const lower: Pt[] = []
  for (const p of pts) {
    while (lower.length >= 2 && cross(lower[lower.length - 2], lower[lower.length - 1], p) <= 0) lower.pop()
    lower.push(p)
  }
  const upper: Pt[] = []
  for (let i = n - 1; i >= 0; i--) {
    const p = pts[i]
    while (upper.length >= 2 && cross(upper[upper.length - 2], upper[upper.length - 1], p) <= 0) upper.pop()
    upper.push(p)
  }
  lower.pop()
  upper.pop()
  return lower.concat(upper)
}

// Ray-casting point-in-polygon test.
export function pointInPolygon(x: number, y: number, poly: Pt[]): boolean {
  let inside = false
  for (let i = 0, j = poly.length - 1; i < poly.length; j = i++) {
    const [xi, yi] = poly[i]
    const [xj, yj] = poly[j]
    const intersect = yi > y !== yj > y && x < ((xj - xi) * (y - yi)) / (yj - yi) + xi
    if (intersect) inside = !inside
  }
  return inside
}

// Decide whether a shape is "overlaid on" a set of moved cells, so it should
// travel with them. `hull` is the convex hull of the moved cells (pre-transform).
//
// ROIs are often drawn ENCIRCLING a cluster, so their vertices sit outside the
// cell cloud — a plain "are the vertices inside the cells" test would wrongly
// exclude them. We therefore accept a shape when its centroid falls inside the
// hull (catches encircling shapes) OR when at least half its points do (catches
// shapes that hug or cross the region boundary).
export function shapeOverlapsHull(points: Pt[], hull: Pt[]): boolean {
  if (points.length === 0 || hull.length < 3) return false
  const [cx, cy] = meanOf(points)
  if (pointInPolygon(cx, cy, hull)) return true
  let inCount = 0
  for (const [x, y] of points) if (pointInPolygon(x, y, hull)) inCount++
  return inCount * 2 >= points.length
}
