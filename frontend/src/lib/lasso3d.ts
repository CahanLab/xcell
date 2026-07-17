export interface ProjectingViewport {
  project(coord: number[]): number[]
}

/** Ray-casting point-in-polygon on screen-space coordinates. */
function inPolygon(px: number, py: number, poly: [number, number][]): boolean {
  let inside = false
  for (let i = 0, j = poly.length - 1; i < poly.length; j = i++) {
    const xi = poly[i][0], yi = poly[i][1]
    const xj = poly[j][0], yj = poly[j][1]
    const intersect = (yi > py) !== (yj > py)
      && px < ((xj - xi) * (py - yi)) / ((yj - yi) || 1e-12) + xi
    if (intersect) inside = !inside
  }
  return inside
}

/**
 * Indices of cells whose 3D position projects to a screen point inside the
 * lasso polygon. Depth-agnostic by design: everything under the outline is
 * captured (rotate + re-lasso to refine).
 */
export function pointsInLassoScreen(
  coordinates: [number, number][],
  z: number[],
  polygonScreen: [number, number][],
  viewport: ProjectingViewport,
): number[] {
  if (polygonScreen.length < 3) return []
  const out: number[] = []
  for (let i = 0; i < coordinates.length; i++) {
    const c = coordinates[i]
    const [sx, sy] = viewport.project([c[0], c[1], z[i] ?? 0])
    if (inPolygon(sx, sy, polygonScreen)) out.push(i)
  }
  return out
}
