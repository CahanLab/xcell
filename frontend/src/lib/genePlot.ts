/**
 * Pure geometry for the gene map: data ↔ screen transform, lasso hit-testing,
 * nearest-point lookup, module colours, and module → gene-set grouping.
 * Kept out of the component so all of it is unit-testable.
 */

export interface PlotTransform {
  toX: (x: number) => number
  toY: (y: number) => number
  fromX: (sx: number) => number
  fromY: (sy: number) => number
  scale: number
}

/** Fit the coordinate extent into a w × h box with a margin, preserving aspect and centring. */
export function fitTransform(coords: [number, number][], w: number, h: number, margin: number): PlotTransform {
  let minX = Infinity, minY = Infinity, maxX = -Infinity, maxY = -Infinity
  for (const [x, y] of coords) {
    if (x < minX) minX = x
    if (x > maxX) maxX = x
    if (y < minY) minY = y
    if (y > maxY) maxY = y
  }
  if (!Number.isFinite(minX)) { minX = maxX = minY = maxY = 0 }
  const dataW = maxX - minX || 1
  const dataH = maxY - minY || 1
  const scale = Math.min((w - 2 * margin) / dataW, (h - 2 * margin) / dataH) || 1
  const offX = (w - dataW * scale) / 2
  const offY = (h - dataH * scale) / 2
  return {
    toX: (x) => offX + (x - minX) * scale,
    toY: (y) => offY + (y - minY) * scale,
    fromX: (sx) => (sx - offX) / scale + minX,
    fromY: (sy) => (sy - offY) / scale + minY,
    scale,
  }
}

function inPolygon(px: number, py: number, poly: [number, number][]): boolean {
  let inside = false
  for (let i = 0, j = poly.length - 1; i < poly.length; j = i++) {
    const xi = poly[i][0], yi = poly[i][1]
    const xj = poly[j][0], yj = poly[j][1]
    const intersect = (yi > py) !== (yj > py) && px < ((xj - xi) * (py - yi)) / ((yj - yi) || 1e-12) + xi
    if (intersect) inside = !inside
  }
  return inside
}

/** Indices of the points whose screen position falls inside a screen-space polygon. */
export function pointsInLasso(coords: [number, number][], polygon: [number, number][], xf: PlotTransform): number[] {
  if (polygon.length < 3) return []
  const out: number[] = []
  for (let i = 0; i < coords.length; i++) {
    if (inPolygon(xf.toX(coords[i][0]), xf.toY(coords[i][1]), polygon)) out.push(i)
  }
  return out
}

/** The point nearest a screen position, if within maxDist pixels. */
export function nearestPoint(coords: [number, number][], sx: number, sy: number, xf: PlotTransform, maxDist: number): number | null {
  let best = -1
  let bestD = maxDist * maxDist
  for (let i = 0; i < coords.length; i++) {
    const dx = xf.toX(coords[i][0]) - sx
    const dy = xf.toY(coords[i][1]) - sy
    const d = dx * dx + dy * dy
    if (d <= bestD) { bestD = d; best = i }
  }
  return best >= 0 ? best : null
}

const PALETTE = [
  '#4ecdc4', '#e9a23b', '#e94560', '#7fb3ff', '#b5e48c', '#f78fb3', '#c084fc', '#ffd166',
  '#06d6a0', '#ff9e64', '#8ecae6', '#ef476f', '#ffb4a2', '#a0c4ff', '#cdb4db', '#90be6d',
  '#f9c74f', '#577590', '#f3722c', '#43aa8b',
]

export function moduleColor(module: number): string {
  return PALETTE[((module % PALETTE.length) + PALETTE.length) % PALETTE.length]
}

/** One gene set per module, in module order (module 0 is the largest by construction). */
export function moduleGeneSets(prefix: string, genes: string[], modules: number[]): { name: string; genes: string[] }[] {
  const groups = new Map<number, string[]>()
  genes.forEach((g, i) => {
    const m = modules[i] ?? 0
    if (!groups.has(m)) groups.set(m, [])
    groups.get(m)!.push(g)
  })
  return Array.from(groups.keys()).sort((a, b) => a - b).map((m) => ({ name: `${prefix} module ${m + 1}`, genes: groups.get(m)! }))
}
