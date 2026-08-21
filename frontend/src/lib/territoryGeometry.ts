/** Pure helpers for territory drawing and rendering.
 *
 * The ring is the outer boundary a section's cuts divide. It is a padded
 * bounding box rather than a hull of the cells: assignment only ever concerns
 * cells, cells only exist inside the tissue, and a box is robust where an alpha
 * shape is fiddly. The padding matters — it gives a freehand cut somewhere to
 * land beyond the outermost cells.
 */

export function ringFromCells(
  coords: [number, number][],
  padFraction = 0.05,
): [number, number][] {
  let minX = Infinity, minY = Infinity, maxX = -Infinity, maxY = -Infinity
  for (const [x, y] of coords) {
    if (!Number.isFinite(x) || !Number.isFinite(y)) continue
    if (x < minX) minX = x
    if (x > maxX) maxX = x
    if (y < minY) minY = y
    if (y > maxY) maxY = y
  }
  if (!Number.isFinite(minX)) return []

  // A single cell (or a perfectly flat row) has no extent to pad; fall back to
  // a unit box so the ring always encloses area.
  const span = Math.max(maxX - minX, maxY - minY) || 1
  const pad = span * padFraction
  return [
    [minX - pad, minY - pad],
    [maxX + pad, minY - pad],
    [maxX + pad, maxY + pad],
    [minX - pad, maxY + pad],
  ]
}

/** Evenly spaced hues, mid-lightness, translucent so cells read through. */
export function faceColor(index: number, total: number): string {
  const hue = (index * 360) / Math.max(total, 1)
  const [r, g, b] = hslToRgb(hue / 360, 0.55, 0.55)
  return `rgba(${r},${g},${b},0.22)`
}

function hslToRgb(h: number, s: number, l: number): [number, number, number] {
  const k = (n: number) => (n + h * 12) % 12
  const a = s * Math.min(l, 1 - l)
  const f = (n: number) => l - a * Math.max(-1, Math.min(k(n) - 3, Math.min(9 - k(n), 1)))
  return [Math.round(f(0) * 255), Math.round(f(8) * 255), Math.round(f(4) * 255)]
}

export function polygonPath(screenPoints: string[]): string {
  if (screenPoints.length < 3) return ''
  return `M ${screenPoints.join(' L ')} Z`
}

/** The labels `.obs` actually holds, from what `/api/obs/<column>` returns.
 *
 * That endpoint sends a categorical column as integer *codes* plus a separate
 * `categories` array, while the backend matches territory sections against the
 * column's string labels. Keying sections by the code produces sections named
 * "1"/"2" that match no cell, and every cell then falls through to unassigned —
 * a total failure that looks exactly like a drawing mistake.
 */
export function sectionLabels(
  values: (string | number | null)[],
  categories: string[] | undefined,
): string[] {
  return values.map((v) => {
    if (v === null || v === undefined) return 'unassigned'
    if (typeof v === 'number' && categories && categories[v] !== undefined) {
      return categories[v]
    }
    return String(v)
  })
}

/** Which section a freshly drawn cut belongs to.
 *
 * The user draws where they are looking, not where the panel's active section
 * happens to be. On a multi-section slide the sections sit side by side, so the
 * cut's own position says which one it divides — filing it against the wrong
 * one produces a cut that divides nothing, which looks like the drawing failed.
 *
 * The midpoint is the test, not the endpoints: cuts are deliberately drawn to
 * overhang the tissue edge so their ends can be extended to the ring.
 */
export function sectionForCut(
  points: [number, number][],
  sections: Record<string, { ring: [number, number][] }>,
): string | null {
  if (points.length === 0) return null
  const [mx, my] = points[Math.floor(points.length / 2)]
  for (const [name, section] of Object.entries(sections)) {
    const ring = section.ring
    if (ring.length < 3) continue
    const xs = ring.map((p) => p[0])
    const ys = ring.map((p) => p[1])
    if (mx >= Math.min(...xs) && mx <= Math.max(...xs)
      && my >= Math.min(...ys) && my <= Math.max(...ys)) {
      return name
    }
  }
  return null
}
