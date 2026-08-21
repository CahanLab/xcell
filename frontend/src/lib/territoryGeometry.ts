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
