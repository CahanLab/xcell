/**
 * Collision-free placement of category labels on a plot.
 *
 * Labels are anchored at cluster centroids, which pile up wherever clusters
 * are small or close. The layout is greedy and deterministic: heavier labels
 * (bigger clusters) are placed first, at their anchor when it is free, else at
 * the nearest free spot on a ring of candidate offsets; a label that finds no
 * free spot within `maxShift` is hidden rather than drawn on top of another.
 * A shifted label remembers its anchor so the renderer can draw a leader line.
 *
 * `fitFontSize` finds the largest font at which every label fits, for small
 * panes where shrinking the text is better than dropping labels.
 *
 * Text is measured with a fixed average glyph width, so the same layout is
 * reproducible in an SVG overlay and on a canvas without a measuring context.
 */

export interface LabelInput {
  id: string
  text: string
  /** anchor, in the same pixel space as `bounds` */
  x: number
  y: number
  /** placement priority; larger wins the contested spot (cluster size) */
  weight?: number
}

export interface PlacedLabel {
  id: string
  text: string
  x: number
  y: number
  /** anchor the label was asked to sit on */
  ax: number
  ay: number
  hidden: boolean
}

export interface LayoutOptions {
  fontPx: number
  bounds: { w: number; h: number }
  /** furthest a label may move from its anchor, px (default 6 × font size) */
  maxShift?: number
  /** minimum gap between label boxes, px */
  padding?: number
}

interface Box { x: number; y: number; w: number; h: number }

/** Average glyph width of a bold system font relative to its size, plus a halo. */
const GLYPH_W = 0.62
const LINE_H = 1.25
const HALO = 3

export function measureLabel(text: string, fontPx: number): { w: number; h: number } {
  return { w: text.length * fontPx * GLYPH_W + 2 * HALO, h: fontPx * LINE_H + 2 * HALO }
}

/** Boxes are centred on (x, y). */
export function boxesOverlap(a: Box, b: Box, padding: number): boolean {
  return Math.abs(a.x - b.x) * 2 < a.w + b.w + 2 * padding && Math.abs(a.y - b.y) * 2 < a.h + b.h + 2 * padding
}

function clampInto(x: number, y: number, box: { w: number; h: number }, bounds: { w: number; h: number }): [number, number] {
  const hw = box.w / 2, hh = box.h / 2
  return [
    Math.min(Math.max(x, hw), Math.max(hw, bounds.w - hw)),
    Math.min(Math.max(y, hh), Math.max(hh, bounds.h - hh)),
  ]
}

/** Candidate offsets: the anchor, then 8 directions at growing radii. */
function* candidates(maxShift: number, step: number): Generator<[number, number]> {
  yield [0, 0]
  const dirs: [number, number][] = [[0, -1], [0, 1], [1, 0], [-1, 0], [1, -1], [-1, -1], [1, 1], [-1, 1]]
  for (let r = step; r <= maxShift; r += step) {
    for (const [dx, dy] of dirs) yield [dx * r, dy * r]
  }
}

export function layoutLabels(labels: LabelInput[], opts: LayoutOptions): PlacedLabel[] {
  const { fontPx, bounds } = opts
  const maxShift = opts.maxShift ?? fontPx * 6
  const padding = opts.padding ?? 2
  const step = Math.max(2, Math.round(fontPx / 2))
  const order = labels
    .map((l, i) => ({ l, i }))
    .sort((a, b) => (b.l.weight ?? 0) - (a.l.weight ?? 0) || a.l.id.localeCompare(b.l.id) || a.i - b.i)
  const placed: Box[] = []
  const out: PlacedLabel[] = new Array(labels.length)
  for (const { l, i } of order) {
    const size = measureLabel(l.text, fontPx)
    let found: [number, number] | null = null
    for (const [dx, dy] of candidates(maxShift, step)) {
      const [cx, cy] = clampInto(l.x + dx, l.y + dy, size, bounds)
      const box = { x: cx, y: cy, ...size }
      if (!placed.some((p) => boxesOverlap(p, box, padding))) { found = [cx, cy]; break }
    }
    if (found) {
      placed.push({ x: found[0], y: found[1], ...size })
      out[i] = { id: l.id, text: l.text, x: found[0], y: found[1], ax: l.x, ay: l.y, hidden: false }
    } else {
      out[i] = { id: l.id, text: l.text, x: l.x, y: l.y, ax: l.x, ay: l.y, hidden: true }
    }
  }
  return out
}

/** Largest integer font size in [min, max] at which no label is hidden, else `min`. */
export function fitFontSize(labels: LabelInput[], opts: { bounds: { w: number; h: number }; max: number; min: number; maxShift?: number; padding?: number }): number {
  const max = Math.max(opts.min, Math.floor(opts.max))
  for (let f = max; f >= opts.min; f--) {
    const placed = layoutLabels(labels, { fontPx: f, bounds: opts.bounds, maxShift: opts.maxShift ?? f * 6, padding: opts.padding })
    if (placed.every((p) => !p.hidden)) return f
  }
  return opts.min
}
