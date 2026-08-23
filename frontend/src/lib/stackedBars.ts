// The arithmetic behind the stacked barplot, kept out of the drawing code.
//
// A stacked bar is a claim about composition — "70% of cluster 0 is proximal" —
// so the ordering and the segment maths are the part worth pinning down.

/** The backend's answer from GET /api/obs/crosstab. */
export interface Crosstab {
  a: string
  b: string
  a_categories: string[]
  b_categories: string[]
  a_colors: (string | null)[] | null
  b_colors: (string | null)[] | null
  /** Rows follow a_categories, columns follow b_categories. */
  counts: number[][]
  n_cells: number
}

export type BarOrder = 'category' | 'alphabetical' | 'total' | 'share'

export interface OrderOptions {
  by: BarOrder
  /** For `share`: the b-category whose proportion sorts the bars. */
  shareOf?: string | null
  /** Bars with fewer cells than this are left out entirely. */
  minCells?: number
}

// Numbered clusters are the common case here, and lexicographic order puts
// '10' before '9' in every one of them.
const collator = new Intl.Collator(undefined, { numeric: true, sensitivity: 'base' })

/** Which bars to draw, in which order, as indices into `a_categories`. */
export function orderBars(xt: Crosstab, opts: OrderOptions): number[] {
  const total = (i: number) => xt.counts[i].reduce((s, n) => s + n, 0)
  const minCells = opts.minCells ?? 0
  const kept = xt.a_categories
    .map((_, i) => i)
    .filter((i) => total(i) >= minCells)

  if (opts.by === 'alphabetical') {
    return [...kept].sort((i, j) => collator.compare(xt.a_categories[i], xt.a_categories[j]))
  }
  if (opts.by === 'total') {
    return [...kept].sort((i, j) => total(j) - total(i))
  }
  if (opts.by === 'share') {
    const col = xt.b_categories.indexOf(opts.shareOf ?? '')
    // The category was renamed or removed since it was chosen; ordering by a
    // column that isn't there would silently reorder by nothing.
    if (col < 0) return kept
    const share = (i: number) => {
      const t = total(i)
      return t === 0 ? 0 : xt.counts[i][col] / t
    }
    return [...kept].sort((i, j) => share(j) - share(i))
  }
  return kept
}

export interface Segment {
  /** Index into `b_categories`. */
  index: number
  count: number
  /** Height as a fraction of the plot's full bar height. */
  fraction: number
  /** Distance from the bar's base to this segment's near edge. */
  start: number
}

export interface StackOptions {
  /** True: every bar fills the height and shows composition. False: bar height
   *  is the cell count, measured against `scaleMax`. */
  normalize: boolean
  scaleMax?: number
}

/** Split one bar's counts into stacked segments, base first. */
export function stackBar(counts: number[], opts: StackOptions): Segment[] {
  const total = counts.reduce((s, n) => s + n, 0)
  if (total === 0) return []
  const denom = opts.normalize ? total : Math.max(1, opts.scaleMax ?? total)

  const segments: Segment[] = []
  let start = 0
  counts.forEach((count, index) => {
    // A zero-count category would be an invisible rectangle that still catches
    // the pointer and still claims a place in the stack.
    if (count === 0) return
    const fraction = count / denom
    segments.push({ index, count, fraction, start })
    start += fraction
  })
  return segments
}

/** Trim a label to the space a 45°-rotated line of text actually has.
 *
 * The heatmap used to truncate by the width of the *column* a label sat under,
 * which left two characters per label once a plot had thirty clusters. A
 * rotated label runs along the diagonal of the space reserved for it, which is
 * far more room than the column below it.
 */
export function fitRotatedLabel(text: string, maxPx: number, charPx: number): string {
  const maxChars = Math.floor(maxPx / charPx)
  if (maxChars >= text.length) return text
  // Below two characters there is nothing left to say, and a bare ellipsis
  // reads as a label rather than as an omission.
  if (maxChars < 2) return ''
  return text.slice(0, maxChars - 1) + '…'
}


/** Whether a 45°-rotated label has room beside its neighbours.
 *
 * Rotated labels sit on parallel diagonals, one per group, so what keeps two
 * of them apart is the perpendicular distance between those diagonals — the
 * group's width over root two, not the width itself. Below one line height
 * they overlap, and a run of small groups turns into a single unreadable
 * smear. Better to leave those unlabelled: the separator lines still show
 * where each group is.
 */
export function rotatedLabelFits(groupWidthPx: number, lineHeightPx: number): boolean {
  return groupWidthPx / Math.SQRT2 >= lineHeightPx
}
