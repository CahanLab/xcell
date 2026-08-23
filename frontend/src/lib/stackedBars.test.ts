import { describe, it, expect } from 'vitest'
import { orderBars, stackBar, fitRotatedLabel, rotatedLabelFits, type Crosstab } from './stackedBars'

// Three clusters split two ways. Cluster '1' is the small one; cluster '2' is
// entirely distal.
const XT: Crosstab = {
  a: 'leiden',
  b: 'territory',
  a_categories: ['0', '1', '2'],
  b_categories: ['dist', 'prox'],
  a_colors: null,
  b_colors: null,
  counts: [[30, 70], [8, 2], [40, 0]],
  n_cells: 150,
}

describe('orderBars', () => {
  it('keeps the dataset\'s own category order by default', () => {
    expect(orderBars(XT, { by: 'category' })).toEqual([0, 1, 2])
  })

  it('sorts by name when asked', () => {
    const named: Crosstab = { ...XT, a_categories: ['beta', 'alpha', 'gamma'] }
    expect(orderBars(named, { by: 'alphabetical' })).toEqual([1, 0, 2])
  })

  it('sorts alphabetically the way a person reads cluster numbers', () => {
    // '10' after '9', not before it — lexicographic order gets every numbered
    // cluster list wrong.
    const many: Crosstab = {
      ...XT,
      a_categories: ['10', '9', '2'],
      counts: [[1, 1], [1, 1], [1, 1]],
    }
    expect(orderBars(many, { by: 'alphabetical' }).map((i) => many.a_categories[i]))
      .toEqual(['2', '9', '10'])
  })

  it('puts the biggest bars first when sorting by size', () => {
    // totals: 100, 10, 40
    expect(orderBars(XT, { by: 'total' })).toEqual([0, 2, 1])
  })

  // The one that makes the plot answer a question: "which clusters are
  // proximal?" is a sort, not a squint.
  it('sorts by how much of each bar is one chosen category', () => {
    // prox share: 0.70, 0.20, 0.00
    expect(orderBars(XT, { by: 'share', shareOf: 'prox' })).toEqual([0, 1, 2])
  })

  it('falls back to category order when the chosen category is gone', () => {
    expect(orderBars(XT, { by: 'share', shareOf: 'nonexistent' })).toEqual([0, 1, 2])
  })

  it('drops bars below the minimum cell count', () => {
    expect(orderBars(XT, { by: 'category', minCells: 20 })).toEqual([0, 2])
  })

  it('keeps every bar when no minimum is set', () => {
    expect(orderBars(XT, { by: 'category', minCells: 0 })).toHaveLength(3)
  })
})

describe('stackBar', () => {
  it('splits a bar into one segment per category, in order', () => {
    const segs = stackBar(XT.counts[0], { normalize: true })
    expect(segs.map((s) => s.index)).toEqual([0, 1])
    expect(segs[0].fraction).toBeCloseTo(0.3)
    expect(segs[1].fraction).toBeCloseTo(0.7)
  })

  it('stacks them end to end, covering the whole bar', () => {
    const segs = stackBar(XT.counts[0], { normalize: true })
    expect(segs[0].start).toBeCloseTo(0)
    expect(segs[0].start + segs[0].fraction).toBeCloseTo(segs[1].start)
    const last = segs[segs.length - 1]
    expect(last.start + last.fraction).toBeCloseTo(1)
  })

  it('leaves out categories with no cells, rather than drawing nothing', () => {
    // A zero-height rectangle is still a hover target and still a legend entry.
    const segs = stackBar(XT.counts[2], { normalize: true })
    expect(segs.map((s) => s.index)).toEqual([0])
  })

  it('carries the raw count alongside the fraction', () => {
    const segs = stackBar(XT.counts[0], { normalize: true })
    expect(segs[0].count).toBe(30)
    expect(segs[1].count).toBe(70)
  })

  it('measures against the tallest bar when showing counts, not proportions', () => {
    const segs = stackBar(XT.counts[1], { normalize: false, scaleMax: 100 })
    // 10 cells against a 100-cell tallest bar fills a tenth of the height.
    expect(segs[0].start + segs[0].fraction + segs[1].fraction).toBeCloseTo(0.1)
  })

  it('has nothing to draw for a bar with no cells', () => {
    expect(stackBar([0, 0], { normalize: true })).toEqual([])
  })
})

describe('fitRotatedLabel', () => {
  // The heatmap truncated by *column* width, so 32 clusters across a plot left
  // two characters per label. A label rotated 45° runs along the diagonal and
  // has far more room than the column it sits under.
  it('keeps a label that fits', () => {
    expect(fitRotatedLabel('prox', 90, 5.5)).toBe('prox')
  })

  it('cuts a long one down with an ellipsis', () => {
    const out = fitRotatedLabel('territory_anterior_posterior', 55, 5.5)
    expect(out.endsWith('…')).toBe(true)
    expect(out.length).toBeLessThan('territory_anterior_posterior'.length)
  })

  it('gives up rather than returning a lone ellipsis', () => {
    expect(fitRotatedLabel('abcdef', 4, 5.5)).toBe('')
  })

  it('never returns more than it was given', () => {
    expect(fitRotatedLabel('ab', 1000, 5.5)).toBe('ab')
  })
})


describe('rotatedLabelFits', () => {
  // Labels rotated 45° run on parallel diagonals, one per group. What keeps two
  // neighbours apart is the *perpendicular* gap between those diagonals, which
  // is the group's width over root two — not the width itself.
  it('labels a group wide enough to hold its own line of text', () => {
    expect(rotatedLabelFits(30, 11)).toBe(true)
  })

  it('skips a group whose neighbour would be written across it', () => {
    // 6px of width is 4px of perpendicular room for an 11px line.
    expect(rotatedLabelFits(6, 11)).toBe(false)
  })

  it('draws the line exactly where the text stops overlapping', () => {
    const need = 11 * Math.SQRT2
    expect(rotatedLabelFits(need + 0.1, 11)).toBe(true)
    expect(rotatedLabelFits(need - 0.1, 11)).toBe(false)
  })
})
