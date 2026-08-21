import { describe, it, expect } from 'vitest'
import { ringFromCells, faceColor, polygonPath, sectionLabels, sectionForCut } from './territoryGeometry'

describe('ringFromCells', () => {
  it('returns a padded bounding box in clockwise corner order', () => {
    const ring = ringFromCells([[0, 0], [10, 10]], 0)
    expect(ring).toEqual([[0, 0], [10, 0], [10, 10], [0, 10]])
  })

  it('pads by a fraction of the larger extent so cuts can reach past the cells', () => {
    const ring = ringFromCells([[0, 0], [10, 10]], 0.1)
    expect(ring[0]).toEqual([-1, -1])
    expect(ring[2]).toEqual([11, 11])
  })

  it('ignores non-finite coordinates rather than producing a NaN ring', () => {
    const ring = ringFromCells([[0, 0], [10, 10], [NaN, 5]], 0)
    expect(ring.flat().every((v) => Number.isFinite(v))).toBe(true)
  })

  it('gives a degenerate single-cell input a ring with area', () => {
    const ring = ringFromCells([[5, 5]], 0.1)
    expect(ring[0][0]).toBeLessThan(ring[1][0])
  })

  it('returns an empty ring when there is nothing to bound', () => {
    expect(ringFromCells([], 0.1)).toEqual([])
  })
})

describe('faceColor', () => {
  it('gives different faces different colors', () => {
    expect(faceColor(0, 4)).not.toBe(faceColor(1, 4))
  })

  it('is stable for the same position', () => {
    expect(faceColor(2, 5)).toBe(faceColor(2, 5))
  })

  it('returns an rgba string carrying the fill transparency', () => {
    expect(faceColor(0, 3)).toMatch(/^rgba\(\d+,\d+,\d+,0\.\d+\)$/)
  })
})

describe('polygonPath', () => {
  it('builds a closed SVG path', () => {
    expect(polygonPath(['1,2', '3,4', '5,6'])).toBe('M 1,2 L 3,4 L 5,6 Z')
  })

  it('returns an empty string when there is nothing to draw', () => {
    expect(polygonPath(['1,2'])).toBe('')
  })
})

describe('sectionLabels', () => {
  it('maps categorical codes to the labels .obs actually holds', () => {
    // /api/obs/<col> returns integer codes plus a categories array. Keying
    // territories by the code produces sections named "1"/"2" that match no
    // cell, and every cell silently falls through to unassigned.
    expect(sectionLabels([2, 2, 1, 3], ['unassigned', 'A', 'B', 'C']))
      .toEqual(['B', 'B', 'A', 'C'])
  })

  it('passes string values through untouched', () => {
    expect(sectionLabels(['A', 'B'], undefined)).toEqual(['A', 'B'])
  })

  it('stringifies numbers when there are no categories to map through', () => {
    expect(sectionLabels([1, 2], undefined)).toEqual(['1', '2'])
  })

  it('leaves a code with no matching category as its own string', () => {
    expect(sectionLabels([7], ['unassigned', 'A'])).toEqual(['7'])
  })

  it('treats a null value as unassigned rather than the string "null"', () => {
    expect(sectionLabels([null], ['unassigned', 'A'])).toEqual(['unassigned'])
  })
})

describe('sectionForCut', () => {
  const sections = {
    A: { ring: [[0, 0], [10, 0], [10, 10], [0, 10]] as [number, number][] },
    B: { ring: [[20, 0], [30, 0], [30, 10], [20, 10]] as [number, number][] },
  }

  it('routes a cut to the section it was actually drawn across', () => {
    // You draw where you are looking, not where the active section is; a cut
    // filed against the wrong section silently divides nothing.
    expect(sectionForCut([[21, 5], [29, 5]], sections)).toBe('B')
  })

  it('uses the midpoint, so a cut overhanging one edge still lands right', () => {
    expect(sectionForCut([[-5, 5], [5, 5], [12, 5]], sections)).toBe('A')
  })

  it('returns null when the cut is nowhere near any section', () => {
    expect(sectionForCut([[100, 100], [110, 110]], sections)).toBeNull()
  })

  it('returns null for a cut with no points', () => {
    expect(sectionForCut([], sections)).toBeNull()
  })
})
