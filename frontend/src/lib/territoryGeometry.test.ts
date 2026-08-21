import { describe, it, expect } from 'vitest'
import { ringFromCells, faceColor, polygonPath } from './territoryGeometry'

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
