import { describe, it, expect } from 'vitest'
import { fitTransform, pointsInLasso, nearestPoint, moduleColor, moduleGeneSets } from './genePlot'

const coords: [number, number][] = [[0, 0], [10, 0], [10, 10], [0, 10], [5, 5]]

describe('fitTransform', () => {
  it('maps the data extent into the canvas with a margin, preserving aspect', () => {
    const xf = fitTransform(coords, 200, 100, 10)
    expect(xf.toX(0)).toBeCloseTo(60)   // letterboxed: 80px square centred in 200×100
    expect(xf.toX(10)).toBeCloseTo(140)
    expect(xf.toY(0)).toBeCloseTo(10)
    expect(xf.toY(10)).toBeCloseTo(90)
    expect(xf.fromX(xf.toX(7.5))).toBeCloseTo(7.5)
    expect(xf.fromY(xf.toY(2.5))).toBeCloseTo(2.5)
  })
  it('survives a degenerate extent', () => {
    const xf = fitTransform([[3, 3], [3, 3]], 100, 100, 0)
    expect(Number.isFinite(xf.toX(3))).toBe(true)
  })
})

describe('pointsInLasso', () => {
  it('returns indices of points inside a screen-space polygon', () => {
    const xf = fitTransform(coords, 200, 200, 0)
    const poly: [number, number][] = [[xf.toX(-1), xf.toY(-1)], [xf.toX(6), xf.toY(-1)], [xf.toX(6), xf.toY(6)], [xf.toX(-1), xf.toY(6)]]
    expect(pointsInLasso(coords, poly, xf)).toEqual([0, 4])
    expect(pointsInLasso(coords, [[0, 0], [1, 0]], xf)).toEqual([])  // fewer than 3 vertices
  })
})

describe('nearestPoint', () => {
  it('finds the closest point within a pixel radius, else null', () => {
    const xf = fitTransform(coords, 200, 200, 0)
    expect(nearestPoint(coords, xf.toX(5) + 2, xf.toY(5) - 2, xf, 8)).toBe(4)
    expect(nearestPoint(coords, xf.toX(5) + 50, xf.toY(5), xf, 8)).toBeNull()
  })
})

describe('moduleColor / moduleGeneSets', () => {
  it('cycles a palette and groups genes by module in module order, largest first', () => {
    expect(moduleColor(0)).toMatch(/^#/)
    expect(moduleColor(0)).not.toBe(moduleColor(1))
    expect(moduleColor(0)).toBe(moduleColor(20))
    const sets = moduleGeneSets('collagens', ['a', 'b', 'c', 'd'], [1, 0, 0, 1])
    expect(sets).toEqual([
      { name: 'collagens module 1', genes: ['b', 'c'] },
      { name: 'collagens module 2', genes: ['a', 'd'] },
    ])
  })
})
