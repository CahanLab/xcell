import { describe, it, expect } from 'vitest'
import { boundsOf, fitView, ZOOM_RANGE } from './viewFit'

describe('boundsOf', () => {
  it('pads the extent by 5% each side', () => {
    expect(boundsOf([[0, 0], [10, 20]])).toEqual({ minX: -0.5, maxX: 10.5, minY: -1, maxY: 21 })
  })

  it('leaves out cells with no coordinates instead of counting them as 0', () => {
    // A subset's embedding is null outside it; `null < x` reads null as 0.
    const b = boundsOf([[5, 6], [null, null], [7, 8], null], 0)
    expect(b).toEqual({ minX: 5, maxX: 7, minY: 6, maxY: 8 })
  })

  it('falls back to the unit square when nothing has coordinates', () => {
    expect(boundsOf([])).toEqual({ minX: 0, maxX: 1, minY: 0, maxY: 1 })
    expect(boundsOf([[null, null]])).toEqual({ minX: 0, maxX: 1, minY: 0, maxY: 1 })
  })
})

const box = (minX: number, maxX: number, minY: number, maxY: number) => ({ minX, maxX, minY, maxY })

describe('fitView', () => {
  it('centres on the bounds and fits the tighter axis', () => {
    const v = fitView(box(0, 20, 0, 10), 800, 600)
    expect(v.target).toEqual([10, 5, 0])
    expect(v.zoom).toBeCloseTo(Math.log2(40) - 1)
  })

  it('lets a diffusion map, a fraction of a unit wide, zoom in past its fit', () => {
    // DCs span ~0.2: the fit is ~11, above the old fixed ceiling of 10, so the
    // first wheel event snapped the view out to 10 and it could not come back.
    const v = fitView(box(-0.11, 0.1, -0.07, 0.07), 800, 600)
    expect(v.zoom).toBeGreaterThan(10)
    expect(v.maxZoom).toBe(v.zoom + ZOOM_RANGE)
    expect(v.minZoom).toBe(v.zoom - ZOOM_RANGE)
  })

  it('keeps zooming out possible on spatial coordinates thousands of units wide', () => {
    const v = fitView(box(0, 20000, 0, 15000), 800, 600)
    expect(v.zoom).toBeLessThan(-5)
    expect(v.minZoom).toBeLessThan(v.zoom)
    expect(v.maxZoom).toBeGreaterThan(v.zoom)
  })

  it('fits on the other axis when one has no extent', () => {
    // A stationary diffusion component is constant across a connected graph.
    const v = fitView(box(0.03, 0.03, -0.1, 0.1), 800, 600)
    expect(v.target).toEqual([0.03, 0, 0])
    expect(v.zoom).toBeCloseTo(Math.log2(600 / 0.2) - 1)
  })

  it('gives a finite view for a single point or no coordinates at all', () => {
    for (const b of [box(1, 1, 2, 2), box(Infinity, -Infinity, Infinity, -Infinity)]) {
      const v = fitView(b, 800, 600)
      expect(Number.isFinite(v.zoom)).toBe(true)
      expect(v.target.every(Number.isFinite)).toBe(true)
    }
  })
})
