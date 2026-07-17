import { describe, it, expect } from 'vitest'
import { pointsInLassoScreen } from './lasso3d'

// Stub viewport: identity projection using x,y (ignores z) so expectations are obvious.
const identityViewport = { project: (c: number[]) => [c[0], c[1]] }

describe('pointsInLassoScreen', () => {
  it('selects points whose projection is inside the polygon', () => {
    const coords: [number, number][] = [[1, 1], [5, 5], [9, 9]]
    const z = [0, 0, 0]
    const square: [number, number][] = [[0, 0], [4, 0], [4, 4], [0, 4]]
    expect(pointsInLassoScreen(coords, z, square, identityViewport)).toEqual([0])
  })

  it('uses the projected coordinate (z affects projection via viewport)', () => {
    // viewport that shifts x by z: a point at x=1,z=10 projects to x=11 (outside)
    const vp = { project: (c: number[]) => [c[0] + c[2], c[1]] }
    const coords: [number, number][] = [[1, 1], [1, 1]]
    const z = [0, 10]
    const square: [number, number][] = [[0, 0], [4, 0], [4, 4], [0, 4]]
    expect(pointsInLassoScreen(coords, z, square, vp)).toEqual([0])
  })

  it('returns empty when polygon has < 3 vertices', () => {
    const coords: [number, number][] = [[1, 1]]
    expect(pointsInLassoScreen(coords, [0], [[0, 0], [1, 1]], identityViewport)).toEqual([])
  })
})
