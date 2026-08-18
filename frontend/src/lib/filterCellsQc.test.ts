import { describe, it, expect } from 'vitest'
import {
  toChartSpace,
  fromChartSpace,
  formatCount,
  passingCount,
} from './filterCellsQc'

describe('log-axis mapping', () => {
  it('is the identity when log is off', () => {
    expect(toChartSpace(250, false)).toBe(250)
    expect(fromChartSpace(250, false)).toBe(250)
  })

  it('maps through log10(1+x) and back', () => {
    expect(toChartSpace(0, true)).toBe(0)
    expect(toChartSpace(999, true)).toBeCloseTo(3, 5)
    expect(fromChartSpace(3, true)).toBe(999)
  })

  it('round-trips integers exactly', () => {
    for (const v of [0, 1, 7, 250, 12345]) {
      expect(fromChartSpace(toChartSpace(v, true), true)).toBe(v)
    }
  })
})

describe('formatCount', () => {
  it('keeps small numbers plain and compacts thousands', () => {
    expect(formatCount(0)).toBe('0')
    expect(formatCount(950)).toBe('950')
    expect(formatCount(1200)).toBe('1.2k')
    expect(formatCount(48000)).toBe('48k')
  })
})

describe('passingCount', () => {
  const counts = [3, 10, 1, 30, 6]
  const genes = [3, 2, 1, 3, 2]

  it('applies all bounds inclusively, like the backend mask', () => {
    expect(passingCount(counts, genes, { minCounts: 3, maxCounts: 20, minGenes: 2 })).toBe(3)
    expect(passingCount(counts, genes, { minCounts: 3, maxCounts: 30 })).toBe(4)
  })

  it('counts everything when no bounds are set', () => {
    expect(passingCount(counts, genes, {})).toBe(5)
  })

  it('a null count fails any counts bound, as it would server-side', () => {
    expect(passingCount([3, null, 6], [1, 1, 1], { minCounts: 1 })).toBe(2)
    expect(passingCount([3, null, 6], [1, 1, 1], { maxCounts: 100 })).toBe(2)
    expect(passingCount([3, null, 6], [1, 1, 1], {})).toBe(3)
  })
})
