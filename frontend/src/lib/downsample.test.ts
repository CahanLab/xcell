import { describe, it, expect } from 'vitest'
import {
  resolveTarget,
  sampleWithoutReplacement,
  seededRng,
  poolIndices,
  availablePools,
  parseSeed,
} from './downsample'

describe('resolveTarget', () => {
  it('takes a count as-is', () => {
    expect(resolveTarget('count', 250, 1000)).toEqual({ n: 250 })
    expect(resolveTarget('count', '250', 1000)).toEqual({ n: 250 })
  })

  it('turns a percent of the pool into a count', () => {
    expect(resolveTarget('percent', 10, 1000)).toEqual({ n: 100 })
    expect(resolveTarget('percent', '12.5', 1000)).toEqual({ n: 125 })
  })

  it('never rounds a percent down to nothing', () => {
    expect(resolveTarget('percent', 0.01, 1000)).toEqual({ n: 1 })
  })

  it('refuses a target that would not thin the pool', () => {
    expect(resolveTarget('count', 1000, 1000).n).toBeNull()
    expect(resolveTarget('count', 5000, 1000).n).toBeNull()
    expect(resolveTarget('percent', 100, 1000).n).toBeNull()
    expect(resolveTarget('percent', 150, 1000).n).toBeNull()
  })

  it('refuses blanks, nonsense, zero and negatives with a reason', () => {
    for (const raw of ['', null, undefined, 'abc', 0, -3, '0.4']) {
      const r = resolveTarget('count', raw, 1000)
      expect(r.n).toBeNull()
      expect(r.reason).toBeTruthy()
    }
    expect(resolveTarget('percent', 0, 1000).n).toBeNull()
    expect(resolveTarget('percent', -1, 1000).n).toBeNull()
  })

  it('truncates a fractional count', () => {
    expect(resolveTarget('count', 12.7, 1000)).toEqual({ n: 12 })
  })

  it('handles an empty pool', () => {
    expect(resolveTarget('count', 5, 0).n).toBeNull()
    expect(resolveTarget('percent', 50, 0).n).toBeNull()
  })
})

describe('sampleWithoutReplacement', () => {
  const pool = Array.from({ length: 200 }, (_, i) => i * 3 + 7)

  it('returns exactly n distinct members of the pool, in ascending order', () => {
    const out = sampleWithoutReplacement(pool, 50, seededRng(1))
    expect(out).toHaveLength(50)
    expect(new Set(out).size).toBe(50)
    const members = new Set(pool)
    for (const v of out) expect(members.has(v)).toBe(true)
    expect(out).toEqual([...out].sort((a, b) => a - b))
  })

  it('is reproducible for a seed and different across seeds', () => {
    const a = sampleWithoutReplacement(pool, 50, seededRng(42))
    const b = sampleWithoutReplacement(pool, 50, seededRng(42))
    const c = sampleWithoutReplacement(pool, 50, seededRng(43))
    expect(a).toEqual(b)
    expect(a).not.toEqual(c)
  })

  it('does not mutate the pool', () => {
    const copy = [...pool]
    sampleWithoutReplacement(pool, 50, seededRng(1))
    expect(pool).toEqual(copy)
  })

  it('samples uniformly — every member is picked about n/N of the time', () => {
    const small = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
    const counts = new Array(10).fill(0)
    const rng = seededRng(7)
    const runs = 4000
    for (let r = 0; r < runs; r++) {
      for (const v of sampleWithoutReplacement(small, 3, rng)) counts[v]++
    }
    // expected 0.3 * 4000 = 1200 each; a biased partial shuffle would show
    // a clear gradient across positions, far outside ±10%.
    for (const c of counts) {
      expect(c).toBeGreaterThan(1080)
      expect(c).toBeLessThan(1320)
    }
  })

  it('returns the whole pool when n equals its size, and nothing for n <= 0', () => {
    expect(sampleWithoutReplacement([5, 3, 9], 3, seededRng(1))).toEqual([3, 5, 9])
    expect(sampleWithoutReplacement([5, 3, 9], 0, seededRng(1))).toEqual([])
  })
})

describe('poolIndices', () => {
  const mask = [true, false, true, true, false]

  it('selection is the selection', () => {
    expect(poolIndices('selection', [4, 1], mask, 5)).toEqual([4, 1])
  })

  it('active is every unmasked cell', () => {
    expect(poolIndices('active', [4, 1], mask, 5)).toEqual([0, 2, 3])
  })

  it('active with no mask is every cell', () => {
    expect(poolIndices('active', [], null, 4)).toEqual([0, 1, 2, 3])
  })

  it('all ignores both', () => {
    expect(poolIndices('all', [4, 1], mask, 5)).toEqual([0, 1, 2, 3, 4])
  })
})

describe('availablePools', () => {
  it('offers only pools that exist, most specific first', () => {
    expect(availablePools(0, false)).toEqual(['all'])
    expect(availablePools(0, true)).toEqual(['active', 'all'])
    expect(availablePools(12, false)).toEqual(['selection', 'all'])
    expect(availablePools(12, true)).toEqual(['selection', 'active', 'all'])
  })
})

describe('parseSeed', () => {
  it('blank means draw one', () => {
    expect(parseSeed('')).toBeNull()
    expect(parseSeed('  ')).toBeNull()
  })
  it('accepts an integer and rejects the rest', () => {
    expect(parseSeed('123')).toBe(123)
    expect(parseSeed('1.5')).toBeNull()
    expect(parseSeed('abc')).toBeNull()
  })
})
