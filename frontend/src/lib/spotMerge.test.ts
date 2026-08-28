import { describe, it, expect } from 'vitest'
import { formatSizeHistogram, vetoVerdict } from './spotMerge'

describe('formatSizeHistogram', () => {
  it('reads back as size:count in ascending size', () => {
    expect(formatSizeHistogram({ '3': 44, '1': 12, '2': 31 })).toBe('1:12  2:31  3:44')
  })

  it('is empty when there is nothing to show', () => {
    expect(formatSizeHistogram({})).toBe('—')
  })
})

describe('vetoVerdict', () => {
  it('says the veto is idle when it blocked nothing', () => {
    expect(vetoVerdict(0, { p10: 0.4, p50: 0.6, p90: 0.9 }, 0.15))
      .toMatch(/blocked nothing/i)
  })

  it('says it is doing real work when the distribution straddles the cut', () => {
    expect(vetoVerdict(37, { p10: 0.11, p50: 0.34, p90: 0.72 }, 0.15))
      .toMatch(/straddle/i)
  })

  it('warns when the threshold sits outside the bulk but still fired', () => {
    expect(vetoVerdict(2, { p10: 0.5, p50: 0.7, p90: 0.95 }, 0.15))
      .toMatch(/rarely/i)
  })

  it('says nothing useful without a distribution', () => {
    expect(vetoVerdict(0, null, 0.15)).toBe('No candidate pairs were tested.')
  })
})
