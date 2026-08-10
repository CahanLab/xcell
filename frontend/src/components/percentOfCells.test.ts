import { describe, it, expect } from 'vitest'
import { percentOfCells } from './ScanpyModal'

describe('percentOfCells', () => {
  it('turns the default 0.1% into a count that scales with the dataset', () => {
    expect(percentOfCells(0.1, 2_683)).toBe(3)
    expect(percentOfCells(0.1, 9_773)).toBe(10)
    expect(percentOfCells(0.1, 200_000)).toBe(200)
  })

  it('never rounds a threshold away to nothing', () => {
    // 0.1% of 200 cells is 0.2. Filtering on 0 keeps every gene, which is not
    // what asking for a threshold means.
    expect(percentOfCells(0.1, 200)).toBe(1)
  })

  it('treats an empty box as no threshold rather than as zero percent', () => {
    expect(percentOfCells('', 10_000)).toBe(0)
    expect(percentOfCells(null, 10_000)).toBe(0)
    expect(percentOfCells(undefined, 10_000)).toBe(0)
  })

  it('refuses nonsense instead of producing NaN', () => {
    expect(percentOfCells('abc', 10_000)).toBe(0)
    expect(percentOfCells(-5, 10_000)).toBe(0)
  })

  it('accepts the string a number input actually gives you', () => {
    expect(percentOfCells('0.5', 10_000)).toBe(50)
  })

  it('handles a dataset that has not loaded yet', () => {
    expect(percentOfCells(0.1, 0)).toBe(1)
  })
})
