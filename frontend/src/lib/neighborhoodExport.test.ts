import { describe, it, expect } from 'vitest'
import { toLongCsv, toMatrixCsv } from './neighborhoodExport'

const result = {
  categories: ['B cell', 'T cell'],
  composition: [[0.7, 0.3], [0.25, 0.75]],
  expected: [[0.5, 0.5], [0.5, 0.5]],
  counts: [[70, 30], [25, 75]],
  zscores: [[4.2, -4.2], [-3.1, 3.1]],
  log2fc: [[0.49, -0.74], [-1.0, 0.58]],
  pvals: [[0.002, 0.002], [0.01, 0.01]],
  qvals: [[0.004, 0.004], [0.0009, 0.0009]],
}

describe('toLongCsv', () => {
  it('emits a header plus one row per ordered pair', () => {
    const lines = toLongCsv(result, [0, 1]).trim().split('\n')
    expect(lines[0]).toBe('row_type,col_type,neighbor_fraction,expected_fraction,n_pairs,zscore,log2fc,pval,qval')
    expect(lines).toHaveLength(1 + 4)
  })

  it('carries every statistic for a pair on one row', () => {
    const row = toLongCsv(result, [0, 1]).trim().split('\n')[1]
    expect(row).toBe('B cell,B cell,0.7,0.5,70,4.2,0.49,0.002,0.004')
  })

  it('writes rows in the displayed order, so the file matches the figure', () => {
    const lines = toLongCsv(result, [1, 0]).trim().split('\n')
    expect(lines[1].startsWith('T cell,T cell')).toBe(true)
  })

  it('quotes a label containing a comma rather than breaking the column count', () => {
    const commas = { ...result, categories: ['Fibro, distal', 'T cell'] }
    const row = toLongCsv(commas, [0, 1]).trim().split('\n')[1]
    expect(row.startsWith('"Fibro, distal","Fibro, distal",')).toBe(true)
  })

  it('does not round a small q-value away to zero', () => {
    const lines = toLongCsv(result, [0, 1]).trim().split('\n')
    expect(lines[3]).toContain('0.0009')
  })
})

describe('toMatrixCsv', () => {
  it('writes a corner cell, column labels, and one labelled row per type', () => {
    const lines = toMatrixCsv(result.zscores, result.categories, [0, 1]).trim().split('\n')
    expect(lines[0]).toBe(',B cell,T cell')
    expect(lines[1]).toBe('B cell,4.2,-4.2')
    expect(lines).toHaveLength(3)
  })

  it('permutes rows and columns together so the diagonal stays the diagonal', () => {
    const lines = toMatrixCsv(result.zscores, result.categories, [1, 0]).trim().split('\n')
    expect(lines[0]).toBe(',T cell,B cell')
    expect(lines[1]).toBe('T cell,3.1,-3.1')
  })
})
