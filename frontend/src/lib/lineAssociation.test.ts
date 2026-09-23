import { describe, it, expect } from 'vitest'
import { parseGeneCap, GENE_CAP_MAX } from './lineAssociation'

describe('parseGeneCap', () => {
  it('reads an empty field as "no cap"', () => {
    expect(parseGeneCap('')).toBeNull()
    expect(parseGeneCap('   ')).toBeNull()
  })

  it('keeps a positive cap', () => {
    expect(parseGeneCap('50')).toBe(50)
    expect(parseGeneCap(' 7 ')).toBe(7)
  })

  it('treats zero and negatives as "no cap" rather than "no genes"', () => {
    expect(parseGeneCap('0')).toBeNull()
    expect(parseGeneCap('-5')).toBeNull()
  })

  it('ignores a value that is not a number', () => {
    expect(parseGeneCap('abc')).toBeNull()
  })

  it('clamps an absurd cap instead of shipping it to the backend', () => {
    expect(parseGeneCap('999999')).toBe(GENE_CAP_MAX)
  })

  it('rounds a fractional entry down to a whole number of genes', () => {
    expect(parseGeneCap('12.7')).toBe(12)
  })
})
