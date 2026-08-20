import { describe, it, expect } from 'vitest'
import { isParamVisible } from './paramVisibility'

describe('isParamVisible', () => {
  it('shows a param with no condition', () => {
    expect(isParamVisible(undefined, {})).toBe(true)
  })

  it('matches an exact value', () => {
    expect(isParamVisible({ param: 'basis', value: 'expression' }, { basis: 'expression' })).toBe(true)
    expect(isParamVisible({ param: 'basis', value: 'expression' }, { basis: 'pca' })).toBe(false)
  })

  it('treats "*" as "any value at all"', () => {
    // Needed for options that only apply once another param is filled in —
    // the union/intersection columns exist only when HVGs are split by group.
    expect(isParamVisible({ param: 'split_by', value: '*' }, { split_by: 'sample' })).toBe(true)
    expect(isParamVisible({ param: 'split_by', value: '*' }, { split_by: '' })).toBe(false)
    expect(isParamVisible({ param: 'split_by', value: '*' }, { split_by: null })).toBe(false)
    expect(isParamVisible({ param: 'split_by', value: '*' }, {})).toBe(false)
  })

  it('does not treat the string "null" as a value', () => {
    // paramValues holds nulls; stringifying them must not read as set.
    expect(isParamVisible({ param: 'x', value: '*' }, { x: undefined })).toBe(false)
  })
})
