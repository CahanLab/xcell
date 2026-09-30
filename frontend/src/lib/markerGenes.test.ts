import { describe, it, expect } from 'vitest'
import { markerGenesParams, maskedCategoryCounts, runnableGroupCount } from './markerGenes'

// The marker-genes request as the modal builds it. The gene subset travels
// only when one is chosen: an empty choice means "all genes", and a stale
// 'highly_variable' on a dataset without that column is a 400.

const base = {
  obsColumn: 'leiden',
  selectedGroups: new Set(['0', '1', '2']),
  nCategories: 3,
  topN: 25,
  geneSubset: '',
  minInGroupFraction: '',
  maxOutGroupFraction: '',
  minFoldChange: '',
}

describe('markerGenesParams', () => {
  it('sends only the column and top_n when nothing else is set', () => {
    expect(markerGenesParams(base)).toEqual({ obs_column: 'leiden', top_n: 25 })
  })

  it('sends the gene subset only when one is chosen', () => {
    expect(markerGenesParams({ ...base, geneSubset: 'highly_variable' }).gene_subset).toBe('highly_variable')
    expect('gene_subset' in markerGenesParams(base)).toBe(false)
  })

  it('names the groups only when not every category is checked', () => {
    expect('groups' in markerGenesParams(base)).toBe(false)
    expect(markerGenesParams({ ...base, selectedGroups: new Set(['0', '2']) }).groups).toEqual(['0', '2'])
  })

  it('scopes the run to the cell mask when one is active', () => {
    expect(markerGenesParams({ ...base, activeCellIndices: [4, 5, 9] }).active_cell_indices).toEqual([4, 5, 9])
  })

  it('sends no cell list without a mask', () => {
    expect('active_cell_indices' in markerGenesParams(base)).toBe(false)
    expect('active_cell_indices' in markerGenesParams({ ...base, activeCellIndices: null })).toBe(false)
  })

  it('parses the optional thresholds and drops blanks', () => {
    const p = markerGenesParams({ ...base, minInGroupFraction: '0.1', maxOutGroupFraction: 'x', minFoldChange: '1.5' })
    expect(p.min_in_group_fraction).toBe(0.1)
    expect('max_out_group_fraction' in p).toBe(false)
    expect(p.min_fold_change).toBe(1.5)
  })
})

describe('maskedCategoryCounts', () => {
  const mask = [true, false, true, true, false]

  it('counts a categorical column by code among the masked cells', () => {
    const col = { values: [0, 0, 1, -1, 1], categories: ['a', 'b'] }
    expect(maskedCategoryCounts(col, mask)).toEqual(new Map([['a', 1], ['b', 1]]))
  })

  it('counts a string column by value', () => {
    const col = { values: ['x', 'y', 'x', null, 'y'] }
    expect(maskedCategoryCounts(col, mask)).toEqual(new Map([['x', 2]]))
  })

  it('leaves out a category with no masked cell', () => {
    const col = { values: [0, 1, 0, 0, 1], categories: ['a', 'b'] }
    expect(maskedCategoryCounts(col, mask).has('b')).toBe(false)
  })
})

describe('runnableGroupCount', () => {
  const picked = new Set(['a', 'b', 'c'])

  it('counts every checked group without a mask', () => {
    expect(runnableGroupCount(picked, null)).toBe(3)
  })

  it('counts only groups with two or more active cells, as the backend tests them', () => {
    expect(runnableGroupCount(picked, new Map([['a', 40], ['b', 1]]))).toBe(1)
    expect(runnableGroupCount(picked, new Map([['a', 40], ['b', 2], ['c', 0]]))).toBe(2)
  })
})
