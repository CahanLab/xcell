import { describe, it, expect } from 'vitest'
import { markerGenesParams } from './markerGenes'

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

  it('parses the optional thresholds and drops blanks', () => {
    const p = markerGenesParams({ ...base, minInGroupFraction: '0.1', maxOutGroupFraction: 'x', minFoldChange: '1.5' })
    expect(p.min_in_group_fraction).toBe(0.1)
    expect('max_out_group_fraction' in p).toBe(false)
    expect(p.min_fold_change).toBe(1.5)
  })
})
