import { describe, it, expect } from 'vitest'
import {
  sanitizeSubsetName, isUsableSubsetName, suggestSubsetName,
  indicesFromMask, maskFromIndices, maskMatchesIndices,
  scopedOutput, derivedBadges, SUBSET_SCOPED_OPS,
} from './cellSubsets'

describe('subset names', () => {
  it('sanitises the way the backend does', () => {
    expect(sanitizeSubsetName('my chondrocytes (E11.5)')).toBe('my_chondrocytes_E11_5')
    expect(sanitizeSubsetName('  chondro  ')).toBe('chondro')
    expect(sanitizeSubsetName('(((')).toBe('')
  })

  it('rejects empty and reserved names', () => {
    expect(isUsableSubsetName('')).toBe(false)
    expect(isUsableSubsetName('   ')).toBe(false)
    expect(isUsableSubsetName('unassigned')).toBe(false)
    expect(isUsableSubsetName('NaN')).toBe(false)
    expect(isUsableSubsetName('chondro')).toBe(true)
  })

  it('suggests the first free subN', () => {
    expect(suggestSubsetName([])).toBe('sub1')
    expect(suggestSubsetName(['sub1', 'chondro'])).toBe('sub2')
    expect(suggestSubsetName(['sub2'])).toBe('sub1')
  })
})

describe('masks and indices', () => {
  it('round-trips', () => {
    const mask = [true, false, true, false]
    expect(indicesFromMask(mask)).toEqual([0, 2])
    expect(maskFromIndices([0, 2], 4)).toEqual(mask)
  })

  it('ignores out-of-range indices rather than growing the mask', () => {
    expect(maskFromIndices([1, 9], 3)).toEqual([false, true, false])
  })

  it('knows when a mask is exactly a subset', () => {
    expect(maskMatchesIndices([true, false, true], [0, 2])).toBe(true)
    expect(maskMatchesIndices([true, true, true], [0, 2])).toBe(false)
    expect(maskMatchesIndices([true, false, false], [0, 2])).toBe(false)
  })
})

describe('scoped outputs', () => {
  it('names what each chain operation writes and what it leaves alone', () => {
    expect(scopedOutput('pca', 'chondro')).toEqual({
      writes: '.obsm["X_pca_chondro"]', spares: '.obsm["X_pca"]',
    })
    expect(scopedOutput('leiden', 'chondro')).toEqual({
      writes: '.obs["leiden_chondro"]', spares: '.obs["leiden"]',
    })
    expect(scopedOutput('neighbors', 'chondro')?.writes).toBe('.obsp["chondro_connectivities"]')
    expect(scopedOutput('highly_variable_genes', 'chondro')?.writes).toBe('.var["highly_variable__chondro"]')
    expect(scopedOutput('umap', 'chondro')?.writes).toBe('.obsm["X_umap_chondro"]')
  })

  it('has nothing to say about operations the subset does not scope', () => {
    expect(scopedOutput('normalize_total', 'chondro')).toBeNull()
    expect(SUBSET_SCOPED_OPS.has('normalize_total')).toBe(false)
    expect(SUBSET_SCOPED_OPS.has('leiden')).toBe(true)
  })

  it('summarises what has been derived', () => {
    expect(derivedBadges({ hvg: 'highly_variable__c', pca: 'X_pca_c', graph: null, umap: ['X_umap_c', 'X_umap_c_alt'], leiden: ['leiden_c'], pca_subsets: ['X_pca_c_noPC1'] }))
      .toEqual(['HVG', 'PCA', 'X_pca_c_noPC1', 'X_umap_c', 'X_umap_c_alt', 'leiden_c'])
    expect(derivedBadges({ hvg: null, pca: null, graph: null, umap: [], leiden: [], pca_subsets: [] })).toEqual([])
  })
})
