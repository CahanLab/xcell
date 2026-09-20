import { describe, it, expect } from 'vitest'
import { mergePcaSubsetLists, pcSourceOptions, type PCASubsetSummary } from './pcaSubsets'

// With a saved subset active, PCA Loadings creates X_pca_<name>_<suffix> —
// and the Neighbors PC-source picker then has to offer it. The picker's list
// was refreshed without the subset scope, so the subset's PC subsets vanished
// the moment Neighbors was selected. These pin what the store holds and what
// the picker offers.

const sub = (obsmKey: string, cellSubset: string | null = null): PCASubsetSummary =>
  ({ obsmKey, suffix: obsmKey.replace(/^X_pca_(chondro_)?/, ''), droppedPcs: [1], nPcsKept: 4, cellSubset })

describe('mergePcaSubsetLists', () => {
  it("puts the subset's own first, then the dataset's, each tagged with its owner", () => {
    const merged = mergePcaSubsetLists(
      [sub('X_pca_chondro_noPC1')], [sub('X_pca_noPC2'), sub('X_pca_noPC3')], 'chondro')
    expect(merged.map((s) => [s.obsmKey, s.cellSubset])).toEqual([
      ['X_pca_chondro_noPC1', 'chondro'], ['X_pca_noPC2', null], ['X_pca_noPC3', null],
    ])
  })
  it('with no subset active is just the dataset list', () => {
    expect(mergePcaSubsetLists([], [sub('X_pca_noPC2')], null).map((s) => s.cellSubset)).toEqual([null])
  })
})

describe('pcSourceOptions', () => {
  const list = [sub('X_pca_chondro_noPC1', 'chondro'), sub('X_pca_noPC2')]

  it("offers the subset's PCA, its PC subsets, then the dataset's PCA and PC subsets", () => {
    expect(pcSourceOptions('X_pca_chondro', list).map((o) => o.value)).toEqual([
      'X_pca_chondro', 'X_pca_chondro_noPC1', 'X_pca', 'X_pca_noPC2',
    ])
  })
  it('labels which is which', () => {
    const labels = Object.fromEntries(pcSourceOptions('X_pca_chondro', list).map((o) => [o.value, o.label]))
    expect(labels['X_pca_chondro']).toMatch(/this subset's PCA/)
    expect(labels['X_pca_chondro_noPC1']).toMatch(/this subset/)
    expect(labels['X_pca_chondro_noPC1']).toMatch(/4 kept/)
    expect(labels['X_pca']).toMatch(/all PCs/)
    expect(labels['X_pca_noPC2']).toMatch(/4 kept/)
  })
  it('with no subset active leads with the dataset PCA', () => {
    expect(pcSourceOptions(null, [sub('X_pca_noPC2')]).map((o) => o.value)).toEqual(['X_pca', 'X_pca_noPC2'])
  })
})
