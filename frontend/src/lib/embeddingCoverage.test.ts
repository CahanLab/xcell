import { describe, it, expect } from 'vitest'
import type { CellSubsetInfo } from './cellSubsets'
import { cellsWithoutCoordinates, supersetEmbedding } from './embeddingCoverage'

// A subset's UMAP has no coordinates for the cells outside it, so a plot on it
// silently shows only the subset. After analysing a subset the modal switches
// the plot there, and Reset Mask then looks broken: the mask is gone but the
// other cells never come back. These pin the way back.

const subset = (over: Partial<CellSubsetInfo> = {}): CellSubsetInfo => ({
  name: 'chondro', obs_key: 'subset_chondro', n_cells: 500, n_total: 1000,
  created_at: null, description: null, parent: null, children: [], depth: 0,
  origin: { kind: 'selection', embedding: 'X_umap' }, steps: {},
  derived: { hvg: null, pca: 'X_pca_chondro', graph: null, umap: ['X_umap_chondro'], leiden: [], pca_subsets: [] },
  embeddings: ['X_pca_chondro', 'X_umap_chondro'],
  decorations: { lines: [], territories: [] },
  ...over,
})

describe('cellsWithoutCoordinates', () => {
  it('counts null rows and NaN coordinates, not real points', () => {
    expect(cellsWithoutCoordinates([[0, 1], null, [NaN, NaN], [2, 3], undefined, [null as any, 1]])).toBe(4)
    expect(cellsWithoutCoordinates([[0, 1], [2, 3]])).toBe(0)
    expect(cellsWithoutCoordinates([])).toBe(0)
  })
})

describe('supersetEmbedding', () => {
  const all = ['X_spatial', 'spatial', 'X_pca', 'X_umap', 'X_pca_chondro', 'X_umap_chondro']

  it('goes back to where the subset was drawn', () => {
    expect(supersetEmbedding('X_umap_chondro', all, [subset()])).toBe('X_umap')
    expect(supersetEmbedding('X_umap_chondro', all, [subset({ origin: { kind: 'selection', embedding: 'spatial' } })])).toBe('spatial')
  })

  it("falls back to the dataset's counterpart when the origin is unknown or gone", () => {
    expect(supersetEmbedding('X_umap_chondro', all, [subset({ origin: null })])).toBe('X_umap')
    expect(supersetEmbedding('X_umap_chondro', all, [subset({ origin: { kind: 'selection', embedding: 'X_umap_old' } })])).toBe('X_umap')
    expect(supersetEmbedding('X_umap_chondro_alt', [...all, 'X_umap_alt'], [subset({ origin: null, embeddings: ['X_umap_chondro_alt'] })])).toBe('X_umap_alt')
  })

  it('never answers with the embedding it is leaving, even if that is the recorded origin', () => {
    const child = subset({ name: 'prolif', parent: 'chondro', origin: { kind: 'selection', embedding: 'X_umap_prolif' }, embeddings: ['X_umap_prolif'] })
    expect(supersetEmbedding('X_umap_prolif', [...all, 'X_umap_prolif'], [subset(), child])).toBe('X_umap')
  })

  it('a child drawn on its parent UMAP goes back one level, to the parent', () => {
    const child = subset({ name: 'prolif', parent: 'chondro', origin: { kind: 'selection', embedding: 'X_umap_chondro' }, embeddings: ['X_umap_prolif'] })
    expect(supersetEmbedding('X_umap_prolif', [...all, 'X_umap_prolif'], [subset(), child])).toBe('X_umap_chondro')
  })

  it('falls back to the preferred embedding for something no subset owns', () => {
    expect(supersetEmbedding('X_predicted', ['X_predicted', 'X_pca', 'X_umap'], [])).toBe('X_umap')
  })

  it('has nowhere to go when it is the only embedding', () => {
    expect(supersetEmbedding('X_umap_chondro', ['X_umap_chondro'], [subset({ origin: null })])).toBeNull()
  })
})
