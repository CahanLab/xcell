import { describe, it, expect, beforeEach } from 'vitest'
import { useStore, Schema } from './store'

// A comparison is a statement about *one* dataset: "these cells versus those
// cells, in this matrix". Cell indices, the split view's second geometry, and
// the resulting DE table are therefore only meaningful next to the dataset they
// were drawn on. These tests pin that down at the store level, because the
// failure mode is silent — primary's row indices are all in range for a larger
// secondary, so the wrong answer looks like a right one.

function makeSchema(embeddings: string[], nCells: number): Schema {
  return {
    n_cells: nCells,
    n_genes: 100,
    embeddings,
    obs_columns: [],
    obs_dtypes: {},
  }
}

// Deliberately mismatched, like the real pair that surfaced this: the primary
// carries derived embeddings the secondary has never heard of.
const PRIMARY = makeSchema(
  ['X_pca', 'X_spatial_k20_none_corr_inj', 'X_umap', 'neighborhood_composition'],
  4212,
)
const SECONDARY = makeSchema(['X_pca', 'X_spatial', 'X_umap'], 9145)

const pristine = useStore.getState()

beforeEach(() => {
  useStore.setState(pristine, true)
  const s = useStore.getState()
  s.loadDatasetIntoSlot('primary', PRIMARY)
  s.loadDatasetIntoSlot('secondary', SECONDARY)
})

describe('comparison groups are scoped to their dataset', () => {
  it('does not carry primary cell indices onto the secondary', () => {
    const s = useStore.getState()
    s.setComparisonGroup1([0, 1, 2], 'primary grp1')
    s.setComparisonGroup2([3000, 3001], 'primary grp2')

    useStore.getState().setActiveSlot('secondary')

    const { comparison } = useStore.getState()
    expect(comparison.group1).toBeNull()
    expect(comparison.group2).toBeNull()
  })

  it('restores each dataset its own groups when the user switches back', () => {
    const s = useStore.getState()
    s.setComparisonGroup1([0, 1, 2], 'primary grp1')

    useStore.getState().setActiveSlot('secondary')
    useStore.getState().setComparisonGroup1([9000, 9001], 'secondary grp1')
    useStore.getState().setActiveSlot('primary')

    const { comparison } = useStore.getState()
    expect(comparison.group1).toEqual([0, 1, 2])
    expect(comparison.group1Label).toBe('primary grp1')
  })

  it('leaves the DE table behind with the dataset it was computed on', () => {
    useStore.getState().setDiffExpResult({
      positive: [], negative: [], group1_count: 3, group2_count: 2,
    })

    useStore.getState().setActiveSlot('secondary')

    expect(useStore.getState().diffExpResult).toBeNull()
  })
})

describe("the split view's second geometry is scoped to its dataset", () => {
  it('never leaves a dataset showing an embedding it does not have', () => {
    const s = useStore.getState()
    s.setSplitView(true)
    s.setSecondEmbedding('neighborhood_composition')

    useStore.getState().setActiveSlot('secondary')

    const { secondEmbedding } = useStore.getState()
    expect(SECONDARY.embeddings).toContain(secondEmbedding)
  })

  it('remembers each dataset its own second embedding', () => {
    const s = useStore.getState()
    s.setSplitView(true)
    s.setSecondEmbedding('neighborhood_composition')

    useStore.getState().setActiveSlot('secondary')
    useStore.getState().setSecondEmbedding('X_spatial')
    useStore.getState().setActiveSlot('primary')

    expect(useStore.getState().secondEmbedding).toBe('neighborhood_composition')
  })

  it('gives a dataset loaded under an open split view something to show', () => {
    const s = useStore.getState()
    s.setSplitView(true)
    s.setSecondEmbedding('neighborhood_composition')

    // Replacing the active slot's dataset — File → Load, same slot.
    useStore.getState().loadDatasetIntoSlot('primary', SECONDARY)

    const { secondEmbedding } = useStore.getState()
    expect(SECONDARY.embeddings).toContain(secondEmbedding)
  })

  it('drops the other dataset\'s coordinates rather than drawing them', () => {
    const s = useStore.getState()
    s.setSplitView(true)
    s.setSecondEmbedding('X_umap')
    s.setSecondEmbeddingData({
      name: 'X_umap',
      coordinates: Array.from({ length: 4212 }, () => [0, 0] as [number, number]),
    })

    useStore.getState().setActiveSlot('secondary')

    expect(useStore.getState().secondEmbeddingData).toBeNull()
  })
})
