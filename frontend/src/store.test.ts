import { describe, it, expect, beforeEach } from 'vitest'
import { useStore, createDefaultDatasetState, Schema } from './store'

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

// A slot is a name, not one of two fixed positions. The backend has always
// keyed its adaptors by an arbitrary string; these pin the frontend to the same
// contract, and pin down what happens when a slot is named that nothing has
// been loaded into — the case that used to be unrepresentable.

const TERTIARY = makeSchema(['X_pca', 'X_umap'], 1600)

describe('slots are named, not numbered', () => {
  it('loads a dataset into a slot that did not exist before', () => {
    useStore.getState().loadDatasetIntoSlot('sample_E11.5', TERTIARY)
    expect(useStore.getState().datasets['sample_E11.5'].schema).toBe(TERTIARY)
  })

  it('leaves the datasets already loaded where they were', () => {
    useStore.getState().loadDatasetIntoSlot('sample_E11.5', TERTIARY)
    const { datasets } = useStore.getState()
    expect(datasets.primary.schema).toBe(PRIMARY)
    expect(datasets.secondary.schema).toBe(SECONDARY)
  })

  it('shows a named slot the same way it shows the first two', () => {
    useStore.getState().loadDatasetIntoSlot('sample_E11.5', TERTIARY)
    useStore.getState().setActiveSlot('sample_E11.5')
    expect(useStore.getState().schema).toBe(TERTIARY)
  })

  it('refuses to make a slot active when there is nothing there to show', () => {
    useStore.getState().setActiveSlot('never_loaded')

    const state = useStore.getState()
    expect(state.activeSlot).toBe('primary')
    // The dangerous outcome is not a throw — it is the flat fields going blank
    // while every panel keeps reading them as the active dataset.
    expect(state.schema).toBe(PRIMARY)
  })

  it('drops a patch aimed at a slot that holds no dataset', () => {
    // An in-flight fetch resolving after its dataset was unloaded, say.
    useStore.getState().patchSlotState('never_loaded', { selectedGenes: ['Sox9'] })
    expect(useStore.getState().datasets['never_loaded']).toBeUndefined()
  })

  it('still activates a slot that exists but has yet to be loaded', () => {
    // 'secondary' exists from the start with a null schema; switching to it is
    // how the UI has always worked and has to keep working.
    useStore.setState({
      datasets: { ...useStore.getState().datasets, secondary: createDefaultDatasetState() },
    })
    useStore.getState().setActiveSlot('secondary')

    expect(useStore.getState().activeSlot).toBe('secondary')
    expect(useStore.getState().schema).toBeNull()
  })
})
