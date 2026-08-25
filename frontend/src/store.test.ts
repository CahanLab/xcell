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

// The audit that came with widening DatasetSlot: state that names an .obs
// column, an embedding's columns, or a drawn line means something only next to
// the dataset it was chosen on, so it cannot sit at the top level. Each of
// these was global.
describe('analysis state that names part of a dataset stays with it', () => {
  it('does not view one dataset at the PCs chosen on another', () => {
    // Both datasets have an X_pca. Chosen on the primary, these dims were
    // silently applied to whatever loaded next — a different projection than
    // the user asked for, or a column the secondary's X_pca does not have.
    useStore.getState().setEmbeddingDims('X_pca', 2, 4)

    useStore.getState().setActiveSlot('secondary')

    expect(useStore.getState().embeddingDims['X_pca']).toBeUndefined()
  })

  it('remembers each dataset the dims chosen on it', () => {
    useStore.getState().setEmbeddingDims('X_pca', 2, 4)
    useStore.getState().setActiveSlot('secondary')
    useStore.getState().setEmbeddingDims('X_pca', 0, 1)
    useStore.getState().setActiveSlot('primary')

    expect(useStore.getState().embeddingDims['X_pca']).toEqual({ x: 2, y: 4, z: undefined })
  })

  it('does not carry a marker-genes column onto a dataset that may not have it', () => {
    useStore.getState().setMarkerGenesColumn('leiden_primary')

    useStore.getState().setActiveSlot('secondary')

    expect(useStore.getState().markerGenesColumn).toBeNull()
  })

  it('unticks category checkboxes belonging to the other dataset', () => {
    // Two datasets both having a 'leiden' is the ordinary case, and its
    // categories are named '0', '1', '2' in both — so the ticks reappear on
    // the new dataset looking deliberate.
    useStore.getState().toggleComparisonCategory('leiden', '3')

    useStore.getState().setActiveSlot('secondary')

    expect(useStore.getState().comparisonCheckedColumn).toBeNull()
    expect(useStore.getState().comparisonCheckedCategories.size).toBe(0)
  })

  it('leaves a line-association table behind with the dataset it was run on', () => {
    useStore.getState().setLineAssociationResult({
      positive: [], negative: [], modules: [],
      n_cells: 4212, n_significant: 0, n_positive: 0, n_negative: 0, n_modules: 0,
      line_name: 'a line only the primary has', test_variable: 'position',
      fdr_threshold: 0.05,
    })

    useStore.getState().setActiveSlot('secondary')

    expect(useStore.getState().lineAssociationResult).toBeNull()
  })
})

// The flat top-level fields are a cache of the active slot's DatasetState.
// A field that lives in both but is not copied across on a slot switch is a
// silent staleness bug — it keeps showing the previous dataset's answer, and
// nothing about it looks wrong. This checks the correspondence mechanically,
// so adding a field to DatasetState forces the question rather than relying on
// whoever adds it remembering syncFlatFields exists.
describe('every mirrored field follows the active dataset', () => {
  it('leaves nothing behind when the active slot changes', () => {
    const datasetFields = Object.keys(createDefaultDatasetState())
    const flatFields = new Set(Object.keys(pristine))
    const mirrored = datasetFields.filter((f) => flatFields.has(f))

    // A value nothing else in the store can equal, per field.
    const marked: Record<string, unknown> = {}
    for (const f of mirrored) marked[f] = `__${f}__`

    const { datasets } = useStore.getState()
    useStore.setState({
      datasets: { ...datasets, secondary: { ...datasets.secondary, ...marked } as never },
    })
    useStore.getState().setActiveSlot('secondary')

    const state = useStore.getState() as unknown as Record<string, unknown>
    const notCopied = mirrored.filter((f) => state[f] !== marked[f])
    expect(notCopied).toEqual([])
  })

  it('actually mirrors something — the check above is not vacuous', () => {
    const datasetFields = Object.keys(createDefaultDatasetState())
    const flatFields = new Set(Object.keys(pristine))
    expect(datasetFields.filter((f) => flatFields.has(f)).length).toBeGreaterThan(20)
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

// Closing a dataset. The backend has had DELETE /api/datasets/{slot} all along
// with nothing calling it; these pin down what the store does around that call.
describe('unloading a dataset', () => {
  it('takes the dataset out of the store', () => {
    useStore.getState().unloadDataset('secondary')
    expect(useStore.getState().datasets.secondary).toBeUndefined()
  })

  it('lands you on the neighbour when you close the one you were looking at', () => {
    useStore.getState().setActiveSlot('secondary')

    useStore.getState().unloadDataset('secondary')

    const state = useStore.getState()
    expect(state.activeSlot).toBe('primary')
    // Not just the pointer: the flat fields every panel reads have to follow.
    expect(state.schema).toBe(PRIMARY)
  })

  it('leaves the active dataset alone when you close another one', () => {
    useStore.getState().unloadDataset('secondary')
    expect(useStore.getState().activeSlot).toBe('primary')
    expect(useStore.getState().schema).toBe(PRIMARY)
  })

  it('refuses to close the last dataset, leaving the app with nothing to show', () => {
    useStore.getState().unloadDataset('secondary')

    useStore.getState().unloadDataset('primary')

    expect(useStore.getState().datasets.primary.schema).toBe(PRIMARY)
    expect(useStore.getState().activeSlot).toBe('primary')
  })

  it('drops the dragged pane sizes, since the grid is a different shape now', () => {
    useStore.getState().setPaneLayout({ cols: [1.4, 0.6], rows: [1] })

    useStore.getState().unloadDataset('secondary')

    expect(useStore.getState().paneLayout).toEqual({ cols: [], rows: [] })
  })
})

describe('a third dataset is not a second-class one', () => {
  it('gives it the display preferences from the user config too', () => {
    // applyUserConfig wrote into 'primary' and 'secondary' by name, so a third
    // dataset silently kept the built-in defaults.
    useStore.getState().loadDatasetIntoSlot('slot3', TERTIARY)
    useStore.setState({ userConfig: { display: { point_size: 9 } } })

    useStore.getState().applyConfigDefaults()

    const { datasets } = useStore.getState()
    expect(datasets.slot3.displayPreferences.pointSize).toBe(9)
    expect(datasets.primary.displayPreferences.pointSize).toBe(9)
  })
})


describe('arranging the workspace', () => {
  it('lets a dataset be called something other than its filename', () => {
    useStore.getState().setDatasetDisplayName('secondary', 'E11.5 rep2')

    expect(useStore.getState().datasets.secondary.displayName).toBe('E11.5 rep2')
    // The name belongs to that dataset, not to the app.
    expect(useStore.getState().datasets.primary.displayName).toBeNull()
  })

  it('keeps a name with its dataset across a slot switch', () => {
    useStore.getState().setDatasetDisplayName('secondary', 'E11.5 rep2')
    useStore.getState().setActiveSlot('secondary')
    useStore.getState().setActiveSlot('primary')

    expect(useStore.getState().datasets.secondary.displayName).toBe('E11.5 rep2')
  })

  it('reorders the tabs, and the panes with them', () => {
    useStore.getState().loadDatasetIntoSlot('slot3', TERTIARY)

    useStore.getState().reorderSlots(0, 2)

    // Pane order is slot order is the key order of `datasets`.
    expect(Object.keys(useStore.getState().datasets)).toEqual(['secondary', 'slot3', 'primary'])
  })

  it('moves the dataset itself, not just its name', () => {
    useStore.getState().reorderSlots(0, 1)

    const [first] = Object.keys(useStore.getState().datasets)
    expect(useStore.getState().datasets[first].schema).toBe(SECONDARY)
  })

  it('leaves the active dataset active after a reorder', () => {
    useStore.getState().setActiveSlot('secondary')
    useStore.getState().reorderSlots(0, 1)

    expect(useStore.getState().activeSlot).toBe('secondary')
    expect(useStore.getState().schema).toBe(SECONDARY)
  })

  it('restores an arrangement from a previous visit', () => {
    useStore.getState().loadDatasetIntoSlot('slot3', TERTIARY)

    useStore.getState().applyWorkspace({
      order: ['slot3', 'primary', 'secondary'],
      names: { slot3: 'E11.5 rep2' },
      cols: [1.5, 0.5, 1],
      rows: [1],
      tiled: true,
    })

    const state = useStore.getState()
    expect(Object.keys(state.datasets)).toEqual(['slot3', 'primary', 'secondary'])
    expect(state.datasets.slot3.displayName).toBe('E11.5 rep2')
    expect(state.paneLayout.cols).toEqual([1.5, 0.5, 1])
    // And it comes back to the view it was left in.
    expect(state.layoutMode).toBe('tiled')
  })

  it('ignores a stored arrangement naming datasets that are not loaded', () => {
    // Last visit had four; this one opened two of them.
    useStore.getState().applyWorkspace({
      order: ['slot9', 'secondary', 'primary', 'slot4'],
      names: { slot9: 'gone' },
      cols: [1, 1, 1, 1],
      rows: [1],
      tiled: true,
    })

    const state = useStore.getState()
    expect(Object.keys(state.datasets)).toEqual(['secondary', 'primary'])
    // Four columns for two panes would lay them out at the wrong widths.
    expect(state.paneLayout.cols).toEqual([])
  })
})


describe('the barplot is a statement about one dataset', () => {
  it('does not carry a barplot onto a dataset whose columns are different', () => {
    // 'leiden' exists in both datasets and means something different in each,
    // which is exactly why the config cannot be global.
    useStore.getState().setBarplotConfig({
      columnA: 'leiden', columnB: 'territory_prox-dist',
      order: 'share', shareOf: 'proximal',
      normalize: true, minCells: 0, showValues: false,
    })

    useStore.getState().setActiveSlot('secondary')

    expect(useStore.getState().barplotConfig).toBeNull()
  })

  it('gives each dataset back its own barplot', () => {
    useStore.getState().setBarplotConfig({
      columnA: 'leiden', columnB: 'territory_prox-dist',
      order: 'category', shareOf: null,
      normalize: true, minCells: 0, showValues: false,
    })
    useStore.getState().setActiveSlot('secondary')
    useStore.getState().setBarplotConfig({
      columnA: 'seurat_clusters', columnB: 'section',
      order: 'total', shareOf: null,
      normalize: false, minCells: 10, showValues: true,
    })
    useStore.getState().setActiveSlot('primary')

    expect(useStore.getState().barplotConfig?.columnA).toBe('leiden')
  })
})

// Reported 2026-08-25. Close the dataset you loaded first, refresh, and every
// request came back 503 "No data loaded for slot 'primary'" while two datasets
// sat healthy in the backend. activeSlot starts as the literal 'primary' and
// nothing on the restore path moved it, so the whole app addressed a slot that
// was not there — invisibly, because appendDataset() omits ?dataset= for
// primary, making "no slot named" and "primary" the same request on the wire.
describe('the active slot always names a dataset that exists', () => {
  it('moves off primary when the session no longer has one', () => {
    useStore.setState(pristine, true)
    const s = useStore.getState()
    s.loadDatasetIntoSlot('slot3', PRIMARY)
    s.loadDatasetIntoSlot('slot4', SECONDARY)
    expect(useStore.getState().activeSlot).toBe('primary')   // the broken state

    useStore.getState().ensureActiveSlot(['slot3', 'slot4'])
    expect(useStore.getState().activeSlot).toBe('slot3')
  })

  it('brings the mirrored fields along, so the panels read the new dataset', () => {
    useStore.setState(pristine, true)
    const s = useStore.getState()
    s.loadDatasetIntoSlot('slot4', SECONDARY)
    useStore.getState().ensureActiveSlot(['slot4'])
    const after = useStore.getState()
    expect(after.activeSlot).toBe('slot4')
    expect(after.schema?.n_cells).toBe(SECONDARY.n_cells)
  })

  it('leaves a valid active slot alone', () => {
    useStore.getState().setActiveSlot('secondary')
    useStore.getState().ensureActiveSlot(['primary', 'secondary'])
    expect(useStore.getState().activeSlot).toBe('secondary')
  })

  it('trusts the backend list over the store, mid-startup', () => {
    // primary is held by the backend but its schema has not landed yet. Moving
    // off it here would fight useSchema and land the user on the wrong tab.
    useStore.setState(pristine, true)
    useStore.getState().loadDatasetIntoSlot('slot3', SECONDARY)
    useStore.getState().ensureActiveSlot(['primary', 'slot3'])
    expect(useStore.getState().activeSlot).toBe('primary')
  })
})
