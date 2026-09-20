import { describe, it, expect } from 'vitest'
import type { CellSubsetInfo } from './cellSubsets'
import { subsetChips, dropSummary, refineTarget } from './subsetTree'

// A subset row is a tree node with chips for everything it owns: the steps
// run on it, its embeddings, its clusterings, and what was drawn on those
// embeddings. These pin the chip order, the wording of the delete confirm,
// and which column a "refine" hand-off targets.

const base = (over: Partial<CellSubsetInfo> = {}): CellSubsetInfo => ({
  name: 'chondro',
  obs_key: 'subset_chondro',
  n_cells: 30,
  n_total: 60,
  created_at: null,
  description: null,
  parent: null,
  children: [],
  depth: 0,
  origin: null,
  steps: {},
  derived: {
    hvg: 'highly_variable__chondro',
    pca: 'X_pca_chondro',
    graph: 'chondro_connectivities',
    umap: ['X_umap_chondro'],
    leiden: ['leiden_chondro'],
    pca_subsets: ['X_pca_chondro_noPC1'],
  },
  embeddings: ['X_pca_chondro', 'X_pca_chondro_noPC1', 'X_umap_chondro'],
  decorations: { lines: ['ridge', 'r2'], territories: ['zones'] },
  ...over,
})

const empty = base({
  derived: { hvg: null, pca: null, graph: null, umap: [], leiden: [], pca_subsets: [] },
  embeddings: [],
  decorations: { lines: [], territories: [] },
})

describe('subsetChips', () => {
  it('lists steps, embeddings, clusterings and decorations in a stable order', () => {
    expect(subsetChips(base()).map((c) => c.label)).toEqual([
      'HVG', 'X_pca_chondro', 'X_pca_chondro_noPC1', 'kNN', 'X_umap_chondro',
      'leiden_chondro', '2 shapes', '1 territory',
    ])
  })

  it('types each chip so the panel knows what a click does', () => {
    const kinds = Object.fromEntries(subsetChips(base()).map((c) => [c.label, c.kind]))
    expect(kinds).toEqual({
      HVG: 'step', X_pca_chondro: 'embedding', X_pca_chondro_noPC1: 'embedding', kNN: 'step',
      X_umap_chondro: 'embedding', leiden_chondro: 'leiden', '2 shapes': 'lines', '1 territory': 'territories',
    })
  })

  it('points a decoration chip at the embedding it was drawn on', () => {
    const chips = subsetChips(base())
    expect(chips.find((c) => c.kind === 'lines')?.embedding).toBe('X_umap_chondro')
    expect(chips.find((c) => c.kind === 'territories')?.embedding).toBe('X_umap_chondro')
  })

  it('singular and plural', () => {
    const one = base({ decorations: { lines: ['a'], territories: ['x', 'y'] } })
    expect(subsetChips(one).slice(-2).map((c) => c.label)).toEqual(['1 shape', '2 territories'])
  })

  it('is empty for a subset with nothing computed', () => {
    expect(subsetChips(empty)).toEqual([])
  })
})

describe('dropSummary', () => {
  it('names what a delete with results removes', () => {
    expect(dropSummary(base())).toBe(
      'highly_variable__chondro, X_pca_chondro, X_pca_chondro_noPC1, chondro_connectivities, ' +
      'X_umap_chondro, leiden_chondro, 2 shapes, 1 territory type')
  })
  it('is empty when there is nothing to remove', () => {
    expect(dropSummary(empty)).toBe('')
  })
})

describe('refineTarget', () => {
  const parent = base({ name: 'p', derived: { ...base().derived, leiden: ['leiden_p', 'leiden_p_r2'] } })
  it("prefers the parent's first leiden column", () => {
    expect(refineTarget(base({ parent: 'p' }), [parent, base({ parent: 'p' })], ['leiden'])).toBe('leiden_p')
  })
  it("falls back to the dataset's leiden for a root", () => {
    expect(refineTarget(base(), [base()], ['leiden', 'cell_type'])).toBe('leiden')
  })
  it('has no target when neither exists', () => {
    expect(refineTarget(base(), [base()], ['cell_type'])).toBeNull()
  })
})
