import { describe, it, expect } from 'vitest'
import { stemFinderParams, stemFinderBlocker, type StemFinderInputs } from './stemfinder'

const base: StemFinderInputs = {
  metrics: new Set(['stemfinder', 'diffometer']),
  markerSource: 'cell_cycle',
  species: 'mouse',
  geneSetGenes: null,
  method: 'gini',
  threshold: '0',
  binarizeOn: 'scaled',
  weightBy: 'equal',
  includeSelf: true,
  graph: 'build',
  useRep: 'X_pca',
  nPcs: '32',
  nNeighbors: '',
  graphKey: '',
  layer: '',
  suffix: '',
  summaryBy: '',
  activeCellIndices: null,
}

describe('stemFinderParams', () => {
  it('sends the species and no markers for the cell-cycle list', () => {
    const p = stemFinderParams(base)
    expect(p.species).toBe('mouse')
    expect('markers' in p).toBe(false)
  })

  it('sends the gene set as markers', () => {
    const p = stemFinderParams({ ...base, markerSource: 'gene_set', geneSetGenes: ['Mcm2', 'Top2a'] })
    expect(p.markers).toEqual(['Mcm2', 'Top2a'])
  })

  it('lists metrics in a fixed order whatever order they were ticked', () => {
    const p = stemFinderParams({ ...base, metrics: new Set(['n_tfs', 'stemfinder']) })
    expect(p.metrics).toEqual(['stemfinder', 'n_tfs'])
  })

  it('leaves k to the backend (√n of the scored cells) when blank', () => {
    expect('n_neighbors' in stemFinderParams(base)).toBe(false)
    expect(stemFinderParams({ ...base, nNeighbors: '50' }).n_neighbors).toBe(50)
  })

  it('sends a blank PC count as all PCs', () => {
    expect(stemFinderParams({ ...base, nPcs: '' }).n_pcs).toBeNull()
  })

  it('names the graph only when reading an existing one', () => {
    const built = stemFinderParams(base)
    expect(built.graph).toBe('build')
    expect('graph_key' in built).toBe(false)
    const existing = stemFinderParams({ ...base, graph: 'existing', graphKey: 'connectivities' })
    expect(existing.graph_key).toBe('connectivities')
    expect('use_rep' in existing).toBe(false)
  })

  it('scopes to the cell mask', () => {
    expect(stemFinderParams({ ...base, activeCellIndices: [1, 2] }).active_cell_indices).toEqual([1, 2])
    expect('active_cell_indices' in stemFinderParams(base)).toBe(false)
  })

  it('omits a blank layer, suffix and summary column', () => {
    const p = stemFinderParams(base)
    expect('layer' in p || 'summary_by' in p).toBe(false)
    expect(p.suffix).toBe('')
    expect(stemFinderParams({ ...base, layer: 'counts', summaryBy: 'leiden' })).toMatchObject({ layer: 'counts', summary_by: 'leiden' })
  })
})

describe('stemFinderBlocker', () => {
  it('is null for the defaults', () => {
    expect(stemFinderBlocker(base)).toBeNull()
  })

  it('needs a metric', () => {
    expect(stemFinderBlocker({ ...base, metrics: new Set() })).toMatch(/metric/i)
  })

  it('needs genes in the chosen gene set', () => {
    expect(stemFinderBlocker({ ...base, markerSource: 'gene_set', geneSetGenes: null })).toMatch(/gene set/i)
  })

  it('needs k of at least 2 when one is given', () => {
    expect(stemFinderBlocker({ ...base, nNeighbors: '1' })).toMatch(/k/i)
    expect(stemFinderBlocker({ ...base, nNeighbors: 'x' })).toMatch(/k/i)
  })

  it('needs a graph when reading an existing one', () => {
    expect(stemFinderBlocker({ ...base, graph: 'existing', graphKey: '' })).toMatch(/graph/i)
  })

  it('needs no graph or markers for the TF count alone', () => {
    expect(stemFinderBlocker({ ...base, metrics: new Set(['n_tfs']), graph: 'existing', graphKey: '' })).toBeNull()
  })

  it('refuses a suffix that would write stemFinder over another score’s column', () => {
    // stemfinder + '_raw' is the raw score's own name; the backend refuses it too.
    expect(stemFinderBlocker({ ...base, suffix: 'raw' })).toMatch(/stemfinder_raw/)
    expect(stemFinderBlocker({ ...base, suffix: ' raw counts ' })).toMatch(/stemfinder_raw_counts/)
    expect(stemFinderBlocker({ ...base, suffix: 'n_TFs' })).toMatch(/stemfinder_n_TFs/)
    expect(stemFinderBlocker({ ...base, suffix: 'rawcounts' })).toBeNull()
    expect(stemFinderBlocker({ ...base, metrics: new Set(['diffometer']), suffix: 'raw' })).toBeNull()
  })

  it('needs a numeric threshold', () => {
    expect(stemFinderBlocker({ ...base, threshold: 'abc' })).toMatch(/threshold/i)
    expect(stemFinderBlocker({ ...base, threshold: '-0.5' })).toBeNull()
  })

  it('says to run PCA first when there is nothing to build a neighbourhood from', () => {
    expect(stemFinderBlocker({ ...base, pcEmbeddingCount: 0, graphCount: 0 })).toMatch(/run pca/i)
    expect(stemFinderBlocker({ ...base, metrics: new Set(['cc_mean']), pcEmbeddingCount: 0, graphCount: 0 })).toBeNull()
  })

  it('rejects expression weighting on scaled expression', () => {
    expect(stemFinderBlocker({ ...base, weightBy: 'expression' })).toMatch(/expression/i)
    expect(stemFinderBlocker({ ...base, weightBy: 'expression', binarizeOn: 'log_normalized' })).toBeNull()
  })
})
