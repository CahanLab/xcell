import { describe, it, expect } from 'vitest'
import { cnvParams, cnvBlocker, cleanKey, CNV_DEFAULTS, type CnvInputs } from './cnv'

const base: CnvInputs = {
  key: 'cnv',
  positions: 'table',
  species: 'mouse',
  chromosomeColumn: '',
  startColumn: '',
  referenceColumn: 'cell_type',
  referenceCategories: new Set(['T cell', 'Fibroblast']),
  includeXY: false,
  cluster: 'leiden',
  ...CNV_DEFAULTS,
  centerCells: true,
  layer: '',
  activeCellIndices: null,
  tableSpecies: ['human', 'mouse'],
  nCells: 5000,
  hierarchicalMaxCells: 20000,
}

describe('cnvParams', () => {
  it('sends the species for the shipped table and no position columns', () => {
    const p = cnvParams(base)
    expect(p.species).toBe('mouse')
    expect('chromosome_column' in p).toBe(false)
    expect(p.reference_column).toBe('cell_type')
    expect(p.reference_categories).toEqual(['T cell', 'Fibroblast'])
    expect(p).toMatchObject({ window: 100, step: 10, min_mean: 0.1, lfc_clip: 3, noise_threshold: 1.5, cluster: 'leiden', resolution: 1, n_clusters: 4, n_pcs: 30, center_cells: true, include_xy: false, key: 'cnv' })
    expect('layer' in p).toBe(false)
    expect('active_cell_indices' in p).toBe(false)
  })

  it('sends the .var columns when positions come from them', () => {
    const p = cnvParams({ ...base, positions: 'columns', chromosomeColumn: 'chrom', startColumn: 'start', species: '' })
    expect(p.chromosome_column).toBe('chrom')
    expect(p.start_column).toBe('start')
    expect('species' in p).toBe(false)
  })

  it('omits the reference when the column is blank', () => {
    const p = cnvParams({ ...base, referenceColumn: '' })
    expect('reference_column' in p).toBe(false)
    expect('reference_categories' in p).toBe(false)
  })

  it('cleans the key and falls back to cnv', () => {
    expect(cnvParams({ ...base, key: ' tumor run 2 ' }).key).toBe('tumor_run_2')
    expect(cnvParams({ ...base, key: '!!!' }).key).toBe('cnv')
    expect(cleanKey('a-b')).toBe('a_b')
  })

  it('coerces numbers and keeps the selection and layer', () => {
    const p = cnvParams({ ...base, window: '50.4', step: 'x', layer: 'counts', activeCellIndices: [1, 2] })
    expect(p.window).toBe(50)
    expect(p.step).toBe(10)
    expect(p.layer).toBe('counts')
    expect(p.active_cell_indices).toEqual([1, 2])
  })
})

describe('cnvBlocker', () => {
  it('passes the defaults', () => {
    expect(cnvBlocker(base)).toBeNull()
  })
  it('needs a species with a shipped table', () => {
    expect(cnvBlocker({ ...base, species: '' })).toMatch(/species/)
    expect(cnvBlocker({ ...base, species: 'zebrafish' })).toMatch(/zebrafish/)
  })
  it('needs both position columns', () => {
    expect(cnvBlocker({ ...base, positions: 'columns', chromosomeColumn: 'c' })).toMatch(/columns/)
  })
  it('needs a reference category once a column is chosen', () => {
    expect(cnvBlocker({ ...base, referenceCategories: new Set() })).toMatch(/reference/)
    expect(cnvBlocker({ ...base, referenceColumn: '', referenceCategories: new Set() })).toBeNull()
  })
  it('checks the window and step', () => {
    expect(cnvBlocker({ ...base, window: '1' })).toMatch(/window/)
    expect(cnvBlocker({ ...base, step: '200' })).toMatch(/step/)
  })
  it('limits hierarchical clustering by cell count', () => {
    expect(cnvBlocker({ ...base, cluster: 'hierarchical', nCells: 30000 })).toMatch(/Leiden/)
    expect(cnvBlocker({ ...base, cluster: 'hierarchical' })).toBeNull()
  })
})
