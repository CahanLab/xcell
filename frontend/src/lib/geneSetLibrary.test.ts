import { describe, it, expect } from 'vitest'
import {
  sortLibraries, filterLibraries, attachOverlap, presenceLabel, importSets,
  pickSpecies, rowPasses, defaultFolderName, type LibraryEntry, type LibrarySet, type OverlapEntry,
} from './geneSetLibrary'

const lib = (over: Partial<LibraryEntry>): LibraryEntry => ({
  source: 'enrichr', id: 'X', name: 'X', description: '', species: 'human', n_sets: 10,
  version: null, url: '', cached: false, fetched_at: null, ...over,
})

describe('sortLibraries', () => {
  it('puts the browsed species first, cached libraries next, then names A→Z', () => {
    const out = sortLibraries([
      lib({ id: 'zeta', name: 'Zeta', species: 'human' }),
      lib({ id: 'beta', name: 'Beta', species: 'mouse' }),
      lib({ id: 'alpha', name: 'Alpha', species: 'human', cached: true }),
      lib({ id: 'gamma', name: 'Gamma', species: 'mouse', cached: true }),
    ], 'mouse')
    expect(out.map((l) => l.id)).toEqual(['gamma', 'beta', 'alpha', 'zeta'])
  })
  it('orders numbered names numerically', () => {
    const mk = (name: string): LibraryEntry => ({ source: 'mgi', id: name, name, description: '', species: 'mouse', n_sets: null, version: null, url: '', cached: false, fetched_at: null })
    expect(sortLibraries([mk('TS10 · E7'), mk('TS2 · E1'), mk('TS1 · E0')], 'mouse').map((e) => e.name)).toEqual(['TS1 · E0', 'TS2 · E1', 'TS10 · E7'])
  })
})

describe('filterLibraries', () => {
  it('matches id, name or description without regard to case; empty text keeps all', () => {
    const entries = [lib({ id: 'GO_BP', name: 'GO BP', description: 'ontology' }), lib({ id: 'KEGG', name: 'KEGG', description: '' })]
    expect(filterLibraries(entries, 'ONTOLOGY').map((l) => l.id)).toEqual(['GO_BP'])
    expect(filterLibraries(entries, 'kegg').map((l) => l.id)).toEqual(['KEGG'])
    expect(filterLibraries(entries, '  ').length).toBe(2)
  })
})

const set = (name: string, genes: string[]): LibrarySet => ({ name, description: '', url: '', n_genes: genes.length, genes })
const ov = (name: string, over: Partial<OverlapEntry>): OverlapEntry => ({
  name, n_genes: 0, n_present: 0, n_exact: 0, n_case_insensitive: 0, n_missing: 0,
  genes_resolved: [], genes_down_resolved: [], genes_missing: [], columns: {}, ...over,
})

describe('overlap presentation', () => {
  it('pairs overlap rows by position and labels presence, flagging case-insensitive hits', () => {
    const rows = attachOverlap([set('A', ['COL1A1', 'COL1A2', 'ZZ'])], [
      ov('A', { n_genes: 3, n_present: 2, n_case_insensitive: 2, n_missing: 1, genes_resolved: ['Col1a1', 'Col1a2'], genes_missing: ['ZZ'] }),
    ])
    expect(rows[0].overlap?.genes_resolved).toEqual(['Col1a1', 'Col1a2'])
    expect(presenceLabel(rows[0])).toBe('2 / 3 (case-insensitive)')
    expect(presenceLabel({ ...rows[0], overlap: ov('A', { n_genes: 3, n_present: 3, n_exact: 3 }) })).toBe('3 / 3')
    expect(presenceLabel({ ...rows[0], overlap: undefined })).toBe('…')
  })

  it('rowPasses applies a minimum present count and an optional per-column minimum', () => {
    const row = attachOverlap([set('A', ['a', 'b'])], [ov('A', { n_genes: 2, n_present: 2, columns: { highly_variable: 1 } })])[0]
    expect(rowPasses(row, { minPresent: 2 })).toBe(true)
    expect(rowPasses(row, { minPresent: 3 })).toBe(false)
    expect(rowPasses(row, { minPresent: 0, column: 'highly_variable', minInColumn: 1 })).toBe(true)
    expect(rowPasses(row, { minPresent: 0, column: 'highly_variable', minInColumn: 2 })).toBe(false)
    // no overlap yet → never filtered out, so rows don't vanish while the check is in flight
    expect(rowPasses({ ...row, overlap: undefined }, { minPresent: 5 })).toBe(true)
  })
})

describe('importSets', () => {
  it('imports the dataset spelling, keeps down genes, records provenance, skips empty sets', () => {
    const rows = attachOverlap([set('A', ['COL1A1']), set('B', ['NOPE'])], [
      ov('A', { n_genes: 1, n_present: 1, genes_resolved: ['Col1a1'], genes_down_resolved: ['Sox9'] }),
      ov('B', { n_genes: 1, n_present: 0, genes_missing: ['NOPE'] }),
    ])
    const out = importSets(rows, { source: 'msigdb', library: 'm2.cgp', libraryName: 'Curated', version: '2026.1' })
    expect(out).toEqual([{
      name: 'A', genes: ['Col1a1'], genesDown: ['Sox9'],
      source: { source: 'msigdb', library: 'm2.cgp', name: 'Curated', version: '2026.1' },
    }])
  })

  it('falls back to the library spelling when no overlap was computed', () => {
    const out = importSets([set('A', ['COL1A1'])], { source: 'enrichr', library: 'GO', libraryName: 'GO', version: null })
    expect(out[0].genes).toEqual(['COL1A1'])
    expect(out[0].genesDown).toBeUndefined()
  })
})

describe('pickSpecies / defaultFolderName', () => {
  it('prefers the dataset guess, then the remembered choice, then human', () => {
    expect(pickSpecies({ species: 'mouse' }, 'human')).toBe('mouse')
    expect(pickSpecies({ species: null }, 'mouse')).toBe('mouse')
    expect(pickSpecies(null, null)).toBe('human')
    expect(pickSpecies(null, 'giraffe')).toBe('human')
  })
  it('names the folder after the library and its source', () => {
    expect(defaultFolderName('Hallmark', 'msigdb')).toBe('Hallmark (MSigDB)')
    expect(defaultFolderName('GO Biological Process 2026', 'enrichr')).toBe('GO Biological Process 2026 (Enrichr)')
  })
})
