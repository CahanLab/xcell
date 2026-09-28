import { describe, it, expect } from 'vitest'
import {
  sortLibraries, filterLibraries, attachOverlap, presenceLabel, importSets,
  pickSpecies, rowPasses, defaultFolderName, collectAllMatching, headerTickState, togglePick, pickRows, unpickNames,
  type LibraryEntry, type LibrarySet, type OverlapEntry, type SetRow,
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

// The set list is paged (50 a page), but a selection is not: the header box
// takes every set the search returned, and ticks survive paging.
describe('selecting across pages', () => {
  const librarySet = (i: number): LibrarySet => ({ name: `set${i}`, description: '', url: '', n_genes: 2, genes: [`G${i}`, 'SHARED'] })
  const all = Array.from({ length: 2345 }, (_, i) => librarySet(i))
  const overlapOf = (s: LibrarySet, i: number): OverlapEntry => ({
    name: s.name, n_genes: 2, n_present: i % 3 === 0 ? 0 : 2, n_exact: 2, n_case_insensitive: 0, n_missing: 0,
    genes_resolved: i % 3 === 0 ? [] : s.genes, genes_down_resolved: [], genes_missing: [], columns: {},
  })

  it('collectAllMatching pages through every match, checks overlap per chunk, and keeps library order', async () => {
    const pages: [number, number][] = []
    const overlapSizes: number[] = []
    const progress: number[] = []
    const rows = await collectAllMatching(
      all.length,
      async (offset, limit) => { pages.push([offset, limit]); return all.slice(offset, offset + limit) },
      async (sets) => { overlapSizes.push(sets.length); return sets.map((s) => ({ ...s, overlap: overlapOf(s, Number(s.name.slice(3))) })) },
      { minPresent: 0 },
      { chunk: 1000, onProgress: (done) => progress.push(done) },
    )
    expect(pages).toEqual([[0, 1000], [1000, 1000], [2000, 1000]])
    expect(overlapSizes).toEqual([1000, 1000, 345])
    expect(progress).toEqual([1000, 2000, 2345])
    expect(rows).toHaveLength(2345)
    expect(rows[0].name).toBe('set0')
    expect(rows[2344].name).toBe('set2344')
  })

  it('applies the row filter, so sets hidden by "≥ N present" are not taken', async () => {
    const rows = await collectAllMatching(
      all.length,
      async (offset, limit) => all.slice(offset, offset + limit),
      async (sets) => sets.map((s) => ({ ...s, overlap: overlapOf(s, Number(s.name.slice(3))) })),
      { minPresent: 1 },
    )
    expect(rows).toHaveLength(2345 - Math.ceil(2345 / 3))
    expect(rows.every((r) => (r.overlap?.n_present ?? 0) >= 1)).toBe(true)
  })

  it('stops early if the library holds fewer sets than the total said', async () => {
    const rows = await collectAllMatching(5000, async (offset, limit) => all.slice(offset, offset + limit), async (s) => s, { minPresent: 0 })
    expect(rows).toHaveLength(2345)
  })

  const row = (name: string): SetRow => ({ name, description: '', url: '', n_genes: 1, genes: ['A'] })

  it('picks are a name → row map that toggles and survives changing pages', () => {
    let picked = togglePick(new Map(), row('a'))
    picked = togglePick(picked, row('b'))
    expect([...picked.keys()]).toEqual(['a', 'b'])
    picked = togglePick(picked, row('a'))
    expect([...picked.keys()]).toEqual(['b'])
    // a later page adds to it rather than replacing it
    picked = pickRows(picked, [row('c'), row('d')])
    expect([...picked.keys()]).toEqual(['b', 'c', 'd'])
    expect([...unpickNames(picked, ['c', 'zzz']).keys()]).toEqual(['b', 'd'])
  })

  it('the header box is ticked only when every match is picked', () => {
    const page = [row('a'), row('b')]
    // one page holds every match
    expect(headerTickState(new Map(), page, page.map((r) => r.name))).toBe('none')
    expect(headerTickState(pickRows(new Map(), [row('a')]), page, page.map((r) => r.name))).toBe('some')
    expect(headerTickState(pickRows(new Map(), page), page, page.map((r) => r.name))).toBe('all')
    // several pages, not yet collected: a full page is only "some"
    expect(headerTickState(pickRows(new Map(), page), page, null)).toBe('some')
    // collected: all 3 matches picked
    expect(headerTickState(pickRows(new Map(), [...page, row('c')]), page, ['a', 'b', 'c'])).toBe('all')
    expect(headerTickState(new Map(), [], ['a'])).toBe('none')
  })
})
