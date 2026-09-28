import { describe, it, expect } from 'vitest'
import {
  parseGeneText,
  makeGeneIndex,
  resolveGenes,
  mergeGeneLists,
  mergeGeneSets,
  minCountFor,
  suggestMergeName,
  setKey,
  folderKey,
  EMPTY_SELECTION,
  toggleSet,
  toggleFolder,
  toggleCategory,
  folderState,
  categoryState,
  selectAllShown,
  deleteSelection,
  selectionSummary,
  gatherSelectedSets,
  filterTree,
  insertGeneSet,
  addFolder,
  updateGeneSet,
  findGeneSet,
  countGeneSets,
  destinationOptions,
  destinationValue,
  destinationFromValue,
  NEW_FOLDER,
  MANAGER_ORDER,
  type Selection,
} from './geneSetManager'
import { createDefaultCategories, type GeneSet, type GeneSetCategoryType } from '../store'

const ORDER: GeneSetCategoryType[] = ['manual', 'gene_clusters', 'diff_exp']

function cats() {
  const c = createDefaultCategories()
  c.manual.geneSets = [
    { id: 's1', name: 'Chondro', genes: ['Sox9', 'Col2a1', 'Acan'] },
    { id: 's2', name: 'Empty', genes: [] },
  ]
  c.manual.folders = [
    {
      id: 'f1', name: 'Wnt (canonical)', expanded: true, createdAt: '',
      geneSets: [
        { id: 'w1', name: 'ligands', genes: ['Wnt1', 'Wnt3a'] },
        { id: 'w2', name: 'targets', genes: ['Axin2', 'Lef1'], genesDown: ['Dkk1'] },
      ],
    },
    {
      id: 'f2', name: 'BMP', expanded: false, createdAt: '',
      geneSets: [{ id: 'b1', name: 'ligands', genes: ['Bmp2', 'Bmp4'] }],
    },
    { id: 'f3', name: 'Empty folder', expanded: false, createdAt: '', geneSets: [] },
  ]
  c.gene_clusters.folders = [
    { id: 'g1', name: 'leiden run', expanded: true, createdAt: '', geneSets: [{ id: 'm1', name: 'module 1', genes: ['Sox9', 'Wnt1'] }] },
  ]
  return c
}

const sk = (cat: GeneSetCategoryType, folderId: string | null, setId: string) => setKey({ cat, folderId, setId })
const fk = (cat: GeneSetCategoryType, folderId: string) => folderKey({ cat, folderId })

// --- parsing -------------------------------------------------------------------

describe('parseGeneText', () => {
  it('splits on whitespace, commas, semicolons, tabs and new lines', () => {
    expect(parseGeneText('Sox9, Col2a1;Acan\tMeis1\nMeis2  Pbx1').genes)
      .toEqual(['Sox9', 'Col2a1', 'Acan', 'Meis1', 'Meis2', 'Pbx1'])
  })

  it('strips quotes and list bullets but keeps hyphens inside names', () => {
    expect(parseGeneText('"Sox9"\n- H2-Ab1\n• mt-Co1\n* Acan\n\'Wnt5a\'').genes)
      .toEqual(['Sox9', 'H2-Ab1', 'mt-Co1', 'Acan', 'Wnt5a'])
  })

  it('drops enumerators from numbered lists', () => {
    expect(parseGeneText('1. Sox9\n2) Acan\n10. Col2a1').genes).toEqual(['Sox9', 'Acan', 'Col2a1'])
  })

  it('drops exact duplicates, keeps first-seen order, and counts them', () => {
    const r = parseGeneText('Sox9 Acan Sox9 Acan Sox9')
    expect(r.genes).toEqual(['Sox9', 'Acan'])
    expect(r.duplicates).toBe(3)
  })

  it('returns nothing for blank input', () => {
    expect(parseGeneText('  \n , ; ').genes).toEqual([])
  })
})

describe('resolveGenes', () => {
  const index = makeGeneIndex(['Sox9', 'Acan', 'Col2a1', 'H2-Ab1'])

  it('keeps exact matches and stores case-insensitive ones in the dataset spelling', () => {
    const r = resolveGenes(['Sox9', 'ACAN', 'col2a1'], index, false)
    expect(r.genes).toEqual(['Sox9', 'Acan', 'Col2a1'])
    expect(r.found).toBe(3)
    expect(r.recased).toEqual([{ from: 'ACAN', to: 'Acan' }, { from: 'col2a1', to: 'Col2a1' }])
    expect(r.missing).toEqual([])
    expect(r.checked).toBe(true)
  })

  it('drops genes the dataset lacks unless told to keep them', () => {
    expect(resolveGenes(['Sox9', 'Foo1'], index, false).genes).toEqual(['Sox9'])
    const kept = resolveGenes(['Sox9', 'Foo1'], index, true)
    expect(kept.genes).toEqual(['Sox9', 'Foo1'])
    expect(kept.missing).toEqual(['Foo1'])
  })

  it('collapses two spellings of one gene into a single entry', () => {
    expect(resolveGenes(['Sox9', 'SOX9', 'sox9'], index, false).genes).toEqual(['Sox9'])
  })

  it('prefers an exact match when the dataset has names differing only by case', () => {
    const tricky = makeGeneIndex(['Mt1', 'MT1'])
    expect(resolveGenes(['MT1', 'Mt1'], tricky, false).genes).toEqual(['MT1', 'Mt1'])
  })

  it('keeps everything as typed when there is no dataset to check against', () => {
    const r = resolveGenes(['Sox9', 'whatever'], null, false)
    expect(r.genes).toEqual(['Sox9', 'whatever'])
    expect(r.checked).toBe(false)
    expect(r.missing).toEqual([])
  })
})

// --- merging -------------------------------------------------------------------

describe('mergeGeneLists', () => {
  const lists = [['A', 'B', 'C'], ['B', 'C', 'D'], ['C', 'D', 'E']]

  it('union (min count 1) keeps first-appearance order', () => {
    expect(mergeGeneLists(lists, 1)).toEqual(['A', 'B', 'C', 'D', 'E'])
  })

  it('intersection (min count n) keeps genes present in every list', () => {
    expect(mergeGeneLists(lists, 3)).toEqual(['C'])
  })

  it('at-least-k is a consensus between the two', () => {
    expect(mergeGeneLists(lists, 2)).toEqual(['B', 'C', 'D'])
  })

  it('counts a gene once per list even when a list repeats it', () => {
    expect(mergeGeneLists([['A', 'A'], ['B']], 2)).toEqual([])
  })

  it('clamps the minimum into 1..n and handles no lists', () => {
    expect(mergeGeneLists(lists, 0)).toEqual(['A', 'B', 'C', 'D', 'E'])
    expect(mergeGeneLists(lists, 9)).toEqual(['C'])
    expect(mergeGeneLists([], 1)).toEqual([])
  })
})

describe('minCountFor', () => {
  it('maps each rule onto a minimum count', () => {
    expect(minCountFor({ kind: 'union' }, 5)).toBe(1)
    expect(minCountFor({ kind: 'intersection' }, 5)).toBe(5)
    expect(minCountFor({ kind: 'atLeast', k: 3 }, 5)).toBe(3)
    expect(minCountFor({ kind: 'atLeast', k: 8 }, 5)).toBe(5)
  })
})

describe('mergeGeneSets', () => {
  it('merges up lists and down lists by the same rule', () => {
    const r = mergeGeneSets(
      [{ genes: ['A', 'B'], genesDown: ['X'] }, { genes: ['B', 'C'], genesDown: ['Y'] }],
      { kind: 'union' },
    )
    expect(r.genes).toEqual(['A', 'B', 'C'])
    expect(r.genesDown).toEqual(['X', 'Y'])
  })

  it('drops from the down list anything already in the up list', () => {
    const r = mergeGeneSets(
      [{ genes: ['A'], genesDown: ['B'] }, { genes: ['B'] }],
      { kind: 'union' },
    )
    expect(r.genes).toEqual(['A', 'B'])
    expect(r.genesDown).toBeUndefined()
  })

  it('an intersection with a set lacking a down list has no down list', () => {
    const r = mergeGeneSets(
      [{ genes: ['A', 'B'], genesDown: ['X'] }, { genes: ['A'] }],
      { kind: 'intersection' },
    )
    expect(r.genes).toEqual(['A'])
    expect(r.genesDown).toBeUndefined()
  })
})

describe('suggestMergeName', () => {
  it('spells out up to three names and counts beyond that', () => {
    expect(suggestMergeName(['a', 'b', 'c'], { kind: 'union' })).toBe('a ∪ b ∪ c')
    expect(suggestMergeName(['a', 'b'], { kind: 'intersection' })).toBe('a ∩ b')
    expect(suggestMergeName(['a', 'b', 'c', 'd'], { kind: 'union' })).toBe('Union of 4 sets')
    expect(suggestMergeName(['a', 'b', 'c', 'd'], { kind: 'intersection' })).toBe('Intersection of 4 sets')
    expect(suggestMergeName(['a', 'b', 'c', 'd'], { kind: 'atLeast', k: 2 })).toBe('In ≥2 of 4 sets')
  })
})

// --- selection -----------------------------------------------------------------

describe('selection', () => {
  it('ticking a folder ticks it and every set inside', () => {
    const c = cats()
    const sel = toggleFolder(EMPTY_SELECTION, 'manual', c.manual.folders[0])
    expect(sel.folders.has(fk('manual', 'f1'))).toBe(true)
    expect(sel.sets.has(sk('manual', 'f1', 'w1'))).toBe(true)
    expect(sel.sets.has(sk('manual', 'f1', 'w2'))).toBe(true)
    expect(folderState(c.manual.folders[0], 'manual', sel)).toBe('all')
    // and again unticks all of it
    const off = toggleFolder(sel, 'manual', c.manual.folders[0])
    expect(off.folders.size + off.sets.size).toBe(0)
  })

  it('unticking one set inside a ticked folder unticks the folder', () => {
    const c = cats()
    let sel = toggleFolder(EMPTY_SELECTION, 'manual', c.manual.folders[0])
    sel = toggleSet(sel, 'manual', c.manual.folders[0], 'w1')
    expect(sel.folders.has(fk('manual', 'f1'))).toBe(false)
    expect(sel.sets.has(sk('manual', 'f1', 'w2'))).toBe(true)
    expect(folderState(c.manual.folders[0], 'manual', sel)).toBe('some')
  })

  it('ticking every set of a folder one by one ticks the folder too', () => {
    const c = cats()
    let sel = toggleSet(EMPTY_SELECTION, 'manual', c.manual.folders[0], 'w1')
    expect(sel.folders.size).toBe(0)
    sel = toggleSet(sel, 'manual', c.manual.folders[0], 'w2')
    expect(sel.folders.has(fk('manual', 'f1'))).toBe(true)
  })

  it('an empty folder can be ticked on its own', () => {
    const c = cats()
    const sel = toggleFolder(EMPTY_SELECTION, 'manual', c.manual.folders[2])
    expect(folderState(c.manual.folders[2], 'manual', sel)).toBe('all')
  })

  it('a top-level set toggles without touching folders', () => {
    let sel = toggleSet(EMPTY_SELECTION, 'manual', null, 's1')
    expect(sel.sets.has(sk('manual', null, 's1'))).toBe(true)
    sel = toggleSet(sel, 'manual', null, 's1')
    expect(sel.sets.size).toBe(0)
  })

  it('a category tick covers its folders and sets', () => {
    const c = cats()
    const sel = toggleCategory(EMPTY_SELECTION, c.manual)
    expect(categoryState(c.manual, sel)).toBe('all')
    expect(sel.folders.size).toBe(3)
    expect(sel.sets.size).toBe(5)
    expect(categoryState(c.gene_clusters, sel)).toBe('none')
    expect(categoryState(c.manual, toggleCategory(sel, c.manual))).toBe('none')
  })
})

// --- delete --------------------------------------------------------------------

describe('deleteSelection', () => {
  it('removes ticked folders whole and ticked sets individually in one pass', () => {
    const c = cats()
    let sel: Selection = toggleFolder(EMPTY_SELECTION, 'manual', c.manual.folders[0])
    sel = toggleFolder(sel, 'manual', c.manual.folders[1])
    sel = toggleSet(sel, 'manual', null, 's2')
    sel = toggleSet(sel, 'gene_clusters', c.gene_clusters.folders[0], 'm1')
    const out = deleteSelection(c, sel)
    expect(out.manual.folders.map((f) => f.id)).toEqual(['f3'])
    expect(out.manual.geneSets.map((s) => s.id)).toEqual(['s1'])
    // Ticking the only set of a folder ticks the folder, so it goes too.
    expect(out.gene_clusters.folders).toEqual([])
  })

  it('leaves untouched categories as the same objects', () => {
    const c = cats()
    const out = deleteSelection(c, toggleSet(EMPTY_SELECTION, 'manual', null, 's1'))
    expect(out.gene_clusters).toBe(c.gene_clusters)
    expect(out.manual).not.toBe(c.manual)
  })

  it('summarises what a delete will remove', () => {
    const c = cats()
    let sel = toggleFolder(EMPTY_SELECTION, 'manual', c.manual.folders[0])
    sel = toggleFolder(sel, 'manual', c.manual.folders[2])
    sel = toggleSet(sel, 'manual', null, 's1')
    sel = toggleSet(sel, 'manual', c.manual.folders[1], 'b1') // BMP's only set → BMP ticked too
    expect(selectionSummary(c, ORDER, sel)).toEqual({ folders: 3, setsInFolders: 3, looseSets: 1 })
  })
})

describe('gatherSelectedSets', () => {
  it('returns ticked sets in tree order with a readable path', () => {
    const c = cats()
    let sel = toggleSet(EMPTY_SELECTION, 'gene_clusters', c.gene_clusters.folders[0], 'm1')
    sel = toggleSet(sel, 'manual', null, 's1')
    sel = toggleSet(sel, 'manual', c.manual.folders[0], 'w2')
    const got = gatherSelectedSets(c, ORDER, sel)
    expect(got.map((g) => g.set.id)).toEqual(['w2', 's1', 'm1'])
    expect(got[0].path).toBe('Manual › Wnt (canonical)')
    expect(got[1].path).toBe('Manual')
  })
})

// --- filter --------------------------------------------------------------------

describe('filterTree', () => {
  it('with no query shows everything, non-empty categories only (Manual always)', () => {
    const t = filterTree(cats(), ORDER, '')
    expect(t.map((x) => x.cat)).toEqual(['manual', 'gene_clusters'])
    expect(t[0].folders).toHaveLength(3)
    expect(t[0].sets.map((s) => s.id)).toEqual(['s1', 's2'])
  })

  it('a folder-name match shows the whole folder', () => {
    const t = filterTree(cats(), ORDER, 'wnt (')
    expect(t).toHaveLength(1)
    expect(t[0].folders).toHaveLength(1)
    expect(t[0].folders[0].sets.map((s) => s.id)).toEqual(['w1', 'w2'])
    expect(t[0].folders[0].whole).toBe(true)
  })

  it('a set-name or gene match shows just those sets', () => {
    const t = filterTree(cats(), ORDER, 'ligands')
    expect(t[0].folders.map((f) => f.folder.id)).toEqual(['f1', 'f2'])
    expect(t[0].folders[0].sets.map((s) => s.id)).toEqual(['w1'])
    expect(t[0].folders[0].whole).toBe(false)

    const bySym = filterTree(cats(), ORDER, 'sox9')
    expect(bySym.map((x) => x.cat)).toEqual(['manual', 'gene_clusters'])
    expect(bySym[0].sets.map((s) => s.id)).toEqual(['s1'])
    expect(bySym[1].folders[0].sets.map((s) => s.id)).toEqual(['m1'])
  })

  it('matches down genes too', () => {
    const t = filterTree(cats(), ORDER, 'dkk1')
    expect(t[0].folders[0].sets.map((s) => s.id)).toEqual(['w2'])
  })

  it('select-all-shown ticks whole folders as folders and partial ones set by set', () => {
    const c = cats()
    const sel = selectAllShown(EMPTY_SELECTION, filterTree(c, ORDER, 'ligands'))
    expect(sel.sets.has(sk('manual', 'f1', 'w1'))).toBe(true)
    expect(sel.sets.has(sk('manual', 'f1', 'w2'))).toBe(false)
    expect(sel.folders.has(fk('manual', 'f1'))).toBe(false)
    // BMP's only set matched, so BMP is shown whole and ticked as a folder.
    expect(sel.folders.has(fk('manual', 'f2'))).toBe(true)
  })
})

// --- insert / update -----------------------------------------------------------

describe('insertGeneSet / addFolder / updateGeneSet', () => {
  const gs = (id: string): GeneSet => ({ id, name: 'new', genes: ['A'] })

  it('adds to a category top level or to a folder', () => {
    let c = insertGeneSet(cats(), { cat: 'diff_exp', folderId: null }, gs('n1'))
    expect(c.diff_exp.geneSets.map((s) => s.id)).toEqual(['n1'])
    c = insertGeneSet(c, { cat: 'manual', folderId: 'f2' }, gs('n2'))
    expect(c.manual.folders[1].geneSets.map((s) => s.id)).toEqual(['b1', 'n2'])
  })

  it('refuses a folder that is not there', () => {
    expect(() => insertGeneSet(cats(), { cat: 'manual', folderId: 'nope' }, gs('n1'))).toThrow(/folder/i)
  })

  it('addFolder appends a folder that insertGeneSet can then target', () => {
    let c = addFolder(cats(), 'manual', { id: 'nf', name: 'Pasted', expanded: true, createdAt: 't', geneSets: [] })
    c = insertGeneSet(c, { cat: 'manual', folderId: 'nf' }, gs('n1'))
    expect(c.manual.folders[c.manual.folders.length - 1].geneSets.map((s) => s.id)).toEqual(['n1'])
  })

  it('updates a set wherever it lives, and an empty down list removes it', () => {
    const key = sk('manual', 'f1', 'w2')
    const c = updateGeneSet(cats(), key, { name: 'readouts', genes: ['Axin2'], genesDown: [] })
    const found = findGeneSet(c, key)
    expect(found?.name).toBe('readouts')
    expect(found?.genes).toEqual(['Axin2'])
    expect(found?.genesDown).toBeUndefined()
    expect('genesDown' in (found ?? {})).toBe(false)
  })

  it('findGeneSet returns null for a key that is gone', () => {
    expect(findGeneSet(cats(), sk('manual', null, 'zzz'))).toBeNull()
  })
})

describe('countGeneSets', () => {
  it('counts sets, folders and non-empty categories', () => {
    expect(countGeneSets(cats())).toEqual({ sets: 6, folders: 4, categories: 2 })
    expect(countGeneSets(createDefaultCategories())).toEqual({ sets: 0, folders: 0, categories: 0 })
  })
})

describe('destinations', () => {
  it('offers Manual first (top level, its folders, a new folder), then non-empty categories', () => {
    const opts = destinationOptions(cats(), MANAGER_ORDER)
    expect(opts.map((o) => o.label)).toEqual([
      'Manual (top level)',
      'Manual › Wnt (canonical)',
      'Manual › BMP',
      'Manual › Empty folder',
      'New folder in Manual…',
      'Gene Clusters (top level)',
      'Gene Clusters › leiden run',
    ])
    expect(opts[4].value).toBe(NEW_FOLDER)
  })

  it('round-trips a destination through its option value', () => {
    for (const d of [{ cat: 'manual' as const, folderId: null }, { cat: 'gene_clusters' as const, folderId: 'g1' }]) {
      expect(destinationFromValue(destinationValue(d))).toEqual(d)
    }
    expect(destinationFromValue(NEW_FOLDER)).toBeNull()
  })

  it('MANAGER_ORDER covers every category, including ones the panel does not list', () => {
    expect([...MANAGER_ORDER].sort()).toEqual(Object.keys(createDefaultCategories()).sort())
  })
})
