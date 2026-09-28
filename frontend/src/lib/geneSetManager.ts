/** Pure operations behind the Gene Set Manager.
 *
 * Everything here maps categories → categories (or gene lists → gene list), so
 * the modal is wiring and the behaviour is pinned by `geneSetManager.test.ts`.
 * Ids are made by the caller: the store owns the id counters, and a lib that
 * minted its own would collide with them.
 */
import type {
  GeneSet,
  GeneSetCategory,
  GeneSetCategoryType,
  GeneSetFolder,
} from '../store'

export type Categories = Record<GeneSetCategoryType, GeneSetCategory>

// ---------------------------------------------------------------------------
// Pasted text → gene symbols

// A numbered list's "1." / "2)" — never a gene symbol.
const ENUMERATOR = /^\d+[.)]$/
// Quotes and brackets around a symbol, and bullets before one. A hyphen is
// only stripped as a leading bullet: H2-Ab1 and mt-Co1 keep theirs.
const LEADING = /^["'`‘’“”•·*\-–—([{]+/
const TRAILING = /["'`‘’“”)\]}]+$/

export function parseGeneText(text: string): { genes: string[]; duplicates: number } {
  const seen = new Set<string>()
  const genes: string[] = []
  let duplicates = 0
  for (const raw of text.split(/[\s,;|]+/)) {
    if (ENUMERATOR.test(raw)) continue
    const tok = raw.replace(LEADING, '').replace(TRAILING, '')
    if (!tok) continue
    if (seen.has(tok)) { duplicates++; continue }
    seen.add(tok)
    genes.push(tok)
  }
  return { genes, duplicates }
}

export interface GeneIndex {
  exact: Set<string>
  // lower-case → the dataset's first spelling of it
  lower: Map<string, string>
}

export function makeGeneIndex(universe: readonly string[]): GeneIndex {
  const exact = new Set(universe)
  const lower = new Map<string, string>()
  for (const g of universe) {
    const k = g.toLowerCase()
    if (!lower.has(k)) lower.set(k, g)
  }
  return { exact, lower }
}

export interface GeneResolution {
  genes: string[]                            // what the set will hold
  found: number                              // distinct dataset genes matched
  recased: { from: string; to: string }[]    // matched only ignoring case
  missing: string[]                          // not in the dataset at all
  checked: boolean                           // false: no dataset to check against
}

/** Match typed symbols against the dataset's genes.
 *
 * Case-insensitive matches are stored in the dataset's own spelling — a set
 * holding `SOX9` in a mouse dataset would colour nothing. An exact match wins,
 * so a dataset with both `Mt1` and `MT1` keeps them apart.
 */
export function resolveGenes(
  tokens: string[],
  index: GeneIndex | null,
  keepMissing: boolean,
): GeneResolution {
  if (!index) {
    return { genes: Array.from(new Set(tokens)), found: 0, recased: [], missing: [], checked: false }
  }
  const out = new Set<string>()
  const recased: { from: string; to: string }[] = []
  const missing: string[] = []
  let found = 0
  for (const tok of tokens) {
    let hit: string | undefined
    if (index.exact.has(tok)) {
      hit = tok
    } else {
      hit = index.lower.get(tok.toLowerCase())
      if (hit) recased.push({ from: tok, to: hit })
    }
    if (hit) {
      if (!out.has(hit)) found++
      out.add(hit)
    } else {
      missing.push(tok)
      if (keepMissing) out.add(tok)
    }
  }
  return { genes: Array.from(out), found, recased, missing, checked: true }
}

// ---------------------------------------------------------------------------
// Merging

export type MergeRule =
  | { kind: 'union' }
  | { kind: 'intersection' }
  | { kind: 'atLeast'; k: number }

/** Genes present in at least `minCount` of the lists, in first-appearance order.
 *  1 is the union, lists.length the intersection. A gene repeated inside one
 *  list still counts once for it. */
export function mergeGeneLists(lists: string[][], minCount: number): string[] {
  if (lists.length === 0) return []
  const need = Math.min(Math.max(1, Math.floor(minCount)), lists.length)
  const counts = new Map<string, number>()
  const order: string[] = []
  for (const list of lists) {
    for (const g of new Set(list)) {
      const c = counts.get(g)
      if (c === undefined) { counts.set(g, 1); order.push(g) } else counts.set(g, c + 1)
    }
  }
  return order.filter((g) => (counts.get(g) ?? 0) >= need)
}

export function minCountFor(rule: MergeRule, n: number): number {
  switch (rule.kind) {
    case 'union': return 1
    case 'intersection': return n
    case 'atLeast': return Math.min(Math.max(1, rule.k), n)
  }
}

/** Merge whole sets. Down lists follow the same rule — a set without one
 *  counts as an empty one — and lose anything that ended up in the up list,
 *  since a gene cannot pull a score both ways. */
export function mergeGeneSets(
  sets: Pick<GeneSet, 'genes' | 'genesDown'>[],
  rule: MergeRule,
): { genes: string[]; genesDown?: string[] } {
  const need = minCountFor(rule, sets.length)
  const genes = mergeGeneLists(sets.map((s) => s.genes), need)
  const up = new Set(genes)
  const down = mergeGeneLists(sets.map((s) => s.genesDown ?? []), need).filter((g) => !up.has(g))
  return down.length > 0 ? { genes, genesDown: down } : { genes }
}

export function suggestMergeName(names: string[], rule: MergeRule): string {
  const n = names.length
  if (rule.kind === 'atLeast') return `In ≥${minCountFor(rule, n)} of ${n} sets`
  if (n <= 3) return names.join(rule.kind === 'union' ? ' ∪ ' : ' ∩ ')
  return `${rule.kind === 'union' ? 'Union' : 'Intersection'} of ${n} sets`
}

// ---------------------------------------------------------------------------
// Addressing and selection

export interface SetRef { cat: GeneSetCategoryType; folderId: string | null; setId: string }
export interface FolderRef { cat: GeneSetCategoryType; folderId: string }

export const setKey = (r: SetRef) => `${r.cat}|${r.folderId ?? ''}|${r.setId}`
export const folderKey = (r: FolderRef) => `${r.cat}|${r.folderId}`

/** Ticked folders and sets. A ticked folder's sets are always ticked with it,
 *  so "which sets" never has to look at folders — and deleting a ticked
 *  folder removes the folder itself, not just what is in it. */
export interface Selection {
  sets: ReadonlySet<string>
  folders: ReadonlySet<string>
}

export const EMPTY_SELECTION: Selection = { sets: new Set(), folders: new Set() }

export type TickState = 'none' | 'some' | 'all'

const folderSetKeys = (cat: GeneSetCategoryType, f: GeneSetFolder) =>
  f.geneSets.map((gs) => setKey({ cat, folderId: f.id, setId: gs.id }))

export function folderState(folder: GeneSetFolder, cat: GeneSetCategoryType, sel: Selection): TickState {
  if (sel.folders.has(folderKey({ cat, folderId: folder.id }))) return 'all'
  return folderSetKeys(cat, folder).some((k) => sel.sets.has(k)) ? 'some' : 'none'
}

export function toggleFolder(sel: Selection, cat: GeneSetCategoryType, folder: GeneSetFolder): Selection {
  const fk = folderKey({ cat, folderId: folder.id })
  const sets = new Set(sel.sets)
  const folders = new Set(sel.folders)
  if (folders.has(fk)) {
    folders.delete(fk)
    for (const k of folderSetKeys(cat, folder)) sets.delete(k)
  } else {
    folders.add(fk)
    for (const k of folderSetKeys(cat, folder)) sets.add(k)
  }
  return { sets, folders }
}

/** Tick or untick one set. Unticking a set inside a ticked folder unticks the
 *  folder (it is no longer wholly chosen); ticking the last unticked set of a
 *  folder ticks the folder, so ticking every set one by one means the same as
 *  ticking the folder. */
export function toggleSet(
  sel: Selection,
  cat: GeneSetCategoryType,
  folder: GeneSetFolder | null,
  setId: string,
): Selection {
  const k = setKey({ cat, folderId: folder?.id ?? null, setId })
  const sets = new Set(sel.sets)
  const folders = new Set(sel.folders)
  if (sets.has(k)) {
    sets.delete(k)
    if (folder) folders.delete(folderKey({ cat, folderId: folder.id }))
  } else {
    sets.add(k)
    if (folder && folderSetKeys(cat, folder).every((fk) => sets.has(fk))) {
      folders.add(folderKey({ cat, folderId: folder.id }))
    }
  }
  return { sets, folders }
}

function categoryKeys(category: GeneSetCategory) {
  const cat = category.type
  return {
    folders: category.folders.map((f) => folderKey({ cat, folderId: f.id })),
    sets: [
      ...category.folders.flatMap((f) => folderSetKeys(cat, f)),
      ...category.geneSets.map((gs) => setKey({ cat, folderId: null, setId: gs.id })),
    ],
  }
}

export function categoryState(category: GeneSetCategory, sel: Selection): TickState {
  const { folders, sets } = categoryKeys(category)
  if (folders.length + sets.length === 0) return 'none'
  const ticked = folders.filter((k) => sel.folders.has(k)).length + sets.filter((k) => sel.sets.has(k)).length
  if (ticked === 0) return 'none'
  return ticked === folders.length + sets.length ? 'all' : 'some'
}

export function toggleCategory(sel: Selection, category: GeneSetCategory): Selection {
  const { folders: fks, sets: sks } = categoryKeys(category)
  const sets = new Set(sel.sets)
  const folders = new Set(sel.folders)
  if (categoryState(category, sel) === 'all') {
    for (const k of fks) folders.delete(k)
    for (const k of sks) sets.delete(k)
  } else {
    for (const k of fks) folders.add(k)
    for (const k of sks) sets.add(k)
  }
  return { sets, folders }
}

// ---------------------------------------------------------------------------
// Filtering

export interface FilteredFolder {
  folder: GeneSetFolder
  key: string
  sets: GeneSet[]   // the sets shown
  whole: boolean    // every set shown (folder-name match, or all matched)
}

export interface FilteredCategory {
  cat: GeneSetCategoryType
  category: GeneSetCategory
  folders: FilteredFolder[]
  sets: GeneSet[]
}

/** What the tree shows for a query. A folder whose name matches shows whole;
 *  otherwise a set shows when its name contains the query or one of its genes
 *  (up or down) starts with it. Empty categories are left out, except Manual
 *  with no query — it is where new sets go. */
export function filterTree(cats: Categories, order: GeneSetCategoryType[], query: string): FilteredCategory[] {
  const q = query.trim().toLowerCase()
  const setMatches = (gs: GeneSet) =>
    gs.name.toLowerCase().includes(q)
    || gs.genes.some((g) => g.toLowerCase().startsWith(q))
    || (gs.genesDown ?? []).some((g) => g.toLowerCase().startsWith(q))

  const out: FilteredCategory[] = []
  for (const cat of order) {
    const category = cats[cat]
    if (!category) continue
    if (!q) {
      if (cat !== 'manual' && category.folders.length === 0 && category.geneSets.length === 0) continue
      out.push({
        cat, category,
        folders: category.folders.map((folder) => ({
          folder, key: folderKey({ cat, folderId: folder.id }), sets: folder.geneSets, whole: true,
        })),
        sets: category.geneSets,
      })
      continue
    }
    const folders: FilteredFolder[] = []
    for (const folder of category.folders) {
      const key = folderKey({ cat, folderId: folder.id })
      if (folder.name.toLowerCase().includes(q)) {
        folders.push({ folder, key, sets: folder.geneSets, whole: true })
        continue
      }
      const sets = folder.geneSets.filter(setMatches)
      if (sets.length > 0) folders.push({ folder, key, sets, whole: sets.length === folder.geneSets.length })
    }
    const sets = category.geneSets.filter(setMatches)
    if (folders.length > 0 || sets.length > 0) out.push({ cat, category, folders, sets })
  }
  return out
}

/** Tick everything the filter shows: folders shown whole as folders, the rest
 *  set by set — so a filtered bulk delete never takes a set the user could not see. */
export function selectAllShown(sel: Selection, tree: FilteredCategory[]): Selection {
  const sets = new Set(sel.sets)
  const folders = new Set(sel.folders)
  for (const c of tree) {
    for (const f of c.folders) {
      if (f.whole) folders.add(f.key)
      for (const gs of f.sets) sets.add(setKey({ cat: c.cat, folderId: f.folder.id, setId: gs.id }))
    }
    for (const gs of c.sets) sets.add(setKey({ cat: c.cat, folderId: null, setId: gs.id }))
  }
  return { sets, folders }
}

// ---------------------------------------------------------------------------
// Reading the selection

export function selectionSummary(
  cats: Categories,
  order: GeneSetCategoryType[],
  sel: Selection,
): { folders: number; setsInFolders: number; looseSets: number } {
  let folders = 0
  let setsInFolders = 0
  let looseSets = 0
  for (const cat of order) {
    const category = cats[cat]
    if (!category) continue
    for (const f of category.folders) {
      if (sel.folders.has(folderKey({ cat, folderId: f.id }))) {
        folders++
        setsInFolders += f.geneSets.length
      } else {
        looseSets += folderSetKeys(cat, f).filter((k) => sel.sets.has(k)).length
      }
    }
    looseSets += category.geneSets.filter((gs) => sel.sets.has(setKey({ cat, folderId: null, setId: gs.id }))).length
  }
  return { folders, setsInFolders, looseSets }
}

export interface SelectedSet {
  key: string
  cat: GeneSetCategoryType
  folderId: string | null
  set: GeneSet
  path: string   // "Manual › Wnt (canonical)"
}

/** Every ticked set, in the order the tree draws them (folders, then a
 *  category's own sets). */
export function gatherSelectedSets(cats: Categories, order: GeneSetCategoryType[], sel: Selection): SelectedSet[] {
  const out: SelectedSet[] = []
  for (const cat of order) {
    const category = cats[cat]
    if (!category) continue
    for (const f of category.folders) {
      for (const gs of f.geneSets) {
        const key = setKey({ cat, folderId: f.id, setId: gs.id })
        if (sel.sets.has(key)) out.push({ key, cat, folderId: f.id, set: gs, path: `${category.name} › ${f.name}` })
      }
    }
    for (const gs of category.geneSets) {
      const key = setKey({ cat, folderId: null, setId: gs.id })
      if (sel.sets.has(key)) out.push({ key, cat, folderId: null, set: gs, path: category.name })
    }
  }
  return out
}

export function countGeneSets(cats: Categories): { sets: number; folders: number; categories: number } {
  let sets = 0
  let folders = 0
  let categories = 0
  for (const c of Object.values(cats)) {
    const n = c.geneSets.length + c.folders.reduce((s, f) => s + f.geneSets.length, 0)
    sets += n
    folders += c.folders.length
    if (n > 0 || c.folders.length > 0) categories++
  }
  return { sets, folders, categories }
}

// ---------------------------------------------------------------------------
// Edits (untouched categories and folders keep their identity)

export function deleteSelection(cats: Categories, sel: Selection): Categories {
  const out = { ...cats }
  for (const cat of Object.keys(cats) as GeneSetCategoryType[]) {
    const category = cats[cat]
    let changed = false
    const folders: GeneSetFolder[] = []
    for (const f of category.folders) {
      if (sel.folders.has(folderKey({ cat, folderId: f.id }))) { changed = true; continue }
      const kept = f.geneSets.filter((gs) => !sel.sets.has(setKey({ cat, folderId: f.id, setId: gs.id })))
      if (kept.length !== f.geneSets.length) { changed = true; folders.push({ ...f, geneSets: kept }) } else folders.push(f)
    }
    const geneSets = category.geneSets.filter((gs) => !sel.sets.has(setKey({ cat, folderId: null, setId: gs.id })))
    if (geneSets.length !== category.geneSets.length) changed = true
    if (changed) out[cat] = { ...category, folders, geneSets }
  }
  return out
}

export interface Destination { cat: GeneSetCategoryType; folderId: string | null }

export function insertGeneSet(cats: Categories, dest: Destination, set: GeneSet): Categories {
  const category = cats[dest.cat]
  if (dest.folderId === null) {
    return { ...cats, [dest.cat]: { ...category, geneSets: [...category.geneSets, set] } }
  }
  if (!category.folders.some((f) => f.id === dest.folderId)) {
    throw new Error('That folder no longer exists — pick another destination.')
  }
  return {
    ...cats,
    [dest.cat]: {
      ...category,
      folders: category.folders.map((f) => (f.id === dest.folderId ? { ...f, geneSets: [...f.geneSets, set] } : f)),
    },
  }
}

export function addFolder(cats: Categories, cat: GeneSetCategoryType, folder: GeneSetFolder): Categories {
  return { ...cats, [cat]: { ...cats[cat], folders: [...cats[cat].folders, folder] } }
}

export function findGeneSet(cats: Categories, key: string): GeneSet | null {
  for (const cat of Object.keys(cats) as GeneSetCategoryType[]) {
    const category = cats[cat]
    for (const f of category.folders) {
      for (const gs of f.geneSets) if (setKey({ cat, folderId: f.id, setId: gs.id }) === key) return gs
    }
    for (const gs of category.geneSets) if (setKey({ cat, folderId: null, setId: gs.id }) === key) return gs
  }
  return null
}

/** Patch one set. An empty `genesDown` removes the down list altogether, so
 *  the set stops being directional rather than carrying an empty one. */
export function updateGeneSet(
  cats: Categories,
  key: string,
  patch: Partial<Pick<GeneSet, 'name' | 'genes' | 'genesDown'>>,
): Categories {
  const apply = (gs: GeneSet): GeneSet => {
    const next: GeneSet = { ...gs, ...patch }
    if (patch.genesDown !== undefined && patch.genesDown.length === 0) delete next.genesDown
    return next
  }
  const out = { ...cats }
  for (const cat of Object.keys(cats) as GeneSetCategoryType[]) {
    const category = cats[cat]
    if (category.geneSets.some((gs) => setKey({ cat, folderId: null, setId: gs.id }) === key)) {
      out[cat] = {
        ...category,
        geneSets: category.geneSets.map((gs) => (setKey({ cat, folderId: null, setId: gs.id }) === key ? apply(gs) : gs)),
      }
      return out
    }
    const fi = category.folders.findIndex((f) => f.geneSets.some((gs) => setKey({ cat, folderId: f.id, setId: gs.id }) === key))
    if (fi >= 0) {
      const f = category.folders[fi]
      const folders = [...category.folders]
      folders[fi] = { ...f, geneSets: f.geneSets.map((gs) => (setKey({ cat, folderId: f.id, setId: gs.id }) === key ? apply(gs) : gs)) }
      out[cat] = { ...category, folders }
      return out
    }
  }
  return cats
}

// ---------------------------------------------------------------------------
// Where a new set goes

/** Every category, in the order the manager lists them. Wider than the Genes
 *  pane's own list, which leaves out `spatial` — the manager is the one place
 *  sets filed there can be seen and cleaned up. */
export const MANAGER_ORDER: GeneSetCategoryType[] = [
  'manual', 'gene_clusters', 'similar_genes', 'diff_exp', 'spatial',
  'marker_genes', 'line_association', 'enrichment',
]

export const NEW_FOLDER = '__new_folder__'

export const destinationValue = (d: Destination) => `${d.cat}|${d.folderId ?? ''}`

export function destinationFromValue(v: string): Destination | null {
  if (v === NEW_FOLDER) return null
  const i = v.indexOf('|')
  const cat = v.slice(0, i) as GeneSetCategoryType
  const folderId = v.slice(i + 1)
  return { cat, folderId: folderId || null }
}

/** Manual — top level, each folder, a new folder — then every other category
 *  that already holds something. New folders are made in Manual only: the
 *  other categories are where tools file their results. */
export function destinationOptions(cats: Categories, order: GeneSetCategoryType[]): { value: string; label: string }[] {
  const out: { value: string; label: string }[] = []
  for (const cat of order) {
    const c = cats[cat]
    if (!c) continue
    if (cat !== 'manual' && c.folders.length === 0 && c.geneSets.length === 0) continue
    out.push({ value: destinationValue({ cat, folderId: null }), label: `${c.name} (top level)` })
    for (const f of c.folders) out.push({ value: destinationValue({ cat, folderId: f.id }), label: `${c.name} › ${f.name}` })
    if (cat === 'manual') out.push({ value: NEW_FOLDER, label: 'New folder in Manual…' })
  }
  return out
}
