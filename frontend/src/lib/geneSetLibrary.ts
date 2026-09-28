/**
 * Pure helpers for the Gene set library browser (external sources: MSigDB,
 * Enrichr, STRING). Everything the modal decides — ordering, filtering,
 * how overlap is shown, what an import contains — lives here so it can be
 * tested without a DOM.
 */
import type { GeneSetSource } from '../store'

export type Species = 'human' | 'mouse'

export const SOURCE_LABELS: Record<string, string> = {
  msigdb: 'MSigDB',
  enrichr: 'Enrichr',
  string: 'STRING',
}

/** One library in a source's catalogue (`GET /api/gene_set_sources/{source}/libraries`). */
export interface LibraryEntry {
  source: string
  id: string
  name: string
  description: string
  species: Species
  n_sets: number | null
  version: string | null
  url: string
  cached: boolean
  fetched_at: string | null
}

/** One set of a cached library as the search route returns it. */
export interface LibrarySet {
  name: string
  description: string
  url: string
  n_genes: number
  genes: string[]
  /** A directional set (CollecTRI regulon): its repressed targets. */
  genes_down?: string[]
}

/** One row of `POST /api/gene_sets/overlap`. */
export interface OverlapEntry {
  name: string
  n_genes: number
  n_present: number
  n_exact: number
  n_case_insensitive: number
  n_missing: number
  genes_resolved: string[]
  genes_down_resolved: string[]
  genes_missing: string[]
  columns: Record<string, number>
}

export interface SetRow extends LibrarySet {
  overlap?: OverlapEntry
}

/** Browsed species first, then libraries already on disk, then by name —
 * numerically where names carry numbers, so Theiler stages read TS1, TS2, …
 * TS10 rather than TS1, TS10, TS11. */
export function sortLibraries(entries: LibraryEntry[], species: Species): LibraryEntry[] {
  return [...entries].sort(
    (a, b) =>
      Number(a.species !== species) - Number(b.species !== species) ||
      Number(!a.cached) - Number(!b.cached) ||
      a.name.localeCompare(b.name, undefined, { numeric: true }),
  )
}

export function filterLibraries(entries: LibraryEntry[], text: string): LibraryEntry[] {
  const t = text.trim().toLowerCase()
  if (!t) return entries
  return entries.filter((e) =>
    [e.id, e.name, e.description].some((x) => (x || '').toLowerCase().includes(t)),
  )
}

/** The overlap route returns one entry per input set, in order. */
export function attachOverlap(sets: LibrarySet[], overlap: OverlapEntry[]): SetRow[] {
  return sets.map((s, i) => ({ ...s, overlap: overlap[i] }))
}

export function presenceLabel(row: SetRow): string {
  const o = row.overlap
  if (!o) return '…'
  const base = `${o.n_present} / ${o.n_genes}`
  return o.n_case_insensitive > 0 ? `${base} (case-insensitive)` : base
}

export interface RowFilter {
  minPresent: number
  column?: string
  minInColumn?: number
}

/** A row with no overlap yet is never filtered out, so rows don't vanish while the check is in flight. */
export function rowPasses(row: SetRow, f: RowFilter): boolean {
  const o = row.overlap
  if (!o) return true
  if (o.n_present < f.minPresent) return false
  const need = f.minInColumn ?? 0
  if (f.column && need > 0 && (o.columns[f.column] ?? 0) < need) return false
  return true
}

export interface ImportMeta {
  source: string
  library: string
  libraryName: string
  version: string | null
}

export interface ImportedSet {
  name: string
  genes: string[]
  genesDown?: string[]
  source: GeneSetSource
}

/**
 * What lands in the Gene Panel: the dataset's own spelling when overlap was
 * computed (so a human-symbol library imports as mouse symbols on mouse
 * data), the library's spelling otherwise. Sets with nothing present are
 * skipped — an empty set is invisible to every downstream picker.
 */
export function importSets(rows: SetRow[], meta: ImportMeta): ImportedSet[] {
  const out: ImportedSet[] = []
  for (const r of rows) {
    const o = r.overlap
    const genes = o ? o.genes_resolved : r.genes
    const down = o && o.genes_down_resolved.length ? o.genes_down_resolved : undefined
    if (genes.length === 0 && !down) continue
    const set: ImportedSet = {
      name: r.name,
      genes,
      source: { source: meta.source, library: meta.library, name: meta.libraryName, version: meta.version },
    }
    if (down) set.genesDown = down
    out.push(set)
  }
  return out
}

function isSpecies(x: unknown): x is Species {
  return x === 'human' || x === 'mouse'
}

/** Dataset guess wins; a remembered choice is next; human (most libraries) last. */
export function pickSpecies(guess: { species: string | null } | null, remembered: string | null): Species {
  if (guess && isSpecies(guess.species)) return guess.species
  if (isSpecies(remembered)) return remembered
  return 'human'
}

export function defaultFolderName(libraryName: string, source: string): string {
  return `${libraryName} (${SOURCE_LABELS[source] ?? source})`
}

// --- selection across pages ---------------------------------------------------
//
// The set list is paged, a selection is not. Picks are held as name → row so a
// set ticked on page 1 is still imported after moving to page 3, and the header
// box can take every set the search returned rather than the 50 on screen.

export type Picked = ReadonlyMap<string, SetRow>

export function togglePick(picked: Picked, row: SetRow): Map<string, SetRow> {
  const next = new Map(picked)
  if (next.has(row.name)) next.delete(row.name); else next.set(row.name, row)
  return next
}

export function pickRows(picked: Picked, rows: SetRow[]): Map<string, SetRow> {
  const next = new Map(picked)
  for (const r of rows) next.set(r.name, r)
  return next
}

export function unpickNames(picked: Picked, names: Iterable<string>): Map<string, SetRow> {
  const next = new Map(picked)
  for (const n of names) next.delete(n)
  return next
}

/** The header box. `complete` is every match of the current search that the
 *  row filter keeps — the page itself when one page holds them all, else the
 *  list gathered by `collectAllMatching`, or null when that is not known yet.
 *  Only a complete list can be "all": a full page of a longer search is "some". */
export function headerTickState(picked: Picked, page: SetRow[], complete: string[] | null): 'none' | 'some' | 'all' {
  if (complete && complete.length > 0 && complete.every((n) => picked.has(n))) return 'all'
  const any = page.some((r) => picked.has(r.name)) || (complete ?? []).some((n) => picked.has(n))
  return any ? 'some' : 'none'
}

/** Every set a search matched, with its overlap, passing the row filter.
 *
 *  Fetches and checks `chunk` sets at a time: one request per chunk keeps
 *  payloads bounded (a whole GO library is ~15,000 sets) and gives progress.
 *  Stops early if a page comes back short, so a stale `total` cannot loop. */
export async function collectAllMatching(
  total: number,
  fetchPage: (offset: number, limit: number) => Promise<LibrarySet[]>,
  overlap: (sets: LibrarySet[]) => Promise<SetRow[]>,
  filter: RowFilter,
  opts: { chunk?: number; onProgress?: (done: number, total: number) => void } = {},
): Promise<SetRow[]> {
  const chunk = Math.max(1, opts.chunk ?? 1000)
  const out: SetRow[] = []
  for (let offset = 0; offset < total; offset += chunk) {
    const sets = await fetchPage(offset, chunk)
    const rows = await overlap(sets)
    for (const r of rows) if (rowPasses(r, filter)) out.push(r)
    opts.onProgress?.(Math.min(offset + sets.length, total), total)
    if (sets.length < chunk) break
  }
  return out
}
