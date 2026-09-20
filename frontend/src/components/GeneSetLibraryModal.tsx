/**
 * GeneSetLibraryModal — browse external gene-set sources (MSigDB, Enrichr,
 * STRING), fetch a library into the local cache once, search it by name or
 * member gene, see how much of each set is present in the loaded dataset
 * (and how many members are highly / spatially variable), and import the
 * chosen sets into a Manual folder in the dataset's own gene spelling.
 *
 * Rollback: delete this file, its mount and the "Gene set library…" menu
 * item / Library button in GenePanel.tsx, `isGeneSetLibraryModalOpen` in
 * store.ts, and lib/geneSetLibrary.ts.
 */
import React, { useCallback, useEffect, useMemo, useState } from 'react'
import { useStore } from '../store'
import { appendDataset, fetchVarBooleanColumns, pollTask, type VarBooleanColumn } from '../hooks/useData'
import { flattenGeneSets } from './GenePanel'
import {
  attachOverlap, defaultFolderName, filterLibraries, importSets, pickSpecies, presenceLabel,
  rowPasses, sortLibraries,
  type LibraryEntry, type LibrarySet, type OverlapEntry, type SetRow, type Species,
} from '../lib/geneSetLibrary'

const API_BASE = '/api'
const PAGE = 50
const SPECIES_KEY = 'xcell_librarySpecies'

const dark = {
  panel: '#16213e', border: '#0f3460', inset: '#0f1625', accent: '#4ecdc4',
  alert: '#e94560', warn: '#e9a23b', muted: '#aaa', dim: '#888', text: '#ddd',
}

const field: React.CSSProperties = {
  background: dark.inset, color: dark.text, border: `1px solid ${dark.border}`,
  borderRadius: 4, padding: '4px 6px', fontSize: 12,
}
const btn: React.CSSProperties = {
  background: dark.accent, color: '#0f1625', border: 'none', borderRadius: 4,
  padding: '5px 12px', fontSize: 12, fontWeight: 600, cursor: 'pointer',
}
const btnGhost: React.CSSProperties = {
  background: 'transparent', color: dark.muted, border: `1px solid ${dark.border}`,
  borderRadius: 4, padding: '4px 10px', fontSize: 12, cursor: 'pointer',
}

interface SourceInfo { name: string; description: string; url: string; kind: 'library' | 'query' }
interface Availability {
  sources: Record<string, SourceInfo>
  cache_dir: string
  cached: { source: string; id: string; species: string; n_sets: number | null }[]
}

async function getJson<T>(url: string): Promise<T> {
  const resp = await fetch(url)
  const body = await resp.json().catch(() => ({}))
  if (!resp.ok) throw new Error(body.detail || `${resp.status} ${resp.statusText}`)
  return body as T
}

async function postJson<T>(url: string, payload: unknown): Promise<T> {
  const resp = await fetch(url, { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(payload) })
  const body = await resp.json().catch(() => ({}))
  if (!resp.ok) throw new Error(body.detail || `${resp.status} ${resp.statusText}`)
  return body as T
}

export default function GeneSetLibraryModal() {
  const isOpen = useStore((s) => s.isGeneSetLibraryModalOpen)
  const setOpen = useStore((s) => s.setGeneSetLibraryModalOpen)
  const addFolderToCategory = useStore((s) => s.addFolderToCategory)
  const geneSetCategories = useStore((s) => s.geneSetCategories)

  const [availability, setAvailability] = useState<Availability | null>(null)
  const [species, setSpecies] = useState<Species>('human')
  const [speciesNote, setSpeciesNote] = useState<string>('')
  const [sourceId, setSourceId] = useState<string>('msigdb')
  const [libraries, setLibraries] = useState<LibraryEntry[]>([])
  const [libFilter, setLibFilter] = useState('')
  const [selectedLib, setSelectedLib] = useState<LibraryEntry | null>(null)
  const [fetching, setFetching] = useState<Record<string, { frac: number; message: string }>>({})
  const [loadingLibs, setLoadingLibs] = useState(false)

  const [q, setQ] = useState('')
  const [gene, setGene] = useState('')
  const [offset, setOffset] = useState(0)
  const [total, setTotal] = useState(0)
  const [rows, setRows] = useState<SetRow[]>([])
  const [searching, setSearching] = useState(false)
  const [columns, setColumns] = useState<VarBooleanColumn[]>([])
  const [activeColumns, setActiveColumns] = useState<string[]>([])
  const [minPresent, setMinPresent] = useState(1)
  const [filterColumn, setFilterColumn] = useState<string>('')
  const [minInColumn, setMinInColumn] = useState(0)
  const [checked, setChecked] = useState<Set<string>>(new Set())
  const [expanded, setExpanded] = useState<Set<string>>(new Set())
  const [folderName, setFolderName] = useState('')
  const [hasDataset, setHasDataset] = useState(true)

  // STRING query panel
  const [seedText, setSeedText] = useState('')
  const [seedSetId, setSeedSetId] = useState('')
  const [stringLimit, setStringLimit] = useState(30)
  const [stringScore, setStringScore] = useState(400)
  const [stringMeta, setStringMeta] = useState<{ unmapped: string[]; nEdges: number } | null>(null)

  const [error, setError] = useState<string | null>(null)
  const [notice, setNotice] = useState<string | null>(null)

  const allGeneSets = useMemo(() => flattenGeneSets(geneSetCategories), [geneSetCategories])
  const importedNames = useMemo(() => {
    const names = new Set<string>()
    if (!selectedLib && sourceId !== 'string') return names
    for (const gs of allGeneSets) {
      if (!gs.source) continue
      if (sourceId === 'string' ? gs.source.source === 'string' : gs.source.library === selectedLib?.id) names.add(gs.name)
    }
    return names
  }, [allGeneSets, selectedLib, sourceId])

  const toast = useCallback((msg: string) => {
    setNotice(msg)
    window.setTimeout(() => setNotice(null), 6000)
  }, [])

  const refreshAvailability = useCallback(() => {
    getJson<Availability>(`${API_BASE}/gene_set_sources`).then(setAvailability).catch(() => {})
  }, [])

  // --- on open: sources, species guess, boolean columns -----------------------
  useEffect(() => {
    if (!isOpen) return
    let cancelled = false
    setError(null)
    getJson<Availability>(`${API_BASE}/gene_set_sources`).then((a) => { if (!cancelled) setAvailability(a) }).catch((e) => { if (!cancelled) setError(String(e.message || e)) })
    const remembered = (() => { try { return localStorage.getItem(SPECIES_KEY) } catch { return null } })()
    getJson<{ species: string | null; method: string | null; confidence: number }>(appendDataset(`${API_BASE}/gene_sets/species_guess`))
      .then((g) => {
        if (cancelled) return
        setHasDataset(true)
        setSpecies(pickSpecies(g, remembered))
        setSpeciesNote(g.species ? `dataset looks ${g.species} (${g.method === 'ensembl_prefix' ? 'Ensembl ids' : 'symbol case'})` : 'could not guess the dataset species')
      })
      .catch(() => {
        if (cancelled) return
        setHasDataset(false)
        setSpecies(pickSpecies(null, remembered))
        setSpeciesNote('no dataset loaded — overlap is unavailable')
      })
    fetchVarBooleanColumns()
      .then((cols) => {
        if (cancelled) return
        setColumns(cols)
        const preferred = cols.map((c) => c.name).filter((n) => /highly_variable|spatially_variable/i.test(n))
        setActiveColumns(preferred)
      })
      .catch(() => { if (!cancelled) setColumns([]) })
    return () => { cancelled = true }
  }, [isOpen])

  // --- catalogue per source / species ----------------------------------------
  const loadLibraries = useCallback(async (refresh = false) => {
    if (sourceId === 'string') { setLibraries([]); return }
    setLoadingLibs(true)
    setError(null)
    try {
      const body = await getJson<{ libraries: LibraryEntry[] }>(`${API_BASE}/gene_set_sources/${sourceId}/libraries?species=${species}&refresh=${refresh}`)
      setLibraries(body.libraries)
    } catch (e) {
      setError(String((e as Error).message || e))
      setLibraries([])
    } finally {
      setLoadingLibs(false)
    }
  }, [sourceId, species])

  useEffect(() => {
    if (!isOpen) return
    setSelectedLib(null)
    setRows([]); setTotal(0); setOffset(0); setChecked(new Set())
    loadLibraries(false)
  }, [isOpen, loadLibraries])

  useEffect(() => {
    if (!isOpen) return
    const onKey = (e: KeyboardEvent) => { if (e.key === 'Escape') setOpen(false) }
    window.addEventListener('keydown', onKey)
    return () => window.removeEventListener('keydown', onKey)
  }, [isOpen, setOpen])

  // --- overlap check for a page of sets --------------------------------------
  const overlapFor = useCallback(async (sets: LibrarySet[]): Promise<SetRow[]> => {
    if (!hasDataset || sets.length === 0) return sets
    try {
      const body = await postJson<{ sets: OverlapEntry[] }>(appendDataset(`${API_BASE}/gene_sets/overlap`), {
        sets: sets.map((s) => ({ name: s.name, genes: s.genes, genesDown: s.genes_down ?? [] })),
        columns: activeColumns,
      })
      return attachOverlap(sets, body.sets)
    } catch (e) {
      setError(`Overlap check failed: ${(e as Error).message || e}`)
      return sets
    }
  }, [hasDataset, activeColumns])

  // --- search a cached library -----------------------------------------------
  // `terms` overrides the current search boxes, for callers that just changed
  // them and cannot wait for the state to settle (the Clear button).
  const search = useCallback(async (lib: LibraryEntry, newOffset: number, terms?: { q: string; gene: string }) => {
    const qq = terms ? terms.q : q
    const gg = terms ? terms.gene : gene
    setSearching(true)
    setError(null)
    try {
      const params = new URLSearchParams({ species: lib.species, q: qq, gene: gg, offset: String(newOffset), limit: String(PAGE) })
      const body = await getJson<{ total: number; sets: LibrarySet[] }>(`${API_BASE}/gene_set_sources/${lib.source}/libraries/${encodeURIComponent(lib.id)}/sets?${params}`)
      setTotal(body.total)
      setOffset(newOffset)
      setRows(body.sets)
      setRows(await overlapFor(body.sets))
    } catch (e) {
      setError(String((e as Error).message || e))
    } finally {
      setSearching(false)
    }
  }, [q, gene, overlapFor])

  const selectLibrary = useCallback((lib: LibraryEntry) => {
    setSelectedLib(lib)
    setChecked(new Set()); setExpanded(new Set())
    setFolderName(defaultFolderName(lib.name, lib.source))
    setRows([]); setTotal(0); setOffset(0)
    if (lib.cached) search(lib, 0)
  }, [search])

  // Re-run the overlap when the column chips change, without a new search.
  useEffect(() => {
    if (rows.length === 0 || !rows.some((r) => r.overlap)) return
    let cancelled = false
    overlapFor(rows).then((r) => { if (!cancelled) setRows(r) })
    return () => { cancelled = true }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [activeColumns])

  const fetchLibrary = useCallback(async (lib: LibraryEntry) => {
    setError(null)
    setFetching((f) => ({ ...f, [lib.id]: { frac: 0, message: 'Starting…' } }))
    try {
      const { task_id } = await postJson<{ task_id: string }>(`${API_BASE}/gene_set_sources/${lib.source}/libraries/${encodeURIComponent(lib.id)}/fetch`, { species })
      const task = await pollTask(task_id, undefined, (s) => {
        setFetching((f) => ({ ...f, [lib.id]: { frac: s.progress ?? 0, message: s.message ?? 'Downloading…' } }))
      })
      if (task.status !== 'completed') throw new Error(task.error || `Fetch ${task.status}`)
      const summary = task.result as { n_sets: number; species: Species; version: string | null }
      const updated: LibraryEntry = { ...lib, cached: true, n_sets: summary.n_sets, species: summary.species, version: summary.version }
      setLibraries((ls) => ls.map((l) => (l.id === lib.id ? updated : l)))
      toast(`Fetched ${lib.name}: ${summary.n_sets.toLocaleString()} sets cached locally`)
      refreshAvailability()
      selectLibrary(updated)
    } catch (e) {
      setError(`Could not fetch ${lib.name}: ${(e as Error).message || e}`)
    } finally {
      setFetching((f) => { const n = { ...f }; delete n[lib.id]; return n })
    }
  }, [species, toast, selectLibrary, refreshAvailability])

  // --- STRING ----------------------------------------------------------------
  const runString = useCallback(async () => {
    const fromSet = seedSetId ? allGeneSets.find((g) => g.id === seedSetId)?.genes ?? [] : []
    const typed = seedText.split(/[\s,;]+/).map((s) => s.trim()).filter(Boolean)
    const seeds = Array.from(new Set([...fromSet, ...typed]))
    if (seeds.length === 0) { setError('Give STRING at least one seed gene'); return }
    setSearching(true)
    setError(null)
    try {
      const body = await postJson<{ sets: { name: string; genes: string[]; description: string; url: string }[]; edges: unknown[]; unmapped: string[] }>(
        `${API_BASE}/gene_set_sources/string/partners`, { genes: seeds, species, limit: stringLimit, required_score: stringScore })
      const sets: LibrarySet[] = body.sets.map((s) => ({ name: s.name, description: s.description, url: s.url, n_genes: s.genes.length, genes: s.genes }))
      setStringMeta({ unmapped: body.unmapped, nEdges: body.edges.length })
      setTotal(sets.length); setOffset(0)
      setChecked(new Set(sets.map((s) => s.name)))
      setFolderName(`STRING partners (${species})`)
      setRows(await overlapFor(sets))
    } catch (e) {
      setError(String((e as Error).message || e))
    } finally {
      setSearching(false)
    }
  }, [seedSetId, seedText, allGeneSets, species, stringLimit, stringScore, overlapFor])

  // --- import ------------------------------------------------------------------
  const doImport = useCallback(() => {
    const chosen = rows.filter((r) => checked.has(r.name))
    const meta = sourceId === 'string'
      ? { source: 'string', library: 'partners', libraryName: 'STRING partners', version: '12.0' }
      : { source: selectedLib!.source, library: selectedLib!.id, libraryName: selectedLib!.name, version: selectedLib!.version }
    const sets = importSets(chosen, meta)
    if (sets.length === 0) { setError('Nothing to import: none of the chosen sets has a gene present in the dataset'); return }
    const name = folderName.trim() || defaultFolderName(meta.libraryName, meta.source)
    addFolderToCategory('manual', name, sets)
    const skipped = chosen.length - sets.length
    toast(`Imported ${sets.length} set${sets.length === 1 ? '' : 's'} into "${name}"${skipped ? ` (${skipped} skipped: no genes present)` : ''}`)
    setChecked(new Set())
  }, [rows, checked, sourceId, selectedLib, folderName, addFolderToCategory, toast])

  if (!isOpen) return null

  const sources = availability?.sources ?? {}
  const visibleLibs = sortLibraries(filterLibraries(libraries, libFilter), species)
  const rowFilter = { minPresent: hasDataset ? minPresent : 0, column: filterColumn || undefined, minInColumn }
  const shownRows = rows.filter((r) => rowPasses(r, rowFilter))
  const nChecked = shownRows.filter((r) => checked.has(r.name)).length
  const canImport = nChecked > 0 && (sourceId === 'string' || !!selectedLib)

  const toggleChecked = (name: string) => setChecked((c) => { const n = new Set(c); n.has(name) ? n.delete(name) : n.add(name); return n })
  const toggleExpanded = (name: string) => setExpanded((c) => { const n = new Set(c); n.has(name) ? n.delete(name) : n.add(name); return n })

  const chooseSpecies = (sp: Species) => {
    setSpecies(sp)
    try { localStorage.setItem(SPECIES_KEY, sp) } catch { /* private window */ }
  }

  return (
    <div
      onClick={() => setOpen(false)}
      style={{ position: 'fixed', inset: 0, background: 'rgba(0,0,0,0.6)', zIndex: 2000, display: 'flex', alignItems: 'center', justifyContent: 'center' }}
    >
      <div
        onClick={(e) => e.stopPropagation()}
        style={{
          background: dark.panel, border: `1px solid ${dark.border}`, borderRadius: 8,
          width: 'min(1140px, 95vw)', height: '86vh', display: 'flex', flexDirection: 'column', color: dark.text, fontSize: 12,
        }}
      >
        {/* header */}
        <div style={{ display: 'flex', alignItems: 'center', gap: 12, padding: '10px 14px', borderBottom: `1px solid ${dark.border}` }}>
          <div style={{ fontSize: 14, fontWeight: 600, color: dark.accent }}>Gene set library</div>
          <div style={{ display: 'flex', alignItems: 'center', gap: 4 }}>
            <span style={{ color: dark.dim }}>Species:</span>
            {(['mouse', 'human'] as Species[]).map((sp) => (
              <button key={sp} onClick={() => chooseSpecies(sp)} style={{ ...btnGhost, padding: '2px 8px', color: species === sp ? dark.accent : dark.muted, borderColor: species === sp ? dark.accent : dark.border }}>
                {sp}
              </button>
            ))}
            <span style={{ color: dark.dim, marginLeft: 6 }}>{speciesNote}</span>
          </div>
          <div style={{ flex: 1 }} />
          {availability && <span style={{ color: dark.dim }} title={availability.cache_dir}>cache: {availability.cached.length} librar{availability.cached.length === 1 ? 'y' : 'ies'}</span>}
          <button onClick={() => setOpen(false)} style={{ ...btnGhost, padding: '2px 8px' }} title="Close (Esc)">✕</button>
        </div>

        {/* source tabs */}
        <div style={{ display: 'flex', gap: 4, padding: '8px 14px 0' }}>
          {Object.entries(sources).map(([id, info]) => (
            <button
              key={id}
              onClick={() => { setSourceId(id); setRows([]); setTotal(0); setChecked(new Set()); setStringMeta(null); setQ(''); setGene('') }}
              title={info.description}
              style={{ ...btnGhost, color: sourceId === id ? dark.accent : dark.muted, borderColor: sourceId === id ? dark.accent : dark.border, background: sourceId === id ? dark.inset : 'transparent' }}
            >
              {info.name}{info.kind === 'query' ? ' (query)' : ''}
            </button>
          ))}
        </div>

        {error && (
          <div style={{ margin: '8px 14px 0', padding: '6px 10px', background: '#3a1f2b', border: `1px solid ${dark.alert}`, borderRadius: 4, color: '#f5b8c4' }}>
            {error}
          </div>
        )}

        <div style={{ display: 'flex', flex: 1, minHeight: 0, padding: 14, gap: 14 }}>
          {/* left: libraries or STRING seeds */}
          <div style={{ width: 300, display: 'flex', flexDirection: 'column', gap: 6, minHeight: 0 }}>
            {sourceId === 'string' ? (
              <StringSeeds
                seedText={seedText} setSeedText={setSeedText}
                seedSetId={seedSetId} setSeedSetId={setSeedSetId}
                limit={stringLimit} setLimit={setStringLimit}
                score={stringScore} setScore={setStringScore}
                sets={allGeneSets.map((g) => ({ id: g.id, name: g.name, n: g.genes.length }))}
                onRun={runString} busy={searching} meta={stringMeta}
              />
            ) : (
              <>
                <div style={{ display: 'flex', gap: 6, alignItems: 'center' }}>
                  <input value={libFilter} onChange={(e) => setLibFilter(e.target.value)} placeholder="Filter libraries…" style={{ ...field, flex: 1 }} />
                  <button onClick={() => loadLibraries(true)} style={btnGhost} title="Re-list this source's libraries from the network">↻</button>
                </div>
                <div style={{ flex: 1, overflowY: 'auto', border: `1px solid ${dark.border}`, borderRadius: 4, background: dark.inset }}>
                  {loadingLibs && <div style={{ padding: 8, color: dark.dim }}>Listing libraries…</div>}
                  {!loadingLibs && visibleLibs.length === 0 && <div style={{ padding: 8, color: dark.dim }}>No libraries.</div>}
                  {visibleLibs.map((lib) => {
                    const active = selectedLib?.id === lib.id
                    const prog = fetching[lib.id]
                    const foreign = lib.species !== species
                    return (
                      <div
                        key={lib.id}
                        onClick={() => selectLibrary(lib)}
                        title={`${lib.description || lib.name}${lib.version ? ` · v${lib.version}` : ''}${lib.fetched_at ? ` · fetched ${lib.fetched_at.slice(0, 10)}` : ''}`}
                        style={{ padding: '5px 8px', cursor: 'pointer', borderBottom: `1px solid ${dark.border}`, background: active ? '#1b2f55' : 'transparent', opacity: foreign ? 0.75 : 1 }}
                      >
                        <div style={{ display: 'flex', alignItems: 'center', gap: 6 }}>
                          <span style={{ flex: 1, color: active ? dark.accent : dark.text, overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap' }}>{lib.name}</span>
                          {foreign && <span style={{ color: dark.warn, fontSize: 10 }} title={`${lib.species}-symbol library; matched case-insensitively`}>{lib.species}</span>}
                          {lib.cached ? (
                            <span style={{ color: dark.accent, fontSize: 10 }} title="Cached locally">✓ {lib.n_sets?.toLocaleString()}</span>
                          ) : prog ? (
                            <span style={{ color: dark.warn, fontSize: 10 }}>{Math.round(prog.frac * 100)}%</span>
                          ) : (
                            <button onClick={(e) => { e.stopPropagation(); fetchLibrary(lib) }} style={{ ...btnGhost, padding: '1px 6px', fontSize: 10 }} title="Download into the local cache">
                              Fetch{lib.n_sets ? ` (${lib.n_sets.toLocaleString()})` : ''}
                            </button>
                          )}
                        </div>
                        {prog && <div style={{ color: dark.dim, fontSize: 10 }}>{prog.message}</div>}
                      </div>
                    )
                  })}
                </div>
              </>
            )}
          </div>

          {/* right: sets */}
          <div style={{ flex: 1, display: 'flex', flexDirection: 'column', gap: 6, minWidth: 0, minHeight: 0 }}>
            {sourceId !== 'string' && (
              <div style={{ display: 'flex', gap: 6, alignItems: 'center' }}>
                <input
                  value={q} onChange={(e) => setQ(e.target.value)}
                  onKeyDown={(e) => { if (e.key === 'Enter' && selectedLib?.cached) search(selectedLib, 0) }}
                  placeholder="Search set names (e.g. collagen, NABA, chondro)" style={{ ...field, flex: 2 }}
                  disabled={!selectedLib?.cached}
                />
                <input
                  value={gene} onChange={(e) => setGene(e.target.value)}
                  onKeyDown={(e) => { if (e.key === 'Enter' && selectedLib?.cached) search(selectedLib, 0) }}
                  placeholder="…containing gene" style={{ ...field, flex: 1 }}
                  disabled={!selectedLib?.cached}
                />
                <button onClick={() => selectedLib?.cached && search(selectedLib, 0)} disabled={!selectedLib?.cached || searching} style={btn}>
                  {searching ? '…' : 'Search'}
                </button>
              </div>
            )}

            {hasDataset && (
              <div style={{ display: 'flex', gap: 8, alignItems: 'center', flexWrap: 'wrap', color: dark.muted }}>
                <span>Count members in:</span>
                {columns.length === 0 && <span style={{ color: dark.dim }}>no boolean .var columns (run Highly Variable Genes first)</span>}
                {columns.map((c) => {
                  const on = activeColumns.includes(c.name)
                  return (
                    <button key={c.name} onClick={() => setActiveColumns((a) => (on ? a.filter((x) => x !== c.name) : [...a, c.name]))}
                      style={{ ...btnGhost, padding: '1px 8px', fontSize: 11, borderRadius: 10, color: on ? '#cfe' : dark.muted, background: on ? dark.border : 'transparent' }}
                      title={`${c.n_true.toLocaleString()} of ${c.n_total.toLocaleString()} genes`}>
                      {c.name}
                    </button>
                  )
                })}
                <span style={{ marginLeft: 'auto' }}>show sets with ≥</span>
                <input type="number" min={0} value={minPresent} onChange={(e) => setMinPresent(Math.max(0, Number(e.target.value) || 0))} style={{ ...field, width: 48 }} />
                <span>present</span>
                {activeColumns.length > 0 && (
                  <>
                    <span>and ≥</span>
                    <input type="number" min={0} value={minInColumn} onChange={(e) => setMinInColumn(Math.max(0, Number(e.target.value) || 0))} style={{ ...field, width: 48 }} />
                    <select value={filterColumn} onChange={(e) => setFilterColumn(e.target.value)} style={field}>
                      <option value="">(any column)</option>
                      {activeColumns.map((c) => <option key={c} value={c}>{c}</option>)}
                    </select>
                  </>
                )}
              </div>
            )}

            <div style={{ flex: 1, overflowY: 'auto', border: `1px solid ${dark.border}`, borderRadius: 4, background: dark.inset, minHeight: 0 }}>
              {sourceId !== 'string' && !selectedLib && <div style={{ padding: 12, color: dark.dim }}>Pick a library on the left. Libraries marked ✓ are cached; others need one download.</div>}
              {sourceId !== 'string' && selectedLib && !selectedLib.cached && !fetching[selectedLib.id] && (
                <div style={{ padding: 12, color: dark.muted }}>
                  <b style={{ color: dark.text }}>{selectedLib.name}</b> is not cached yet.{' '}
                  <button onClick={() => fetchLibrary(selectedLib)} style={btn}>Fetch it</button>
                  <div style={{ color: dark.dim, marginTop: 6 }}>{selectedLib.description}</div>
                </div>
              )}
              {sourceId !== 'string' && selectedLib && fetching[selectedLib.id] && (
                <div style={{ padding: 12, color: dark.muted }}>
                  Downloading {selectedLib.name}… {Math.round(fetching[selectedLib.id].frac * 100)}% — {fetching[selectedLib.id].message}
                </div>
              )}
              {sourceId !== 'string' && selectedLib?.cached && !searching && rows.length === 0 && total === 0 && (q || gene) && (
                <div style={{ padding: 12, color: dark.muted }}>
                  No sets in <b style={{ color: dark.text }}>{selectedLib.name}</b> match{q ? <> “{q}”</> : null}{q && gene ? ' and' : ''}{gene ? <> gene “{gene}”</> : null}.{' '}
                  <button onClick={() => { setQ(''); setGene(''); search(selectedLib, 0, { q: '', gene: '' }) }} style={{ ...btnGhost, padding: '1px 8px' }}>Clear search</button>
                </div>
              )}
              {sourceId === 'string' && rows.length === 0 && !searching && (
                <div style={{ padding: 12, color: dark.dim }}>Enter seed genes (or pick a gene set) and query STRING for their interaction partners. Each seed becomes one set.</div>
              )}
              {rows.length > 0 && (
                <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: 12 }}>
                  <thead>
                    <tr style={{ color: dark.dim, textAlign: 'left', position: 'sticky', top: 0, background: dark.inset }}>
                      <th style={{ padding: '4px 6px', width: 24 }}>
                        <input type="checkbox" checked={shownRows.length > 0 && nChecked === shownRows.length}
                          onChange={(e) => setChecked(e.target.checked ? new Set(shownRows.map((r) => r.name)) : new Set())} title="Select all shown" />
                      </th>
                      <th style={{ padding: '4px 6px' }}>Set</th>
                      <th style={{ padding: '4px 6px', textAlign: 'right' }}>genes</th>
                      {hasDataset && <th style={{ padding: '4px 6px', textAlign: 'right' }}>present</th>}
                      {hasDataset && activeColumns.map((c) => <th key={c} style={{ padding: '4px 6px', textAlign: 'right' }}>{c}</th>)}
                    </tr>
                  </thead>
                  <tbody>
                    {shownRows.map((r) => {
                      const isOpenRow = expanded.has(r.name)
                      const already = importedNames.has(r.name)
                      return (
                        <React.Fragment key={r.name}>
                          <tr style={{ borderTop: `1px solid ${dark.border}` }}>
                            <td style={{ padding: '3px 6px' }}><input type="checkbox" checked={checked.has(r.name)} onChange={() => toggleChecked(r.name)} /></td>
                            <td style={{ padding: '3px 6px', maxWidth: 420 }}>
                              <span onClick={() => toggleExpanded(r.name)} style={{ cursor: 'pointer', color: dark.text }} title={r.description || r.name}>
                                {isOpenRow ? '▾ ' : '▸ '}{r.name}
                              </span>
                              {r.url && <a href={r.url} target="_blank" rel="noreferrer" style={{ color: dark.dim, marginLeft: 6, textDecoration: 'none' }} title={r.url}>↗</a>}
                              {already && <span style={{ color: dark.accent, marginLeft: 6, fontSize: 10 }} title="Already imported from this library">✓ imported</span>}
                              {r.description && !isOpenRow && <div style={{ color: dark.dim, fontSize: 10, overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap' }}>{r.description}</div>}
                            </td>
                            <td style={{ padding: '3px 6px', textAlign: 'right', color: dark.muted }}>{r.n_genes}</td>
                            {hasDataset && (
                              <td style={{ padding: '3px 6px', textAlign: 'right', color: r.overlap && r.overlap.n_present === 0 ? dark.alert : dark.text }} title={r.overlap ? `${r.overlap.n_exact} exact, ${r.overlap.n_case_insensitive} case-insensitive, ${r.overlap.n_missing} missing` : 'checking…'}>
                                {presenceLabel(r).replace(' (case-insensitive)', '')}{r.overlap && r.overlap.n_case_insensitive > 0 && <span style={{ color: dark.warn }} title="Some members matched only case-insensitively"> Aa</span>}
                              </td>
                            )}
                            {hasDataset && activeColumns.map((c) => (
                              <td key={c} style={{ padding: '3px 6px', textAlign: 'right', color: dark.muted }}>{r.overlap ? (r.overlap.columns[c] ?? 0) : '…'}</td>
                            ))}
                          </tr>
                          {isOpenRow && (
                            <tr>
                              <td />
                              <td colSpan={2 + (hasDataset ? 1 + activeColumns.length : 0)} style={{ padding: '2px 6px 8px', color: dark.muted, fontSize: 11, lineHeight: 1.5 }}>
                                {r.description && <div style={{ color: dark.dim, marginBottom: 4 }}>{r.description}</div>}
                                {r.overlap ? (
                                  <>
                                    <span style={{ color: dark.dim }}>present: </span>{r.overlap.genes_resolved.join(', ') || '—'}
                                    {r.overlap.genes_missing.length > 0 && (
                                      <div><span style={{ color: dark.dim }}>missing: </span><span style={{ color: '#c99' }}>{r.overlap.genes_missing.join(', ')}{r.overlap.n_missing > r.overlap.genes_missing.length ? ` … (+${r.overlap.n_missing - r.overlap.genes_missing.length})` : ''}</span></div>
                                    )}
                                  </>
                                ) : r.genes.join(', ')}
                              </td>
                            </tr>
                          )}
                        </React.Fragment>
                      )
                    })}
                  </tbody>
                </table>
              )}
            </div>

            {/* footer */}
            <div style={{ display: 'flex', alignItems: 'center', gap: 8, flexWrap: 'wrap' }}>
              {total > 0 && sourceId !== 'string' && selectedLib && (
                <>
                  <button onClick={() => search(selectedLib, Math.max(0, offset - PAGE))} disabled={offset === 0 || searching} style={btnGhost}>‹</button>
                  <span style={{ color: dark.dim }}>{offset + 1}–{Math.min(offset + PAGE, total)} of {total.toLocaleString()}{shownRows.length < rows.length ? ` (${rows.length - shownRows.length} hidden by filter)` : ''}</span>
                  <button onClick={() => search(selectedLib, offset + PAGE)} disabled={offset + PAGE >= total || searching} style={btnGhost}>›</button>
                </>
              )}
              {sourceId === 'string' && total > 0 && <span style={{ color: dark.dim }}>{total} set{total === 1 ? '' : 's'}</span>}
              <div style={{ flex: 1 }} />
              <span style={{ color: dark.dim }}>into folder</span>
              <input value={folderName} onChange={(e) => setFolderName(e.target.value)} style={{ ...field, width: 240 }} placeholder="Folder name" />
              <button onClick={doImport} disabled={!canImport} style={{ ...btn, opacity: canImport ? 1 : 0.5 }} title={hasDataset ? 'Imports the dataset spelling of each present gene' : 'Imports the library spelling (no dataset loaded)'}>
                Import {nChecked > 0 ? nChecked : ''} set{nChecked === 1 ? '' : 's'}
              </button>
            </div>
          </div>
        </div>

        {notice && (
          <div style={{ position: 'fixed', bottom: 16, left: '50%', transform: 'translateX(-50%)', background: '#2d4a43', color: '#d6f5ee', border: `1px solid ${dark.accent}`, padding: '8px 16px', borderRadius: 6, fontSize: 12, zIndex: 2100, maxWidth: '80vw' }}>
            {notice}
          </div>
        )}
      </div>
    </div>
  )
}

function StringSeeds(props: {
  seedText: string; setSeedText: (s: string) => void
  seedSetId: string; setSeedSetId: (s: string) => void
  limit: number; setLimit: (n: number) => void
  score: number; setScore: (n: number) => void
  sets: { id: string; name: string; n: number }[]
  onRun: () => void; busy: boolean
  meta: { unmapped: string[]; nEdges: number } | null
}) {
  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: 8, color: dark.muted }}>
      <div>
        <div style={{ color: dark.dim, marginBottom: 2 }}>Seed genes from a gene set</div>
        <select value={props.seedSetId} onChange={(e) => props.setSeedSetId(e.target.value)} style={{ ...field, width: '100%' }}>
          <option value="">(none)</option>
          {props.sets.map((s) => <option key={s.id} value={s.id}>{s.name} ({s.n})</option>)}
        </select>
      </div>
      <div>
        <div style={{ color: dark.dim, marginBottom: 2 }}>…and/or typed (space, comma or newline separated)</div>
        <textarea value={props.seedText} onChange={(e) => props.setSeedText(e.target.value)} rows={5} placeholder="Col1a1 Sox9 Runx2" style={{ ...field, width: '100%', resize: 'vertical', fontFamily: 'inherit' }} />
      </div>
      <div style={{ display: 'flex', gap: 8, alignItems: 'center' }}>
        <span>partners / seed</span>
        <input type="number" min={1} max={500} value={props.limit} onChange={(e) => props.setLimit(Math.max(1, Number(e.target.value) || 1))} style={{ ...field, width: 56 }} />
        <span title="STRING combined score, 0–1000 (400 medium, 700 high)">min score</span>
        <input type="number" min={0} max={1000} step={50} value={props.score} onChange={(e) => props.setScore(Math.min(1000, Math.max(0, Number(e.target.value) || 0)))} style={{ ...field, width: 64 }} />
      </div>
      <button onClick={props.onRun} disabled={props.busy} style={btn}>{props.busy ? 'Querying STRING…' : 'Query STRING'}</button>
      {props.meta && (
        <div style={{ color: dark.dim, fontSize: 11 }}>
          {props.meta.nEdges} edges.{props.meta.unmapped.length > 0 && <> Not in STRING: <span style={{ color: '#c99' }}>{props.meta.unmapped.join(', ')}</span></>}
        </div>
      )}
      <div style={{ color: dark.dim, fontSize: 11 }}>Each seed becomes one set, seed first, partners by descending combined score. Live query, not cached.</div>
    </div>
  )
}
