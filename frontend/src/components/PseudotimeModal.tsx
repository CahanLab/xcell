/**
 * Pseudotime (Analyze → Cells → Pseudotime (DPT)): diffusion pseudotime on a
 * diffusion map, from a root the user chooses.
 *
 * The root is the whole decision — DPT is a distance from it — so this modal
 * is mostly a root picker: the most potent cell by stemFinder (when a score
 * exists), a column's extreme, a cell group, the current selection, or a tip
 * of a diffusion component. DPT runs on the map's own cells; the cell mask is
 * not sent (scope the map instead, by running Diffusion map on a subset).
 */

import { useCallback, useEffect, useMemo, useState } from 'react'
import { useStore } from '../store'
import { appendDataset, refreshCellSubsets, refreshSchema, useObsSummaries } from '../hooks/useData'
import { datasetIdentity } from '../lib/datasetIdentity'
import { dptBlocker, dptOutputName, dptParams, type DptInputs, type RootMode } from '../lib/diffusion'

interface DiffmapInfo {
  key: string
  graph_key: string
  n_comps: number
  n_cells: number
  cell_subset: string | null
  eigenvalues: number[]
  n_stationary: number
  view_dims: number[]
}

interface Potency { column: string; direction: 'min' | 'max'; label: string }

interface Listing { diffmaps: DiffmapInfo[]; potency: Potency[] }

interface Result {
  key_added: string
  diffmap_key: string
  n_dcs: number
  root_index: number
  root_name: string
  root_rule: string
  n_cells: number
  n_unreachable: number
  warnings: string[]
  cell_subset?: string
}

/** The radio list. The two component tips share one entry with an end picker. */
type RootChoice = 'potency' | 'obs_min' | 'obs_max' | 'group' | 'cells' | 'dc'

const ROOT_CHOICES: { key: RootChoice; label: string; hint: string }[] = [
  { key: 'potency', label: 'Most potent cell (stemFinder)', hint: 'The least differentiated cell by a stemFinder score: the lowest stemfinder, or the highest diffOmeter.' },
  { key: 'obs_min', label: 'Lowest value of a column', hint: 'The cell with the smallest value of a numeric .obs column.' },
  { key: 'obs_max', label: 'Highest value of a column', hint: 'The cell with the largest value of a numeric .obs column.' },
  { key: 'group', label: 'A cell group', hint: 'The cell nearest the centre of a category, measured the way DPT measures distance.' },
  { key: 'cells', label: 'The current selection', hint: 'One selected cell is the root; for several, the one nearest their centre.' },
  { key: 'dc', label: 'Tip of a diffusion component', hint: 'One end of a diffusion component — the classic choice when nothing is annotated.' },
]

const dark = {
  overlay: {
    position: 'fixed' as const, inset: 0, background: 'rgba(0,0,0,0.6)',
    display: 'flex', alignItems: 'center', justifyContent: 'center', zIndex: 1000,
  },
  panel: {
    background: '#16213e', border: '1px solid #0f3460', borderRadius: 8,
    width: 'min(520px, 94vw)', maxHeight: '90vh', display: 'flex',
    flexDirection: 'column' as const, boxShadow: '0 8px 32px rgba(0,0,0,0.5)',
  },
  header: {
    display: 'flex', alignItems: 'center', justifyContent: 'space-between',
    padding: '12px 16px', borderBottom: '1px solid #0f3460',
  },
  title: { margin: 0, fontSize: 15, color: '#e94560', fontWeight: 600 },
  close: { background: 'transparent', border: 'none', color: '#888', fontSize: 20, cursor: 'pointer', lineHeight: 1 },
  body: { padding: '14px 16px', overflowY: 'auto' as const },
  label: { fontSize: 10, textTransform: 'uppercase' as const, letterSpacing: 0.6, color: '#666', margin: '12px 0 6px' },
  row: { display: 'flex', alignItems: 'center', gap: 8, marginBottom: 6, fontSize: 12, color: '#ccc' },
  input: {
    background: '#0f1625', border: '1px solid #0f3460', borderRadius: 4,
    color: '#ddd', fontSize: 12, padding: '5px 8px', outline: 'none', width: 70,
  },
  select: {
    background: '#0f1625', border: '1px solid #0f3460', borderRadius: 4,
    color: '#ddd', fontSize: 12, padding: '5px 8px', outline: 'none', maxWidth: 280,
  },
  hint: { fontSize: 10.5, color: '#777', lineHeight: 1.4, margin: '0 0 6px 22px' },
  notice: { padding: '8px 10px', borderRadius: 4, fontSize: 11.5, lineHeight: 1.45, marginBottom: 8 },
  actions: { display: 'flex', gap: 8, padding: '12px 16px', borderTop: '1px solid #0f3460', alignItems: 'center' },
  primary: {
    padding: '8px 16px', borderRadius: 4, border: '1px solid #4ecdc4',
    background: '#4ecdc4', color: '#000', fontSize: 12, fontWeight: 600, cursor: 'pointer',
  },
  ghost: {
    padding: '8px 16px', borderRadius: 4, border: '1px solid #0f3460',
    background: 'transparent', color: '#aaa', fontSize: 12, cursor: 'pointer',
  },
}

const mapLabel = (m: DiffmapInfo) =>
  `${m.key} · ${m.graph_key} · ${m.n_cells.toLocaleString()} cells`

export default function PseudotimeModal() {
  const isOpen = useStore((s) => s.isPseudotimeModalOpen)
  const setOpen = useStore((s) => s.setPseudotimeModalOpen)
  const activeSlot = useStore((s) => s.activeSlot)
  const schema = useStore((s) => s.schema)
  const activeCellMask = useStore((s) => s.activeCellMask)
  const activeSubsetName = useStore((s) => s.activeSubsetName)
  const selection = useStore((s) => s.selectedCellIndices)
  const selectColorColumn = useStore((s) => s.setSelectedColorColumn)
  const setColorMode = useStore((s) => s.setColorMode)
  const setColorBy = useStore((s) => s.setColorBy)
  const refreshObsSummaries = useStore((s) => s.refreshObsSummaries)
  const { summaries } = useObsSummaries()
  const dataset = datasetIdentity(activeSlot, schema)

  const [listing, setListing] = useState<Listing | null>(null)
  const [listError, setListError] = useState<string | null>(null)
  const [diffmapKey, setDiffmapKey] = useState('')
  const [nDcs, setNDcs] = useState('10')
  const [choice, setChoice] = useState<RootChoice>('dc')
  const [potencyColumn, setPotencyColumn] = useState('')
  const [column, setColumn] = useState('')
  const [groupValue, setGroupValue] = useState('')
  const [component, setComponent] = useState('1')
  const [dcEnd, setDcEnd] = useState<'dc_min' | 'dc_max'>('dc_min')
  const [keyAdded, setKeyAdded] = useState('')

  const [busy, setBusy] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const [result, setResult] = useState<Result | null>(null)

  // A map's own defaults: components to use, and the first informative DC.
  const applyMapDefaults = (m: DiffmapInfo | undefined) => {
    setNDcs(String(Math.min(10, m?.n_comps ?? 10)))
    setComponent(String(m?.view_dims?.[0] ?? 1))
  }

  // Everything dataset-shaped resets on open and on a dataset change: this
  // modal never unmounts, so a pick from the last dataset would otherwise be sent.
  useEffect(() => {
    if (!isOpen) return
    let cancelled = false
    setListing(null); setListError(null); setResult(null); setError(null)
    fetch(appendDataset('/api/scanpy/diffmaps', activeSlot))
      .then(async (r) => {
        const data = await r.json()
        if (!r.ok) throw new Error(data.detail || 'Could not read the dataset')
        return data as Listing
      })
      .then((l) => {
        if (cancelled) return
        setListing(l)
        // A saved subset's own map is the natural one while it is active.
        const preferred = activeSubsetName
          ? l.diffmaps.find((m) => m.key === `X_diffmap_${activeSubsetName}`)
          : undefined
        const first = preferred ?? l.diffmaps.find((m) => m.key === 'X_diffmap') ?? l.diffmaps[0]
        setDiffmapKey(first?.key ?? '')
        applyMapDefaults(first)
        setPotencyColumn(l.potency[0]?.column ?? '')
        setChoice(l.potency.length ? 'potency' : 'dc')
        setDcEnd('dc_min')
        setColumn(''); setGroupValue(''); setKeyAdded('')
      })
      .catch((e) => { if (!cancelled) setListError(e instanceof Error ? e.message : String(e)) })
    return () => { cancelled = true }
    // activeSubsetName only seeds the default map on open or a dataset change.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [isOpen, dataset])

  useEffect(() => {
    if (!isOpen) return
    const onKey = (e: KeyboardEvent) => { if (e.key === 'Escape' && !busy) setOpen(false) }
    window.addEventListener('keydown', onKey)
    return () => window.removeEventListener('keydown', onKey)
  }, [isOpen, busy, setOpen])

  const numericColumns = useMemo(
    () => summaries.filter((s) => s.dtype === 'numeric').map((s) => s.name), [summaries])
  const categoricalColumns = useMemo(
    () => summaries.filter((s) => s.dtype === 'category').map((s) => s.name), [summaries])
  const groupValues = useMemo(
    () => (summaries.find((s) => s.name === column)?.categories ?? []).map((c) => c.value),
    [summaries, column])

  const map = listing?.diffmaps.find((m) => m.key === diffmapKey)
  const potency = listing?.potency ?? []
  const rootMode: RootMode = choice === 'dc' ? dcEnd : choice
  const inputs: DptInputs = {
    diffmapKey,
    nDcs,
    rootMode,
    potencyColumn,
    potencyDirection: potency.find((p) => p.column === potencyColumn)?.direction ?? 'min',
    column,
    groupValue,
    component,
    selection,
    keyAdded,
  }
  const blocker = listing ? dptBlocker(inputs) : 'Reading the diffusion maps…'

  const run = useCallback(async () => {
    if (blocker) return
    setBusy(true); setError(null); setResult(null)
    try {
      const resp = await fetch(appendDataset('/api/scanpy/dpt', activeSlot), {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(dptParams(inputs)),
      })
      const data = await resp.json()
      if (!resp.ok) throw new Error(data.detail || 'Request failed')
      const res = data as Result
      setResult(res)
      // A new .obs column: both the schema and the Cell Manager lists.
      await refreshSchema()
      refreshObsSummaries()
      // The subset tree lists a subset's pseudotimes; it refetches on scanpy
      // history, which this tool does not write to.
      if (res.cell_subset) void refreshCellSubsets()
      // A re-run into the same column keeps the column name, so the cached
      // colouring must be dropped for the new values to be fetched.
      setColorBy(null)
      selectColorColumn(res.key_added)
      setColorMode('metadata')
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e))
    } finally {
      setBusy(false)
    }
    // `inputs` is rebuilt every render from the state above; listing it keeps
    // Run from sending a stale pick.
  }, [blocker, inputs, activeSlot, refreshObsSummaries, setColorBy, selectColorColumn, setColorMode])

  if (!isOpen) return null

  const close = () => { if (!busy) setOpen(false) }
  const nComps = map?.n_comps ?? 0

  return (
    <div style={dark.overlay} onClick={close}>
      <div style={dark.panel} onClick={(e) => e.stopPropagation()}>
        <div style={dark.header}>
          <h2 style={dark.title}>Pseudotime (DPT)</h2>
          <button style={dark.close} onClick={close}>&times;</button>
        </div>

        <div style={dark.body}>
          <div style={{ fontSize: 11.5, color: '#999', lineHeight: 1.45, marginBottom: 8 }}>
            Diffusion pseudotime (Haghverdi et al., 2016): each cell&apos;s distance from a root cell,
            measured along the diffusion map&apos;s graph. Cells the root cannot reach get none.
          </div>

          {listError && <div style={{ ...dark.notice, background: 'rgba(233,69,96,0.12)', color: '#f3a3b2' }}>{listError}</div>}
          {!listing && !listError && <div style={{ fontSize: 12, color: '#888' }}>Reading the diffusion maps…</div>}
          {listing && listing.diffmaps.length === 0 && (
            <div style={{ ...dark.notice, background: 'rgba(233,162,59,0.12)', color: '#f0c987' }}>
              No diffusion map yet. Run Analyze → Cells → Diffusion map first, on any graph.
            </div>
          )}

          {listing && listing.diffmaps.length > 0 && (
            <>
              {activeCellMask && (
                <div style={{ ...dark.notice, border: '1px solid #0f3460', color: '#e9a23b' }}>
                  The cell mask is not used: pseudotime covers the diffusion map&apos;s own cells. To
                  scope it, run Diffusion map on a subset{activeSubsetName ? ` (${activeSubsetName})` : ''}.
                </div>
              )}

              <div style={dark.label}>Diffusion map</div>
              <div style={dark.row}>
                <select
                  style={dark.select}
                  value={diffmapKey}
                  onChange={(e) => {
                    setDiffmapKey(e.target.value)
                    applyMapDefaults(listing.diffmaps.find((m) => m.key === e.target.value))
                  }}
                >
                  {listing.diffmaps.map((m) => <option key={m.key} value={m.key}>{mapLabel(m)}</option>)}
                </select>
              </div>
              <div style={dark.row}>
                Components used
                <input style={dark.input} type="number" min={2} max={nComps || undefined} value={nDcs}
                  onChange={(e) => setNDcs(e.target.value)} />
                <span style={{ color: '#777', fontSize: 11 }}>of {nComps} (scanpy&apos;s default is 10)</span>
              </div>
              {map && map.n_stationary > 1 && (
                <div style={{ ...dark.notice, border: '1px solid #0f3460', color: '#e9a23b' }}>
                  This map&apos;s graph has {map.n_stationary} disconnected parts; cells outside the
                  root&apos;s part will have no pseudotime.
                </div>
              )}

              <div style={dark.label}>Root</div>
              {ROOT_CHOICES.filter((c) => c.key !== 'potency' || potency.length > 0).map((c) => (
                <div key={c.key}>
                  <label style={{ ...dark.row, marginBottom: 2, cursor: 'pointer' }}>
                    <input type="radio" name="dptRoot" checked={choice === c.key} onChange={() => setChoice(c.key)} />
                    {c.label}
                    {c.key === 'cells' && (
                      <span style={{ color: selection.length ? '#4ecdc4' : '#777', fontSize: 11 }}>
                        ({selection.length.toLocaleString()} selected)
                      </span>
                    )}
                  </label>
                  {choice === c.key && (
                    <div style={{ ...dark.row, marginLeft: 22 }}>
                      {c.key === 'potency' && (
                        <select style={dark.select} value={potencyColumn} onChange={(e) => setPotencyColumn(e.target.value)}>
                          {potency.map((p) => <option key={p.column} value={p.column}>{p.column} ({p.label})</option>)}
                        </select>
                      )}
                      {(c.key === 'obs_min' || c.key === 'obs_max') && (
                        <select style={dark.select} value={column} onChange={(e) => setColumn(e.target.value)}>
                          <option value="">Choose a numeric column…</option>
                          {numericColumns.map((n) => <option key={n} value={n}>{n}</option>)}
                        </select>
                      )}
                      {c.key === 'group' && (
                        <>
                          <select style={dark.select} value={column}
                            onChange={(e) => { setColumn(e.target.value); setGroupValue('') }}>
                            <option value="">Choose a column…</option>
                            {categoricalColumns.map((n) => <option key={n} value={n}>{n}</option>)}
                          </select>
                          {column && (
                            <select style={dark.select} value={groupValue} onChange={(e) => setGroupValue(e.target.value)}>
                              <option value="">Choose a group…</option>
                              {groupValues.map((v) => <option key={v} value={v}>{v}</option>)}
                            </select>
                          )}
                        </>
                      )}
                      {c.key === 'dc' && (
                        <>
                          <select style={dark.select} value={dcEnd} onChange={(e) => setDcEnd(e.target.value as 'dc_min' | 'dc_max')}>
                            <option value="dc_min">Low end of</option>
                            <option value="dc_max">High end of</option>
                          </select>
                          <select style={dark.select} value={component} onChange={(e) => setComponent(e.target.value)}>
                            {Array.from({ length: nComps }, (_, i) => i).map((i) => (
                              <option key={i} value={String(i)}>DC{i}{map && i < map.n_stationary ? ' (stationary)' : ''}</option>
                            ))}
                          </select>
                        </>
                      )}
                    </div>
                  )}
                  {choice === c.key && <div style={dark.hint}>{c.hint}</div>}
                </div>
              ))}

              <div style={dark.label}>Output</div>
              <div style={dark.row}>
                .obs column
                <input style={{ ...dark.input, width: 220 }} value={keyAdded}
                  placeholder={diffmapKey ? dptOutputName(diffmapKey) : ''}
                  onChange={(e) => setKeyAdded(e.target.value)} />
              </div>
            </>
          )}

          {error && <div style={{ ...dark.notice, background: 'rgba(233,69,96,0.12)', color: '#f3a3b2', marginTop: 8 }}>{error}</div>}
          {result && (
            <div style={{ ...dark.notice, border: '1px solid #0f3460', color: '#ccc', marginTop: 8 }}>
              Pseudotime → <code>.obs[&quot;{result.key_added}&quot;]</code> from {result.root_rule.replace(/`/g, '')}{' '}
              (cell {result.root_name}), over {result.n_cells.toLocaleString()} cells using {result.n_dcs} components.
              {result.warnings.map((w) => <div key={w} style={{ color: '#e9a23b', marginTop: 4 }}>{w}</div>)}
            </div>
          )}
        </div>

        <div style={dark.actions}>
          <button
            style={{ ...dark.primary, opacity: blocker || busy ? 0.5 : 1, cursor: blocker || busy ? 'not-allowed' : 'pointer' }}
            disabled={!!blocker || busy}
            title={blocker ?? undefined}
            onClick={run}
          >
            {busy ? 'Computing…' : 'Compute pseudotime'}
          </button>
          <button style={dark.ghost} onClick={close} disabled={busy}>Close</button>
          {blocker && listing && listing.diffmaps.length > 0 && (
            <span style={{ fontSize: 11, color: '#888' }}>{blocker}</span>
          )}
        </div>
      </div>
    </div>
  )
}
