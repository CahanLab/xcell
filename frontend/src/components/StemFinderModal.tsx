/**
 * Differentiation (Analyze → Cells → Differentiation): PyStemFinder scores.
 *
 * Less differentiated cells vary more in their expression of cell cycle genes
 * than their neighbours do; stemFinder scores that heterogeneity in each
 * cell's kNN neighbourhood. The backend (xcell/stemfinder.py) prepares the
 * expression and the graph, so this modal only chooses metrics, markers and
 * the neighbourhood. PyStemFinder is optional: without it the modal shows how
 * to install it.
 */

import { useCallback, useEffect, useMemo, useState } from 'react'
import { useStore } from '../store'
import { appendDataset, cancelTask, pollTask, refreshSchema, useObsSummaries } from '../hooks/useData'
import { flattenGeneSets } from './GenePanel'
import { datasetIdentity } from '../lib/datasetIdentity'
import { indicesFromMask } from '../lib/cellSubsets'
import { STEM_METRICS, stemFinderBlocker, stemFinderParams, type StemMetric, type StemFinderInputs } from '../lib/stemfinder'

interface GraphInfo { key: string; label: string; n_edges: number }

interface Status {
  available: boolean
  version: string | null
  install_hint: string | null
  n_cells: number
  default_n_neighbors: number
  pc_embeddings: string[]
  pc_dims: Record<string, number>
  graphs: GraphInfo[]
  species: string | null
  x_scale: string
  layers: string[]
}

interface ColumnStats { n: number; min: number | null; median: number | null; max: number | null }

interface Result {
  columns: string[]
  n_cells_scored: number
  n_markers_used: number
  n_markers_missing: number
  markers_missing: string[]
  n_tfs_used: number
  n_neighbors: number | null
  graph_source: string | null
  species: string | null
  stats: Record<string, ColumnStats>
  summary: { group: string; n: number; medians: Record<string, number | null> }[] | null
}

const dark = {
  overlay: {
    position: 'fixed' as const, inset: 0, background: 'rgba(0,0,0,0.6)',
    display: 'flex', alignItems: 'center', justifyContent: 'center', zIndex: 1000,
  },
  panel: {
    background: '#16213e', border: '1px solid #0f3460', borderRadius: 8,
    width: 'min(560px, 94vw)', maxHeight: '90vh', display: 'flex',
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
    color: '#ddd', fontSize: 12, padding: '5px 8px', outline: 'none', width: 80,
  },
  select: {
    background: '#0f1625', border: '1px solid #0f3460', borderRadius: 4,
    color: '#ddd', fontSize: 12, padding: '5px 8px', outline: 'none', maxWidth: 260,
  },
  hint: { fontSize: 10.5, color: '#777', lineHeight: 1.4, margin: '2px 0 6px 22px' },
  notice: { padding: '8px 10px', borderRadius: 4, fontSize: 11.5, lineHeight: 1.45, marginBottom: 8 },
  toggle: { fontSize: 11, color: '#4ecdc4', cursor: 'pointer', margin: '12px 0 4px', userSelect: 'none' as const },
  actions: { display: 'flex', gap: 8, padding: '12px 16px', borderTop: '1px solid #0f3460', alignItems: 'center' },
  primary: {
    padding: '8px 16px', borderRadius: 4, border: '1px solid #4ecdc4',
    background: '#4ecdc4', color: '#000', fontSize: 12, fontWeight: 600, cursor: 'pointer',
  },
  ghost: {
    padding: '8px 16px', borderRadius: 4, border: '1px solid #0f3460',
    background: 'transparent', color: '#aaa', fontSize: 12, cursor: 'pointer',
  },
  chip: {
    padding: '3px 8px', borderRadius: 10, border: '1px solid #4ecdc4', background: 'transparent',
    color: '#4ecdc4', fontSize: 11, cursor: 'pointer',
  },
  table: { width: '100%', borderCollapse: 'collapse' as const, fontSize: 11.5, color: '#ccc' },
  th: { textAlign: 'left' as const, color: '#888', fontWeight: 500, padding: '3px 6px', borderBottom: '1px solid #0f3460' },
  td: { padding: '3px 6px', borderBottom: '1px solid rgba(15,52,96,0.5)' },
}

const fmt = (v: number | null | undefined, digits = 3) =>
  v === null || v === undefined ? '—' : Math.abs(v) >= 100 ? v.toFixed(0) : v.toFixed(digits)

export default function StemFinderModal() {
  const isOpen = useStore((s) => s.isStemFinderModalOpen)
  const setOpen = useStore((s) => s.setStemFinderModalOpen)
  const activeSlot = useStore((s) => s.activeSlot)
  const schema = useStore((s) => s.schema)
  const activeCellMask = useStore((s) => s.activeCellMask)
  const activeSubsetName = useStore((s) => s.activeSubsetName)
  const geneSetCategories = useStore((s) => s.geneSetCategories)
  const selectColorColumn = useStore((s) => s.setSelectedColorColumn)
  const setColorMode = useStore((s) => s.setColorMode)
  const refreshObsSummaries = useStore((s) => s.refreshObsSummaries)
  const { summaries } = useObsSummaries()
  const dataset = datasetIdentity(activeSlot, schema)

  const [status, setStatus] = useState<Status | null>(null)
  const [statusError, setStatusError] = useState<string | null>(null)
  const [metrics, setMetrics] = useState<Set<StemMetric>>(() => new Set(['stemfinder', 'diffometer']))
  const [markerSource, setMarkerSource] = useState<'cell_cycle' | 'gene_set'>('cell_cycle')
  const [species, setSpecies] = useState('mouse')
  const [geneSetId, setGeneSetId] = useState('')
  const [method, setMethod] = useState<StemFinderInputs['method']>('gini')
  const [threshold, setThreshold] = useState('0')
  const [binarizeOn, setBinarizeOn] = useState<StemFinderInputs['binarizeOn']>('scaled')
  const [weightBy, setWeightBy] = useState<StemFinderInputs['weightBy']>('equal')
  const [includeSelf, setIncludeSelf] = useState(true)
  const [graph, setGraph] = useState<'build' | 'existing'>('build')
  const [useRep, setUseRep] = useState('X_pca')
  const [nPcs, setNPcs] = useState('32')
  const [nNeighbors, setNNeighbors] = useState('')
  const [graphKey, setGraphKey] = useState('')
  const [layer, setLayer] = useState('')
  const [suffix, setSuffix] = useState('')
  const [summaryBy, setSummaryBy] = useState('')
  const [showAdvanced, setShowAdvanced] = useState(false)

  const [busy, setBusy] = useState(false)
  const [taskId, setTaskId] = useState<string | null>(null)
  const [progress, setProgress] = useState<string | null>(null)
  const [error, setError] = useState<string | null>(null)
  const [result, setResult] = useState<Result | null>(null)

  const nActive = useMemo(() => (activeCellMask ? activeCellMask.reduce((n, b) => n + (b ? 1 : 0), 0) : 0), [activeCellMask])

  // Everything dataset-shaped resets when the dataset changes: this modal
  // never unmounts, so a pick from the last dataset would otherwise be sent.
  useEffect(() => {
    if (!isOpen) return
    let cancelled = false
    setStatus(null); setStatusError(null); setResult(null); setError(null)
    fetch(appendDataset('/api/stemfinder/status', activeSlot))
      .then(async (r) => {
        const data = await r.json()
        if (!r.ok) throw new Error(data.detail || 'Could not read the dataset')
        return data as Status
      })
      .then((s) => {
        if (cancelled) return
        setStatus(s)
        if (s.species) setSpecies(s.species)
        // A saved subset's own PCA is the natural space for its cells.
        const subsetPca = activeSubsetName ? `X_pca_${activeSubsetName}` : null
        setUseRep(subsetPca && s.pc_embeddings.includes(subsetPca) ? subsetPca
          : s.pc_embeddings.includes('X_pca') ? 'X_pca' : (s.pc_embeddings[0] ?? ''))
        setGraphKey(s.graphs[0]?.key ?? '')
        if (s.pc_embeddings.length === 0 && s.graphs.length > 0) setGraph('existing')
        setLayer(''); setSummaryBy('')
        setSuffix(activeCellMask && activeSubsetName ? activeSubsetName : '')
      })
      .catch((e) => { if (!cancelled) setStatusError(e instanceof Error ? e.message : String(e)) })
    return () => { cancelled = true }
    // activeCellMask/activeSubsetName seed defaults only on open or a dataset change.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [isOpen, dataset])

  useEffect(() => {
    if (!isOpen) return
    const onKey = (e: KeyboardEvent) => { if (e.key === 'Escape' && !busy) setOpen(false) }
    window.addEventListener('keydown', onKey)
    return () => window.removeEventListener('keydown', onKey)
  }, [isOpen, busy, setOpen])

  const geneSets = useMemo(
    () => flattenGeneSets(geneSetCategories).filter((g) => g.genes.length > 0),
    [geneSetCategories],
  )
  const geneSetGenes = geneSets.find((g) => g.id === geneSetId)?.genes ?? null
  const categoricalColumns = useMemo(
    () => summaries.filter((s) => s.dtype === 'category').map((s) => s.name),
    [summaries],
  )

  const inputs: StemFinderInputs = {
    metrics, markerSource, species, geneSetGenes, method, threshold, binarizeOn, weightBy,
    includeSelf, graph, useRep, nPcs, nNeighbors, graphKey, layer, suffix, summaryBy,
    activeCellIndices: null,   // filled at Run; O(n) is not worth paying every render
  }
  const blocker = status?.available ? stemFinderBlocker(inputs) : 'PyStemFinder is not installed.'
  const usesGraph = metrics.has('stemfinder') || metrics.has('diffometer')
  const usesMarkers = usesGraph || metrics.has('cc_mean')
  const nScored = activeCellMask ? nActive : (status?.n_cells ?? 0)
  const defaultK = Math.round(Math.sqrt(nScored))

  const run = useCallback(async () => {
    if (blocker) return
    setBusy(true); setError(null); setResult(null); setProgress('Starting…')
    try {
      const resp = await fetch(appendDataset('/api/stemfinder', activeSlot), {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(stemFinderParams({
          ...inputs,
          activeCellIndices: activeCellMask ? indicesFromMask(activeCellMask) : null,
        })),
      })
      const data = await resp.json()
      if (!resp.ok) throw new Error(data.detail || 'Request failed')
      setTaskId(data.task_id)
      const task = await pollTask(data.task_id, activeSlot, (s) => {
        if (s.message) setProgress(s.message)
      })
      if (task.status === 'cancelled') return
      if (task.status !== 'completed') throw new Error(task.error || `Task ${task.status}`)
      const res = task.result as unknown as Result
      setResult(res)
      // New .obs columns: both the schema and the Cell Manager lists.
      await refreshSchema()
      refreshObsSummaries()
      if (res.columns.length) {
        selectColorColumn(res.columns[0])
        setColorMode('metadata')
      }
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e))
    } finally {
      setBusy(false); setTaskId(null); setProgress(null)
    }
    // `inputs` is rebuilt every render from the state above; listing it keeps
    // Run from sending a stale pick.
  }, [blocker, inputs, activeCellMask, activeSlot, refreshObsSummaries, selectColorColumn, setColorMode])

  const cancel = useCallback(() => {
    if (taskId) cancelTask(taskId, activeSlot).catch(() => {})
  }, [taskId, activeSlot])

  const toggleMetric = (m: StemMetric) => setMetrics((prev) => {
    const next = new Set(prev)
    if (next.has(m)) next.delete(m); else next.add(m)
    return next
  })

  const colorBy = (column: string) => {
    selectColorColumn(column)
    setColorMode('metadata')
  }

  if (!isOpen) return null

  const close = () => { if (!busy) setOpen(false) }

  return (
    <div style={dark.overlay} onClick={close}>
      <div style={dark.panel} onClick={(e) => e.stopPropagation()}>
        <div style={dark.header}>
          <h2 style={dark.title}>Differentiation (stemFinder)</h2>
          <button style={dark.close} onClick={close}>&times;</button>
        </div>

        <div style={dark.body}>
          <div style={{ fontSize: 11.5, color: '#999', lineHeight: 1.45, marginBottom: 8 }}>
            Less differentiated cells vary more in their expression of cell cycle genes than their
            neighbours do. <a href="https://github.com/CahanLab/PyStemFinder" target="_blank" rel="noreferrer" style={{ color: '#4ecdc4' }}>PyStemFinder</a>{' '}
            scores that heterogeneity in each cell&apos;s kNN neighbourhood (Noller &amp; Cahan, 2024).
          </div>

          {statusError && <div style={{ ...dark.notice, background: 'rgba(233,69,96,0.12)', color: '#f3a3b2' }}>{statusError}</div>}
          {!status && !statusError && <div style={{ fontSize: 12, color: '#888' }}>Checking PyStemFinder…</div>}
          {status && !status.available && (
            <div style={{ ...dark.notice, background: 'rgba(233,162,59,0.12)', color: '#f0c987' }}>
              {status.install_hint}
            </div>
          )}

          {status?.available && (
            <>
              {activeCellMask && (
                <div style={{ ...dark.notice, border: '1px solid #0f3460', color: '#e9a23b' }}>
                  Scoped to the cell mask: {nActive.toLocaleString()} active cells
                  {activeSubsetName ? ` (${activeSubsetName})` : ''}. Scaling, the neighbourhood and k = √n
                  use these cells only; cells outside get no score.
                </div>
              )}

              <div style={dark.label}>Metrics</div>
              {STEM_METRICS.map((m) => (
                <div key={m.key}>
                  <label style={{ ...dark.row, marginBottom: 0, cursor: 'pointer' }}>
                    <input type="checkbox" checked={metrics.has(m.key)} onChange={() => toggleMetric(m.key)} />
                    {m.label}
                  </label>
                  <div style={dark.hint}>{m.hint}</div>
                </div>
              ))}

              {usesMarkers && (
                <>
                  <div style={dark.label}>Markers</div>
                  <label style={{ ...dark.row, cursor: 'pointer' }}>
                    <input type="radio" name="sfMarkers" checked={markerSource === 'cell_cycle'} onChange={() => setMarkerSource('cell_cycle')} />
                    Cell cycle genes (S + G2M)
                  </label>
                  <label style={{ ...dark.row, cursor: 'pointer' }}>
                    <input type="radio" name="sfMarkers" checked={markerSource === 'gene_set'} onChange={() => setMarkerSource('gene_set')} />
                    A gene set
                    {markerSource === 'gene_set' && (
                      <select style={dark.select} value={geneSetId} onChange={(e) => setGeneSetId(e.target.value)}>
                        <option value="">Choose…</option>
                        {geneSets.map((g) => <option key={g.id} value={g.id}>{g.name} ({g.genes.length})</option>)}
                      </select>
                    )}
                  </label>
                </>
              )}
              {(markerSource === 'cell_cycle' && usesMarkers) || metrics.has('n_tfs') ? (
                <div style={dark.row}>
                  Species
                  <select style={dark.select} value={species} onChange={(e) => setSpecies(e.target.value)}>
                    <option value="mouse">mouse</option>
                    <option value="human">human</option>
                    <option value="celegans">C. elegans</option>
                  </select>
                  <span style={{ fontSize: 10.5, color: '#666' }}>
                    {status.species ? `guessed from gene names: ${status.species}` : 'could not guess from gene names'}
                    ; matched case-insensitively
                  </span>
                </div>
              ) : null}

              {usesGraph && (
                <>
                  <div style={dark.label}>Neighbourhood</div>
                  <label style={{ ...dark.row, cursor: 'pointer' }}>
                    <input type="radio" name="sfGraph" checked={graph === 'build'} onChange={() => setGraph('build')}
                           disabled={status.pc_embeddings.length === 0} />
                    Build a kNN graph for this run
                  </label>
                  {graph === 'build' && (
                    <div style={{ ...dark.row, marginLeft: 22, flexWrap: 'wrap' }}>
                      <select style={dark.select} value={useRep} onChange={(e) => setUseRep(e.target.value)}>
                        {status.pc_embeddings.map((k) => <option key={k} value={k}>{k} ({status.pc_dims[k]} PCs)</option>)}
                      </select>
                      PCs <input style={{ ...dark.input, width: 50 }} value={nPcs} placeholder="all" onChange={(e) => setNPcs(e.target.value)} />
                      k <input style={{ ...dark.input, width: 70 }} value={nNeighbors} placeholder={`√n = ${defaultK}`} onChange={(e) => setNNeighbors(e.target.value)} />
                    </div>
                  )}
                  {graph === 'build' && (
                    <div style={dark.hint}>
                      stemFinder uses k = √n (here {defaultK.toLocaleString()}), much larger than a clustering graph. The graph is
                      discarded after the run. Ideally the PCA left the cell cycle genes out of its HVGs.
                    </div>
                  )}
                  <label style={{ ...dark.row, cursor: 'pointer' }}>
                    <input type="radio" name="sfGraph" checked={graph === 'existing'} onChange={() => setGraph('existing')}
                           disabled={status.graphs.length === 0} />
                    Use an existing graph
                    {graph === 'existing' && (
                      <select style={dark.select} value={graphKey} onChange={(e) => setGraphKey(e.target.value)}>
                        {status.graphs.map((g) => <option key={g.key} value={g.key}>{g.label} ({g.key})</option>)}
                      </select>
                    )}
                  </label>
                </>
              )}

              <div style={dark.label}>Output</div>
              <div style={dark.row}>
                Column suffix
                <input style={{ ...dark.input, width: 120 }} value={suffix} placeholder="none" onChange={(e) => setSuffix(e.target.value)} />
                <span style={{ fontSize: 10.5, color: '#666' }}>
                  {suffix.trim() ? `→ stemfinder_${suffix.trim().replace(/[^A-Za-z0-9_]+/g, '_').replace(/^_+|_+$/g, '')}` : 'overwrites stemfinder, diffometer, …'}
                </span>
              </div>
              <div style={dark.row}>
                Summarize by
                <select style={dark.select} value={summaryBy} onChange={(e) => setSummaryBy(e.target.value)}>
                  <option value="">(none)</option>
                  {categoricalColumns.map((c) => <option key={c} value={c}>{c}</option>)}
                </select>
                <span style={{ fontSize: 10.5, color: '#666' }}>per-group medians, least differentiated first</span>
              </div>

              <div style={dark.toggle} onClick={() => setShowAdvanced((v) => !v)}>
                {showAdvanced ? '▼' : '▶'} Advanced
              </div>
              {showAdvanced && (
                <>
                  {metrics.has('stemfinder') && (
                    <div style={dark.row}>
                      stemFinder method
                      <select style={dark.select} value={method} onChange={(e) => setMethod(e.target.value as StemFinderInputs['method'])}>
                        <option value="gini">gini (binarized, default)</option>
                        <option value="stdev">stdev (log-normalized)</option>
                        <option value="variance">variance (log-normalized)</option>
                      </select>
                    </div>
                  )}
                  {(metrics.has('diffometer') || (metrics.has('stemfinder') && method === 'gini')) && (
                    <>
                      <div style={dark.row}>
                        Binarize
                        <select style={dark.select} value={binarizeOn} onChange={(e) => setBinarizeOn(e.target.value as StemFinderInputs['binarizeOn'])}>
                          <option value="scaled">scaled expression (above each gene&apos;s mean)</option>
                          <option value="log_normalized">log-normalized expression (detected)</option>
                        </select>
                        at
                        <input style={{ ...dark.input, width: 50 }} value={threshold} onChange={(e) => setThreshold(e.target.value)} />
                      </div>
                    </>
                  )}
                  {metrics.has('diffometer') && (
                    <div style={dark.row}>
                      diffOmeter weights
                      <select style={dark.select} value={weightBy} onChange={(e) => setWeightBy(e.target.value as StemFinderInputs['weightBy'])}>
                        <option value="equal">equal</option>
                        <option value="presence">presence (fraction of cells above threshold)</option>
                        <option value="expression" disabled={binarizeOn === 'scaled'}>mean expression</option>
                      </select>
                      <label style={{ cursor: 'pointer' }}>
                        <input type="checkbox" checked={includeSelf} onChange={(e) => setIncludeSelf(e.target.checked)} /> include the cell
                      </label>
                    </div>
                  )}
                  <div style={dark.row}>
                    Expression
                    <select style={dark.select} value={layer} onChange={(e) => setLayer(e.target.value)}>
                      <option value="">X ({status.x_scale.replace(/_/g, ' ')})</option>
                      {status.layers.map((l) => <option key={l} value={l}>layer: {l}</option>)}
                    </select>
                  </div>
                  <div style={dark.hint}>
                    Counts are normalized to 10,000 and log1p&apos;d; log-scale data is used as is. Scaling is per gene, over the scored cells.
                  </div>
                </>
              )}

              {error && <div style={{ ...dark.notice, marginTop: 10, background: 'rgba(233,69,96,0.12)', color: '#f3a3b2' }}>{error}</div>}

              {result && (
                <div style={{ ...dark.notice, marginTop: 10, background: 'rgba(78,205,196,0.08)', color: '#bfeae6' }}>
                  Scored {result.n_cells_scored.toLocaleString()} cells
                  {result.n_markers_used ? ` with ${result.n_markers_used} markers` : ''}
                  {result.n_markers_missing ? ` (${result.n_markers_missing} not in this dataset)` : ''}
                  {result.n_neighbors ? `, k = ${result.n_neighbors}` : result.graph_source ? `, ${result.graph_source} graph` : ''}
                  {result.n_tfs_used ? `; ${result.n_tfs_used} TFs present` : ''}.
                  <div style={{ display: 'flex', gap: 6, flexWrap: 'wrap', marginTop: 6 }}>
                    {result.columns.map((c) => (
                      <button key={c} style={dark.chip} onClick={() => colorBy(c)}
                              title={`Colour by ${c}: median ${fmt(result.stats[c]?.median)}, range ${fmt(result.stats[c]?.min)}–${fmt(result.stats[c]?.max)}`}>
                        {c}
                      </button>
                    ))}
                  </div>
                  {result.summary && result.summary.length > 0 && (
                    <table style={{ ...dark.table, marginTop: 8 }}>
                      <thead>
                        <tr>
                          <th style={dark.th}>{summaryBy || 'group'}</th>
                          <th style={dark.th}>cells</th>
                          {result.columns.map((c) => <th key={c} style={dark.th}>{c}</th>)}
                        </tr>
                      </thead>
                      <tbody>
                        {result.summary.map((g) => (
                          <tr key={g.group}>
                            <td style={dark.td}>{g.group}</td>
                            <td style={dark.td}>{g.n.toLocaleString()}</td>
                            {result.columns.map((c) => <td key={c} style={dark.td}>{fmt(g.medians[c])}</td>)}
                          </tr>
                        ))}
                      </tbody>
                    </table>
                  )}
                </div>
              )}
            </>
          )}
        </div>

        <div style={dark.actions}>
          <span style={{ flex: 1, fontSize: 11, color: '#888' }}>
            {busy ? progress : blocker && status?.available ? blocker : status?.version ? `pystemfinder ${status.version}` : ''}
          </span>
          {busy ? (
            <button style={dark.ghost} onClick={cancel} disabled={!taskId}>Cancel</button>
          ) : (
            <button style={dark.ghost} onClick={close}>Close</button>
          )}
          <button style={{ ...dark.primary, opacity: blocker || busy ? 0.5 : 1 }} onClick={run} disabled={!!blocker || busy}
                  title={blocker ?? 'Score the cells and colour by the first score'}>
            {busy ? 'Running…' : 'Run'}
          </button>
        </div>
      </div>
    </div>
  )
}
