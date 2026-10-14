/**
 * CNV inference (Analyze → Cells → CNV inference), inferCNV-style.
 *
 * Expression relative to reference cells, clipped, smoothed in windows along
 * each chromosome, centred per cell, then clustered into clones. The backend
 * (xcell/cnv.py) does the work; this modal chooses where gene positions come
 * from, which cells are the normal reference, and how clones are called. It
 * writes X_<key>, <key>_score and <key>_clone, and can save a CNV heatmap
 * figure of the result.
 */

import { useCallback, useEffect, useMemo, useState } from 'react'
import { useStore } from '../store'
import { appendDataset, cancelTask, createFigure, pollTask, refreshSchema } from '../hooks/useData'
import { datasetIdentity } from '../lib/datasetIdentity'
import { indicesFromMask } from '../lib/cellSubsets'
import { CNV_DEFAULTS, cleanKey, cnvBlocker, cnvParams, type CnvCluster, type CnvInputs } from '../lib/cnv'

interface Status {
  n_cells: number
  n_genes: number
  species: string | null
  table_species: string[]
  var_columns: { name: string; kind: 'numeric' | 'category' }[]
  obs_categorical: { name: string; categories: string[] }[]
  layers: string[]
  x_scale: string
  keys: string[]
  hierarchical_max_cells: number
}

interface CloneRow {
  clone: number
  name: string
  n_cells: number
  mean_score: number | null
  reference_fraction: number | null
  top_group: string | null
  top_group_fraction: number | null
  chromosome_means: Record<string, number | null>
}

interface Result {
  key: string
  species: string | null
  positions_source: string
  position_matches: Record<string, number>
  n_genes_placed: number
  n_genes_lineage_excluded?: number
  n_genes_filtered: number
  n_genes_used: number
  n_windows: number
  n_cells: number
  reference: { column: string; categories: string[]; counts?: Record<string, number>; bounds_categories?: string[]; pooled?: boolean } | null
  chromosomes: { name: string; start: number; size: number }[]
  score_column: string
  clone_column: string | null
  chromosome_matrix: string
  clones: CloneRow[] | null
  warnings: string[]
}

const dark = {
  overlay: {
    position: 'fixed' as const, inset: 0, background: 'rgba(0,0,0,0.6)',
    display: 'flex', alignItems: 'center', justifyContent: 'center', zIndex: 1000,
  },
  panel: {
    background: '#16213e', border: '1px solid #0f3460', borderRadius: 8,
    width: 'min(620px, 94vw)', maxHeight: '90vh', display: 'flex',
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
  row: { display: 'flex', alignItems: 'center', gap: 8, marginBottom: 6, fontSize: 12, color: '#ccc', flexWrap: 'wrap' as const },
  input: {
    background: '#0f1625', border: '1px solid #0f3460', borderRadius: 4,
    color: '#ddd', fontSize: 12, padding: '5px 8px', outline: 'none', width: 70,
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
  catBox: {
    maxHeight: 130, overflowY: 'auto' as const, border: '1px solid #0f3460', borderRadius: 4,
    padding: '4px 8px', marginLeft: 22, marginBottom: 6, background: '#0f1625',
  },
  table: { width: '100%', borderCollapse: 'collapse' as const, fontSize: 11.5, color: '#ccc' },
  th: { textAlign: 'left' as const, color: '#888', fontWeight: 500, padding: '3px 6px', borderBottom: '1px solid #0f3460' },
  td: { padding: '3px 6px', borderBottom: '1px solid rgba(15,52,96,0.5)' },
}

const fmt = (v: number | null | undefined, digits = 3) => (v === null || v === undefined ? '—' : v.toFixed(digits))
const pct = (v: number | null | undefined) => (v === null || v === undefined ? '—' : `${Math.round(v * 100)}%`)

const CHROM_GUESSES = ['chromosome', 'chrom', 'chr', 'seqname', 'seqnames', 'chromosome_name']
const START_GUESSES = ['start', 'start_position', 'gene_start', 'txstart', 'begin']

export default function CnvModal() {
  const isOpen = useStore((s) => s.isCnvModalOpen)
  const setOpen = useStore((s) => s.setCnvModalOpen)
  const activeSlot = useStore((s) => s.activeSlot)
  const schema = useStore((s) => s.schema)
  const activeCellMask = useStore((s) => s.activeCellMask)
  const activeSubsetName = useStore((s) => s.activeSubsetName)
  const selectColorColumn = useStore((s) => s.setSelectedColorColumn)
  const setColorMode = useStore((s) => s.setColorMode)
  const refreshObsSummaries = useStore((s) => s.refreshObsSummaries)
  const refreshFigures = useStore((s) => s.refreshFigures)
  const setActiveFigureId = useStore((s) => s.setActiveFigureId)
  const setCenterPanelView = useStore((s) => s.setCenterPanelView)
  const dataset = datasetIdentity(activeSlot, schema)

  const [status, setStatus] = useState<Status | null>(null)
  const [statusError, setStatusError] = useState<string | null>(null)
  const [key, setKey] = useState('cnv')
  const [positions, setPositions] = useState<'table' | 'columns'>('table')
  const [species, setSpecies] = useState('')
  const [chromosomeColumn, setChromosomeColumn] = useState('')
  const [startColumn, setStartColumn] = useState('')
  const [referenceColumn, setReferenceColumn] = useState('')
  const [referenceCategories, setReferenceCategories] = useState<Set<string>>(() => new Set())
  const [includeXY, setIncludeXY] = useState(false)
  const [cluster, setCluster] = useState<CnvCluster>('leiden')
  const [resolution, setResolution] = useState(CNV_DEFAULTS.resolution)
  const [nClusters, setNClusters] = useState(CNV_DEFAULTS.nClusters)
  const [nPcs, setNPcs] = useState(CNV_DEFAULTS.nPcs)
  const [window_, setWindow] = useState(CNV_DEFAULTS.window)
  const [step, setStep] = useState(CNV_DEFAULTS.step)
  const [minMean, setMinMean] = useState(CNV_DEFAULTS.minMean)
  const [lfcClip, setLfcClip] = useState(CNV_DEFAULTS.lfcClip)
  const [noiseThreshold, setNoiseThreshold] = useState(CNV_DEFAULTS.noiseThreshold)
  const [centerCells, setCenterCells] = useState(true)
  const [excludeLineageGenes, setExcludeLineageGenes] = useState(true)
  const [minReferenceCells, setMinReferenceCells] = useState(CNV_DEFAULTS.minReferenceCells)
  const [layer, setLayer] = useState('')
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
    fetch(appendDataset('/api/cnv/status', activeSlot))
      .then(async (r) => {
        const data = await r.json()
        if (!r.ok) throw new Error(data.detail || 'Could not read the dataset')
        return data as Status
      })
      .then((s) => {
        if (cancelled) return
        setStatus(s)
        const cats = s.var_columns.filter((c) => c.kind === 'category').map((c) => c.name)
        const nums = s.var_columns.filter((c) => c.kind === 'numeric').map((c) => c.name)
        const chromGuess = cats.find((c) => CHROM_GUESSES.includes(c.toLowerCase())) ?? ''
        const startGuess = nums.find((c) => START_GUESSES.includes(c.toLowerCase())) ?? ''
        const tableOk = !!s.species && s.table_species.includes(s.species)
        setPositions(tableOk || !(chromGuess && startGuess) ? 'table' : 'columns')
        setSpecies(s.species ?? (s.table_species[0] ?? ''))
        setChromosomeColumn(chromGuess); setStartColumn(startGuess)
        setReferenceColumn(''); setReferenceCategories(new Set())
        setLayer(''); setKey(activeCellMask && activeSubsetName ? `cnv_${cleanKey(activeSubsetName)}` : 'cnv')
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

  const refCategories = useMemo(
    () => status?.obs_categorical.find((c) => c.name === referenceColumn)?.categories ?? [],
    [status, referenceColumn],
  )

  const inputs: CnvInputs = {
    key, positions, species, chromosomeColumn, startColumn, referenceColumn, referenceCategories, includeXY,
    cluster, resolution, nClusters, nPcs, window: window_, step, minMean, lfcClip, noiseThreshold, centerCells, layer,
    excludeLineageGenes, minReferenceCells,
    activeCellIndices: null,   // filled at Run; O(n) is not worth paying every render
    tableSpecies: status?.table_species,
    nCells: activeCellMask ? nActive : status?.n_cells,
    hierarchicalMaxCells: status?.hierarchical_max_cells,
  }
  const blocker = status ? cnvBlocker(inputs) : 'Reading the dataset…'

  const run = useCallback(async () => {
    if (blocker) return
    setBusy(true); setError(null); setResult(null); setProgress('Starting…')
    try {
      const resp = await fetch(appendDataset('/api/cnv', activeSlot), {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(cnvParams({
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
      setStatus((s) => (s && !s.keys.includes(res.key) ? { ...s, keys: [...s.keys, res.key].sort() } : s))
      // New .obs columns and a new score matrix: the schema and the Cell Manager lists.
      await refreshSchema()
      refreshObsSummaries()
      selectColorColumn(res.clone_column ?? res.score_column)
      setColorMode('metadata')
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

  const colorBy = (column: string) => {
    selectColorColumn(column)
    setColorMode('metadata')
  }

  const makeFigure = useCallback(async (cnvKey: string, obsColumn: string | null) => {
    setError(null)
    try {
      const rec = await createFigure({ kind: 'cnv_heatmap', inputs: { key: cnvKey, obs_column: obsColumn } }, activeSlot)
      refreshFigures(); setActiveFigureId(rec.id); setCenterPanelView('figures'); setOpen(false)
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e))
    }
  }, [activeSlot, refreshFigures, setActiveFigureId, setCenterPanelView, setOpen])

  const toggleCategory = (c: string) => setReferenceCategories((prev) => {
    const next = new Set(prev)
    if (next.has(c)) next.delete(c); else next.add(c)
    return next
  })

  if (!isOpen) return null

  const close = () => { if (!busy) setOpen(false) }
  const cleaned = cleanKey(key) || 'cnv'
  const catColumns = status?.var_columns.filter((c) => c.kind === 'category') ?? []
  const numColumns = status?.var_columns.filter((c) => c.kind === 'numeric') ?? []

  return (
    <div style={dark.overlay} onClick={close}>
      <div style={dark.panel} onClick={(e) => e.stopPropagation()}>
        <div style={dark.header}>
          <h2 style={dark.title}>CNV inference</h2>
          <button style={dark.close} onClick={close}>&times;</button>
        </div>

        <div style={dark.body}>
          <div style={{ fontSize: 11.5, color: '#999', lineHeight: 1.45, marginBottom: 8 }}>
            A chromosomal gain or loss nudges the expression of most genes in the region. Each gene is compared
            with reference cells, clipped, and averaged in windows along the chromosome (inferCNV; Tirosh et al. 2016).
            Cells with the same gains and losses are then clustered into clones.
          </div>

          {statusError && <div style={{ ...dark.notice, background: 'rgba(233,69,96,0.12)', color: '#f3a3b2' }}>{statusError}</div>}
          {!status && !statusError && <div style={{ fontSize: 12, color: '#888' }}>Reading the dataset…</div>}

          {status && (
            <>
              {activeCellMask && (
                <div style={{ ...dark.notice, border: '1px solid #0f3460', color: '#e9a23b' }}>
                  Scoped to the cell mask: {nActive.toLocaleString()} active cells
                  {activeSubsetName ? ` (${activeSubsetName})` : ''}. Reference means, the gene filter and the clones use
                  these cells only; cells outside get no values.
                </div>
              )}

              <div style={dark.label}>Gene positions</div>
              <label style={{ ...dark.row, cursor: 'pointer' }}>
                <input type="radio" name="cnvPos" checked={positions === 'table'} onChange={() => setPositions('table')} />
                Shipped GENCODE table
                {positions === 'table' && (
                  <select style={dark.select} value={species} onChange={(e) => setSpecies(e.target.value)}>
                    <option value="">Choose species…</option>
                    {status.table_species.map((sp) => <option key={sp} value={sp}>{sp}</option>)}
                  </select>
                )}
                {positions === 'table' && (
                  <span style={{ fontSize: 10.5, color: '#666' }}>
                    {status.species ? `guessed from gene names: ${status.species}` : 'could not guess the species from gene names'}
                  </span>
                )}
              </label>
              {positions === 'table' && (
                <div style={dark.hint}>Genes match by Ensembl ID or symbol (case-insensitive). GRCh38 / GRCm39 coordinates.</div>
              )}
              <label style={{ ...dark.row, cursor: 'pointer' }}>
                <input type="radio" name="cnvPos" checked={positions === 'columns'} onChange={() => setPositions('columns')}
                       disabled={catColumns.length === 0 || numColumns.length === 0} />
                Columns of .var
                {positions === 'columns' && (
                  <>
                    chromosome
                    <select style={dark.select} value={chromosomeColumn} onChange={(e) => setChromosomeColumn(e.target.value)}>
                      <option value="">Choose…</option>
                      {catColumns.map((c) => <option key={c.name} value={c.name}>{c.name}</option>)}
                    </select>
                    start
                    <select style={dark.select} value={startColumn} onChange={(e) => setStartColumn(e.target.value)}>
                      <option value="">Choose…</option>
                      {numColumns.map((c) => <option key={c.name} value={c.name}>{c.name}</option>)}
                    </select>
                  </>
                )}
              </label>
              <label style={{ ...dark.row, cursor: 'pointer', marginLeft: 22 }}>
                <input type="checkbox" checked={includeXY} onChange={(e) => setIncludeXY(e.target.checked)} />
                Include X and Y
                <span style={{ fontSize: 10.5, color: '#666' }}>off by default: a reference of the other sex reads as a whole-chromosome event</span>
              </label>

              <div style={dark.label}>Reference (normal) cells</div>
              <div style={dark.row}>
                Column
                <select style={dark.select} value={referenceColumn}
                        onChange={(e) => { setReferenceColumn(e.target.value); setReferenceCategories(new Set()) }}>
                  <option value="">(none: compare with the mean over all cells)</option>
                  {status.obs_categorical.map((c) => <option key={c.name} value={c.name}>{c.name}</option>)}
                </select>
              </div>
              {referenceColumn && (
                <div style={dark.catBox}>
                  {refCategories.map((c) => (
                    <label key={c} style={{ display: 'flex', alignItems: 'center', gap: 6, fontSize: 11.5, color: '#ccc', cursor: 'pointer', padding: '1px 0' }}>
                      <input type="checkbox" checked={referenceCategories.has(c)} onChange={() => toggleCategory(c)} />
                      {c}
                    </label>
                  ))}
                </div>
              )}
              <div style={dark.hint}>
                {referenceColumn
                  ? 'Tick the categories you trust to be diploid (immune, stromal, endothelial cells). With several, a gene within the range of their means counts as normal.'
                  : 'Without a reference, events shared by every cell are invisible and the quietest clone is not necessarily normal. Works when most cells are normal, or clones differ.'}
              </div>

              <div style={dark.label}>Clones</div>
              <div style={dark.row}>
                <select style={dark.select} value={cluster} onChange={(e) => setCluster(e.target.value as CnvCluster)}>
                  <option value="leiden">Leiden on the CNV profile</option>
                  <option value="hierarchical">Hierarchical (Ward), fixed number</option>
                  <option value="none">Do not call clones</option>
                </select>
                {cluster === 'leiden' && (
                  <>resolution <input style={dark.input} value={resolution} onChange={(e) => setResolution(e.target.value)} /></>
                )}
                {cluster === 'hierarchical' && (
                  <>clones <input style={dark.input} value={nClusters} onChange={(e) => setNClusters(e.target.value)} /></>
                )}
                {cluster !== 'none' && (
                  <>PCs <input style={{ ...dark.input, width: 50 }} value={nPcs} onChange={(e) => setNPcs(e.target.value)} /></>
                )}
              </div>
              <div style={dark.hint}>
                Clones are named by rising CNV burden: clone_1 is the quietest. With a reference, the table below shows how much of it
                each clone holds, so a tumor clone has none and a normal one most.
              </div>

              <div style={dark.label}>Output</div>
              <div style={dark.row}>
                Key
                <input style={{ ...dark.input, width: 120 }} value={key} onChange={(e) => setKey(e.target.value)} />
                <span style={{ fontSize: 10.5, color: status.keys.includes(cleaned) ? '#e9a23b' : '#666' }}>
                  → X_{cleaned}, {cleaned}_score{cluster !== 'none' ? `, ${cleaned}_clone` : ''}
                  {status.keys.includes(cleaned) ? ' (replaces the stored run of this name)' : ''}
                </span>
              </div>

              <div style={dark.toggle} onClick={() => setShowAdvanced((v) => !v)}>
                {showAdvanced ? '▼' : '▶'} Advanced
              </div>
              {showAdvanced && (
                <>
                  <div style={dark.row}>
                    Window <input style={dark.input} value={window_} onChange={(e) => setWindow(e.target.value)} /> genes,
                    every <input style={dark.input} value={step} onChange={(e) => setStep(e.target.value)} /> genes
                  </div>
                  <div style={dark.hint}>inferCNV smooths 101 genes at every gene; infercnvpy 100 every 10. A smaller step gives finer breakpoints and more windows.</div>
                  <div style={dark.row}>
                    Keep genes with mean log expression ≥ <input style={dark.input} value={minMean} onChange={(e) => setMinMean(e.target.value)} />
                  </div>
                  <div style={dark.row}>
                    Clip at ± <input style={dark.input} value={lfcClip} onChange={(e) => setLfcClip(e.target.value)} />
                    noise threshold <input style={dark.input} value={noiseThreshold} onChange={(e) => setNoiseThreshold(e.target.value)} /> × SD
                  </div>
                  <div style={dark.hint}>Window values below the threshold × the matrix-wide standard deviation are set to 0 (inferCNV&apos;s dynamic threshold). 0 disables.</div>
                  <label style={{ ...dark.row, cursor: 'pointer' }}>
                    <input type="checkbox" checked={centerCells} onChange={(e) => setCenterCells(e.target.checked)} />
                    Centre each cell at its median window value
                  </label>
                  <label style={{ ...dark.row, cursor: 'pointer' }}>
                    <input type="checkbox" checked={excludeLineageGenes} onChange={(e) => setExcludeLineageGenes(e.target.checked)} />
                    Exclude immunoglobulin, T-cell receptor and MHC genes
                  </label>
                  <div style={dark.hint}>Clonally rearranged, lineage-restricted loci read as focal events in every plasma, B or T cell.</div>
                  <div style={dark.row}>
                    A reference category needs ≥ <input style={dark.input} value={minReferenceCells} onChange={(e) => setMinReferenceCells(e.target.value)} /> cells to set the bounds
                  </div>
                  <div style={dark.hint}>Smaller categories still count as reference cells; their noisy means would widen the bounds and hide real events. If none qualifies, all reference cells are pooled.</div>
                  <div style={dark.row}>
                    Expression
                    <select style={dark.select} value={layer} onChange={(e) => setLayer(e.target.value)}>
                      <option value="">X ({status.x_scale.replace(/_/g, ' ')})</option>
                      {status.layers.map((l) => <option key={l} value={l}>layer: {l}</option>)}
                    </select>
                  </div>
                  <div style={dark.hint}>Counts are normalized to 10,000 and log1p&apos;d; log-scale data is used as is. Scaled data is refused.</div>
                </>
              )}

              {status.keys.length > 0 && !result && (
                <div style={{ ...dark.notice, marginTop: 10, border: '1px solid #0f3460', color: '#aaa' }}>
                  Stored runs: {status.keys.map((k) => (
                    <button key={k} style={{ ...dark.chip, marginLeft: 6 }}
                            onClick={() => makeFigure(k, schema?.obs_dtypes?.[`${k}_clone`] ? `${k}_clone` : null)}
                            title={`Save a CNV heatmap figure of ${k}${schema?.obs_dtypes?.[`${k}_clone`] ? `, grouped by ${k}_clone` : ''}`}>
                      {k}: heatmap figure…
                    </button>
                  ))}
                </div>
              )}

              {error && <div style={{ ...dark.notice, marginTop: 10, background: 'rgba(233,69,96,0.12)', color: '#f3a3b2' }}>{error}</div>}

              {result && (
                <div style={{ ...dark.notice, marginTop: 10, background: 'rgba(78,205,196,0.08)', color: '#bfeae6' }}>
                  {result.n_genes_used.toLocaleString()} genes in {result.n_windows.toLocaleString()} windows over{' '}
                  {result.chromosomes.length} chromosomes, for {result.n_cells.toLocaleString()} cells
                  {result.n_genes_filtered || result.n_genes_lineage_excluded
                    ? ` (${[result.n_genes_filtered ? `${result.n_genes_filtered.toLocaleString()} placed genes below the expression floor` : '', result.n_genes_lineage_excluded ? `${result.n_genes_lineage_excluded.toLocaleString()} Ig/TCR/MHC genes excluded` : ''].filter(Boolean).join(', ')})`
                    : ''}.
                  {' '}Positions: {result.positions_source.replace('_', ' ')}.
                  {result.warnings.map((w) => (
                    <div key={w} style={{ marginTop: 6, color: '#e9a23b' }}>{w}</div>
                  ))}
                  <div style={{ display: 'flex', gap: 6, flexWrap: 'wrap', marginTop: 8 }}>
                    {result.clone_column && (
                      <button style={dark.chip} onClick={() => colorBy(result.clone_column!)}>Colour by {result.clone_column}</button>
                    )}
                    <button style={dark.chip} onClick={() => colorBy(result.score_column)}>Colour by {result.score_column}</button>
                    <button style={{ ...dark.chip, borderColor: '#e9a23b', color: '#e9a23b' }}
                            onClick={() => makeFigure(result.key, result.clone_column)}>
                      Heatmap figure…
                    </button>
                  </div>
                  <div style={{ fontSize: 10.5, color: '#888', marginTop: 6 }}>
                    Per-chromosome means are in the score matrix {result.chromosome_matrix} (Colour → score matrices), so the
                    embedding can be coloured by one chromosome.
                  </div>
                  {result.clones && result.clones.length > 0 && (
                    <table style={{ ...dark.table, marginTop: 8 }}>
                      <thead>
                        <tr>
                          <th style={dark.th}>clone</th>
                          <th style={dark.th}>cells</th>
                          <th style={dark.th}>CNV score</th>
                          {result.reference && <th style={dark.th}>reference cells</th>}
                          {result.reference && <th style={dark.th}>mostly</th>}
                        </tr>
                      </thead>
                      <tbody>
                        {result.clones.map((c) => (
                          <tr key={c.name}>
                            <td style={dark.td}>{c.name}</td>
                            <td style={dark.td}>{c.n_cells.toLocaleString()}</td>
                            <td style={dark.td}>{fmt(c.mean_score)}</td>
                            {result.reference && <td style={dark.td}>{pct(c.reference_fraction)}</td>}
                            {result.reference && <td style={dark.td}>{c.top_group ?? '—'} ({pct(c.top_group_fraction)})</td>}
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
            {busy ? progress : blocker && status ? blocker : ''}
          </span>
          {busy ? (
            <button style={dark.ghost} onClick={cancel} disabled={!taskId}>Cancel</button>
          ) : (
            <button style={dark.ghost} onClick={close}>Close</button>
          )}
          <button style={{ ...dark.primary, opacity: blocker || busy ? 0.5 : 1 }} onClick={run} disabled={!!blocker || busy}
                  title={blocker ?? 'Infer CNVs, call clones, and colour by clone'}>
            {busy ? 'Running…' : 'Run'}
          </button>
        </div>
      </div>
    </div>
  )
}
