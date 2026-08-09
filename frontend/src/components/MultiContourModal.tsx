/**
 * Contour tool (Analyze → Spatial → Contour).
 *
 * Unified single- and multi-gene-set spatial contouring:
 *   - 1 source  → a banded expression column (classic contourize).
 *   - ≥2 sources → fused tissue annotation: contour each, review/bin the "high"
 *                  cutoff per module, then assign each spot a tissue (overlaps
 *                  resolved by spatial + X_pca neighbors). Requires X_pca.
 *
 * Sources come from a gene-set picker popup (so many sets don't crowd the form);
 * the current Gene Panel selection is also offered as a source. Grid resolution
 * and smoothing sigma are prefilled from data-aware suggestions.
 *
 * Backend: /scanpy/contour_suggest, /scanpy/contourize (single),
 * /scanpy/multicontour/prepare + /finalize (multi).
 *
 * Rollback: delete this file, remove its mount + Spatial "Contour" launcher in
 * ScanpyModal, and the isMultiContourModalOpen state in store.ts.
 */

import { useEffect, useMemo, useState } from 'react'
import { useStore, GeneSet, GeneSetCategoryType, cfgDefault } from '../store'
import { appendDataset, pollTask } from '../hooks/useData'
import {
  adviseContour, adviseModules, type Advice, type ContourGeometry,
} from '../lib/contourAdvice'

const API_BASE = '/api'
const SELECTION_KEY = '__selection__'

const CATEGORY_ORDER: GeneSetCategoryType[] = ['manual', 'gene_clusters', 'similar_genes', 'diff_exp', 'spatial', 'marker_genes', 'line_association']

interface Source {
  key: string
  name: string
  genes: string[]
}

function getAllGeneSets(
  categories: Record<GeneSetCategoryType, { geneSets: GeneSet[]; folders: { name: string; geneSets: GeneSet[] }[] }>
): Source[] {
  const all: Source[] = []
  for (const catType of CATEGORY_ORDER) {
    const cat = categories[catType]
    for (const gs of cat.geneSets) all.push({ key: gs.id, name: gs.name, genes: gs.genes })
    for (const folder of cat.folders) {
      for (const gs of folder.geneSets) all.push({ key: gs.id, name: `${folder.name} / ${gs.name}`, genes: gs.genes })
    }
  }
  return all
}

interface ModuleReview {
  name: string
  n_genes: number
  thresholds: number[]
  band_values: number[]
  histogram: number[]
  auto_cutoff: number
}

interface PrepareResult {
  token: string
  modules: ModuleReview[]
  params: Record<string, unknown>
  missing_genes?: Record<string, string[]>  // per-module genes dropped (not in dataset)
}

interface DoneResult {
  annotation_key: string
  missing_genes?: string[]  // single-contour genes dropped (not in dataset)
  // multi
  categories?: string[]
  counts?: Record<string, number>
  n_resolved?: number
  // single
  contour_levels?: number
  n_genes?: number
}

type Phase = 'select' | 'review' | 'done'

export default function MultiContourModal() {
  const isOpen = useStore((s) => s.isMultiContourModalOpen)
  const setOpen = useStore((s) => s.setMultiContourModalOpen)
  const geneSetCategories = useStore((s) => s.geneSetCategories)
  const selectedGenes = useStore((s) => s.selectedGenes)
  const setSelectedColorColumn = useStore((s) => s.setSelectedColorColumn)
  const refreshObsSummaries = useStore((s) => s.refreshObsSummaries)
  const addScanpyAction = useStore((s) => s.addScanpyAction)

  const namedSets = useMemo(() => getAllGeneSets(geneSetCategories).filter((g) => g.genes.length > 0), [geneSetCategories])

  const [phase, setPhase] = useState<Phase>('select')
  const [pickerOpen, setPickerOpen] = useState(false)
  const [sources, setSources] = useState<Source[]>([])
  const [contourLevels, setContourLevels] = useState(() => cfgDefault(['contour', 'contour_levels'], 3))
  const [gridRes, setGridRes] = useState<string>('')
  const [smoothSigma, setSmoothSigma] = useState<string>('')
  const [logTransform, setLogTransform] = useState(() => cfgDefault(['contour', 'log_transform'], false))
  const [columnName, setColumnName] = useState('') // single path; blank = auto

  const [busy, setBusy] = useState(false)
  const [error, setError] = useState<string | null>(null)

  const [prep, setPrep] = useState<PrepareResult | null>(null)
  const [cutoffs, setCutoffs] = useState<Record<string, number>>({})
  const [profileK, setProfileK] = useState(() => cfgDefault(['contour', 'profile_k'], 15))
  const [outName, setOutName] = useState(() => cfgDefault(['contour', 'out_name'], 'tissue'))
  const [saveQc, setSaveQc] = useState(() => cfgDefault(['contour', 'save_qc'], false))
  const [doneResult, setDoneResult] = useState<DoneResult | null>(null)

  const [obsColumns, setObsColumns] = useState<string[]>([])
  const [sectionCol, setSectionCol] = useState<string>('')

  // Inputs the advice reasons from. geom carries the spot spacing and extent —
  // without them nothing can be said about σ, which is measured in grid pixels.
  const [geom, setGeom] = useState<ContourGeometry | null>(null)
  const [sectionCandidates, setSectionCandidates] = useState<string[]>([])
  const [matrixScale, setMatrixScale] = useState<string | null>(null)
  const [matrixMax, setMatrixMax] = useState<number | null>(null)
  const [showGuide, setShowGuide] = useState(false)
  const hasPca = !!useStore((s) => s.schema)?.embeddings?.includes('X_pca')

  // Load categorical obs columns; auto-detect a section column.
  useEffect(() => {
    if (!isOpen) return
    let cancelled = false
    fetch(appendDataset(`${API_BASE}/obs/summaries`))
      .then((r) => (r.ok ? r.json() : null))
      .then((d) => {
        if (cancelled || !d) return
        const cols: string[] = (d.summaries || d || [])
          .filter((s: { dtype?: string }) => s.dtype === 'category')
          .map((s: { name: string }) => s.name)
        setObsColumns(cols)
        // A section column has a handful of levels, not hundreds — a per-cell
        // barcode column is categorical too and would be a useless suggestion.
        setSectionCandidates(
          (d.summaries || d || [])
            .filter((s: { dtype?: string; categories?: unknown[] }) =>
              s.dtype === 'category'
              && (s.categories?.length ?? 0) >= 2
              && (s.categories?.length ?? 0) <= 20)
            .map((s: { name: string }) => s.name),
        )
        const detected = cols.find((c) => c.toLowerCase() === 'section')
          || cols.find((c) => c.toLowerCase() === 'sample')
        if (detected) setSectionCol((prev) => (prev ? prev : detected))
      })
      .catch(() => {})
    return () => { cancelled = true }
  }, [isOpen])

  // Whether .X holds raw counts is the one thing the log-transform checkbox
  // cannot be answered without, and datasets rarely say. list_layers already
  // attaches layer_scale's verdict, so no new endpoint is needed.
  useEffect(() => {
    if (!isOpen) return
    let cancelled = false
    fetch(appendDataset(`${API_BASE}/scanpy/layers`))
      .then((r) => (r.ok ? r.json() : null))
      .then((d) => {
        if (cancelled || !d) return
        const x = (d.layers || []).find((l: { name: string }) => l.name === 'X')
        setMatrixScale(x?.scale?.verdict ?? null)
        setMatrixMax(x?.scale?.stats?.max ?? null)
      })
      .catch(() => {})
    return () => { cancelled = true }
  }, [isOpen])

  // Prefill data-aware suggestions when the modal opens.
  useEffect(() => {
    if (!isOpen) return
    let cancelled = false
    fetch(appendDataset(`${API_BASE}/scanpy/contour_suggest`))
      .then((r) => (r.ok ? r.json() : null))
      .then((d) => {
        if (cancelled || !d) return
        setGridRes((prev) => (prev ? prev : String(d.grid_res)))
        setSmoothSigma((prev) => (prev ? prev : String(d.smooth_sigma)))
        setGeom({
          nSpots: d.n_spots, medianSpacing: d.median_spacing, extent: d.extent,
          suggestedGridRes: d.grid_res, suggestedSigma: d.smooth_sigma,
        })
      })
      .catch(() => {})
    return () => { cancelled = true }
  }, [isOpen])

  if (!isOpen) return null

  const close = () => {
    setOpen(false)
    setPhase('select'); setPickerOpen(false); setSources([]); setPrep(null)
    setCutoffs({}); setDoneResult(null); setError(null); setBusy(false)
    setGridRes(''); setSmoothSigma(''); setColumnName('')
  }

  const isPicked = (key: string) => sources.some((s) => s.key === key)
  const togglePick = (src: Source) => {
    setSources((prev) => (prev.some((s) => s.key === src.key) ? prev.filter((s) => s.key !== src.key) : [...prev, src]))
  }

  const commonParams = () => {
    const body: Record<string, unknown> = { contour_levels: contourLevels, log_transform: logTransform }
    if (gridRes.trim()) body.grid_res = parseInt(gridRes, 10)
    if (smoothSigma.trim()) body.smooth_sigma = parseFloat(smoothSigma)
    if (sectionCol) body.section_col = sectionCol
    return body
  }

  // 1 source → classic single contour (banded column)
  const runSingle = async () => {
    setError(null); setBusy(true)
    try {
      const body: Record<string, unknown> = {
        ...commonParams(), genes: sources[0].genes,
      }
      if (columnName.trim()) body.annotation_key = columnName.trim()
      const resp = await fetch(appendDataset(`${API_BASE}/scanpy/contourize`), {
        method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(body),
      })
      if (!resp.ok) { setError((await resp.json()).detail || 'Contour failed'); setBusy(false); return }
      const { task_id } = await resp.json()
      const task = await pollTask(task_id)
      if (task.status !== 'completed') { setError(task.error || `Contour ${task.status}`); setBusy(false); return }
      const result = task.result as unknown as DoneResult
      finishDone(result, 'contourize', { genes: sources[0].name })
    } catch (e) {
      setError((e as Error).message)
    } finally {
      setBusy(false)
    }
  }

  // ≥2 sources → multi-contour fusion (prepare → review)
  const runPrepare = async () => {
    setError(null); setBusy(true)
    try {
      const geneSets: Record<string, string[]> = {}
      for (const s of sources) geneSets[s.name] = s.genes
      const resp = await fetch(appendDataset(`${API_BASE}/scanpy/multicontour/prepare`), {
        method: 'POST', headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ ...commonParams(), gene_sets: geneSets }),
      })
      if (!resp.ok) { setError((await resp.json()).detail || 'Prepare failed'); setBusy(false); return }
      const { task_id } = await resp.json()
      const task = await pollTask(task_id)
      if (task.status !== 'completed') { setError(task.error || `Prepare ${task.status}`); setBusy(false); return }
      const result = task.result as unknown as PrepareResult
      setPrep(result)
      setCutoffs(Object.fromEntries(result.modules.map((m) => [m.name, m.auto_cutoff])))
      setPhase('review')
    } catch (e) {
      setError((e as Error).message)
    } finally {
      setBusy(false)
    }
  }

  const runFinalize = async () => {
    if (!prep) return
    setError(null); setBusy(true)
    try {
      const resp = await fetch(appendDataset(`${API_BASE}/scanpy/multicontour/finalize`), {
        method: 'POST', headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          token: prep.token, cutoffs, profile_k: profileK,
          out_name: outName.trim() || 'tissue', save_qc: saveQc, params: prep.params,
        }),
      })
      if (!resp.ok) { setError((await resp.json()).detail || 'Finalize failed'); setBusy(false); return }
      const result = (await resp.json()) as DoneResult
      finishDone(result, 'multicontour', { gene_sets: Object.keys(cutoffs), profile_k: profileK })
    } catch (e) {
      setError((e as Error).message)
    } finally {
      setBusy(false)
    }
  }

  const finishDone = (result: DoneResult, action: string, params: Record<string, unknown>) => {
    setDoneResult(result)
    refreshObsSummaries()
    addScanpyAction({
      action, params: { ...params, out: result.annotation_key },
      result: result as unknown as Record<string, unknown>, timestamp: new Date().toISOString(),
    })
    setPhase('done')
  }

  const highCount = (m: ModuleReview, cutoff: number) =>
    m.band_values.reduce((sum, bv, i) => (bv >= cutoff ? sum + m.histogram[i] : sum), 0)

  const nSources = sources.length

  const advice = useMemo(() => adviseContour({
    gridRes: parseInt(gridRes || '0', 10) || 0,
    smoothSigma: parseFloat(smoothSigma || '0') || 0,
    contourLevels, logTransform, sectionCol, sectionCandidates, nSources,
    smallestSourceSize: sources.length
      ? Math.min(...sources.map((s) => s.genes.length)) : null,
    hasPca, matrixScale, matrixMax,
  }, geom), [gridRes, smoothSigma, contourLevels, logTransform, sectionCol,
             sectionCandidates, nSources, sources, hasPca, matrixScale,
             matrixMax, geom])

  const moduleAdvice: Advice[] = useMemo(() => (prep
    ? adviseModules(
        prep.modules.map((m) => ({
          name: m.name, highCount: highCount(m, cutoffs[m.name] ?? m.auto_cutoff),
        })),
        geom?.nSpots ?? 0, profileK)
    : []), [prep, cutoffs, profileK, geom])

  return (
    <div style={styles.overlay} onClick={close}>
      <div style={styles.modal} onClick={(e) => e.stopPropagation()}>
        <div style={styles.header}>
          <span style={styles.title}>Contour</span>
          <button style={styles.close} onClick={close}>×</button>
        </div>

        {error && <div style={styles.error}>{error}</div>}

        {phase === 'select' && (
          <div>
            <div style={styles.note}>
              Pick gene sets to contour. <strong>One</strong> set → a banded expression
              column. <strong>Two or more</strong> → a fused tissue annotation (needs
              <code> X_pca</code>; run PCA first). Grid resolution and smoothing are prefilled
              from data-aware suggestions.
            </div>

            <button style={styles.secondary} onClick={() => setPickerOpen(true)}>
              Choose gene sets…
            </button>
            <div style={styles.chips}>
              {nSources === 0 && <span style={styles.dim}>No gene sets chosen yet.</span>}
              {sources.map((s) => (
                <span key={s.key} style={styles.chip}>
                  {s.name} <span style={styles.dim}>({s.genes.length})</span>
                  <button style={styles.chipX} onClick={() => togglePick(s)}>×</button>
                </span>
              ))}
            </div>

            <div style={styles.paramGrid}>
              <Param label="Contour levels" tip="How finely the score range is divided. Thresholds are spaced evenly in value, not in spot count, so the upper bands stay small on a skewed field. More levels give finer control over the cutoff; they do not change the field.">
                <input type="number" min={2} max={8} value={contourLevels}
                  onChange={(e) => setContourLevels(parseInt(e.target.value || '3', 10))} style={styles.input} />
              </Param>
              <Param label="Grid resolution" tip="Interpolation grid per axis. Sigma is measured in these pixels, so changing this rescales the smoothing by the same factor. Prefilled to roughly match the spot spacing.">
                <input type="number" placeholder="auto" value={gridRes}
                  onChange={(e) => setGridRes(e.target.value)} style={styles.input} />
              </Param>
              <Param label="Smoothing sigma" tip="Gaussian width in grid pixels. What matters is the radius it works out to in spot spacings — under 1 it cannot reach the next spot and the bands speckle; over about 6 it spans a zone and tissues merge.">
                <input type="number" step="0.5" placeholder="auto" value={smoothSigma}
                  onChange={(e) => setSmoothSigma(e.target.value)} style={styles.input} />
              </Param>
              <Param label="Log transform" tip="log1p before per-gene clipping. Each gene is already clipped to its 1st–99th percentile, so this is for a bulk multiplicative scale — raw counts — not for one hot spot. On an already-logged matrix it flattens the field.">
                <input type="checkbox" checked={logTransform} onChange={(e) => setLogTransform(e.target.checked)} />
              </Param>
              <Param label="Section column" tip="Contours each section on its own grid. Without it the interpolation spans the gap between sections and expression bleeds across. Use Define Sections to create one.">
                <select value={sectionCol} onChange={(e) => setSectionCol(e.target.value)} style={styles.input}>
                  <option value="">— treat as one tissue —</option>
                  {obsColumns.map((c) => <option key={c} value={c}>{c}</option>)}
                </select>
              </Param>
              {nSources === 1 && (
                <Param label="Column name" tip="Name for the resulting banded .obs column (blank = auto).">
                  <input type="text" placeholder="auto" value={columnName}
                    onChange={(e) => setColumnName(e.target.value)} style={styles.input} />
                </Param>
              )}
            </div>

            <AdviceList items={advice} />
            <button style={styles.ghost} onClick={() => setShowGuide((v) => !v)}>
              {showGuide ? 'Hide' : 'Show'} how to choose these
            </button>
            {showGuide && <ContourGuide />}

            <div style={{ ...styles.actions, marginTop: 14 }}>
              {nSources <= 1 ? (
                <button style={styles.primary} disabled={busy || nSources !== 1} onClick={runSingle}>
                  {busy ? 'Running…' : 'Run contour'}
                </button>
              ) : (
                <button style={styles.primary} disabled={busy} onClick={runPrepare}>
                  {busy ? 'Computing…' : `Compute (${nSources} sets)`}
                </button>
              )}
            </div>
          </div>
        )}

        {phase === 'review' && prep && (
          <div>
            <div style={styles.note}>
              For each module choose the cutoff at/above which a spot is "high" (auto = top band).
              Spots high in exactly one module get that tissue; multi-high spots are resolved by
              spatial + PCA-profile neighbors; the rest stay <code>unassigned</code>.
            </div>
            {prep.missing_genes && Object.keys(prep.missing_genes).length > 0 && (
              <div style={styles.warn}>
                Some genes weren't found in this dataset and were skipped:
                {Object.entries(prep.missing_genes).map(([name, genes]) => (
                  <div key={name}><strong>{name}</strong>: {genes.join(', ')}</div>
                ))}
              </div>
            )}
            {prep.modules.map((m) => (
              <div key={m.name} style={styles.moduleRow}>
                <div style={styles.moduleHead}>
                  <strong>{m.name}</strong>
                  <span style={styles.dim}> — {highCount(m, cutoffs[m.name] ?? m.auto_cutoff)} spots high</span>
                </div>
                <BandBars m={m} cutoff={cutoffs[m.name] ?? m.auto_cutoff} />
                <div style={styles.sliderRow}>
                  <span style={styles.dim}>cutoff:</span>
                  <select
                    value={String(cutoffs[m.name] ?? m.auto_cutoff)}
                    onChange={(e) => setCutoffs((c) => ({ ...c, [m.name]: parseFloat(e.target.value) }))}
                    style={styles.input}
                  >
                    {m.thresholds.map((t, i) => (
                      <option key={i} value={String(t)}>band ≥ {t.toFixed(3)}</option>
                    ))}
                  </select>
                </div>
              </div>
            ))}

            <div style={styles.paramGrid}>
              <Param label="Profile k" tip="Nearest unambiguous spatial neighbours, ranked in PCA space, that vote on a spot high in more than one module. Large k reaches past the local neighbourhood and votes for whatever is regionally dominant.">
                <input type="number" min={1} value={profileK}
                  onChange={(e) => setProfileK(parseInt(e.target.value || '15', 10))} style={styles.input} />
              </Param>
              <Param label="Column name" tip="Name of the resulting tissue .obs column.">
                <input type="text" value={outName} onChange={(e) => setOutName(e.target.value)} style={styles.input} />
              </Param>
              <Param label="Save QC columns" tip="Also write <name>_status (single/resolved/unassigned) and per-module <set>_high.">
                <input type="checkbox" checked={saveQc} onChange={(e) => setSaveQc(e.target.checked)} />
              </Param>
            </div>

            <AdviceList items={moduleAdvice} />
            <button style={styles.ghost} onClick={() => setShowGuide((v) => !v)}>
              {showGuide ? 'Hide' : 'Show'} how to choose these
            </button>
            {showGuide && <ContourGuide />}

            <div style={{ ...styles.actions, marginTop: 14 }}>
              <button style={styles.secondary} disabled={busy} onClick={() => setPhase('select')}>Back</button>
              <button style={styles.primary} disabled={busy || !outName.trim()} onClick={runFinalize}>
                {busy ? 'Finalizing…' : 'Finalize'}
              </button>
            </div>
          </div>
        )}

        {phase === 'done' && doneResult && (
          <div>
            <div style={styles.note}>
              Created <strong>{doneResult.annotation_key}</strong>
              {doneResult.categories
                ? ` (${doneResult.n_resolved ?? 0} conflicts resolved).`
                : ` — banded column, ${doneResult.contour_levels} levels from ${doneResult.n_genes} gene${doneResult.n_genes === 1 ? '' : 's'}.`}
            </div>
            {doneResult.missing_genes && doneResult.missing_genes.length > 0 && (
              <div style={styles.warn}>
                Skipped genes not found in this dataset: {doneResult.missing_genes.join(', ')}
              </div>
            )}
            {doneResult.categories && (
              <div style={styles.setList}>
                {doneResult.categories.map((c) => (
                  <div key={c} style={styles.chipRow}>
                    <span>{c}</span><span style={styles.dim}> — {doneResult.counts?.[c] ?? 0} spots</span>
                  </div>
                ))}
              </div>
            )}
            <div style={styles.actions}>
              <button style={styles.secondary} onClick={() => { setPhase('select'); setPrep(null); setDoneResult(null) }}>Run again</button>
              <button style={styles.primary} onClick={() => { setSelectedColorColumn(doneResult.annotation_key); close() }}>
                Color by {doneResult.annotation_key}
              </button>
            </div>
          </div>
        )}

        {/* Gene-set picker popup */}
        {pickerOpen && (
          <div style={styles.pickerOverlay} onClick={() => setPickerOpen(false)}>
            <div style={styles.picker} onClick={(e) => e.stopPropagation()}>
              <div style={styles.header}>
                <span style={styles.title}>Choose gene sets</span>
                <button style={styles.close} onClick={() => setPickerOpen(false)}>×</button>
              </div>
              <div style={styles.setList}>
                <label style={{ ...styles.chipRow, opacity: selectedGenes.length ? 1 : 0.5 }}>
                  <input type="checkbox" disabled={!selectedGenes.length}
                    checked={isPicked(SELECTION_KEY)}
                    onChange={() => togglePick({ key: SELECTION_KEY, name: 'Current selection', genes: selectedGenes })} />
                  <span style={{ marginLeft: 8 }}>Current Gene Panel selection</span>
                  <span style={styles.dim}> ({selectedGenes.length} genes)</span>
                </label>
                {namedSets.length === 0 && <div style={styles.dim}>No gene sets available. Create some in the Gene Panel first.</div>}
                {namedSets.map((g) => (
                  <label key={g.key} style={styles.chipRow}>
                    <input type="checkbox" checked={isPicked(g.key)} onChange={() => togglePick(g)} />
                    <span style={{ marginLeft: 8 }}>{g.name}</span>
                    <span style={styles.dim}> ({g.genes.length} genes)</span>
                  </label>
                ))}
              </div>
              <div style={styles.actions}>
                <button style={styles.primary} onClick={() => setPickerOpen(false)}>Done ({nSources})</button>
              </div>
            </div>
          </div>
        )}
      </div>
    </div>
  )
}

function Param({ label, tip, children }: { label: string; tip: string; children: React.ReactNode }) {
  return (
    <label style={styles.param} title={tip}>
      <span style={styles.paramLabel}>{label}</span>
      {children}
    </label>
  )
}

function BandBars({ m, cutoff }: { m: ModuleReview; cutoff: number }) {
  const max = Math.max(1, ...m.histogram)
  return (
    <div style={styles.bars}>
      {m.band_values.map((bv, i) => (
        <div key={i} style={styles.barCol} title={`band ≥ ${bv.toFixed(3)}: ${m.histogram[i]} spots`}>
          <div style={{
            ...styles.bar,
            height: `${(m.histogram[i] / max) * 40 + 2}px`,
            backgroundColor: bv >= cutoff ? '#4ecdc4' : '#3a3a52',
          }} />
        </div>
      ))}
    </div>
  )
}

/**
 * What to pick and what each choice costs — not a restatement of the labels.
 *
 * Leads with the grid×σ coupling because it is the one thing the form actively
 * hides: σ is measured in grid pixels, so raising the grid shrinks the real
 * smoothing by the same factor.
 */
function ContourGuide() {
  return (
    <div style={styles.guide}>
      <div style={styles.guideHead}>Starting points</div>
      <table style={styles.guideTable}>
        <thead>
          <tr style={{ color: '#888', textAlign: 'left' }}>
            <th style={styles.th}>Data</th><th style={styles.th}>Use</th>
          </tr>
        </thead>
        <tbody>
          <tr>
            <td style={styles.td}>Visium <span style={styles.dim}>(55 µm spots, 2–5k)</span></td>
            <td style={styles.td}>The prefilled grid and σ. 3 levels, cutoff at the top band.</td>
          </tr>
          <tr>
            <td style={styles.td}>Visium HD, or binned to a fine grid</td>
            <td style={styles.td}>Prefilled grid; raise σ with it so the smoothing still reaches ~2 spot spacings.</td>
          </tr>
          <tr>
            <td style={styles.td}>Xenium / single-cell resolution</td>
            <td style={styles.td}>Spacing is irregular — trust the reported radius above, not σ itself.</td>
          </tr>
          <tr>
            <td style={styles.td}>Several sections on one slide</td>
            <td style={styles.td}>Set the section column. Without it the grid spans the gap and expression bleeds across.</td>
          </tr>
        </tbody>
      </table>

      <div style={styles.guideHead}>Trade-offs</div>
      <ul style={styles.guideList}>
        <li>
          <b>Grid and σ are one setting, not two.</b> σ is measured in grid
          pixels, and a pixel is <code>extent / grid</code>. Doubling the grid
          halves the real smoothing radius. So if a map looks blurry, raising
          the grid appears to sharpen it — but it sharpened by removing
          smoothing, which is not the same as resolving structure.
        </li>
        <li>
          <b>Bands are equal-width, not equal-count.</b> Thresholds are spaced
          evenly between zero and the field's maximum. A module high in one
          corner gives a strongly skewed field, so the top band holds few spots
          — that is the shape of the field, not weak expression.
        </li>
        <li>
          <b>Levels only subdivide the cutoff.</b> More levels give finer
          control over where "high" starts. They do not change the interpolated
          field and cannot reveal structure the grid and σ did not preserve.
        </li>
        <li>
          <b>Log transform and the percentile clip overlap.</b> Each gene is
          already clipped to its 1st–99th percentile before averaging, so the
          extreme tail is handled. Log is for when the <i>bulk</i> of the
          distribution is multiplicative — raw counts — not for one hot spot.
        </li>
        <li>
          <b>The cutoff is where the tissue call happens.</b> Everything before
          it is a continuous field; the cutoff turns it into a claim. The spot
          count beside each choice is the honest signal, not the picture.
        </li>
      </ul>

      <div style={styles.guideHead}>Avoid</div>
      <ul style={styles.guideList}>
        <li>Raising the grid to sharpen a blurry map — that removes smoothing rather than adding resolution.</li>
        <li>Contouring several sections without a section column.</li>
        <li>Comparing band thresholds between runs. <code>vmax</code> is per-run, so the same number means different things.</li>
        <li>Choosing cutoffs by which map looks cleanest. A clean map is what this produces when a module is high nearly everywhere.</li>
      </ul>
    </div>
  )
}

function AdviceList({ items }: { items: Advice[] }) {
  if (items.length === 0) return null
  return (
    <div style={styles.advice}>
      {items.map((a, i) => (
        <div key={i} style={{
          fontSize: 11, lineHeight: 1.45, padding: '4px 8px', borderRadius: 2,
          color: a.level === 'warn' ? '#f0c987' : '#aaa',
          background: a.level === 'warn' ? 'rgba(233,162,59,0.12)' : 'transparent',
          borderLeft: `2px solid ${a.level === 'warn' ? '#e9a23b' : '#0f3460'}`,
        }}>
          {a.level === 'warn' ? '⚠ ' : ''}{a.text}
        </div>
      ))}
    </div>
  )
}

const styles: Record<string, React.CSSProperties> = {
  guide: { marginTop: 8, padding: 10, background: '#0f1625', borderRadius: 4, fontSize: 11, lineHeight: 1.5, color: '#bbb' },
  guideHead: { color: '#4ecdc4', fontSize: 11, fontWeight: 600, textTransform: 'uppercase', letterSpacing: 0.4, margin: '10px 0 6px' },
  guideTable: { borderCollapse: 'collapse', width: '100%', marginBottom: 4 },
  th: { padding: '2px 6px 4px 0', fontWeight: 500, borderBottom: '1px solid #0f3460' },
  td: { padding: '4px 6px 4px 0', verticalAlign: 'top', borderBottom: '1px solid rgba(15,52,96,0.5)' },
  guideList: { margin: 0, paddingLeft: 16, display: 'grid', gap: 5 },
  ghost: { background: 'none', border: '1px solid #0f3460', color: '#aaa', borderRadius: 4, padding: '4px 10px', fontSize: 11, cursor: 'pointer' },
  advice: { display: 'grid', gap: 5, marginBottom: 12 },
  overlay: { position: 'fixed', inset: 0, backgroundColor: 'rgba(0,0,0,0.6)', zIndex: 1000, display: 'flex', alignItems: 'center', justifyContent: 'center' },
  modal: { backgroundColor: '#16213e', color: '#eee', borderRadius: 8, padding: 20, width: 560, maxWidth: '90vw', maxHeight: '85vh', overflowY: 'auto', boxShadow: '0 8px 32px rgba(0,0,0,0.5)', position: 'relative' },
  header: { display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: 12 },
  title: { fontSize: 16, fontWeight: 600 },
  close: { background: 'none', border: 'none', color: '#aaa', fontSize: 22, cursor: 'pointer', lineHeight: 1 },
  error: { backgroundColor: 'rgba(233,69,96,0.15)', color: '#ff7a90', padding: '8px 10px', borderRadius: 4, marginBottom: 12, fontSize: 13 },
  note: { fontSize: 12.5, color: '#bbb', backgroundColor: 'rgba(78,205,196,0.08)', padding: 10, borderRadius: 4, borderLeft: '3px solid #4ecdc4', marginBottom: 14, lineHeight: 1.5 },
  warn: { fontSize: 12, color: '#ffcf6a', backgroundColor: 'rgba(255,207,106,0.10)', padding: 10, borderRadius: 4, borderLeft: '3px solid #ffcf6a', marginBottom: 14, lineHeight: 1.5 },
  chips: { display: 'flex', flexWrap: 'wrap', gap: 6, margin: '10px 0 14px' },
  chip: { display: 'inline-flex', alignItems: 'center', gap: 4, backgroundColor: '#0f3460', borderRadius: 12, padding: '3px 8px', fontSize: 12 },
  chipX: { background: 'none', border: 'none', color: '#aaa', cursor: 'pointer', fontSize: 14, lineHeight: 1, padding: 0 },
  setList: { maxHeight: 240, overflowY: 'auto', border: '1px solid #1a1a2e', borderRadius: 4, padding: 8, marginBottom: 14 },
  chipRow: { display: 'flex', alignItems: 'center', padding: '4px 2px', fontSize: 13, cursor: 'pointer' },
  dim: { color: '#888', fontSize: 12 },
  paramGrid: { display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 10, marginBottom: 14 },
  param: { display: 'flex', flexDirection: 'column', gap: 4, fontSize: 12 },
  paramLabel: { color: '#bbb' },
  input: { padding: '5px 7px', fontSize: 13, backgroundColor: '#0f3460', color: '#eee', border: '1px solid #1a1a2e', borderRadius: 4 },
  actions: { display: 'flex', justifyContent: 'flex-end', gap: 10, marginTop: 6 },
  primary: { padding: '8px 16px', fontSize: 13, backgroundColor: '#4ecdc4', color: '#0a0a1a', border: 'none', borderRadius: 4, cursor: 'pointer', fontWeight: 600 },
  secondary: { padding: '8px 16px', fontSize: 13, backgroundColor: '#0f3460', color: '#eee', border: '1px solid #1a1a2e', borderRadius: 4, cursor: 'pointer' },
  moduleRow: { border: '1px solid #1a1a2e', borderRadius: 4, padding: 10, marginBottom: 10 },
  moduleHead: { fontSize: 13, marginBottom: 6 },
  bars: { display: 'flex', alignItems: 'flex-end', gap: 4, height: 46, marginBottom: 6 },
  barCol: { flex: 1, display: 'flex', alignItems: 'flex-end', justifyContent: 'center' },
  bar: { width: '100%', borderRadius: '2px 2px 0 0' },
  sliderRow: { display: 'flex', alignItems: 'center', gap: 8 },
  pickerOverlay: { position: 'fixed', inset: 0, backgroundColor: 'rgba(0,0,0,0.5)', zIndex: 1100, display: 'flex', alignItems: 'center', justifyContent: 'center' },
  picker: { backgroundColor: '#16213e', color: '#eee', borderRadius: 8, padding: 20, width: 460, maxWidth: '90vw', maxHeight: '80vh', overflowY: 'auto', boxShadow: '0 8px 32px rgba(0,0,0,0.6)' },
}
