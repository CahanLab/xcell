import { useCallback, useEffect, useMemo, useState } from 'react'
import { appendDataset, pollTask, refreshSchema, useObsSummaries } from '../hooks/useData'
import { useStore } from '../store'
import { LayerScaleBadge, layerOptionLabel, type LayerInfo } from './LayerScaleInfo'
import { datasetIdentity } from '../lib/datasetIdentity'

/**
 * PySingleCellNet cell-type classification.
 *
 * Two jobs, one modal. **Classify** runs a trained classifier over the loaded
 * dataset; **Train** builds one from a labelled dataset and writes it to disk.
 *
 * The design pressure here is that PySCN fails quietly rather than loudly: a
 * query sharing few genes with the training data still returns confident
 * scores, because missing genes are filled with zeros. So the classifier is
 * inspected against this dataset *before* the run, and the resulting gene
 * coverage is the most prominent thing on screen.
 */

const dark = {
  panel: '#16213e', border: '#0f3460', field: '#0f3460', text: '#eee',
  sub: '#aaa', faint: '#888', accent: '#4ecdc4', inset: '#0f1625',
  warn: '#e9a23b', error: '#e94560',
}

const SEVERITY_COLOR: Record<string, string> = {
  ok: dark.accent, warn: dark.warn, error: dark.error,
}

interface ClassifierMeta {
  classes: string[]
  cell_type_classes: string[]
  n_classes: number
  n_genes: number
  n_gene_pairs: number
  n_trees: number
  train_params: Record<string, unknown>
  path?: string
  file_size_mb?: number
}

interface GeneOverlap {
  n_required: number
  n_found: number
  frac_found: number
  missing: string[]
  n_missing: number
  n_found_case_insensitive: number
  case_mismatch_only: boolean
  severity: 'ok' | 'warn' | 'error'
}

interface Inspection {
  classifier: ClassifierMeta
  gene_overlap: GeneOverlap
  colors: Record<string, string>
}

interface CompositionEntry {
  name: string
  n_cells: number
  frac: number
  mean_score: number
  color: string
  threshold?: number | null
}

interface ClassifyResult {
  key: string
  n_cells: number
  classes: string[]
  composition: CompositionEntry[]
  class_types: { name: string; n_cells: number; frac: number; color: string }[]
  gene_overlap: GeneOverlap
  obs_columns: string[]
  argmax_column: string
  score_column: string
  type_column: string | null
  obsm_key: string
}

interface Preprocessing {
  source_scale: string
  detected_scale: string
  normalize: boolean
  log1p: boolean
  hvg_flavor: string
  snapshot_counts: boolean
  overridden: boolean
  uncertain: boolean
  reason: string
}

interface TrainResult {
  path: string
  preprocessing: Preprocessing
  classifier: ClassifierMeta
  n_cells_used: number
  groupby: string
  dropped_labels: string[]
  n_genes_dropped_underscore: number
  training_counts: { name: string; n_cells: number }[]
}

interface Status {
  available: boolean
  version: string | null
  error: string | null
  install_hint: string
  layers: LayerInfo[]
  categorical_obs: string[]
  n_cells: number
  n_genes: number
}

export default function PyscnModal() {
  const isOpen = useStore((s) => s.isPyscnModalOpen)
  const setOpen = useStore((s) => s.setPyscnModalOpen)
  const selectColorColumn = useStore((s) => s.setSelectedColorColumn)
  const setColorMode = useStore((s) => s.setColorMode)
  const activeSlot = useStore((s) => s.activeSlot)
  const schema = useStore((s) => s.schema)
  // Everything below describes one specific dataset. Loading a different file —
  // including into the slot already selected — invalidates all of it.
  const dataset = datasetIdentity(activeSlot, schema)
  const { refresh: refreshObs } = useObsSummaries()

  const [tab, setTab] = useState<'classify' | 'train'>('classify')
  const [status, setStatus] = useState<Status | null>(null)
  const [busy, setBusy] = useState(false)
  const [progress, setProgress] = useState<{ frac: number; message: string } | null>(null)
  const [error, setError] = useState<string | null>(null)

  // --- classify state ---
  const [path, setPath] = useState('')
  const [inspection, setInspection] = useState<Inspection | null>(null)
  const [inspecting, setInspecting] = useState(false)
  const [layer, setLayer] = useState('X')
  const [resultKey, setResultKey] = useState('SCN')
  const [caseInsensitive, setCaseInsensitive] = useState(false)
  const [categorize, setCategorize] = useState(true)
  const [quantile, setQuantile] = useState(0.05)
  const [result, setResult] = useState<ClassifyResult | null>(null)

  // --- train state ---
  const [groupby, setGroupby] = useState('')
  const [outPath, setOutPath] = useState('')
  const [nCellsPerType, setNCellsPerType] = useState(100)
  const [nTopGenes, setNTopGenes] = useState(30)
  const [nTopGenePairs, setNTopGenePairs] = useState(40)
  const [nTrees, setNTrees] = useState(1000)
  const [nComps, setNComps] = useState(30)
  const [sourceScale, setSourceScale] = useState('auto')
  const [trainResult, setTrainResult] = useState<TrainResult | null>(null)

  // Drop anything computed against a dataset that is no longer loaded. Results
  // and the gene-coverage inspection are the dangerous ones: left on screen
  // they describe the wrong data while looking authoritative.
  useEffect(() => {
    setResult(null)
    setTrainResult(null)
    setInspection(null)
    setError(null)
  }, [dataset])

  useEffect(() => {
    if (!isOpen) return
    setError(null)
    fetch(appendDataset('/api/pyscn/status'))
      .then((r) => r.json())
      .then((s: Status) => {
        setStatus(s)
        // Keep the current choices only if they still exist here.
        const names = (s.layers ?? []).map((l) => l.name)
        setLayer((cur) => (names.includes(cur) ? cur : 'X'))
        const cats = s.categorical_obs ?? []
        setGroupby((cur) => (cats.includes(cur) ? cur : (cats[0] ?? '')))
      })
      .catch((e) => setError(String(e)))
  }, [isOpen, dataset])

  // Re-check the classifier against the new data rather than making the user
  // remember to. The path is still valid; only what it was measured against
  // changed.
  useEffect(() => {
    if (!isOpen || !path.trim() || inspection) return
    inspect()
    // `inspect` is stable per path; re-running on `inspection` would loop.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [isOpen, dataset])

  useEffect(() => {
    if (!isOpen) return
    const onKey = (e: KeyboardEvent) => { if (e.key === 'Escape' && !busy) setOpen(false) }
    window.addEventListener('keydown', onKey)
    return () => window.removeEventListener('keydown', onKey)
  }, [isOpen, busy, setOpen])

  const inspect = useCallback(async () => {
    if (!path.trim()) return
    setInspecting(true); setError(null); setInspection(null)
    try {
      const resp = await fetch(appendDataset('/api/pyscn/inspect_classifier'), {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ path: path.trim() }),
      })
      const data = await resp.json()
      if (!resp.ok) throw new Error(data.detail || 'Could not read that classifier')
      setInspection(data)
      // A pure case mismatch has one obvious fix — pre-arm it.
      if (data.gene_overlap?.case_mismatch_only) setCaseInsensitive(true)
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e))
    } finally {
      setInspecting(false)
    }
  }, [path])

  const runTask = useCallback(async (url: string, body: unknown) => {
    setBusy(true); setError(null); setProgress({ frac: 0, message: 'Starting…' })
    try {
      const resp = await fetch(appendDataset(url), {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(body),
      })
      const data = await resp.json()
      if (!resp.ok) throw new Error(data.detail || 'Request failed')
      const task = await pollTask(data.task_id, undefined, (s) => {
        setProgress((prev) => ({
          frac: typeof s.progress === 'number' ? s.progress : (prev?.frac ?? 0),
          message: s.message ?? prev?.message ?? 'Working…',
        }))
      })
      if (task.status !== 'completed') throw new Error(task.error || `Task ${task.status}`)
      return task.result
    } finally {
      setBusy(false); setProgress(null)
    }
  }, [])

  const runClassify = useCallback(async () => {
    try {
      const res = await runTask('/api/pyscn/classify', {
        path: path.trim(),
        key: resultKey.trim() || 'SCN',
        layer: layer === 'X' ? null : layer,
        case_insensitive: caseInsensitive,
        categorize,
        quantile,
      }) as unknown as ClassifyResult
      setResult(res)
      // New obs columns and a new obsm score matrix — both dropdown sources.
      await refreshSchema()
      refreshObs()
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e))
    }
  }, [runTask, path, resultKey, layer, caseInsensitive, categorize, quantile, refreshObs])

  const runTrain = useCallback(async () => {
    try {
      const res = await runTask('/api/pyscn/train', {
        groupby,
        out_path: outPath.trim(),
        n_cells_per_type: nCellsPerType > 0 ? nCellsPerType : null,
        n_top_genes: nTopGenes,
        n_top_gene_pairs: nTopGenePairs,
        n_trees: nTrees,
        n_comps: nComps,
        layer: layer === 'X' ? null : layer,
        source_scale: sourceScale === 'auto' ? null : sourceScale,
      }) as unknown as TrainResult
      setTrainResult(res)
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e))
    }
  }, [runTask, groupby, outPath, nCellsPerType, nTopGenes, nTopGenePairs, nTrees, nComps, layer, sourceScale])

  const colorBy = useCallback((column: string) => {
    selectColorColumn(column)
    setColorMode('metadata')
    setOpen(false)
  }, [selectColorColumn, setColorMode, setOpen])

  const selectedLayer = useMemo(
    () => status?.layers?.find((l) => l.name === layer),
    [status, layer],
  )
  // The scale verdict already rides along on /api/pyscn/status's layer list,
  // so the Train tab can name what auto-detect will do without another call.
  const detectedScale = selectedLayer?.scale

  if (!isOpen) return null

  const overlap = inspection?.gene_overlap
  const canClassify = !!inspection && !busy && (status?.available ?? false)

  return (
    <div style={overlayStyle} onClick={() => { if (!busy) setOpen(false) }}>
      <div style={cardStyle} onClick={(e) => e.stopPropagation()}>
        <div style={{ display: 'flex', alignItems: 'baseline', justifyContent: 'space-between' }}>
          <h3 style={{ margin: 0, color: dark.text, fontSize: 16 }}>
            Cell typing — PySingleCellNet
          </h3>
          <span style={{ fontSize: 11, color: dark.faint }}>
            {status?.available ? `v${status.version}` : 'not installed'}
          </span>
        </div>

        {/* Which dataset this acts on. With two slots loaded and results that
            persist on screen, "what am I about to classify?" must never be a
            guess. */}
        <div style={{ fontSize: 11, color: dark.sub, marginTop: 4 }}>
          <span style={{ color: dark.faint }}>Dataset: </span>
          <code style={{ color: dark.text }}>{schema?.filename ?? '—'}</code>
          <span style={{ color: dark.faint }}>
            {' '}({activeSlot}
            {schema ? `, ${schema.n_cells.toLocaleString()} cells` : ''})
          </span>
        </div>

        <div style={{ display: 'flex', gap: 4, margin: '12px 0' }}>
          {(['classify', 'train'] as const).map((t) => (
            <button
              key={t}
              onClick={() => setTab(t)}
              disabled={busy}
              style={{
                ...tabButton,
                background: tab === t ? dark.field : 'transparent',
                color: tab === t ? dark.text : dark.sub,
                borderBottom: tab === t ? `2px solid ${dark.accent}` : '2px solid transparent',
              }}
            >
              {t === 'classify' ? 'Classify' : 'Train'}
            </button>
          ))}
        </div>

        {status && !status.available && (
          <div style={{ ...noticeStyle, background: 'rgba(233,162,59,0.14)', color: '#f0c987' }}>
            {status.install_hint}
            <div style={{ marginTop: 6, color: dark.faint, fontSize: 10 }}>
              You can still inspect a classifier against this dataset below —
              that only needs scikit-learn.
            </div>
          </div>
        )}

        {error && (
          <div style={{ ...noticeStyle, background: 'rgba(233,69,96,0.15)', color: '#ffb3bd' }}>
            {error}
          </div>
        )}

        {/* ---------------------------------------------------------- */}
        {tab === 'classify' && !result && (
          <>
            <label style={labelStyle}>Classifier file (.pkl)</label>
            <div style={{ display: 'flex', gap: 6 }}>
              <input
                value={path}
                onChange={(e) => { setPath(e.target.value); setInspection(null) }}
                onKeyDown={(e) => { if (e.key === 'Enter') inspect() }}
                placeholder="/path/to/classifier.pkl"
                style={{ ...inputStyle, flex: 1 }}
              />
              <button onClick={inspect} disabled={!path.trim() || inspecting} style={ghostButton}>
                {inspecting ? 'Reading…' : 'Inspect'}
              </button>
            </div>

            {inspection && (
              <>
                <div style={sectionStyle}>
                  <div style={sectionTitle}>Classifier</div>
                  <div style={{ display: 'flex', flexWrap: 'wrap', gap: 4, marginBottom: 6 }}>
                    {inspection.classifier.cell_type_classes.map((c) => (
                      <span key={c} style={{ ...pill, borderColor: inspection.colors[c] ?? dark.border }}>
                        <span style={{ ...swatch, background: inspection.colors[c] ?? '#888' }} />
                        {c}
                      </span>
                    ))}
                  </div>
                  <div style={{ color: dark.faint, fontSize: 11 }}>
                    {inspection.classifier.n_classes} classes ·{' '}
                    {inspection.classifier.n_gene_pairs} gene pairs over{' '}
                    {inspection.classifier.n_genes} genes ·{' '}
                    {inspection.classifier.n_trees} trees
                  </div>
                </div>

                {overlap && (
                  <div style={{
                    ...sectionStyle,
                    borderLeft: `3px solid ${SEVERITY_COLOR[overlap.severity]}`,
                  }}>
                    <div style={sectionTitle}>Gene coverage</div>
                    <div style={{ color: SEVERITY_COLOR[overlap.severity], fontWeight: 600 }}>
                      {overlap.n_found} / {overlap.n_required} classifier genes present
                      {' '}({(overlap.frac_found * 100).toFixed(0)}%)
                    </div>
                    <div style={{ color: dark.sub, fontSize: 11, marginTop: 4, lineHeight: 1.45 }}>
                      PySingleCellNet fills genes it cannot find with zeros, so
                      missing genes quietly degrade every prediction rather than
                      raising an error.
                      {overlap.severity === 'error' &&
                        ' At this coverage the scores are not meaningful.'}
                    </div>
                    {overlap.case_mismatch_only && (
                      <label style={{ ...checkRow, marginTop: 8, color: dark.warn }}>
                        <input
                          type="checkbox"
                          checked={caseInsensitive}
                          onChange={(e) => setCaseInsensitive(e.target.checked)}
                        />
                        Match gene symbols ignoring case — recovers{' '}
                        {overlap.n_found_case_insensitive} / {overlap.n_required}
                        {' '}(this dataset and the classifier use different symbol casing)
                      </label>
                    )}
                    {overlap.n_missing > 0 && !overlap.case_mismatch_only && (
                      <div style={{ color: dark.faint, fontSize: 10, marginTop: 6 }}>
                        Missing: {overlap.missing.slice(0, 12).join(', ')}
                        {overlap.n_missing > 12 ? ` … +${overlap.n_missing - 12} more` : ''}
                      </div>
                    )}
                  </div>
                )}

                <div style={{ display: 'flex', gap: 10, marginTop: 10 }}>
                  <div style={{ flex: 1 }}>
                    <div style={{ display: 'flex', alignItems: 'center', gap: 6 }}>
                      <span style={labelStyle}>Expression source</span>
                      <LayerScaleBadge layer={selectedLayer} />
                    </div>
                    <select value={layer} onChange={(e) => setLayer(e.target.value)} style={inputStyle}>
                      {(status?.layers ?? []).map((l) => (
                        <option key={l.name} value={l.name}>{layerOptionLabel(l)}</option>
                      ))}
                    </select>
                  </div>
                  <div style={{ width: 130 }}>
                    <span style={labelStyle}>Result prefix</span>
                    <input
                      value={resultKey}
                      onChange={(e) => setResultKey(e.target.value)}
                      style={inputStyle}
                    />
                  </div>
                </div>

                {selectedLayer?.scale?.verdict === 'z_scored' && (
                  <div style={{ ...noticeStyle, background: 'rgba(233,69,96,0.15)', color: '#ffb3bd' }}>
                    This layer looks scaled / z-scored. The top-scoring-pair
                    transform asks "is gene A above gene B in this cell", which
                    per-gene centering reorders — the result would be
                    meaningless. Pick a counts or log-normalized source.
                  </div>
                )}

                <div style={{ color: dark.faint, fontSize: 10, marginTop: 6, lineHeight: 1.5 }}>
                  Writes <code>{resultKey || 'SCN'}_class_argmax</code> (the call),{' '}
                  <code>{resultKey || 'SCN'}_class_score</code> (its confidence)
                  {categorize && <> and <code>{resultKey || 'SCN'}_class_type</code></>}
                  {' '}to <code>.obs</code>, plus a per-class score matrix at{' '}
                  <code>.obsm[&apos;{resultKey || 'SCN'}_score&apos;]</code>.
                </div>

                <label style={checkRow}>
                  <input type="checkbox" checked={categorize}
                    onChange={(e) => setCategorize(e.target.checked)} />
                  Also call each cell Singular / Ambiguous / None / Rand
                </label>
                {categorize && (
                  <div style={{ marginLeft: 22, marginTop: 4 }}>
                    <span style={{ ...labelStyle, display: 'inline' }}>threshold quantile </span>
                    <input
                      type="number" step={0.01} min={0} max={0.5} value={quantile}
                      onChange={(e) => setQuantile(Math.min(0.5, Math.max(0, parseFloat(e.target.value) || 0)))}
                      style={{ ...inputStyle, width: 70, display: 'inline-block' }}
                    />
                    <div style={{ color: dark.faint, fontSize: 10, marginTop: 3, lineHeight: 1.45 }}>
                      Per-class threshold, as in PySCN&apos;s <code>comp_ct_thresh</code>.
                      Cells over exactly one threshold are Singular; over several,
                      Ambiguous — PySCN splits those into Intermediate and Hybrid
                      using a cell-type graph xcell does not have.
                    </div>
                  </div>
                )}
              </>
            )}
          </>
        )}

        {tab === 'classify' && result && (
          <ClassifyResultView result={result} onColorBy={colorBy} onAgain={() => setResult(null)} />
        )}

        {/* ---------------------------------------------------------- */}
        {tab === 'train' && !trainResult && (
          <>
            <label style={labelStyle}>Cell type labels (.obs column)</label>
            <select value={groupby} onChange={(e) => setGroupby(e.target.value)} style={inputStyle}>
              {(status?.categorical_obs ?? []).map((c) => <option key={c} value={c}>{c}</option>)}
            </select>
            {!status?.categorical_obs?.length && (
              <div style={{ color: dark.warn, fontSize: 11, marginTop: 4 }}>
                This dataset has no categorical .obs column to train on.
              </div>
            )}

            <div style={{ display: 'flex', alignItems: 'center', gap: 6, marginTop: 10 }}>
              <span style={labelStyle}>Counts source</span>
              <LayerScaleBadge layer={selectedLayer} />
            </div>
            <select value={layer} onChange={(e) => setLayer(e.target.value)} style={inputStyle}>
              {(status?.layers ?? []).map((l) => (
                <option key={l.name} value={l.name}>{layerOptionLabel(l)}</option>
              ))}
            </select>
            <label style={labelStyle}>Scale of that matrix</label>
            <select value={sourceScale} onChange={(e) => setSourceScale(e.target.value)}
              style={inputStyle}>
              <option value="auto">
                Auto-detect{detectedScale ? ` — ${detectedScale.label}` : ''}
              </option>
              <option value="raw_counts">Raw counts — normalize, then log1p</option>
              <option value="normalized_linear">Normalized, not logged — log1p only</option>
              <option value="log_normalized">Already log-normalized — use as-is</option>
              <option value="log_transformed">Already log-scaled — use as-is</option>
            </select>
            <div style={{ color: dark.faint, fontSize: 10, marginTop: 4, lineHeight: 1.45 }}>
              Decides what preprocessing is still owed. A reference distributed
              as log-normalized values must not be normalized and logged again —
              that distorts the marker ranking used to pick gene pairs. All of
              this happens on a private copy; the loaded dataset is not modified.
            </div>
            {sourceScale === 'auto' && detectedScale?.verdict === 'unknown' && (
              <div style={{ ...noticeStyle, background: 'rgba(233,162,59,0.14)', color: '#f0c987' }}>
                The scale of this matrix could not be identified. Training will
                leave it untransformed, which is the safer guess — set it
                explicitly above if you know what it is.
              </div>
            )}

            <label style={labelStyle}>Save classifier to</label>
            <input
              value={outPath}
              onChange={(e) => setOutPath(e.target.value)}
              placeholder="/path/to/my_classifier.pkl"
              style={inputStyle}
            />

            <div style={{ display: 'flex', gap: 8, marginTop: 10, flexWrap: 'wrap' }}>
              <NumField label="cells / type" value={nCellsPerType} onChange={setNCellsPerType}
                hint="0 = use every cell" />
              <NumField label="top genes" value={nTopGenes} onChange={setNTopGenes} />
              <NumField label="gene pairs" value={nTopGenePairs} onChange={setNTopGenePairs} />
              <NumField label="trees" value={nTrees} onChange={setNTrees} />
              <NumField label="PCs" value={nComps} onChange={setNComps} />
            </div>
          </>
        )}

        {tab === 'train' && trainResult && (
          <TrainResultView
            result={trainResult}
            onUse={() => {
              setPath(trainResult.path)
              setTrainResult(null)
              setTab('classify')
              setInspection(null)
            }}
            onAgain={() => setTrainResult(null)}
          />
        )}

        {/* ---------------------------------------------------------- */}
        {progress && (
          <div style={{ marginTop: 12 }}>
            <div style={{ height: 4, background: dark.inset, borderRadius: 2, overflow: 'hidden' }}>
              <div style={{
                width: `${Math.round(progress.frac * 100)}%`, height: '100%',
                background: dark.accent, transition: 'width 200ms',
              }} />
            </div>
            <div style={{ color: dark.sub, fontSize: 11, marginTop: 4 }}>{progress.message}</div>
          </div>
        )}

        <div style={{ display: 'flex', justifyContent: 'flex-end', gap: 8, marginTop: 16 }}>
          <button onClick={() => setOpen(false)} disabled={busy} style={ghostButton}>Close</button>
          {tab === 'classify' && !result && (
            <button onClick={runClassify} disabled={!canClassify} style={primaryButton(canClassify)}>
              {busy ? 'Classifying…' : 'Classify cells'}
            </button>
          )}
          {tab === 'train' && !trainResult && (
            <button
              onClick={runTrain}
              disabled={busy || !groupby || !outPath.trim() || !status?.available}
              style={primaryButton(!busy && !!groupby && !!outPath.trim() && !!status?.available)}
            >
              {busy ? 'Training…' : 'Train classifier'}
            </button>
          )}
        </div>
      </div>
    </div>
  )
}

function ClassifyResultView({
  result, onColorBy, onAgain,
}: {
  result: ClassifyResult
  onColorBy: (column: string) => void
  onAgain: () => void
}) {
  const max = Math.max(...result.composition.map((c) => c.n_cells), 1)
  return (
    <>
      <div style={{ ...noticeStyle, background: 'rgba(78,205,196,0.12)', color: '#9be7d8' }}>
        Classified {result.n_cells.toLocaleString()} cells into{' '}
        {result.composition.length} type{result.composition.length === 1 ? '' : 's'}.
      </div>

      <div style={sectionStyle}>
        <div style={sectionTitle}>Composition</div>
        {result.composition.map((c) => (
          <div key={c.name} style={{ marginBottom: 5 }}>
            <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: 11 }}>
              <span style={{ color: dark.text, display: 'flex', alignItems: 'center', gap: 5 }}>
                <span style={{ ...swatch, background: c.color }} />
                {c.name}
              </span>
              <span style={{ color: dark.faint, fontVariantNumeric: 'tabular-nums' }}>
                {c.n_cells.toLocaleString()} ({(c.frac * 100).toFixed(1)}%) · mean{' '}
                {c.mean_score.toFixed(2)}
              </span>
            </div>
            <div style={{ height: 4, background: dark.inset, borderRadius: 2, marginTop: 2 }}>
              <div style={{
                width: `${(c.n_cells / max) * 100}%`, height: '100%',
                background: c.color, borderRadius: 2,
              }} />
            </div>
          </div>
        ))}
      </div>

      {result.class_types.length > 0 && (
        <div style={sectionStyle}>
          <div style={sectionTitle}>Call quality</div>
          <div style={{ display: 'flex', flexWrap: 'wrap', gap: 4 }}>
            {result.class_types.map((t) => (
              <span key={t.name} style={{ ...pill, borderColor: t.color }}>
                <span style={{ ...swatch, background: t.color }} />
                {t.name} — {t.n_cells.toLocaleString()} ({(t.frac * 100).toFixed(0)}%)
              </span>
            ))}
          </div>
        </div>
      )}

      <div style={{ display: 'flex', flexWrap: 'wrap', gap: 6, marginTop: 12 }}>
        <button style={ghostButton} onClick={() => onColorBy(result.argmax_column)}>
          Color by cell type
        </button>
        <button style={ghostButton} onClick={() => onColorBy(result.score_column)}>
          Color by confidence
        </button>
        {result.type_column && (
          <button style={ghostButton} onClick={() => onColorBy(result.type_column!)}>
            Color by call quality
          </button>
        )}
        <button style={ghostButton} onClick={onAgain}>Classify again</button>
      </div>

      <div style={{ color: dark.faint, fontSize: 10, marginTop: 10, lineHeight: 1.5 }}>
        Per-class scores are in <code>.obsm[&apos;{result.obsm_key}&apos;]</code>. The
        Genes panel lists them under <code>◈ {result.obsm_key}</code> so you can
        colour by any single class, and the embedding picker can plot two
        classes against each other.
      </div>
    </>
  )
}

function TrainResultView({
  result, onUse, onAgain,
}: {
  result: TrainResult
  onUse: () => void
  onAgain: () => void
}) {
  return (
    <>
      <div style={{ ...noticeStyle, background: 'rgba(78,205,196,0.12)', color: '#9be7d8' }}>
        Trained on {result.n_cells_used.toLocaleString()} cells across{' '}
        {result.classifier.cell_type_classes.length} types.
      </div>
      <div style={sectionStyle}>
        <div style={sectionTitle}>Preprocessing applied</div>
        <div style={{ color: dark.text, fontSize: 11, lineHeight: 1.45 }}>
          {result.preprocessing.reason}
        </div>
        <div style={{ color: dark.faint, fontSize: 10, marginTop: 3 }}>
          detected <code>{result.preprocessing.detected_scale}</code>
          {result.preprocessing.overridden
            && <> · overridden to <code>{result.preprocessing.source_scale}</code></>}
          {' '}· normalize_total {result.preprocessing.normalize ? 'yes' : 'no'}
          {' '}· log1p {result.preprocessing.log1p ? 'yes' : 'no'}
          {' '}· HVG {result.preprocessing.hvg_flavor}
        </div>
      </div>
      <div style={sectionStyle}>
        <div style={sectionTitle}>Saved to</div>
        <code style={{ color: dark.text, fontSize: 11, wordBreak: 'break-all' }}>{result.path}</code>
        <div style={{ color: dark.faint, fontSize: 10, marginTop: 3 }}>
          {result.classifier.file_size_mb} MB · {result.classifier.n_gene_pairs} gene
          pairs over {result.classifier.n_genes} genes
        </div>
      </div>
      {result.dropped_labels.length > 0 && (
        <div style={{ ...noticeStyle, background: 'rgba(233,162,59,0.14)', color: '#f0c987' }}>
          Dropped {result.dropped_labels.join(', ')} — fewer than 3 cells each.
        </div>
      )}
      {result.n_genes_dropped_underscore > 0 && (
        <div style={{ ...noticeStyle, background: 'rgba(233,162,59,0.14)', color: '#f0c987' }}>
          Excluded {result.n_genes_dropped_underscore} gene(s) whose symbols
          contain an underscore — PySingleCellNet encodes gene pairs as
          <code> geneA_geneB</code> and splits on <code>_</code>, so those
          symbols cannot be represented.
        </div>
      )}
      <div style={sectionStyle}>
        <div style={sectionTitle}>Cells per type used</div>
        <div style={{ display: 'flex', flexWrap: 'wrap', gap: 4 }}>
          {result.training_counts.map((t) => (
            <span key={t.name} style={pill}>{t.name} — {t.n_cells}</span>
          ))}
        </div>
      </div>
      <div style={{ display: 'flex', gap: 6, marginTop: 12 }}>
        <button style={ghostButton} onClick={onUse}>Use it to classify this dataset</button>
        <button style={ghostButton} onClick={onAgain}>Train another</button>
      </div>
    </>
  )
}

function NumField({
  label, value, onChange, hint,
}: {
  label: string; value: number; onChange: (n: number) => void; hint?: string
}) {
  return (
    <div style={{ width: 92 }}>
      <span style={{ ...labelStyle, marginTop: 0 }}>{label}</span>
      <input
        type="number" min={0} value={value}
        onChange={(e) => onChange(Math.max(0, parseInt(e.target.value) || 0))}
        style={inputStyle}
      />
      {hint && <div style={{ color: dark.faint, fontSize: 9, marginTop: 2 }}>{hint}</div>}
    </div>
  )
}

const overlayStyle: React.CSSProperties = {
  position: 'fixed', inset: 0, background: 'rgba(0,0,0,0.5)', zIndex: 1000,
  display: 'flex', alignItems: 'center', justifyContent: 'center',
}

const cardStyle: React.CSSProperties = {
  background: dark.panel, color: dark.text, padding: 20, borderRadius: 8,
  width: 560, maxHeight: '86vh', overflowY: 'auto', fontSize: 13,
  border: `1px solid ${dark.border}`,
}

const tabButton: React.CSSProperties = {
  padding: '6px 14px', fontSize: 12, border: 'none', borderRadius: '4px 4px 0 0',
  cursor: 'pointer',
}

const labelStyle: React.CSSProperties = {
  display: 'block', fontSize: 11, color: dark.sub, margin: '10px 0 4px',
}

const inputStyle: React.CSSProperties = {
  width: '100%', padding: '5px 7px', fontSize: 12, background: dark.field,
  color: dark.text, border: `1px solid ${dark.border}`, borderRadius: 4,
  boxSizing: 'border-box',
}

const sectionStyle: React.CSSProperties = {
  marginTop: 10, padding: '8px 10px', background: dark.inset, borderRadius: 4,
}

const sectionTitle: React.CSSProperties = {
  fontSize: 10, color: dark.faint, textTransform: 'uppercase',
  letterSpacing: 0.5, marginBottom: 5,
}

const noticeStyle: React.CSSProperties = {
  marginTop: 10, padding: '7px 9px', borderRadius: 4, fontSize: 11, lineHeight: 1.45,
}

const checkRow: React.CSSProperties = {
  display: 'flex', alignItems: 'flex-start', gap: 6, marginTop: 10,
  fontSize: 11, color: dark.sub, lineHeight: 1.45, cursor: 'pointer',
}

const pill: React.CSSProperties = {
  display: 'inline-flex', alignItems: 'center', gap: 4, padding: '2px 7px',
  fontSize: 11, borderRadius: 10, border: `1px solid ${dark.border}`,
  color: dark.text,
}

const swatch: React.CSSProperties = {
  width: 8, height: 8, borderRadius: 2, display: 'inline-block', flexShrink: 0,
}

const ghostButton: React.CSSProperties = {
  padding: '5px 11px', fontSize: 12, background: 'transparent', color: dark.sub,
  border: `1px solid ${dark.border}`, borderRadius: 4, cursor: 'pointer',
}

function primaryButton(enabled: boolean): React.CSSProperties {
  return {
    padding: '5px 13px', fontSize: 12,
    background: enabled ? dark.accent : dark.field,
    color: enabled ? '#16213e' : dark.faint,
    border: 'none', borderRadius: 4,
    cursor: enabled ? 'pointer' : 'not-allowed',
    fontWeight: 600,
  }
}
