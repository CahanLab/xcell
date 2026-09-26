import { Component, useCallback, useEffect, useMemo, useRef, useState, type ReactNode } from 'react'
import { useStore } from '../store'
import {
  appendDataset, fetchFigures, fetchFigure, fetchFigureData, updateFigure, deleteFigure, createFigure, attachFigureToRecord,
} from '../hooks/useData'
import {
  FIGURE_KIND_LABELS, figureFilename, svgToPngBlob, canvasToPngBlob, blobToBase64, downloadBlob, paramsToBarplotConfig,
  type FigureRecord, type FigureSummary, type FigureData, type EnrichmentHeatmapData, type EnrichmentNetworkData,
  type CrosstabData, type ExpressionHeatmapData,
} from '../lib/figures'
import { FIGURE_SCHEMAS, coerceParams, type ParamField } from '../lib/figureSchemas'
import { standaloneSvg, downloadText } from '../lib/svgExport'
import EnrichmentHeatmapFigure from './figures/EnrichmentHeatmapFigure'
import EnrichmentNetworkFigure from './figures/EnrichmentNetworkFigure'
import CompositionBarplotFigure from './figures/CompositionBarplotFigure'
import ExpressionHeatmapFigure, { heatmapLegendFor } from './figures/ExpressionHeatmapFigure'
import NewFigureModal from './figures/NewFigureModal'

/** The Figures tab: a gallery of saved figure records (uns['xcell_figures']),
 *  each rendered from its stored inputs + params, with an editable params
 *  form (live preview, explicit Save), export, and a link back into the
 *  analysis record. A figure is data + a spec; pixels are only an export. */

const BG = '#16213e'
const ACCENT = '#4ecdc4'
const ALERT = '#e94560'

const styles = {
  root: { display: 'flex', height: '100%', minHeight: 0, backgroundColor: '#1a1a2e', color: '#eee' },
  rail: { width: '230px', flexShrink: 0, borderRight: '1px solid #0f3460', display: 'flex', flexDirection: 'column' as const, minHeight: 0 },
  railHead: { padding: '10px 12px', borderBottom: '1px solid #0f3460', display: 'flex', alignItems: 'center', justifyContent: 'space-between' },
  railList: { flex: 1, overflowY: 'auto' as const },
  item: (active: boolean) => ({
    padding: '8px 12px', cursor: 'pointer', borderBottom: '1px solid #0f1625',
    backgroundColor: active ? '#0f3460' : 'transparent',
  }),
  itemTitle: { fontSize: '12px', color: '#eee', whiteSpace: 'nowrap' as const, overflow: 'hidden', textOverflow: 'ellipsis' },
  itemMeta: { fontSize: '10px', color: '#888', marginTop: '2px' },
  canvas: { flex: 1, minWidth: 0, display: 'flex', flexDirection: 'column' as const, minHeight: 0 },
  toolbar: { display: 'flex', alignItems: 'center', gap: '8px', padding: '8px 12px', borderBottom: '1px solid #0f3460', flexWrap: 'wrap' as const },
  plotArea: { flex: 1, overflow: 'auto', padding: '16px' },
  side: { width: '290px', flexShrink: 0, borderLeft: '1px solid #0f3460', overflowY: 'auto' as const, padding: '10px 12px' },
  label: { display: 'block', fontSize: '11px', color: '#aaa', margin: '8px 0 3px' },
  input: { width: '100%', padding: '5px 7px', fontSize: '12px', backgroundColor: '#0f3460', color: '#eee', border: '1px solid #1a1a2e', borderRadius: '4px', boxSizing: 'border-box' as const },
  select: { width: '100%', padding: '5px 7px', fontSize: '12px', backgroundColor: '#0f3460', color: '#eee', border: '1px solid #1a1a2e', borderRadius: '4px' },
  button: { padding: '6px 12px', fontSize: '12px', borderRadius: '4px', cursor: 'pointer', border: 'none', backgroundColor: '#0f3460', color: '#ccc' },
  primary: { backgroundColor: ALERT, color: '#fff' },
  success: { backgroundColor: ACCENT, color: '#000' },
  disabled: { opacity: 0.5, cursor: 'not-allowed' },
  muted: { fontSize: '11px', color: '#888' },
  section: { fontSize: '11px', color: ACCENT, textTransform: 'uppercase' as const, letterSpacing: '0.04em', margin: '14px 0 4px' },
  error: { fontSize: '12px', color: ALERT, padding: '8px 12px', backgroundColor: 'rgba(233,69,96,0.1)', borderRadius: '4px', margin: '12px' },
  toast: { fontSize: '11px', color: ACCENT },
}

interface RecordStepLite { index: number; action: string; title: string }

/** A renderer bug must show a message in the plot area, never blank the app
 *  (there is no error boundary above the centre panel). Keyed on the figure
 *  id + data so a fresh figure retries. */
class RendererBoundary extends Component<{ children: ReactNode }, { error: string | null }> {
  state = { error: null as string | null }
  static getDerivedStateFromError(e: Error) { return { error: e.message } }
  render() {
    if (this.state.error) return <div style={styles.error}>The figure could not be drawn: {this.state.error}</div>
    return this.props.children
  }
}

function ParamInput({ field, value, onChange, dynamicOptions }: { field: ParamField; value: unknown; onChange: (v: unknown) => void; dynamicOptions?: string[] }) {
  if (field.type === 'bool') {
    return (
      <label style={{ display: 'flex', alignItems: 'center', gap: '6px', fontSize: '12px', color: '#ccc', margin: '8px 0 3px' }}>
        <input type="checkbox" checked={Boolean(value)} onChange={(e) => onChange(e.target.checked)} /> {field.label}
      </label>
    )
  }
  if (field.type === 'select') {
    const options = field.optionsFrom
      ? [{ value: '', label: '— none —' }, ...(dynamicOptions ?? []).map((v) => ({ value: v, label: v }))]
      : field.options!
    return (
      <>
        <label style={styles.label} title={field.help}>{field.label}</label>
        <select style={styles.select} value={value == null ? '' : String(value)} onChange={(e) => onChange(field.optionsFrom && e.target.value === '' ? null : e.target.value)}>
          {options.map((o) => <option key={o.value} value={o.value}>{o.label}</option>)}
        </select>
      </>
    )
  }
  return (
    <>
      <label style={styles.label} title={field.help}>{field.label}{field.type === 'nullable-number' ? ' (blank = off)' : ''}</label>
      <input
        style={styles.input} type="number" min={field.min} max={field.max} step={field.step ?? (field.type === 'int' ? 1 : 'any')}
        value={value == null ? '' : String(value)} onChange={(e) => onChange(e.target.value)}
      />
    </>
  )
}

export default function FiguresView() {
  const activeSlot = useStore((s) => s.activeSlot)
  const figures = useStore((s) => s.figures)
  const setFigures = useStore((s) => s.setFigures)
  const activeFigureId = useStore((s) => s.activeFigureId)
  const setActiveFigureId = useStore((s) => s.setActiveFigureId)
  const figuresVersion = useStore((s) => s.figuresVersion)
  const refreshFigures = useStore((s) => s.refreshFigures)
  const setAnalysisRecordOpen = useStore((s) => s.setAnalysisRecordOpen)

  const [record, setRecord] = useState<FigureRecord | null>(null)
  const [form, setForm] = useState<Record<string, unknown>>({})
  const [title, setTitle] = useState('')
  const [caption, setCaption] = useState('')
  const [data, setData] = useState<FigureData | null>(null)
  const [dataError, setDataError] = useState<string | null>(null)
  const [error, setError] = useState<string | null>(null)
  const [busy, setBusy] = useState(false)
  const [toast, setToast] = useState<string | null>(null)
  const [pngScale, setPngScale] = useState(2)
  const [showNew, setShowNew] = useState(false)
  const [steps, setSteps] = useState<RecordStepLite[]>([])
  const svgRef = useRef<SVGSVGElement>(null)
  const canvasRef = useRef<HTMLCanvasElement>(null)
  const plotRef = useRef<HTMLDivElement>(null)
  const [plotWidth, setPlotWidth] = useState(800)
  // Every preview request gets a sequence number; only the latest response
  // may land, so a slow older request can never overwrite a newer one, and a
  // request made for figure A is dropped once B is active. The payload also
  // names its figure, and the view only draws data tagged for the current one.
  const reqSeq = useRef(0)
  const [dataVersion, setDataVersion] = useState(0)

  // Reconcile the active id against the *fetched* list, never the stale one:
  // a just-created figure is set active before its refetch lands, and a guard
  // on the old list would bounce it back to the first figure.
  useEffect(() => {
    fetchFigures(activeSlot)
      .then((list) => {
        setFigures(list)
        const cur = useStore.getState().activeFigureId
        if (!cur || !list.some((f) => f.id === cur)) setActiveFigureId(list[0]?.id ?? null)
      })
      .catch((e) => setError((e as Error).message))
  }, [activeSlot, figuresVersion, setFigures, setActiveFigureId])

  useEffect(() => {
    const el = plotRef.current
    if (!el) return
    const ro = new ResizeObserver(() => setPlotWidth(Math.max(320, el.clientWidth - 32)))
    ro.observe(el)
    return () => ro.disconnect()
  }, [])

  // Load the record when the active figure changes; reset the form from it.
  useEffect(() => {
    if (!activeFigureId) { setRecord(null); setData(null); return }
    let cancelled = false
    // The previous figure's data must not reach the new figure's renderer,
    // and any preview still in flight for it is now stale.
    reqSeq.current += 1
    setData(null); setError(null); setDataError(null); setToast(null)
    fetchFigure(activeFigureId, activeSlot)
      .then((r) => {
        if (cancelled) return
        setRecord(r); setForm({ ...r.params }); setTitle(r.title); setCaption(r.caption ?? '')
      })
      .catch((e) => { if (!cancelled) setError((e as Error).message) })
    return () => { cancelled = true }
  }, [activeFigureId, activeSlot])

  useEffect(() => {
    fetch(appendDataset('/api/record', activeSlot))
      .then((r) => r.json())
      .then((rec: { steps?: RecordStepLite[] }) => setSteps((rec.steps ?? []).map((s) => ({ index: s.index, action: s.action, title: s.title }))))
      .catch(() => setSteps([]))
  }, [activeSlot, activeFigureId])

  const coerced = useMemo(() => (record ? coerceParams(record.kind, form) : {}), [record, form])
  const dirty = useMemo(() => {
    if (!record) return false
    if (title !== record.title || caption !== (record.caption ?? '')) return true
    return Object.keys(coerced).some((k) => JSON.stringify(coerced[k]) !== JSON.stringify(record.params[k]))
  }, [record, coerced, title, caption])

  // Preview: fetch data for the current form values, debounced.
  useEffect(() => {
    if (!record) return
    const id = record.id
    const override: Record<string, unknown> = {}
    for (const k of Object.keys(coerced)) if (JSON.stringify(coerced[k]) !== JSON.stringify(record.params[k])) override[k] = coerced[k]
    const t = setTimeout(() => {
      const seq = ++reqSeq.current
      fetchFigureData(id, Object.keys(override).length ? override : null, activeSlot)
        .then((d) => { if (seq === reqSeq.current) { setData(d); setDataError(null); setDataVersion((v) => v + 1) } })
        .catch((e) => { if (seq === reqSeq.current) { setData(null); setDataError((e as Error).message) } })
    }, 300)
    return () => clearTimeout(t)
  }, [record, coerced, activeSlot])

  const save = useCallback(async () => {
    if (!record || !dirty) return
    setBusy(true); setError(null)
    try {
      const body: { title?: string; caption?: string; params?: Record<string, unknown> } = {}
      if (title !== record.title) body.title = title
      if (caption !== (record.caption ?? '')) body.caption = caption
      const delta: Record<string, unknown> = {}
      for (const k of Object.keys(coerced)) if (JSON.stringify(coerced[k]) !== JSON.stringify(record.params[k])) delta[k] = coerced[k]
      if (Object.keys(delta).length) body.params = delta
      const updated = await updateFigure(record.id, body, activeSlot)
      setRecord(updated); setForm({ ...updated.params }); setTitle(updated.title); setCaption(updated.caption ?? '')
      refreshFigures()
      setToast('Saved')
    } catch (e) {
      setError((e as Error).message)
    } finally {
      setBusy(false)
    }
  }, [record, dirty, title, caption, coerced, activeSlot, refreshFigures])

  const revert = useCallback(() => {
    if (!record) return
    setForm({ ...record.params }); setTitle(record.title); setCaption(record.caption ?? '')
  }, [record])

  const isCanvasKind = record?.kind === 'expression_heatmap'

  // The heatmap is a canvas (a genes × bins matrix is too many cells for SVG), so it exports as PNG only.
  const renderPng = useCallback(async (scale: number): Promise<Blob> => {
    if (isCanvasKind) {
      if (!canvasRef.current || !data) throw new Error('The figure has not drawn yet')
      return canvasToPngBlob(canvasRef.current, { scale, title, legend: heatmapLegendFor(data as ExpressionHeatmapData), background: BG })
    }
    if (!svgRef.current) throw new Error('The figure has not drawn yet')
    return svgToPngBlob(svgRef.current, scale, BG)
  }, [isCanvasKind, data, title])

  const exportSvg = useCallback(() => {
    if (!svgRef.current || !record || isCanvasKind) return
    downloadText(figureFilename(record, 'svg'), standaloneSvg(svgRef.current.outerHTML, { background: BG }), 'image/svg+xml')
  }, [record, isCanvasKind])

  const exportPng = useCallback(async () => {
    if (!record) return
    try {
      downloadBlob(figureFilename(record, 'png'), await renderPng(pngScale))
    } catch (e) {
      setError((e as Error).message)
    }
  }, [record, pngScale, renderPng])

  const attach = useCallback(async () => {
    if (!record) return
    setBusy(true); setError(null)
    try {
      const b64 = await blobToBase64(await renderPng(2))
      const out = await attachFigureToRecord(record.id, b64, caption || title, activeSlot)
      setToast(out.step_index != null ? `Attached to record step ${out.step_index}` : 'Attached to the record')
    } catch (e) {
      setError((e as Error).message)
    } finally {
      setBusy(false)
    }
  }, [record, caption, title, activeSlot, renderPng])

  const duplicate = useCallback(async () => {
    if (!record) return
    setBusy(true); setError(null)
    try {
      const copy = await createFigure({ kind: record.kind, title: `${record.title} copy`, caption: record.caption, inputs: record.inputs, params: record.params }, activeSlot)
      refreshFigures(); setActiveFigureId(copy.id)
    } catch (e) {
      setError((e as Error).message)
    } finally {
      setBusy(false)
    }
  }, [record, activeSlot, refreshFigures, setActiveFigureId])

  const remove = useCallback(async () => {
    if (!record) return
    if (!window.confirm(`Delete figure "${record.title}"? The analysis record keeps its steps; only the figure spec goes.`)) return
    setBusy(true); setError(null)
    try {
      await deleteFigure(record.id, activeSlot)
      setActiveFigureId(null); refreshFigures()
    } catch (e) {
      setError((e as Error).message)
    } finally {
      setBusy(false)
    }
  }, [record, activeSlot, refreshFigures, setActiveFigureId])

  const fields = record ? FIGURE_SCHEMAS[record.kind] : []
  const provenanceSteps = record ? steps.filter((s) => record.provenance.steps.includes(s.index)) : []
  const createdStep = record?.provenance.created_step != null ? steps.find((s) => s.index === record.provenance.created_step) : undefined

  // The payload names its figure and kind; anything else is stale and not drawn.
  const dataMatchesKind = Boolean(data && record && data.figure_id === record.id && data.kind === record.kind)
  const dynamicOptions = data && 'b_categories' in data ? (data as CrosstabData).b_categories : undefined
  const renderer = record && data && dataMatchesKind ? (
    record.kind === 'enrichment_heatmap' ? (
      <EnrichmentHeatmapFigure ref={svgRef} data={data as EnrichmentHeatmapData} params={coerced} title={title} width={plotWidth} background={BG} />
    ) : record.kind === 'enrichment_network' ? (
      <EnrichmentNetworkFigure ref={svgRef} data={data as EnrichmentNetworkData} params={coerced} title={title} width={plotWidth} height={Math.max(360, Math.round(plotWidth * 0.66))} background={BG} />
    ) : record.kind === 'composition_barplot' ? (
      <CompositionBarplotFigure ref={svgRef} data={data as CrosstabData} config={paramsToBarplotConfig(coerced)} width={plotWidth} height={Math.max(360, Math.round(plotWidth * 0.6))} title={title} legend="inline" background={BG} />
    ) : (
      <div>
        <div style={{ fontSize: '13px', color: '#eee', fontWeight: 600, marginBottom: '6px' }}>{title}</div>
        <ExpressionHeatmapFigure ref={canvasRef} data={data as ExpressionHeatmapData} legends="inline" />
      </div>
    )
  ) : null

  return (
    <div style={styles.root} data-figures-view>
      <div style={styles.rail}>
        <div style={styles.railHead}>
          <span style={{ fontSize: '12px', color: '#aaa' }}>Figures ({figures.length})</span>
          <button style={{ ...styles.button, ...styles.primary, padding: '4px 10px' }} onClick={() => setShowNew(true)}>New figure…</button>
        </div>
        <div style={styles.railList}>
          {figures.length === 0 && (
            <div style={{ ...styles.muted, padding: '12px' }}>
              No figures yet. Run a batch enrichment (Analyze → Genes → Gene-set Enrichment → All groups) and click <em>Heatmap figure…</em>, or start one here.
            </div>
          )}
          {figures.map((f: FigureSummary) => (
            <div key={f.id} style={styles.item(f.id === activeFigureId)} onClick={() => setActiveFigureId(f.id)} title={f.title}>
              <div style={styles.itemTitle}>{f.title}</div>
              <div style={styles.itemMeta}>{FIGURE_KIND_LABELS[f.kind]} · {f.id}{f.n_provenance_steps ? ` · ${f.n_provenance_steps} step${f.n_provenance_steps === 1 ? '' : 's'}` : ''}</div>
            </div>
          ))}
        </div>
      </div>

      <div style={styles.canvas}>
        <div style={styles.toolbar}>
          <button style={{ ...styles.button, ...(!record || !data || isCanvasKind ? styles.disabled : {}) }} disabled={!record || !data || isCanvasKind} onClick={exportSvg} title={isCanvasKind ? 'The expression heatmap is a canvas: PNG only' : 'Vector export'}>Export SVG</button>
          <button style={{ ...styles.button, ...(!record || !data ? styles.disabled : {}) }} disabled={!record || !data} onClick={exportPng}>Export PNG</button>
          <select style={{ ...styles.select, width: '64px' }} value={pngScale} onChange={(e) => setPngScale(Number(e.target.value))} title="PNG scale">
            {[1, 2, 3, 4].map((s) => <option key={s} value={s}>{s}×</option>)}
          </select>
          <button style={{ ...styles.button, ...(!record || !data || busy || dirty ? styles.disabled : {}) }} disabled={!record || !data || busy || dirty} onClick={attach} title={dirty ? 'Save the figure first: the record must be able to reproduce what it shows' : "Rasterise and attach to the analysis record, linked to this figure's step"}>Attach to record</button>
          <button style={{ ...styles.button, ...(!record || busy ? styles.disabled : {}) }} disabled={!record || busy} onClick={duplicate}>Duplicate</button>
          <button style={{ ...styles.button, ...(!record || busy ? styles.disabled : {}) }} disabled={!record || busy} onClick={remove}>Delete</button>
          {toast && <span style={styles.toast}>{toast}</span>}
        </div>
        {error && <div style={styles.error}>{error}</div>}
        <div style={styles.plotArea} ref={plotRef}>
          {!record && <div style={styles.muted}>Select a figure, or create one.</div>}
          {record && dataError && (
            <div style={{ ...styles.error, margin: 0 }}>
              {dataError}
              <div style={{ ...styles.muted, marginTop: '6px' }}>The figure spec is kept so its provenance is not lost; re-run the analysis it points at to draw it again.</div>
            </div>
          )}
          {record && (!data || !dataMatchesKind) && !dataError && <div style={styles.muted}>Loading…</div>}
          <RendererBoundary key={`${record?.id ?? ''}:${dataVersion}`}>{renderer}</RendererBoundary>
        </div>
      </div>

      <div style={styles.side}>
        {record ? (
          <>
            <div style={styles.section}>Figure</div>
            <label style={styles.label}>Title</label>
            <input style={styles.input} value={title} onChange={(e) => setTitle(e.target.value)} />
            <label style={styles.label}>Caption</label>
            <textarea style={{ ...styles.input, minHeight: '48px', resize: 'vertical' as const }} value={caption} onChange={(e) => setCaption(e.target.value)} />
            <div style={styles.section}>Parameters</div>
            {fields.map((f) => (
              <ParamInput key={f.name} field={f} value={form[f.name]} dynamicOptions={dynamicOptions} onChange={(v) => setForm((prev) => ({ ...prev, [f.name]: v }))} />
            ))}
            <div style={{ display: 'flex', gap: '6px', marginTop: '12px' }}>
              <button style={{ ...styles.button, ...styles.success, ...(!dirty || busy ? styles.disabled : {}) }} disabled={!dirty || busy} onClick={save}>Save</button>
              <button style={{ ...styles.button, ...(!dirty ? styles.disabled : {}) }} disabled={!dirty} onClick={revert}>Revert</button>
            </div>
            <div style={styles.section}>Provenance</div>
            <div style={styles.muted}>
              {createdStep ? <div>Created at step {createdStep.index}: {createdStep.title}</div> : <div>Created step not in the record.</div>}
              {provenanceSteps.length > 0 ? (
                <div style={{ marginTop: '4px' }}>
                  Built from:
                  {provenanceSteps.map((s) => (
                    <div key={s.index} style={{ color: '#ccc', cursor: 'pointer' }} onClick={() => setAnalysisRecordOpen(true)} title="Open the analysis record">
                      #{s.index} {s.title}
                    </div>
                  ))}
                </div>
              ) : <div style={{ marginTop: '4px' }}>No input steps matched in the record.</div>}
              <div style={{ marginTop: '6px' }}>Inputs: {JSON.stringify(record.inputs)}</div>
              <div>Created {record.created_at}{record.updated_at !== record.created_at ? ` · edited ${record.updated_at}` : ''}</div>
            </div>
          </>
        ) : <div style={styles.muted}>Parameters appear here.</div>}
      </div>

      {showNew && (
        <NewFigureModal
          onClose={() => setShowNew(false)}
          onCreated={(rec) => { setShowNew(false); refreshFigures(); setActiveFigureId(rec.id) }}
        />
      )}
    </div>
  )
}
