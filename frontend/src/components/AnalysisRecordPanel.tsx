import { useCallback, useEffect, useState } from 'react'
import { useStore } from '../store'
import { appendDataset } from '../hooks/useData'
import FileBrowser from './FileBrowser'

/**
 * The analysis record: what this session did, and the export that makes it
 * reproducible outside xcell.
 *
 * Recording is always on — the backend logs every mutating operation whether or
 * not this panel has ever been opened, because the alternative (a record button
 * users forget to press) loses the analysis outright. What the user controls
 * here is the *span* the exported document covers ("Start report here"), the
 * annotations, and the figures.
 *
 * Each step arrives already translated by the backend, including its fidelity,
 * so the panel shows the same claim the exported notebook will make rather than
 * a second opinion computed here.
 */

const dark = {
  panel: '#16213e',
  border: '#0f3460',
  inset: '#0f1625',
  accent: '#4ecdc4',
  alert: '#e94560',
  warn: '#e9a23b',
  text: '#ccc',
  sub: '#aaa',
  faint: '#888',
}

type Fidelity = 'exact' | 'xcell' | 'manual'

interface RecordStep {
  index: number
  action: string
  title: string
  summary: string
  fidelity: Fidelity
  warnings: string[]
  code: string[]
  note: string | null
  figure_ids: string[]
  n_active: number | null
  n_total: number | null
  in_report: boolean
  timestamp: string
}

interface RecordFigure {
  id: string
  caption: string
  step_index: number | null
}

interface RecordCounts {
  total: number
  exact: number
  xcell: number
  manual: number
  subset: number
}

interface AnalysisRecordData {
  title: string
  abstract: string
  source: { path?: string; kind?: string; n_cells?: number; n_genes?: number }
  report_start: number
  steps: RecordStep[]
  figures: RecordFigure[]
  counts: RecordCounts
}

const FIDELITY_STYLE: Record<Fidelity, { label: string; color: string; title: string }> = {
  exact: {
    label: 'exact',
    color: dark.accent,
    title: 'The exported code is the library call xcell actually made',
  },
  xcell: {
    label: 'xcell',
    color: dark.warn,
    title: 'Reproducible by calling xcell’s own Python API',
  },
  manual: {
    label: 'manual',
    color: dark.faint,
    title: 'Described in the export, but not reproduced as code',
  },
}

export default function AnalysisRecordPanel() {
  const isOpen = useStore((s) => s.isAnalysisRecordOpen)
  const setOpen = useStore((s) => s.setAnalysisRecordOpen)
  const activeSlot = useStore((s) => s.activeSlot)

  const [data, setData] = useState<AnalysisRecordData | null>(null)
  const [error, setError] = useState<string | null>(null)
  const [busy, setBusy] = useState(false)

  const [title, setTitle] = useState('')
  const [abstract, setAbstract] = useState('')
  const [editingNote, setEditingNote] = useState<number | null>(null)
  const [noteDraft, setNoteDraft] = useState('')
  const [expanded, setExpanded] = useState<Set<number>>(new Set())

  const [format, setFormat] = useState<'ipynb' | 'md' | 'both'>('ipynb')
  const [outputDir, setOutputDir] = useState('')
  const [filename, setFilename] = useState('analysis')
  const [includeFigures, setIncludeFigures] = useState(true)
  const [includeCode, setIncludeCode] = useState(true)
  const [browsing, setBrowsing] = useState(false)
  const [written, setWritten] = useState<string[]>([])

  const load = useCallback(async () => {
    setError(null)
    try {
      const r = await fetch(appendDataset('/api/record'))
      if (!r.ok) throw new Error((await r.json().catch(() => ({}))).detail || `HTTP ${r.status}`)
      const body: AnalysisRecordData = await r.json()
      setData(body)
      setTitle(body.title)
      setAbstract(body.abstract)
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e))
    }
  }, [])

  useEffect(() => {
    if (isOpen) {
      setWritten([])
      load()
    }
  }, [isOpen, activeSlot, load])

  useEffect(() => {
    if (!isOpen) return
    const onKey = (e: KeyboardEvent) => { if (e.key === 'Escape') setOpen(false) }
    window.addEventListener('keydown', onKey)
    return () => window.removeEventListener('keydown', onKey)
  }, [isOpen, setOpen])

  const post = useCallback(async (url: string, body?: unknown, method = 'POST') => {
    setBusy(true)
    setError(null)
    try {
      const r = await fetch(appendDataset(url), {
        method,
        headers: { 'Content-Type': 'application/json' },
        ...(body === undefined ? {} : { body: JSON.stringify(body) }),
      })
      const payload = await r.json().catch(() => ({}))
      if (!r.ok) throw new Error(payload.detail || `HTTP ${r.status}`)
      return payload
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e))
      return null
    } finally {
      setBusy(false)
    }
  }, [])

  if (!isOpen) return null

  const saveMeta = async () => {
    await post('/api/record/meta', { title, abstract }, 'PUT')
    load()
  }

  const saveNote = async (index: number) => {
    await post(`/api/record/step/${index}/note`, { note: noteDraft })
    setEditingNote(null)
    load()
  }

  const markStart = async () => {
    await post('/api/record/mark')
    load()
  }

  const clearRecord = async () => {
    if (!window.confirm('Clear the recorded steps? The loaded dataset stays; only the history is dropped.')) return
    await post('/api/record', undefined, 'DELETE')
    load()
  }

  const removeFigure = async (id: string) => {
    await post(`/api/record/figure/${id}`, undefined, 'DELETE')
    load()
  }

  const runExport = async () => {
    if (!outputDir.trim()) {
      setError('Choose a folder to export into.')
      return
    }
    setWritten([])
    const result = await post('/api/record/export', {
      output_dir: outputDir.trim(),
      filename: filename.trim() || 'analysis',
      format,
      include_figures: includeFigures,
      include_code: includeCode,
    })
    if (result) setWritten((result.files as { path: string }[]).map((f) => f.path))
  }

  const steps = data?.steps ?? []
  const counts = data?.counts
  const figuresByStep = new Map<number, RecordFigure[]>()
  const looseFigures: RecordFigure[] = []
  for (const f of data?.figures ?? []) {
    if (f.step_index === null) looseFigures.push(f)
    else figuresByStep.set(f.step_index, [...(figuresByStep.get(f.step_index) ?? []), f])
  }

  return (
    <div
      onClick={() => setOpen(false)}
      style={{
        position: 'fixed', inset: 0, backgroundColor: 'rgba(0,0,0,0.6)',
        display: 'flex', alignItems: 'center', justifyContent: 'center', zIndex: 1000,
      }}
    >
      <div
        onClick={(e) => e.stopPropagation()}
        style={{
          backgroundColor: dark.panel, border: `1px solid ${dark.border}`,
          borderRadius: 8, width: 'min(860px, 94vw)', maxHeight: '90vh',
          display: 'flex', flexDirection: 'column', boxShadow: '0 8px 32px rgba(0,0,0,0.5)',
        }}
      >
        {/* Header */}
        <div style={{
          display: 'flex', alignItems: 'center', justifyContent: 'space-between',
          padding: '12px 16px', borderBottom: `1px solid ${dark.border}`,
        }}>
          <div>
            <div style={{ fontSize: 15, fontWeight: 600, color: '#fff' }}>Analysis record</div>
            <div style={{ fontSize: 11, color: dark.faint, marginTop: 2 }}>
              {data?.source?.path
                ? `${data.source.path} — ${(data.source.n_cells ?? 0).toLocaleString()} cells × ${(data.source.n_genes ?? 0).toLocaleString()} genes`
                : 'No source recorded'}
            </div>
          </div>
          <div style={{ display: 'flex', gap: 6 }}>
            <button onClick={load} style={ghost} title="Re-read the record">Refresh</button>
            <button onClick={() => setOpen(false)} style={{ ...ghost, fontSize: 16 }}>×</button>
          </div>
        </div>

        {error && (
          <div style={{
            margin: '10px 16px 0', padding: '8px 10px', borderRadius: 4,
            background: 'rgba(233,69,96,0.15)', color: '#ff8fa3', fontSize: 12,
          }}>{error}</div>
        )}

        <div style={{ overflowY: 'auto', padding: '12px 16px', flex: 1 }}>
          {/* Reproducibility summary — the same claim the export will make. */}
          {counts && (
            <div style={{
              display: 'flex', gap: 14, flexWrap: 'wrap', alignItems: 'baseline',
              padding: '8px 10px', background: dark.inset, borderRadius: 4,
              fontSize: 11, color: dark.sub, marginBottom: 12,
            }}>
              <span style={{ color: '#fff', fontWeight: 600 }}>
                {counts.total} step{counts.total === 1 ? '' : 's'} in the report
              </span>
              <span><b style={{ color: dark.accent }}>{counts.exact}</b> re-run as written</span>
              <span><b style={{ color: dark.warn }}>{counts.xcell}</b> need xcell</span>
              <span><b style={{ color: dark.faint }}>{counts.manual}</b> manual</span>
              {counts.subset > 0 && (
                <span style={{ color: dark.warn }}>
                  {counts.subset} ran on a cell selection
                </span>
              )}
            </div>
          )}

          {/* Document metadata */}
          <div style={{ display: 'grid', gap: 6, marginBottom: 14 }}>
            <input
              value={title}
              onChange={(e) => setTitle(e.target.value)}
              onBlur={saveMeta}
              placeholder="Title (e.g. Cortex atlas — clustering)"
              style={input}
            />
            <textarea
              value={abstract}
              onChange={(e) => setAbstract(e.target.value)}
              onBlur={saveMeta}
              placeholder="Abstract — what this analysis set out to do"
              rows={2}
              style={{ ...input, resize: 'vertical', fontFamily: 'inherit' }}
            />
          </div>

          {/* Steps */}
          <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', marginBottom: 6 }}>
            <span style={sectionLabel}>Steps</span>
            <span style={{ display: 'flex', gap: 6 }}>
              <button onClick={markStart} style={ghost} title="Everything after this point becomes the report; nothing is discarded">
                Start report here
              </button>
              <button onClick={clearRecord} style={{ ...ghost, color: dark.alert }}>Clear</button>
            </span>
          </div>

          <div style={{ border: `1px solid ${dark.border}`, borderRadius: 4, overflow: 'hidden' }}>
            {steps.length === 0 && (
              <div style={{ padding: 14, fontSize: 12, color: dark.faint }}>
                Nothing recorded yet. Run an analysis and it will appear here.
              </div>
            )}
            {steps.map((step) => {
              const style = FIDELITY_STYLE[step.fidelity]
              const isOpenRow = expanded.has(step.index)
              const figs = figuresByStep.get(step.index) ?? []
              return (
                <div
                  key={step.index}
                  style={{
                    borderTop: step.index === 0 ? 'none' : `1px solid ${dark.border}`,
                    padding: '8px 10px',
                    background: step.in_report ? 'transparent' : 'rgba(255,255,255,0.02)',
                    opacity: step.in_report ? 1 : 0.55,
                  }}
                >
                  <div style={{ display: 'flex', gap: 8, alignItems: 'baseline' }}>
                    <span style={{ fontSize: 10, color: dark.faint, width: 20, flexShrink: 0 }}>
                      {step.index}
                    </span>
                    <div style={{ flex: 1, minWidth: 0 }}>
                      <div style={{ display: 'flex', gap: 6, alignItems: 'baseline', flexWrap: 'wrap' }}>
                        <span style={{ fontSize: 12, fontWeight: 600, color: '#fff' }}>{step.title}</span>
                        <span title={style.title} style={{
                          fontSize: 9, textTransform: 'uppercase', letterSpacing: 0.4,
                          color: style.color, border: `1px solid ${style.color}`,
                          borderRadius: 3, padding: '0 4px',
                        }}>{style.label}</span>
                        {step.n_active !== null && (
                          <span title="This operation ran on an active cell selection, not the whole dataset" style={{
                            fontSize: 9, textTransform: 'uppercase', letterSpacing: 0.4,
                            color: dark.warn, background: 'rgba(233,162,59,0.15)',
                            borderRadius: 3, padding: '0 4px',
                          }}>
                            selection {step.n_active.toLocaleString()}/{(step.n_total ?? 0).toLocaleString()}
                          </span>
                        )}
                      </div>
                      <div style={{ fontSize: 11, color: dark.sub, marginTop: 2 }}>{step.summary}</div>
                      {step.warnings.map((w, i) => (
                        <div key={i} style={{ fontSize: 10, color: dark.warn, marginTop: 3 }}>⚠ {w}</div>
                      ))}
                      {step.note && (
                        <div style={{
                          fontSize: 11, color: dark.accent, marginTop: 4,
                          borderLeft: `2px solid ${dark.accent}`, paddingLeft: 6,
                        }}>{step.note}</div>
                      )}
                      {figs.length > 0 && (
                        <div style={{ fontSize: 10, color: dark.faint, marginTop: 4 }}>
                          {figs.map((f) => (
                            <span key={f.id} style={{ marginRight: 8 }}>
                              ◈ {f.caption || 'figure'}
                              <button onClick={() => removeFigure(f.id)} style={{ ...ghost, padding: '0 3px', fontSize: 10 }}>×</button>
                            </span>
                          ))}
                        </div>
                      )}

                      {editingNote === step.index ? (
                        <div style={{ display: 'flex', gap: 6, marginTop: 6 }}>
                          <input
                            autoFocus
                            value={noteDraft}
                            onChange={(e) => setNoteDraft(e.target.value)}
                            onKeyDown={(e) => { if (e.key === 'Enter') saveNote(step.index) }}
                            placeholder="Why this step, or what it showed"
                            style={{ ...input, flex: 1, fontSize: 11 }}
                          />
                          <button onClick={() => saveNote(step.index)} style={ghost}>Save</button>
                          <button onClick={() => setEditingNote(null)} style={ghost}>Cancel</button>
                        </div>
                      ) : (
                        <div style={{ display: 'flex', gap: 8, marginTop: 4 }}>
                          <button
                            onClick={() => { setEditingNote(step.index); setNoteDraft(step.note ?? '') }}
                            style={{ ...ghost, fontSize: 10, padding: '1px 4px' }}
                          >
                            {step.note ? 'Edit note' : 'Add note'}
                          </button>
                          {step.code.length > 0 && (
                            <button
                              onClick={() => setExpanded((prev) => {
                                const next = new Set(prev)
                                if (next.has(step.index)) next.delete(step.index); else next.add(step.index)
                                return next
                              })}
                              style={{ ...ghost, fontSize: 10, padding: '1px 4px' }}
                            >
                              {isOpenRow ? 'Hide code' : 'Show code'}
                            </button>
                          )}
                        </div>
                      )}

                      {isOpenRow && step.code.length > 0 && (
                        <pre style={{
                          margin: '6px 0 0', padding: 8, background: dark.inset,
                          borderRadius: 3, fontSize: 10.5, color: '#d6e6e4',
                          overflowX: 'auto', whiteSpace: 'pre',
                        }}>{step.code.join('\n')}</pre>
                      )}
                    </div>
                  </div>
                </div>
              )
            })}
          </div>

          {looseFigures.length > 0 && (
            <div style={{ marginTop: 10, fontSize: 11, color: dark.sub }}>
              <span style={sectionLabel}>Unattached figures</span>
              {looseFigures.map((f) => (
                <span key={f.id} style={{ marginLeft: 8 }}>
                  ◈ {f.caption || 'figure'}
                  <button onClick={() => removeFigure(f.id)} style={{ ...ghost, padding: '0 3px' }}>×</button>
                </span>
              ))}
            </div>
          )}

          <div style={{ fontSize: 10.5, color: dark.faint, marginTop: 8 }}>
            Capture a figure with the <b>◧</b> button on the plot — it attaches to the
            most recent step.
          </div>

          {/* Export */}
          <div style={{ marginTop: 18, paddingTop: 14, borderTop: `1px solid ${dark.border}` }}>
            <span style={sectionLabel}>Export</span>
            <div style={{ display: 'flex', gap: 14, alignItems: 'center', margin: '8px 0', flexWrap: 'wrap' }}>
              {(['ipynb', 'md', 'both'] as const).map((f) => (
                <label key={f} style={{ fontSize: 12, color: dark.text, display: 'flex', gap: 4, alignItems: 'center' }}>
                  <input type="radio" checked={format === f} onChange={() => setFormat(f)} />
                  {f === 'ipynb' ? 'Notebook (.ipynb)' : f === 'md' ? 'Markdown (.md)' : 'Both'}
                </label>
              ))}
              <label style={{ fontSize: 12, color: dark.text, display: 'flex', gap: 4, alignItems: 'center' }}>
                <input type="checkbox" checked={includeCode} onChange={(e) => setIncludeCode(e.target.checked)} />
                code cells
              </label>
              <label style={{ fontSize: 12, color: dark.text, display: 'flex', gap: 4, alignItems: 'center' }}>
                <input type="checkbox" checked={includeFigures} onChange={(e) => setIncludeFigures(e.target.checked)} />
                figures
              </label>
            </div>

            <div style={{ display: 'flex', gap: 6, alignItems: 'center' }}>
              <input
                value={outputDir}
                onChange={(e) => setOutputDir(e.target.value)}
                placeholder="/path/to/folder"
                style={{ ...input, flex: 1 }}
              />
              <button
                onClick={() => setBrowsing((b) => !b)}
                style={{ ...ghost, color: browsing ? dark.accent : dark.sub }}
              >
                Browse
              </button>
              <input
                value={filename}
                onChange={(e) => setFilename(e.target.value)}
                placeholder="analysis"
                style={{ ...input, width: 130 }}
              />
              <button
                onClick={runExport}
                disabled={busy}
                style={{
                  ...ghost, background: dark.accent, color: '#0b1020',
                  fontWeight: 600, borderColor: dark.accent,
                  opacity: busy ? 0.6 : 1,
                }}
              >
                {busy ? 'Writing…' : 'Export'}
              </button>
            </div>

            {browsing && (
              <div style={{ marginTop: 8, padding: 8, background: dark.inset, borderRadius: 4 }}>
                <div style={{ fontSize: 10, color: dark.faint, marginBottom: 6 }}>
                  Navigate to the folder to write into — the notebook goes next to
                  your data so it can re-read it.
                </div>
                <FileBrowser
                  kind="data"
                  height={180}
                  onError={setError}
                  onNavigate={(dir) => setOutputDir(dir)}
                  onSelect={() => { /* folders only; navigating is the selection */ }}
                />
                <button onClick={() => setBrowsing(false)} style={{ ...ghost, marginTop: 6 }}>Done</button>
              </div>
            )}

            {written.length > 0 && (
              <div style={{
                marginTop: 10, padding: '8px 10px', borderRadius: 4,
                background: 'rgba(78,205,196,0.12)', fontSize: 11, color: '#bfeae6',
              }}>
                <div style={{ fontWeight: 600, marginBottom: 3 }}>Written:</div>
                {written.map((p) => <div key={p} style={{ fontFamily: 'monospace', fontSize: 10.5 }}>{p}</div>)}
              </div>
            )}
          </div>
        </div>
      </div>
    </div>
  )
}

const input: React.CSSProperties = {
  background: dark.inset,
  border: `1px solid ${dark.border}`,
  borderRadius: 4,
  color: dark.text,
  fontSize: 12,
  padding: '6px 8px',
  outline: 'none',
}

const ghost: React.CSSProperties = {
  background: 'transparent',
  border: `1px solid ${dark.border}`,
  borderRadius: 4,
  color: dark.sub,
  cursor: 'pointer',
  fontSize: 11,
  padding: '4px 8px',
}

const sectionLabel: React.CSSProperties = {
  fontSize: 10,
  textTransform: 'uppercase',
  letterSpacing: 0.6,
  color: dark.faint,
}
