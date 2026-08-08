import { useCallback, useEffect, useState } from 'react'
import { useStore } from '../store'
import { appendDataset, pollTask, refreshSchema } from '../hooks/useData'

/**
 * Localize — predict where each dissociated cell came from, using a spatial
 * dataset as the map.
 *
 * The panel is arranged around the one thing that makes this trustworthy: you
 * are shown the gene overlap *before* running, offered a cross-validation that
 * measures accuracy on the reference itself, and told afterwards how many cells
 * the method declined to place. A coordinate map is smooth and convincing
 * whether or not it is right, so none of that is optional decoration.
 */

const API = '/api'

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

interface ReferenceSlot {
  slot: string
  filename: string
  n_cells: number
  n_genes: number
  has_spatial: boolean
  is_query: boolean
}

interface Overlap {
  n_shared: number
  n_reference_genes: number
  frac_of_reference: number
  n_missing: number
  missing_from_query: string[]
  sufficient: boolean
  severity: 'ok' | 'warn' | 'error'
}

interface CrossValidation {
  median_error: number | null
  median_error_normalized: number | null
  baseline_centroid: number | null
  baseline_random: number | null
  structure_correlation: number | null
  confidence_error_correlation: number | null
  fraction_within: Record<string, number | null>
  per_group: Record<string, { n: number; median_error: number | null }>
  same_platform: boolean
  n_holdout: number
}

interface LocalizeResult {
  embedding_name: string
  n_cells: number
  n_unplaced: number
  n_shared_genes: number
  n_missing_genes: number
  n_reference_cells: number
  section_col: string | null
  confidence_distribution: {
    median: number
    q25: number
    q75: number
    'n_below_0.2': number
    'n_below_0.5': number
  }
}

const TIPS: Record<string, string> = {
  k: 'How many spatial cells vote on each prediction. Too few is noisy; too many drags every cell toward the tissue centre.',
  transform: 'Applied to each dataset separately — that is what removes platform-level differences in per-gene capture. z-score is the safe default across platforms; rank additionally survives any monotone difference.',
  metric: 'How similarity between two cells is measured. Correlation on z-scored data and cosine coincide.',
  aggregation: 'How the k neighbours become one point. weighted_mean is the classic estimator; densest lands the cell in real tissue when its neighbours sit in two separate patches, though it cannot say which patch; best_match snaps to a real reference cell.',
  min_confidence: 'Cells whose neighbours disagree about location this badly get no coordinate at all, rather than a fabricated one. Leave at 0 to place everything and filter later.',
}

export default function LocalizeModal() {
  const isOpen = useStore((s) => s.isLocalizeModalOpen)
  const setOpen = useStore((s) => s.setLocalizeModalOpen)
  const refreshObsSummaries = useStore((s) => s.refreshObsSummaries)

  const [refs, setRefs] = useState<ReferenceSlot[]>([])
  const [reference, setReference] = useState('secondary')
  const [overlap, setOverlap] = useState<Overlap | null>(null)
  const [sectionOptions, setSectionOptions] = useState<string[]>([])

  const [k, setK] = useState('15')
  const [transform, setTransform] = useState('zscore')
  const [metric, setMetric] = useState('correlation')
  const [aggregation, setAggregation] = useState('weighted_mean')
  const [minConfidence, setMinConfidence] = useState('0')
  const [sectionCol, setSectionCol] = useState('')
  const [keyAdded, setKeyAdded] = useState('X_spatial_pred')

  const [busy, setBusy] = useState<'' | 'check' | 'run'>('')
  const [progress, setProgress] = useState<number | null>(null)
  const [error, setError] = useState<string | null>(null)
  const [cv, setCv] = useState<CrossValidation | null>(null)
  const [result, setResult] = useState<LocalizeResult | null>(null)

  const close = () => {
    setOpen(false)
    setError(null); setCv(null); setResult(null); setBusy(''); setProgress(null)
  }

  useEffect(() => {
    if (!isOpen) return
    const onKey = (e: KeyboardEvent) => { if (e.key === 'Escape') close() }
    window.addEventListener('keydown', onKey)
    return () => window.removeEventListener('keydown', onKey)
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [isOpen])

  // Which slots can be a reference, and how well the panels match.
  const loadSuggest = useCallback(async (slot: string) => {
    setError(null)
    try {
      const r = await fetch(appendDataset(`${API}/localize/suggest?reference=${slot}`))
      const body = await r.json()
      if (!r.ok) {
        setOverlap(null)
        // A slot with no coordinates is a normal state to be in, not an error
        // worth a red box — the picker below already says which are usable.
        if (r.status !== 400) setError(body.detail || `HTTP ${r.status}`)
        return
      }
      setRefs(body.references || [])
      setOverlap(body.overlap || null)
      setSectionOptions(body.reference_sections || [])
      setK((prev) => (prev === '15' && body.suggested_k ? String(body.suggested_k) : prev))
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e))
    }
  }, [])

  useEffect(() => {
    if (isOpen) loadSuggest(reference)
  }, [isOpen, reference, loadSuggest])

  const body = () => ({
    reference,
    k: Number(k) || 15,
    transform, metric, aggregation,
    min_confidence: Number(minConfidence) || 0,
    section_col: sectionCol || null,
  })

  const runCheck = async () => {
    setBusy('check'); setError(null); setCv(null); setProgress(0)
    try {
      const r = await fetch(appendDataset(`${API}/localize/cross_validate`), {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ ...body(), holdout_fraction: 0.2 }),
      })
      const payload = await r.json()
      if (!r.ok) throw new Error(payload.detail || `HTTP ${r.status}`)
      const status = await pollTask(payload.task_id, undefined,
        (s) => setProgress(s.progress ?? null))
      if (status.status !== 'completed') throw new Error(status.error || 'Check failed')
      setCv(status.result as unknown as CrossValidation)
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e))
    } finally {
      setBusy(''); setProgress(null)
    }
  }

  const runLocalize = async () => {
    setBusy('run'); setError(null); setResult(null); setProgress(0)
    try {
      const r = await fetch(appendDataset(`${API}/localize/prepare`), {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ ...body(), key_added: keyAdded }),
      })
      const payload = await r.json()
      if (!r.ok) throw new Error(payload.detail || `HTTP ${r.status}`)
      const status = await pollTask(payload.task_id, undefined,
        (s) => setProgress(s.progress ?? null))
      if (status.status !== 'completed') throw new Error(status.error || 'Localize failed')
      setResult(status.result as unknown as LocalizeResult)
      // A new embedding *and* new .obs columns, so both refresh channels are
      // needed: the schema drives the embedding picker, the summaries version
      // drives the Cell Manager's column list.
      await refreshSchema()
      refreshObsSummaries()
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e))
    } finally {
      setBusy(''); setProgress(null)
    }
  }

  if (!isOpen) return null

  const usable = refs.filter((r) => r.has_spatial && !r.is_query)
  const blocked = overlap != null && !overlap.sufficient

  return (
    <div onClick={close} style={overlayStyle}>
      <div onClick={(e) => e.stopPropagation()} style={cardStyle}>
        <div style={headerStyle}>
          <div>
            <div style={{ fontSize: 15, fontWeight: 600, color: '#fff' }}>
              Localize — predict spatial coordinates
            </div>
            <div style={{ fontSize: 11, color: dark.faint, marginTop: 2 }}>
              Place this dataset's cells on a tissue map borrowed from a spatial dataset
            </div>
          </div>
          <button onClick={close} style={{ ...ghost, fontSize: 16 }}>×</button>
        </div>

        <div style={{ overflowY: 'auto', padding: '12px 16px', flex: 1 }}>
          {error && <div style={errBox}>{error}</div>}

          {/* Reference */}
          <div style={sectionLabel}>Spatial reference</div>
          {usable.length === 0 ? (
            <div style={{ ...noticeStyle, background: 'rgba(233,162,59,0.14)', color: '#f0c987' }}>
              No loaded dataset has spatial coordinates to borrow. Load one into the
              other slot with <b>File → Load…</b>, then reopen this tool.
            </div>
          ) : (
            <select value={reference} onChange={(e) => setReference(e.target.value)}
                    style={{ ...input, width: '100%' }}>
              {usable.map((r) => (
                <option key={r.slot} value={r.slot}>
                  {r.slot} — {r.filename} ({r.n_cells.toLocaleString()} cells)
                </option>
              ))}
            </select>
          )}

          {/* Gene overlap — shown before anything runs, on purpose. */}
          {overlap && (
            <div style={{
              ...noticeStyle,
              background: overlap.severity === 'ok' ? 'rgba(78,205,196,0.12)'
                : overlap.severity === 'warn' ? 'rgba(233,162,59,0.14)'
                : 'rgba(233,69,96,0.15)',
              color: overlap.severity === 'ok' ? '#bfeae6'
                : overlap.severity === 'warn' ? '#f0c987' : '#ff8fa3',
            }}>
              <b>{overlap.n_shared.toLocaleString()}</b> of the reference's{' '}
              {overlap.n_reference_genes.toLocaleString()} genes are present here
              ({(overlap.frac_of_reference * 100).toFixed(0)}%).
              {!overlap.sufficient && ' Too few to predict from — the result would look confident and mean nothing.'}
              {overlap.sufficient && overlap.severity !== 'ok' &&
                ' Expect a weaker mapping; check the accuracy below before trusting it.'}
              {overlap.n_missing > 0 && (
                <div style={{ marginTop: 4, fontSize: 10, opacity: 0.8 }}>
                  Missing e.g. {overlap.missing_from_query.slice(0, 8).join(', ')}
                  {overlap.n_missing > 8 ? ` … (+${overlap.n_missing - 8})` : ''}
                </div>
              )}
            </div>
          )}

          {/* Parameters */}
          <div style={{ ...sectionLabel, marginTop: 14 }}>Parameters</div>
          <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 8 }}>
            <Field label="Neighbours (k)" tip={TIPS.k}>
              <input value={k} onChange={(e) => setK(e.target.value)} style={input} />
            </Field>
            <Field label="Expression transform" tip={TIPS.transform}>
              <select value={transform} onChange={(e) => setTransform(e.target.value)} style={input}>
                <option value="zscore">z-score per dataset (default)</option>
                <option value="rank">rank within cell</option>
                <option value="log1p">log1p</option>
                <option value="none">none</option>
              </select>
            </Field>
            <Field label="Similarity" tip={TIPS.metric}>
              <select value={metric} onChange={(e) => setMetric(e.target.value)} style={input}>
                <option value="correlation">correlation</option>
                <option value="cosine">cosine</option>
                <option value="euclidean">euclidean</option>
              </select>
            </Field>
            <Field label="Aggregation" tip={TIPS.aggregation}>
              <select value={aggregation} onChange={(e) => setAggregation(e.target.value)} style={input}>
                <option value="weighted_mean">weighted mean</option>
                <option value="median">median</option>
                <option value="densest">densest cluster</option>
                <option value="best_match">best match</option>
              </select>
            </Field>
            <Field label="Min confidence" tip={TIPS.min_confidence}>
              <input value={minConfidence} onChange={(e) => setMinConfidence(e.target.value)} style={input} />
            </Field>
            <Field label="Reference section column" tip="Keeps neighbourhoods inside one tissue section, so a coordinate is never averaged across the gap between cuts.">
              <select value={sectionCol} onChange={(e) => setSectionCol(e.target.value)} style={input}>
                <option value="">none</option>
                {sectionOptions.map((c) => <option key={c} value={c}>{c}</option>)}
              </select>
            </Field>
          </div>

          {/* Accuracy check */}
          <div style={{ ...sectionLabel, marginTop: 16 }}>Accuracy</div>
          <div style={{ fontSize: 11, color: dark.sub, marginBottom: 6 }}>
            Holds out a fifth of the reference and predicts it from the rest, with
            these exact parameters. Ground truth is known there, so the error is
            exact — but both halves come from the same platform, so the real
            scRNA-seq mapping will be worse than this.
          </div>
          <button onClick={runCheck} disabled={!!busy || blocked || usable.length === 0}
                  style={{ ...ghost, opacity: busy || blocked ? 0.5 : 1 }}>
            {busy === 'check' ? 'Checking…' : 'Check accuracy on the reference'}
          </button>

          {cv && (
            <div style={{ marginTop: 8, padding: 10, background: dark.inset, borderRadius: 4 }}>
              <div style={{ display: 'flex', gap: 16, flexWrap: 'wrap', fontSize: 12, color: dark.text }}>
                <span><b style={{ color: dark.accent }}>{fmt(cv.median_error)}</b> median error</span>
                <span style={{ color: dark.sub }}>vs {fmt(cv.baseline_centroid)} predicting the tissue centre</span>
                <span style={{ color: dark.sub }}>vs {fmt(cv.baseline_random)} at random</span>
              </div>
              <div style={{ fontSize: 11, color: dark.sub, marginTop: 6 }}>
                {pct(cv.fraction_within?.['10%'])} of held-out cells land within 10% of
                the tissue diameter. Structure preserved r={fmt(cv.structure_correlation, 2)};
                confidence tracks error r={fmt(cv.confidence_error_correlation, 2)}.
              </div>
              {beatsBaseline(cv) === false && (
                <div style={{ ...noticeStyle, background: 'rgba(233,69,96,0.15)', color: '#ff8fa3' }}>
                  This does not beat simply predicting the middle of the tissue. The
                  shared genes carry little positional information — a map from these
                  parameters would not mean anything.
                </div>
              )}
              {Object.keys(cv.per_group || {}).length > 0 && (
                <div style={{ marginTop: 6, fontSize: 11, color: dark.sub }}>
                  {Object.entries(cv.per_group).map(([g, v]) => (
                    <span key={g} style={{ marginRight: 12 }}>{g}: {fmt(v.median_error)}</span>
                  ))}
                </div>
              )}
            </div>
          )}

          {/* Run */}
          <div style={{ ...sectionLabel, marginTop: 16 }}>Apply</div>
          <div style={{ display: 'flex', gap: 6, alignItems: 'center' }}>
            <input value={keyAdded} onChange={(e) => setKeyAdded(e.target.value)}
                   style={{ ...input, flex: 1 }} placeholder="X_spatial_pred" />
            <button onClick={runLocalize} disabled={!!busy || blocked || usable.length === 0}
                    style={{
                      ...ghost, background: dark.accent, color: '#0b1020',
                      fontWeight: 600, border: `1px solid ${dark.accent}`,
                      opacity: busy || blocked || usable.length === 0 ? 0.5 : 1,
                    }}>
              {busy === 'run' ? 'Localizing…' : 'Localize'}
            </button>
          </div>

          {progress !== null && busy && (
            <div style={{ marginTop: 8, height: 4, background: dark.inset, borderRadius: 2 }}>
              <div style={{
                width: `${Math.round((progress || 0) * 100)}%`, height: '100%',
                background: dark.accent, borderRadius: 2, transition: 'width 200ms',
              }} />
            </div>
          )}

          {result && (
            <div style={{ marginTop: 10, padding: 10, background: 'rgba(78,205,196,0.12)', borderRadius: 4 }}>
              <div style={{ fontSize: 12, color: '#bfeae6', fontWeight: 600 }}>
                Wrote <code>.obsm['{result.embedding_name}']</code> for{' '}
                {(result.n_cells - result.n_unplaced).toLocaleString()} of{' '}
                {result.n_cells.toLocaleString()} cells
                {result.n_unplaced > 0 && ` (${result.n_unplaced.toLocaleString()} left unplaced)`}.
              </div>
              <div style={{ fontSize: 11, color: dark.sub, marginTop: 5 }}>
                Median confidence {result.confidence_distribution.median.toFixed(2)}.{' '}
                <b>{result.confidence_distribution['n_below_0.2'].toLocaleString()}</b> cells
                score below 0.2 — their transcriptional neighbours disagree about where
                they belong, so their coordinates carry little information.
              </div>
              <div style={{ fontSize: 11, color: dark.sub, marginTop: 5 }}>
                Select <b>{result.embedding_name}</b> in the embedding picker, then
                colour by <b>{result.embedding_name}_confidence</b> to see which regions
                of the map to trust.
              </div>
            </div>
          )}
        </div>
      </div>
    </div>
  )
}

function beatsBaseline(cv: CrossValidation): boolean | null {
  if (cv.median_error == null || cv.baseline_centroid == null) return null
  return cv.median_error < cv.baseline_centroid
}

function fmt(v: number | null | undefined, digits = 1): string {
  return v == null ? '—' : v.toFixed(digits)
}

function pct(v: number | null | undefined): string {
  return v == null ? '—' : `${Math.round(v * 100)}%`
}

function Field({ label, tip, children }: {
  label: string; tip?: string; children: React.ReactNode
}) {
  return (
    <label style={{ display: 'block' }} title={tip}>
      <div style={{ fontSize: 11, color: dark.sub, marginBottom: 3 }}>{label}</div>
      {children}
    </label>
  )
}

const overlayStyle: React.CSSProperties = {
  position: 'fixed', inset: 0, backgroundColor: 'rgba(0,0,0,0.6)',
  display: 'flex', alignItems: 'center', justifyContent: 'center', zIndex: 1000,
}

const cardStyle: React.CSSProperties = {
  backgroundColor: dark.panel, border: `1px solid ${dark.border}`,
  borderRadius: 8, width: 'min(680px, 94vw)', maxHeight: '90vh',
  display: 'flex', flexDirection: 'column', boxShadow: '0 8px 32px rgba(0,0,0,0.5)',
}

const headerStyle: React.CSSProperties = {
  display: 'flex', alignItems: 'center', justifyContent: 'space-between',
  padding: '12px 16px', borderBottom: `1px solid ${dark.border}`,
}

const input: React.CSSProperties = {
  background: dark.inset, border: `1px solid ${dark.border}`, borderRadius: 4,
  color: dark.text, fontSize: 12, padding: '6px 8px', outline: 'none', width: '100%',
}

const ghost: React.CSSProperties = {
  background: 'transparent', border: `1px solid ${dark.border}`, borderRadius: 4,
  color: dark.sub, cursor: 'pointer', fontSize: 12, padding: '6px 10px',
}

const sectionLabel: React.CSSProperties = {
  fontSize: 10, textTransform: 'uppercase', letterSpacing: 0.6,
  color: dark.faint, marginBottom: 6,
}

const noticeStyle: React.CSSProperties = {
  marginTop: 8, padding: '8px 10px', borderRadius: 4, fontSize: 11.5,
}

const errBox: React.CSSProperties = {
  marginBottom: 10, padding: '8px 10px', borderRadius: 4,
  background: 'rgba(233,69,96,0.15)', color: '#ff8fa3', fontSize: 12,
}
