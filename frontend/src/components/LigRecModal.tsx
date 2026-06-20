/**
 * Ligand-Receptor tool (Analyze → Spatial → Ligand-Receptor).
 *
 * Detects ligand-receptor signaling at cellular resolution (a CytoSignal-style
 * port). Scores each cell for each L-R interaction from a spatial neighborhood,
 * tests significance against a spatial permutation null, and ranks interactions.
 * The user reviews the ranked table and visualizes a chosen interaction's score
 * as a continuous .obs column on the embedding.
 *
 * Backend: /scanpy/ligrec/suggest, /scanpy/ligrec/prepare (background task),
 * /scanpy/ligrec/finalize.
 *
 * Rollback: delete this file, remove its mount + the Spatial "Ligand-Receptor"
 * launcher in ScanpyModal, and the isLigRecModalOpen state in store.ts.
 */
import { useEffect, useMemo, useState } from 'react'
import { useStore } from '../store'
import { appendDataset, pollTask } from '../hooks/useData'

const API_BASE = '/api'

interface SummaryRow {
  interaction: string
  type: string
  ligand: string
  receptor: string
  classification?: string
  n_signif: number
  frac_signif: number
  mean_score: number
}

interface PrepareResult {
  summary: SummaryRow[]
  params: Record<string, unknown>
  n_tested: number
  n_significant: number
  n_dropped_capped: number
  interactions: string[]
}

type Phase = 'config' | 'running' | 'results'

const TIPS: Record<string, string> = {
  radius: 'Diffusion reach in coordinate units. Default ≈ 3× the median nearest-neighbor distance; smaller for short-range/contact-only signaling.',
  n_perm: 'Permutations for the significance null. More = finer p-values; the default targets ~100k null draws for your cell count.',
  min_cells: 'Drop interactions whose ligand/receptor genes are detected in fewer than this many cells (removes noise from rarely-expressed genes).',
  p_thresh: 'BH-adjusted p-value cutoff for calling a cell "significant".',
  recep_smooth: 'Average the receptor over direct spatial neighbors. Turn on for sparse / noisy data.',
  types: 'Diffusion = secreted ligands (Gaussian neighborhood). Contact = membrane-bound ligands (direct Delaunay neighbors). The database tags each pair.',
  max_pairs: 'Cap on the number of interactions scored; the most-expressed pairs are kept. Raise for a fuller scan (slower).',
  section_col: 'Optional categorical .obs column of tissue sections. When set, neighborhoods and the permutation null never cross section boundaries. Leave as one tissue if your sample is a single section.',
  gene_subset: 'Restrict eligible genes to a boolean .var column (e.g. highly_variable, spatially_variable). Off by default — L-R scoring normally wants all L-R genes regardless of variability.',
}

function Field({ label, tip, children }: { label: string; tip: string; children: React.ReactNode }) {
  return (
    <div style={{ marginBottom: 12 }}>
      <label style={{ display: 'block', fontSize: 11, color: '#aaa', marginBottom: 4 }}>{label}</label>
      {children}
      <div style={{ fontSize: 10, color: '#666', marginTop: 3, lineHeight: 1.4 }}>{tip}</div>
    </div>
  )
}

const numInput: React.CSSProperties = {
  width: '100%', padding: '6px 8px', fontSize: 12, backgroundColor: '#0f3460',
  color: '#eee', border: '1px solid #1a1a2e', borderRadius: 4,
}

export default function LigRecModal() {
  const isOpen = useStore((s) => s.isLigRecModalOpen)
  const setOpen = useStore((s) => s.setLigRecModalOpen)
  const setSelectedColorColumn = useStore((s) => s.setSelectedColorColumn)
  const refreshObsSummaries = useStore((s) => s.refreshObsSummaries)
  const addScanpyAction = useStore((s) => s.addScanpyAction)

  const [phase, setPhase] = useState<Phase>('config')
  const [radius, setRadius] = useState('')
  const [nPerm, setNPerm] = useState('')
  const [minCells, setMinCells] = useState('')
  const [pThresh, setPThresh] = useState('0.05')
  const [maxPairs, setMaxPairs] = useState('400')
  const [signalType, setSignalType] = useState<'both' | 'diffusion' | 'contact'>('both')
  const [recepSmooth, setRecepSmooth] = useState(false)
  const [sectionCol, setSectionCol] = useState('')
  const [geneSubset, setGeneSubset] = useState('')
  const [sectionOptions, setSectionOptions] = useState<string[]>([])
  const [geneSubsetOptions, setGeneSubsetOptions] = useState<{ name: string; n_true: number }[]>([])

  const [busy, setBusy] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const [prep, setPrep] = useState<PrepareResult | null>(null)
  const [checked, setChecked] = useState<Set<string>>(new Set())
  const [writeSig, setWriteSig] = useState(false)
  const [progress, setProgress] = useState<{ frac: number; message: string; startedAt: number; now: number } | null>(null)
  const [reusedPrior, setReusedPrior] = useState(false)

  const close = () => {
    setOpen(false)
    setPhase('config')
    setPrep(null)
    setError(null)
    setBusy(false)
    setChecked(new Set())
    setProgress(null)
    setReusedPrior(false)
  }

  // On open: prefill data-driven defaults, and reuse a prior result (if this
  // dataset already has one) so the user can re-select interactions without
  // re-running.
  useEffect(() => {
    if (!isOpen) return
    let cancelled = false
    fetch(appendDataset(`${API_BASE}/scanpy/ligrec/suggest`))
      .then((r) => (r.ok ? r.json() : null))
      .then((d) => {
        if (cancelled || !d) return
        setRadius((p) => (p ? p : String(d.radius)))
        setNPerm((p) => (p ? p : String(d.n_perm)))
        setMinCells((p) => (p ? p : String(d.min_cells)))
      })
      .catch(() => {})
    fetch(appendDataset(`${API_BASE}/scanpy/ligrec/result`))
      .then((r) => (r.ok ? r.json() : null))
      .then((d) => {
        if (cancelled || !d || !d.summary) return
        setPrep(d as PrepareResult)
        const top = (d.summary as SummaryRow[]).filter((s) => s.n_signif > 0).slice(0, 3).map((s) => s.interaction)
        setChecked(new Set(top))
        setReusedPrior(true)
        setPhase('results')
      })
      .catch(() => {})
    // Categorical .obs columns -> optional Section column.
    fetch(appendDataset(`${API_BASE}/obs/summaries`))
      .then((r) => (r.ok ? r.json() : null))
      .then((d) => {
        if (cancelled || !Array.isArray(d)) return
        setSectionOptions(d.filter((s: { dtype: string }) => s.dtype === 'category').map((s: { name: string }) => s.name))
      })
      .catch(() => {})
    // Boolean .var columns (e.g. highly_variable) -> optional gene subset.
    fetch(appendDataset(`${API_BASE}/var/boolean_columns`))
      .then((r) => (r.ok ? r.json() : null))
      .then((d) => {
        if (cancelled || !Array.isArray(d)) return
        setGeneSubsetOptions(d as { name: string; n_true: number }[])
      })
      .catch(() => {})
    return () => { cancelled = true }
  }, [isOpen])

  useEffect(() => {
    if (!isOpen) return
    const onKey = (e: KeyboardEvent) => { if (e.key === 'Escape') close() }
    document.addEventListener('keydown', onKey)
    return () => document.removeEventListener('keydown', onKey)
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [isOpen])

  const runPrepare = async () => {
    setError(null); setBusy(true); setPhase('running')
    setReusedPrior(false)
    setProgress({ frac: 0, message: 'Starting…', startedAt: Date.now(), now: Date.now() })
    try {
      const body: Record<string, unknown> = {
        p_thresh: parseFloat(pThresh) || 0.05,
        recep_smooth: recepSmooth,
        max_pairs: parseInt(maxPairs, 10) || 400,
      }
      if (radius.trim()) body.radius = parseFloat(radius)
      if (nPerm.trim()) body.n_perm = parseInt(nPerm, 10)
      if (minCells.trim()) body.min_cells = parseInt(minCells, 10)
      if (signalType !== 'both') body.types = [signalType]
      if (sectionCol) body.section_col = sectionCol
      if (geneSubset) body.gene_subset = geneSubset

      const resp = await fetch(appendDataset(`${API_BASE}/scanpy/ligrec/prepare`), {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(body),
      })
      if (!resp.ok) {
        const d = await resp.json().catch(() => ({}))
        throw new Error(d.detail || 'Failed to start scoring')
      }
      const { task_id } = await resp.json()
      const task = await pollTask(task_id, undefined, (s) => {
        setProgress((prev) => ({
          frac: typeof s.progress === 'number' ? s.progress : (prev?.frac ?? 0),
          message: s.message ?? prev?.message ?? 'Working…',
          startedAt: prev?.startedAt ?? Date.now(),
          now: Date.now(),
        }))
      })
      if (task.status !== 'completed') {
        throw new Error(task.error || `Scoring ${task.status}`)
      }
      const result = task.result as unknown as PrepareResult
      setPrep(result)
      // Pre-select the top significant interactions (up to 3).
      const top = result.summary.filter((s) => s.n_signif > 0).slice(0, 3).map((s) => s.interaction)
      setChecked(new Set(top))
      setPhase('results')
    } catch (e) {
      setError((e as Error).message)
      setPhase('config')
    } finally {
      setBusy(false)
    }
  }

  const toggle = (it: string) =>
    setChecked((prev) => {
      const next = new Set(prev)
      next.has(it) ? next.delete(it) : next.add(it)
      return next
    })

  const visualize = async () => {
    if (!prep || checked.size === 0) return
    setError(null); setBusy(true)
    try {
      const resp = await fetch(appendDataset(`${API_BASE}/scanpy/ligrec/finalize`), {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          interactions: [...checked],
          write_significance: writeSig,
        }),
      })
      if (!resp.ok) {
        const d = await resp.json().catch(() => ({}))
        throw new Error(d.detail || 'Failed to write columns')
      }
      const result = await resp.json()
      refreshObsSummaries()
      addScanpyAction({
        action: 'ligrec',
        params: { interactions: [...checked], write_significance: writeSig },
        result,
        timestamp: new Date().toISOString(),
      })
      setSelectedColorColumn(result.annotation_key)
      close()
    } catch (e) {
      setError((e as Error).message)
    } finally {
      setBusy(false)
    }
  }

  const sigCount = useMemo(() => prep?.summary.filter((s) => s.n_signif > 0).length ?? 0, [prep])

  if (!isOpen) return null

  return (
    <div
      onClick={close}
      style={{
        position: 'fixed', inset: 0, backgroundColor: 'rgba(0,0,0,0.6)',
        display: 'flex', alignItems: 'center', justifyContent: 'center', zIndex: 100,
      }}
    >
      <div
        onClick={(e) => e.stopPropagation()}
        style={{
          backgroundColor: '#16213e', border: '1px solid #0f3460', borderRadius: 8,
          padding: '20px 24px', width: 620, maxWidth: '92vw', maxHeight: '88vh',
          overflowY: 'auto', color: '#eee',
        }}
      >
        <div style={{ fontSize: 16, fontWeight: 600, marginBottom: 4 }}>Ligand-Receptor signaling</div>
        <div style={{ fontSize: 12, color: '#aaa', marginBottom: 16 }}>
          Detect where ligand-receptor signaling occurs across the tissue (CytoSignal-style).
          Parameters are pre-filled from your data — hover the tips before changing them.
        </div>

        {phase === 'running' && (() => {
          const frac = progress?.frac ?? 0
          const pct = Math.round(frac * 100)
          const elapsed = progress ? (progress.now - progress.startedAt) / 1000 : 0
          const eta = frac > 0.02 ? (elapsed * (1 - frac)) / frac : null
          return (
            <div style={{ padding: '20px 0' }}>
              <div style={{ fontSize: 13, color: '#9be7d8', marginBottom: 10 }}>
                {progress?.message ?? 'Scoring interactions…'}
              </div>
              <div style={{ height: 10, backgroundColor: '#0f1625', borderRadius: 5, overflow: 'hidden', border: '1px solid #0f3460' }}>
                <div style={{ width: `${pct}%`, height: '100%', backgroundColor: '#4ecdc4', transition: 'width 0.3s ease' }} />
              </div>
              <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: 11, color: '#888', marginTop: 6 }}>
                <span>{pct}%</span>
                <span>
                  {elapsed.toFixed(0)}s elapsed
                  {eta != null && ` · ~${eta.toFixed(0)}s left`}
                </span>
              </div>
            </div>
          )
        })()}

        {phase === 'config' && (
          <>
            <div style={{ display: 'flex', gap: 16 }}>
              <div style={{ flex: 1 }}>
                <Field label="Signaling type" tip={TIPS.types}>
                  <select value={signalType} onChange={(e) => setSignalType(e.target.value as typeof signalType)} style={numInput}>
                    <option value="both">Both (diffusion + contact)</option>
                    <option value="diffusion">Diffusion (secreted)</option>
                    <option value="contact">Contact (membrane)</option>
                  </select>
                </Field>
                <Field label="Diffusion radius" tip={TIPS.radius}>
                  <input type="number" value={radius} onChange={(e) => setRadius(e.target.value)} placeholder="auto" style={numInput} />
                </Field>
                <Field label="Permutations" tip={TIPS.n_perm}>
                  <input type="number" value={nPerm} onChange={(e) => setNPerm(e.target.value)} placeholder="auto" style={numInput} />
                </Field>
              </div>
              <div style={{ flex: 1 }}>
                <Field label="Min cells / gene" tip={TIPS.min_cells}>
                  <input type="number" value={minCells} onChange={(e) => setMinCells(e.target.value)} placeholder="auto" style={numInput} />
                </Field>
                <Field label="p-value threshold" tip={TIPS.p_thresh}>
                  <input type="number" step="0.01" value={pThresh} onChange={(e) => setPThresh(e.target.value)} style={numInput} />
                </Field>
                <Field label="Max interactions" tip={TIPS.max_pairs}>
                  <input type="number" value={maxPairs} onChange={(e) => setMaxPairs(e.target.value)} style={numInput} />
                </Field>
              </div>
            </div>
            <div style={{ display: 'flex', gap: 16 }}>
              <div style={{ flex: 1 }}>
                <Field label="Section column" tip={TIPS.section_col}>
                  <select value={sectionCol} onChange={(e) => setSectionCol(e.target.value)} style={numInput}>
                    <option value="">— treat as one tissue —</option>
                    {sectionOptions.map((n) => (
                      <option key={n} value={n}>{n}</option>
                    ))}
                  </select>
                </Field>
              </div>
              <div style={{ flex: 1 }}>
                <Field label="Gene subset" tip={TIPS.gene_subset}>
                  <select value={geneSubset} onChange={(e) => setGeneSubset(e.target.value)} style={numInput}>
                    <option value="">All genes</option>
                    {geneSubsetOptions.map((c) => (
                      <option key={c.name} value={c.name}>{c.name} ({c.n_true.toLocaleString()})</option>
                    ))}
                  </select>
                </Field>
              </div>
            </div>
            <label style={{ display: 'flex', alignItems: 'center', fontSize: 12, color: '#ccc', marginBottom: 4 }}>
              <input type="checkbox" checked={recepSmooth} onChange={(e) => setRecepSmooth(e.target.checked)} style={{ marginRight: 6 }} />
              Smooth receptor over neighbors
            </label>
            <div style={{ fontSize: 10, color: '#666', marginBottom: 14 }}>{TIPS.recep_smooth}</div>

            {error && <div style={errBox}>{error}</div>}

            <div style={{ display: 'flex', justifyContent: 'flex-end', gap: 8 }}>
              <button onClick={close} style={ghost}>Cancel</button>
              <button onClick={runPrepare} disabled={busy} style={primary(!busy)}>Score interactions</button>
            </div>
          </>
        )}

        {phase === 'results' && prep && (
          <>
            <div style={{ fontSize: 12, color: '#9be7d8', marginBottom: 10 }}>
              Tested {prep.n_tested} interactions · {sigCount} with significant signaling.
              {prep.n_dropped_capped > 0 && (
                <span style={{ color: '#e9a23b' }}> {prep.n_dropped_capped} lower-expressed pairs were capped.</span>
              )}
            </div>
            {reusedPrior && (
              <div style={{ fontSize: 11, color: '#aaa', marginBottom: 10, fontStyle: 'italic' }}>
                Showing a previous run on this dataset. Pick interactions to visualize, or
                use ← Change parameters to re-run with different settings.
              </div>
            )}

            <div style={{ border: '1px solid #0f3460', borderRadius: 4, overflow: 'hidden', marginBottom: 12 }}>
              <div style={{ display: 'flex', fontSize: 10, color: '#888', backgroundColor: '#0f1625', padding: '6px 8px', fontWeight: 600 }}>
                <span style={{ width: 22 }} />
                <span style={{ flex: 1 }}>Interaction</span>
                <span style={{ width: 64 }}>Type</span>
                <span style={{ width: 64, textAlign: 'right' }}>Sig cells</span>
                <span style={{ width: 64, textAlign: 'right' }}>Mean</span>
              </div>
              <div style={{ maxHeight: 320, overflowY: 'auto' }}>
                {prep.summary.map((s) => (
                  <label
                    key={s.interaction}
                    style={{
                      display: 'flex', alignItems: 'center', fontSize: 11, padding: '4px 8px',
                      borderTop: '1px solid #0f3460', cursor: 'pointer',
                      backgroundColor: checked.has(s.interaction) ? '#13243f' : 'transparent',
                      color: s.n_signif > 0 ? '#eee' : '#888',
                    }}
                  >
                    <span style={{ width: 22 }}>
                      <input type="checkbox" checked={checked.has(s.interaction)} onChange={() => toggle(s.interaction)} />
                    </span>
                    <span style={{ flex: 1, overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap' }}
                      title={s.classification || s.interaction}>
                      {s.ligand} → {s.receptor}
                    </span>
                    <span style={{ width: 64, color: s.type === 'contact' ? '#c39bff' : '#7fd4c8' }}>{s.type}</span>
                    <span style={{ width: 64, textAlign: 'right', fontWeight: s.n_signif > 0 ? 700 : 400 }}>{s.n_signif.toLocaleString()}</span>
                    <span style={{ width: 64, textAlign: 'right' }}>{s.mean_score.toFixed(2)}</span>
                  </label>
                ))}
              </div>
            </div>

            <label style={{ display: 'flex', alignItems: 'center', fontSize: 12, color: '#ccc', marginBottom: 12 }}>
              <input type="checkbox" checked={writeSig} onChange={(e) => setWriteSig(e.target.checked)} style={{ marginRight: 6 }} />
              Also write a significant/ns column for each selected interaction
            </label>

            {error && <div style={errBox}>{error}</div>}

            <div style={{ display: 'flex', justifyContent: 'space-between', gap: 8 }}>
              <button onClick={() => setPhase('config')} style={ghost}>← Change parameters</button>
              <div style={{ display: 'flex', gap: 8 }}>
                <button onClick={close} style={ghost}>Close</button>
                <button onClick={visualize} disabled={busy || checked.size === 0} style={primary(!busy && checked.size > 0)}>
                  {busy ? 'Writing…' : `Visualize ${checked.size || ''}`.trim()}
                </button>
              </div>
            </div>
          </>
        )}
      </div>
    </div>
  )
}

const ghost: React.CSSProperties = {
  padding: '6px 14px', fontSize: 12, backgroundColor: 'transparent', color: '#aaa',
  border: '1px solid #0f3460', borderRadius: 4, cursor: 'pointer',
}
function primary(enabled: boolean): React.CSSProperties {
  return {
    padding: '6px 14px', fontSize: 12, backgroundColor: enabled ? '#4ecdc4' : '#1a1a2e',
    color: enabled ? '#16213e' : '#555', border: 'none', borderRadius: 4,
    cursor: enabled ? 'pointer' : 'not-allowed', fontWeight: 600,
  }
}
const errBox: React.CSSProperties = {
  fontSize: 11, color: '#e94560', backgroundColor: 'rgba(233,69,96,0.15)',
  padding: 8, borderRadius: 4, marginBottom: 12,
}
