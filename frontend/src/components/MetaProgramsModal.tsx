/**
 * NMF meta-programs (Analyze → Genes → NMF Meta-Programs).
 *
 * The multi-sample half of GeneNMF. Factorizes each sample on its own at every
 * rank in a sweep, then clusters every resulting program by gene-weight
 * similarity into consensus "meta-programs". A program fitted on pooled data
 * can be a batch effect; one that turns up independently in several samples,
 * at several ranks, cannot — which is what sample coverage measures and what
 * the single-sample NMF tool cannot tell you.
 *
 * The similarity heatmap is the diagnostic: each meta-program should read as a
 * bright block on the diagonal. A dim or bleeding block, or a negative
 * silhouette, means the consensus is not real.
 *
 * Backend: /gene_nmf/meta/run (background task), /gene_nmf/meta/{result,runs,columns}.
 * Per-cell scores land in obsm[key] → score pills.
 *
 * Rollback: delete this file, remove its mount in App.tsx, the Genes "NMF
 * Meta-Programs" launcher in ScanpyModal, and isMetaProgramsModalOpen in store.ts.
 */
import { useEffect, useMemo, useRef, useState } from 'react'
import { useStore } from '../store'
import { appendDataset, pollTask, refreshSchema, useDataActions } from '../hooks/useData'
import { defaultTransformFor, transformNoteFor, type NmfTransform } from '../lib/nmfTransform'
import { clusterBands, similarityColor, coverageLabel } from '../lib/metaPrograms'

const API_BASE = '/api'

interface MetaProgram {
  name: string
  genes: string[]
  weights: number[]
  n_programs: number
  n_genes: number
  sample_coverage: number
  silhouette: number
  mean_similarity: number
}

interface MetaResult {
  obsm_key?: string
  key?: string
  metaprograms: MetaProgram[]
  composition: Record<string, Record<string, number>>
  samples: string[]
  program_labels: string[]
  program_samples: string[]
  clusters: number[]
  order: number[]
  similarity: number[][]
  skipped: Array<{ sample: string; n_cells: number; reason: string }>
  ks_used: number[]
  n_programs: number
  n_dropped: number
}

interface SampleColumn { name: string; n_samples: number; min_cells: number }
interface BoolColumn { name: string; n_true: number }
interface LayerInfo { name: string; verdict?: string }

type Phase = 'config' | 'running' | 'results'
type Tab = 'programs' | 'similarity' | 'composition'

const TIPS = {
  sample: 'The .obs column separating samples, sections or donors. Each is factorized on its own — that independence is what makes a recurrent program trustworthy.',
  ks: 'Ranks to fit per sample. Sweeping means a program need not be an artifact of one arbitrary k. GeneNMF sweeps 4–9; wider costs proportionally more time.',
  nMp: 'How many consensus meta-programs to cut the program tree into. Start near the widest k you swept and check the heatmap — over-cutting splits one block in two.',
  genes: 'NMF is normally run on highly variable genes. For multi-sample work a batch-aware HVG column is better still, since genes variable in only one sample are exactly what you do not want driving a consensus.',
  minCells: 'Samples smaller than this are skipped rather than contributing a noisy factorization.',
  specificity: 'Down-weights genes loading on many programs before the consensus is cut. The meta-program default (5) is higher than the single-sample tool\'s, because averaging across a cluster smooths the weighting first.',
  weightExplained: 'A meta-program keeps its top consensus genes up to this share of total weight. Higher = longer signatures. 0.8 here rather than 0.5, since after specificity weighting the top genes hold most of the mass.',
  minConfidence: 'A gene must appear in more than this fraction of a cluster\'s own programs to enter the consensus — the filter that removes genes carried by a single outlier program.',
  metric: 'How program similarity is measured: cosine over the full weight vectors (GeneNMF ≥0.6, more sensitive) or Jaccard over the top-gene sets.',
  key: 'Where results go: .obsm[key] for per-cell meta-program scores.',
  layer: 'Which matrix to factorize. NMF needs non-negative values — raw counts or a log-normalized layer. Scaled/centered data will be rejected.',
}

const dark = {
  panel: '#16213e', border: '#0f3460', inset: '#0f1625',
  accent: '#4ecdc4', alert: '#e94560', warn: '#e9a23b', muted: '#aaa', dim: '#888',
}
const field: React.CSSProperties = {
  width: '100%', padding: '6px 8px', fontSize: 12, backgroundColor: dark.border,
  color: '#eee', border: '1px solid #1a1a2e', borderRadius: 4,
}
const btn: React.CSSProperties = {
  padding: '8px 14px', fontSize: 12, fontWeight: 600, borderRadius: 4,
  border: 'none', cursor: 'pointer', backgroundColor: dark.accent, color: '#06231f',
}
const btnGhost: React.CSSProperties = {
  padding: '4px 9px', fontSize: 11, borderRadius: 4, cursor: 'pointer',
  border: `1px solid ${dark.border}`, backgroundColor: 'transparent', color: dark.muted,
}

function Field({ label, tip, children }: { label: string; tip: string; children: React.ReactNode }) {
  return (
    <div style={{ marginBottom: 12 }}>
      <label style={{ display: 'block', fontSize: 11, color: dark.muted, marginBottom: 4 }}>{label}</label>
      {children}
      <div style={{ fontSize: 10, color: '#666', marginTop: 3, lineHeight: 1.4 }}>{tip}</div>
    </div>
  )
}

/** Programs x programs similarity, in dendrogram order, with cluster bands. */
function SimilarityHeatmap({ result }: { result: MetaResult }) {
  const ref = useRef<HTMLCanvasElement | null>(null)
  const n = result.program_labels.length
  const bands = useMemo(
    () => clusterBands(result.clusters, result.order),
    [result.clusters, result.order],
  )

  useEffect(() => {
    const canvas = ref.current
    if (!canvas || !n) return
    const size = Math.min(520, Math.max(240, n * 6))
    const cell = size / n
    const dpr = window.devicePixelRatio || 1
    canvas.width = size * dpr
    canvas.height = size * dpr
    canvas.style.width = `${size}px`
    canvas.style.height = `${size}px`
    const ctx = canvas.getContext('2d')
    if (!ctx) return
    ctx.scale(dpr, dpr)
    ctx.fillStyle = dark.inset
    ctx.fillRect(0, 0, size, size)
    for (let i = 0; i < n; i++) {
      const row = result.similarity[result.order[i]]
      if (!row) continue
      for (let j = 0; j < n; j++) {
        ctx.fillStyle = similarityColor(row[result.order[j]])
        // +1 avoids hairline gaps between cells at fractional sizes
        ctx.fillRect(j * cell, i * cell, cell + 1, cell + 1)
      }
    }
    ctx.strokeStyle = 'rgba(255,255,255,0.55)'
    ctx.lineWidth = 1
    for (const band of bands) {
      if (band.cluster < 0) continue
      const x = band.start * cell
      const w = band.size * cell
      ctx.strokeRect(x + 0.5, x + 0.5, w - 1, w - 1)
    }
  }, [result, n, bands])

  return (
    <div>
      <div style={{ fontSize: 11, color: dark.muted, marginBottom: 8, lineHeight: 1.5 }}>
        Every program compared to every other, ordered by the clustering. Each
        boxed block on the diagonal is one meta-program — a bright, tight block
        is a real consensus; a dim block or one bleeding into its neighbours is not.
      </div>
      <canvas ref={ref} style={{ borderRadius: 4, border: `1px solid ${dark.border}` }} />
      <div style={{ display: 'flex', alignItems: 'center', gap: 8, marginTop: 8, fontSize: 10, color: dark.dim }}>
        <span>0</span>
        <div style={{
          flex: 1, height: 8, borderRadius: 4,
          background: `linear-gradient(to right, ${similarityColor(0)}, ${similarityColor(0.5)}, ${similarityColor(1)})`,
        }} />
        <span>1 &middot; program similarity</span>
      </div>
    </div>
  )
}

export default function MetaProgramsModal() {
  const isOpen = useStore((s) => s.isMetaProgramsModalOpen)
  const setOpen = useStore((s) => s.setMetaProgramsModalOpen)
  const addFolderToCategory = useStore((s) => s.addFolderToCategory)
  const addScanpyAction = useStore((s) => s.addScanpyAction)
  const refreshObsSummaries = useStore((s) => s.refreshObsSummaries)
  const { colorByScore } = useDataActions()

  const [phase, setPhase] = useState<Phase>('config')
  const [tab, setTab] = useState<Tab>('programs')
  const [sampleColumns, setSampleColumns] = useState<SampleColumn[]>([])
  const [sampleColumn, setSampleColumn] = useState('')
  const [kMin, setKMin] = useState('4')
  const [kMax, setKMax] = useState('9')
  const [nMp, setNMp] = useState('10')
  const [geneSubset, setGeneSubset] = useState('')
  const [boolColumns, setBoolColumns] = useState<BoolColumn[]>([])
  const [layers, setLayers] = useState<LayerInfo[]>([])
  const [layer, setLayer] = useState('X')
  const [transform, setTransform] = useState<NmfTransform>('log1p')
  const [transformTouched, setTransformTouched] = useState(false)
  const [resultKey, setResultKey] = useState('MP')

  const [showAdvanced, setShowAdvanced] = useState(false)
  const [minCells, setMinCells] = useState('10')
  const [specificity, setSpecificity] = useState('5')
  const [weightExplained, setWeightExplained] = useState('0.8')
  const [minConfidence, setMinConfidence] = useState('0.5')
  const [metric, setMetric] = useState<'cosine' | 'jaccard'>('cosine')
  const [seed, setSeed] = useState('0')

  const [error, setError] = useState<string | null>(null)
  const [notice, setNotice] = useState<string | null>(null)
  const [result, setResult] = useState<MetaResult | null>(null)
  const [reusedPrior, setReusedPrior] = useState(false)
  const [expanded, setExpanded] = useState<string | null>(null)
  const [progress, setProgress] = useState<{ frac: number; message: string; startedAt: number; now: number } | null>(null)

  const close = () => {
    setOpen(false)
    setPhase('config'); setResult(null); setError(null); setNotice(null)
    setProgress(null); setReusedPrior(false); setExpanded(null)
    setTransformTouched(false); setTab('programs')
  }

  useEffect(() => {
    if (!isOpen) return
    let cancelled = false

    fetch(appendDataset(`${API_BASE}/gene_nmf/meta/columns`))
      .then((r) => (r.ok ? r.json() : null))
      .then((d) => {
        if (cancelled || !Array.isArray(d?.columns)) return
        const cols = d.columns as SampleColumn[]
        setSampleColumns(cols)
        setSampleColumn((prev) => {
          if (prev) return prev
          const preferred = cols.find((c) => /sample|section|donor|batch|orig/i.test(c.name))
          return (preferred ?? cols[0])?.name ?? ''
        })
      })
      .catch(() => {})

    fetch(appendDataset(`${API_BASE}/var/boolean_columns`))
      .then((r) => (r.ok ? r.json() : null))
      .then((d) => {
        if (cancelled || !Array.isArray(d)) return
        setBoolColumns(d as BoolColumn[])
        setGeneSubset((prev) => prev
          || (d as BoolColumn[]).find((c) => c.name === 'highly_variable')?.name || '')
      })
      .catch(() => {})

    fetch(appendDataset(`${API_BASE}/scanpy/layers`))
      .then((r) => (r.ok ? r.json() : null))
      .then((d) => {
        if (cancelled) return
        setLayers(Array.isArray(d?.layers)
          ? (d.layers as Array<{ name: string; scale?: { verdict?: string } }>)
              .map((l) => ({ name: l.name, verdict: l.scale?.verdict }))
          : [])
      })
      .catch(() => {})

    fetch(appendDataset(`${API_BASE}/gene_nmf/meta/result`))
      .then((r) => (r.ok ? r.json() : null))
      .then((d) => {
        if (cancelled || !d || !Array.isArray(d.metaprograms) || !d.metaprograms.length) return
        setResult({ ...(d as MetaResult), obsm_key: d.key })
        setReusedPrior(true)
        setPhase('results')
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

  const layerVerdict = useMemo(
    () => layers.find((l) => l.name === layer)?.verdict,
    [layers, layer],
  )
  useEffect(() => {
    if (transformTouched) return
    setTransform(defaultTransformFor(layerVerdict))
  }, [layerVerdict, transformTouched])

  const selectedColumn = sampleColumns.find((c) => c.name === sampleColumn)
  const ks = useMemo(() => {
    const lo = Math.max(2, parseInt(kMin, 10) || 4)
    const hi = Math.max(lo, parseInt(kMax, 10) || lo)
    return Array.from({ length: hi - lo + 1 }, (_, i) => lo + i)
  }, [kMin, kMax])
  const nRuns = (selectedColumn?.n_samples ?? 0) * ks.length

  const run = async (overwrite = false) => {
    setError(null); setNotice(null); setReusedPrior(false)
    setPhase('running')
    setProgress({ frac: 0, message: 'Starting…', startedAt: Date.now(), now: Date.now() })
    try {
      const body: Record<string, unknown> = {
        sample_column: sampleColumn,
        ks,
        n_mp: parseInt(nMp, 10) || 10,
        key: resultKey || 'MP',
        gene_subset: geneSubset || null,
        layer: layer === 'X' ? null : layer,
        transform,
        min_cells: parseInt(minCells, 10) || 10,
        specificity_weight: parseFloat(specificity),
        weight_explained: parseFloat(weightExplained),
        min_confidence: parseFloat(minConfidence),
        metric,
        seed: parseInt(seed, 10) || 0,
        overwrite,
      }
      const resp = await fetch(appendDataset(`${API_BASE}/gene_nmf/meta/run`), {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(body),
      })
      if (!resp.ok) {
        const d = await resp.json().catch(() => ({}))
        throw new Error(d.detail || 'Failed to start')
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
      if (task.status !== 'completed') throw new Error(task.error || `Run ${task.status}`)

      const res = task.result as unknown as MetaResult
      setResult(res)
      setPhase('results')
      await refreshSchema()
      refreshObsSummaries()
      addScanpyAction({
        action: 'gene_nmf_meta',
        params: { ...body },
        result: { n_metaprograms: res.metaprograms.length, obsm_key: res.obsm_key },
        timestamp: new Date().toISOString(),
      })
    } catch (e) {
      setError((e as Error).message)
      setPhase('config')
    }
  }

  const saveAll = () => {
    if (!result) return
    addFolderToCategory(
      'gene_clusters',
      `${result.obsm_key ?? 'MP'} meta-programs`,
      result.metaprograms.map((mp) => ({ name: mp.name, genes: mp.genes })),
    )
    setNotice(`Saved ${result.metaprograms.length} meta-programs to Gene Panel → Gene Clusters.`)
  }

  if (!isOpen) return null
  const nSamples = result?.samples.length ?? 0

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
          backgroundColor: dark.panel, border: `1px solid ${dark.border}`, borderRadius: 8,
          padding: '20px 24px', width: phase === 'results' ? 760 : 580, maxWidth: '94vw',
          maxHeight: '90vh', overflowY: 'auto', color: '#eee',
        }}
      >
        <div style={{ fontSize: 16, fontWeight: 600, marginBottom: 4 }}>NMF meta-programs</div>
        <div style={{ fontSize: 12, color: dark.muted, marginBottom: 16, lineHeight: 1.5 }}>
          Factorizes each sample separately at several ranks, then keeps the programs that
          recur. A program found once can be a batch effect; one that turns up independently
          across samples and ranks is a real, shared piece of biology.
        </div>

        {error && (
          <div style={{
            backgroundColor: 'rgba(233,69,96,0.12)', border: `1px solid ${dark.alert}`,
            borderRadius: 4, padding: '8px 10px', fontSize: 12, color: '#ffb3c0', marginBottom: 12,
          }}>
            {error}
            {/already exists/.test(error) && (
              <button style={{ ...btnGhost, marginLeft: 10 }} onClick={() => run(true)}>
                Overwrite it
              </button>
            )}
          </div>
        )}

        {phase === 'running' && (() => {
          const frac = progress?.frac ?? 0
          const pct = Math.round(frac * 100)
          const elapsed = progress ? (progress.now - progress.startedAt) / 1000 : 0
          const eta = frac > 0.02 ? (elapsed * (1 - frac)) / frac : null
          return (
            <div style={{ padding: '20px 0' }}>
              <div style={{ fontSize: 13, color: '#9be7d8', marginBottom: 10 }}>
                {progress?.message ?? 'Factorizing…'}
              </div>
              <div style={{ height: 10, backgroundColor: dark.inset, borderRadius: 5, overflow: 'hidden', border: `1px solid ${dark.border}` }}>
                <div style={{ width: `${pct}%`, height: '100%', backgroundColor: dark.accent, transition: 'width 0.3s ease' }} />
              </div>
              <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: 11, color: dark.dim, marginTop: 6 }}>
                <span>{pct}%</span>
                <span>{elapsed.toFixed(0)}s elapsed{eta != null && ` · ~${eta.toFixed(0)}s left`}</span>
              </div>
            </div>
          )
        })()}

        {phase === 'config' && (
          <>
            <div style={{ display: 'flex', gap: 16 }}>
              <div style={{ flex: 1 }}>
                <Field label="Sample column" tip={TIPS.sample}>
                  <select value={sampleColumn} onChange={(e) => setSampleColumn(e.target.value)} style={field}>
                    {sampleColumns.length === 0 && <option value="">— none found —</option>}
                    {sampleColumns.map((c) => (
                      <option key={c.name} value={c.name}>
                        {c.name} ({c.n_samples} samples)
                      </option>
                    ))}
                  </select>
                </Field>
                <Field label="Ranks to sweep (k)" tip={TIPS.ks}>
                  <div style={{ display: 'flex', gap: 8, alignItems: 'center' }}>
                    <input type="number" min={2} value={kMin} onChange={(e) => setKMin(e.target.value)} style={field} />
                    <span style={{ color: dark.dim, fontSize: 11 }}>to</span>
                    <input type="number" min={2} value={kMax} onChange={(e) => setKMax(e.target.value)} style={field} />
                  </div>
                </Field>
                <Field label="Meta-programs (nMP)" tip={TIPS.nMp}>
                  <input type="number" min={1} value={nMp} onChange={(e) => setNMp(e.target.value)} style={field} />
                </Field>
              </div>
              <div style={{ flex: 1 }}>
                <Field label="Genes" tip={TIPS.genes}>
                  <select value={geneSubset} onChange={(e) => setGeneSubset(e.target.value)} style={field}>
                    <option value="">All genes</option>
                    {boolColumns.map((c) => (
                      <option key={c.name} value={c.name}>{c.name} ({c.n_true.toLocaleString()})</option>
                    ))}
                  </select>
                </Field>
                <Field label="Source matrix" tip={TIPS.layer}>
                  <select value={layer} onChange={(e) => setLayer(e.target.value)} style={{ ...field, marginBottom: 6 }}>
                    {(layers.length ? layers : [{ name: 'X' }]).map((l) => (
                      <option key={l.name} value={l.name}>{l.name}</option>
                    ))}
                  </select>
                  <select
                    value={transform}
                    onChange={(e) => { setTransformTouched(true); setTransform(e.target.value as NmfTransform) }}
                    style={field}
                  >
                    <option value="log1p">normalize + log1p</option>
                    <option value="none">use as-is</option>
                  </select>
                  {transformNoteFor(layerVerdict) && (
                    <div style={{ fontSize: 10, color: dark.warn, marginTop: 4, lineHeight: 1.4 }}>
                      {transformNoteFor(layerVerdict)}
                    </div>
                  )}
                </Field>
                <Field label="Result name" tip={TIPS.key}>
                  <input value={resultKey} onChange={(e) => setResultKey(e.target.value)} style={field} />
                </Field>
              </div>
            </div>

            {nRuns > 0 && (
              <div style={{ fontSize: 11, color: dark.muted, marginBottom: 10 }}>
                {selectedColumn?.n_samples} samples × {ks.length} ranks ={' '}
                <strong style={{ color: '#eee' }}>{nRuns} factorizations</strong>
                {selectedColumn && selectedColumn.min_cells < 50 && (
                  <span style={{ color: dark.warn }}>
                    {' '}· smallest sample has {selectedColumn.min_cells} cells
                  </span>
                )}
              </div>
            )}

            <button style={{ ...btnGhost, marginBottom: showAdvanced ? 10 : 0 }} onClick={() => setShowAdvanced((v) => !v)}>
              {showAdvanced ? '▾' : '▸'} Advanced
            </button>

            {showAdvanced && (
              <div style={{
                display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '0 16px',
                backgroundColor: dark.inset, border: `1px solid ${dark.border}`,
                borderRadius: 4, padding: '12px 14px 2px', marginBottom: 12,
              }}>
                <Field label="Specificity weight" tip={TIPS.specificity}>
                  <input type="number" step="1" min={0} value={specificity} onChange={(e) => setSpecificity(e.target.value)} style={field} />
                </Field>
                <Field label="Weight explained" tip={TIPS.weightExplained}>
                  <input type="number" step="0.05" min={0.05} max={1} value={weightExplained} onChange={(e) => setWeightExplained(e.target.value)} style={field} />
                </Field>
                <Field label="Min gene confidence" tip={TIPS.minConfidence}>
                  <input type="number" step="0.05" min={0} max={1} value={minConfidence} onChange={(e) => setMinConfidence(e.target.value)} style={field} />
                </Field>
                <Field label="Similarity metric" tip={TIPS.metric}>
                  <select value={metric} onChange={(e) => setMetric(e.target.value as 'cosine' | 'jaccard')} style={field}>
                    <option value="cosine">cosine (weights)</option>
                    <option value="jaccard">jaccard (gene sets)</option>
                  </select>
                </Field>
                <Field label="Min cells per sample" tip={TIPS.minCells}>
                  <input type="number" min={2} value={minCells} onChange={(e) => setMinCells(e.target.value)} style={field} />
                </Field>
                <Field label="Random seed" tip="Fixes the initialization so a run is reproducible.">
                  <input type="number" value={seed} onChange={(e) => setSeed(e.target.value)} style={field} />
                </Field>
              </div>
            )}

            <div style={{ display: 'flex', justifyContent: 'flex-end', gap: 8, marginTop: 12 }}>
              <button style={btnGhost} onClick={close}>Cancel</button>
              <button
                style={{ ...btn, ...(sampleColumn ? {} : { opacity: 0.5, cursor: 'not-allowed' }) }}
                disabled={!sampleColumn}
                onClick={() => run(false)}
              >
                Find meta-programs
              </button>
            </div>
          </>
        )}

        {phase === 'results' && result && (
          <>
            {reusedPrior && (
              <div style={{ fontSize: 11, color: dark.warn, marginBottom: 10 }}>
                Showing the stored <code>{result.obsm_key}</code> run. Change settings and re-run for a new one.
              </div>
            )}
            <div style={{
              display: 'flex', gap: 18, flexWrap: 'wrap', fontSize: 11, color: dark.muted,
              backgroundColor: dark.inset, border: `1px solid ${dark.border}`,
              borderRadius: 4, padding: '8px 12px', marginBottom: 12,
            }}>
              <span><strong style={{ color: '#eee' }}>{result.metaprograms.length}</strong> meta-programs</span>
              <span>from {result.n_programs} programs</span>
              <span>{nSamples} samples × k={result.ks_used.join(',')}</span>
              {result.n_dropped > 0 && (
                <span style={{ color: dark.warn }}>{result.n_dropped} empty, dropped</span>
              )}
              {result.skipped.length > 0 && (
                <span style={{ color: dark.warn }}>
                  skipped {result.skipped.map((s) => `${s.sample} (${s.n_cells} cells)`).join(', ')}
                </span>
              )}
            </div>

            <div style={{ display: 'flex', gap: 6, marginBottom: 12 }}>
              {(['programs', 'similarity', 'composition'] as Tab[]).map((t) => (
                <button
                  key={t}
                  onClick={() => setTab(t)}
                  style={{
                    ...btnGhost,
                    // Restate the whole shorthand: mixing `border` with a
                    // `borderColor` longhand makes React drop one on rerender.
                    ...(tab === t
                      ? { color: dark.accent, border: `1px solid ${dark.accent}` }
                      : {}),
                    textTransform: 'capitalize',
                  }}
                >
                  {t}
                </button>
              ))}
            </div>

            {notice && <div style={{ fontSize: 11, color: dark.accent, marginBottom: 10 }}>{notice}</div>}

            {tab === 'programs' && (
              <div style={{ maxHeight: '46vh', overflowY: 'auto', marginBottom: 12 }}>
                {result.metaprograms.map((mp) => {
                  const isOpenRow = expanded === mp.name
                  const weak = mp.silhouette < 0.05
                  return (
                    <div key={mp.name} style={{
                      border: `1px solid ${weak ? dark.warn : dark.border}`, borderRadius: 4,
                      padding: '8px 10px', marginBottom: 6, backgroundColor: dark.inset,
                    }}>
                      <div style={{ display: 'flex', alignItems: 'center', gap: 10 }}>
                        <div style={{ fontSize: 12, fontWeight: 600, minWidth: 46 }}>{mp.name}</div>
                        <div style={{ flex: 1, height: 6, backgroundColor: '#0a0f1c', borderRadius: 3, overflow: 'hidden' }}>
                          <div style={{ width: `${mp.sample_coverage * 100}%`, height: '100%', backgroundColor: dark.accent }} />
                        </div>
                        <div style={{ fontSize: 10, color: dark.dim, minWidth: 78, textAlign: 'right' }}>
                          {coverageLabel(mp.sample_coverage, nSamples)}
                        </div>
                        <div
                          style={{ fontSize: 10, color: weak ? dark.warn : dark.dim, minWidth: 52, textAlign: 'right' }}
                          title="Average silhouette width — how cleanly this meta-program separates from the others"
                        >
                          sil {mp.silhouette.toFixed(2)}
                        </div>
                        <div style={{ fontSize: 10, color: dark.dim, minWidth: 74, textAlign: 'right' }}>
                          {mp.n_programs}p · {mp.n_genes}g
                        </div>
                        <button style={btnGhost} onClick={() => colorByScore(result.obsm_key ?? 'MP', mp.name)}>Color</button>
                        <button style={btnGhost} onClick={() => setExpanded(isOpenRow ? null : mp.name)}>
                          {isOpenRow ? 'Hide' : 'Genes'}
                        </button>
                      </div>
                      <div style={{ fontSize: 11, color: dark.muted, marginTop: 5, lineHeight: 1.5 }}>
                        {(isOpenRow ? mp.genes : mp.genes.slice(0, 12)).join(', ')}
                        {!isOpenRow && mp.genes.length > 12 && ` … +${mp.genes.length - 12}`}
                      </div>
                      {weak && (
                        <div style={{ fontSize: 10, color: dark.warn, marginTop: 4 }}>
                          Low silhouette — these programs do not group cleanly; treat this
                          meta-program as a leftover cluster rather than a consensus.
                        </div>
                      )}
                    </div>
                  )
                })}
              </div>
            )}

            {tab === 'similarity' && <SimilarityHeatmap result={result} />}

            {tab === 'composition' && (
              <div style={{ maxHeight: '46vh', overflowY: 'auto', marginBottom: 12 }}>
                <div style={{ fontSize: 11, color: dark.muted, marginBottom: 8, lineHeight: 1.5 }}>
                  How many programs each sample contributed to each meta-program. A row
                  concentrated in one column is sample-specific, not a consensus.
                </div>
                <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: 11 }}>
                  <thead>
                    <tr>
                      <th style={{ textAlign: 'left', padding: '4px 6px', color: dark.muted }}>Meta-program</th>
                      {result.samples.map((s) => (
                        <th key={s} style={{ textAlign: 'right', padding: '4px 6px', color: dark.muted }}>{s}</th>
                      ))}
                    </tr>
                  </thead>
                  <tbody>
                    {result.metaprograms.map((mp) => (
                      <tr key={mp.name} style={{ borderTop: `1px solid ${dark.border}` }}>
                        <td style={{ padding: '4px 6px' }}>{mp.name}</td>
                        {result.samples.map((s) => {
                          const v = result.composition[mp.name]?.[s] ?? 0
                          return (
                            <td key={s} style={{
                              textAlign: 'right', padding: '4px 6px',
                              color: v === 0 ? '#555' : '#eee',
                            }}>{v}</td>
                          )
                        })}
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            )}

            <div style={{ display: 'flex', justifyContent: 'space-between', gap: 8 }}>
              <button style={btnGhost} onClick={() => { setPhase('config'); setResult(null); setNotice(null) }}>
                ← New run
              </button>
              <div style={{ display: 'flex', gap: 8 }}>
                <button style={btnGhost} onClick={saveAll}>Save all as gene sets</button>
                <button style={btn} onClick={close}>Done</button>
              </div>
            </div>
          </>
        )}
      </div>
    </div>
  )
}
