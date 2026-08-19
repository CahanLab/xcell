/**
 * NMF gene programs (Analyze → Genes → NMF Programs).
 *
 * Factorizes expression into a handful of non-negative "gene programs" — a
 * Python port of GeneNMF's single-sample path (carmonalab/GeneNMF). Each
 * program is a ranked gene set plus a per-cell usage score, so a program can
 * be colored onto the plot like any score column or saved into the Gene Panel
 * for downstream scoring.
 *
 * Unlike Cluster Genes (which partitions genes — one gene, one module), NMF is
 * additive: a gene can carry weight in several programs, and each cell mixes
 * programs rather than belonging to one. That is what makes it the right tool
 * for overlapping states like cycling, hypoxia or interferon response.
 *
 * Backend: /gene_nmf/run (background task), /gene_nmf/result, /gene_nmf/runs.
 * Per-cell usage lands in obsm[key] → score pills; loadings in varm[key_loadings].
 *
 * Rollback: delete this file, remove its mount in App.tsx, the Genes "NMF
 * Programs" launcher in ScanpyModal, and isGeneNmfModalOpen in store.ts.
 */
import { useEffect, useMemo, useState } from 'react'
import { useStore } from '../store'
import { appendDataset, pollTask, refreshSchema, useDataActions, type ObsSummary } from '../hooks/useData'
import { defaultTransformFor, transformNoteFor, type NmfTransform } from '../lib/nmfTransform'

const API_BASE = '/api'

interface Program {
  name: string
  index: number
  genes: string[]
  weights: number[]
  factor_weight?: number
}

interface NmfResult {
  obsm_key: string
  programs: Program[]
  n_programs: number
  n_dropped: number
  n_iter: number
  converged: boolean
  relative_error: number | null
  factor_weights: number[]
  n_genes_used: number
  n_cells_used: number
  key?: string
}

interface BoolColumn { name: string; n_true: number }
interface LayerInfo { name: string; verdict?: string }

type Phase = 'config' | 'running' | 'results'
type CellScope = 'all' | 'selection' | 'annotation'

const TIPS = {
  k: 'How many programs to extract. NMF does not choose this for you: too few merges distinct states, too many splits one state across factors. 5–15 is the usual range; try a couple of values and keep the one whose programs read as coherent biology.',
  genes: 'NMF is normally run on highly variable genes — including every gene lets housekeeping dominate every factor. Pick a boolean .var column (run Highly Variable Genes first if you have none).',
  cells: 'Fit on all cells, the current selection, or one annotation group. Programs are only defined for the cells the model saw; every other cell scores NaN.',
  layer: 'Which matrix to factorize. NMF needs non-negative values — raw counts or a log-normalized layer. Scaled/centered data will be rejected.',
  key: 'Where the results go: .obsm[key] for per-cell usage, .varm[key_loadings] for gene weights. Use a fresh name to keep several runs side by side.',
  specificity: 'Down-weights genes that load on many programs before picking each program\'s gene set, so shared/housekeeping genes stop crowding out the genes that actually distinguish a program. It is an exponent, so it bites fast: 1 sharpens programs, 3+ shrinks them to a handful of genes, and past that programs collapse to one gene and get dropped. 0 disables it.',
  weightExplained: 'A program\'s gene set is its top genes up to this fraction of the program\'s total weight. Higher = longer, more inclusive signatures.',
  maxGenes: 'Hard cap on genes per program, applied after the weight cutoff.',
  l1: 'Sparsity pressure, as a fraction of each factor\'s own scale. Above 0 it zeroes out weak gene loadings, giving crisper but narrower programs. 0 matches GeneNMF\'s default.',
  seed: 'NMF starts from a random initialization; the seed makes a run reproducible. Different seeds can find slightly different programs — a program that survives several seeds is a real one.',
  maxIter: 'Upper bound on solver passes. The run stops earlier once the fit stops improving.',
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

export default function GeneNmfModal() {
  const isOpen = useStore((s) => s.isGeneNmfModalOpen)
  const setOpen = useStore((s) => s.setGeneNmfModalOpen)
  const addFolderToCategory = useStore((s) => s.addFolderToCategory)
  const addScanpyAction = useStore((s) => s.addScanpyAction)
  const refreshObsSummaries = useStore((s) => s.refreshObsSummaries)
  const selectedCellIndices = useStore((s) => s.selectedCellIndices)
  const { colorByScore } = useDataActions()

  const [phase, setPhase] = useState<Phase>('config')
  const [k, setK] = useState('10')
  const [geneSubset, setGeneSubset] = useState('')
  const [boolColumns, setBoolColumns] = useState<BoolColumn[]>([])
  const [layers, setLayers] = useState<LayerInfo[]>([])
  const [layer, setLayer] = useState('X')
  const [transform, setTransform] = useState<NmfTransform>('log1p')
  // Set from the layer's detected scale unless the user overrides it.
  const [transformTouched, setTransformTouched] = useState(false)
  const [resultKey, setResultKey] = useState('NMF')
  const [cellScope, setCellScope] = useState<CellScope>('all')
  const [annColumn, setAnnColumn] = useState('')
  const [annValues, setAnnValues] = useState<string[]>([])
  const [summaries, setSummaries] = useState<ObsSummary[]>([])

  const [showAdvanced, setShowAdvanced] = useState(false)
  const [specificity, setSpecificity] = useState('1')
  const [weightExplained, setWeightExplained] = useState('0.5')
  const [maxGenes, setMaxGenes] = useState('200')
  const [l1w, setL1w] = useState('0')
  const [seed, setSeed] = useState('0')
  const [maxIter, setMaxIter] = useState('500')

  const [error, setError] = useState<string | null>(null)
  const [notice, setNotice] = useState<string | null>(null)
  const [result, setResult] = useState<NmfResult | null>(null)
  const [reusedPrior, setReusedPrior] = useState(false)
  const [expanded, setExpanded] = useState<string | null>(null)
  const [progress, setProgress] = useState<{ frac: number; message: string; startedAt: number; now: number } | null>(null)

  const close = () => {
    setOpen(false)
    setPhase('config'); setResult(null); setError(null); setNotice(null)
    setProgress(null); setReusedPrior(false); setExpanded(null)
    setTransformTouched(false)
  }

  // On open: gene subset columns, layers, annotation columns, and any prior run.
  useEffect(() => {
    if (!isOpen) return
    let cancelled = false

    fetch(appendDataset(`${API_BASE}/var/boolean_columns`))
      .then((r) => (r.ok ? r.json() : null))
      .then((d) => {
        if (cancelled || !Array.isArray(d)) return
        setBoolColumns(d as BoolColumn[])
        setGeneSubset((prev) => {
          if (prev) return prev
          return (d as BoolColumn[]).find((c) => c.name === 'highly_variable')?.name ?? ''
        })
      })
      .catch(() => {})

    fetch(appendDataset(`${API_BASE}/scanpy/layers`))
      .then((r) => (r.ok ? r.json() : null))
      .then((d) => {
        if (cancelled) return
        const got = Array.isArray(d?.layers)
          ? (d.layers as Array<{ name: string; scale?: { verdict?: string } }>).map(
              (l) => ({ name: l.name, verdict: l.scale?.verdict }),
            )
          : []
        setLayers(got)
      })
      .catch(() => {})

    fetch(appendDataset(`${API_BASE}/obs/summaries`))
      .then((r) => (r.ok ? r.json() : null))
      .then((d) => {
        if (cancelled || !Array.isArray(d)) return
        setSummaries((d as ObsSummary[]).filter((s) => s.dtype === 'category'))
      })
      .catch(() => {})

    fetch(appendDataset(`${API_BASE}/gene_nmf/result`))
      .then((r) => (r.ok ? r.json() : null))
      .then((d) => {
        if (cancelled || !d || !Array.isArray(d.programs) || d.programs.length === 0) return
        const params = (d.params ?? {}) as Record<string, unknown>
        setResult({
          obsm_key: d.key,
          key: d.key,
          programs: d.programs as Program[],
          n_programs: (d.metrics?.n_programs as number) ?? d.programs.length,
          n_dropped: (d.metrics?.n_dropped as number) ?? 0,
          n_iter: (d.metrics?.n_iter as number) ?? 0,
          converged: Boolean(d.metrics?.converged),
          relative_error: (d.metrics?.relative_error as number) ?? null,
          factor_weights: (d.programs as Program[]).map((p) => p.factor_weight ?? 0),
          n_genes_used: (params.n_genes_used as number) ?? 0,
          n_cells_used: (params.n_cells_used as number) ?? 0,
        })
        if (typeof params.k === 'number') setK(String(params.k))
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

  // xcell's usual log1p default is wrong on a matrix that is already logged,
  // so follow the detected scale until the user says otherwise.
  useEffect(() => {
    if (transformTouched) return
    setTransform(defaultTransformFor(layerVerdict))
  }, [layerVerdict, transformTouched])

  const annCategories = useMemo(
    () => summaries.find((s) => s.name === annColumn)?.categories?.map((c) => c.value) ?? [],
    [summaries, annColumn],
  )

  const run = async (overwrite = false) => {
    setError(null); setNotice(null); setReusedPrior(false)
    setPhase('running')
    setProgress({ frac: 0, message: 'Starting…', startedAt: Date.now(), now: Date.now() })
    try {
      const body: Record<string, unknown> = {
        k: parseInt(k, 10) || 10,
        key: resultKey || 'NMF',
        gene_subset: geneSubset || null,
        layer: layer === 'X' ? null : layer,
        transform,
        specificity_weight: parseFloat(specificity),
        weight_explained: parseFloat(weightExplained),
        max_genes: parseInt(maxGenes, 10) || 200,
        l1_w: parseFloat(l1w) || 0,
        seed: parseInt(seed, 10) || 0,
        max_iter: parseInt(maxIter, 10) || 500,
        overwrite,
        cell_context: cellScope,
      }
      if (cellScope === 'selection') body.cell_indices = selectedCellIndices
      if (cellScope === 'annotation') {
        body.annotation_column = annColumn
        body.annotation_values = annValues
      }

      const resp = await fetch(appendDataset(`${API_BASE}/gene_nmf/run`), {
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

      const res = task.result as unknown as NmfResult
      setResult(res)
      setPhase('results')
      // New .obsm score matrix — the score pills and axis pickers read the schema.
      await refreshSchema()
      refreshObsSummaries()
      addScanpyAction({
        action: 'gene_nmf',
        params: { ...body },
        result: { n_programs: res.n_programs, obsm_key: res.obsm_key },
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
      `${result.obsm_key} programs`,
      result.programs.map((p) => ({ name: p.name, genes: p.genes })),
    )
    setNotice(`Saved ${result.programs.length} programs to Gene Panel → Gene Clusters.`)
  }

  if (!isOpen) return null

  const maxWeight = result ? Math.max(...result.factor_weights, 1e-9) : 1

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
          padding: '20px 24px', width: phase === 'results' ? 720 : 560, maxWidth: '94vw',
          maxHeight: '90vh', overflowY: 'auto', color: '#eee',
        }}
      >
        <div style={{ fontSize: 16, fontWeight: 600, marginBottom: 4 }}>NMF gene programs</div>
        <div style={{ fontSize: 12, color: dark.muted, marginBottom: 16, lineHeight: 1.5 }}>
          Decomposes expression into additive gene programs. Unlike gene clustering, a gene can
          belong to several programs and each cell mixes them — which is what lets overlapping
          states (cycling, hypoxia, interferon) come apart.
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
                <Field label="Number of programs (k)" tip={TIPS.k}>
                  <input type="number" min={2} value={k} onChange={(e) => setK(e.target.value)} style={field} />
                </Field>
                <Field label="Genes" tip={TIPS.genes}>
                  <select value={geneSubset} onChange={(e) => setGeneSubset(e.target.value)} style={field}>
                    <option value="">All genes</option>
                    {boolColumns.map((c) => (
                      <option key={c.name} value={c.name}>{c.name} ({c.n_true.toLocaleString()})</option>
                    ))}
                  </select>
                </Field>
                <Field label="Cells" tip={TIPS.cells}>
                  <select value={cellScope} onChange={(e) => setCellScope(e.target.value as CellScope)} style={field}>
                    <option value="all">All cells</option>
                    <option value="selection">
                      Current selection ({selectedCellIndices?.length ?? 0})
                    </option>
                    <option value="annotation">By annotation…</option>
                  </select>
                </Field>
                {cellScope === 'annotation' && (
                  <>
                    <select
                      value={annColumn}
                      onChange={(e) => { setAnnColumn(e.target.value); setAnnValues([]) }}
                      style={{ ...field, marginBottom: 6 }}
                    >
                      <option value="">— column —</option>
                      {summaries.map((s) => <option key={s.name} value={s.name}>{s.name}</option>)}
                    </select>
                    <select
                      multiple
                      value={annValues}
                      onChange={(e) => setAnnValues(Array.from(e.target.selectedOptions, (o) => o.value))}
                      style={{ ...field, height: 90 }}
                    >
                      {annCategories.map((v) => <option key={v} value={v}>{v}</option>)}
                    </select>
                  </>
                )}
              </div>
              <div style={{ flex: 1 }}>
                <Field label="Source matrix" tip={TIPS.layer}>
                  <select
                    value={layer}
                    onChange={(e) => setLayer(e.target.value)}
                    style={{ ...field, marginBottom: 6 }}
                  >
                    {(layers.length ? layers : [{ name: 'X' }]).map((l) => (
                      <option key={l.name} value={l.name}>{l.name}</option>
                    ))}
                  </select>
                  <select
                    value={transform}
                    onChange={(e) => {
                      setTransformTouched(true)
                      setTransform(e.target.value as NmfTransform)
                    }}
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

            <button
              style={{ ...btnGhost, marginBottom: showAdvanced ? 10 : 0 }}
              onClick={() => setShowAdvanced((v) => !v)}
            >
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
                <Field label="Max genes per program" tip={TIPS.maxGenes}>
                  <input type="number" min={1} value={maxGenes} onChange={(e) => setMaxGenes(e.target.value)} style={field} />
                </Field>
                <Field label="L1 sparsity (gene loadings)" tip={TIPS.l1}>
                  <input type="number" step="0.05" min={0} max={0.95} value={l1w} onChange={(e) => setL1w(e.target.value)} style={field} />
                </Field>
                <Field label="Random seed" tip={TIPS.seed}>
                  <input type="number" value={seed} onChange={(e) => setSeed(e.target.value)} style={field} />
                </Field>
                <Field label="Max iterations" tip={TIPS.maxIter}>
                  <input type="number" min={10} value={maxIter} onChange={(e) => setMaxIter(e.target.value)} style={field} />
                </Field>
              </div>
            )}

            <div style={{ display: 'flex', justifyContent: 'flex-end', gap: 8, marginTop: 12 }}>
              <button style={btnGhost} onClick={close}>Cancel</button>
              <button
                style={{
                  ...btn,
                  ...(cellScope === 'annotation' && (!annColumn || annValues.length === 0)
                    ? { opacity: 0.5, cursor: 'not-allowed' } : {}),
                }}
                disabled={cellScope === 'annotation' && (!annColumn || annValues.length === 0)}
                onClick={() => run(false)}
              >
                Find programs
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
              <span><strong style={{ color: '#eee' }}>{result.n_programs}</strong> programs</span>
              <span>{result.n_genes_used.toLocaleString()} genes × {result.n_cells_used.toLocaleString()} cells</span>
              <span>{result.n_iter} iterations{result.converged ? '' : ' (hit the cap)'}</span>
              {result.relative_error != null && (
                <span>residual {(result.relative_error * 100).toFixed(1)}%</span>
              )}
              {result.n_dropped > 0 && (
                <span style={{ color: dark.warn }}>
                  {result.n_dropped} dropped (one gene carried the whole program)
                </span>
              )}
            </div>

            {notice && (
              <div style={{ fontSize: 11, color: dark.accent, marginBottom: 10 }}>{notice}</div>
            )}

            <div style={{ maxHeight: '46vh', overflowY: 'auto', marginBottom: 12 }}>
              {result.programs.map((p, i) => {
                const isOpenRow = expanded === p.name
                const w = result.factor_weights[i] ?? 0
                return (
                  <div
                    key={p.name}
                    style={{
                      border: `1px solid ${dark.border}`, borderRadius: 4,
                      padding: '8px 10px', marginBottom: 6, backgroundColor: dark.inset,
                    }}
                  >
                    <div style={{ display: 'flex', alignItems: 'center', gap: 10 }}>
                      <div style={{ fontSize: 12, fontWeight: 600, minWidth: 62 }}>{p.name}</div>
                      <div style={{ flex: 1, height: 6, backgroundColor: '#0a0f1c', borderRadius: 3, overflow: 'hidden' }}>
                        <div style={{ width: `${(w / maxWeight) * 100}%`, height: '100%', backgroundColor: dark.accent }} />
                      </div>
                      <div style={{ fontSize: 10, color: dark.dim, minWidth: 54, textAlign: 'right' }}>
                        {p.genes.length} genes
                      </div>
                      <button style={btnGhost} onClick={() => colorByScore(result.obsm_key, p.name)}>
                        Color
                      </button>
                      <button style={btnGhost} onClick={() => setExpanded(isOpenRow ? null : p.name)}>
                        {isOpenRow ? 'Hide' : 'Genes'}
                      </button>
                    </div>
                    <div style={{ fontSize: 11, color: dark.muted, marginTop: 5, lineHeight: 1.5 }}>
                      {(isOpenRow ? p.genes : p.genes.slice(0, 12)).join(', ')}
                      {!isOpenRow && p.genes.length > 12 && ` … +${p.genes.length - 12}`}
                    </div>
                  </div>
                )
              })}
            </div>

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
