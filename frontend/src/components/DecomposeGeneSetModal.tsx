/**
 * DecomposeGeneSetModal — take a gene set apart into expression programs.
 *
 * A set whose members do not co-vary (the collagens) is several expression
 * configurations under one name. This modal first measures that — how much
 * of the set's variance one pattern explains — then runs PCA or NMF on the
 * cells × set-genes submatrix. Per-cell program scores land in a score
 * matrix (`.obsm[key]`, colourable like any set score and viewable as an
 * embedding), per-gene loadings in `.varm`, and each program can be saved as
 * a gene set (PCA programs are directional: up and down lists).
 *
 * Opened from a gene-set row's ⋯ menu; the store holds the source set
 * (`decomposeSource`; null = closed).
 *
 * Rollback: delete this file, its mount in App.tsx, the "Decompose into
 * programs…" item in GenePanel.tsx, `decomposeSource` in store.ts, and
 * lib/geneSetDecomposition.ts.
 */
import React, { useEffect, useMemo, useState } from 'react'
import { useStore, cfgDefault } from '../store'
import { appendDataset, pollTask, refreshSchema, useDataActions, useObsSummaries } from '../hooks/useData'
import { coherenceVerdict, programWeight, programsToGeneSets, type Coherence, type Program } from '../lib/geneSetDecomposition'

const API_BASE = '/api'

type CellContext = 'all' | 'selection' | 'annotation'
type Method = 'pca' | 'nmf'

interface Result {
  key: string
  obsm_key: string
  method: Method
  k: number
  program_names: string[]
  programs: Program[]
  variance_ratio: number[] | null
  factor_weights: number[] | null
  n_dropped: number
  n_genes_used: number
  genes_missing: string[]
  n_missing: number
  n_cells: number
  coherence: Coherence
}

const dark = {
  panel: '#16213e', border: '#0f3460', inset: '#0f1625', accent: '#4ecdc4',
  alert: '#e94560', warn: '#e9a23b', muted: '#aaa', dim: '#888', text: '#ddd',
}
const field: React.CSSProperties = {
  background: dark.inset, color: dark.text, border: `1px solid ${dark.border}`,
  borderRadius: 4, padding: '5px 8px', fontSize: 12, width: '100%', boxSizing: 'border-box',
}
const btn: React.CSSProperties = {
  background: dark.accent, color: '#0f1625', border: 'none', borderRadius: 4,
  padding: '6px 14px', fontSize: 12, fontWeight: 600, cursor: 'pointer',
}
const btnGhost: React.CSSProperties = {
  background: 'transparent', color: dark.muted, border: `1px solid ${dark.border}`,
  borderRadius: 4, padding: '5px 12px', fontSize: 12, cursor: 'pointer',
}

function Field({ label, tip, children }: { label: string; tip?: string; children: React.ReactNode }) {
  return (
    <div style={{ marginBottom: 10 }} title={tip}>
      <div style={{ fontSize: 11, color: dark.dim, marginBottom: 3 }}>{label}</div>
      {children}
    </div>
  )
}

function sanitizeKey(name: string): string {
  return name.replace(/[^A-Za-z0-9_]+/g, '_').replace(/^_+|_+$/g, '').slice(0, 40)
}

export default function DecomposeGeneSetModal() {
  const source = useStore((s) => s.decomposeSource)
  const setSource = useStore((s) => s.setDecomposeSource)
  const selectedCellIndices = useStore((s) => s.selectedCellIndices)
  const activeCellMask = useStore((s) => s.activeCellMask)
  const addScanpyAction = useStore((s) => s.addScanpyAction)
  const refreshObsSummaries = useStore((s) => s.refreshObsSummaries)
  const addFolderToCategory = useStore((s) => s.addFolderToCategory)
  const setEmbeddingDims = useStore((s) => s.setEmbeddingDims)
  const setEmbedding = useStore((s) => s.setEmbedding)
  const setSelectedEmbedding = useStore((s) => s.setSelectedEmbedding)
  const { colorByScore } = useDataActions()
  const { summaries } = useObsSummaries()

  const [key, setKey] = useState('')
  const [method, setMethod] = useState<Method>(() => cfgDefault(['gene_set_decomposition', 'method'], 'pca' as Method))
  const [k, setK] = useState<number>(() => cfgDefault(['gene_set_decomposition', 'k'], 3))
  const [kTouched, setKTouched] = useState(false)
  const [threshold, setThreshold] = useState<number>(() => cfgDefault(['gene_set_decomposition', 'loading_threshold'], 0.2))
  const [layer, setLayer] = useState<string>('X')
  const [availableLayers, setAvailableLayers] = useState<{ name: string; density: number }[]>([])
  const [cellContext, setCellContext] = useState<CellContext>('all')
  const [annotationColumn, setAnnotationColumn] = useState('')
  const [annotationValues, setAnnotationValues] = useState<Set<string>>(new Set())
  const [coherence, setCoherence] = useState<Coherence | null>(null)
  const [coherenceError, setCoherenceError] = useState<string | null>(null)
  const [phase, setPhase] = useState<'config' | 'running' | 'results'>('config')
  const [progress, setProgress] = useState<{ frac: number; message: string }>({ frac: 0, message: '' })
  const [error, setError] = useState<string | null>(null)
  const [result, setResult] = useState<Result | null>(null)
  const [expanded, setExpanded] = useState<string | null>(null)
  const [saved, setSaved] = useState(false)

  const categoricalColumns = useMemo(() => summaries.filter((s) => s.dtype === 'category').map((s) => s.name), [summaries])
  const annotationSummary = useMemo(() => summaries.find((s) => s.name === annotationColumn), [summaries, annotationColumn])

  // Same mask × context intersection as the other gene-set analyses.
  const resolveScope = (): { ctx: CellContext; indices: number[] } => {
    let ctx: CellContext = cellContext
    let indices = cellContext === 'selection' ? selectedCellIndices : []
    if (activeCellMask) {
      if (cellContext === 'selection') indices = selectedCellIndices.filter((i) => activeCellMask[i])
      else if (cellContext === 'all') {
        ctx = 'selection'
        indices = []
        for (let i = 0; i < activeCellMask.length; i++) if (activeCellMask[i]) indices.push(i)
      }
    }
    return { ctx, indices }
  }

  const scopeBody = () => {
    const { ctx, indices } = resolveScope()
    return {
      layer: layer && layer !== 'X' ? layer : null,
      cell_context: ctx,
      ...(ctx === 'selection' ? { cell_indices: indices } : {}),
      ...(ctx === 'annotation' ? { annotation_column: annotationColumn, annotation_values: Array.from(annotationValues) } : {}),
    }
  }

  useEffect(() => {
    if (!source) return
    setKey(sanitizeKey(source.name))
    setPhase('config'); setError(null); setResult(null); setExpanded(null); setSaved(false)
    setCoherence(null); setCoherenceError(null); setKTouched(false)
    setCellContext(selectedCellIndices.length > 0 ? 'selection' : 'all')
    setAnnotationValues(new Set())
    setLayer('X')
    fetch(appendDataset(`${API_BASE}/scanpy/layers`))
      .then((r) => (r.ok ? r.json() : Promise.reject(new Error(`HTTP ${r.status}`))))
      .then((d) => setAvailableLayers(d.layers || [{ name: 'X', density: 0 }]))
      .catch(() => setAvailableLayers([{ name: 'X', density: 0 }]))
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [source])

  useEffect(() => {
    if (!annotationColumn && categoricalColumns.length > 0) setAnnotationColumn(categoricalColumns[0])
  }, [categoricalColumns, annotationColumn])

  // Coherence: cheap, synchronous, re-run whenever the cell scope changes.
  useEffect(() => {
    if (!source || phase !== 'config') return
    if (cellContext === 'annotation' && annotationValues.size === 0) return
    let cancelled = false
    setCoherence(null); setCoherenceError(null)
    fetch(appendDataset(`${API_BASE}/gene_sets/coherence`), {
      method: 'POST', headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ genes: source.genes, ...scopeBody() }),
    })
      .then(async (r) => { if (!r.ok) throw new Error((await r.json().catch(() => ({}))).detail || `HTTP ${r.status}`); return r.json() })
      .then((c: Coherence) => {
        if (cancelled) return
        setCoherence(c)
        if (!kTouched) setK(Math.max(1, c.suggested_k))
      })
      .catch((e) => { if (!cancelled) setCoherenceError(e instanceof Error ? e.message : String(e)) })
    return () => { cancelled = true }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [source, phase, cellContext, annotationColumn, annotationValues, activeCellMask, layer])

  useEffect(() => {
    if (!source) return
    const onKey = (e: KeyboardEvent) => { if (e.key === 'Escape' && phase !== 'running') setSource(null) }
    window.addEventListener('keydown', onKey)
    return () => window.removeEventListener('keydown', onKey)
  }, [source, phase, setSource])

  if (!source) return null

  const nGenes = source.genes.length
  const kMax = Math.max(1, method === 'pca' ? nGenes - 1 : nGenes)
  const selectionDisabled = selectedCellIndices.length === 0
  const annotationDisabled = categoricalColumns.length === 0

  const run = async (overwrite = false) => {
    setPhase('running'); setError(null); setProgress({ frac: 0, message: 'Starting…' })
    try {
      const body = {
        genes: source.genes, key: key.trim(), method, k: Math.min(k, kMax),
        loading_threshold: threshold, overwrite, ...scopeBody(),
      }
      const resp = await fetch(appendDataset(`${API_BASE}/gene_sets/decompose`), {
        method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(body),
      })
      if (!resp.ok) throw new Error((await resp.json().catch(() => ({}))).detail || `HTTP ${resp.status}`)
      const { task_id } = await resp.json()
      const task = await pollTask(task_id, undefined, (s) => setProgress({ frac: s.progress ?? 0, message: s.message ?? '' }))
      if (task.status !== 'completed') throw new Error(task.error || `Run ${task.status}`)
      const res = task.result as unknown as Result
      await refreshSchema()
      refreshObsSummaries()
      addScanpyAction({ action: 'gene_set_decomposition', params: body as Record<string, unknown>, result: res as unknown as Record<string, unknown>, timestamp: new Date().toISOString() })
      setResult(res)
      setSaved(false)
      setPhase('results')
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e))
      setPhase('config')
    }
  }

  const saveAll = () => {
    if (!result) return
    const sets = programsToGeneSets(source.name, result.method, result.programs)
    if (sets.length === 0) return
    addFolderToCategory('gene_clusters', `${source.name} programs (${result.method.toUpperCase()}, k=${result.k})`, sets)
    setSaved(true)
  }

  const viewAsEmbedding = () => {
    if (!result || result.program_names.length < 2) return
    setEmbeddingDims(result.obsm_key, 0, 1)
    setEmbedding(null)
    setSelectedEmbedding(result.obsm_key)
    setSource(null)
  }

  const verdict = coherence ? coherenceVerdict(coherence) : null
  const overlayStyle: React.CSSProperties = { position: 'fixed', inset: 0, background: 'rgba(0,0,0,0.6)', zIndex: 100, display: 'flex', alignItems: 'center', justifyContent: 'center' }
  const cardStyle: React.CSSProperties = { background: dark.panel, border: `1px solid ${dark.border}`, borderRadius: 8, width: 520, maxWidth: '94vw', maxHeight: '88vh', overflowY: 'auto', padding: 16, color: dark.text, fontSize: 12 }

  return (
    <div style={overlayStyle} onClick={() => phase !== 'running' && setSource(null)}>
      <div style={cardStyle} onClick={(e) => e.stopPropagation()}>
        <div style={{ fontSize: 14, fontWeight: 600, color: dark.accent, marginBottom: 4 }}>Decompose a gene set</div>
        <div style={{ color: dark.muted, marginBottom: 12 }}>
          <span style={{ color: dark.text }}>{source.name}</span> ({nGenes} gene{nGenes === 1 ? '' : 's'}). Genes that share a name need not share an expression pattern; this finds the patterns.
        </div>

        {error && (
          <div style={{ padding: '6px 10px', background: '#3a1f2b', border: `1px solid ${dark.alert}`, borderRadius: 4, color: '#f5b8c4', marginBottom: 10 }}>
            {error}
            {/already exists/.test(error) && (
              <div style={{ marginTop: 6 }}>
                <button onClick={() => run(true)} style={{ ...btnGhost, color: dark.warn, borderColor: dark.warn }}>Overwrite it</button>
              </div>
            )}
          </div>
        )}

        {phase === 'config' && (
          <>
            <div style={{ background: dark.inset, border: `1px solid ${dark.border}`, borderRadius: 4, padding: '8px 10px', marginBottom: 12 }}>
              {coherenceError && <span style={{ color: dark.alert }}>{coherenceError}</span>}
              {!coherence && !coherenceError && <span style={{ color: dark.dim }}>Measuring how many patterns this set holds…</span>}
              {coherence && verdict && (
                <>
                  <div style={{ color: verdict.kind === 'one' ? dark.accent : dark.warn, marginBottom: 6 }}>{verdict.text}</div>
                  {(() => {
                    const pves = (coherence.eigen_pve ?? []).slice(0, 10)
                    const scale = 30 / Math.max(...pves, coherence.noise_edge_pve ?? 0, 1e-6)
                    const edge = coherence.noise_edge_pve
                    return (
                      <div style={{ display: 'flex', alignItems: 'flex-end', gap: 3, height: 40, position: 'relative' }} title="Share of the set's variance carried by each successive pattern; teal bars stand above the noise floor (Marchenko–Pastur edge, dashed)">
                        {pves.map((v, i) => (
                          <div key={i} style={{ width: 14, display: 'flex', flexDirection: 'column', alignItems: 'center', gap: 2 }}>
                            <div style={{ width: '100%', height: Math.max(2, Math.round(v * scale)), background: edge != null ? (v > edge ? dark.accent : dark.border) : (i < Math.max(1, coherence.suggested_k) ? dark.accent : dark.border) }} />
                            <span style={{ fontSize: 9, color: dark.dim }}>{i + 1}</span>
                          </div>
                        ))}
                        {edge != null && (
                          <div style={{ position: 'absolute', left: 0, width: pves.length * 17, bottom: 13 + Math.round(edge * scale), borderTop: `1px dashed ${dark.warn}`, pointerEvents: 'none' }} />
                        )}
                        <span style={{ fontSize: 10, color: dark.dim, marginLeft: 6, alignSelf: 'center' }}>
                          {coherence.n_genes} genes{coherence.mean_abs_corr != null ? `, mean |r| ${coherence.mean_abs_corr.toFixed(3)}` : ''}{edge != null ? `, noise floor ${(edge * 100).toFixed(1)}%` : ''}
                        </span>
                      </div>
                    )
                  })()}
                </>
              )}
            </div>

            <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr 1fr', gap: 10 }}>
              <Field label="Method" tip="PCA: signed programs (up and down gene lists), fast, exact. NMF: additive non-negative programs, like the NMF Programs tool.">
                <select value={method} onChange={(e) => setMethod(e.target.value as Method)} style={field}>
                  <option value="pca">PCA</option>
                  <option value="nmf">NMF</option>
                </select>
              </Field>
              <Field label={`Programs (k, ≤ ${kMax})`} tip="Defaults to the number of patterns the eigen-spectrum suggests">
                <input type="number" min={1} max={kMax} value={k} onChange={(e) => { setKTouched(true); setK(Math.max(1, Math.min(kMax, Number(e.target.value) || 1))) }} style={field} />
              </Field>
              {method === 'pca' ? (
                <Field label="Loading cut" tip="A gene joins a program when its loading is at least this fraction of the program's largest |loading|">
                  <input type="number" min={0} max={1} step={0.05} value={threshold} onChange={(e) => setThreshold(Math.max(0, Math.min(1, Number(e.target.value) || 0)))} style={field} />
                </Field>
              ) : <div />}
            </div>
            <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 10 }}>
              <Field label="Key (the score matrix name)">
                <input value={key} onChange={(e) => setKey(sanitizeKey(e.target.value))} style={field} />
              </Field>
              <Field label="Source matrix" tip="Pick a smoothed layer (Preprocess → Smooth) to decompose denoised expression; sparse counts make every correlation small">
                <select value={layer} onChange={(e) => setLayer(e.target.value)} style={field}>
                  {availableLayers.length === 0 && <option value="X">.X (log-normalised)</option>}
                  {availableLayers.map((L) => (
                    <option key={L.name} value={L.name}>{L.name === 'X' ? '.X (log-normalised)' : L.name}{L.density > 0 ? ` — ${(L.density * 100).toFixed(1)}% dense` : ''}</option>
                  ))}
                </select>
              </Field>
            </div>

            <Field label="Cells">
              <div style={{ display: 'flex', flexDirection: 'column', gap: 4, color: dark.text }}>
                <label><input type="radio" checked={cellContext === 'all'} onChange={() => setCellContext('all')} /> All cells{activeCellMask ? ' (active set)' : ''}</label>
                <label style={{ color: selectionDisabled ? '#555' : dark.text }} title={selectionDisabled ? 'Lasso-select cells first' : undefined}>
                  <input type="radio" disabled={selectionDisabled} checked={cellContext === 'selection'} onChange={() => setCellContext('selection')} /> Current selection ({selectedCellIndices.length.toLocaleString()} cells)
                </label>
                <label style={{ color: annotationDisabled ? '#555' : dark.text }}>
                  <input type="radio" disabled={annotationDisabled} checked={cellContext === 'annotation'} onChange={() => setCellContext('annotation')} /> Annotation category
                </label>
              </div>
            </Field>
            {cellContext === 'annotation' && (
              <div style={{ paddingLeft: 18, marginBottom: 10 }}>
                <select value={annotationColumn} onChange={(e) => { setAnnotationColumn(e.target.value); setAnnotationValues(new Set()) }} style={{ ...field, marginBottom: 6 }}>
                  {categoricalColumns.map((c) => <option key={c} value={c}>{c}</option>)}
                </select>
                <div style={{ maxHeight: 120, overflowY: 'auto', border: `1px solid ${dark.border}`, borderRadius: 4, padding: 4 }}>
                  {(annotationSummary?.categories ?? []).map((c) => c.value).map((v) => (
                    <label key={v} style={{ display: 'block', color: dark.text }}>
                      <input type="checkbox" checked={annotationValues.has(v)} onChange={(e) => setAnnotationValues((prev) => { const n = new Set(prev); e.target.checked ? n.add(v) : n.delete(v); return n })} /> {v}
                    </label>
                  ))}
                </div>
              </div>
            )}

            <div style={{ display: 'flex', gap: 8, justifyContent: 'flex-end' }}>
              <button onClick={() => setSource(null)} style={btnGhost}>Cancel</button>
              <button
                onClick={() => run(false)}
                disabled={nGenes < 2 || !key.trim() || (cellContext === 'annotation' && annotationValues.size === 0)}
                style={{ ...btn, opacity: nGenes < 2 || !key.trim() ? 0.5 : 1 }}
              >
                Run
              </button>
            </div>
          </>
        )}

        {phase === 'running' && (
          <div style={{ marginBottom: 12 }}>
            <div style={{ height: 8, background: dark.border, borderRadius: 4, overflow: 'hidden' }}>
              <div style={{ width: `${Math.round(progress.frac * 100)}%`, height: '100%', background: dark.accent, transition: 'width 0.3s' }} />
            </div>
            <div style={{ color: dark.dim, marginTop: 4 }}>{progress.message}</div>
          </div>
        )}

        {phase === 'results' && result && (
          <div>
            <div style={{ display: 'flex', gap: 14, flexWrap: 'wrap', color: dark.muted, background: dark.inset, border: `1px solid ${dark.border}`, borderRadius: 4, padding: '8px 10px', marginBottom: 10 }}>
              <span><b style={{ color: dark.text }}>{result.program_names.length}</b> {result.method.toUpperCase()} program{result.program_names.length === 1 ? '' : 's'}</span>
              <span>{result.n_genes_used} genes × {result.n_cells.toLocaleString()} cells</span>
              <span>one pattern = {Math.round(result.coherence.eigengene_pve * 100)}% of variance</span>
              {result.variance_ratio && <span>k explains {Math.round(result.variance_ratio.reduce((a, b) => a + b, 0) * 100)}%</span>}
              {result.n_dropped > 0 && <span style={{ color: dark.warn }}>{result.n_dropped} dropped (one gene carried it)</span>}
              {result.n_missing > 0 && <span style={{ color: dark.warn }}>{result.n_missing} gene{result.n_missing === 1 ? '' : 's'} not in dataset</span>}
            </div>
            <div style={{ color: dark.dim, fontSize: 11, marginBottom: 8 }}>
              Scores in <code>.obsm['{result.obsm_key}']</code> (score pills in the Genes panel); loadings in <code>.varm['{result.obsm_key}_loadings']</code>.
            </div>
            <div style={{ maxHeight: '46vh', overflowY: 'auto', marginBottom: 10 }}>
              {result.programs.map((p) => {
                const open = expanded === p.name
                const w = programWeight(p, result.programs)
                return (
                  <div key={p.name} style={{ border: `1px solid ${dark.border}`, borderRadius: 4, padding: '8px 10px', marginBottom: 6, background: dark.inset }}>
                    <div style={{ display: 'flex', alignItems: 'center', gap: 10 }}>
                      <div style={{ fontWeight: 600, minWidth: 40 }}>{p.name}</div>
                      <div style={{ flex: 1, height: 6, background: '#0a0f1c', borderRadius: 3, overflow: 'hidden' }}>
                        <div style={{ width: `${Math.round(w * 100)}%`, height: '100%', background: dark.accent }} />
                      </div>
                      <div style={{ fontSize: 10, color: dark.dim, minWidth: 90, textAlign: 'right' }}>
                        {p.variance_ratio != null ? `${(p.variance_ratio * 100).toFixed(1)}% var · ` : ''}{p.genes.length}↑{p.genes_down.length > 0 ? ` ${p.genes_down.length}↓` : ''}
                      </div>
                      <button style={btnGhost} onClick={() => colorByScore(result.obsm_key, p.name)} title="Color cells by this program's score">Color</button>
                      <button style={btnGhost} onClick={() => setExpanded(open ? null : p.name)}>{open ? 'Hide' : 'Genes'}</button>
                    </div>
                    <div style={{ fontSize: 11, color: dark.muted, marginTop: 5, lineHeight: 1.5 }}>
                      <span style={{ color: dark.accent }}>↑ </span>{(open ? p.genes : p.genes.slice(0, 10)).join(', ')}{!open && p.genes.length > 10 ? ` … +${p.genes.length - 10}` : ''}
                      {p.genes_down.length > 0 && (
                        <div><span style={{ color: '#ff9e64' }}>↓ </span>{(open ? p.genes_down : p.genes_down.slice(0, 10)).join(', ')}{!open && p.genes_down.length > 10 ? ` … +${p.genes_down.length - 10}` : ''}</div>
                      )}
                    </div>
                  </div>
                )
              })}
            </div>
            <div style={{ display: 'flex', gap: 8, justifyContent: 'flex-end', flexWrap: 'wrap' }}>
              <button onClick={saveAll} disabled={saved} style={{ ...btn, opacity: saved ? 0.5 : 1 }} title="One gene set per program, under Gene Clusters (PCA programs keep their down lists)">
                {saved ? 'Saved' : 'Save programs as gene sets'}
              </button>
              {result.program_names.length >= 2 && (
                <button onClick={viewAsEmbedding} style={btnGhost} title="Plot cells by their first two program scores">View {result.program_names[0]} × {result.program_names[1]}</button>
              )}
              <button onClick={() => setPhase('config')} style={btnGhost}>Adjust &amp; re-run</button>
              <button onClick={() => setSource(null)} style={btnGhost}>Done</button>
            </div>
          </div>
        )}
      </div>
    </div>
  )
}
