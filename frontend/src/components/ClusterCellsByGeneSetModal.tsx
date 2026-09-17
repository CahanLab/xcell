/**
 * ClusterCellsByGeneSetModal — cluster cells using only one gene set's genes.
 *
 * PCA → kNN → Leiden (→ UMAP) on the cells × set-genes submatrix, all
 * written under suffixed keys (`X_pca_<key>`, `<key>_connectivities`,
 * `leiden_<key>`, `X_umap_<key>`) so the dataset's own embedding and
 * clustering are untouched. Opened from a gene-set row's ⋯ menu; the store
 * holds the source set (`clusterCellsSource`; null = closed).
 *
 * Rollback: delete this file, its mount in App.tsx, the "Cluster cells on
 * this set…" item in GenePanel.tsx, and `clusterCellsSource` in store.ts.
 */
import React, { useEffect, useMemo, useState } from 'react'
import { useStore, cfgDefault } from '../store'
import { appendDataset, pollTask, refreshSchema, useDataActions, useObsSummaries } from '../hooks/useData'

const API_BASE = '/api'

type CellContext = 'all' | 'selection' | 'annotation'

interface Result {
  key: string
  obs_column: string
  embedding: string | null
  pca_key: string
  graph_key: string
  n_genes_used: number
  genes_missing: string[]
  n_missing: number
  n_cells: number
  n_comps: number
  n_neighbors: number
  resolution: number
  n_clusters: number
  cluster_sizes: Record<string, number>
  variance_ratio: number[]
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

export function sanitizeKey(name: string): string {
  return name.replace(/[^A-Za-z0-9_]+/g, '_').replace(/^_+|_+$/g, '').slice(0, 40)
}

export default function ClusterCellsByGeneSetModal() {
  const source = useStore((s) => s.clusterCellsSource)
  const setSource = useStore((s) => s.setClusterCellsSource)
  const selectedCellIndices = useStore((s) => s.selectedCellIndices)
  const activeCellMask = useStore((s) => s.activeCellMask)
  const addScanpyAction = useStore((s) => s.addScanpyAction)
  const refreshObsSummaries = useStore((s) => s.refreshObsSummaries)
  const { selectColorColumn, selectEmbedding } = useDataActions()
  const { summaries } = useObsSummaries()

  const [key, setKey] = useState('')
  const [nComps, setNComps] = useState<number>(() => cfgDefault(['cluster_cells_by_gene_set', 'n_comps'], 20))
  const [nNeighbors, setNNeighbors] = useState<number>(() => cfgDefault(['cluster_cells_by_gene_set', 'n_neighbors'], 15))
  const [resolution, setResolution] = useState<number>(() => cfgDefault(['cluster_cells_by_gene_set', 'resolution'], 0.3))
  const [runUmap, setRunUmap] = useState<boolean>(() => cfgDefault(['cluster_cells_by_gene_set', 'run_umap'], true))
  const [scale, setScale] = useState(true)
  const [cellContext, setCellContext] = useState<CellContext>('all')
  const [annotationColumn, setAnnotationColumn] = useState('')
  const [annotationValues, setAnnotationValues] = useState<Set<string>>(new Set())
  const [phase, setPhase] = useState<'config' | 'running' | 'results'>('config')
  const [progress, setProgress] = useState<{ frac: number; message: string }>({ frac: 0, message: '' })
  const [error, setError] = useState<string | null>(null)
  const [result, setResult] = useState<Result | null>(null)

  const categoricalColumns = useMemo(() => summaries.filter((s) => s.dtype === 'category').map((s) => s.name), [summaries])
  const annotationSummary = useMemo(() => summaries.find((s) => s.name === annotationColumn), [summaries, annotationColumn])

  useEffect(() => {
    if (!source) return
    setKey(sanitizeKey(source.name))
    setPhase('config'); setError(null); setResult(null)
    setCellContext(selectedCellIndices.length > 0 ? 'selection' : 'all')
    setAnnotationValues(new Set())
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [source])

  useEffect(() => {
    if (!annotationColumn && categoricalColumns.length > 0) setAnnotationColumn(categoricalColumns[0])
  }, [categoricalColumns, annotationColumn])

  useEffect(() => {
    if (!source) return
    const onKey = (e: KeyboardEvent) => { if (e.key === 'Escape' && phase !== 'running') setSource(null) }
    window.addEventListener('keydown', onKey)
    return () => window.removeEventListener('keydown', onKey)
  }, [source, phase, setSource])

  if (!source) return null

  const nGenes = source.genes.length
  const selectionDisabled = selectedCellIndices.length === 0
  const annotationDisabled = categoricalColumns.length === 0

  const run = async (overwrite = false) => {
    setPhase('running'); setError(null); setProgress({ frac: 0, message: 'Starting…' })
    try {
      // Same mask × context intersection as Cluster genes: a masked-out cell
      // never enters the clustering, whichever context is chosen.
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
      const body = {
        genes: source.genes, key: key.trim(), n_comps: nComps, n_neighbors: nNeighbors, resolution,
        run_umap: runUmap, scale, overwrite,
        cell_context: ctx,
        ...(ctx === 'selection' ? { cell_indices: indices } : {}),
        ...(ctx === 'annotation' ? { annotation_column: annotationColumn, annotation_values: Array.from(annotationValues) } : {}),
      }
      const resp = await fetch(appendDataset(`${API_BASE}/gene_sets/cluster_cells`), {
        method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(body),
      })
      if (!resp.ok) throw new Error((await resp.json().catch(() => ({}))).detail || `HTTP ${resp.status}`)
      const { task_id } = await resp.json()
      const task = await pollTask(task_id, undefined, (s) => setProgress({ frac: s.progress ?? 0, message: s.message ?? '' }))
      if (task.status !== 'completed') throw new Error(task.error || `Run ${task.status}`)
      const res = task.result as unknown as Result
      await refreshSchema()
      refreshObsSummaries()
      addScanpyAction({ action: 'cluster_cells_by_gene_set', params: body as Record<string, unknown>, result: res as unknown as Record<string, unknown>, timestamp: new Date().toISOString() })
      setResult(res)
      setPhase('results')
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e))
      setPhase('config')
    }
  }

  const overlayStyle: React.CSSProperties = { position: 'fixed', inset: 0, background: 'rgba(0,0,0,0.6)', zIndex: 100, display: 'flex', alignItems: 'center', justifyContent: 'center' }
  const cardStyle: React.CSSProperties = { background: dark.panel, border: `1px solid ${dark.border}`, borderRadius: 8, width: 440, maxWidth: '94vw', maxHeight: '88vh', overflowY: 'auto', padding: 16, color: dark.text, fontSize: 12 }

  return (
    <div style={overlayStyle} onClick={() => phase !== 'running' && setSource(null)}>
      <div style={cardStyle} onClick={(e) => e.stopPropagation()}>
        <div style={{ fontSize: 14, fontWeight: 600, color: dark.accent, marginBottom: 4 }}>Cluster cells on a gene set</div>
        <div style={{ color: dark.muted, marginBottom: 12 }}>
          <span style={{ color: dark.text }}>{source.name}</span> ({nGenes} gene{nGenes === 1 ? '' : 's'}) — PCA → neighbours → Leiden{runUmap ? ' → UMAP' : ''}, on these genes only.
          The dataset's own PCA, graph and clusters are left alone.
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
            <div style={{ fontSize: 13, fontWeight: 600, marginBottom: 6 }}>Results</div>
            <div style={{ color: dark.muted, marginBottom: 8 }}>
              <span style={{ color: dark.text }}>{result.n_clusters} cluster{result.n_clusters === 1 ? '' : 's'}</span> from {result.n_genes_used} genes × {result.n_cells.toLocaleString()} cells
              {result.n_missing > 0 && <span style={{ color: dark.warn }}> · {result.n_missing} gene{result.n_missing === 1 ? '' : 's'} not in dataset</span>}
              <div style={{ color: dark.dim, fontSize: 11, marginTop: 2 }}>
                labels in <code>.obs['{result.obs_column}']</code>{result.embedding && <>, layout in <code>.obsm['{result.embedding}']</code></>}; {result.n_comps} PCs explain {(result.variance_ratio.reduce((a, b) => a + b, 0) * 100).toFixed(0)}% of the set's variance
              </div>
            </div>
            <div style={{ marginBottom: 10 }}>
              {Object.entries(result.cluster_sizes).sort((a, b) => b[1] - a[1]).map(([label, n]) => (
                <div key={label} style={{ display: 'flex', alignItems: 'center', gap: 8, fontSize: 11, marginBottom: 3 }}>
                  <span style={{ width: 60, color: dark.muted }}>cluster {label}</span>
                  <div style={{ flex: 1, height: 7, background: dark.border, borderRadius: 3, overflow: 'hidden' }}>
                    <div style={{ width: `${Math.round((n / result.n_cells) * 100)}%`, height: '100%', background: dark.accent }} />
                  </div>
                  <span style={{ width: 70, textAlign: 'right', color: dark.muted }}>{n.toLocaleString()}</span>
                </div>
              ))}
            </div>
            <div style={{ display: 'flex', gap: 8, justifyContent: 'flex-end', flexWrap: 'wrap' }}>
              <button onClick={() => { selectColorColumn(result.obs_column); setSource(null) }} style={btn}>Color by clusters</button>
              {result.embedding && (
                <button onClick={() => { selectEmbedding(result.embedding!); selectColorColumn(result.obs_column); setSource(null) }} style={btn}>Show UMAP</button>
              )}
              <button onClick={() => setPhase('config')} style={btnGhost}>Adjust &amp; re-run</button>
              <button onClick={() => setSource(null)} style={btnGhost}>Done</button>
            </div>
          </div>
        )}

        {phase === 'config' && (
          <>
            <Field label="Key (names leiden_<key>, X_umap_<key>, …)">
              <input value={key} onChange={(e) => setKey(sanitizeKey(e.target.value))} style={field} />
            </Field>
            <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr 1fr', gap: 10 }}>
              <Field label="PCs" tip="Clamped to the number of genes minus one">
                <input type="number" min={1} max={100} value={nComps} onChange={(e) => setNComps(Math.max(1, Number(e.target.value) || 1))} style={field} />
              </Field>
              <Field label="Neighbours">
                <input type="number" min={2} max={200} value={nNeighbors} onChange={(e) => setNNeighbors(Math.max(2, Number(e.target.value) || 2))} style={field} />
              </Field>
              <Field label="Resolution" tip="Higher = more clusters. A gene-set PCA fragments easily: at 1.0 a 33-gene set gave 100 clusters on 24k cells, so start low.">
                <input type="number" min={0.05} max={10} step={0.1} value={resolution} onChange={(e) => setResolution(Math.max(0.05, Number(e.target.value) || 0.05))} style={field} />
              </Field>
            </div>
            <div style={{ display: 'flex', gap: 16, marginBottom: 10, color: dark.muted }}>
              <label><input type="checkbox" checked={runUmap} onChange={(e) => setRunUmap(e.target.checked)} /> compute a UMAP of this clustering</label>
              <label title="z-score each gene before PCA so a highly expressed member does not dominate"><input type="checkbox" checked={scale} onChange={(e) => setScale(e.target.checked)} /> scale genes</label>
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
      </div>
    </div>
  )
}
