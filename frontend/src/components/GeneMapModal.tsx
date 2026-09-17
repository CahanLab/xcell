/**
 * GeneMapModal — genes as points.
 *
 * Builds a gene–gene similarity from up to three channels (expression across
 * cells, annotation membership in cached libraries, STRING edges), clusters
 * it into modules, lays the genes out in 2-D, and draws them on a canvas:
 * hover for the gene, lasso to make a gene set. A second view is the
 * clustered similarity heatmap; a third lists the modules for saving.
 *
 * Opened from a gene-set row (⋯ → Map genes…, genes fixed) or from the Genes
 * panel header (⋯ → Gene map…, pick a gene subset here). The store holds the
 * source (`geneMapSource`; null = closed).
 *
 * Rollback: delete this file, its mount in App.tsx, the two launchers in
 * GenePanel.tsx, `geneMapSource` in store.ts, and lib/genePlot.ts.
 */
import React, { useCallback, useEffect, useMemo, useRef, useState } from 'react'
import { useStore, cfgDefault } from '../store'
import { appendDataset, pollTask } from '../hooks/useData'
import GeneSubsetPicker, { useGeneSubset } from './GeneSubsetPicker'
import { fitTransform, moduleColor, moduleGeneSets, nearestPoint, pointsInLasso } from '../lib/genePlot'
import { clusterBands, similarityColor } from '../lib/metaPrograms'
import type { LibraryEntry, Species } from '../lib/geneSetLibrary'

const API_BASE = '/api'

type Tab = 'map' | 'similarity' | 'modules'

interface MapData {
  key: string
  genes: string[]
  coords: [number, number][]
  modules: number[]
  order: number[]
  module_sizes: number[]
  n_modules: number
  channels: Record<string, Record<string, unknown>>
  channel_weights: Record<string, number>
  params: Record<string, unknown>
  n_cells: number
  similarity?: number[][]
}

const dark = {
  panel: '#16213e', border: '#0f3460', inset: '#0f1625', accent: '#4ecdc4',
  alert: '#e94560', warn: '#e9a23b', muted: '#aaa', dim: '#888', text: '#ddd',
}
const field: React.CSSProperties = {
  background: dark.inset, color: dark.text, border: `1px solid ${dark.border}`,
  borderRadius: 4, padding: '4px 6px', fontSize: 12, boxSizing: 'border-box',
}
const btn: React.CSSProperties = {
  background: dark.accent, color: '#0f1625', border: 'none', borderRadius: 4,
  padding: '6px 14px', fontSize: 12, fontWeight: 600, cursor: 'pointer',
}
const btnGhost: React.CSSProperties = {
  background: 'transparent', color: dark.muted, border: `1px solid ${dark.border}`,
  borderRadius: 4, padding: '5px 12px', fontSize: 12, cursor: 'pointer',
}

async function getJson<T>(url: string): Promise<T> {
  const r = await fetch(url)
  const body = await r.json().catch(() => ({}))
  if (!r.ok) throw new Error(body.detail || `HTTP ${r.status}`)
  return body as T
}

function sanitizeKey(name: string): string {
  return name.replace(/[^A-Za-z0-9_]+/g, '_').replace(/^_+|_+$/g, '').slice(0, 40) || 'map'
}

export default function GeneMapModal() {
  const source = useStore((s) => s.geneMapSource)
  const setSource = useStore((s) => s.setGeneMapSource)
  const selectedCellIndices = useStore((s) => s.selectedCellIndices)
  const activeCellMask = useStore((s) => s.activeCellMask)
  const addGeneSetToCategory = useStore((s) => s.addGeneSetToCategory)
  const addFolderToCategory = useStore((s) => s.addFolderToCategory)
  const addScanpyAction = useStore((s) => s.addScanpyAction)
  const subset = useGeneSubset()

  const [phase, setPhase] = useState<'config' | 'running' | 'results'>('config')
  const [key, setKey] = useState('map')
  const [useExpr, setUseExpr] = useState(true)
  const [exprWeight, setExprWeight] = useState<number>(() => cfgDefault(['gene_map', 'expression_weight'], 1.0))
  const [metric, setMetric] = useState<string>(() => cfgDefault(['gene_map', 'expression_metric'], 'bicor'))
  const [layer, setLayer] = useState('X')
  const [layers, setLayers] = useState<{ name: string; density: number }[]>([])
  const [useAnn, setUseAnn] = useState(true)
  const [annWeight, setAnnWeight] = useState<number>(() => cfgDefault(['gene_map', 'annotation_weight'], 1.0))
  const [cachedLibs, setCachedLibs] = useState<LibraryEntry[]>([])
  const [chosenLibs, setChosenLibs] = useState<Set<string>>(new Set())
  const [useString, setUseString] = useState(false)
  const [stringWeight, setStringWeight] = useState<number>(() => cfgDefault(['gene_map', 'string_weight'], 0.5))
  const [species, setSpecies] = useState<Species>('mouse')
  const [stringScore, setStringScore] = useState(400)
  const [nNeighbors, setNNeighbors] = useState<number>(() => cfgDefault(['gene_map', 'n_neighbors'], 15))
  const [resolution, setResolution] = useState<number>(() => cfgDefault(['gene_map', 'resolution'], 1.0))
  const [embedding, setEmbedding] = useState<string>(() => cfgDefault(['gene_map', 'embedding'], 'umap'))
  const [scopeSelection, setScopeSelection] = useState(false)
  const [progress, setProgress] = useState<{ frac: number; message: string }>({ frac: 0, message: '' })
  const [error, setError] = useState<string | null>(null)
  const [data, setData] = useState<MapData | null>(null)
  const [tab, setTab] = useState<Tab>('map')
  const [notice, setNotice] = useState<string | null>(null)

  const toast = useCallback((m: string) => { setNotice(m); window.setTimeout(() => setNotice(null), 5000) }, [])

  useEffect(() => {
    if (!source) return
    setPhase('config'); setError(null); setData(null); setTab('map')
    setKey(sanitizeKey(source.genes ? source.name : 'gene_map'))
    setScopeSelection(false)
    getJson<{ layers: { name: string; density: number }[] }>(appendDataset(`${API_BASE}/scanpy/layers`))
      .then((d) => setLayers(d.layers || [{ name: 'X', density: 0 }])).catch(() => setLayers([{ name: 'X', density: 0 }]))
    getJson<{ cached: LibraryEntry[] }>(`${API_BASE}/gene_set_sources`)
      .then((d) => setCachedLibs(d.cached || [])).catch(() => setCachedLibs([]))
    getJson<{ species: string | null }>(appendDataset(`${API_BASE}/gene_sets/species_guess`))
      .then((g) => { if (g.species === 'human' || g.species === 'mouse') setSpecies(g.species) }).catch(() => {})
  }, [source])

  useEffect(() => {
    if (!source) return
    const onKey = (e: KeyboardEvent) => { if (e.key === 'Escape' && phase !== 'running') setSource(null) }
    window.addEventListener('keydown', onKey)
    return () => window.removeEventListener('keydown', onKey)
  }, [source, phase, setSource])

  const nGenesHint = source?.genes ? source.genes.length : (subset.selection.geneSets.reduce((n, s) => n + s.genes.length, 0) || null)

  const run = async (overwrite = false) => {
    if (!source) return
    setPhase('running'); setError(null); setProgress({ frac: 0, message: 'Starting…' })
    try {
      let cellBody: Record<string, unknown> = { cell_context: 'all' }
      if (scopeSelection && selectedCellIndices.length > 0) {
        const idx = activeCellMask ? selectedCellIndices.filter((i) => activeCellMask[i]) : selectedCellIndices
        cellBody = { cell_context: 'selection', cell_indices: idx }
      } else if (activeCellMask) {
        const idx: number[] = []
        for (let i = 0; i < activeCellMask.length; i++) if (activeCellMask[i]) idx.push(i)
        cellBody = { cell_context: 'selection', cell_indices: idx }
      }
      const body: Record<string, unknown> = {
        key: key.trim(),
        ...(source.genes ? { genes: source.genes } : { gene_subset: subset.spec }),
        expression_weight: useExpr ? exprWeight : 0,
        expression_metric: metric,
        layer: layer !== 'X' ? layer : null,
        annotation_weight: useAnn ? annWeight : 0,
        annotation_libraries: useAnn ? cachedLibs.filter((l) => chosenLibs.has(`${l.source}/${l.species}/${l.id}`)).map((l) => ({ source: l.source, id: l.id, species: l.species })) : [],
        string_weight: useString ? stringWeight : 0,
        string_species: species,
        string_required_score: stringScore,
        n_neighbors: nNeighbors, resolution, embedding, overwrite,
        ...cellBody,
      }
      const resp = await fetch(appendDataset(`${API_BASE}/gene_map/run`), { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(body) })
      if (!resp.ok) throw new Error((await resp.json().catch(() => ({}))).detail || `HTTP ${resp.status}`)
      const { task_id } = await resp.json()
      const task = await pollTask(task_id, undefined, (s) => setProgress({ frac: s.progress ?? 0, message: s.message ?? '' }))
      if (task.status !== 'completed') throw new Error(task.error || `Run ${task.status}`)
      const res = task.result as Record<string, unknown>
      addScanpyAction({ action: 'gene_map', params: body, result: res, timestamp: new Date().toISOString() })
      const full = await getJson<MapData>(appendDataset(`${API_BASE}/gene_map/${encodeURIComponent(key.trim())}`))
      setData(full)
      setPhase('results')
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e))
      setPhase('config')
    }
  }

  const loadSimilarity = useCallback(async () => {
    if (!data || data.similarity) return
    try {
      const full = await getJson<MapData>(appendDataset(`${API_BASE}/gene_map/${encodeURIComponent(data.key)}?similarity=true`))
      setData((d) => (d ? { ...d, similarity: full.similarity } : d))
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e))
    }
  }, [data])

  useEffect(() => { if (tab === 'similarity') loadSimilarity() }, [tab, loadSimilarity])

  if (!source) return null

  const canRun = key.trim().length > 0 && (source.genes ? source.genes.length >= 3 : subset.spec !== null) &&
    (useExpr || (useAnn && chosenLibs.size > 0) || useString)

  const overlayStyle: React.CSSProperties = { position: 'fixed', inset: 0, background: 'rgba(0,0,0,0.6)', zIndex: 100, display: 'flex', alignItems: 'center', justifyContent: 'center' }
  const cardStyle: React.CSSProperties = { background: dark.panel, border: `1px solid ${dark.border}`, borderRadius: 8, width: phase === 'results' ? 'min(1120px, 95vw)' : 620, maxWidth: '95vw', maxHeight: '90vh', overflowY: 'auto', padding: 16, color: dark.text, fontSize: 12, display: 'flex', flexDirection: 'column', gap: 10 }

  return (
    <div style={overlayStyle} onClick={() => phase !== 'running' && setSource(null)}>
      <div style={cardStyle} onClick={(e) => e.stopPropagation()}>
        <div style={{ display: 'flex', alignItems: 'center', gap: 10 }}>
          <div style={{ fontSize: 14, fontWeight: 600, color: dark.accent }}>Gene map</div>
          <div style={{ color: dark.muted }}>
            {source.genes ? <><span style={{ color: dark.text }}>{source.name}</span> ({source.genes.length} genes)</> : 'genes as points, from a subset you choose'}
          </div>
          <div style={{ flex: 1 }} />
          {data && (
            <div style={{ display: 'flex', gap: 4 }}>
              {(['map', 'similarity', 'modules'] as Tab[]).map((t) => (
                <button key={t} onClick={() => setTab(t)} style={{ ...btnGhost, color: tab === t ? dark.accent : dark.muted, borderColor: tab === t ? dark.accent : dark.border }}>{t}</button>
              ))}
            </div>
          )}
          <button onClick={() => phase !== 'running' && setSource(null)} style={{ ...btnGhost, padding: '2px 8px' }}>✕</button>
        </div>

        {error && (
          <div style={{ padding: '6px 10px', background: '#3a1f2b', border: `1px solid ${dark.alert}`, borderRadius: 4, color: '#f5b8c4' }}>
            {error}
            {/already exists/.test(error) && <div style={{ marginTop: 6 }}><button onClick={() => run(true)} style={{ ...btnGhost, color: dark.warn, borderColor: dark.warn }}>Overwrite it</button></div>}
          </div>
        )}

        {phase === 'config' && (
          <>
            {!source.genes && (
              <div>
                <div style={{ color: dark.dim, marginBottom: 4 }}>Genes ({subset.label}{nGenesHint ? `, ${nGenesHint}` : ''})</div>
                <GeneSubsetPicker control={subset} />
              </div>
            )}
            <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr 1fr', gap: 10 }}>
              <Channel title="Expression" on={useExpr} setOn={setUseExpr} weight={exprWeight} setWeight={setExprWeight} tip="Correlation across cells, mapped to [0, 1]">
                <select value={metric} onChange={(e) => setMetric(e.target.value)} style={{ ...field, width: '100%' }}>
                  <option value="bicor">bicor (robust)</option><option value="pearson">Pearson</option><option value="spearman">Spearman</option>
                </select>
                <select value={layer} onChange={(e) => setLayer(e.target.value)} style={{ ...field, width: '100%', marginTop: 4 }} title="A smoothed layer makes sparse-count correlations usable">
                  {layers.length === 0 && <option value="X">.X (log-normalised)</option>}
                  {layers.map((L) => <option key={L.name} value={L.name}>{L.name === 'X' ? '.X (log-normalised)' : L.name}</option>)}
                </select>
                <label style={{ display: 'block', marginTop: 4, color: dark.muted }} title="Restrict the expression channel to the current lasso selection">
                  <input type="checkbox" checked={scopeSelection} disabled={selectedCellIndices.length === 0} onChange={(e) => setScopeSelection(e.target.checked)} /> selection only ({selectedCellIndices.length.toLocaleString()})
                </label>
              </Channel>
              <Channel title="Annotation" on={useAnn} setOn={setUseAnn} weight={annWeight} setWeight={setAnnWeight} tip="Cosine similarity of which cached library sets each gene belongs to">
                {cachedLibs.length === 0 ? (
                  <div style={{ color: dark.warn }}>No cached libraries — fetch some in the Gene set library first.</div>
                ) : (
                  <div style={{ maxHeight: 110, overflowY: 'auto', border: `1px solid ${dark.border}`, borderRadius: 4, padding: 4 }}>
                    {cachedLibs.map((l) => {
                      const id = `${l.source}/${l.species}/${l.id}`
                      return (
                        <label key={id} style={{ display: 'block', color: dark.text, whiteSpace: 'nowrap', overflow: 'hidden', textOverflow: 'ellipsis' }} title={`${l.name} (${l.source}, ${l.species}, ${l.n_sets ?? '?'} sets)`}>
                          <input type="checkbox" checked={chosenLibs.has(id)} onChange={(e) => setChosenLibs((c) => { const n = new Set(c); e.target.checked ? n.add(id) : n.delete(id); return n })} /> {l.name} <span style={{ color: dark.dim }}>({l.species})</span>
                        </label>
                      )
                    })}
                  </div>
                )}
              </Channel>
              <Channel title="STRING" on={useString} setOn={setUseString} weight={stringWeight} setWeight={setStringWeight} tip="Combined interaction score of each STRING edge among the genes (live query)">
                <select value={species} onChange={(e) => setSpecies(e.target.value as Species)} style={{ ...field, width: '100%' }}>
                  <option value="mouse">mouse</option><option value="human">human</option>
                </select>
                <input type="number" min={0} max={1000} step={50} value={stringScore} onChange={(e) => setStringScore(Math.min(1000, Math.max(0, Number(e.target.value) || 0)))} style={{ ...field, width: '100%', marginTop: 4 }} title="Minimum combined score (0–1000)" />
              </Channel>
            </div>
            <div style={{ display: 'grid', gridTemplateColumns: 'repeat(4, 1fr)', gap: 10 }}>
              <Labeled label="Neighbours" tip="kNN for modules and the UMAP"><input type="number" min={2} max={100} value={nNeighbors} onChange={(e) => setNNeighbors(Math.max(2, Number(e.target.value) || 2))} style={{ ...field, width: '100%' }} /></Labeled>
              <Labeled label="Resolution" tip="Leiden; higher = more modules"><input type="number" min={0.1} step={0.1} value={resolution} onChange={(e) => setResolution(Math.max(0.05, Number(e.target.value) || 0.05))} style={{ ...field, width: '100%' }} /></Labeled>
              <Labeled label="Layout"><select value={embedding} onChange={(e) => setEmbedding(e.target.value)} style={{ ...field, width: '100%' }}><option value="umap">UMAP</option><option value="mds">MDS</option></select></Labeled>
              <Labeled label="Key"><input value={key} onChange={(e) => setKey(sanitizeKey(e.target.value))} style={{ ...field, width: '100%' }} /></Labeled>
            </div>
            <div style={{ display: 'flex', gap: 8, justifyContent: 'flex-end' }}>
              <button onClick={() => setSource(null)} style={btnGhost}>Cancel</button>
              <button onClick={() => run(false)} disabled={!canRun} style={{ ...btn, opacity: canRun ? 1 : 0.5 }} title={canRun ? '' : 'Pick genes and at least one channel (annotation needs a cached library)'}>Build map</button>
            </div>
          </>
        )}

        {phase === 'running' && (
          <div>
            <div style={{ height: 8, background: dark.border, borderRadius: 4, overflow: 'hidden' }}>
              <div style={{ width: `${Math.round(progress.frac * 100)}%`, height: '100%', background: dark.accent, transition: 'width 0.3s' }} />
            </div>
            <div style={{ color: dark.dim, marginTop: 4 }}>{progress.message}</div>
          </div>
        )}

        {phase === 'results' && data && (
          <>
            <div style={{ display: 'flex', gap: 14, flexWrap: 'wrap', color: dark.muted, background: dark.inset, border: `1px solid ${dark.border}`, borderRadius: 4, padding: '6px 10px' }}>
              <span><b style={{ color: dark.text }}>{data.genes.length}</b> genes</span>
              <span><b style={{ color: dark.text }}>{data.n_modules}</b> modules</span>
              {Object.entries(data.channel_weights).map(([c, w]) => (
                <span key={c}>{c} {Math.round(w * 100)}%{c === 'annotation' && data.channels.annotation ? ` (${String(data.channels.annotation.n_genes_annotated)} of ${data.genes.length} annotated, ${String(data.channels.annotation.n_terms)} sets)` : ''}{c === 'string' && data.channels.string ? ` (${String(data.channels.string.n_edges)} edges)` : ''}</span>
              ))}
              <span style={{ color: dark.dim }}>stored as <code>uns['xcell_gene_maps']['{data.key}']</code></span>
              <div style={{ flex: 1 }} />
              <button onClick={() => setPhase('config')} style={{ ...btnGhost, padding: '2px 8px' }}>Adjust &amp; re-run</button>
            </div>
            {tab === 'map' && <GeneScatter data={data} onSaveSet={(name, genes) => { addGeneSetToCategory('manual', name, genes); toast(`Saved "${name}" (${genes.length} genes) under Manual`) }} />}
            {tab === 'similarity' && <SimilarityView data={data} />}
            {tab === 'modules' && (
              <ModulesView data={data} onSaveAll={() => { const sets = moduleGeneSets(source.name, data.genes, data.modules); addFolderToCategory('gene_clusters', `${source.name} map modules (${data.key})`, sets); toast(`Saved ${sets.length} modules under Gene Clusters`) }} onSaveOne={(m) => { const set = moduleGeneSets(source.name, data.genes, data.modules)[m]; if (set) { addGeneSetToCategory('manual', set.name, set.genes); toast(`Saved "${set.name}" under Manual`) } }} />
            )}
          </>
        )}

        {notice && (
          <div style={{ position: 'fixed', bottom: 16, left: '50%', transform: 'translateX(-50%)', background: '#2d4a43', color: '#d6f5ee', border: `1px solid ${dark.accent}`, padding: '8px 16px', borderRadius: 6, fontSize: 12, zIndex: 2100 }}>{notice}</div>
        )}
      </div>
    </div>
  )
}

function Labeled({ label, tip, children }: { label: string; tip?: string; children: React.ReactNode }) {
  return <div title={tip}><div style={{ fontSize: 11, color: dark.dim, marginBottom: 3 }}>{label}</div>{children}</div>
}

function Channel({ title, on, setOn, weight, setWeight, tip, children }: { title: string; on: boolean; setOn: (b: boolean) => void; weight: number; setWeight: (n: number) => void; tip: string; children: React.ReactNode }) {
  return (
    <div style={{ border: `1px solid ${on ? dark.accent : dark.border}`, borderRadius: 4, padding: 8, opacity: on ? 1 : 0.6 }} title={tip}>
      <div style={{ display: 'flex', alignItems: 'center', gap: 6, marginBottom: 6 }}>
        <label style={{ fontWeight: 600, color: dark.text }}><input type="checkbox" checked={on} onChange={(e) => setOn(e.target.checked)} /> {title}</label>
        <div style={{ flex: 1 }} />
        <span style={{ color: dark.dim }}>weight</span>
        <input type="number" min={0} step={0.1} value={weight} onChange={(e) => setWeight(Math.max(0, Number(e.target.value) || 0))} style={{ ...field, width: 52 }} disabled={!on} />
      </div>
      {children}
    </div>
  )
}

const MAP_W = 760
const MAP_H = 520

function GeneScatter({ data, onSaveSet }: { data: MapData; onSaveSet: (name: string, genes: string[]) => void }) {
  const canvasRef = useRef<HTMLCanvasElement | null>(null)
  const [hover, setHover] = useState<number | null>(null)
  const [lasso, setLasso] = useState<[number, number][]>([])
  const [drawing, setDrawing] = useState(false)
  const [selected, setSelected] = useState<number[]>([])
  const [showLabels, setShowLabels] = useState(data.genes.length <= 150)
  const [setName, setSetName] = useState('')
  const xf = useMemo(() => fitTransform(data.coords, MAP_W, MAP_H, 24), [data.coords])
  const selectedSet = useMemo(() => new Set(selected), [selected])

  useEffect(() => {
    const canvas = canvasRef.current
    if (!canvas) return
    const dpr = Math.min(window.devicePixelRatio || 1, 2)
    canvas.width = MAP_W * dpr; canvas.height = MAP_H * dpr
    canvas.style.width = `${MAP_W}px`; canvas.style.height = `${MAP_H}px`
    const ctx = canvas.getContext('2d')
    if (!ctx) return
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0)
    ctx.fillStyle = dark.inset
    ctx.fillRect(0, 0, MAP_W, MAP_H)
    const r = data.genes.length > 800 ? 2.5 : 4
    const dim = selected.length > 0
    for (let i = 0; i < data.genes.length; i++) {
      const x = xf.toX(data.coords[i][0]), y = xf.toY(data.coords[i][1])
      const on = !dim || selectedSet.has(i)
      ctx.globalAlpha = on ? 0.95 : 0.25
      ctx.fillStyle = moduleColor(data.modules[i])
      ctx.beginPath(); ctx.arc(x, y, i === hover ? r + 2 : r, 0, Math.PI * 2); ctx.fill()
    }
    ctx.globalAlpha = 1
    if (showLabels) {
      ctx.font = '10px system-ui, sans-serif'
      ctx.textBaseline = 'middle'
      for (let i = 0; i < data.genes.length; i++) {
        if (dim && !selectedSet.has(i)) continue
        const x = xf.toX(data.coords[i][0]), y = xf.toY(data.coords[i][1])
        ctx.fillStyle = 'rgba(0,0,0,0.7)'; ctx.fillText(data.genes[i], x + r + 3, y + 1)
        ctx.fillStyle = '#e8e8e8'; ctx.fillText(data.genes[i], x + r + 2, y)
      }
    }
    if (hover != null) {
      const x = xf.toX(data.coords[hover][0]), y = xf.toY(data.coords[hover][1])
      ctx.strokeStyle = '#fff'; ctx.lineWidth = 1.5
      ctx.beginPath(); ctx.arc(x, y, r + 3, 0, Math.PI * 2); ctx.stroke()
    }
  }, [data, xf, hover, selectedSet, selected.length, showLabels])

  const local = (e: React.MouseEvent): [number, number] => {
    const rect = (e.currentTarget as HTMLElement).getBoundingClientRect()
    return [e.clientX - rect.left, e.clientY - rect.top]
  }

  const onMouseDown = (e: React.MouseEvent) => { const p = local(e); setDrawing(true); setLasso([p]) }
  const onMouseMove = (e: React.MouseEvent) => {
    const p = local(e)
    if (drawing) { setLasso((l) => [...l, p]); return }
    setHover(nearestPoint(data.coords, p[0], p[1], xf, 10))
  }
  const finish = () => {
    if (!drawing) return
    setDrawing(false)
    if (lasso.length >= 3) {
      const idx = pointsInLasso(data.coords, lasso, xf)
      setSelected(idx)
      if (idx.length > 0 && !setName) setSetName(`${data.key} lasso ${idx.length}`)
    } else {
      setSelected([])
    }
    setLasso([])
  }

  const hoverGene = hover != null ? data.genes[hover] : null
  const legend = data.module_sizes.map((n, m) => ({ m, n })).slice(0, 20)

  return (
    <div style={{ display: 'flex', gap: 12, minHeight: 0 }}>
      <div style={{ position: 'relative', width: MAP_W, height: MAP_H, cursor: 'crosshair', flexShrink: 0 }}
        onMouseDown={onMouseDown} onMouseMove={onMouseMove} onMouseUp={finish} onMouseLeave={() => { finish(); setHover(null) }}>
        <canvas ref={canvasRef} style={{ borderRadius: 4, border: `1px solid ${dark.border}`, display: 'block' }} />
        <svg width={MAP_W} height={MAP_H} style={{ position: 'absolute', inset: 0, pointerEvents: 'none' }}>
          {lasso.length > 1 && <path d={`M ${lasso.map((p) => p.join(' ')).join(' L ')} Z`} fill="rgba(78,205,196,0.15)" stroke={dark.accent} strokeWidth={1.5} strokeDasharray="4 3" />}
        </svg>
        {hoverGene && hover != null && (
          <div style={{ position: 'absolute', left: Math.min(MAP_W - 160, xf.toX(data.coords[hover][0]) + 12), top: Math.max(0, xf.toY(data.coords[hover][1]) - 28), background: '#000c', color: '#fff', padding: '3px 8px', borderRadius: 4, fontSize: 11, pointerEvents: 'none', whiteSpace: 'nowrap' }}>
            {hoverGene} · module {data.modules[hover] + 1}
            {data.channels.annotation && Array.isArray((data.channels.annotation as { terms_per_gene?: number[] }).terms_per_gene) ? ` · ${(data.channels.annotation as { terms_per_gene: number[] }).terms_per_gene[hover]} sets` : ''}
          </div>
        )}
      </div>
      <div style={{ flex: 1, minWidth: 220, display: 'flex', flexDirection: 'column', gap: 8 }}>
        <div style={{ color: dark.dim }}>Drag to lasso genes. Hover for names.</div>
        <label style={{ color: dark.muted }}><input type="checkbox" checked={showLabels} onChange={(e) => setShowLabels(e.target.checked)} /> gene labels</label>
        <div style={{ display: 'flex', flexWrap: 'wrap', gap: 4 }}>
          {legend.map(({ m, n }) => (
            <span key={m} style={{ fontSize: 10, color: dark.muted, display: 'inline-flex', alignItems: 'center', gap: 3 }}>
              <span style={{ width: 9, height: 9, borderRadius: 5, background: moduleColor(m), display: 'inline-block' }} /> {m + 1} ({n})
            </span>
          ))}
          {data.module_sizes.length > 20 && <span style={{ fontSize: 10, color: dark.dim }}>+{data.module_sizes.length - 20} more</span>}
        </div>
        {selected.length > 0 && (
          <div style={{ border: `1px solid ${dark.accent}`, borderRadius: 4, padding: 8, background: dark.inset }}>
            <div style={{ color: dark.text, marginBottom: 4 }}>{selected.length} gene{selected.length === 1 ? '' : 's'} selected</div>
            <div style={{ color: dark.muted, fontSize: 11, maxHeight: 120, overflowY: 'auto', marginBottom: 6, lineHeight: 1.5 }}>{selected.map((i) => data.genes[i]).join(', ')}</div>
            <input value={setName} onChange={(e) => setSetName(e.target.value)} style={{ ...field, width: '100%', marginBottom: 6 }} placeholder="Gene set name" />
            <div style={{ display: 'flex', gap: 6 }}>
              <button onClick={() => { if (setName.trim()) onSaveSet(setName.trim(), selected.map((i) => data.genes[i])) }} style={btn} disabled={!setName.trim()}>Save as gene set</button>
              <button onClick={() => setSelected([])} style={btnGhost}>Clear</button>
            </div>
          </div>
        )}
      </div>
    </div>
  )
}

function SimilarityView({ data }: { data: MapData }) {
  const ref = useRef<HTMLCanvasElement | null>(null)
  const [hover, setHover] = useState<{ i: number; j: number } | null>(null)
  const n = data.genes.length
  const size = Math.min(560, Math.max(240, n * 6))
  const bands = useMemo(() => clusterBands(data.modules, data.order), [data.modules, data.order])

  useEffect(() => {
    const canvas = ref.current
    if (!canvas || !data.similarity) return
    const cell = size / n
    const dpr = window.devicePixelRatio || 1
    canvas.width = size * dpr; canvas.height = size * dpr
    canvas.style.width = `${size}px`; canvas.style.height = `${size}px`
    const ctx = canvas.getContext('2d')
    if (!ctx) return
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0)
    ctx.fillStyle = dark.inset; ctx.fillRect(0, 0, size, size)
    for (let i = 0; i < n; i++) {
      const row = data.similarity[data.order[i]]
      if (!row) continue
      for (let j = 0; j < n; j++) {
        ctx.fillStyle = similarityColor(row[data.order[j]])
        ctx.fillRect(j * cell, i * cell, cell + 1, cell + 1)
      }
    }
    ctx.strokeStyle = 'rgba(255,255,255,0.55)'; ctx.lineWidth = 1
    for (const band of bands) {
      if (band.cluster < 0) continue
      const x = band.start * cell, w = band.size * cell
      ctx.strokeRect(x + 0.5, x + 0.5, w - 1, w - 1)
    }
  }, [data, n, size, bands])

  if (n > 1500) return <div style={{ color: dark.warn }}>The similarity heatmap is drawn for up to 1,500 genes; this map has {n}.</div>
  if (!data.similarity) return <div style={{ color: dark.dim }}>Loading similarity…</div>
  const cell = size / n
  return (
    <div style={{ display: 'flex', gap: 12 }}>
      <canvas ref={ref} style={{ borderRadius: 4, border: `1px solid ${dark.border}`, cursor: 'crosshair', flexShrink: 0 }}
        onMouseMove={(e) => { const r = e.currentTarget.getBoundingClientRect(); const i = Math.floor((e.clientY - r.top) / cell), j = Math.floor((e.clientX - r.left) / cell); setHover(i >= 0 && j >= 0 && i < n && j < n ? { i, j } : null) }}
        onMouseLeave={() => setHover(null)} />
      <div style={{ color: dark.muted, lineHeight: 1.6, minWidth: 200 }}>
        <div>Every gene against every other, in dendrogram order; boxed blocks are modules.</div>
        <div style={{ display: 'flex', alignItems: 'center', gap: 6, margin: '8px 0', fontSize: 10, color: dark.dim }}>
          <span>0</span><div style={{ flex: 1, height: 8, borderRadius: 4, background: `linear-gradient(to right, ${similarityColor(0)}, ${similarityColor(0.5)}, ${similarityColor(1)})` }} /><span>1</span>
        </div>
        {hover && data.similarity && (
          <div style={{ color: dark.text }}>
            {data.genes[data.order[hover.i]]} × {data.genes[data.order[hover.j]]}: <b>{data.similarity[data.order[hover.i]][data.order[hover.j]].toFixed(3)}</b>
          </div>
        )}
      </div>
    </div>
  )
}

function ModulesView({ data, onSaveAll, onSaveOne }: { data: MapData; onSaveAll: () => void; onSaveOne: (m: number) => void }) {
  const [open, setOpen] = useState<number | null>(null)
  const groups = useMemo(() => moduleGeneSets('m', data.genes, data.modules), [data.genes, data.modules])
  return (
    <div>
      <div style={{ display: 'flex', alignItems: 'center', gap: 8, marginBottom: 8 }}>
        <span style={{ color: dark.muted }}>{groups.length} modules from Leiden on the similarity's kNN graph.</span>
        <div style={{ flex: 1 }} />
        <button onClick={onSaveAll} style={btn}>Save all modules as gene sets</button>
      </div>
      <div style={{ maxHeight: '58vh', overflowY: 'auto' }}>
        {groups.map((g, m) => (
          <div key={m} style={{ border: `1px solid ${dark.border}`, borderRadius: 4, padding: '6px 10px', marginBottom: 6, background: dark.inset }}>
            <div style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
              <span style={{ width: 10, height: 10, borderRadius: 5, background: moduleColor(m), display: 'inline-block' }} />
              <b>module {m + 1}</b><span style={{ color: dark.dim }}>{g.genes.length} genes</span>
              <div style={{ flex: 1 }} />
              <button onClick={() => onSaveOne(m)} style={btnGhost}>Save</button>
              <button onClick={() => setOpen(open === m ? null : m)} style={btnGhost}>{open === m ? 'Hide' : 'Genes'}</button>
            </div>
            <div style={{ color: dark.muted, fontSize: 11, marginTop: 4, lineHeight: 1.5 }}>
              {(open === m ? g.genes : g.genes.slice(0, 14)).join(', ')}{open !== m && g.genes.length > 14 ? ` … +${g.genes.length - 14}` : ''}
            </div>
          </div>
        ))}
      </div>
    </div>
  )
}
