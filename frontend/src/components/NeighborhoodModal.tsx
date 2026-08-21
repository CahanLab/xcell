/**
 * Neighborhood tool (Analyze → Spatial → Neighborhood).
 *
 * Quantifies which cell types surround each cell type in space. Builds a
 * spatial kNN (or radius) graph, computes each cell's neighbor-type fractions,
 * and tests every pair of types for co-location against a label-shuffling null
 * (histoCAT / squidpy-style permutation test, abundance-controlled). Results
 * render as a types × types heatmap: Enrichment (z, red = together more than
 * chance, blue = less) or Composition (% of a type's neighbors), ordered by
 * hierarchical clustering so co-locating types sit together, and exportable
 * as CSV (long or matrix) or SVG.
 *
 * Backend: /scanpy/neighborhood/run (background task), /scanpy/neighborhood/result.
 * Per-cell fractions land in obsm['neighborhood_composition'] → score pills.
 *
 * Rollback: delete this file, remove its mount in App.tsx, the Spatial
 * "Neighborhood" launcher in ScanpyModal, and isNeighborhoodModalOpen in store.ts.
 */
import { useEffect, useMemo, useRef, useState } from 'react'
import { useStore } from '../store'
import { appendDataset, pollTask, refreshSchema, type ObsSummary } from '../hooks/useData'
import { resolveCategoryPalette } from '../lib/cellColors'
import { zDomain, divergingColor, seqColor, formatZ, formatPct } from '../lib/neighborhoodViz'
import {
  hierarchicalOrder, dendrogramSegments, type Linkage, type DistanceMetric,
} from '../lib/hclust'
import { toLongCsv, toMatrixCsv } from '../lib/neighborhoodExport'
import { standaloneSvg, downloadText } from '../lib/svgExport'

const API_BASE = '/api'

interface NeighborhoodResult {
  categories: string[]
  composition: number[][]
  counts: number[][]
  zscores: number[][]
  pvals: number[][]
  qvals: number[][]
  log2fc: number[][]
  expected: number[][]
  n_cells: number
  n_edges: number
  mean_neighbors: number
  n_perms: number
  params: Record<string, unknown>
}

type Phase = 'config' | 'running' | 'results'
type View = 'z' | 'comp'
type Ordering = 'clustered' | 'input'

const TIPS: Record<string, string> = {
  column: 'Categorical .obs column (clusters or cell types) whose spatial neighborhoods to quantify.',
  mode: 'kNN: each cell\'s k nearest cells (robust to density differences; the usual choice). Radius: all cells within a fixed distance, in coordinate units.',
  n_neighs: 'Neighbors per cell for the kNN graph. 5–15 is standard (10 ≈ the CODEX "window" of Schürch et al.); results are robust across this range.',
  radius: 'Neighborhood radius in coordinate units (only for radius mode).',
  n_perms: 'Label permutations for the null. 1000 gives stable z-scores and q-values ≥ 0.002 resolution.',
  section_col: 'Optional categorical .obs column of tissue sections. When set, neighborhoods and the permutation null never cross section boundaries.',
}

const ORDER_TIPS: Record<string, string> = {
  order: 'Clustered orders types by how similar their neighborhoods are, which is what makes blocks of co-locating types visible. Input order is the column\u2019s own category order.',
  clusterOn: 'Which matrix the ordering is computed from. Enrichment groups types that like (or avoid) the same partners; composition groups types whose neighborhoods have the same makeup.',
  metric: 'Correlation compares the shape of two types\u2019 profiles and ignores how strong they are \u2014 the usual choice. Euclidean also weights magnitude; cosine sits between the two.',
  linkage: 'Average is the robust default. Complete makes tight, equal-sized blocks; single chains types together and can produce one long ladder; ward minimises within-block variance and pairs with euclidean.',
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

const rgbTuple = (s: string): [number, number, number] => {
  const m = s.match(/\d+/g) ?? ['0', '0', '0']
  return [Number(m[0]), Number(m[1]), Number(m[2])]
}
const luma = ([r, g, b]: [number, number, number]) => 0.299 * r + 0.587 * g + 0.114 * b

export default function NeighborhoodModal() {
  const isOpen = useStore((s) => s.isNeighborhoodModalOpen)
  const setOpen = useStore((s) => s.setNeighborhoodModalOpen)
  const refreshObsSummaries = useStore((s) => s.refreshObsSummaries)
  const addScanpyAction = useStore((s) => s.addScanpyAction)

  const [phase, setPhase] = useState<Phase>('config')
  const [column, setColumn] = useState('')
  const [mode, setMode] = useState<'knn' | 'radius'>('knn')
  const [nNeighs, setNNeighs] = useState('10')
  const [radius, setRadius] = useState('')
  const [nPerms, setNPerms] = useState('1000')
  const [sectionCol, setSectionCol] = useState('')
  const [summaries, setSummaries] = useState<ObsSummary[]>([])

  const [busy, setBusy] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const [result, setResult] = useState<NeighborhoodResult | null>(null)
  const [reusedPrior, setReusedPrior] = useState(false)
  const [progress, setProgress] = useState<{ frac: number; message: string; startedAt: number; now: number } | null>(null)
  const [view, setView] = useState<View>('z')
  const [hover, setHover] = useState<{ i: number; j: number } | null>(null)
  // Ordering lives here, not in Results, so switching views does not reshuffle
  // the heatmap under the user.
  const [ordering, setOrdering] = useState<Ordering>('clustered')
  const [clusterOn, setClusterOn] = useState<View>('z')
  const [metric, setMetric] = useState<DistanceMetric>('correlation')
  const [linkage, setLinkage] = useState<Linkage>('average')

  const close = () => {
    setOpen(false)
    setPhase('config')
    setResult(null)
    setError(null)
    setBusy(false)
    setProgress(null)
    setReusedPrior(false)
    setHover(null)
  }

  // On open: load categorical columns, and reuse a prior result if one exists.
  useEffect(() => {
    if (!isOpen) return
    let cancelled = false
    fetch(appendDataset(`${API_BASE}/obs/summaries`))
      .then((r) => (r.ok ? r.json() : null))
      .then((d) => {
        if (cancelled || !Array.isArray(d)) return
        const cats = (d as ObsSummary[]).filter((s) => s.dtype === 'category')
        setSummaries(cats)
        setColumn((prev) => {
          if (prev) return prev
          const preferred = cats.find((s) => /leiden|cell_type|cluster|type/i.test(s.name)) ?? cats[0]
          return preferred?.name ?? ''
        })
      })
      .catch(() => {})
    fetch(appendDataset(`${API_BASE}/scanpy/neighborhood/result`))
      .then((r) => (r.ok ? r.json() : null))
      .then((d) => {
        if (cancelled || !d || !d.categories) return
        setResult(d as NeighborhoodResult)
        const col = (d.params?.column as string) ?? ''
        if (col) setColumn(col)
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

  const run = async () => {
    setError(null); setBusy(true); setPhase('running')
    setReusedPrior(false)
    setProgress({ frac: 0, message: 'Starting…', startedAt: Date.now(), now: Date.now() })
    try {
      const body: Record<string, unknown> = {
        column,
        mode,
        n_perms: parseInt(nPerms, 10) || 1000,
      }
      if (mode === 'knn') body.n_neighs = parseInt(nNeighs, 10) || 10
      if (mode === 'radius') body.radius = parseFloat(radius)
      if (sectionCol) body.section_col = sectionCol

      const resp = await fetch(appendDataset(`${API_BASE}/scanpy/neighborhood/run`), {
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
      if (task.status !== 'completed') {
        throw new Error(task.error || `Analysis ${task.status}`)
      }
      const res = task.result as unknown as NeighborhoodResult
      setResult(res)
      setPhase('results')
      // New obsm score matrix -> the pills/axis pickers read the schema.
      await refreshSchema()
      refreshObsSummaries()
      addScanpyAction({
        action: 'neighborhood_enrichment',
        params: { ...body },
        result: { n_types: res.categories.length, n_edges: res.n_edges },
        timestamp: new Date().toISOString(),
      })
    } catch (e) {
      setError((e as Error).message)
      setPhase('config')
    } finally {
      setBusy(false)
    }
  }

  // Cluster swatch colors: dataset-author colors win, generated palette fills in.
  const palette = useMemo(() => {
    if (!result) return []
    const colName = (result.params?.column as string) ?? column
    const byValue = new Map<string, string>()
    summaries.find((s) => s.name === colName)?.categories?.forEach((c) => {
      if (c.color) byValue.set(c.value, c.color)
    })
    return resolveCategoryPalette(
      result.categories.length,
      result.categories.map((c) => byValue.get(c)),
    )
  }, [result, summaries, column])

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
          padding: '20px 24px', width: phase === 'results' ? 760 : 560, maxWidth: '94vw',
          maxHeight: '90vh', overflowY: 'auto', color: '#eee',
        }}
      >
        <div style={{ fontSize: 16, fontWeight: 600, marginBottom: 4 }}>Cell-type neighborhoods</div>
        <div style={{ fontSize: 12, color: '#aaa', marginBottom: 16 }}>
          Which types are found around each type — and which pairs co-locate more (or less)
          than expected from their abundances (permutation test, histoCAT/squidpy-style).
        </div>

        {phase === 'running' && (() => {
          const frac = progress?.frac ?? 0
          const pct = Math.round(frac * 100)
          const elapsed = progress ? (progress.now - progress.startedAt) / 1000 : 0
          const eta = frac > 0.02 ? (elapsed * (1 - frac)) / frac : null
          return (
            <div style={{ padding: '20px 0' }}>
              <div style={{ fontSize: 13, color: '#9be7d8', marginBottom: 10 }}>
                {progress?.message ?? 'Computing…'}
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
                <Field label="Cell type / cluster column" tip={TIPS.column}>
                  <select value={column} onChange={(e) => setColumn(e.target.value)} style={numInput}>
                    {summaries.map((s) => (
                      <option key={s.name} value={s.name}>{s.name}</option>
                    ))}
                  </select>
                </Field>
                <Field label="Neighborhood" tip={TIPS.mode}>
                  <select value={mode} onChange={(e) => setMode(e.target.value as 'knn' | 'radius')} style={numInput}>
                    <option value="knn">k nearest neighbors</option>
                    <option value="radius">radius</option>
                  </select>
                </Field>
                {mode === 'knn' ? (
                  <Field label="Neighbors (k)" tip={TIPS.n_neighs}>
                    <input type="number" value={nNeighs} onChange={(e) => setNNeighs(e.target.value)} style={numInput} />
                  </Field>
                ) : (
                  <Field label="Radius" tip={TIPS.radius}>
                    <input type="number" value={radius} onChange={(e) => setRadius(e.target.value)} placeholder="coordinate units" style={numInput} />
                  </Field>
                )}
              </div>
              <div style={{ flex: 1 }}>
                <Field label="Permutations" tip={TIPS.n_perms}>
                  <input type="number" value={nPerms} onChange={(e) => setNPerms(e.target.value)} style={numInput} />
                </Field>
                <Field label="Section column" tip={TIPS.section_col}>
                  <select value={sectionCol} onChange={(e) => setSectionCol(e.target.value)} style={numInput}>
                    <option value="">— treat as one tissue —</option>
                    {summaries.map((s) => (
                      <option key={s.name} value={s.name}>{s.name}</option>
                    ))}
                  </select>
                </Field>
              </div>
            </div>

            {error && <div style={errBox}>{error}</div>}

            <div style={{ display: 'flex', justifyContent: 'flex-end', gap: 8 }}>
              <button onClick={close} style={ghost}>Cancel</button>
              <button onClick={run} disabled={busy || !column} style={primary(!busy && !!column)}>
                Quantify neighborhoods
              </button>
            </div>
          </>
        )}

        {phase === 'results' && result && (
          <Results
            result={result}
            palette={palette}
            view={view}
            setView={setView}
            hover={hover}
            setHover={setHover}
            reusedPrior={reusedPrior}
            onBack={() => setPhase('config')}
            onClose={close}
            error={error}
            ordering={ordering}
            setOrdering={setOrdering}
            clusterOn={clusterOn}
            setClusterOn={setClusterOn}
            metric={metric}
            setMetric={setMetric}
            linkage={linkage}
            setLinkage={setLinkage}
          />
        )}
      </div>
    </div>
  )
}

function Results({
  result, palette, view, setView, hover, setHover, reusedPrior, onBack, onClose, error,
  ordering, setOrdering, clusterOn, setClusterOn, metric, setMetric, linkage, setLinkage,
}: {
  result: NeighborhoodResult
  palette: [number, number, number][]
  view: View
  setView: (v: View) => void
  hover: { i: number; j: number } | null
  setHover: (h: { i: number; j: number } | null) => void
  reusedPrior: boolean
  onBack: () => void
  onClose: () => void
  error: string | null
  ordering: Ordering
  setOrdering: (o: Ordering) => void
  clusterOn: View
  setClusterOn: (v: View) => void
  metric: DistanceMetric
  setMetric: (m: DistanceMetric) => void
  linkage: Linkage
  setLinkage: (l: Linkage) => void
}) {
  const cats = result.categories
  const K = cats.length
  const dom = useMemo(() => zDomain(result.zscores), [result])
  const maxComp = useMemo(() => Math.max(...result.composition.flat(), 1e-9), [result])
  const svgRef = useRef<SVGSVGElement>(null)
  const [exportOpen, setExportOpen] = useState(false)

  // Types are ordered by the similarity of their *rows* — each row being one
  // type's neighborhood profile. Columns take the same order: the matrix is
  // near-symmetric (i next to j implies j next to i), and a shared order is
  // what keeps self-neighborhood on the diagonal where it can be read.
  const clustering = useMemo(() => {
    if (ordering !== 'clustered' || K < 3) return null
    const source = clusterOn === 'z' ? result.zscores : result.composition
    return hierarchicalOrder(source, { metric, linkage })
  }, [ordering, clusterOn, metric, linkage, result, K])

  const order = useMemo(
    () => clustering?.order ?? cats.map((_, i) => i),
    [clustering, cats],
  )
  const dendro = useMemo(() => (clustering ? dendrogramSegments(clustering) : null), [clustering])

  const cell = Math.max(13, Math.min(26, Math.floor(440 / K)))
  const labelW = 108
  const labelH = 84
  const dendroH = dendro ? 46 : 0
  const grid = K * cell
  const legendH = 46
  const width = labelW + grid + 16
  const height = dendroH + labelH + grid + legendH

  const fill = (i: number, j: number) =>
    view === 'z' ? divergingColor(result.zscores[i][j], dom) : seqColor(result.composition[i][j] / maxComp)

  const trunc = (s: string, n = 14) => (s.length > n ? s.slice(0, n - 1) + '…' : s)

  const h = hover
  const params = result.params ?? {}
  const graphDesc = params.mode === 'radius' ? `radius ${params.radius}` : `k=${params.n_neighs ?? 10}`

  const fileBase = `neighborhood_${String(params.column ?? 'types')}`.replace(/[^\w.-]+/g, '_')
  const doExport = (kind: 'long' | 'matrix' | 'svg') => {
    setExportOpen(false)
    if (kind === 'long') {
      downloadText(`${fileBase}_stats.csv`, toLongCsv(result, order), 'text/csv')
    } else if (kind === 'matrix') {
      const values = view === 'z' ? result.zscores : result.composition
      downloadText(
        `${fileBase}_${view === 'z' ? 'zscore' : 'composition'}.csv`,
        toMatrixCsv(values, cats, order),
        'text/csv',
      )
    } else if (svgRef.current) {
      downloadText(
        `${fileBase}_${view === 'z' ? 'zscore' : 'composition'}.svg`,
        standaloneSvg(svgRef.current.outerHTML, { background: '#16213e' }),
        'image/svg+xml',
      )
    }
  }

  return (
    <>
      <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', marginBottom: 8 }}>
        <div style={{ fontSize: 12, color: '#9be7d8' }}>
          {String(params.column ?? '')} · {K} types · {result.n_cells.toLocaleString()} cells · {graphDesc} · {result.n_perms.toLocaleString()} permutations
        </div>
        <div style={{ display: 'flex', gap: 4 }}>
          <button onClick={() => setView('z')} style={toggle(view === 'z')}>Enrichment (z)</button>
          <button onClick={() => setView('comp')} style={toggle(view === 'comp')}>Composition</button>
        </div>
      </div>
      {reusedPrior && (
        <div style={{ fontSize: 11, color: '#aaa', marginBottom: 8, fontStyle: 'italic' }}>
          Showing a previous run on this dataset. Use ← Change parameters to re-run.
        </div>
      )}

      {/* Ordering controls — defaults are the answer for most datasets; they
          are exposed because linkage changes what the blocks look like. */}
      <div style={{
        display: 'flex', alignItems: 'center', gap: 10, flexWrap: 'wrap',
        padding: '6px 8px', marginBottom: 8, backgroundColor: '#0f1625',
        border: '1px solid #0f3460', borderRadius: 4,
      }}>
        <MiniSelect
          label="Order" tip={ORDER_TIPS.order} value={ordering}
          onChange={(v) => setOrdering(v as Ordering)}
          options={[['clustered', 'Clustered'], ['input', 'Input order']]}
          disabled={K < 3}
        />
        {ordering === 'clustered' && K >= 3 && (
          <>
            <MiniSelect
              label="Cluster on" tip={ORDER_TIPS.clusterOn} value={clusterOn}
              onChange={(v) => setClusterOn(v as View)}
              options={[['z', 'Enrichment'], ['comp', 'Composition']]}
            />
            <MiniSelect
              label="Distance" tip={ORDER_TIPS.metric} value={metric}
              onChange={(v) => setMetric(v as DistanceMetric)}
              options={[['correlation', 'Correlation'], ['euclidean', 'Euclidean'], ['cosine', 'Cosine']]}
            />
            <MiniSelect
              label="Linkage" tip={ORDER_TIPS.linkage} value={linkage}
              onChange={(v) => setLinkage(v as Linkage)}
              options={[['average', 'Average'], ['complete', 'Complete'], ['single', 'Single'], ['ward', 'Ward']]}
            />
          </>
        )}
        <div style={{ marginLeft: 'auto', position: 'relative' }}>
          <button onClick={() => setExportOpen((o) => !o)} style={toggle(exportOpen)}>Export ▾</button>
          {exportOpen && (
            <>
              <div onClick={() => setExportOpen(false)} style={{ position: 'fixed', inset: 0, zIndex: 10 }} />
              <div style={{
                position: 'absolute', right: 0, top: '100%', marginTop: 4, zIndex: 11,
                backgroundColor: '#16213e', border: '1px solid #0f3460', borderRadius: 4,
                minWidth: 210, boxShadow: '0 4px 14px rgba(0,0,0,0.5)', overflow: 'hidden',
              }}>
                {([
                  ['long', 'Statistics table (CSV)', 'every pair, all statistics'],
                  ['matrix', `${view === 'z' ? 'Enrichment' : 'Composition'} matrix (CSV)`, 'as shown, in display order'],
                  ['svg', 'Figure (SVG)', 'vector, editable in Illustrator'],
                ] as const).map(([kind, label, hint]) => (
                  <button
                    key={kind}
                    onClick={() => doExport(kind)}
                    style={{
                      display: 'block', width: '100%', textAlign: 'left', padding: '7px 10px',
                      fontSize: 11, backgroundColor: 'transparent', color: '#eee',
                      border: 'none', borderBottom: '1px solid #0f3460', cursor: 'pointer',
                    }}
                  >
                    {label}
                    <div style={{ fontSize: 10, color: '#777' }}>{hint}</div>
                  </button>
                ))}
              </div>
            </>
          )}
        </div>
      </div>

      <div style={{ fontSize: 11, color: '#888', marginBottom: 6 }}>
        {view === 'z'
          ? 'Row × column: are the two types spatial neighbors more (red) or less (blue) often than chance, given their abundances? Dot = q < 0.05.'
          : 'Row × column: what fraction of the row type’s neighbors are the column type? Rows sum to 100%.'}
      </div>

      <svg ref={svgRef} width={width} height={height} style={{ display: 'block', margin: '0 auto' }}>
        {/* dendrogram over the shared row/column order */}
        {dendro && dendro.segments.map((seg, idx) => {
          const px = (x: number) => labelW + x * cell + cell / 2
          const py = (y: number) => dendroH - 4 - (y / dendro.maxHeight) * (dendroH - 10)
          return (
            <line
              key={`d${idx}`}
              x1={px(seg.x1)} y1={py(seg.y1)} x2={px(seg.x2)} y2={py(seg.y2)}
              stroke="#4a6a8a" strokeWidth={1}
            />
          )
        })}
        {/* column labels (rotated) + swatches */}
        {order.map((c, jPos) => {
          const x = labelW + jPos * cell + cell / 2
          const [r, g, b] = palette[c] ?? [136, 136, 136]
          return (
            <g key={`c${c}`}>
              <rect x={x - 4} y={dendroH + labelH - 12} width={8} height={8} rx={2}
                fill={`rgb(${r},${g},${b})`} />
              <text x={x} y={dendroH + labelH - 18} fill={h && h.j === c ? '#fff' : '#aaa'} fontSize={10}
                textAnchor="start" transform={`rotate(-45 ${x} ${dendroH + labelH - 18})`}>
                {trunc(cats[c])}<title>{cats[c]}</title>
              </text>
            </g>
          )
        })}
        {/* row labels + swatches */}
        {order.map((c, iPos) => {
          const y = dendroH + labelH + iPos * cell + cell / 2
          const [r, g, b] = palette[c] ?? [136, 136, 136]
          return (
            <g key={`r${c}`}>
              <text x={labelW - 16} y={y + 3} fill={h && h.i === c ? '#fff' : '#aaa'} fontSize={10} textAnchor="end">
                {trunc(cats[c])}<title>{cats[c]}</title>
              </text>
              <rect x={labelW - 12} y={y - 4} width={8} height={8} rx={2} fill={`rgb(${r},${g},${b})`} />
            </g>
          )
        })}
        {/* cells */}
        <g onMouseLeave={() => setHover(null)}>
          {order.map((i, iPos) =>
            order.map((j, jPos) => {
              const bg = fill(i, j)
              const sig = view === 'z' && result.qvals[i][j] < 0.05
              const hovered = h && h.i === i && h.j === j
              return (
                <g key={`${i}-${j}`}>
                  <rect
                    x={labelW + jPos * cell + 1} y={dendroH + labelH + iPos * cell + 1}
                    width={cell - 2} height={cell - 2} rx={2}
                    fill={bg}
                    stroke={hovered ? '#eee' : 'none'} strokeWidth={1.5}
                    onMouseEnter={() => setHover({ i, j })}
                  />
                  {sig && (
                    <circle
                      cx={labelW + jPos * cell + cell - 5} cy={dendroH + labelH + iPos * cell + 5} r={1.8}
                      fill={luma(rgbTuple(bg)) > 140 ? '#101418' : '#eee'}
                      pointerEvents="none"
                    />
                  )}
                </g>
              )
            }),
          )}
        </g>
        {/* scale legend */}
        <defs>
          <linearGradient id="nbLegend" x1="0" y1="0" x2="1" y2="0">
            {[0, 0.25, 0.5, 0.75, 1].map((t) => (
              <stop key={t} offset={`${t * 100}%`}
                stopColor={view === 'z' ? divergingColor((t * 2 - 1) * dom, dom) : seqColor(t)} />
            ))}
          </linearGradient>
        </defs>
        <rect x={labelW} y={dendroH + labelH + grid + 18} width={150} height={8} rx={2} fill="url(#nbLegend)" />
        <text x={labelW} y={dendroH + labelH + grid + 40} fill="#888" fontSize={9} textAnchor="start">
          {view === 'z' ? `≤ ${formatZ(-dom)}` : '0%'}
        </text>
        <text x={labelW + 75} y={dendroH + labelH + grid + 40} fill="#888" fontSize={9} textAnchor="middle">
          {view === 'z' ? 'z = 0' : ''}
        </text>
        <text x={labelW + 150} y={dendroH + labelH + grid + 40} fill="#888" fontSize={9} textAnchor="end">
          {view === 'z' ? `≥ ${formatZ(dom)}` : formatPct(maxComp)}
        </text>
        {view === 'z' && (
          <text x={labelW + 190} y={dendroH + labelH + grid + 27} fill="#888" fontSize={9}>
            • q &lt; 0.05
          </text>
        )}
      </svg>

      {/* readout bar: value leads, labels follow */}
      <div style={{
        marginTop: 6, minHeight: 30, padding: '6px 10px', backgroundColor: '#0f1625',
        border: '1px solid #0f3460', borderRadius: 4, fontSize: 12,
      }}>
        {h ? (() => {
          const z = result.zscores[h.i][h.j]
          const q = result.qvals[h.i][h.j]
          const obsF = result.composition[h.i][h.j]
          const expF = result.expected[h.i][h.j]
          const n = result.counts[h.i][h.j]
          return (
            <span>
              <strong style={{ color: view === 'z' ? (z > 0 ? '#f2886b' : '#82aaf2') : '#7ce8da', fontSize: 13 }}>
                {view === 'z' ? `z ${formatZ(z)}` : formatPct(obsF)}
              </strong>
              <span style={{ color: '#ccc' }}> — {cats[h.i]} → {cats[h.j]}</span>
              <span style={{ color: '#888' }}>
                {' '}· {formatPct(obsF)} of neighbors vs {formatPct(expF)} expected
                · q {q < 0.001 ? '< 0.001' : q.toFixed(3)} · {n.toLocaleString()} pairs
              </span>
            </span>
          )
        })() : (
          <span style={{ color: '#666' }}>Hover a cell: row type → how much of its neighborhood is the column type.</span>
        )}
      </div>

      {error && <div style={{ ...errBox, marginTop: 10 }}>{error}</div>}

      <div style={{ display: 'flex', justifyContent: 'space-between', gap: 8, marginTop: 12 }}>
        <button onClick={onBack} style={ghost}>← Change parameters</button>
        <button onClick={onClose} style={primary(true)}>Done</button>
      </div>
    </>
  )
}

/** Compact labelled select for the ordering strip. The tip rides on the
 *  title attribute rather than a caption line — four of these stacked with
 *  captions would push the heatmap off screen. */
function MiniSelect({ label, tip, value, onChange, options, disabled }: {
  label: string
  tip: string
  value: string
  onChange: (v: string) => void
  options: [string, string][]
  disabled?: boolean
}) {
  return (
    <label title={tip} style={{ display: 'flex', alignItems: 'center', gap: 4, fontSize: 11, color: '#888' }}>
      {label}
      <select
        value={value}
        disabled={disabled}
        onChange={(e) => onChange(e.target.value)}
        style={{
          padding: '3px 6px', fontSize: 11, backgroundColor: '#0f3460',
          color: disabled ? '#666' : '#eee', border: '1px solid #1a1a2e', borderRadius: 3,
          cursor: disabled ? 'not-allowed' : 'pointer',
        }}
      >
        {options.map(([v, l]) => <option key={v} value={v}>{l}</option>)}
      </select>
    </label>
  )
}

function toggle(active: boolean): React.CSSProperties {
  return {
    padding: '4px 10px', fontSize: 11, borderRadius: 4, cursor: 'pointer',
    backgroundColor: active ? '#4ecdc4' : 'transparent',
    color: active ? '#16213e' : '#aaa',
    border: `1px solid ${active ? '#4ecdc4' : '#0f3460'}`,
    fontWeight: active ? 600 : 400,
  }
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
