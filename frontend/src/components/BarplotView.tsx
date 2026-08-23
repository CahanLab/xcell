/**
 * BarplotView — stacked composition barplot for the center panel.
 *
 * One bar per category of column A, split by the proportion of those cells
 * carrying each category of column B. "What is each cluster made of?"
 *
 * SVG rather than canvas: the plot is tens of rectangles, not tens of
 * thousands, and it exports as a figure without a second rendering path.
 */

import { useCallback, useEffect, useMemo, useRef, useState } from 'react'
import { useStore } from '../store'
import { appendDataset } from '../hooks/useData'
import { resolveCategoryPalette } from '../lib/cellColors'
import { standaloneSvg, downloadText } from '../lib/svgExport'
import { orderBars, stackBar, fitRotatedLabel, type Crosstab } from '../lib/stackedBars'
import { FloatingPanel } from './PlotLegends'
import BarplotConfigModal from './BarplotConfigModal'

const M = { top: 16, right: 16, bottom: 96, left: 60 }
const BAR_GAP = 0.25          // share of a slot left empty between bars
const CHAR_PX = 5.6           // 10px sans-serif, near enough for truncation
const LABEL_PX = M.bottom * Math.SQRT2 - 12   // diagonal room for a 45° label

export default function BarplotView() {
  const config = useStore((s) => s.barplotConfig)
  const setConfig = useStore((s) => s.setBarplotConfig)
  const activeSlot = useStore((s) => s.activeSlot)
  const schema = useStore((s) => s.schema)

  const [data, setData] = useState<Crosstab | null>(null)
  const [error, setError] = useState<string | null>(null)
  const [loading, setLoading] = useState(false)
  const [configOpen, setConfigOpen] = useState(false)
  const [hover, setHover] = useState<{ x: number; y: number; text: string } | null>(null)
  const [size, setSize] = useState({ w: 900, h: 520 })
  const boxRef = useRef<HTMLDivElement>(null)
  const svgRef = useRef<SVGSVGElement>(null)

  // Refetch whenever the columns or the dataset change. Not on the ordering
  // knobs — those rearrange what is already here.
  const colA = config?.columnA
  const colB = config?.columnB
  useEffect(() => {
    if (!colA || !colB) { setData(null); return }
    let stale = false
    setLoading(true)
    setError(null)
    fetch(appendDataset(`/api/obs/crosstab?a=${encodeURIComponent(colA)}&b=${encodeURIComponent(colB)}`))
      .then(async (r) => {
        const body = await r.json()
        if (!r.ok) throw new Error(body.detail || `HTTP ${r.status}`)
        return body as Crosstab
      })
      .then((d) => { if (!stale) setData(d) })
      .catch((e) => { if (!stale) { setError((e as Error).message); setData(null) } })
      .finally(() => { if (!stale) setLoading(false) })
    return () => { stale = true }
  }, [colA, colB, activeSlot])

  // Depends on `config` because the plot box does not exist until there is one
  // — before that the empty state renders instead, so a mount-only effect found
  // a null ref and the plot kept its default size forever.
  useEffect(() => {
    const el = boxRef.current
    if (!el) return
    const ro = new ResizeObserver(([entry]) => {
      const { width, height } = entry.contentRect
      setSize({ w: Math.max(320, width - 18), h: Math.max(240, height - 18) })
    })
    ro.observe(el)
    return () => ro.disconnect()
  }, [config])

  const palette = useMemo(() => {
    if (!data) return []
    return resolveCategoryPalette(data.b_categories.length, data.b_colors ?? undefined)
      .map(([r, g, b]) => `rgb(${r},${g},${b})`)
  }, [data])

  const bars = useMemo(() => {
    if (!data || !config) return []
    return orderBars(data, {
      by: config.order, shareOf: config.shareOf, minCells: config.minCells,
    })
  }, [data, config])

  const scaleMax = useMemo(() => {
    if (!data) return 1
    return Math.max(1, ...bars.map((i) => data.counts[i].reduce((s, n) => s + n, 0)))
  }, [data, bars])

  const exportSvg = useCallback(() => {
    const svg = svgRef.current
    if (!svg) return
    downloadText(
      `${config?.columnA}_by_${config?.columnB}.svg`,
      standaloneSvg(svg.outerHTML, { background: '#1a1a2e' }),
      'image/svg+xml',
    )
  }, [config])

  if (!config) {
    return (
      <div style={styles.empty}>
        <div style={{ marginBottom: 12 }}>
          A stacked barplot shows what each group is made of — one bar per
          category of one column, split by the proportions of another.
        </div>
        <button style={styles.primaryBtn} onClick={() => setConfigOpen(true)}>
          Choose columns…
        </button>
        {configOpen && (
          <BarplotConfigModal
            schema={schema}
            initial={null}
            onClose={() => setConfigOpen(false)}
            onApply={(c) => { setConfig(c); setConfigOpen(false) }}
          />
        )}
      </div>
    )
  }

  const plotW = Math.max(40, size.w - M.left - M.right)
  const plotH = Math.max(40, size.h - M.top - M.bottom)
  const slot = bars.length > 0 ? plotW / bars.length : plotW
  const barW = slot * (1 - BAR_GAP)

  return (
    <div style={styles.wrap}>
      <div style={styles.toolbar}>
        <span style={styles.title}>
          {config.columnA} <span style={{ color: '#666' }}>split by</span> {config.columnB}
        </span>
        <span style={styles.meta}>
          {bars.length} {bars.length === 1 ? 'bar' : 'bars'}
          {data && bars.length < data.a_categories.length
            && ` · ${data.a_categories.length - bars.length} below ${config.minCells} cells`}
          {config.normalize ? ' · proportion' : ' · cell count'}
        </span>
        <div style={{ flex: 1 }} />
        <button style={styles.btn} onClick={() => setConfigOpen(true)}>Columns…</button>
        <button style={styles.btn} onClick={exportSvg} disabled={!data}>Export SVG</button>
      </div>

      <div ref={boxRef} style={styles.canvasBox}>
        {loading && <div style={styles.note}>Counting…</div>}
        {error && <div style={styles.error}>{error}</div>}
        {data && !error && (
          <svg ref={svgRef} width={size.w} height={size.h} style={{ display: 'block' }}>
            {/* y axis */}
            {(config.normalize ? [0, 0.25, 0.5, 0.75, 1] : [0, 0.5, 1]).map((t) => {
              const y = M.top + plotH * (1 - t)
              return (
                <g key={t}>
                  <line x1={M.left} x2={M.left + plotW} y1={y} y2={y}
                        stroke="#0f3460" strokeWidth={1} />
                  <text x={M.left - 8} y={y + 3} textAnchor="end"
                        fill="#888" fontSize={10} fontFamily="sans-serif">
                    {config.normalize ? `${Math.round(t * 100)}%` : Math.round(t * scaleMax)}
                  </text>
                </g>
              )
            })}
            <text
              transform={`translate(14 ${M.top + plotH / 2}) rotate(-90)`}
              textAnchor="middle" fill="#888" fontSize={11} fontFamily="sans-serif"
            >
              {config.normalize ? 'proportion of cells' : 'cells'}
            </text>

            {bars.map((rowIdx, pos) => {
              const x = M.left + pos * slot + (slot - barW) / 2
              const total = data.counts[rowIdx].reduce((s, n) => s + n, 0)
              const segs = stackBar(data.counts[rowIdx], {
                normalize: config.normalize, scaleMax,
              })
              const name = data.a_categories[rowIdx]
              const labelText = fitRotatedLabel(name, LABEL_PX, CHAR_PX)
              return (
                <g key={name}>
                  {segs.map((seg) => {
                    const h = seg.fraction * plotH
                    const y = M.top + plotH - (seg.start + seg.fraction) * plotH
                    const pct = (seg.count / total) * 100
                    return (
                      <rect
                        key={seg.index}
                        x={x} y={y} width={barW} height={Math.max(0.5, h)}
                        fill={palette[seg.index]}
                        onMouseMove={(e) => setHover({
                          x: e.clientX, y: e.clientY,
                          text: `${name} · ${data.b_categories[seg.index]}: `
                            + `${seg.count.toLocaleString()} cells (${pct.toFixed(1)}%)`,
                        })}
                        onMouseLeave={() => setHover(null)}
                      />
                    )
                  })}
                  {config.showValues && (
                    <text x={x + barW / 2} y={M.top + plotH - (config.normalize ? plotH : (total / scaleMax) * plotH) - 4}
                          textAnchor="middle" fill="#888" fontSize={9} fontFamily="sans-serif">
                      {total.toLocaleString()}
                    </text>
                  )}
                  {/* Category label: rotated 45° and ending at the bar's centre,
                      so it points at the bar it names instead of the next one. */}
                  {labelText && (
                    <text
                      transform={`translate(${x + barW / 2} ${M.top + plotH + 8}) rotate(-45)`}
                      textAnchor="end" fill="#ccc" fontSize={10} fontFamily="sans-serif"
                    >
                      <title>{name}</title>
                      {labelText}
                    </text>
                  )}
                </g>
              )
            })}
            <line x1={M.left} x2={M.left + plotW} y1={M.top + plotH} y2={M.top + plotH}
                  stroke="#0f3460" strokeWidth={1} />
          </svg>
        )}

        {data && !error && (
          <FloatingPanel id="barplot-legend" title={config.columnB}>
            {data.b_categories.map((cat, i) => (
              <div key={cat} style={styles.legendRow}>
                <div style={{ ...styles.swatch, backgroundColor: palette[i] }} />
                <span>{cat}</span>
              </div>
            ))}
          </FloatingPanel>
        )}

        {hover && (
          <div style={{ ...styles.tooltip, left: hover.x + 12, top: hover.y - 28 }}>
            {hover.text}
          </div>
        )}
      </div>

      {configOpen && (
        <BarplotConfigModal
          schema={schema}
          initial={config}
          categoriesOfB={data?.b_categories ?? []}
          onClose={() => setConfigOpen(false)}
          onApply={(c) => { setConfig(c); setConfigOpen(false) }}
        />
      )}
    </div>
  )
}

const styles = {
  wrap: { display: 'flex', flexDirection: 'column' as const, height: '100%', overflow: 'hidden' },
  toolbar: {
    display: 'flex', alignItems: 'center', gap: '10px',
    padding: '6px 10px', borderBottom: '1px solid #0f3460', backgroundColor: '#16213e',
  },
  title: { fontSize: '12px', color: '#eee' },
  meta: { fontSize: '10px', color: '#888' },
  btn: {
    padding: '4px 9px', fontSize: '11px', color: '#aaa',
    backgroundColor: '#0f3460', border: '1px solid #1a1a2e',
    borderRadius: '4px', cursor: 'pointer',
  },
  primaryBtn: {
    padding: '6px 14px', fontSize: '12px', color: '#000',
    backgroundColor: '#4ecdc4', border: 'none', borderRadius: '4px', cursor: 'pointer',
  },
  canvasBox: { flex: 1, position: 'relative' as const, overflow: 'auto', padding: '8px' },
  empty: {
    display: 'flex', flexDirection: 'column' as const, alignItems: 'center',
    justifyContent: 'center', height: '100%', maxWidth: '460px', margin: '0 auto',
    textAlign: 'center' as const, fontSize: '12px', color: '#888', lineHeight: 1.6,
  },
  note: { padding: '12px', fontSize: '12px', color: '#888' },
  error: { padding: '12px', fontSize: '12px', color: '#e94560' },
  legendRow: {
    display: 'flex', alignItems: 'center', gap: '8px',
    fontSize: '11px', color: '#ccc', marginBottom: '3px',
  },
  swatch: { width: '11px', height: '11px', borderRadius: '2px', flex: '0 0 auto' },
  tooltip: {
    position: 'fixed' as const, pointerEvents: 'none' as const, zIndex: 30,
    padding: '4px 8px', fontSize: '11px', color: '#eee',
    backgroundColor: 'rgba(15, 22, 37, 0.95)', border: '1px solid #0f3460',
    borderRadius: '4px', whiteSpace: 'nowrap' as const,
  },
}
