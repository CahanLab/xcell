/** The CNV heatmap canvas (Figures tab): rows are cells binned within their
 *  group, or one mean row per group; columns are genome windows in karyotype
 *  order with chromosome separators. Diverging ramp centred on 0, so a gain
 *  is warm and a loss cool whatever the colormap. Canvas, not SVG: hundreds
 *  of rows × thousands of windows. Exports as PNG. */

import { useState, useEffect, useRef, useCallback, forwardRef } from 'react'
import { divergingColor, type CnvHeatmapData, type CanvasLegend } from '../../lib/figures'

const RAMP_STOPS: Record<string, string[]> = {
  rdbu: ['rgb(5,48,97)', 'rgb(103,169,207)', 'rgb(247,247,247)', 'rgb(239,138,98)', 'rgb(103,0,31)'],
  prgn: ['rgb(64,0,75)', 'rgb(175,141,195)', 'rgb(247,247,247)', 'rgb(127,191,123)', 'rgb(0,68,27)'],
  roma: ['rgb(126,23,0)', 'rgb(219,165,95)', 'rgb(247,247,247)', 'rgb(96,176,176)', 'rgb(1,51,80)'],
}

const GROUP_COLORS = [
  '#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd',
  '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf',
]

const LABEL_WIDTH = 96
const GROUP_BAR_WIDTH = 8
const CHROM_LABEL_HEIGHT = 22
const MIN_ROW_HEIGHT = 2
const MIN_CELL_WIDTH = 1.5
const MAX_ROW_HEIGHT = 14
const MAX_HEIGHT = 700

function uniqueGroups(data: CnvHeatmapData): string[] {
  const out: string[] = []
  for (const g of data.row_groups) if (g && !out.includes(g)) out.push(g)
  return out
}

const fmtVmax = (v: number) => (v >= 1 ? v.toFixed(1) : v.toPrecision(2))

/** The legend a PNG export should carry for this data. */
export function cnvHeatmapLegendFor(data: CnvHeatmapData): CanvasLegend {
  return {
    gradient: RAMP_STOPS[data.colormap] ?? RAMP_STOPS.rdbu,
    gradientLabel: 'CNV (log expression vs reference)',
    gradientRange: `−${fmtVmax(data.vmax)} … +${fmtVmax(data.vmax)}`,
    groups: uniqueGroups(data).map((name, i) => ({ name, color: GROUP_COLORS[i % GROUP_COLORS.length] })),
  }
}

export interface CnvHeatmapFigureProps {
  data: CnvHeatmapData
  legends?: 'inline' | 'none'
}

export const CnvHeatmapFigure = forwardRef<HTMLCanvasElement, CnvHeatmapFigureProps>(function CnvHeatmapFigure(
  { data, legends = 'inline' },
  ref,
) {
  const canvasRef = useRef<HTMLCanvasElement | null>(null)
  const setCanvas = useCallback((el: HTMLCanvasElement | null) => {
    canvasRef.current = el
    if (typeof ref === 'function') ref(el)
    else if (ref) (ref as React.MutableRefObject<HTMLCanvasElement | null>).current = el
  }, [ref])
  const containerRef = useRef<HTMLDivElement>(null)
  const [containerWidth, setContainerWidth] = useState(800)
  useEffect(() => {
    const el = containerRef.current
    if (!el) return
    const ro = new ResizeObserver(() => setContainerWidth(el.clientWidth || 800))
    ro.observe(el)
    setContainerWidth(el.clientWidth || 800)
    return () => ro.disconnect()
  }, [])
  const [tooltip, setTooltip] = useState<{ x: number; y: number; text: string } | null>(null)

  const nRows = data.matrix.length
  const nCols = nRows > 0 ? data.matrix[0].length : 0
  const groups = uniqueGroups(data)
  const hasGroups = groups.length > 0
  const left = LABEL_WIDTH + (hasGroups ? GROUP_BAR_WIDTH + 2 : 0)
  // Fill the container, but never below MIN_CELL_WIDTH per window: a narrow
  // pane scrolls sideways rather than collapsing 500 windows into 100 px.
  const cellWidth = nCols ? Math.max(MIN_CELL_WIDTH, (containerWidth - left - 20) / nCols) : 1
  const heatmapWidth = cellWidth * nCols
  // Rows shrink to fit a tall matrix; a 400-row heatmap is still readable at 2 px.
  const rowHeight = nRows ? Math.max(MIN_ROW_HEIGHT, Math.min(MAX_ROW_HEIGHT, Math.floor(MAX_HEIGHT / nRows))) : MAX_ROW_HEIGHT
  const matrixHeight = nRows * rowHeight
  const width = left + heatmapWidth
  const height = matrixHeight + CHROM_LABEL_HEIGHT
  const showRowLabels = rowHeight >= 9

  useEffect(() => {
    const canvas = canvasRef.current
    if (!canvas || nRows === 0 || nCols === 0) return
    const dpr = window.devicePixelRatio || 1
    canvas.width = width * dpr
    canvas.height = height * dpr
    canvas.style.width = `${width}px`
    canvas.style.height = `${height}px`
    const ctx = canvas.getContext('2d')
    if (!ctx) return
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0)
    ctx.fillStyle = '#1a1a2e'
    ctx.fillRect(0, 0, width, height)

    // Cells. Pixel-aligned columns: with thousands of windows a column is
    // under a pixel wide, so each is drawn from its rounded left edge to the
    // next column's, never leaving hairline gaps.
    for (let r = 0; r < nRows; r++) {
      const row = data.matrix[r]
      const y = r * rowHeight
      for (let c = 0; c < nCols; c++) {
        const x0 = left + Math.floor(c * cellWidth)
        const x1 = left + Math.floor((c + 1) * cellWidth)
        ctx.fillStyle = divergingColor(row[c], data.vmax, data.colormap)
        ctx.fillRect(x0, y, Math.max(1, x1 - x0), rowHeight)
      }
    }

    // Row labels and group bar.
    if (showRowLabels) {
      ctx.font = '10px monospace'
      ctx.textBaseline = 'middle'
      ctx.fillStyle = '#ccc'
      for (let r = 0; r < nRows; r++) {
        const label = data.row_labels[r]
        ctx.fillText(label.length > 14 ? `${label.slice(0, 13)}…` : label, 4, r * rowHeight + rowHeight / 2)
      }
    }
    if (hasGroups) {
      let prev: string | null = null
      let runStart = 0
      const flush = (end: number) => {
        if (prev == null) return
        const gi = groups.indexOf(prev)
        ctx.fillStyle = GROUP_COLORS[gi % GROUP_COLORS.length]
        ctx.fillRect(LABEL_WIDTH, runStart * rowHeight, GROUP_BAR_WIDTH, (end - runStart) * rowHeight)
        // When rows are too thin for per-row labels, name the group once beside its bar.
        if (!showRowLabels && (end - runStart) * rowHeight >= 10) {
          ctx.font = '10px sans-serif'
          ctx.textBaseline = 'middle'
          ctx.fillStyle = '#ccc'
          const label = prev.length > 13 ? `${prev.slice(0, 12)}…` : prev
          ctx.fillText(label, 4, ((runStart + end) / 2) * rowHeight)
        }
      }
      for (let r = 0; r < nRows; r++) {
        const g = data.row_groups[r]
        if (g !== prev) {
          flush(r)
          if (r > 0) {
            ctx.strokeStyle = 'rgba(255,255,255,0.7)'
            ctx.lineWidth = 1
            ctx.beginPath(); ctx.moveTo(LABEL_WIDTH, r * rowHeight + 0.5); ctx.lineTo(width, r * rowHeight + 0.5); ctx.stroke()
          }
          prev = g
          runStart = r
        }
      }
      flush(nRows)
    }

    // Chromosome separators and labels.
    ctx.font = '10px sans-serif'
    ctx.textBaseline = 'top'
    ctx.textAlign = 'center'
    for (const grp of data.column_groups) {
      const x = left + Math.floor(grp.start * cellWidth)
      const w = grp.size * cellWidth
      if (grp.start > 0) {
        ctx.strokeStyle = '#1a1a2e'
        ctx.lineWidth = 1
        ctx.beginPath(); ctx.moveTo(x + 0.5, 0); ctx.lineTo(x + 0.5, matrixHeight); ctx.stroke()
      }
      const name = grp.name.replace(/^chr/, '')
      if (w >= name.length * 6.5) {
        ctx.fillStyle = '#ccc'
        ctx.fillText(name, x + w / 2, matrixHeight + 5)
      }
    }
    ctx.textAlign = 'left'
    ctx.textBaseline = 'alphabetic'
  }, [data, nRows, nCols, width, height, left, cellWidth, rowHeight, matrixHeight, hasGroups, groups, showRowLabels])

  const handleMouseMove = useCallback((e: React.MouseEvent) => {
    const canvas = canvasRef.current
    if (!canvas) return
    const rect = canvas.getBoundingClientRect()
    const x = e.clientX - rect.left
    const y = e.clientY - rect.top
    const row = Math.floor(y / rowHeight)
    const col = Math.floor((x - left) / cellWidth)
    if (row >= 0 && row < nRows && col >= 0 && col < nCols) {
      const chrom = data.column_groups.find((g) => col >= g.start && col < g.start + g.size)
      const val = data.matrix[row][col]
      setTooltip({ x: e.clientX, y: e.clientY, text: `${data.row_labels[row]} | ${chrom?.name ?? `window ${col + 1}`} | ${val >= 0 ? '+' : ''}${val.toFixed(3)}` })
    } else {
      setTooltip(null)
    }
  }, [data, nRows, nCols, cellWidth, left, rowHeight])

  const stops = RAMP_STOPS[data.colormap] ?? RAMP_STOPS.rdbu

  return (
    <div ref={containerRef} style={{ flex: 1, overflow: 'auto', padding: '8px', position: 'relative' }}>
      <canvas ref={setCanvas} data-figure-canvas onMouseMove={handleMouseMove} onMouseLeave={() => setTooltip(null)} />
      {tooltip && (
        <div style={{
          position: 'fixed', left: tooltip.x + 12, top: tooltip.y - 8, padding: '4px 8px', backgroundColor: 'rgba(0,0,0,0.85)',
          color: '#eee', fontSize: '11px', fontFamily: 'monospace', borderRadius: '4px', pointerEvents: 'none', zIndex: 1000, whiteSpace: 'nowrap',
        }}>
          {tooltip.text}
        </div>
      )}
      {legends === 'inline' && (
        <div style={{ display: 'flex', gap: '18px', alignItems: 'flex-start', padding: '6px 2px', flexWrap: 'wrap' }}>
          <div>
            <div style={{ fontSize: '10px', color: '#aaa', marginBottom: '2px' }}>CNV (log expression vs reference)</div>
            <div style={{ width: '120px', height: '10px', borderRadius: '2px', background: `linear-gradient(to right, ${stops.join(', ')})` }} />
            <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: '9px', color: '#888', marginTop: '2px' }}>
              <span>−{fmtVmax(data.vmax)}</span><span>0</span><span>+{fmtVmax(data.vmax)}</span>
            </div>
            <div style={{ fontSize: '9px', color: '#777', marginTop: '2px' }}>{data.n_cells.toLocaleString()} cells in {nRows} rows; {nCols} windows</div>
          </div>
          {hasGroups && (
            <div>
              <div style={{ fontSize: '10px', color: '#aaa', marginBottom: '2px' }}>Groups</div>
              {groups.map((g, i) => (
                <div key={g} style={{ display: 'flex', alignItems: 'center', gap: '6px', marginBottom: '2px' }}>
                  <div style={{ width: '10px', height: '10px', borderRadius: '2px', backgroundColor: GROUP_COLORS[i % GROUP_COLORS.length] }} />
                  <span style={{ fontSize: '10px', color: '#aaa' }}>{g}</span>
                </div>
              ))}
            </div>
          )}
        </div>
      )}
    </div>
  )
})

export default CnvHeatmapFigure
