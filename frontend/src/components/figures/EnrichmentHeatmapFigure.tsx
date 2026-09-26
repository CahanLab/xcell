import { forwardRef, useMemo } from 'react'
import { divergingColor, truncate, type EnrichmentHeatmapData } from '../../lib/figures'

/** Gene sets × groups heatmap (PySingleCellNet convention): a diverging
 *  colour centred on 0, rows the union of each column's top sets. Pure SVG so
 *  it exports as vector; the root carries data-figure-svg for the exporter. */

export interface EnrichmentHeatmapFigureProps {
  data: EnrichmentHeatmapData
  params: Record<string, unknown>
  title?: string
  width?: number
  background?: string
}

const FONT = 'Inter, system-ui, sans-serif'

export const EnrichmentHeatmapFigure = forwardRef<SVGSVGElement, EnrichmentHeatmapFigureProps>(function EnrichmentHeatmapFigure(
  { data, params, title, width, background = '#16213e' },
  ref,
) {
  const cell = Math.max(6, Number(params.cell_size ?? 14))
  const labelMax = Math.max(8, Number(params.label_max_chars ?? 40))
  const colormap = String(params.colormap ?? 'rdbu')
  const showValues = Boolean(params.show_values)
  const fontSize = Math.max(7, Math.min(12, cell * 0.7))

  const vmax = useMemo(() => {
    const p = params.vmax
    if (typeof p === 'number' && p > 0) return p
    let m = 0
    for (const row of data.values) for (const v of row) m = Math.max(m, Math.abs(v))
    return m || 1
  }, [data.values, params.vmax])

  const rowLabels = data.rows.map((r) => `${truncate(r.name, labelMax)}${r.members.length ? ` (+${r.members.length})` : ''}`)
  const colLabels = data.cols.map((c) => truncate(c.label, 24))
  const rowLabelW = Math.min(360, 8 + Math.max(0, ...rowLabels.map((l) => l.length)) * fontSize * 0.62)
  const colLabelH = Math.min(160, 8 + Math.max(0, ...colLabels.map((l) => l.length)) * fontSize * 0.55)
  const titleH = title ? 24 : 0
  const legendH = 44
  const gridW = data.cols.length * cell
  const gridH = data.rows.length * cell
  const left = rowLabelW + 8
  const top = titleH + colLabelH + 8
  const svgW = Math.max(width ?? 0, left + gridW + 24)
  const svgH = top + gridH + legendH + 16

  if (data.rows.length === 0) {
    return (
      <svg ref={ref} data-figure-svg width={Math.max(width ?? 480, 480)} height={80} style={{ background, fontFamily: FONT }}>
        {title && <text x={12} y={20} fill="#eee" fontSize={13} fontWeight={600}>{title}</text>}
        <text x={12} y={titleH + 30} fill="#aaa" fontSize={12}>{data.note ?? 'Nothing to draw'}</text>
      </svg>
    )
  }

  const legendW = 160
  const legendX = left
  const legendY = top + gridH + 14
  const stops = [-1, -0.5, 0, 0.5, 1]

  return (
    <svg ref={ref} data-figure-svg width={svgW} height={svgH} style={{ background, fontFamily: FONT, display: 'block' }}>
      {title && <text x={12} y={17} fill="#eee" fontSize={13} fontWeight={600}>{title}</text>}
      {/* column labels, rotated */}
      {colLabels.map((label, j) => (
        <text
          key={j} fill="#ccc" fontSize={fontSize}
          transform={`translate(${left + j * cell + cell / 2 + fontSize * 0.35}, ${top - 6}) rotate(-45)`}
          textAnchor="start"
        >
          {label}
        </text>
      ))}
      {/* row labels */}
      {rowLabels.map((label, i) => (
        <text key={i} x={left - 6} y={top + i * cell + cell / 2 + fontSize * 0.35} fill="#ccc" fontSize={fontSize} textAnchor="end">
          <title>{[data.rows[i].name, data.rows[i].library, ...(data.rows[i].members.length ? ['also: ' + data.rows[i].members.join(', ')] : [])].filter(Boolean).join(' · ')}</title>
          {label}
        </text>
      ))}
      {/* cells */}
      {data.values.map((row, i) => row.map((v, j) => (
        <g key={`${i}-${j}`}>
          <rect x={left + j * cell} y={top + i * cell} width={cell} height={cell} fill={divergingColor(v, vmax, colormap)} stroke={background} strokeWidth={0.5}>
            <title>{`${data.rows[i].name} × ${data.cols[j].label}: ${data.value_label} ${v.toFixed(2)}, padj ${data.padj[i][j].toExponential(1)}`}</title>
          </rect>
          {showValues && v !== 0 && (
            <text x={left + j * cell + cell / 2} y={top + i * cell + cell / 2 + fontSize * 0.3} fill={Math.abs(v) / vmax > 0.6 ? '#fff' : '#222'} fontSize={Math.max(6, fontSize - 2)} textAnchor="middle">
              {Math.abs(v) >= 10 ? v.toFixed(0) : v.toFixed(1)}
            </text>
          )}
        </g>
      )))}
      {/* legend */}
      <defs>
        <linearGradient id="enrich-heatmap-legend" x1="0" x2="1" y1="0" y2="0">
          {stops.map((s, k) => <stop key={k} offset={`${(k / (stops.length - 1)) * 100}%`} stopColor={divergingColor(s * vmax, vmax, colormap)} />)}
        </linearGradient>
      </defs>
      <rect x={legendX} y={legendY} width={legendW} height={10} fill="url(#enrich-heatmap-legend)" stroke="#0f3460" strokeWidth={0.5} />
      <text x={legendX} y={legendY + 22} fill="#aaa" fontSize={9}>−{vmax.toFixed(1)}</text>
      <text x={legendX + legendW / 2} y={legendY + 22} fill="#aaa" fontSize={9} textAnchor="middle">0</text>
      <text x={legendX + legendW} y={legendY + 22} fill="#aaa" fontSize={9} textAnchor="end">{vmax.toFixed(1)}</text>
      <text x={legendX + legendW + 10} y={legendY + 9} fill="#ccc" fontSize={10}>{data.value_label}{data.n_collapsed ? ` · ${data.n_collapsed} rows collapsed` : ''}</text>
      {data.note && <text x={legendX} y={legendY + 36} fill="#e9a23b" fontSize={9}>{data.note}</text>}
    </svg>
  )
})

export default EnrichmentHeatmapFigure
