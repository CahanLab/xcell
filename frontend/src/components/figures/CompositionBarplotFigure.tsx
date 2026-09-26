import { forwardRef, useMemo } from 'react'
import { resolveCategoryPalette } from '../../lib/cellColors'
import { orderBars, stackBar, fitRotatedLabel, type Crosstab } from '../../lib/stackedBars'
import type { BarplotConfigLike } from '../../lib/figures'

/** Stacked composition barplot: one bar per category of column A, split by
 *  the proportions of column B. Shared by the Barplot tab (which adds its own
 *  floating legend and tooltip) and the Figures tab (which asks for an inline
 *  legend so the export is self-contained). SVG, so it exports as vector. */

export interface CompositionBarplotFigureProps {
  data: Crosstab
  config: Omit<BarplotConfigLike, 'columnA' | 'columnB'>
  width: number
  height: number
  title?: string
  legend?: 'inline' | 'none'
  background?: string
  onHover?: (hover: { x: number; y: number; text: string } | null) => void
}

const M = { top: 16, right: 16, bottom: 96, left: 60 }
const BAR_GAP = 0.25          // share of a slot left empty between bars
const CHAR_PX = 5.6           // 10px sans-serif, near enough for truncation
const LABEL_PX = M.bottom * Math.SQRT2 - 12   // diagonal room for a 45° label
const LEGEND_W = 150

export function barplotPalette(data: Crosstab): string[] {
  return resolveCategoryPalette(data.b_categories.length, data.b_colors ?? undefined)
    .map(([r, g, b]) => `rgb(${r},${g},${b})`)
}

export const CompositionBarplotFigure = forwardRef<SVGSVGElement, CompositionBarplotFigureProps>(function CompositionBarplotFigure(
  { data, config, width, height, title, legend = 'none', background, onHover },
  ref,
) {
  const palette = useMemo(() => barplotPalette(data), [data])
  const bars = useMemo(() => orderBars(data, { by: config.order, shareOf: config.shareOf, minCells: config.minCells }), [data, config.order, config.shareOf, config.minCells])
  const scaleMax = useMemo(() => Math.max(1, ...bars.map((i) => data.counts[i].reduce((s, n) => s + n, 0))), [data, bars])

  const titleH = title ? 22 : 0
  const legendW = legend === 'inline' ? LEGEND_W : 0
  const plotW = Math.max(40, width - M.left - M.right - legendW)
  const plotH = Math.max(40, height - M.top - M.bottom - titleH)
  const top = M.top + titleH
  const slot = bars.length > 0 ? plotW / bars.length : plotW
  const barW = slot * (1 - BAR_GAP)

  return (
    <svg ref={ref} data-figure-svg width={width} height={height} style={{ display: 'block', background }}>
      {title && <text x={12} y={15} fill="#eee" fontSize={13} fontWeight={600} fontFamily="sans-serif">{title}</text>}
      {(config.normalize ? [0, 0.25, 0.5, 0.75, 1] : [0, 0.5, 1]).map((t) => {
        const y = top + plotH * (1 - t)
        return (
          <g key={t}>
            <line x1={M.left} x2={M.left + plotW} y1={y} y2={y} stroke="#0f3460" strokeWidth={1} />
            <text x={M.left - 8} y={y + 3} textAnchor="end" fill="#888" fontSize={10} fontFamily="sans-serif">
              {config.normalize ? `${Math.round(t * 100)}%` : Math.round(t * scaleMax)}
            </text>
          </g>
        )
      })}
      <text transform={`translate(14 ${top + plotH / 2}) rotate(-90)`} textAnchor="middle" fill="#888" fontSize={11} fontFamily="sans-serif">
        {config.normalize ? 'proportion of cells' : 'cells'}
      </text>
      {bars.map((rowIdx, pos) => {
        const x = M.left + pos * slot + (slot - barW) / 2
        const total = data.counts[rowIdx].reduce((s, n) => s + n, 0)
        const segs = stackBar(data.counts[rowIdx], { normalize: config.normalize, scaleMax })
        const name = data.a_categories[rowIdx]
        const labelText = fitRotatedLabel(name, LABEL_PX, CHAR_PX)
        return (
          <g key={name}>
            {segs.map((seg) => {
              const h = seg.fraction * plotH
              const y = top + plotH - (seg.start + seg.fraction) * plotH
              const pct = (seg.count / total) * 100
              return (
                <rect
                  key={seg.index} x={x} y={y} width={barW} height={Math.max(0.5, h)} fill={palette[seg.index]}
                  onMouseMove={onHover ? (e) => onHover({ x: e.clientX, y: e.clientY, text: `${name} · ${data.b_categories[seg.index]}: ${seg.count.toLocaleString()} cells (${pct.toFixed(1)}%)` }) : undefined}
                  onMouseLeave={onHover ? () => onHover(null) : undefined}
                >
                  {!onHover && <title>{`${name} · ${data.b_categories[seg.index]}: ${seg.count.toLocaleString()} cells (${pct.toFixed(1)}%)`}</title>}
                </rect>
              )
            })}
            {config.showValues && (
              <text x={x + barW / 2} y={top + plotH - (config.normalize ? plotH : (total / scaleMax) * plotH) - 4} textAnchor="middle" fill="#888" fontSize={9} fontFamily="sans-serif">
                {total.toLocaleString()}
              </text>
            )}
            {/* Category label: rotated 45° and ending at the bar's centre, so it points at the bar it names. */}
            {labelText && (
              <text transform={`translate(${x + barW / 2} ${top + plotH + 8}) rotate(-45)`} textAnchor="end" fill="#ccc" fontSize={10} fontFamily="sans-serif">
                <title>{name}</title>
                {labelText}
              </text>
            )}
          </g>
        )
      })}
      <line x1={M.left} x2={M.left + plotW} y1={top + plotH} y2={top + plotH} stroke="#0f3460" strokeWidth={1} />
      {legend === 'inline' && (
        <g transform={`translate(${M.left + plotW + 16} ${top})`}>
          <text x={0} y={10} fill="#aaa" fontSize={10} fontWeight={600} fontFamily="sans-serif">{data.b}</text>
          {data.b_categories.map((cat, i) => (
            <g key={cat} transform={`translate(0 ${20 + i * 14})`}>
              <rect x={0} y={0} width={10} height={10} fill={palette[i]} />
              <text x={14} y={9} fill="#ccc" fontSize={10} fontFamily="sans-serif">{fitRotatedLabel(cat, LEGEND_W - 20, CHAR_PX)}</text>
            </g>
          ))}
        </g>
      )}
    </svg>
  )
})

export default CompositionBarplotFigure
