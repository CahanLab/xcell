import React from 'react'

// ---------------------------------------------------------------------------
// Pure helpers — histogram binning and threshold-to-indices.
// Shared by SelectByExpressionModal and the GenePanel highlight overlay.
// ---------------------------------------------------------------------------

export type ThresholdMode = 'above' | 'below' | 'between'

export interface Histogram {
  binEdges: number[]
  counts: number[]
  min: number
  max: number
  zeroVariance: boolean
}

export function computeHistogram(values: Float32Array | number[] | (number | null)[], nBins = 60): Histogram {
  let min = Infinity
  let max = -Infinity
  for (let i = 0; i < values.length; i++) {
    const v = values[i]
    if (v == null) continue
    if (v < min) min = v
    if (v > max) max = v
  }
  if (!isFinite(min) || !isFinite(max)) {
    return { binEdges: [0, 1], counts: new Array(nBins).fill(0), min: 0, max: 1, zeroVariance: true }
  }
  if (min === max) {
    let nonNullCount = 0
    for (let i = 0; i < values.length; i++) {
      if (values[i] != null) nonNullCount++
    }
    return { binEdges: [min, min], counts: [nonNullCount], min, max, zeroVariance: true }
  }
  const width = (max - min) / nBins
  const counts = new Array(nBins).fill(0)
  for (let i = 0; i < values.length; i++) {
    const v = values[i]
    if (v == null) continue
    const idx = Math.min(nBins - 1, Math.floor((v - min) / width))
    counts[idx]++
  }
  const binEdges: number[] = new Array(nBins + 1)
  for (let i = 0; i <= nBins; i++) binEdges[i] = min + i * width
  return { binEdges, counts, min, max, zeroVariance: false }
}

export function matchingIndices(
  values: Float32Array | number[] | (number | null)[],
  mode: ThresholdMode,
  lo: number,
  hi: number
): number[] {
  const out: number[] = []
  for (let i = 0; i < values.length; i++) {
    const v = values[i]
    if (v == null) continue
    if (mode === 'above' && v >= lo) out.push(i)
    else if (mode === 'below' && v <= lo) out.push(i)
    else if (mode === 'between' && v >= lo && v <= hi) out.push(i)
  }
  return out
}

// Build a Uint8Array mask (1 = matches threshold, 0 = doesn't) over the
// full cell array. Same semantics as matchingIndices but mask-shaped — used
// by the highlight overlay so getColor() can do O(1) per-cell lookups.
export function thresholdMask(
  values: Float32Array | number[] | (number | null)[],
  mode: ThresholdMode,
  lo: number,
  hi: number
): Uint8Array {
  const out = new Uint8Array(values.length)
  for (let i = 0; i < values.length; i++) {
    const v = values[i]
    if (v == null) continue
    if (mode === 'above' && v >= lo) out[i] = 1
    else if (mode === 'below' && v <= lo) out[i] = 1
    else if (mode === 'between' && v >= lo && v <= hi) out[i] = 1
  }
  return out
}

// Pick a sensible default threshold for a given mode from observed values.
// For Above/Below: median of non-zero values, falling back to the 25th percentile
// and then to the midpoint of [min, max]. For Between: 25th and 75th percentile.
export function defaultThresholds(
  values: Float32Array | number[] | (number | null)[],
  mode: ThresholdMode,
  min: number,
  max: number
): { lo: number; hi: number } {
  if (values.length === 0 || min === max) {
    return { lo: min, hi: max }
  }
  const sorted = (Array.from(values) as (number | null)[])
    .filter((v): v is number => v != null)
    .sort((a, b) => a - b)
  const percentile = (p: number) =>
    sorted[Math.min(sorted.length - 1, Math.max(0, Math.floor(p * sorted.length)))]
  if (mode === 'between') {
    return { lo: percentile(0.25), hi: percentile(0.75) }
  }
  // above/below
  const nonZero = sorted.filter((v) => v > 0)
  if (nonZero.length > 0) {
    const median = nonZero[Math.floor(nonZero.length / 2)]
    if (median > 0) return { lo: median, hi: median }
  }
  // Next fallback: 25th percentile of all non-null values
  const p25 = sorted[Math.min(sorted.length - 1, Math.max(0, Math.floor(0.25 * sorted.length)))]
  if (p25 !== undefined && p25 !== min) return { lo: p25, hi: p25 }
  // Final fallback: midpoint of [min, max]
  const mid = (min + max) / 2
  return { lo: mid, hi: mid }
}

// ---------------------------------------------------------------------------
// Histogram chart
// ---------------------------------------------------------------------------

const DEFAULT_CHART_WIDTH = 460
const DEFAULT_CHART_HEIGHT = 140
const CHART_PADDING = { top: 6, right: 6, bottom: 22, left: 6 }

export function HistogramChart({
  histogram,
  mode,
  lo,
  hi,
  onChangeLo,
  onChangeHi,
  width = DEFAULT_CHART_WIDTH,
  height = DEFAULT_CHART_HEIGHT,
  barColor = '#4ecdc4',
}: {
  histogram: Histogram
  mode: ThresholdMode
  lo: number
  hi: number
  onChangeLo: (v: number) => void
  onChangeHi: (v: number) => void
  width?: number
  height?: number
  barColor?: string
}) {
  const innerW = width - CHART_PADDING.left - CHART_PADDING.right
  const innerH = height - CHART_PADDING.top - CHART_PADDING.bottom
  const maxCount = Math.max(...histogram.counts, 1)
  const nBins = histogram.counts.length
  const barW = innerW / nBins
  const svgRef = React.useRef<SVGSVGElement | null>(null)

  const valueToX = (v: number) => {
    const clamped = Math.max(histogram.min, Math.min(histogram.max, v))
    const frac = (clamped - histogram.min) / (histogram.max - histogram.min || 1)
    return CHART_PADDING.left + frac * innerW
  }

  const xToValue = (x: number) => {
    const localX = Math.max(0, Math.min(innerW, x - CHART_PADDING.left))
    const frac = localX / innerW
    return histogram.min + frac * (histogram.max - histogram.min)
  }

  const startDrag = (which: 'lo' | 'hi') => (e: React.MouseEvent<SVGRectElement>) => {
    e.preventDefault()
    const svg = svgRef.current
    if (!svg) return
    const onMove = (ev: MouseEvent) => {
      const rect = svg.getBoundingClientRect()
      const x = ev.clientX - rect.left
      const v = xToValue(x)
      if (which === 'lo') onChangeLo(v)
      else onChangeHi(v)
    }
    const onUp = () => {
      document.removeEventListener('mousemove', onMove)
      document.removeEventListener('mouseup', onUp)
    }
    document.addEventListener('mousemove', onMove)
    document.addEventListener('mouseup', onUp)
  }

  const loX = valueToX(lo)
  const hiX = valueToX(hi)

  return (
    <svg ref={svgRef} width={width} height={height} style={{ display: 'block' }}>
      <rect
        x={CHART_PADDING.left}
        y={CHART_PADDING.top}
        width={innerW}
        height={innerH}
        fill="#0f1625"
      />
      {histogram.counts.map((count, i) => {
        const h = (count / maxCount) * innerH
        return (
          <rect
            key={i}
            x={CHART_PADDING.left + i * barW}
            y={CHART_PADDING.top + innerH - h}
            width={Math.max(1, barW - 1)}
            height={h}
            fill={barColor}
          />
        )
      })}

      {mode === 'above' && <CutoffLine x={loX} innerH={innerH} onMouseDown={startDrag('lo')} />}
      {mode === 'below' && <CutoffLine x={loX} innerH={innerH} onMouseDown={startDrag('lo')} />}
      {mode === 'between' && (
        <>
          <CutoffLine x={loX} innerH={innerH} onMouseDown={startDrag('lo')} />
          <CutoffLine x={hiX} innerH={innerH} onMouseDown={startDrag('hi')} />
        </>
      )}

      <text x={CHART_PADDING.left} y={height - 6} fill="#888" fontSize="10">
        {histogram.min.toFixed(2)}
      </text>
      <text
        x={CHART_PADDING.left + innerW}
        y={height - 6}
        fill="#888"
        fontSize="10"
        textAnchor="end"
      >
        {histogram.max.toFixed(2)}
      </text>
    </svg>
  )
}

function CutoffLine({
  x,
  innerH,
  onMouseDown,
}: {
  x: number
  innerH: number
  onMouseDown: (e: React.MouseEvent<SVGRectElement>) => void
}) {
  return (
    <g>
      <line
        x1={x}
        x2={x}
        y1={CHART_PADDING.top}
        y2={CHART_PADDING.top + innerH}
        stroke="#e94560"
        strokeWidth={2}
      />
      <rect
        x={x - 6}
        y={CHART_PADDING.top}
        width={12}
        height={innerH}
        fill="transparent"
        style={{ cursor: 'ew-resize' }}
        onMouseDown={onMouseDown}
      />
    </g>
  )
}
