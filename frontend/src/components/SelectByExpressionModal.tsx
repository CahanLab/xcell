import React, { useEffect, useMemo, useState } from 'react'
import { MESSAGES } from '../messages'
import { useStore } from '../store'
import {
  createAnnotation,
  addLabelToAnnotation,
  labelCells,
  useDataActions,
  useObsSummaries,
} from '../hooks/useData'

// ---------------------------------------------------------------------------
// Pure helpers — histogram binning and threshold-to-indices.
// Exported for future unit tests; not used outside this file currently.
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

const CHART_WIDTH = 460
const CHART_HEIGHT = 140
const CHART_PADDING = { top: 6, right: 6, bottom: 22, left: 6 }

function HistogramChart({
  histogram,
  mode,
  lo,
  hi,
  onChangeLo,
  onChangeHi,
}: {
  histogram: Histogram
  mode: ThresholdMode
  lo: number
  hi: number
  onChangeLo: (v: number) => void
  onChangeHi: (v: number) => void
}) {
  const innerW = CHART_WIDTH - CHART_PADDING.left - CHART_PADDING.right
  const innerH = CHART_HEIGHT - CHART_PADDING.top - CHART_PADDING.bottom
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
    <svg ref={svgRef} width={CHART_WIDTH} height={CHART_HEIGHT} style={{ display: 'block' }}>
      {/* Background */}
      <rect
        x={CHART_PADDING.left}
        y={CHART_PADDING.top}
        width={innerW}
        height={innerH}
        fill="#0f1625"
      />
      {/* Bars */}
      {histogram.counts.map((count, i) => {
        const h = (count / maxCount) * innerH
        return (
          <rect
            key={i}
            x={CHART_PADDING.left + i * barW}
            y={CHART_PADDING.top + innerH - h}
            width={Math.max(1, barW - 1)}
            height={h}
            fill="#4ecdc4"
          />
        )
      })}

      {/* Cutoff line(s) */}
      {mode === 'above' && <CutoffLine x={loX} innerH={innerH} onMouseDown={startDrag('lo')} />}
      {mode === 'below' && <CutoffLine x={loX} innerH={innerH} onMouseDown={startDrag('lo')} />}
      {mode === 'between' && (
        <>
          <CutoffLine x={loX} innerH={innerH} onMouseDown={startDrag('lo')} />
          <CutoffLine x={hiX} innerH={innerH} onMouseDown={startDrag('hi')} />
        </>
      )}

      {/* X axis labels */}
      <text x={CHART_PADDING.left} y={CHART_HEIGHT - 6} fill="#888" fontSize="10">
        {histogram.min.toFixed(2)}
      </text>
      <text
        x={CHART_PADDING.left + innerW}
        y={CHART_HEIGHT - 6}
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
      {/* Visible line */}
      <line
        x1={x}
        x2={x}
        y1={CHART_PADDING.top}
        y2={CHART_PADDING.top + innerH}
        stroke="#e94560"
        strokeWidth={2}
      />
      {/* Wide invisible hit-target for drag */}
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

// ---------------------------------------------------------------------------
// Component
// ---------------------------------------------------------------------------

export default function SelectByExpressionModal() {
  const source = useStore((s) => s.selectByExpressionSource)
  const setSource = useStore((s) => s.setSelectByExpressionSource)
  const activeSlot = useStore((s) => s.activeSlot)
  const schema = useStore((s) => s.schema)
  const expressionData = useStore((s) => s.expressionData)
  const selectedGenes = useStore((s) => s.selectedGenes)
  const selectedGeneSetName = useStore((s) => s.selectedGeneSetName)
  const colorMode = useStore((s) => s.colorMode)
  const { colorByGene, colorByGenes } = useDataActions()
  const { summaries } = useObsSummaries()
  const existingColumnNames = useMemo(
    () => new Set(summaries.map((s) => s.name)),
    [summaries]
  )

  // Close on Escape
  useEffect(() => {
    if (!source) return
    const onKey = (e: KeyboardEvent) => {
      if (e.key === 'Escape') setSource(null)
    }
    document.addEventListener('keydown', onKey)
    return () => document.removeEventListener('keydown', onKey)
  }, [source, setSource])

  // Auto-color the plot when the modal opens, if it isn't already coloring by this source.
  useEffect(() => {
    if (!source) return
    if (source.type === 'gene') {
      const alreadyColoring =
        colorMode === 'expression' &&
        selectedGenes.length === 1 &&
        selectedGenes[0] === source.gene &&
        selectedGeneSetName === null
      if (!alreadyColoring) {
        colorByGene(source.gene)
      }
    } else {
      const alreadyColoring =
        colorMode === 'expression' &&
        selectedGeneSetName === source.name &&
        selectedGenes.length === source.genes.length
      if (!alreadyColoring) {
        colorByGenes(source.genes, undefined, source.name)
      }
    }
    // Deliberately only run when `source` identity changes — we don't want
    // to re-fetch on every store tick.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [source])

  const [openedInSlot, setOpenedInSlot] = useState<string | null>(null)

  useEffect(() => {
    if (!source) {
      setOpenedInSlot(null)
      return
    }
    if (openedInSlot === null) {
      setOpenedInSlot(activeSlot)
      return
    }
    if (openedInSlot !== activeSlot) {
      setSource(null)
    }
  }, [source, activeSlot, openedInSlot, setSource])

  // Histogram is memoized on expressionData identity.
  const histogram = useMemo(() => {
    if (!expressionData) return null
    return computeHistogram(expressionData.values)
  }, [expressionData])

  const [mode, setMode] = useState<ThresholdMode>('above')
  const [lo, setLo] = useState<number>(0)
  const [hi, setHi] = useState<number>(0)

  // When the histogram becomes available for a new source, reset mode/lo/hi
  // to sensible defaults based on the actual value distribution.
  useEffect(() => {
    if (!histogram || histogram.zeroVariance) return
    const defaults = defaultThresholds(
      expressionData?.values ?? [],
      mode,
      histogram.min,
      histogram.max
    )
    setLo(defaults.lo)
    setHi(defaults.hi)
    // Only fire when the histogram identity changes (new expressionData for a new source).
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [histogram])

  const matchCount = useMemo(() => {
    if (!expressionData || !histogram || histogram.zeroVariance) return 0
    // Fast count without building the indices array; skip nulls.
    const values = expressionData.values
    let count = 0
    for (let i = 0; i < values.length; i++) {
      const v = values[i]
      if (v == null) continue
      if (mode === 'above' && v >= lo) count++
      else if (mode === 'below' && v <= lo) count++
      else if (mode === 'between' && v >= lo && v <= hi) count++
    }
    return count
  }, [expressionData, histogram, mode, lo, hi])

  // Auto-swap to maintain lo <= hi when in Between mode.
  useEffect(() => {
    if (mode === 'between' && lo > hi) {
      setLo(hi)
      setHi(lo)
    }
  }, [mode, lo, hi])

  type Action = 'updateSelection' | 'labelCells'
  type SubAction = 'replace' | 'add' | 'intersect'

  const selectedCellIndices = useStore((s) => s.selectedCellIndices)
  const setSelectedCellIndices = useStore((s) => s.setSelectedCellIndices)
  const setComparisonGroup1 = useStore((s) => s.setComparisonGroup1)
  const setComparisonGroup2 = useStore((s) => s.setComparisonGroup2)
  const setDiffExpModalOpen = useStore((s) => s.setDiffExpModalOpen)
  const refreshObsSummaries = useStore((s) => s.refreshObsSummaries)

  const [action, setAction] = useState<Action>('updateSelection')
  const [subAction, setSubAction] = useState<SubAction>('replace')

  // When the existing selection goes empty, force sub-action back to 'replace'.
  useEffect(() => {
    if (selectedCellIndices.length === 0 && subAction !== 'replace') {
      setSubAction('replace')
    }
  }, [selectedCellIndices, subAction])

  const defaultAnnotationName = useMemo(() => {
    if (!source) return ''
    const base = source.type === 'gene' ? source.gene : source.name
    return `${base}_${mode}`.replace(/\s+/g, '_')
  }, [source, mode])

  const [annotationName, setAnnotationName] = useState('')
  const [userEditedName, setUserEditedName] = useState(false)
  type ApplyStatus =
    | { kind: 'idle' }
    | { kind: 'running' }
    | { kind: 'error'; message: string }
    | {
        kind: 'success'
        highCount: number
        lowCount: number
        annotationName: string
        highLabel: string
        lowLabel: string
        highIndices: number[]
        lowIndices: number[]
      }

  const [applyStatus, setApplyStatus] = useState<ApplyStatus>({ kind: 'idle' })

  const nameCollision =
    action === 'labelCells' &&
    annotationName.trim().length > 0 &&
    existingColumnNames.has(annotationName.trim())

  const [moreOptionsOpen, setMoreOptionsOpen] = useState(false)
  const [highLabel, setHighLabel] = useState<string>(MESSAGES.selectByExpression.defaultHighLabel)
  const [lowLabel, setLowLabel] = useState<string>(MESSAGES.selectByExpression.defaultLowLabel)
  type LabelContext = 'selection' | 'all'
  const [labelContext, setLabelContext] = useState<LabelContext>('selection')

  // When the existing selection becomes empty, force context to 'all'.
  useEffect(() => {
    if (selectedCellIndices.length === 0 && labelContext === 'selection') {
      setLabelContext('all')
    }
  }, [selectedCellIndices, labelContext])

  // Keep annotationName in sync with the default until the user has edited it.
  useEffect(() => {
    if (!userEditedName) setAnnotationName(defaultAnnotationName)
  }, [defaultAnnotationName, userEditedName])

  const handleApply = async () => {
    if (!expressionData) return

    // Bail if expressionData is stale relative to the current schema.
    // This can happen briefly after a filter_cells / delete_cells op if
    // cleanup missed something, and we must not send out-of-range indices
    // to the backend — it rejects them strictly in diff exp.
    const nCells = schema?.n_cells
    if (nCells != null && expressionData.values.length !== nCells) {
      setApplyStatus({
        kind: 'error',
        message: 'Expression data is out of sync with the dataset. Close and reopen the modal.',
      })
      return
    }

    // Clamp any out-of-range cell indices for safety. This is belt-and-
    // suspenders on top of the ScanpyModal cleanup — if a selection ever
    // survives a cell-count change via a path we haven't plugged, we
    // still won't hand bad indices to the backend.
    const inRange = (i: number): boolean =>
      Number.isInteger(i) && i >= 0 && (nCells == null || i < nCells)
    const validSelection = selectedCellIndices.filter(inRange)

    const matching = matchingIndices(expressionData.values, mode, lo, hi)

    if (action === 'updateSelection') {
      let final: number[]
      if (subAction === 'replace') {
        final = matching
      } else if (subAction === 'add') {
        const existing = new Set(validSelection)
        for (const i of matching) existing.add(i)
        final = Array.from(existing)
      } else {
        // intersect
        const matchingSet = new Set(matching)
        final = validSelection.filter((i) => matchingSet.has(i))
      }
      setSelectedCellIndices(final)
      setSource(null)
      return
    }

    // Label cells branch
    const contextIndices =
      labelContext === 'selection' ? validSelection : null // null => all cells
    let high: number[]
    let low: number[]
    if (contextIndices === null) {
      high = matching
      const matchingSet = new Set(matching)
      const total = expressionData.values.length
      low = []
      for (let i = 0; i < total; i++) {
        if (!matchingSet.has(i)) low.push(i)
      }
    } else {
      if (contextIndices.length === 0) {
        setApplyStatus({ kind: 'error', message: MESSAGES.selectByExpression.emptyContextError })
        return
      }
      const matchingSet = new Set(matching)
      high = contextIndices.filter((i) => matchingSet.has(i))
      low = contextIndices.filter((i) => !matchingSet.has(i))
    }

    const highLabelTrimmed = highLabel.trim() || MESSAGES.selectByExpression.defaultHighLabel
    const lowLabelTrimmed = lowLabel.trim() || MESSAGES.selectByExpression.defaultLowLabel
    const name = annotationName.trim()
    if (!name) {
      setApplyStatus({ kind: 'error', message: MESSAGES.selectByExpression.emptyNameError })
      return
    }

    setApplyStatus({ kind: 'running' })
    try {
      await createAnnotation(name)
      await addLabelToAnnotation(name, highLabelTrimmed)
      await addLabelToAnnotation(name, lowLabelTrimmed)
      if (high.length > 0) await labelCells(name, highLabelTrimmed, high)
      if (low.length > 0) await labelCells(name, lowLabelTrimmed, low)
      setApplyStatus({
        kind: 'success',
        highCount: high.length,
        lowCount: low.length,
        annotationName: name,
        highLabel: highLabelTrimmed,
        lowLabel: lowLabelTrimmed,
        highIndices: high,
        lowIndices: low,
      })
      refreshObsSummaries()
    } catch (err) {
      setApplyStatus({
        kind: 'error',
        message: (err as Error).message || MESSAGES.selectByExpression.failedToLabelCells,
      })
    }
  }

  if (!source) return null

  const title =
    source.type === 'gene'
      ? MESSAGES.selectByExpression.titleGene(source.gene)
      : MESSAGES.selectByExpression.titleGeneSet(source.name)

  let body: React.ReactNode
  if (!expressionData || !histogram) {
    body = <div style={{ color: '#888', padding: '24px 0' }}>{MESSAGES.selectByExpression.loading}</div>
  } else if (histogram.zeroVariance) {
    body = (
      <div style={{ color: '#e94560', padding: '24px 0' }}>
        {MESSAGES.selectByExpression.zeroVariance(histogram.min)}
      </div>
    )
  } else {
    body = (
      <div style={{ padding: '8px 0' }}>
        {/* Mode selector */}
        <div style={{ display: 'flex', gap: '4px', marginBottom: '8px', alignItems: 'center' }}>
          <span style={{ color: '#888', marginRight: '4px' }}>{MESSAGES.selectByExpression.thresholdModeLabel}</span>
          {(['above', 'below', 'between'] as ThresholdMode[]).map((m) => (
            <button
              key={m}
              onClick={() => {
                setMode(m)
                const d = defaultThresholds(expressionData.values, m, histogram.min, histogram.max)
                setLo(d.lo)
                setHi(d.hi)
              }}
              style={{
                padding: '4px 10px',
                backgroundColor: mode === m ? '#4ecdc4' : '#0f3460',
                color: mode === m ? '#16213e' : '#ccc',
                border: '1px solid #0f3460',
                borderRadius: '3px',
                cursor: 'pointer',
                fontSize: '11px',
                textTransform: 'capitalize',
              }}
            >
              {m}
            </button>
          ))}
        </div>

        <HistogramChart
          histogram={histogram}
          mode={mode}
          lo={lo}
          hi={hi}
          onChangeLo={setLo}
          onChangeHi={setHi}
        />

        {/* Numeric inputs + match counter */}
        <div style={{ display: 'flex', gap: '8px', alignItems: 'center', marginTop: '8px' }}>
          {mode === 'between' ? (
            <>
              <label style={{ color: '#888' }}>
                {MESSAGES.selectByExpression.loInputLabel}
                <input
                  type="number"
                  value={lo}
                  step="0.01"
                  onChange={(e) => setLo(Number(e.target.value))}
                  style={{
                    width: '70px',
                    backgroundColor: '#0f1625',
                    border: '1px solid #0f3460',
                    color: '#ccc',
                    padding: '3px 6px',
                    fontSize: '11px',
                  }}
                />
              </label>
              <label style={{ color: '#888' }}>
                {MESSAGES.selectByExpression.hiInputLabel}
                <input
                  type="number"
                  value={hi}
                  step="0.01"
                  onChange={(e) => setHi(Number(e.target.value))}
                  style={{
                    width: '70px',
                    backgroundColor: '#0f1625',
                    border: '1px solid #0f3460',
                    color: '#ccc',
                    padding: '3px 6px',
                    fontSize: '11px',
                  }}
                />
              </label>
            </>
          ) : (
            <label style={{ color: '#888' }}>
              {MESSAGES.selectByExpression.thresholdInputLabel}
              <input
                type="number"
                value={lo}
                step="0.01"
                onChange={(e) => setLo(Number(e.target.value))}
                style={{
                  width: '70px',
                  backgroundColor: '#0f1625',
                  border: '1px solid #0f3460',
                  color: '#ccc',
                  padding: '3px 6px',
                  fontSize: '11px',
                }}
              />
            </label>
          )}
          <span style={{ marginLeft: 'auto', color: '#4ecdc4' }}>
            {MESSAGES.selectByExpression.matchCounter(matchCount, expressionData.values.length)}
          </span>
        </div>

        {/* Action selector */}
        <div style={{ marginTop: '14px', paddingTop: '10px', borderTop: '1px solid #0f3460' }}>
          <div style={{ color: '#888', marginBottom: '6px' }}>{MESSAGES.selectByExpression.actionLabel}</div>
          <label style={{ display: 'block', marginBottom: '4px', color: '#ccc' }}>
            <input
              type="radio"
              checked={action === 'updateSelection'}
              onChange={() => setAction('updateSelection')}
            />{' '}
            {MESSAGES.selectByExpression.updateSelectionLabel}
          </label>
          {action === 'updateSelection' && (
            <div style={{ paddingLeft: '22px', display: 'flex', gap: '12px' }}>
              {(['replace', 'add', 'intersect'] as SubAction[]).map((sa) => {
                const disabled = sa !== 'replace' && selectedCellIndices.length === 0
                return (
                  <label
                    key={sa}
                    style={{ color: disabled ? '#555' : '#ccc', textTransform: 'capitalize' }}
                    title={disabled ? MESSAGES.selectByExpression.noExistingSelectionTooltip : undefined}
                  >
                    <input
                      type="radio"
                      checked={subAction === sa}
                      disabled={disabled}
                      onChange={() => setSubAction(sa)}
                    />{' '}
                    {sa}
                  </label>
                )
              })}
            </div>
          )}
          <label style={{ display: 'block', marginTop: '8px', marginBottom: '4px', color: '#ccc' }}>
            <input
              type="radio"
              checked={action === 'labelCells'}
              onChange={() => setAction('labelCells')}
            />{' '}
            {MESSAGES.selectByExpression.labelCellsLabel}
          </label>
          {action === 'labelCells' && (
            <div style={{ paddingLeft: '22px' }}>
              <label style={{ color: '#888', display: 'flex', gap: '6px', alignItems: 'center' }}>
                {MESSAGES.selectByExpression.annotationNameLabel}
                <input
                  type="text"
                  value={annotationName}
                  onChange={(e) => {
                    setUserEditedName(true)
                    setAnnotationName(e.target.value)
                  }}
                  style={{
                    flex: 1,
                    backgroundColor: '#0f1625',
                    border: '1px solid #0f3460',
                    color: '#ccc',
                    padding: '3px 6px',
                    fontSize: '11px',
                  }}
                />
              </label>
              {nameCollision && (
                <div style={{ color: '#e94560', fontSize: '11px', marginTop: '4px' }}>
                  {MESSAGES.selectByExpression.annotationCollision(annotationName.trim())}
                </div>
              )}
              <div style={{ marginTop: '6px' }}>
                <button
                  onClick={() => setMoreOptionsOpen((o) => !o)}
                  style={{
                    backgroundColor: 'transparent',
                    border: 'none',
                    color: '#4ecdc4',
                    cursor: 'pointer',
                    fontSize: '11px',
                    padding: 0,
                  }}
                >
                  {moreOptionsOpen ? MESSAGES.selectByExpression.moreOptionsOpen : MESSAGES.selectByExpression.moreOptionsClosed}
                </button>
                {moreOptionsOpen && (
                  <div
                    style={{
                      marginTop: '6px',
                      padding: '8px',
                      backgroundColor: '#0f1625',
                      borderRadius: '3px',
                      display: 'flex',
                      flexDirection: 'column',
                      gap: '6px',
                    }}
                  >
                    <label style={{ color: '#888', display: 'flex', gap: '6px', alignItems: 'center' }}>
                      {MESSAGES.selectByExpression.highLabelFieldLabel}
                      <input
                        type="text"
                        value={highLabel}
                        onChange={(e) => setHighLabel(e.target.value)}
                        style={{
                          flex: 1,
                          backgroundColor: '#16213e',
                          border: '1px solid #0f3460',
                          color: '#ccc',
                          padding: '3px 6px',
                          fontSize: '11px',
                        }}
                      />
                    </label>
                    <label style={{ color: '#888', display: 'flex', gap: '6px', alignItems: 'center' }}>
                      {MESSAGES.selectByExpression.lowLabelFieldLabel}
                      <input
                        type="text"
                        value={lowLabel}
                        onChange={(e) => setLowLabel(e.target.value)}
                        style={{
                          flex: 1,
                          backgroundColor: '#16213e',
                          border: '1px solid #0f3460',
                          color: '#ccc',
                          padding: '3px 6px',
                          fontSize: '11px',
                        }}
                      />
                    </label>
                    <div style={{ color: '#888' }}>{MESSAGES.selectByExpression.contextLabel}</div>
                    <label
                      style={{
                        color: selectedCellIndices.length === 0 ? '#555' : '#ccc',
                      }}
                      title={selectedCellIndices.length === 0 ? MESSAGES.selectByExpression.noExistingSelectionTooltip : undefined}
                    >
                      <input
                        type="radio"
                        checked={labelContext === 'selection'}
                        disabled={selectedCellIndices.length === 0}
                        onChange={() => setLabelContext('selection')}
                      />{' '}
                      {MESSAGES.selectByExpression.contextCurrentSelectionLabel(selectedCellIndices.length)}
                    </label>
                    <label style={{ color: '#ccc' }}>
                      <input
                        type="radio"
                        checked={labelContext === 'all'}
                        onChange={() => setLabelContext('all')}
                      />{' '}
                      {MESSAGES.selectByExpression.contextAllCellsLabel}
                    </label>
                  </div>
                )}
              </div>
            </div>
          )}
        </div>
      </div>
    )
  }

  return (
    <div
      style={{
        position: 'fixed',
        inset: 0,
        backgroundColor: 'rgba(0, 0, 0, 0.6)',
        display: 'flex',
        alignItems: 'center',
        justifyContent: 'center',
        zIndex: 1000,
      }}
      onClick={() => setSource(null)}
    >
      <div
        style={{
          backgroundColor: '#16213e',
          border: '1px solid #0f3460',
          borderRadius: '6px',
          padding: '16px 20px',
          width: '520px',
          maxWidth: '95vw',
          color: '#ccc',
          fontSize: '12px',
        }}
        onClick={(e) => e.stopPropagation()}
      >
        <div
          style={{
            fontSize: '14px',
            fontWeight: 600,
            marginBottom: '12px',
            color: '#e94560',
          }}
        >
          {title}
        </div>
        {body}
        {applyStatus.kind === 'error' && (
          <div style={{ color: '#e94560', fontSize: '11px', marginTop: '8px' }}>
            {applyStatus.message}
          </div>
        )}
        <div
          style={{
            display: 'flex',
            justifyContent: 'flex-end',
            gap: '8px',
            marginTop: '14px',
            alignItems: 'center',
          }}
        >
          {applyStatus.kind === 'success' ? (
            <>
              <span style={{ marginRight: 'auto', color: '#4ecdc4', fontSize: '11px' }}>
                {MESSAGES.selectByExpression.successFooter(applyStatus.highCount, applyStatus.highLabel, applyStatus.lowCount, applyStatus.lowLabel)}
              </span>
              <button
                style={{
                  padding: '6px 14px',
                  backgroundColor: '#4ecdc4',
                  color: '#16213e',
                  border: '1px solid #4ecdc4',
                  borderRadius: '3px',
                  cursor: 'pointer',
                  fontSize: '11px',
                  fontWeight: 600,
                }}
                onClick={() => {
                  setComparisonGroup1(applyStatus.highIndices, applyStatus.highLabel)
                  setComparisonGroup2(applyStatus.lowIndices, applyStatus.lowLabel)
                  setDiffExpModalOpen(true)
                  setSource(null)
                }}
              >
                {MESSAGES.selectByExpression.openDiffExpButton}
              </button>
              <button
                style={{
                  padding: '6px 14px',
                  backgroundColor: '#0f3460',
                  color: '#ccc',
                  border: '1px solid #0f3460',
                  borderRadius: '3px',
                  cursor: 'pointer',
                  fontSize: '11px',
                }}
                onClick={() => setSource(null)}
              >
                {MESSAGES.selectByExpression.closeButton}
              </button>
            </>
          ) : (
            <>
              <button
                style={{
                  padding: '6px 14px',
                  backgroundColor: '#0f3460',
                  color: '#ccc',
                  border: '1px solid #0f3460',
                  borderRadius: '3px',
                  cursor: 'pointer',
                  fontSize: '11px',
                }}
                onClick={() => setSource(null)}
              >
                {MESSAGES.selectByExpression.cancelButton}
              </button>
              <button
                disabled={
                  applyStatus.kind === 'running' ||
                  (action === 'labelCells' && nameCollision) ||
                  !!histogram?.zeroVariance ||
                  (action === 'labelCells' &&
                    labelContext === 'selection' &&
                    selectedCellIndices.length === 0)
                }
                title={
                  histogram?.zeroVariance
                    ? MESSAGES.selectByExpression.zeroVarianceTooltip
                    : action === 'labelCells' && nameCollision
                    ? MESSAGES.selectByExpression.collisionTooltip
                    : action === 'labelCells' &&
                      labelContext === 'selection' &&
                      selectedCellIndices.length === 0
                    ? MESSAGES.selectByExpression.emptyContextTooltip
                    : undefined
                }
                style={{
                  padding: '6px 14px',
                  backgroundColor: '#4ecdc4',
                  color: '#16213e',
                  border: '1px solid #4ecdc4',
                  borderRadius: '3px',
                  cursor:
                    applyStatus.kind === 'running' ||
                    (action === 'labelCells' && nameCollision) ||
                    histogram?.zeroVariance ||
                    (action === 'labelCells' &&
                      labelContext === 'selection' &&
                      selectedCellIndices.length === 0)
                      ? 'not-allowed'
                      : 'pointer',
                  fontSize: '11px',
                  fontWeight: 600,
                  opacity:
                    applyStatus.kind === 'running' ||
                    (action === 'labelCells' && nameCollision) ||
                    histogram?.zeroVariance ||
                    (action === 'labelCells' &&
                      labelContext === 'selection' &&
                      selectedCellIndices.length === 0)
                      ? 0.6
                      : 1,
                }}
                onClick={handleApply}
              >
                {applyStatus.kind === 'running' ? MESSAGES.selectByExpression.labelingButton : MESSAGES.selectByExpression.applyButton}
              </button>
            </>
          )}
        </div>
      </div>
    </div>
  )
}
