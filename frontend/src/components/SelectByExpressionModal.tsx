import { useEffect, useMemo, useState } from 'react'
import { MESSAGES } from '../messages'
import { useStore, cfgDefault } from '../store'
import {
  createAnnotation,
  addLabelToAnnotation,
  labelCells,
  useDataActions,
  useObsSummaries,
  refreshSchema,
} from '../hooks/useData'
import {
  HistogramChart,
  ThresholdMode,
  computeHistogram,
  defaultThresholds,
  matchingIndices,
} from '../utils/histogram'

export type { ThresholdMode, Histogram } from '../utils/histogram'
export { computeHistogram, matchingIndices, defaultThresholds } from '../utils/histogram'

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

  const [mode, setMode] = useState<ThresholdMode>(() => cfgDefault(['select_by_expression', 'mode'], 'above' as ThresholdMode))
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

  const [action, setAction] = useState<Action>(() => cfgDefault(['select_by_expression', 'action'], 'updateSelection' as Action))
  const [subAction, setSubAction] = useState<SubAction>(() => cfgDefault(['select_by_expression', 'sub_action'], 'replace' as SubAction))

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
  const [labelContext, setLabelContext] = useState<LabelContext>(() => cfgDefault(['select_by_expression', 'label_context'], 'selection' as LabelContext))

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
      // Refresh schema so Compare Cells / Heatmap dropdowns see the new column.
      await refreshSchema()
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
