import { useEffect, useMemo, useState } from 'react'
import { useStore } from '../store'
import { MESSAGES } from '../messages'
import {
  applyGeneMask,
  clearGeneMask,
  fetchBooleanColumnValues,
  type BooleanColumnValuesResponse,
} from '../hooks/useData'

type ColumnState = 'off' | 'keep' | 'hide'

export default function GeneMaskModal() {
  const open = useStore((s) => s.geneMaskModalOpen)
  const setOpen = useStore((s) => s.setGeneMaskModalOpen)
  const config = useStore((s) => s.geneMaskConfig)

  const [loading, setLoading] = useState(false)
  const [loadError, setLoadError] = useState<string | null>(null)
  const [applyError, setApplyError] = useState<string | null>(null)
  const [applying, setApplying] = useState(false)
  const [columnValues, setColumnValues] = useState<BooleanColumnValuesResponse | null>(null)
  const [states, setStates] = useState<Record<string, ColumnState>>({})
  const [combineMode, setCombineMode] = useState<'or' | 'and'>('or')

  // Close on Escape
  useEffect(() => {
    if (!open) return
    const onKey = (e: KeyboardEvent) => {
      if (e.key === 'Escape') setOpen(false)
    }
    document.addEventListener('keydown', onKey)
    return () => document.removeEventListener('keydown', onKey)
  }, [open, setOpen])

  // Fetch column values on open
  useEffect(() => {
    if (!open) return
    setLoading(true)
    setLoadError(null)
    setApplyError(null)
    fetchBooleanColumnValues()
      .then((data) => {
        setColumnValues(data)
        // Initialize states from the current backend config (if any)
        const initial: Record<string, ColumnState> = {}
        for (const col of Object.keys(data.columns)) {
          initial[col] = 'off'
        }
        if (config?.active) {
          for (const c of config.keepColumns) {
            if (c in initial) initial[c] = 'keep'
          }
          for (const c of config.hideColumns) {
            if (c in initial) initial[c] = 'hide'
          }
          setCombineMode(config.keepCombineMode)
        } else {
          setCombineMode('or')
        }
        setStates(initial)
      })
      .catch((err) => setLoadError((err as Error).message))
      .finally(() => setLoading(false))
  }, [open, config])

  // Client-side preview count
  const preview = useMemo(() => {
    if (!columnValues) return null
    const n = columnValues.n_genes
    const keepCols = Object.entries(states).filter(([, v]) => v === 'keep').map(([k]) => k)
    const hideCols = Object.entries(states).filter(([, v]) => v === 'hide').map(([k]) => k)

    // Build keep set
    let keepSet: Set<number>
    if (keepCols.length === 0) {
      keepSet = new Set<number>()
      for (let i = 0; i < n; i++) keepSet.add(i)
    } else if (combineMode === 'and') {
      keepSet = new Set<number>(columnValues.columns[keepCols[0]] ?? [])
      for (let i = 1; i < keepCols.length; i++) {
        const arr = new Set(columnValues.columns[keepCols[i]] ?? [])
        keepSet = new Set([...keepSet].filter((x) => arr.has(x)))
      }
    } else {
      keepSet = new Set<number>()
      for (const c of keepCols) {
        for (const idx of columnValues.columns[c] ?? []) keepSet.add(idx)
      }
    }

    // Build hide set (union)
    const hideSet = new Set<number>()
    for (const c of hideCols) {
      for (const idx of columnValues.columns[c] ?? []) hideSet.add(idx)
    }

    let visible = 0
    for (const idx of keepSet) {
      if (!hideSet.has(idx)) visible++
    }
    return { visible, total: n }
  }, [columnValues, states, combineMode])

  const hasActiveMask = config?.active === true

  const handleApply = async () => {
    if (!columnValues) return
    const keepColumns = Object.entries(states).filter(([, v]) => v === 'keep').map(([k]) => k)
    const hideColumns = Object.entries(states).filter(([, v]) => v === 'hide').map(([k]) => k)
    setApplying(true)
    setApplyError(null)
    try {
      await applyGeneMask({ keepColumns, hideColumns, keepCombineMode: combineMode })
      setOpen(false)
    } catch (err) {
      const msg = err instanceof Error ? err.message : 'Unknown error'
      // Prefer a friendly message for the zero-visible case
      if (msg.toLowerCase().includes('0 visible')) {
        setApplyError(MESSAGES.geneMask.noneVisibleError)
      } else {
        setApplyError(msg)
      }
    } finally {
      setApplying(false)
    }
  }

  const handleClear = async () => {
    setApplying(true)
    setApplyError(null)
    try {
      await clearGeneMask()
      setOpen(false)
    } catch (err) {
      setApplyError(err instanceof Error ? err.message : 'Unknown error')
    } finally {
      setApplying(false)
    }
  }

  if (!open) return null

  const boolCols = columnValues ? Object.keys(columnValues.columns) : []

  return (
    <div
      style={{
        position: 'fixed',
        top: 0,
        left: 0,
        right: 0,
        bottom: 0,
        backgroundColor: 'rgba(0, 0, 0, 0.6)',
        display: 'flex',
        alignItems: 'center',
        justifyContent: 'center',
        zIndex: 1000,
      }}
      onClick={() => setOpen(false)}
    >
      <div
        style={{
          backgroundColor: '#16213e',
          border: '1px solid #0f3460',
          borderRadius: '6px',
          padding: '20px',
          minWidth: '460px',
          maxWidth: '560px',
          color: '#ccc',
        }}
        onClick={(e) => e.stopPropagation()}
      >
        <div
          style={{
            display: 'flex',
            justifyContent: 'space-between',
            alignItems: 'center',
            marginBottom: '8px',
          }}
        >
          <h3 style={{ margin: 0, fontSize: '14px', color: '#ccc' }}>
            {MESSAGES.geneMask.title}
          </h3>
          <button
            onClick={() => setOpen(false)}
            style={{
              background: 'transparent',
              border: 'none',
              color: '#888',
              cursor: 'pointer',
              fontSize: '16px',
            }}
            title="Close"
          >
            ✕
          </button>
        </div>

        <div style={{ fontSize: '11px', color: '#888', marginBottom: '12px' }}>
          {MESSAGES.geneMask.description}
        </div>

        {loading && <div style={{ fontSize: '11px', color: '#888' }}>Loading…</div>}
        {loadError && <div style={{ fontSize: '11px', color: '#ff6b6b' }}>{loadError}</div>}

        {!loading && !loadError && boolCols.length === 0 && (
          <div style={{ fontSize: '11px', color: '#888' }}>
            {MESSAGES.geneMask.noBoolColumns}
          </div>
        )}

        {!loading && !loadError && boolCols.length > 0 && (
          <div style={{ maxHeight: '320px', overflowY: 'auto', marginBottom: '12px' }}>
            {boolCols.map((col) => {
              const indices = columnValues!.columns[col] ?? []
              const state = states[col] ?? 'off'
              return (
                <div key={col} style={{ marginBottom: '10px' }}>
                  <div style={{ fontSize: '12px', color: '#ccc', marginBottom: '2px' }}>
                    {col}{' '}
                    <span style={{ fontSize: '10px', color: '#777' }}>
                      ({MESSAGES.geneMask.columnLabel(indices.length, columnValues!.n_genes)})
                    </span>
                  </div>
                  <div style={{ display: 'flex', gap: '12px', fontSize: '11px' }}>
                    {(['off', 'keep', 'hide'] as ColumnState[]).map((s) => (
                      <label key={s} style={{ cursor: 'pointer' }}>
                        <input
                          type="radio"
                          checked={state === s}
                          onChange={() => setStates((prev) => ({ ...prev, [col]: s }))}
                          style={{ marginRight: '4px' }}
                        />
                        {s === 'off' && MESSAGES.geneMask.stateOff}
                        {s === 'keep' && MESSAGES.geneMask.stateKeep}
                        {s === 'hide' && MESSAGES.geneMask.stateHide}
                      </label>
                    ))}
                  </div>
                </div>
              )
            })}

            <div style={{ marginTop: '14px', paddingTop: '10px', borderTop: '1px solid #0f3460' }}>
              <div style={{ fontSize: '11px', color: '#888', marginBottom: '4px' }}>
                {MESSAGES.geneMask.combineLabel}
              </div>
              <div style={{ display: 'flex', gap: '12px', fontSize: '11px' }}>
                <label style={{ cursor: 'pointer' }}>
                  <input
                    type="radio"
                    checked={combineMode === 'or'}
                    onChange={() => setCombineMode('or')}
                    style={{ marginRight: '4px' }}
                  />
                  {MESSAGES.geneMask.combineAny}
                </label>
                <label style={{ cursor: 'pointer' }}>
                  <input
                    type="radio"
                    checked={combineMode === 'and'}
                    onChange={() => setCombineMode('and')}
                    style={{ marginRight: '4px' }}
                  />
                  {MESSAGES.geneMask.combineAll}
                </label>
              </div>
            </div>

            {preview && (
              <div style={{ marginTop: '10px', fontSize: '11px', color: '#4ecdc4' }}>
                {MESSAGES.geneMask.previewLabel(preview.visible, preview.total)}
              </div>
            )}
          </div>
        )}

        {applyError && (
          <div style={{ fontSize: '11px', color: '#ff6b6b', marginBottom: '8px' }}>
            {applyError}
          </div>
        )}

        <div
          style={{
            display: 'flex',
            justifyContent: 'space-between',
            gap: '8px',
            marginTop: '12px',
          }}
        >
          <button
            onClick={handleClear}
            disabled={!hasActiveMask || applying}
            style={{
              padding: '4px 10px',
              fontSize: '11px',
              backgroundColor: 'transparent',
              color: hasActiveMask ? '#ccc' : '#555',
              border: '1px solid #0f3460',
              borderRadius: '3px',
              cursor: hasActiveMask && !applying ? 'pointer' : 'not-allowed',
            }}
          >
            {MESSAGES.geneMask.clearButton}
          </button>
          <div style={{ display: 'flex', gap: '8px' }}>
            <button
              onClick={() => setOpen(false)}
              disabled={applying}
              style={{
                padding: '4px 10px',
                fontSize: '11px',
                backgroundColor: 'transparent',
                color: '#aaa',
                border: '1px solid #0f3460',
                borderRadius: '3px',
                cursor: applying ? 'not-allowed' : 'pointer',
              }}
            >
              {MESSAGES.geneMask.cancelButton}
            </button>
            <button
              onClick={handleApply}
              disabled={applying || !columnValues || boolCols.length === 0}
              style={{
                padding: '4px 12px',
                fontSize: '11px',
                backgroundColor: '#4ecdc4',
                color: '#0a0a1a',
                border: 'none',
                borderRadius: '3px',
                cursor: applying ? 'wait' : 'pointer',
              }}
            >
              {MESSAGES.geneMask.applyButton}
            </button>
          </div>
        </div>
      </div>
    </div>
  )
}
