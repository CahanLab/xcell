import { useEffect, useMemo, useState } from 'react'
import type { ObsSummary } from '../hooks/useData'

interface MergeLabelsModalProps {
  columnName: string
  summary: ObsSummary | null
  onClose: () => void
  onMerge: (labels: string[], newLabel: string) => Promise<void>
}

export default function MergeLabelsModal({ columnName, summary, onClose, onMerge }: MergeLabelsModalProps) {
  const labels = useMemo(() => (summary?.categories || []).map((c) => c.value), [summary])

  const [checked, setChecked] = useState<Set<string>>(new Set())
  const [newLabel, setNewLabel] = useState('')
  const [busy, setBusy] = useState(false)
  const [error, setError] = useState<string | null>(null)

  useEffect(() => {
    setChecked(new Set())
    setNewLabel('')
    setError(null)
    setBusy(false)
  }, [columnName])

  // Default the new-label input to the joined names so users can edit instead
  // of starting from blank. Don't overwrite once they've typed something.
  const [userTouched, setUserTouched] = useState(false)
  useEffect(() => {
    if (userTouched) return
    if (checked.size >= 2) {
      setNewLabel([...checked].join('+'))
    } else {
      setNewLabel('')
    }
  }, [checked, userTouched])

  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      if (e.key === 'Escape') onClose()
    }
    document.addEventListener('keydown', onKey)
    return () => document.removeEventListener('keydown', onKey)
  }, [onClose])

  const toggle = (label: string) => {
    setChecked((prev) => {
      const next = new Set(prev)
      if (next.has(label)) next.delete(label)
      else next.add(label)
      return next
    })
  }

  const selectAll = () => setChecked(new Set(labels))
  const clearAll = () => setChecked(new Set())

  const trimmedNew = newLabel.trim()
  const canSubmit = checked.size >= 2 && trimmedNew.length > 0 && !busy

  const handleMerge = async () => {
    if (!canSubmit) return
    setBusy(true)
    setError(null)
    try {
      await onMerge([...checked], trimmedNew)
      onClose()
    } catch (err) {
      setError((err as Error).message || 'Merge failed')
      setBusy(false)
    }
  }

  return (
    <div
      onClick={onClose}
      style={{
        position: 'fixed',
        inset: 0,
        backgroundColor: 'rgba(0, 0, 0, 0.6)',
        display: 'flex',
        alignItems: 'center',
        justifyContent: 'center',
        zIndex: 100,
      }}
    >
      <div
        onClick={(e) => e.stopPropagation()}
        style={{
          backgroundColor: '#16213e',
          border: '1px solid #0f3460',
          borderRadius: '8px',
          padding: '20px 24px',
          minWidth: '380px',
          maxWidth: '480px',
          color: '#eee',
        }}
      >
        <div style={{ fontSize: '15px', fontWeight: 600, marginBottom: '8px' }}>Merge labels</div>
        <div style={{ fontSize: '12px', color: '#aaa', marginBottom: '12px' }}>
          Column: <span style={{ color: '#eee' }}>{columnName}</span>
        </div>

        <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '6px' }}>
          <label style={{ fontSize: '11px', color: '#888' }}>
            Select labels to merge ({checked.size} selected)
          </label>
          <div style={{ display: 'flex', gap: '6px' }}>
            <button
              onClick={selectAll}
              style={{
                fontSize: '10px',
                padding: '2px 6px',
                backgroundColor: 'transparent',
                color: '#4ecdc4',
                border: '1px solid #0f3460',
                borderRadius: '3px',
                cursor: 'pointer',
              }}
            >
              All
            </button>
            <button
              onClick={clearAll}
              style={{
                fontSize: '10px',
                padding: '2px 6px',
                backgroundColor: 'transparent',
                color: '#888',
                border: '1px solid #0f3460',
                borderRadius: '3px',
                cursor: 'pointer',
              }}
            >
              Clear
            </button>
          </div>
        </div>

        <div
          style={{
            maxHeight: '220px',
            overflowY: 'auto',
            border: '1px solid #0f3460',
            borderRadius: '4px',
            padding: '6px 8px',
            marginBottom: '12px',
            backgroundColor: '#0f1625',
          }}
        >
          {labels.length === 0 ? (
            <div style={{ fontSize: '12px', color: '#666' }}>No labels in this column.</div>
          ) : (
            labels.map((label) => (
              <label
                key={label}
                style={{
                  display: 'flex',
                  alignItems: 'center',
                  fontSize: '12px',
                  color: '#ccc',
                  padding: '2px 0',
                  cursor: 'pointer',
                }}
              >
                <input
                  type="checkbox"
                  checked={checked.has(label)}
                  onChange={() => toggle(label)}
                  style={{ marginRight: '6px' }}
                />
                <span style={{ overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap' }}>
                  {label}
                </span>
              </label>
            ))
          )}
        </div>

        <div style={{ marginBottom: '12px' }}>
          <label style={{ display: 'block', fontSize: '11px', color: '#888', marginBottom: '4px' }}>
            New label name
          </label>
          <input
            type="text"
            value={newLabel}
            onChange={(e) => {
              setNewLabel(e.target.value)
              setUserTouched(true)
            }}
            placeholder="e.g. ab"
            style={{
              width: '100%',
              padding: '6px 8px',
              fontSize: '12px',
              backgroundColor: '#0f3460',
              color: '#eee',
              border: '1px solid #1a1a2e',
              borderRadius: '4px',
            }}
          />
          <div style={{ fontSize: '10px', color: '#666', marginTop: '4px' }}>
            If this name matches an existing label, the selected labels merge into it.
          </div>
        </div>

        {error && (
          <div
            style={{
              fontSize: '11px',
              color: '#e94560',
              backgroundColor: 'rgba(233, 69, 96, 0.15)',
              padding: '8px',
              borderRadius: '4px',
              marginBottom: '12px',
            }}
          >
            {error}
          </div>
        )}

        <div style={{ display: 'flex', justifyContent: 'flex-end', gap: '8px' }}>
          <button
            onClick={onClose}
            style={{
              padding: '6px 14px',
              fontSize: '12px',
              backgroundColor: 'transparent',
              color: '#aaa',
              border: '1px solid #0f3460',
              borderRadius: '4px',
              cursor: 'pointer',
            }}
          >
            Cancel
          </button>
          <button
            onClick={handleMerge}
            disabled={!canSubmit}
            style={{
              padding: '6px 14px',
              fontSize: '12px',
              backgroundColor: canSubmit ? '#4ecdc4' : '#1a1a2e',
              color: canSubmit ? '#16213e' : '#555',
              border: 'none',
              borderRadius: '4px',
              cursor: canSubmit ? 'pointer' : 'not-allowed',
              fontWeight: 600,
            }}
          >
            {busy ? 'Merging…' : 'Merge'}
          </button>
        </div>
      </div>
    </div>
  )
}
