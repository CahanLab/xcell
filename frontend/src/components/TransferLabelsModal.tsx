import { useEffect, useMemo, useState } from 'react'
import type {
  ObsSummary,
  TransferObsLabelsParams,
  TransferObsLabelsResult,
  TransferRenameMode,
} from '../hooks/useData'

interface TransferLabelsModalProps {
  // The column whose "..." menu was used; defaults to the parent being refined.
  targetColumnDefault: string
  summaries: ObsSummary[]
  onClose: () => void
  onApply: (params: TransferObsLabelsParams) => Promise<TransferObsLabelsResult>
  onColorBy: (column: string) => void
}

const dark = {
  panel: '#16213e',
  border: '#0f3460',
  field: '#0f3460',
  text: '#eee',
  sub: '#aaa',
  faint: '#888',
  accent: '#4ecdc4',
  inset: '#0f1625',
}

const labelStyle: React.CSSProperties = {
  display: 'block',
  fontSize: '11px',
  color: dark.faint,
  marginBottom: '4px',
}

const selectStyle: React.CSSProperties = {
  width: '100%',
  padding: '6px 8px',
  fontSize: '12px',
  backgroundColor: dark.field,
  color: dark.text,
  border: `1px solid #1a1a2e`,
  borderRadius: '4px',
}

const inputStyle: React.CSSProperties = { ...selectStyle }

/** A column is a plausible "subcluster" source if it carries an "unassigned"
 *  category — that's how masked-then-subclustered cells are written. */
function hasUnassigned(s: ObsSummary): boolean {
  return (s.categories || []).some((c) => c.value === 'unassigned')
}

export default function TransferLabelsModal({
  targetColumnDefault,
  summaries,
  onClose,
  onApply,
  onColorBy,
}: TransferLabelsModalProps) {
  // Only categorical/string columns can take part.
  const columns = useMemo(
    () => summaries.filter((s) => s.dtype === 'category' || s.dtype === 'string'),
    [summaries]
  )
  const columnNames = useMemo(() => columns.map((c) => c.name), [columns])

  const [targetColumn, setTargetColumn] = useState(targetColumnDefault)
  // Best guess for the source: the first column (other than the target) that
  // looks like a subcluster result, else the first other column.
  const guessSource = useMemo(() => {
    const others = columnNames.filter((n) => n !== targetColumnDefault)
    const sub = columns.find((c) => c.name !== targetColumnDefault && hasUnassigned(c))
    return sub?.name ?? others[0] ?? ''
  }, [columns, columnNames, targetColumnDefault])
  const [sourceColumn, setSourceColumn] = useState(guessSource)

  const [renameMode, setRenameMode] = useState<TransferRenameMode>('parent_prefix')
  const [customPrefix, setCustomPrefix] = useState('')
  const [unassignedText, setUnassignedText] = useState('unassigned')

  const [overwrite, setOverwrite] = useState(false)
  const [newName, setNewName] = useState(`${targetColumnDefault}_refined`)

  const [busy, setBusy] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const [result, setResult] = useState<TransferObsLabelsResult | null>(null)

  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      if (e.key === 'Escape') onClose()
    }
    document.addEventListener('keydown', onKey)
    return () => document.removeEventListener('keydown', onKey)
  }, [onClose])

  // Keep the suggested new-column name in sync with the chosen target until the
  // user starts editing it.
  const [nameTouched, setNameTouched] = useState(false)
  useEffect(() => {
    if (!nameTouched) setNewName(`${targetColumn}_refined`)
  }, [targetColumn, nameTouched])

  const sourceSummary = columns.find((c) => c.name === sourceColumn)
  const outColumn = overwrite ? targetColumn : newName.trim()

  const sampleSourceLabels = useMemo(
    () =>
      (sourceSummary?.categories || [])
        .map((c) => c.value)
        .filter((v) => v !== unassignedText.trim())
        .slice(0, 3),
    [sourceSummary, unassignedText]
  )

  const previewExample = useMemo(() => {
    const ex = sampleSourceLabels[0] ?? '0'
    if (renameMode === 'replace') return ex
    if (renameMode === 'custom_prefix') return `${customPrefix}${ex}`
    return `${targetColumn || 'Parent'}.${ex}`
  }, [renameMode, customPrefix, sampleSourceLabels, targetColumn])

  const canSubmit =
    !busy &&
    !!targetColumn &&
    !!sourceColumn &&
    targetColumn !== sourceColumn &&
    !!outColumn

  const handleApply = async () => {
    if (!canSubmit) return
    setBusy(true)
    setError(null)
    try {
      const unassignedValues = unassignedText
        .split(',')
        .map((s) => s.trim())
        .filter(Boolean)
      const res = await onApply({
        targetColumn,
        sourceColumn,
        outColumn,
        renameMode,
        prefix: customPrefix,
        unassignedValues: unassignedValues.length ? unassignedValues : undefined,
      })
      setResult(res)
    } catch (err) {
      setError((err as Error).message || 'Transfer failed')
    } finally {
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
          backgroundColor: dark.panel,
          border: `1px solid ${dark.border}`,
          borderRadius: '8px',
          padding: '20px 24px',
          minWidth: '440px',
          maxWidth: '560px',
          color: dark.text,
        }}
      >
        <div style={{ fontSize: '15px', fontWeight: 600, marginBottom: '4px' }}>
          Refine category with subcluster labels
        </div>
        <div style={{ fontSize: '12px', color: dark.sub, marginBottom: '16px' }}>
          Push the labels from a subcluster column into a parent category. Cells the
          subcluster left as <em>unassigned</em> keep their parent label.
        </div>

        {result ? (
          <>
            <div
              style={{
                fontSize: '12px',
                color: '#9be7d8',
                backgroundColor: 'rgba(78, 205, 196, 0.12)',
                padding: '10px 12px',
                borderRadius: '4px',
                marginBottom: '14px',
                lineHeight: 1.6,
              }}
            >
              Wrote <strong>{result.out_column}</strong>:{' '}
              {result.n_overridden.toLocaleString()} cells overridden,{' '}
              {result.n_kept.toLocaleString()} kept their parent label,{' '}
              {result.categories.length} categories total.
            </div>
            <div style={{ display: 'flex', justifyContent: 'flex-end', gap: '8px' }}>
              <button onClick={onClose} style={ghostButton}>
                Close
              </button>
              <button
                onClick={() => {
                  onColorBy(result.out_column)
                  onClose()
                }}
                style={primaryButton(true)}
              >
                Color by {result.out_column}
              </button>
            </div>
          </>
        ) : (
          <>
            <div style={{ display: 'flex', gap: '12px', marginBottom: '14px' }}>
              <div style={{ flex: 1 }}>
                <label style={labelStyle}>Parent category (to refine)</label>
                <select
                  value={targetColumn}
                  onChange={(e) => setTargetColumn(e.target.value)}
                  style={selectStyle}
                >
                  {columnNames.map((n) => (
                    <option key={n} value={n}>
                      {n}
                    </option>
                  ))}
                </select>
              </div>
              <div style={{ flex: 1 }}>
                <label style={labelStyle}>New labels from (subcluster)</label>
                <select
                  value={sourceColumn}
                  onChange={(e) => setSourceColumn(e.target.value)}
                  style={selectStyle}
                >
                  {columnNames
                    .filter((n) => n !== targetColumn)
                    .map((n) => (
                      <option key={n} value={n}>
                        {n}
                        {hasUnassigned(columns.find((c) => c.name === n)!) ? ' ⟂' : ''}
                      </option>
                    ))}
                </select>
              </div>
            </div>

            {targetColumn === sourceColumn && (
              <div style={{ fontSize: '11px', color: '#e9a23b', marginBottom: '10px' }}>
                Parent and source must be different columns.
              </div>
            )}

            <div style={{ marginBottom: '14px' }}>
              <label style={labelStyle}>Name the incoming labels</label>
              {(
                [
                  ['parent_prefix', 'Prefix with parent label', 'recommended — keeps provenance & avoids "0"/"1" clashes'],
                  ['replace', 'Use new labels as-is', ''],
                  ['custom_prefix', 'Custom prefix', ''],
                ] as [TransferRenameMode, string, string][]
              ).map(([mode, title, hint]) => (
                <label
                  key={mode}
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
                    type="radio"
                    name="renameMode"
                    checked={renameMode === mode}
                    onChange={() => setRenameMode(mode)}
                    style={{ marginRight: '6px' }}
                  />
                  <span>{title}</span>
                  {mode === 'custom_prefix' && renameMode === 'custom_prefix' && (
                    <input
                      type="text"
                      value={customPrefix}
                      onChange={(e) => setCustomPrefix(e.target.value)}
                      placeholder="prefix_"
                      style={{ ...inputStyle, width: '110px', marginLeft: '8px', padding: '2px 6px' }}
                    />
                  )}
                  {hint && <span style={{ marginLeft: '8px', color: dark.faint, fontSize: '10px' }}>{hint}</span>}
                </label>
              ))}
              <div style={{ fontSize: '10px', color: dark.faint, marginTop: '6px' }}>
                Example: a source label <code>{sampleSourceLabels[0] ?? '0'}</code> →{' '}
                <code style={{ color: dark.accent }}>{previewExample}</code>
              </div>
            </div>

            <div style={{ marginBottom: '14px' }}>
              <label style={labelStyle}>Treat these source values as "unassigned" (keep parent)</label>
              <input
                type="text"
                value={unassignedText}
                onChange={(e) => setUnassignedText(e.target.value)}
                placeholder="unassigned"
                style={inputStyle}
              />
              <div style={{ fontSize: '10px', color: dark.faint, marginTop: '4px' }}>
                Comma-separated. Empty/NaN cells are always treated as unassigned.
              </div>
            </div>

            <div style={{ marginBottom: '16px' }}>
              <label style={labelStyle}>Output</label>
              <label style={{ display: 'flex', alignItems: 'center', fontSize: '12px', color: '#ccc', padding: '2px 0' }}>
                <input
                  type="radio"
                  name="output"
                  checked={!overwrite}
                  onChange={() => setOverwrite(false)}
                  style={{ marginRight: '6px' }}
                />
                <span style={{ marginRight: '8px' }}>New column</span>
                <input
                  type="text"
                  value={newName}
                  onChange={(e) => {
                    setNewName(e.target.value)
                    setNameTouched(true)
                  }}
                  disabled={overwrite}
                  style={{ ...inputStyle, flex: 1, padding: '4px 6px', opacity: overwrite ? 0.5 : 1 }}
                />
              </label>
              <label style={{ display: 'flex', alignItems: 'center', fontSize: '12px', color: '#ccc', padding: '2px 0' }}>
                <input
                  type="radio"
                  name="output"
                  checked={overwrite}
                  onChange={() => setOverwrite(true)}
                  style={{ marginRight: '6px' }}
                />
                <span>Overwrite <strong>{targetColumn}</strong> in place</span>
              </label>
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
              <button onClick={onClose} style={ghostButton}>
                Cancel
              </button>
              <button onClick={handleApply} disabled={!canSubmit} style={primaryButton(canSubmit)}>
                {busy ? 'Applying…' : 'Apply'}
              </button>
            </div>
          </>
        )}
      </div>
    </div>
  )
}

const ghostButton: React.CSSProperties = {
  padding: '6px 14px',
  fontSize: '12px',
  backgroundColor: 'transparent',
  color: '#aaa',
  border: `1px solid ${dark.border}`,
  borderRadius: '4px',
  cursor: 'pointer',
}

function primaryButton(enabled: boolean): React.CSSProperties {
  return {
    padding: '6px 14px',
    fontSize: '12px',
    backgroundColor: enabled ? dark.accent : '#1a1a2e',
    color: enabled ? dark.panel : '#555',
    border: 'none',
    borderRadius: '4px',
    cursor: enabled ? 'pointer' : 'not-allowed',
    fontWeight: 600,
  }
}
