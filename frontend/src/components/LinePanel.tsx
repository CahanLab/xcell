import { useState, useCallback } from 'react'
import { useStore } from '../store'

const styles = {
  panel: {
    position: 'absolute' as const,
    bottom: '20px',
    left: '20px',
    backgroundColor: 'rgba(22, 33, 62, 0.95)',
    borderRadius: '8px',
    padding: '12px',
    minWidth: '200px',
    maxWidth: '280px',
    maxHeight: '300px',
    overflowY: 'auto' as const,
  },
  header: {
    display: 'flex',
    justifyContent: 'space-between',
    alignItems: 'center',
    marginBottom: '10px',
  },
  title: {
    fontSize: '12px',
    fontWeight: 600,
    color: '#4ecdc4',
    textTransform: 'uppercase' as const,
    letterSpacing: '0.5px',
  },
  lineItem: {
    display: 'flex',
    alignItems: 'center',
    padding: '8px',
    backgroundColor: '#0a0f1a',
    borderRadius: '4px',
    marginBottom: '6px',
    cursor: 'pointer',
  },
  lineItemActive: {
    backgroundColor: '#1a3a4a',
    border: '1px solid #4ecdc4',
  },
  lineName: {
    flex: 1,
    fontSize: '13px',
    color: '#ddd',
    overflow: 'hidden',
    textOverflow: 'ellipsis',
    whiteSpace: 'nowrap' as const,
  },
  lineActions: {
    display: 'flex',
    gap: '4px',
  },
  iconButton: {
    padding: '2px 6px',
    fontSize: '10px',
    backgroundColor: 'transparent',
    color: '#888',
    border: '1px solid #444',
    borderRadius: '3px',
    cursor: 'pointer',
  },
  actionBar: {
    display: 'flex',
    gap: '6px',
    marginTop: '10px',
    paddingTop: '10px',
    borderTop: '1px solid #0f3460',
  },
  button: {
    flex: 1,
    padding: '6px 8px',
    fontSize: '11px',
    backgroundColor: '#0f3460',
    color: '#aaa',
    border: '1px solid #1a1a2e',
    borderRadius: '4px',
    cursor: 'pointer',
  },
  primaryButton: {
    backgroundColor: '#4ecdc4',
    color: '#000',
    borderColor: '#4ecdc4',
  },
  smoothingRow: {
    display: 'flex',
    alignItems: 'center',
    gap: '8px',
    marginTop: '8px',
    fontSize: '11px',
    color: '#888',
  },
  smallInput: {
    width: '50px',
    padding: '4px 6px',
    fontSize: '11px',
    backgroundColor: '#0f3460',
    color: '#eee',
    border: '1px solid #1a1a2e',
    borderRadius: '3px',
    textAlign: 'center' as const,
  },
  emptyState: {
    fontSize: '12px',
    color: '#666',
    textAlign: 'center' as const,
    padding: '12px',
  },
  projectionInfo: {
    marginTop: '8px',
    padding: '8px',
    backgroundColor: '#0a0f1a',
    borderRadius: '4px',
    fontSize: '11px',
    color: '#aaa',
  },
}

export default function LinePanel() {
  const {
    drawnLines,
    activeLineId,
    setActiveLine,
    removeLine,
    renameLine,
    smoothLine,
    lineSmoothingParams,
    setLineSmoothingParams,
    projectCellsOntoLine,
    cellProjections,
    clearProjections,
  } = useStore()

  const [editingId, setEditingId] = useState<string | null>(null)
  const [editName, setEditName] = useState('')

  const handleStartEdit = useCallback((id: string, name: string) => {
    setEditingId(id)
    setEditName(name)
  }, [])

  const handleSaveEdit = useCallback(() => {
    if (editingId && editName.trim()) {
      renameLine(editingId, editName.trim())
    }
    setEditingId(null)
    setEditName('')
  }, [editingId, editName, renameLine])

  const activeLine = drawnLines.find((l) => l.id === activeLineId)

  if (drawnLines.length === 0) {
    return null
  }

  return (
    <div style={styles.panel}>
      <div style={styles.header}>
        <span style={styles.title}>Lines ({drawnLines.length})</span>
      </div>

      {drawnLines.map((line) => (
        <div
          key={line.id}
          style={{
            ...styles.lineItem,
            ...(line.id === activeLineId ? styles.lineItemActive : {}),
          }}
          onClick={() => setActiveLine(line.id === activeLineId ? null : line.id)}
        >
          {editingId === line.id ? (
            <input
              type="text"
              value={editName}
              onChange={(e) => setEditName(e.target.value)}
              onBlur={handleSaveEdit}
              onKeyDown={(e) => {
                if (e.key === 'Enter') handleSaveEdit()
                if (e.key === 'Escape') {
                  setEditingId(null)
                  setEditName('')
                }
              }}
              onClick={(e) => e.stopPropagation()}
              autoFocus
              style={{
                ...styles.smallInput,
                flex: 1,
                width: 'auto',
                textAlign: 'left' as const,
              }}
            />
          ) : (
            <span
              style={styles.lineName}
              onDoubleClick={(e) => {
                e.stopPropagation()
                handleStartEdit(line.id, line.name)
              }}
              title={`${line.name}${line.smoothedPoints ? ' (smoothed)' : ''}\nDouble-click to rename`}
            >
              {line.name}
              {line.smoothedPoints && <span style={{ color: '#4ecdc4', marginLeft: '4px' }}>~</span>}
            </span>
          )}
          <div style={styles.lineActions}>
            <button
              style={styles.iconButton}
              onClick={(e) => {
                e.stopPropagation()
                removeLine(line.id)
              }}
              title="Delete line"
            >
              x
            </button>
          </div>
        </div>
      ))}

      {activeLineId && activeLine && (
        <>
          <div style={styles.smoothingRow}>
            <span>Smooth:</span>
            <span>Window</span>
            <input
              type="number"
              min="3"
              max="21"
              step="2"
              value={lineSmoothingParams.windowSize}
              onChange={(e) => setLineSmoothingParams({ windowSize: parseInt(e.target.value) || 5 })}
              style={styles.smallInput}
            />
            <span>Iter</span>
            <input
              type="number"
              min="1"
              max="10"
              value={lineSmoothingParams.iterations}
              onChange={(e) => setLineSmoothingParams({ iterations: parseInt(e.target.value) || 1 })}
              style={styles.smallInput}
            />
          </div>

          <div style={styles.actionBar}>
            <button
              style={styles.button}
              onClick={() => smoothLine(activeLineId)}
              title="Apply smoothing to the line"
            >
              Smooth
            </button>
            <button
              style={{ ...styles.button, ...styles.primaryButton }}
              onClick={projectCellsOntoLine}
              title="Project active cells onto this line"
            >
              Project
            </button>
          </div>

          {cellProjections && cellProjections.length > 0 && (
            <div style={styles.projectionInfo}>
              <div style={{ marginBottom: '4px', fontWeight: 500, color: '#4ecdc4' }}>
                Projections: {cellProjections.length} cells
              </div>
              <div>
                Avg distance: {(cellProjections.reduce((s, p) => s + p.distanceToLine, 0) / cellProjections.length).toFixed(3)}
              </div>
              <button
                style={{ ...styles.button, marginTop: '6px', width: '100%' }}
                onClick={clearProjections}
              >
                Clear Projections
              </button>
            </div>
          )}
        </>
      )}
    </div>
  )
}
