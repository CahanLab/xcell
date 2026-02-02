import { useState, useCallback } from 'react'
import { useStore } from '../store'

const styles = {
  panel: {
    width: '280px',
    flex: '0 0 auto',
    minHeight: '100px',
    maxHeight: '250px',
    backgroundColor: '#16213e',
    borderRight: '1px solid #0f3460',
    borderTop: '1px solid #0f3460',
    display: 'flex',
    flexDirection: 'column' as const,
    overflow: 'hidden',
  },
  header: {
    padding: '12px 16px',
    borderBottom: '1px solid #0f3460',
    display: 'flex',
    justifyContent: 'space-between',
    alignItems: 'center',
  },
  title: {
    fontSize: '14px',
    fontWeight: 600,
    color: '#4ecdc4',
  },
  content: {
    flex: 1,
    overflowY: 'auto' as const,
    padding: '8px 0',
  },
  shapeRow: {
    display: 'flex',
    alignItems: 'center',
    padding: '8px 16px',
    gap: '8px',
    borderBottom: '1px solid #0a0f1a',
  },
  shapeRowActive: {
    backgroundColor: '#0f3460',
  },
  shapeName: {
    flex: 1,
    fontSize: '12px',
    color: '#ddd',
    overflow: 'hidden',
    textOverflow: 'ellipsis',
    whiteSpace: 'nowrap' as const,
    cursor: 'pointer',
  },
  shapeNameInput: {
    flex: 1,
    padding: '2px 6px',
    fontSize: '12px',
    backgroundColor: '#0a0f1a',
    color: '#eee',
    border: '1px solid #1a1a2e',
    borderRadius: '3px',
    outline: 'none',
  },
  embeddingBadge: {
    fontSize: '9px',
    color: '#666',
    backgroundColor: '#0a0f1a',
    padding: '2px 4px',
    borderRadius: '2px',
  },
  cellCount: {
    fontSize: '11px',
    color: '#888',
    minWidth: '30px',
    textAlign: 'right' as const,
  },
  iconButton: {
    padding: '4px 6px',
    fontSize: '12px',
    backgroundColor: 'transparent',
    color: '#888',
    border: '1px solid #444',
    borderRadius: '3px',
    cursor: 'pointer',
    display: 'flex',
    alignItems: 'center',
    justifyContent: 'center',
  },
  iconButtonActive: {
    backgroundColor: '#4ecdc4',
    color: '#000',
    borderColor: '#4ecdc4',
  },
  iconButtonDisabled: {
    opacity: 0.4,
    cursor: 'not-allowed',
  },
  addButton: {
    backgroundColor: '#4ecdc4',
    color: '#000',
    borderColor: '#4ecdc4',
  },
  emptyState: {
    padding: '12px 16px',
    textAlign: 'center' as const,
    color: '#666',
    fontSize: '11px',
    lineHeight: '1.4',
  },
  smoothingSection: {
    padding: '8px 16px',
    borderTop: '1px solid #0f3460',
    backgroundColor: '#0a0f1a',
  },
  smoothingRow: {
    display: 'flex',
    alignItems: 'center',
    gap: '8px',
    fontSize: '11px',
    color: '#888',
  },
  smallInput: {
    width: '40px',
    padding: '3px 4px',
    fontSize: '10px',
    backgroundColor: '#0f3460',
    color: '#eee',
    border: '1px solid #1a1a2e',
    borderRadius: '3px',
    textAlign: 'center' as const,
  },
  smoothButton: {
    padding: '4px 8px',
    fontSize: '10px',
    backgroundColor: '#0f3460',
    color: '#aaa',
    border: '1px solid #1a1a2e',
    borderRadius: '3px',
    cursor: 'pointer',
    marginLeft: 'auto',
  },
}

export default function ShapeManager() {
  const {
    drawnLines,
    activeLineId,
    selectedEmbedding,
    selectedCellIndices,
    setActiveLine,
    removeLine,
    renameLine,
    setLineVisibility,
    projectSelectedCellsOntoLine,
    clearLineProjections,
    smoothLine,
    lineSmoothingParams,
    setLineSmoothingParams,
  } = useStore()

  const [editingId, setEditingId] = useState<string | null>(null)
  const [editName, setEditName] = useState('')

  // Filter lines to show only those for current embedding
  const currentEmbeddingLines = drawnLines.filter(
    (l) => l.embeddingName === selectedEmbedding
  )

  const handleStartEdit = useCallback((id: string, name: string, e: React.MouseEvent) => {
    e.stopPropagation()
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

  const handleProjectCells = useCallback((lineId: string, e: React.MouseEvent) => {
    e.stopPropagation()
    if (selectedCellIndices.length === 0) {
      alert('Select cells first using the lasso tool or by clicking a category in the Cells panel.')
      return
    }
    projectSelectedCellsOntoLine(lineId)
  }, [selectedCellIndices, projectSelectedCellsOntoLine])

  const activeLine = drawnLines.find((l) => l.id === activeLineId)

  // Always show the panel, even if empty
  const hasAnyLines = drawnLines.length > 0

  return (
    <div style={styles.panel}>
      <div style={styles.header}>
        <span style={styles.title}>Shapes</span>
      </div>

      <div style={styles.content}>
        {currentEmbeddingLines.length === 0 && (
          <div style={styles.emptyState}>
            {hasAnyLines
              ? `No shapes on "${selectedEmbedding}". Draw a line using the Draw tool.`
              : 'No shapes yet. Use the Draw tool to create a line.'}
          </div>
        )}
        {currentEmbeddingLines.map((line) => (
          <div
            key={line.id}
            style={{
              ...styles.shapeRow,
              ...(line.id === activeLineId ? styles.shapeRowActive : {}),
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
                style={styles.shapeNameInput}
              />
            ) : (
              <span
                style={styles.shapeName}
                onDoubleClick={(e) => handleStartEdit(line.id, line.name, e)}
                title={`${line.name}\nEmbedding: ${line.embeddingName}\nDouble-click to rename`}
              >
                {line.name}
                {line.smoothedPoints && <span style={{ color: '#4ecdc4', marginLeft: '4px' }}>~</span>}
              </span>
            )}

            <span style={styles.cellCount} title={`${line.projections.length} cells projected`}>
              {line.projections.length}
            </span>

            {/* Add cells button */}
            <button
              style={{
                ...styles.iconButton,
                ...styles.addButton,
                ...(selectedCellIndices.length === 0 ? styles.iconButtonDisabled : {}),
              }}
              onClick={(e) => handleProjectCells(line.id, e)}
              title={selectedCellIndices.length > 0
                ? `Project ${selectedCellIndices.length} selected cells onto this line`
                : 'Select cells first to project'}
            >
              +
            </button>

            {/* Visibility toggle */}
            <button
              style={{
                ...styles.iconButton,
                ...(line.visible ? styles.iconButtonActive : {}),
              }}
              onClick={(e) => {
                e.stopPropagation()
                setLineVisibility(line.id, !line.visible)
              }}
              title={line.visible ? 'Hide line' : 'Show line'}
            >
              {line.visible ? '👁' : '○'}
            </button>

            {/* Delete button */}
            <button
              style={styles.iconButton}
              onClick={(e) => {
                e.stopPropagation()
                removeLine(line.id)
              }}
              title="Delete line and projections"
            >
              ×
            </button>
          </div>
        ))}
      </div>

      {/* Smoothing controls for active line */}
      {activeLineId && activeLine && activeLine.embeddingName === selectedEmbedding && (
        <div style={styles.smoothingSection}>
          <div style={styles.smoothingRow}>
            <span>Smooth:</span>
            <span>Win</span>
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
            <button
              style={styles.smoothButton}
              onClick={() => smoothLine(activeLineId)}
              title="Apply smoothing"
            >
              Apply
            </button>
          </div>
          {activeLine.projections.length > 0 && (
            <div style={{ marginTop: '6px' }}>
              <button
                style={{ ...styles.smoothButton, marginLeft: 0, color: '#e94560' }}
                onClick={() => clearLineProjections(activeLineId)}
                title="Clear all projections for this line"
              >
                Clear Projections
              </button>
            </div>
          )}
        </div>
      )}
    </div>
  )
}
