import { useState, useCallback, useEffect } from 'react'
import { useStore } from '../store'
import { useLineAssociation, createLineEmbedding } from '../hooks/useData'

const API_BASE = '/api'

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
    activeCellMask,
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
  const [associationError, setAssociationError] = useState<string | null>(null)
  const [isCreatingEmbedding, setIsCreatingEmbedding] = useState(false)
  const [embeddingMessage, setEmbeddingMessage] = useState<string | null>(null)
  const [geneSubsetColumns, setGeneSubsetColumns] = useState<{ name: string; n_true: number; n_total: number }[]>([])
  const [selectedGeneColumns, setSelectedGeneColumns] = useState<string[]>([])
  const [geneSubsetOperation, setGeneSubsetOperation] = useState<'intersection' | 'union'>('intersection')
  const [testVariable, setTestVariable] = useState<'position' | 'distance'>('position')

  const { runAssociation, isLineAssociationLoading } = useLineAssociation()

  // Fetch available boolean columns for gene subsetting (on mount + after scanpy actions)
  const scanpyActionHistory = useStore((state) => state.scanpyActionHistory)
  useEffect(() => {
    fetch(`${API_BASE}/var/boolean_columns`)
      .then((res) => res.json())
      .then(setGeneSubsetColumns)
      .catch(() => setGeneSubsetColumns([]))
  }, [scanpyActionHistory])

  // Get setSchema to force refresh after creating embedding
  const setSchema = useStore((state) => state.setSchema)

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

  const handleFindAssociatedGenes = useCallback(async (line: typeof drawnLines[0]) => {
    setAssociationError(null)
    try {
      // Determine which cells to use:
      // 1. If active cell mask exists, use only masked (visible) cells
      // 2. Otherwise, use all cells
      let cellIndices: number[] | undefined = undefined

      if (activeCellMask) {
        // Get indices where mask is true
        cellIndices = activeCellMask
          .map((visible, idx) => visible ? idx : -1)
          .filter(idx => idx >= 0)
      }

      // Build gene subset spec
      let geneSubset: string | { columns: string[]; operation: string } | null = null
      if (selectedGeneColumns.length === 1) {
        geneSubset = selectedGeneColumns[0]
      } else if (selectedGeneColumns.length > 1) {
        geneSubset = { columns: selectedGeneColumns, operation: geneSubsetOperation }
      }

      await runAssociation(line.name, { cellIndices, geneSubset, testVariable })
    } catch (err) {
      setAssociationError((err as Error).message)
    }
  }, [runAssociation, activeCellMask, selectedGeneColumns, geneSubsetOperation, testVariable])

  const handleCreateProjectionEmbedding = useCallback(async (line: typeof drawnLines[0]) => {
    setEmbeddingMessage(null)
    setIsCreatingEmbedding(true)
    try {
      // Determine which cells to use:
      // 1. If active cell mask exists, use only masked (visible) cells
      // 2. Otherwise, use all cells
      let cellIndices: number[] | undefined = undefined

      if (activeCellMask) {
        cellIndices = activeCellMask
          .map((visible, idx) => visible ? idx : -1)
          .filter(idx => idx >= 0)
      }

      const result = await createLineEmbedding(
        { lineName: line.name, cellIndices },
        drawnLines
      )

      // Force schema refresh to pick up the new embedding
      const schemaResponse = await fetch('/api/schema')
      const schema = await schemaResponse.json()
      setSchema(schema)

      setEmbeddingMessage(`Created embedding "${result.embedding_name}"`)
    } catch (err) {
      setEmbeddingMessage(`Error: ${(err as Error).message}`)
    } finally {
      setIsCreatingEmbedding(false)
    }
  }, [drawnLines, setSchema, activeCellMask])

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

          {/* Gene subset selector for association */}
          {geneSubsetColumns.length > 0 && (
            <div style={{ marginTop: '8px' }}>
              <div style={{ fontSize: '10px', color: '#888', marginBottom: '4px' }}>
                Genes {selectedGeneColumns.length === 0 ? '(all)' : ''}:
              </div>
              <div style={{ display: 'flex', flexWrap: 'wrap', gap: '4px' }}>
                {geneSubsetColumns.map((col) => {
                  const isSelected = selectedGeneColumns.includes(col.name)
                  return (
                    <button
                      key={col.name}
                      onClick={() => {
                        if (isSelected) {
                          setSelectedGeneColumns(selectedGeneColumns.filter(c => c !== col.name))
                        } else {
                          setSelectedGeneColumns([...selectedGeneColumns, col.name])
                        }
                      }}
                      style={{
                        padding: '2px 6px',
                        fontSize: '9px',
                        backgroundColor: isSelected ? '#4ecdc4' : '#0f3460',
                        color: isSelected ? '#000' : '#aaa',
                        border: `1px solid ${isSelected ? '#4ecdc4' : '#1a1a2e'}`,
                        borderRadius: '10px',
                        cursor: 'pointer',
                        fontWeight: isSelected ? 600 : 400,
                      }}
                      title={`${col.n_true.toLocaleString()} of ${col.n_total.toLocaleString()} genes`}
                    >
                      {col.name} ({col.n_true.toLocaleString()})
                    </button>
                  )
                })}
              </div>
              {selectedGeneColumns.length >= 2 && (
                <div style={{ marginTop: '4px', display: 'flex', alignItems: 'center', gap: '4px', fontSize: '9px' }}>
                  <span style={{ color: '#888' }}>Combine:</span>
                  <button
                    onClick={() => setGeneSubsetOperation('intersection')}
                    style={{
                      padding: '1px 5px',
                      fontSize: '9px',
                      backgroundColor: geneSubsetOperation === 'intersection' ? '#4ecdc4' : '#0f3460',
                      color: geneSubsetOperation === 'intersection' ? '#000' : '#aaa',
                      border: '1px solid #1a1a2e',
                      borderRadius: '3px',
                      cursor: 'pointer',
                    }}
                  >
                    AND
                  </button>
                  <button
                    onClick={() => setGeneSubsetOperation('union')}
                    style={{
                      padding: '1px 5px',
                      fontSize: '9px',
                      backgroundColor: geneSubsetOperation === 'union' ? '#4ecdc4' : '#0f3460',
                      color: geneSubsetOperation === 'union' ? '#000' : '#aaa',
                      border: '1px solid #1a1a2e',
                      borderRadius: '3px',
                      cursor: 'pointer',
                    }}
                  >
                    OR
                  </button>
                </div>
              )}
            </div>
          )}

          {/* Test variable toggle */}
          <div style={{ marginTop: '8px' }}>
            <div style={{ fontSize: '10px', color: '#888', marginBottom: '4px' }}>
              Test against:
            </div>
            <div style={{ display: 'flex', gap: '4px' }}>
              <button
                onClick={() => setTestVariable('position')}
                style={{
                  padding: '2px 8px',
                  fontSize: '9px',
                  backgroundColor: testVariable === 'position' ? '#4ecdc4' : '#0f3460',
                  color: testVariable === 'position' ? '#000' : '#aaa',
                  border: `1px solid ${testVariable === 'position' ? '#4ecdc4' : '#1a1a2e'}`,
                  borderRadius: '3px',
                  cursor: 'pointer',
                  fontWeight: testVariable === 'position' ? 600 : 400,
                }}
                title="Test gene expression vs. position along the line (0=start, 1=end)"
              >
                Position along line
              </button>
              <button
                onClick={() => setTestVariable('distance')}
                style={{
                  padding: '2px 8px',
                  fontSize: '9px',
                  backgroundColor: testVariable === 'distance' ? '#4ecdc4' : '#0f3460',
                  color: testVariable === 'distance' ? '#000' : '#aaa',
                  border: `1px solid ${testVariable === 'distance' ? '#4ecdc4' : '#1a1a2e'}`,
                  borderRadius: '3px',
                  cursor: 'pointer',
                  fontWeight: testVariable === 'distance' ? 600 : 400,
                }}
                title="Test gene expression vs. perpendicular distance from the line"
              >
                Distance from line
              </button>
            </div>
          </div>

          {/* Find Associated Genes button */}
          <div style={{ marginTop: '8px' }}>
            <button
              style={{
                ...styles.smoothButton,
                marginLeft: 0,
                backgroundColor: '#4ecdc4',
                color: '#000',
                fontWeight: 500,
                opacity: isLineAssociationLoading ? 0.6 : 1,
              }}
              onClick={() => handleFindAssociatedGenes(activeLine)}
              disabled={isLineAssociationLoading}
              title={testVariable === 'position'
                ? "Find genes whose expression is associated with position along this line"
                : "Find genes whose expression is associated with distance from this line"
              }
            >
              {isLineAssociationLoading ? 'Analyzing...' : 'Find Associated Genes'}
            </button>
            {associationError && (
              <div style={{ marginTop: '4px', fontSize: '10px', color: '#e94560' }}>
                {associationError}
              </div>
            )}
          </div>

          {/* Create Projection Embedding button */}
          <div style={{ marginTop: '6px' }}>
            <button
              style={{
                ...styles.smoothButton,
                marginLeft: 0,
                opacity: isCreatingEmbedding ? 0.6 : 1,
              }}
              onClick={() => handleCreateProjectionEmbedding(activeLine)}
              disabled={isCreatingEmbedding}
              title="Create a new embedding where X=position along line, Y=distance from line"
            >
              {isCreatingEmbedding ? 'Creating...' : 'Create Projection Embedding'}
            </button>
            {embeddingMessage && (
              <div style={{
                marginTop: '4px',
                fontSize: '10px',
                color: embeddingMessage.startsWith('Error') ? '#e94560' : '#4ecdc4',
              }}>
                {embeddingMessage}
              </div>
            )}
          </div>
        </div>
      )}
    </div>
  )
}
