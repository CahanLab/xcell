import { useState, useCallback, useEffect } from 'react'
import { useStore } from '../store'
import { useLineAssociation, createLineEmbedding, appendDataset } from '../hooks/useData'

const API_BASE = '/api'

const styles = {
  panel: {
    width: '280px',
    flex: '0 0 auto',
    backgroundColor: '#16213e',
    borderRight: '1px solid #0f3460',
    borderTop: '1px solid #0f3460',
    display: 'flex',
    flexDirection: 'column' as const,
    overflow: 'hidden',
  },
  header: {
    padding: '8px 16px',
    borderBottom: '1px solid #0f3460',
    display: 'flex',
    justifyContent: 'space-between',
    alignItems: 'center',
    cursor: 'pointer',
    userSelect: 'none' as const,
  },
  title: {
    fontSize: '14px',
    fontWeight: 600,
    color: '#4ecdc4',
  },
  collapseIcon: {
    fontSize: '10px',
    color: '#888',
  },
  content: {
    flex: 1,
    overflowY: 'auto' as const,
    padding: '8px 0',
  },
  lineRow: {
    display: 'flex',
    alignItems: 'center',
    padding: '8px 16px',
    gap: '6px',
    borderBottom: '1px solid #0a0f1a',
  },
  lineRowActive: {
    backgroundColor: '#0f3460',
  },
  lineName: {
    flex: 1,
    fontSize: '12px',
    color: '#ddd',
    overflow: 'hidden',
    textOverflow: 'ellipsis',
    whiteSpace: 'nowrap' as const,
    cursor: 'pointer',
  },
  lineNameInput: {
    flex: 1,
    padding: '2px 6px',
    fontSize: '12px',
    backgroundColor: '#0a0f1a',
    color: '#eee',
    border: '1px solid #1a1a2e',
    borderRadius: '3px',
    outline: 'none',
  },
  cellCount: {
    fontSize: '11px',
    color: '#888',
    minWidth: '24px',
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
  // Modal styles
  overlay: {
    position: 'fixed' as const,
    top: 0,
    left: 0,
    right: 0,
    bottom: 0,
    backgroundColor: 'rgba(0, 0, 0, 0.7)',
    display: 'flex',
    alignItems: 'center',
    justifyContent: 'center',
    zIndex: 2000,
  },
  modal: {
    backgroundColor: '#16213e',
    borderRadius: '12px',
    border: '1px solid #0f3460',
    width: '480px',
    maxWidth: '95vw',
    maxHeight: '80vh',
    display: 'flex',
    flexDirection: 'column' as const,
    boxShadow: '0 8px 32px rgba(0, 0, 0, 0.4)',
  },
  modalHeader: {
    padding: '16px 20px',
    borderBottom: '1px solid #0f3460',
    display: 'flex',
    justifyContent: 'space-between',
    alignItems: 'center',
  },
  modalTitle: {
    fontSize: '16px',
    fontWeight: 600,
    color: '#e94560',
    margin: 0,
  },
  modalClose: {
    background: 'none',
    border: 'none',
    color: '#aaa',
    fontSize: '20px',
    cursor: 'pointer',
    padding: '4px 8px',
  },
  modalContent: {
    flex: 1,
    overflow: 'auto',
    padding: '20px',
  },
  section: {
    marginBottom: '20px',
  },
  sectionTitle: {
    fontSize: '12px',
    fontWeight: 600,
    color: '#4ecdc4',
    marginBottom: '8px',
  },
  row: {
    display: 'flex',
    alignItems: 'center',
    gap: '8px',
    fontSize: '12px',
    color: '#ccc',
  },
  smallInput: {
    width: '48px',
    padding: '4px 6px',
    fontSize: '11px',
    backgroundColor: '#0f3460',
    color: '#eee',
    border: '1px solid #1a1a2e',
    borderRadius: '4px',
    textAlign: 'center' as const,
  },
  actionButton: {
    padding: '6px 12px',
    fontSize: '11px',
    backgroundColor: '#0f3460',
    color: '#aaa',
    border: '1px solid #1a1a2e',
    borderRadius: '4px',
    cursor: 'pointer',
  },
  primaryActionButton: {
    padding: '6px 12px',
    fontSize: '11px',
    backgroundColor: '#4ecdc4',
    color: '#000',
    border: '1px solid #4ecdc4',
    borderRadius: '4px',
    cursor: 'pointer',
    fontWeight: 500,
  },
  pillButton: {
    padding: '3px 8px',
    fontSize: '10px',
    borderRadius: '10px',
    cursor: 'pointer',
    border: '1px solid #1a1a2e',
  },
  toggleButton: {
    padding: '3px 10px',
    fontSize: '10px',
    borderRadius: '4px',
    cursor: 'pointer',
    border: '1px solid #1a1a2e',
  },
}

// ---------- Line Tools Modal ----------

function LineToolsModal({
  line,
  onClose,
}: {
  line: ReturnType<typeof useStore.getState>['drawnLines'][0]
  onClose: () => void
}) {
  const {
    drawnLines,
    activeCellMask,
    smoothLine,
    clearLineProjections,
    lineSmoothingParams,
    setLineSmoothingParams,
    setSchema,
  } = useStore()

  const [associationError, setAssociationError] = useState<string | null>(null)
  const [isCreatingEmbedding, setIsCreatingEmbedding] = useState(false)
  const [embeddingMessage, setEmbeddingMessage] = useState<string | null>(null)
  const [geneSubsetColumns, setGeneSubsetColumns] = useState<{ name: string; n_true: number; n_total: number }[]>([])
  const [selectedGeneColumns, setSelectedGeneColumns] = useState<string[]>([])
  const [geneSubsetOperation, setGeneSubsetOperation] = useState<'intersection' | 'union'>('intersection')
  const [testVariable, setTestVariable] = useState<'position' | 'distance'>('position')

  const { runAssociation, isLineAssociationLoading } = useLineAssociation()

  const scanpyActionHistory = useStore((state) => state.scanpyActionHistory)
  useEffect(() => {
    fetch(appendDataset(`${API_BASE}/var/boolean_columns`))
      .then((res) => res.json())
      .then(setGeneSubsetColumns)
      .catch(() => setGeneSubsetColumns([]))
  }, [scanpyActionHistory])

  const getCellIndices = useCallback(() => {
    if (!activeCellMask) return undefined
    return activeCellMask
      .map((visible, idx) => (visible ? idx : -1))
      .filter((idx) => idx >= 0)
  }, [activeCellMask])

  const handleFindAssociatedGenes = useCallback(async () => {
    setAssociationError(null)
    try {
      const cellIndices = getCellIndices()
      let geneSubset: string | { columns: string[]; operation: string } | null = null
      if (selectedGeneColumns.length === 1) {
        geneSubset = selectedGeneColumns[0]
      } else if (selectedGeneColumns.length > 1) {
        geneSubset = { columns: selectedGeneColumns, operation: geneSubsetOperation }
      }
      await runAssociation(line.name, { cellIndices, geneSubset, testVariable })
      onClose()
    } catch (err) {
      setAssociationError((err as Error).message)
    }
  }, [runAssociation, getCellIndices, selectedGeneColumns, geneSubsetOperation, testVariable, line.name, onClose])

  const handleCreateProjectionEmbedding = useCallback(async () => {
    setEmbeddingMessage(null)
    setIsCreatingEmbedding(true)
    try {
      const cellIndices = getCellIndices()
      const result = await createLineEmbedding(
        { lineName: line.name, cellIndices },
        drawnLines
      )
      const schemaResponse = await fetch(appendDataset('/api/schema'))
      const schema = await schemaResponse.json()
      setSchema(schema)
      setEmbeddingMessage(`Created embedding "${result.embedding_name}"`)
    } catch (err) {
      setEmbeddingMessage(`Error: ${(err as Error).message}`)
    } finally {
      setIsCreatingEmbedding(false)
    }
  }, [drawnLines, setSchema, getCellIndices, line.name])

  return (
    <div style={styles.overlay} onClick={onClose}>
      <div style={styles.modal} onClick={(e) => e.stopPropagation()}>
        <div style={styles.modalHeader}>
          <h2 style={styles.modalTitle}>Line Tools: {line.name}</h2>
          <button style={styles.modalClose} onClick={onClose}>&times;</button>
        </div>

        <div style={styles.modalContent}>
          {/* Smoothing */}
          <div style={styles.section}>
            <div style={styles.sectionTitle}>Smoothing</div>
            <div style={styles.row}>
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
              <span>Iterations</span>
              <input
                type="number"
                min="1"
                max="10"
                value={lineSmoothingParams.iterations}
                onChange={(e) => setLineSmoothingParams({ iterations: parseInt(e.target.value) || 1 })}
                style={styles.smallInput}
              />
              <button
                style={styles.actionButton}
                onClick={() => smoothLine(line.id)}
                title="Apply smoothing to this line"
              >
                Smooth
              </button>
            </div>
            {line.smoothedPoints && (
              <div style={{ marginTop: '6px', fontSize: '11px', color: '#4ecdc4' }}>
                Line is smoothed
              </div>
            )}
          </div>

          {/* Projections */}
          {line.projections.length > 0 && (
            <div style={styles.section}>
              <div style={styles.sectionTitle}>Projections</div>
              <div style={styles.row}>
                <span>{line.projections.length} cells projected</span>
                <button
                  style={{ ...styles.actionButton, color: '#e94560' }}
                  onClick={() => clearLineProjections(line.id)}
                  title="Clear all projections for this line"
                >
                  Clear Projections
                </button>
              </div>
            </div>
          )}

          {/* Gene Association */}
          <div style={styles.section}>
            <div style={styles.sectionTitle}>Gene Association</div>

            {/* Gene subset selector */}
            {geneSubsetColumns.length > 0 && (
              <div style={{ marginBottom: '10px' }}>
                <div style={{ fontSize: '11px', color: '#888', marginBottom: '6px' }}>
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
                            setSelectedGeneColumns(selectedGeneColumns.filter((c) => c !== col.name))
                          } else {
                            setSelectedGeneColumns([...selectedGeneColumns, col.name])
                          }
                        }}
                        style={{
                          ...styles.pillButton,
                          backgroundColor: isSelected ? '#4ecdc4' : '#0f3460',
                          color: isSelected ? '#000' : '#aaa',
                          borderColor: isSelected ? '#4ecdc4' : '#1a1a2e',
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
                  <div style={{ marginTop: '6px', display: 'flex', alignItems: 'center', gap: '6px', fontSize: '10px' }}>
                    <span style={{ color: '#888' }}>Combine:</span>
                    <button
                      onClick={() => setGeneSubsetOperation('intersection')}
                      style={{
                        ...styles.toggleButton,
                        backgroundColor: geneSubsetOperation === 'intersection' ? '#4ecdc4' : '#0f3460',
                        color: geneSubsetOperation === 'intersection' ? '#000' : '#aaa',
                      }}
                    >
                      AND
                    </button>
                    <button
                      onClick={() => setGeneSubsetOperation('union')}
                      style={{
                        ...styles.toggleButton,
                        backgroundColor: geneSubsetOperation === 'union' ? '#4ecdc4' : '#0f3460',
                        color: geneSubsetOperation === 'union' ? '#000' : '#aaa',
                      }}
                    >
                      OR
                    </button>
                  </div>
                )}
              </div>
            )}

            {/* Test variable */}
            <div style={{ marginBottom: '10px' }}>
              <div style={{ fontSize: '11px', color: '#888', marginBottom: '6px' }}>Test against:</div>
              <div style={{ display: 'flex', gap: '6px' }}>
                <button
                  onClick={() => setTestVariable('position')}
                  style={{
                    ...styles.toggleButton,
                    backgroundColor: testVariable === 'position' ? '#4ecdc4' : '#0f3460',
                    color: testVariable === 'position' ? '#000' : '#aaa',
                    borderColor: testVariable === 'position' ? '#4ecdc4' : '#1a1a2e',
                    fontWeight: testVariable === 'position' ? 600 : 400,
                  }}
                  title="Test gene expression vs. position along the line (0=start, 1=end)"
                >
                  Position along line
                </button>
                <button
                  onClick={() => setTestVariable('distance')}
                  style={{
                    ...styles.toggleButton,
                    backgroundColor: testVariable === 'distance' ? '#4ecdc4' : '#0f3460',
                    color: testVariable === 'distance' ? '#000' : '#aaa',
                    borderColor: testVariable === 'distance' ? '#4ecdc4' : '#1a1a2e',
                    fontWeight: testVariable === 'distance' ? 600 : 400,
                  }}
                  title="Test gene expression vs. perpendicular distance from the line"
                >
                  Distance from line
                </button>
              </div>
            </div>

            <button
              style={{
                ...styles.primaryActionButton,
                opacity: isLineAssociationLoading ? 0.6 : 1,
              }}
              onClick={handleFindAssociatedGenes}
              disabled={isLineAssociationLoading}
              title={testVariable === 'position'
                ? 'Find genes whose expression is associated with position along this line'
                : 'Find genes whose expression is associated with distance from this line'}
            >
              {isLineAssociationLoading ? 'Analyzing...' : 'Find Associated Genes'}
            </button>
            {associationError && (
              <div style={{ marginTop: '6px', fontSize: '11px', color: '#e94560' }}>
                {associationError}
              </div>
            )}
          </div>

          {/* Projection Embedding */}
          <div style={styles.section}>
            <div style={styles.sectionTitle}>Projection Embedding</div>
            <button
              style={{
                ...styles.actionButton,
                opacity: isCreatingEmbedding ? 0.6 : 1,
              }}
              onClick={handleCreateProjectionEmbedding}
              disabled={isCreatingEmbedding}
              title="Create a new embedding where X=position along line, Y=distance from line"
            >
              {isCreatingEmbedding ? 'Creating...' : 'Create Projection Embedding'}
            </button>
            {embeddingMessage && (
              <div style={{
                marginTop: '6px',
                fontSize: '11px',
                color: embeddingMessage.startsWith('Error') ? '#e94560' : '#4ecdc4',
              }}>
                {embeddingMessage}
              </div>
            )}
          </div>
        </div>
      </div>
    </div>
  )
}

// ---------- Lines Panel ----------

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
  } = useStore()

  const [editingId, setEditingId] = useState<string | null>(null)
  const [editName, setEditName] = useState('')
  const [toolsLineId, setToolsLineId] = useState<string | null>(null)
  const [collapsed, setCollapsed] = useState(false)

  const currentEmbeddingLines = drawnLines.filter(
    (l) => l.embeddingName === selectedEmbedding
  )
  const hasAnyLines = drawnLines.length > 0
  const toolsLine = toolsLineId ? drawnLines.find((l) => l.id === toolsLineId) : null

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

  return (
    <>
      <div style={{
        ...styles.panel,
        ...(collapsed ? { minHeight: 0 } : { minHeight: '80px', maxHeight: '250px' }),
      }}>
        <div
          style={styles.header}
          onClick={() => setCollapsed(!collapsed)}
          title={collapsed ? 'Expand Lines panel' : 'Collapse Lines panel'}
        >
          <span style={styles.title}>Lines</span>
          <span style={styles.collapseIcon}>{collapsed ? '\u25B6' : '\u25BC'}</span>
        </div>

        {!collapsed && (
          <div style={styles.content}>
            {currentEmbeddingLines.length === 0 && (
              <div style={styles.emptyState}>
                {hasAnyLines
                  ? `No lines on "${selectedEmbedding}". Draw a line using the Draw tool.`
                  : 'No lines yet. Use the Draw tool to create a line.'}
              </div>
            )}
            {currentEmbeddingLines.map((line) => (
              <div
                key={line.id}
                style={{
                  ...styles.lineRow,
                  ...(line.id === activeLineId ? styles.lineRowActive : {}),
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
                    style={styles.lineNameInput}
                  />
                ) : (
                  <span
                    style={styles.lineName}
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

                {/* Tools button */}
                <button
                  style={styles.iconButton}
                  onClick={(e) => {
                    e.stopPropagation()
                    setToolsLineId(line.id)
                  }}
                  title="Line tools: smooth, analyze, project"
                >
                  &#9881;
                </button>

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
                  {line.visible ? '\u25CF' : '\u25CB'}
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
                  &times;
                </button>
              </div>
            ))}
          </div>
        )}
      </div>

      {/* Line Tools Modal */}
      {toolsLine && (
        <LineToolsModal
          line={toolsLine}
          onClose={() => setToolsLineId(null)}
        />
      )}
    </>
  )
}
