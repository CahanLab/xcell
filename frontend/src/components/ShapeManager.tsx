import { useState, useCallback, useEffect } from 'react'
import { useStore, userConfigGet } from '../store'
import { useLineAssociation, createLineEmbedding, appendDataset, cancelTask } from '../hooks/useData'

// Pull line_association defaults from user config, with hardcoded fallbacks.
// Evaluated lazily inside useState so that if the config loads after the
// component mounts (shouldn't, but possible), any remount gets fresh values.
function laDefault<T>(path: string, fallback: T): T {
  const cfg = useStore.getState().userConfig
  return userConfigGet(cfg, ['line_association', path], fallback)
}

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
    updateLineAppearance,
  } = useStore()

  const [associationError, setAssociationError] = useState<string | null>(null)
  const [isCreatingEmbedding, setIsCreatingEmbedding] = useState(false)
  const [embeddingMessage, setEmbeddingMessage] = useState<string | null>(null)
  const [geneSubsetColumns, setGeneSubsetColumns] = useState<{ name: string; n_true: number; n_total: number }[]>([])
  const [selectedGeneColumns, setSelectedGeneColumns] = useState<string[]>([])
  const [geneSubsetOperation, setGeneSubsetOperation] = useState<'intersection' | 'union'>('intersection')
  const [testVariable, setTestVariable] = useState<'position' | 'distance'>(() => laDefault('test_variable', 'position'))
  const [nSplineKnots, setNSplineKnots] = useState<number>(() => laDefault('n_spline_knots', 5))
  const [fdrThreshold, setFdrThreshold] = useState<number>(() => laDefault('fdr_threshold', 0.05))
  const [topN, setTopN] = useState<number>(() => laDefault('top_n', 50))
  const [clusterGenes, setClusterGenes] = useState<boolean>(() => laDefault('cluster_genes', false))

  const { runAssociation, isLineAssociationLoading } = useLineAssociation()
  const activeTaskId = useStore((state) => state.activeTaskId)

  const scanpyActionHistory = useStore((state) => state.scanpyActionHistory)
  useEffect(() => {
    fetch(appendDataset(`${API_BASE}/var/boolean_columns`))
      .then((res) => res.json())
      .then(setGeneSubsetColumns)
      .catch(() => setGeneSubsetColumns([]))
  }, [scanpyActionHistory])

  const getCellIndices = useCallback(() => {
    // If this line has projected cells, use those
    if (line.projections && line.projections.length > 0) {
      return line.projections.map((p) => p.cellIndex)
    }
    // Otherwise fall back to active cell mask
    if (!activeCellMask) return undefined
    return activeCellMask
      .map((visible, idx) => (visible ? idx : -1))
      .filter((idx) => idx >= 0)
  }, [line.projections, activeCellMask])

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
      await runAssociation(line.name, { cellIndices, geneSubset, testVariable, nSplineKnots, fdrThreshold, topN, clusterGenes })
      onClose()
    } catch (err) {
      const message = (err as Error).message
      if (message !== 'cancelled') {
        setAssociationError(message)
      }
    }
  }, [runAssociation, getCellIndices, selectedGeneColumns, geneSubsetOperation, testVariable, nSplineKnots, fdrThreshold, topN, clusterGenes, line.name, onClose])

  const handleCancelAssociation = useCallback(async () => {
    if (activeTaskId) {
      try { await cancelTask(activeTaskId) } catch { /* may have completed */ }
    }
  }, [activeTaskId])

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

          {/* Appearance */}
          <div style={styles.section}>
            <div style={styles.sectionTitle}>Appearance</div>
            <div style={styles.row}>
              <span>Color</span>
              <input
                type="color"
                value={line.strokeColor || '#4ecdc4'}
                onChange={(e) => updateLineAppearance(line.id, { strokeColor: e.target.value })}
                style={{ width: '32px', height: '24px', border: 'none', background: 'none', cursor: 'pointer' }}
              />
              <span>Width</span>
              <input
                type="number"
                min="1"
                max="10"
                value={line.strokeWidth || 2}
                onChange={(e) => updateLineAppearance(line.id, { strokeWidth: Math.max(1, Math.min(10, parseInt(e.target.value) || 2)) })}
                style={styles.smallInput}
              />
            </div>
            <div style={{ ...styles.row, marginTop: '6px' }}>
              <span>Fill</span>
              <label style={{ display: 'flex', alignItems: 'center', gap: '4px', cursor: 'pointer', fontSize: '12px', color: '#aaa' }}>
                <input
                  type="checkbox"
                  checked={!!line.fillColor}
                  onChange={(e) => updateLineAppearance(line.id, { fillColor: e.target.checked ? (line.strokeColor || '#4ecdc4') : null })}
                />
                Enable
              </label>
              {line.fillColor && (
                <input
                  type="color"
                  value={line.fillColor}
                  onChange={(e) => updateLineAppearance(line.id, { fillColor: e.target.value })}
                  style={{ width: '32px', height: '24px', border: 'none', background: 'none', cursor: 'pointer' }}
                />
              )}
            </div>
            <div style={{ ...styles.row, marginTop: '6px' }}>
              <span>Closed</span>
              <label style={{ display: 'flex', alignItems: 'center', gap: '4px', cursor: 'pointer', fontSize: '12px', color: '#aaa' }}>
                <input
                  type="checkbox"
                  checked={line.closed || false}
                  onChange={(e) => updateLineAppearance(line.id, { closed: e.target.checked })}
                />
                {line.closed ? 'Yes' : 'No'}
              </label>
            </div>
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

            {/* Analysis parameters */}
            <div style={{ marginBottom: '10px' }}>
              <div style={{ fontSize: '11px', color: '#888', marginBottom: '6px' }}>Parameters:</div>
              <div style={{ display: 'flex', alignItems: 'center', gap: '8px', marginBottom: '6px', fontSize: '11px', color: '#ccc' }}>
                <span title="Number of interior knots for the cubic B-spline. More knots capture more complex expression patterns but risk overfitting.">Spline knots</span>
                <input
                  type="number"
                  min="2"
                  max="20"
                  value={nSplineKnots}
                  onChange={(e) => setNSplineKnots(Math.max(2, Math.min(20, parseInt(e.target.value) || 5)))}
                  style={styles.smallInput}
                  title="Interior knots for the cubic B-spline basis (2-20). Higher = more flexible fit."
                />
                <span title="FDR (Benjamini-Hochberg) threshold for significance.">FDR</span>
                <input
                  type="number"
                  min="0.001"
                  max="0.5"
                  step="0.01"
                  value={fdrThreshold}
                  onChange={(e) => setFdrThreshold(Math.max(0.001, Math.min(0.5, parseFloat(e.target.value) || 0.05)))}
                  style={{ ...styles.smallInput, width: '56px' }}
                  title="FDR significance threshold (0.001-0.5)"
                />
              </div>
              <div style={{ display: 'flex', alignItems: 'center', gap: '8px', fontSize: '11px', color: '#ccc' }}>
                <span title={clusterGenes
                  ? 'Maximum number of genes to return per module.'
                  : 'Maximum number of genes to return per direction (positive/negative).'}>
                  {clusterGenes ? 'Max genes/module' : 'Max genes/direction'}
                </span>
                <input
                  type="number"
                  min="10"
                  max="500"
                  step="10"
                  value={topN}
                  onChange={(e) => setTopN(Math.max(10, Math.min(500, parseInt(e.target.value) || 50)))}
                  style={styles.smallInput}
                  title="Maximum genes returned (10-500)"
                />
              </div>
              <label
                style={{ display: 'flex', alignItems: 'center', gap: '6px', marginTop: '8px', fontSize: '11px', color: '#ccc', cursor: 'pointer' }}
                title="Cluster significant genes by expression profile shape into modules (increasing, decreasing, peak, trough, complex). When off, only positive/negative lists are returned."
              >
                <input
                  type="checkbox"
                  checked={clusterGenes}
                  onChange={(e) => setClusterGenes(e.target.checked)}
                />
                Cluster genes into modules
              </label>
            </div>

            <button
              style={{
                ...styles.primaryActionButton,
                opacity: isLineAssociationLoading && !activeTaskId ? 0.6 : 1,
                ...(isLineAssociationLoading && activeTaskId ? { backgroundColor: '#e94560', color: '#fff' } : {}),
              }}
              onClick={isLineAssociationLoading && activeTaskId ? handleCancelAssociation : handleFindAssociatedGenes}
              disabled={isLineAssociationLoading && !activeTaskId}
              title={isLineAssociationLoading
                ? 'Click to cancel'
                : testVariable === 'position'
                  ? 'Find genes whose expression is associated with position along this line'
                  : 'Find genes whose expression is associated with distance from this line'}
            >
              {isLineAssociationLoading
                ? (activeTaskId ? 'Cancel' : 'Analyzing...')
                : 'Find Associated Genes'}
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

// ---------- Multi-Line Tools Modal ----------

function MultiLineToolsModal({
  lines,
  onClose,
}: {
  lines: ReturnType<typeof useStore.getState>['drawnLines']
  onClose: () => void
}) {
  const { scanpyActionHistory } = useStore()
  const { runMultiAssociation, isLineAssociationLoading } = useLineAssociation()
  const activeTaskId = useStore((state) => state.activeTaskId)

  const [reversals, setReversals] = useState<Record<string, boolean>>({})
  const [associationError, setAssociationError] = useState<string | null>(null)
  const [geneSubsetColumns, setGeneSubsetColumns] = useState<{ name: string; n_true: number; n_total: number }[]>([])
  const [selectedGeneColumns, setSelectedGeneColumns] = useState<string[]>([])
  const [geneSubsetOperation, setGeneSubsetOperation] = useState<'intersection' | 'union'>('intersection')
  const [testVariable, setTestVariable] = useState<'position' | 'distance'>(() => laDefault('test_variable', 'position'))
  const [nSplineKnots, setNSplineKnots] = useState<number>(() => laDefault('n_spline_knots', 5))
  const [fdrThreshold, setFdrThreshold] = useState<number>(() => laDefault('fdr_threshold', 0.05))
  const [topN, setTopN] = useState<number>(() => laDefault('top_n', 50))
  const [clusterGenes, setClusterGenes] = useState<boolean>(() => laDefault('cluster_genes', false))

  const API_BASE = '/api'
  const totalCells = lines.reduce((sum, l) => sum + l.projections.length, 0)

  useEffect(() => {
    fetch(appendDataset(`${API_BASE}/var/boolean_columns`))
      .then((res) => res.json())
      .then(setGeneSubsetColumns)
      .catch(() => setGeneSubsetColumns([]))
  }, [scanpyActionHistory])

  const toggleReversal = useCallback((lineId: string) => {
    setReversals((prev) => ({ ...prev, [lineId]: !prev[lineId] }))
  }, [])

  const handleRun = useCallback(async () => {
    setAssociationError(null)
    try {
      let geneSubset: string | { columns: string[]; operation: string } | null = null
      if (selectedGeneColumns.length === 1) {
        geneSubset = selectedGeneColumns[0]
      } else if (selectedGeneColumns.length > 1) {
        geneSubset = { columns: selectedGeneColumns, operation: geneSubsetOperation }
      }

      await runMultiAssociation({
        lines: lines.map((l) => ({
          name: l.name,
          cellIndices: l.projections.map((p) => p.cellIndex),
          reversed: !!reversals[l.id],
        })),
        geneSubset,
        testVariable,
        nSplineKnots,
        fdrThreshold,
        topN,
        clusterGenes,
      })
      onClose()
    } catch (err) {
      const message = (err as Error).message
      if (message !== 'cancelled') {
        setAssociationError(message)
      }
    }
  }, [runMultiAssociation, lines, reversals, selectedGeneColumns, geneSubsetOperation, testVariable, nSplineKnots, fdrThreshold, topN, clusterGenes, onClose])

  const handleCancelAssociation = useCallback(async () => {
    if (activeTaskId) {
      try { await cancelTask(activeTaskId) } catch { /* may have completed */ }
    }
  }, [activeTaskId])

  return (
    <div style={styles.overlay} onClick={onClose}>
      <div style={styles.modal} onClick={(e) => e.stopPropagation()}>
        <div style={styles.modalHeader}>
          <h2 style={styles.modalTitle}>
            Line Association: {lines.length} lines ({totalCells.toLocaleString()} cells)
          </h2>
          <button style={styles.modalClose} onClick={onClose}>&times;</button>
        </div>

        <div style={styles.modalContent}>
          {/* Line list with direction toggles */}
          <div style={styles.section}>
            <div style={styles.sectionTitle}>Lines</div>
            {lines.map((line) => (
              <div key={line.id} style={{
                display: 'flex',
                alignItems: 'center',
                gap: '8px',
                padding: '6px 0',
                fontSize: '12px',
                color: '#ccc',
                borderBottom: '1px solid #0a0f1a',
              }}>
                <span style={{ flex: 1, overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap' as const }}>
                  {line.name}
                </span>
                <span style={{ fontSize: '11px', color: '#888' }}>
                  {line.projections.length} cells
                </span>
                <button
                  onClick={() => toggleReversal(line.id)}
                  style={{
                    padding: '2px 8px',
                    fontSize: '11px',
                    borderRadius: '4px',
                    cursor: 'pointer',
                    border: '1px solid #1a1a2e',
                    backgroundColor: reversals[line.id] ? '#e94560' : '#0f3460',
                    color: reversals[line.id] ? '#fff' : '#aaa',
                    fontWeight: reversals[line.id] ? 600 : 400,
                  }}
                  title={reversals[line.id] ? 'Direction: reversed (click to reset)' : 'Direction: as drawn (click to reverse)'}
                >
                  {reversals[line.id] ? '\u2190' : '\u2192'}
                </button>
              </div>
            ))}
          </div>

          {/* Gene subset selector */}
          {geneSubsetColumns.length > 0 && (
            <div style={styles.section}>
              <div style={styles.sectionTitle}>
                Genes {selectedGeneColumns.length === 0 ? '(all)' : ''}
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
          <div style={styles.section}>
            <div style={styles.sectionTitle}>Test against</div>
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
              >
                Distance from line
              </button>
            </div>
          </div>

          {/* Parameters */}
          <div style={styles.section}>
            <div style={styles.sectionTitle}>Parameters</div>
            <div style={{ display: 'flex', alignItems: 'center', gap: '8px', marginBottom: '6px', fontSize: '11px', color: '#ccc' }}>
              <span>Spline knots</span>
              <input type="number" min="2" max="20" value={nSplineKnots}
                onChange={(e) => setNSplineKnots(Math.max(2, Math.min(20, parseInt(e.target.value) || 5)))}
                style={styles.smallInput} />
              <span>FDR</span>
              <input type="number" min="0.001" max="0.5" step="0.01" value={fdrThreshold}
                onChange={(e) => setFdrThreshold(Math.max(0.001, Math.min(0.5, parseFloat(e.target.value) || 0.05)))}
                style={{ ...styles.smallInput, width: '56px' }} />
            </div>
            <div style={{ display: 'flex', alignItems: 'center', gap: '8px', fontSize: '11px', color: '#ccc' }}>
              <span>{clusterGenes ? 'Max genes/module' : 'Max genes/direction'}</span>
              <input type="number" min="10" max="500" step="10" value={topN}
                onChange={(e) => setTopN(Math.max(10, Math.min(500, parseInt(e.target.value) || 50)))}
                style={styles.smallInput} />
            </div>
            <label
              style={{ display: 'flex', alignItems: 'center', gap: '6px', marginTop: '8px', fontSize: '11px', color: '#ccc', cursor: 'pointer' }}
              title="Cluster significant genes by expression profile shape into modules."
            >
              <input
                type="checkbox"
                checked={clusterGenes}
                onChange={(e) => setClusterGenes(e.target.checked)}
              />
              Cluster genes into modules
            </label>
          </div>

          {/* Run button */}
          <button
            style={{
              ...styles.primaryActionButton,
              opacity: isLineAssociationLoading && !activeTaskId ? 0.6 : 1,
              ...(isLineAssociationLoading && activeTaskId ? { backgroundColor: '#e94560', color: '#fff' } : {}),
            }}
            onClick={isLineAssociationLoading && activeTaskId ? handleCancelAssociation : handleRun}
            disabled={isLineAssociationLoading && !activeTaskId}
          >
            {isLineAssociationLoading
              ? (activeTaskId ? 'Cancel' : 'Analyzing...')
              : 'Find Associated Genes'}
          </button>
          {associationError && (
            <div style={{ marginTop: '6px', fontSize: '11px', color: '#e94560' }}>
              {associationError}
            </div>
          )}
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
  const [checkedLineIds, setCheckedLineIds] = useState<Set<string>>(new Set())
  const [multiLineModalOpen, setMultiLineModalOpen] = useState(false)

  const currentEmbeddingLines = drawnLines.filter(
    (l) => l.embeddingName === selectedEmbedding
  )
  const hasAnyLines = drawnLines.length > 0
  const toolsLine = toolsLineId ? drawnLines.find((l) => l.id === toolsLineId) : null

  const checkedLines = currentEmbeddingLines.filter((l) => checkedLineIds.has(l.id))
  const checkedTotalCells = checkedLines.reduce((sum, l) => sum + l.projections.length, 0)

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

  const handleToggleCheck = useCallback((lineId: string, e: React.MouseEvent) => {
    e.stopPropagation()
    setCheckedLineIds((prev) => {
      const next = new Set(prev)
      if (next.has(lineId)) {
        next.delete(lineId)
      } else {
        next.add(lineId)
      }
      return next
    })
  }, [])

  return (
    <>
      <div style={{
        ...styles.panel,
        ...(collapsed ? { minHeight: 0 } : { minHeight: '80px', maxHeight: '250px' }),
      }}>
        <div
          style={styles.header}
          onClick={() => setCollapsed(!collapsed)}
          title={collapsed ? 'Expand Shapes panel' : 'Collapse Shapes panel'}
        >
          <span style={styles.title}>Shapes</span>
          <span style={styles.collapseIcon}>{collapsed ? '\u25B6' : '\u25BC'}</span>
        </div>

        {!collapsed && (
          <div style={styles.content}>
            {currentEmbeddingLines.length === 0 && (
              <div style={styles.emptyState}>
                {hasAnyLines
                  ? `No shapes on "${selectedEmbedding}". Use the Draw tool to create one.`
                  : 'No shapes yet. Use the Draw tool to create a line or shape.'}
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
                {line.projections.length > 0 && (
                  <input
                    type="checkbox"
                    checked={checkedLineIds.has(line.id)}
                    onChange={() => {}}
                    onClick={(e) => handleToggleCheck(line.id, e)}
                    style={{ marginRight: '4px', cursor: 'pointer', flexShrink: 0 }}
                    title={`Include in multi-line analysis (${line.projections.length} cells)`}
                  />
                )}
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
            {checkedLineIds.size > 0 && checkedLines.length > 0 && (
              <div style={{
                padding: '8px 16px',
                borderTop: '1px solid #0f3460',
                display: 'flex',
                alignItems: 'center',
                justifyContent: 'space-between',
                gap: '8px',
                fontSize: '11px',
              }}>
                <span style={{ color: '#888' }}>
                  {checkedLines.length} line{checkedLines.length !== 1 ? 's' : ''} ({checkedTotalCells.toLocaleString()} cells)
                </span>
                <button
                  style={{
                    padding: '4px 10px',
                    fontSize: '11px',
                    backgroundColor: '#4ecdc4',
                    color: '#000',
                    border: '1px solid #4ecdc4',
                    borderRadius: '4px',
                    cursor: 'pointer',
                    fontWeight: 500,
                  }}
                  onClick={() => setMultiLineModalOpen(true)}
                >
                  Find Associated Genes
                </button>
              </div>
            )}
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
      {multiLineModalOpen && checkedLines.length > 0 && (
        <MultiLineToolsModal
          lines={checkedLines}
          onClose={() => setMultiLineModalOpen(false)}
        />
      )}
    </>
  )
}
