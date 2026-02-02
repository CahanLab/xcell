import { useCallback, useEffect, useState } from 'react'
import { useStore } from './store'
import { useSchema, useEmbedding, useColorBy, useDataActions, exportAnnotations } from './hooks/useData'
import ScatterPlot from './components/ScatterPlot'
import GenePanel from './components/GenePanel'
import CellPanel from './components/CellPanel'
import DisplaySettings from './components/DisplaySettings'
import DiffExpModal from './components/DiffExpModal'
import ShapeManager from './components/ShapeManager'

const styles = {
  container: {
    display: 'flex',
    flexDirection: 'column' as const,
    height: '100vh',
    backgroundColor: '#1a1a2e',
  },
  header: {
    display: 'flex',
    alignItems: 'center',
    justifyContent: 'space-between',
    padding: '12px 20px',
    backgroundColor: '#16213e',
    borderBottom: '1px solid #0f3460',
  },
  titleGroup: {
    display: 'flex',
    alignItems: 'baseline',
    gap: '12px',
  },
  title: {
    fontSize: '20px',
    fontWeight: 600,
    color: '#e94560',
  },
  statsInline: {
    fontSize: '11px',
    color: '#666',
  },
  controls: {
    display: 'flex',
    gap: '12px',
    alignItems: 'center',
  },
  exportButton: {
    padding: '6px 10px',
    fontSize: '12px',
    backgroundColor: '#0f3460',
    color: '#aaa',
    border: '1px solid #1a1a2e',
    borderRadius: '4px',
    cursor: 'pointer',
    display: 'flex',
    alignItems: 'center',
    gap: '4px',
  },
  controlGroup: {
    display: 'flex',
    alignItems: 'center',
    gap: '8px',
  },
  label: {
    fontSize: '14px',
    color: '#aaa',
  },
  select: {
    padding: '6px 12px',
    fontSize: '14px',
    backgroundColor: '#0f3460',
    color: '#eee',
    border: '1px solid #1a1a2e',
    borderRadius: '4px',
    cursor: 'pointer',
  },
  stats: {
    fontSize: '13px',
    color: '#888',
  },
  coloringInfo: {
    fontSize: '12px',
    color: '#e94560',
    backgroundColor: '#0f3460',
    padding: '4px 10px',
    borderRadius: '4px',
  },
  toolButton: {
    padding: '6px 12px',
    fontSize: '13px',
    backgroundColor: '#0f3460',
    color: '#aaa',
    border: '1px solid #1a1a2e',
    borderRadius: '4px',
    cursor: 'pointer',
    display: 'flex',
    alignItems: 'center',
    gap: '6px',
  },
  toolButtonActive: {
    backgroundColor: '#e94560',
    color: '#fff',
    borderColor: '#e94560',
  },
  selectionInfo: {
    fontSize: '12px',
    color: '#ffd700',
    backgroundColor: '#0f3460',
    padding: '4px 10px',
    borderRadius: '4px',
    display: 'flex',
    alignItems: 'center',
    gap: '8px',
  },
  clearButton: {
    padding: '2px 6px',
    fontSize: '11px',
    backgroundColor: 'transparent',
    color: '#ffd700',
    border: '1px solid #ffd700',
    borderRadius: '3px',
    cursor: 'pointer',
  },
  body: {
    flex: 1,
    display: 'flex',
    overflow: 'hidden',
  },
  main: {
    flex: 1,
    position: 'relative' as const,
    overflow: 'hidden',
  },
  loading: {
    position: 'absolute' as const,
    top: '50%',
    left: '50%',
    transform: 'translate(-50%, -50%)',
    fontSize: '18px',
    color: '#aaa',
  },
  error: {
    position: 'absolute' as const,
    top: '50%',
    left: '50%',
    transform: 'translate(-50%, -50%)',
    padding: '20px',
    backgroundColor: '#4a1c1c',
    border: '1px solid #e94560',
    borderRadius: '8px',
    color: '#e94560',
  },
  legend: {
    position: 'absolute' as const,
    bottom: '20px',
    right: '20px',
    padding: '12px',
    backgroundColor: 'rgba(22, 33, 62, 0.9)',
    borderRadius: '8px',
    maxHeight: '200px',
    overflowY: 'auto' as const,
  },
  legendTitle: {
    fontSize: '12px',
    fontWeight: 600,
    marginBottom: '8px',
    color: '#aaa',
  },
  legendItem: {
    display: 'flex',
    alignItems: 'center',
    gap: '8px',
    fontSize: '12px',
    marginBottom: '4px',
  },
  legendColor: {
    width: '12px',
    height: '12px',
    borderRadius: '2px',
  },
  expressionLegend: {
    position: 'absolute' as const,
    bottom: '20px',
    right: '20px',
    padding: '12px',
    backgroundColor: 'rgba(22, 33, 62, 0.9)',
    borderRadius: '8px',
  },
  embeddingSelector: {
    position: 'absolute' as const,
    bottom: '20px',
    left: '20px',
    padding: '8px 12px',
    backgroundColor: 'rgba(22, 33, 62, 0.9)',
    borderRadius: '8px',
    display: 'flex',
    alignItems: 'center',
    gap: '8px',
  },
  embeddingLabel: {
    fontSize: '11px',
    color: '#888',
  },
  embeddingSelect: {
    padding: '4px 8px',
    fontSize: '12px',
    backgroundColor: '#0f3460',
    color: '#eee',
    border: '1px solid #1a1a2e',
    borderRadius: '4px',
    cursor: 'pointer',
  },
  colorBar: {
    width: '120px',
    height: '12px',
    borderRadius: '2px',
    marginBottom: '4px',
  },
  colorBarLabels: {
    display: 'flex',
    justifyContent: 'space-between',
    fontSize: '10px',
    color: '#888',
  },
  stackButton: {
    marginTop: '8px',
    padding: '4px 8px',
    fontSize: '10px',
    backgroundColor: '#0f3460',
    color: '#aaa',
    border: '1px solid #1a1a2e',
    borderRadius: '3px',
    cursor: 'pointer',
    width: '100%',
  },
  stackButtonActive: {
    backgroundColor: '#4ecdc4',
    color: '#000',
    borderColor: '#4ecdc4',
  },
}

const CATEGORY_COLORS = [
  'rgb(31, 119, 180)',
  'rgb(255, 127, 14)',
  'rgb(44, 160, 44)',
  'rgb(214, 39, 40)',
  'rgb(148, 103, 189)',
  'rgb(140, 86, 75)',
  'rgb(227, 119, 194)',
  'rgb(127, 127, 127)',
  'rgb(188, 189, 34)',
  'rgb(23, 190, 207)',
]

function CategoryLegend({ colorBy }: { colorBy: { name: string; categories?: string[]; dtype: string } }) {
  if (colorBy.dtype !== 'category' || !colorBy.categories) {
    return null
  }

  return (
    <div style={styles.legend}>
      <div style={styles.legendTitle}>{colorBy.name}</div>
      {colorBy.categories.map((cat, i) => (
        <div key={cat} style={styles.legendItem}>
          <div
            style={{
              ...styles.legendColor,
              backgroundColor: CATEGORY_COLORS[i % CATEGORY_COLORS.length],
            }}
          />
          <span>{cat}</span>
        </div>
      ))}
    </div>
  )
}

const COLOR_SCALE_GRADIENTS: Record<string, string> = {
  viridis: 'linear-gradient(to right, rgb(68,1,84), rgb(59,82,139), rgb(33,145,140), rgb(94,201,98), rgb(253,231,37))',
  plasma: 'linear-gradient(to right, rgb(13,8,135), rgb(126,3,168), rgb(204,71,120), rgb(248,149,64), rgb(240,249,33))',
  magma: 'linear-gradient(to right, rgb(0,0,4), rgb(81,18,124), rgb(183,55,121), rgb(252,137,97), rgb(252,253,191))',
  inferno: 'linear-gradient(to right, rgb(0,0,4), rgb(66,10,104), rgb(147,38,103), rgb(221,81,58), rgb(252,165,10), rgb(252,255,164))',
  cividis: 'linear-gradient(to right, rgb(0,32,81), rgb(82,95,110), rgb(152,136,62), rgb(253,234,69))',
  coolwarm: 'linear-gradient(to right, rgb(59,76,192), rgb(112,146,208), rgb(197,197,197), rgb(230,128,103), rgb(180,4,38))',
  blues: 'linear-gradient(to right, rgb(247,251,255), rgb(107,174,214), rgb(8,48,107))',
  reds: 'linear-gradient(to right, rgb(255,245,240), rgb(251,106,74), rgb(103,0,13))',
}

function ContinuousLegend({ name, min, max }: { name: string; min: number; max: number }) {
  return (
    <div style={styles.expressionLegend}>
      <div style={styles.legendTitle}>{name}</div>
      <div
        style={{
          ...styles.colorBar,
          background: 'linear-gradient(to right, rgb(0,50,255), rgb(255,50,0))',
        }}
      />
      <div style={styles.colorBarLabels}>
        <span>{min.toFixed(2)}</span>
        <span>{max.toFixed(2)}</span>
      </div>
    </div>
  )
}

export default function App() {
  const {
    isLoading,
    error,
    selectedEmbedding,
    colorMode,
    expressionData,
    selectedGenes,
    interactionMode,
    selectedCellIndices,
    setInteractionMode,
    setSelectedCellIndices,
    clearSelection,
    cellSortOrder,
    sortCellsByExpression,
    resetCellOrder,
    displayPreferences,
    drawnLines,
    addLine,
  } = useStore()

  const schema = useSchema()
  const embedding = useEmbedding()
  const colorBy = useColorBy()
  const { selectEmbedding } = useDataActions()

  // State for line naming dialog
  const [pendingLinePoints, setPendingLinePoints] = useState<[number, number][] | null>(null)
  const [newLineName, setNewLineName] = useState('')

  // Handle escape key to exit lasso/draw mode
  useEffect(() => {
    const handleKeyDown = (e: KeyboardEvent) => {
      if (e.key === 'Escape') {
        setInteractionMode('pan')
        setPendingLinePoints(null)
      }
    }
    window.addEventListener('keydown', handleKeyDown)
    return () => window.removeEventListener('keydown', handleKeyDown)
  }, [setInteractionMode])

  const handleSelectionComplete = useCallback(
    (indices: number[]) => {
      setSelectedCellIndices(indices)
    },
    [setSelectedCellIndices]
  )

  const toggleLassoMode = useCallback(() => {
    setInteractionMode(interactionMode === 'lasso' ? 'pan' : 'lasso')
  }, [interactionMode, setInteractionMode])

  const toggleDrawMode = useCallback(() => {
    setInteractionMode(interactionMode === 'draw' ? 'pan' : 'draw')
  }, [interactionMode, setInteractionMode])

  const handleLineDrawn = useCallback((points: [number, number][]) => {
    setPendingLinePoints(points)
    setNewLineName(`Line ${drawnLines.length + 1}`)
    setInteractionMode('pan')
  }, [drawnLines.length, setInteractionMode])

  const handleSaveLine = useCallback(() => {
    if (pendingLinePoints && newLineName.trim() && selectedEmbedding) {
      addLine(newLineName.trim(), pendingLinePoints, selectedEmbedding)
      setPendingLinePoints(null)
      setNewLineName('')
    }
  }, [pendingLinePoints, newLineName, selectedEmbedding, addLine])

  const handleCancelLine = useCallback(() => {
    setPendingLinePoints(null)
    setNewLineName('')
  }, [])

  const handleExport = useCallback(async () => {
    try {
      const tsv = await exportAnnotations()
      const blob = new Blob([tsv], { type: 'text/tab-separated-values' })
      const url = URL.createObjectURL(blob)
      const a = document.createElement('a')
      a.href = url
      a.download = 'cell_annotations.tsv'
      document.body.appendChild(a)
      a.click()
      document.body.removeChild(a)
      URL.revokeObjectURL(url)
    } catch (err) {
      alert(`Failed to export: ${(err as Error).message}`)
    }
  }, [])

  return (
    <div style={styles.container}>
      <header style={styles.header}>
        <div style={styles.titleGroup}>
          <h1 style={styles.title}>XCell</h1>
          {schema && (
            <span style={styles.statsInline}>
              {schema.n_cells.toLocaleString()} cells · {schema.n_genes.toLocaleString()} genes
            </span>
          )}
        </div>

        <div style={styles.controls}>
          {schema && (
            <>
              <button
                style={{
                  ...styles.toolButton,
                  ...(interactionMode === 'lasso' ? styles.toolButtonActive : {}),
                }}
                onClick={toggleLassoMode}
                title="Toggle lasso selection (Escape to exit)"
              >
                <span>&#10022;</span> Lasso
              </button>

              <button
                style={{
                  ...styles.toolButton,
                  ...(interactionMode === 'draw' ? { ...styles.toolButtonActive, backgroundColor: '#4ecdc4' } : {}),
                }}
                onClick={toggleDrawMode}
                title="Draw a line for trajectory analysis (Escape to exit)"
              >
                <span>&#9998;</span> Draw
              </button>

              <DisplaySettings />

              <button
                style={styles.exportButton}
                onClick={handleExport}
                title="Export annotations as TSV"
              >
                Export
              </button>

              {selectedCellIndices.length > 0 && (
                <div style={styles.selectionInfo}>
                  {selectedCellIndices.length.toLocaleString()} cells selected
                  <button style={styles.clearButton} onClick={clearSelection}>
                    Clear
                  </button>
                </div>
              )}
            </>
          )}
        </div>
      </header>

      <div style={styles.body}>
        <div style={{
          display: 'flex',
          flexDirection: 'column',
          height: '100%',
          overflow: 'hidden',
          flexShrink: 0,
        }}>
          <CellPanel />
          <ShapeManager />
        </div>

        <main style={styles.main}>
          {isLoading && <div style={styles.loading}>Loading...</div>}

          {error && (
            <div style={styles.error}>
              <strong>Error:</strong> {error}
            </div>
          )}

          {embedding && !error && (
            <>
              <ScatterPlot
                embedding={embedding}
                colorBy={colorBy}
                expressionData={expressionData}
                colorMode={colorMode}
                interactionMode={interactionMode}
                selectedCellIndices={selectedCellIndices}
                onSelectionComplete={handleSelectionComplete}
                onLineDrawn={handleLineDrawn}
              />

              {/* Embedding selector - bottom left */}
              {schema && schema.embeddings.length > 1 && (
                <div style={styles.embeddingSelector}>
                  <span style={styles.embeddingLabel}>Embedding:</span>
                  <select
                    style={styles.embeddingSelect}
                    value={selectedEmbedding || ''}
                    onChange={(e) => selectEmbedding(e.target.value)}
                  >
                    {schema.embeddings.map((emb) => (
                      <option key={emb} value={emb}>
                        {emb}
                      </option>
                    ))}
                  </select>
                </div>
              )}
              {/* Legends - bottom right */}
              {colorMode === 'metadata' && colorBy && colorBy.dtype === 'category' && (
                <CategoryLegend colorBy={colorBy} />
              )}
              {colorMode === 'metadata' && colorBy && colorBy.dtype === 'numeric' && (
                <ContinuousLegend
                  name={colorBy.name}
                  min={Math.min(...(colorBy.values.filter((v) => v !== null) as number[]))}
                  max={Math.max(...(colorBy.values.filter((v) => v !== null) as number[]))}
                />
              )}
              {colorMode === 'expression' && expressionData && (
                <div style={styles.expressionLegend}>
                  <div style={styles.legendTitle}>
                    {selectedGenes.length === 1 ? selectedGenes[0] : `${selectedGenes.length} genes (mean)`}
                  </div>
                  <div style={{ ...styles.colorBar, background: COLOR_SCALE_GRADIENTS[displayPreferences.colorScale] || COLOR_SCALE_GRADIENTS.viridis }} />
                  <div style={styles.colorBarLabels}>
                    <span>{expressionData.min.toFixed(2)}</span>
                    <span>{expressionData.max.toFixed(2)}</span>
                  </div>
                  <button
                    style={{
                      ...styles.stackButton,
                      ...(cellSortOrder ? styles.stackButtonActive : {}),
                    }}
                    onClick={cellSortOrder ? resetCellOrder : sortCellsByExpression}
                    title={cellSortOrder ? 'Reset to default cell order' : 'Sort cells so high-expression cells render on top'}
                  >
                    {cellSortOrder ? 'Reset Order' : 'Stack by Expression'}
                  </button>
                </div>
              )}
            </>
          )}

          {!embedding && !isLoading && !error && (
            <div style={styles.loading}>
              No data loaded. Start the backend with an h5ad file.
            </div>
          )}
        </main>

        <GenePanel />
      </div>

      <DiffExpModal />

      {/* Line naming dialog */}
      {pendingLinePoints && (
        <div
          style={{
            position: 'fixed',
            top: 0,
            left: 0,
            right: 0,
            bottom: 0,
            backgroundColor: 'rgba(0, 0, 0, 0.6)',
            zIndex: 1000,
            display: 'flex',
            alignItems: 'center',
            justifyContent: 'center',
          }}
          onClick={handleCancelLine}
        >
          <div
            style={{
              backgroundColor: '#16213e',
              border: '1px solid #0f3460',
              borderRadius: '8px',
              padding: '20px',
              minWidth: '300px',
            }}
            onClick={(e) => e.stopPropagation()}
          >
            <div style={{ fontSize: '14px', fontWeight: 600, color: '#4ecdc4', marginBottom: '16px' }}>
              Save Line
            </div>
            <input
              type="text"
              value={newLineName}
              onChange={(e) => setNewLineName(e.target.value)}
              onKeyDown={(e) => e.key === 'Enter' && handleSaveLine()}
              placeholder="Line name..."
              autoFocus
              style={{
                width: '100%',
                padding: '8px 12px',
                fontSize: '14px',
                backgroundColor: '#0f3460',
                color: '#eee',
                border: '1px solid #1a1a2e',
                borderRadius: '4px',
                marginBottom: '16px',
                outline: 'none',
              }}
            />
            <div style={{ display: 'flex', gap: '8px', justifyContent: 'flex-end' }}>
              <button
                onClick={handleCancelLine}
                style={{
                  padding: '8px 16px',
                  fontSize: '13px',
                  backgroundColor: '#0f3460',
                  color: '#aaa',
                  border: '1px solid #1a1a2e',
                  borderRadius: '4px',
                  cursor: 'pointer',
                }}
              >
                Cancel
              </button>
              <button
                onClick={handleSaveLine}
                disabled={!newLineName.trim()}
                style={{
                  padding: '8px 16px',
                  fontSize: '13px',
                  backgroundColor: '#4ecdc4',
                  color: '#000',
                  border: 'none',
                  borderRadius: '4px',
                  cursor: 'pointer',
                }}
              >
                Save
              </button>
            </div>
          </div>
        </div>
      )}
    </div>
  )
}
