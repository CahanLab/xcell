import { useCallback, useEffect } from 'react'
import { useStore } from './store'
import { useSchema, useEmbedding, useColorBy, useDataActions } from './hooks/useData'
import ScatterPlot from './components/ScatterPlot'
import GenePanel from './components/GenePanel'
import CellPanel from './components/CellPanel'
import DisplaySettings from './components/DisplaySettings'
import DiffExpModal from './components/DiffExpModal'

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
  title: {
    fontSize: '20px',
    fontWeight: 600,
    color: '#e94560',
  },
  controls: {
    display: 'flex',
    gap: '16px',
    alignItems: 'center',
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
    left: '20px',
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
    left: '20px',
    padding: '12px',
    backgroundColor: 'rgba(22, 33, 62, 0.9)',
    borderRadius: '8px',
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

function ExpressionLegend({ gene, min, max }: { gene: string; min: number; max: number }) {
  const colorScale = useStore((state) => state.displayPreferences.colorScale)
  const gradient = COLOR_SCALE_GRADIENTS[colorScale] || COLOR_SCALE_GRADIENTS.viridis

  return (
    <div style={styles.expressionLegend}>
      <div style={styles.legendTitle}>{gene}</div>
      <div style={{ ...styles.colorBar, background: gradient }} />
      <div style={styles.colorBarLabels}>
        <span>{min.toFixed(2)}</span>
        <span>{max.toFixed(2)}</span>
      </div>
    </div>
  )
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
    selectedColorColumn,
    colorMode,
    expressionData,
    selectedGenes,
    interactionMode,
    selectedCellIndices,
    setInteractionMode,
    setSelectedCellIndices,
    clearSelection,
  } = useStore()

  const schema = useSchema()
  const embedding = useEmbedding()
  const colorBy = useColorBy()
  const { selectEmbedding } = useDataActions()

  // Handle escape key to exit lasso mode
  useEffect(() => {
    const handleKeyDown = (e: KeyboardEvent) => {
      if (e.key === 'Escape') {
        setInteractionMode('pan')
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

  const getColoringLabel = () => {
    if (colorMode === 'expression' && selectedGenes.length > 0) {
      return selectedGenes.length === 1 ? selectedGenes[0] : `${selectedGenes.length} genes`
    }
    if (colorMode === 'metadata' && selectedColorColumn) {
      return selectedColorColumn
    }
    return null
  }

  const coloringLabel = getColoringLabel()

  return (
    <div style={styles.container}>
      <header style={styles.header}>
        <h1 style={styles.title}>XCell</h1>

        <div style={styles.controls}>
          {schema && (
            <>
              <div style={styles.controlGroup}>
                <span style={styles.label}>Embedding:</span>
                <select
                  style={styles.select}
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

              <DisplaySettings />

              {selectedCellIndices.length > 0 && (
                <div style={styles.selectionInfo}>
                  {selectedCellIndices.length.toLocaleString()} cells selected
                  <button style={styles.clearButton} onClick={clearSelection}>
                    Clear
                  </button>
                </div>
              )}

              {coloringLabel && (
                <div style={styles.coloringInfo}>
                  Coloring: {coloringLabel}
                </div>
              )}

              <span style={styles.stats}>
                {schema.n_cells.toLocaleString()} cells &middot; {schema.n_genes.toLocaleString()} genes
              </span>
            </>
          )}
        </div>
      </header>

      <div style={styles.body}>
        <CellPanel />

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
              />
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
                <ExpressionLegend
                  gene={selectedGenes.length === 1 ? selectedGenes[0] : `${selectedGenes.length} genes (mean)`}
                  min={expressionData.min}
                  max={expressionData.max}
                />
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
    </div>
  )
}
