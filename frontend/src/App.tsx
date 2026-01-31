import { useStore } from './store'
import { useSchema, useEmbedding, useColorBy, useDataActions } from './hooks/useData'
import ScatterPlot from './components/ScatterPlot'
import GenePanel from './components/GenePanel'

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
    gap: '20px',
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
    background: 'linear-gradient(to right, rgb(68,1,84), rgb(59,82,139), rgb(33,145,140), rgb(94,201,98), rgb(253,231,37))',
    marginBottom: '4px',
  },
  colorBarLabels: {
    display: 'flex',
    justifyContent: 'space-between',
    fontSize: '10px',
    color: '#888',
  },
}

// Color palette (must match ScatterPlot)
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

function ExpressionLegend({ gene, min, max }: { gene: string; min: number; max: number }) {
  return (
    <div style={styles.expressionLegend}>
      <div style={styles.legendTitle}>{gene}</div>
      <div style={styles.colorBar} />
      <div style={styles.colorBarLabels}>
        <span>{min.toFixed(2)}</span>
        <span>{max.toFixed(2)}</span>
      </div>
    </div>
  )
}

export default function App() {
  const { isLoading, error, selectedEmbedding, selectedColorColumn, colorMode, expressionData, selectedGenes } = useStore()
  const schema = useSchema()
  const embedding = useEmbedding()
  const colorBy = useColorBy()
  const { selectEmbedding, selectColorColumn } = useDataActions()

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

              <div style={styles.controlGroup}>
                <span style={styles.label}>Color by metadata:</span>
                <select
                  style={styles.select}
                  value={colorMode === 'metadata' ? selectedColorColumn || '' : ''}
                  onChange={(e) => selectColorColumn(e.target.value || null)}
                >
                  <option value="">None</option>
                  {schema.obs_columns.map((col) => (
                    <option key={col} value={col}>
                      {col} ({schema.obs_dtypes[col]})
                    </option>
                  ))}
                </select>
              </div>

              <span style={styles.stats}>
                {schema.n_cells.toLocaleString()} cells &middot; {schema.n_genes.toLocaleString()} genes
              </span>
            </>
          )}
        </div>
      </header>

      <div style={styles.body}>
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
              />
              {colorMode === 'metadata' && colorBy && <CategoryLegend colorBy={colorBy} />}
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
    </div>
  )
}
