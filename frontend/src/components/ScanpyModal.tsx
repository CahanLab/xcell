import { useState, useCallback, useEffect } from 'react'
import { useStore, ScanpyActionRecord } from '../store'

const API_BASE = '/api'

// Type definitions for scanpy functions
interface ParamDef {
  name: string
  label: string
  type: 'number' | 'text' | 'select'
  default: string | number | null
  description: string
  options?: string[]
}

interface FunctionDef {
  label: string
  description: string
  prerequisites: string[]
  params: ParamDef[]
}

interface CategoryDef {
  label: string
  functions: Record<string, FunctionDef>
}

// Scanpy function definitions organized by category
const SCANPY_FUNCTIONS: Record<string, CategoryDef> = {
  preprocessing: {
    label: 'Preprocessing',
    functions: {
      filter_genes: {
        label: 'Filter Genes',
        description: 'Remove genes based on counts or number of cells expressing',
        prerequisites: [],
        params: [
          { name: 'min_counts', label: 'Min counts', type: 'number', default: null, description: 'Minimum total counts' },
          { name: 'max_counts', label: 'Max counts', type: 'number', default: null, description: 'Maximum total counts' },
          { name: 'min_cells', label: 'Min cells', type: 'number', default: null, description: 'Minimum cells expressing' },
          { name: 'max_cells', label: 'Max cells', type: 'number', default: null, description: 'Maximum cells expressing' },
        ],
      },
      filter_cells: {
        label: 'Filter Cells',
        description: 'Remove cells based on counts or number of genes expressed',
        prerequisites: [],
        params: [
          { name: 'min_counts', label: 'Min counts', type: 'number', default: null, description: 'Minimum total counts' },
          { name: 'max_counts', label: 'Max counts', type: 'number', default: null, description: 'Maximum total counts' },
          { name: 'min_genes', label: 'Min genes', type: 'number', default: null, description: 'Minimum genes expressed' },
          { name: 'max_genes', label: 'Max genes', type: 'number', default: null, description: 'Maximum genes expressed' },
        ],
      },
      normalize_total: {
        label: 'Normalize Total',
        description: 'Normalize counts per cell to a target sum',
        prerequisites: [],
        params: [
          { name: 'target_sum', label: 'Target sum', type: 'number', default: 10000, description: 'Target sum per cell (null = median)' },
        ],
      },
      log1p: {
        label: 'Log1p Transform',
        description: 'Apply log(x + 1) transformation',
        prerequisites: [],
        params: [],
      },
    },
  },
  dimensionality_reduction: {
    label: 'Dimensionality Reduction',
    functions: {
      pca: {
        label: 'PCA',
        description: 'Principal component analysis',
        prerequisites: [],
        params: [
          { name: 'n_comps', label: 'Components', type: 'number', default: 50, description: 'Number of principal components' },
          { name: 'svd_solver', label: 'SVD solver', type: 'select', default: 'arpack', options: ['arpack', 'randomized', 'auto'], description: 'SVD algorithm' },
        ],
      },
    },
  },
  graph_building: {
    label: 'Graph Building',
    functions: {
      neighbors: {
        label: 'Neighbors',
        description: 'Compute neighborhood graph',
        prerequisites: ['pca'],
        params: [
          { name: 'n_neighbors', label: 'Neighbors', type: 'number', default: 15, description: 'Number of neighbors' },
          { name: 'n_pcs', label: 'PCs to use', type: 'number', default: null, description: 'Number of PCs (null = all)' },
          { name: 'metric', label: 'Metric', type: 'select', default: 'euclidean', options: ['euclidean', 'cosine', 'manhattan'], description: 'Distance metric' },
        ],
      },
    },
  },
  embedding: {
    label: 'Embedding',
    functions: {
      umap: {
        label: 'UMAP',
        description: 'Compute UMAP embedding',
        prerequisites: ['neighbors'],
        params: [
          { name: 'min_dist', label: 'Min distance', type: 'number', default: 0.5, description: 'Minimum distance between points' },
          { name: 'spread', label: 'Spread', type: 'number', default: 1.0, description: 'Spread of embedding' },
          { name: 'n_components', label: 'Dimensions', type: 'number', default: 2, description: 'Number of dimensions' },
        ],
      },
    },
  },
  clustering: {
    label: 'Clustering',
    functions: {
      leiden: {
        label: 'Leiden',
        description: 'Leiden clustering algorithm',
        prerequisites: ['neighbors'],
        params: [
          { name: 'resolution', label: 'Resolution', type: 'number', default: 1.0, description: 'Higher = more clusters' },
          { name: 'key_added', label: 'Column name', type: 'text', default: 'leiden', description: 'Name for cluster labels' },
        ],
      },
    },
  },
}

type CategoryKey = string
type FunctionKey = string

interface ParamValues {
  [key: string]: string | number | null
}

interface PrerequisiteStatus {
  satisfied: boolean
  missing: string[]
}

const styles = {
  overlay: {
    position: 'fixed' as const,
    top: 0,
    left: 0,
    right: 0,
    bottom: 0,
    backgroundColor: 'rgba(0, 0, 0, 0.6)',
    zIndex: 1000,
    display: 'flex',
    alignItems: 'center',
    justifyContent: 'center',
  },
  modal: {
    backgroundColor: '#16213e',
    border: '1px solid #0f3460',
    borderRadius: '8px',
    padding: '24px',
    width: '500px',
    maxWidth: '90vw',
    maxHeight: '80vh',
    overflow: 'auto',
  },
  header: {
    display: 'flex',
    justifyContent: 'space-between',
    alignItems: 'center',
    marginBottom: '20px',
  },
  title: {
    fontSize: '18px',
    fontWeight: 600,
    color: '#e94560',
  },
  closeButton: {
    background: 'none',
    border: 'none',
    color: '#aaa',
    fontSize: '20px',
    cursor: 'pointer',
    padding: '4px 8px',
  },
  categoryTabs: {
    display: 'flex',
    flexWrap: 'wrap' as const,
    gap: '8px',
    marginBottom: '16px',
  },
  tab: {
    padding: '6px 12px',
    fontSize: '12px',
    backgroundColor: '#0f3460',
    color: '#aaa',
    border: '1px solid #1a1a2e',
    borderRadius: '4px',
    cursor: 'pointer',
  },
  tabActive: {
    backgroundColor: '#e94560',
    color: '#fff',
    borderColor: '#e94560',
  },
  functionSelect: {
    width: '100%',
    padding: '8px 12px',
    fontSize: '14px',
    backgroundColor: '#0f3460',
    color: '#eee',
    border: '1px solid #1a1a2e',
    borderRadius: '4px',
    marginBottom: '16px',
  },
  description: {
    fontSize: '13px',
    color: '#888',
    marginBottom: '16px',
    padding: '8px',
    backgroundColor: '#0f3460',
    borderRadius: '4px',
  },
  prerequisiteWarning: {
    fontSize: '12px',
    color: '#ffd700',
    backgroundColor: 'rgba(255, 215, 0, 0.1)',
    padding: '8px',
    borderRadius: '4px',
    marginBottom: '16px',
  },
  paramsSection: {
    marginBottom: '16px',
  },
  paramRow: {
    display: 'flex',
    alignItems: 'center',
    gap: '12px',
    marginBottom: '12px',
  },
  paramLabel: {
    width: '120px',
    fontSize: '13px',
    color: '#aaa',
  },
  paramInput: {
    flex: 1,
    padding: '6px 10px',
    fontSize: '13px',
    backgroundColor: '#0f3460',
    color: '#eee',
    border: '1px solid #1a1a2e',
    borderRadius: '4px',
  },
  paramDescription: {
    fontSize: '11px',
    color: '#666',
    marginTop: '2px',
  },
  buttonRow: {
    display: 'flex',
    justifyContent: 'space-between',
    gap: '12px',
    marginTop: '20px',
  },
  runButton: {
    flex: 1,
    padding: '10px 16px',
    fontSize: '14px',
    backgroundColor: '#4ecdc4',
    color: '#000',
    border: 'none',
    borderRadius: '4px',
    cursor: 'pointer',
    fontWeight: 500,
  },
  runButtonDisabled: {
    backgroundColor: '#333',
    color: '#666',
    cursor: 'not-allowed',
  },
  cancelButton: {
    padding: '10px 16px',
    fontSize: '14px',
    backgroundColor: 'transparent',
    color: '#aaa',
    border: '1px solid #0f3460',
    borderRadius: '4px',
    cursor: 'pointer',
  },
  historySection: {
    marginTop: '20px',
    borderTop: '1px solid #0f3460',
    paddingTop: '16px',
  },
  historyTitle: {
    fontSize: '13px',
    fontWeight: 500,
    color: '#aaa',
    marginBottom: '8px',
  },
  historyItem: {
    fontSize: '12px',
    color: '#666',
    padding: '4px 8px',
    backgroundColor: '#0f3460',
    borderRadius: '3px',
    marginBottom: '4px',
  },
  result: {
    marginTop: '16px',
    padding: '12px',
    backgroundColor: '#0f3460',
    borderRadius: '4px',
  },
  resultSuccess: {
    borderLeft: '3px solid #4ecdc4',
  },
  resultError: {
    borderLeft: '3px solid #e94560',
  },
  resultText: {
    fontSize: '13px',
    color: '#eee',
  },
}

export default function ScanpyModal() {
  const { isScanpyModalOpen, setScanpyModalOpen, schema, setSchema, scanpyActionHistory, addScanpyAction } = useStore()

  const [selectedCategory, setSelectedCategory] = useState<CategoryKey>('preprocessing')
  const [selectedFunction, setSelectedFunction] = useState<FunctionKey>('filter_genes')
  const [paramValues, setParamValues] = useState<ParamValues>({})
  const [isRunning, setIsRunning] = useState(false)
  const [result, setResult] = useState<{ success: boolean; message: string } | null>(null)
  const [prereqStatus, setPrereqStatus] = useState<PrerequisiteStatus | null>(null)

  // Get current function definition
  const categoryDef = SCANPY_FUNCTIONS[selectedCategory]
  const functionDef: FunctionDef | undefined = categoryDef?.functions[selectedFunction]

  // Initialize param values when function changes
  useEffect(() => {
    if (functionDef) {
      const defaults: ParamValues = {}
      functionDef.params.forEach((param) => {
        defaults[param.name] = param.default
      })
      setParamValues(defaults)
      setResult(null)
    }
  }, [selectedFunction, functionDef])

  // Check prerequisites when function changes
  useEffect(() => {
    if (!functionDef || functionDef.prerequisites.length === 0) {
      setPrereqStatus({ satisfied: true, missing: [] })
      return
    }

    fetch(`${API_BASE}/scanpy/prerequisites/${selectedFunction}`)
      .then((res) => res.json())
      .then(setPrereqStatus)
      .catch(() => setPrereqStatus({ satisfied: false, missing: ['unknown'] }))
  }, [selectedFunction, functionDef, scanpyActionHistory])

  // Handle category change
  const handleCategoryChange = useCallback((category: CategoryKey) => {
    setSelectedCategory(category)
    const firstFunction = Object.keys(SCANPY_FUNCTIONS[category].functions)[0]
    setSelectedFunction(firstFunction)
    setResult(null)
  }, [])

  // Handle param change
  const handleParamChange = useCallback((paramName: string, value: string) => {
    setParamValues((prev) => ({
      ...prev,
      [paramName]: value === '' ? null : (isNaN(Number(value)) ? value : Number(value)),
    }))
  }, [])

  // Refresh schema after operations that change data shape
  const refreshSchema = useCallback(async () => {
    try {
      const response = await fetch(`${API_BASE}/schema`)
      if (response.ok) {
        const newSchema = await response.json()
        setSchema(newSchema)
      }
    } catch (err) {
      console.error('Failed to refresh schema:', err)
    }
  }, [setSchema])

  // Run the selected function
  const handleRun = useCallback(async () => {
    if (!functionDef || isRunning) return
    if (prereqStatus && !prereqStatus.satisfied) return

    setIsRunning(true)
    setResult(null)

    try {
      const response = await fetch(`${API_BASE}/scanpy/${selectedFunction}`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(paramValues),
      })

      const data = await response.json()

      if (!response.ok) {
        throw new Error(data.detail || `HTTP ${response.status}`)
      }

      // Build result message
      let message = 'Completed successfully'
      if (data.n_genes_removed !== undefined) {
        message = `Removed ${data.n_genes_removed} genes (${data.n_genes_before} → ${data.n_genes_after})`
      } else if (data.n_cells_removed !== undefined) {
        message = `Removed ${data.n_cells_removed} cells (${data.n_cells_before} → ${data.n_cells_after})`
      } else if (data.n_clusters !== undefined) {
        message = `Found ${data.n_clusters} clusters`
      } else if (data.embedding_name) {
        message = `Created embedding: ${data.embedding_name}`
      }

      setResult({ success: true, message })

      // Add to history
      const actionRecord: ScanpyActionRecord = {
        action: selectedFunction,
        params: paramValues as Record<string, unknown>,
        result: data,
        timestamp: new Date().toISOString(),
      }
      addScanpyAction(actionRecord)

      // Refresh schema if data shape may have changed
      if (['filter_genes', 'filter_cells', 'pca', 'umap', 'leiden'].includes(selectedFunction)) {
        await refreshSchema()
      }
    } catch (err) {
      setResult({ success: false, message: (err as Error).message })
    } finally {
      setIsRunning(false)
    }
  }, [functionDef, isRunning, prereqStatus, selectedFunction, paramValues, addScanpyAction, refreshSchema])

  if (!isScanpyModalOpen) return null

  return (
    <div style={styles.overlay} onClick={() => setScanpyModalOpen(false)}>
      <div style={styles.modal} onClick={(e) => e.stopPropagation()}>
        <div style={styles.header}>
          <div style={styles.title}>Scanpy Analysis</div>
          <button style={styles.closeButton} onClick={() => setScanpyModalOpen(false)}>
            &times;
          </button>
        </div>

        {/* Category tabs */}
        <div style={styles.categoryTabs}>
          {(Object.keys(SCANPY_FUNCTIONS) as CategoryKey[]).map((cat) => (
            <button
              key={cat}
              style={{
                ...styles.tab,
                ...(selectedCategory === cat ? styles.tabActive : {}),
              }}
              onClick={() => handleCategoryChange(cat)}
            >
              {SCANPY_FUNCTIONS[cat].label}
            </button>
          ))}
        </div>

        {/* Function select within category */}
        <select
          style={styles.functionSelect}
          value={selectedFunction}
          onChange={(e) => {
            setSelectedFunction(e.target.value)
            setResult(null)
          }}
        >
          {Object.entries(categoryDef.functions).map(([key, func]) => (
            <option key={key} value={key}>
              {func.label}
            </option>
          ))}
        </select>

        {/* Description */}
        {functionDef && (
          <div style={styles.description}>{functionDef.description}</div>
        )}

        {/* Prerequisite warning */}
        {prereqStatus && !prereqStatus.satisfied && (
          <div style={styles.prerequisiteWarning}>
            Requires: {prereqStatus.missing.map((p) => p.toUpperCase()).join(', ')} to be computed first
          </div>
        )}

        {/* Parameters */}
        {functionDef && functionDef.params.length > 0 && (
          <div style={styles.paramsSection}>
            {functionDef.params.map((param) => (
              <div key={param.name}>
                <div style={styles.paramRow}>
                  <label style={styles.paramLabel}>{param.label}</label>
                  {param.type === 'select' ? (
                    <select
                      style={styles.paramInput}
                      value={paramValues[param.name] ?? ''}
                      onChange={(e) => handleParamChange(param.name, e.target.value)}
                    >
                      {param.options?.map((opt) => (
                        <option key={opt} value={opt}>
                          {opt}
                        </option>
                      ))}
                    </select>
                  ) : (
                    <input
                      type={param.type === 'number' ? 'number' : 'text'}
                      style={styles.paramInput}
                      value={paramValues[param.name] ?? ''}
                      onChange={(e) => handleParamChange(param.name, e.target.value)}
                      placeholder={param.default === null ? '(optional)' : String(param.default)}
                    />
                  )}
                </div>
                <div style={styles.paramDescription}>{param.description}</div>
              </div>
            ))}
          </div>
        )}

        {/* Result display */}
        {result && (
          <div
            style={{
              ...styles.result,
              ...(result.success ? styles.resultSuccess : styles.resultError),
            }}
          >
            <div style={styles.resultText}>
              {result.success ? '✓ ' : '✗ '}
              {result.message}
            </div>
          </div>
        )}

        {/* Buttons */}
        <div style={styles.buttonRow}>
          <button style={styles.cancelButton} onClick={() => setScanpyModalOpen(false)}>
            Close
          </button>
          <button
            style={{
              ...styles.runButton,
              ...((isRunning || (prereqStatus && !prereqStatus.satisfied)) ? styles.runButtonDisabled : {}),
            }}
            onClick={handleRun}
            disabled={isRunning || (prereqStatus !== null && !prereqStatus.satisfied)}
          >
            {isRunning ? 'Running...' : 'Run'}
          </button>
        </div>

        {/* History section */}
        {scanpyActionHistory.length > 0 && (
          <div style={styles.historySection}>
            <div style={styles.historyTitle}>Session History ({scanpyActionHistory.length})</div>
            {scanpyActionHistory.slice(-5).reverse().map((action, i) => (
              <div key={i} style={styles.historyItem}>
                {action.action} - {new Date(action.timestamp).toLocaleTimeString()}
              </div>
            ))}
          </div>
        )}

        {/* Data info */}
        {schema && (
          <div style={{ marginTop: '12px', fontSize: '11px', color: '#666' }}>
            Current data: {schema.n_cells.toLocaleString()} cells, {schema.n_genes.toLocaleString()} genes
          </div>
        )}
      </div>
    </div>
  )
}
