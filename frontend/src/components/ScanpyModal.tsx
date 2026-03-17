import { useState, useCallback, useEffect } from 'react'
import { useStore, ScanpyActionRecord } from '../store'
import { appendDataset } from '../hooks/useData'

const API_BASE = '/api'

// Variance data for visualization
interface VarianceData {
  variance_ratio: number[]
  cumulative_variance: number[]
  n_comps_used: number
  n_comps_computed: number
  elbow_index: number | null
}

// Simple SVG-based variance chart component
function VarianceChart({ data, width = 300, height = 150 }: { data: VarianceData; width?: number; height?: number }) {
  const padding = { top: 20, right: 40, bottom: 30, left: 45 }
  const chartWidth = width - padding.left - padding.right
  const chartHeight = height - padding.top - padding.bottom

  const nPCs = Math.min(data.variance_ratio.length, 50) // Show max 50 PCs
  const varRatios = data.variance_ratio.slice(0, nPCs)
  const cumVar = data.cumulative_variance.slice(0, nPCs)

  const maxVar = Math.max(...varRatios) * 1.1
  const barWidth = Math.max(2, (chartWidth / nPCs) - 1)

  // Scale functions
  const xScale = (i: number) => padding.left + (i / nPCs) * chartWidth
  const yScaleVar = (v: number) => padding.top + chartHeight - (v / maxVar) * chartHeight
  const yScaleCum = (v: number) => padding.top + chartHeight - v * chartHeight

  // Build cumulative line path
  const cumLinePath = cumVar.map((v, i) =>
    `${i === 0 ? 'M' : 'L'} ${xScale(i + 0.5)} ${yScaleCum(v)}`
  ).join(' ')

  return (
    <svg width={width} height={height} style={{ backgroundColor: '#0a0a1a', borderRadius: '4px' }}>
      {/* Y-axis labels - variance ratio */}
      <text x={padding.left - 5} y={padding.top} fontSize="9" fill="#888" textAnchor="end">
        {(maxVar * 100).toFixed(0)}%
      </text>
      <text x={padding.left - 5} y={padding.top + chartHeight} fontSize="9" fill="#888" textAnchor="end">
        0%
      </text>
      <text x={padding.left - 25} y={padding.top + chartHeight / 2} fontSize="9" fill="#666" textAnchor="middle" transform={`rotate(-90, ${padding.left - 25}, ${padding.top + chartHeight / 2})`}>
        Var. Ratio
      </text>

      {/* Y-axis labels - cumulative (right side) */}
      <text x={width - padding.right + 5} y={padding.top} fontSize="9" fill="#4ecdc4" textAnchor="start">
        100%
      </text>
      <text x={width - padding.right + 5} y={padding.top + chartHeight} fontSize="9" fill="#4ecdc4" textAnchor="start">
        0%
      </text>

      {/* X-axis label */}
      <text x={padding.left + chartWidth / 2} y={height - 5} fontSize="9" fill="#666" textAnchor="middle">
        Principal Component
      </text>

      {/* Variance ratio bars */}
      {varRatios.map((v, i) => (
        <rect
          key={i}
          x={xScale(i) + 1}
          y={yScaleVar(v)}
          width={barWidth}
          height={yScaleVar(0) - yScaleVar(v)}
          fill={i < data.n_comps_used ? '#e94560' : '#444'}
          opacity={0.8}
        />
      ))}

      {/* Cumulative variance line */}
      <path
        d={cumLinePath}
        fill="none"
        stroke="#4ecdc4"
        strokeWidth={2}
      />

      {/* Elbow point marker */}
      {data.elbow_index !== null && data.elbow_index < nPCs && (
        <>
          <line
            x1={xScale(data.elbow_index + 0.5)}
            y1={padding.top}
            x2={xScale(data.elbow_index + 0.5)}
            y2={padding.top + chartHeight}
            stroke="#ffd700"
            strokeWidth={1}
            strokeDasharray="3,3"
          />
          <circle
            cx={xScale(data.elbow_index + 0.5)}
            cy={yScaleCum(cumVar[data.elbow_index])}
            r={4}
            fill="#ffd700"
          />
          <text
            x={xScale(data.elbow_index + 0.5)}
            y={padding.top - 5}
            fontSize="8"
            fill="#ffd700"
            textAnchor="middle"
          >
            elbow
          </text>
        </>
      )}

      {/* N comps used indicator */}
      {data.n_comps_used < nPCs && (
        <line
          x1={xScale(data.n_comps_used)}
          y1={padding.top}
          x2={xScale(data.n_comps_used)}
          y2={padding.top + chartHeight}
          stroke="#e94560"
          strokeWidth={1}
          opacity={0.5}
        />
      )}

      {/* Legend */}
      <rect x={padding.left + 5} y={padding.top + 2} width={8} height={8} fill="#e94560" />
      <text x={padding.left + 16} y={padding.top + 9} fontSize="8" fill="#888">Used PCs</text>
      <line x1={padding.left + 70} y1={padding.top + 6} x2={padding.left + 82} y2={padding.top + 6} stroke="#4ecdc4" strokeWidth={2} />
      <text x={padding.left + 85} y={padding.top + 9} fontSize="8" fill="#888">Cumulative</text>
    </svg>
  )
}

// Type definitions for scanpy functions
interface ParamDef {
  name: string
  label: string
  type: 'number' | 'text' | 'select' | 'gene_subset' | 'textarea'
  default: string | number | null
  description: string
  options?: string[]
  visibleWhen?: { param: string; value: string }
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
          { name: 'min_cells', label: 'Min cells', type: 'number', default: 25, description: 'Minimum cells expressing' },
          { name: 'max_cells', label: 'Max cells', type: 'number', default: null, description: 'Maximum cells expressing' },
        ],
      },
      exclude_genes: {
        label: 'Exclude Genes',
        description: 'Remove genes by name or regex pattern (e.g. ^mt- for mitochondrial, ^Gm\\d+ for predicted)',
        prerequisites: [],
        params: [
          { name: 'gene_names', label: 'Gene names', type: 'textarea', default: null, description: 'One gene name per line (exact match)' },
          { name: 'patterns', label: 'Regex patterns', type: 'text', default: null, description: 'Comma-separated regex patterns (e.g. ^mt-,^Gm\\d+)' },
        ],
      },
      filter_cells: {
        label: 'Filter Cells',
        description: 'Remove cells based on counts or number of genes expressed',
        prerequisites: [],
        params: [
          { name: 'min_counts', label: 'Min counts', type: 'number', default: null, description: 'Minimum total counts' },
          { name: 'max_counts', label: 'Max counts', type: 'number', default: null, description: 'Maximum total counts' },
          { name: 'min_genes', label: 'Min genes', type: 'number', default: 25, description: 'Minimum genes expressed' },
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
      highly_variable_genes: {
        label: 'Highly Variable Genes',
        description: 'Identify highly variable genes for downstream analysis',
        prerequisites: [],
        params: [
          { name: 'n_top_genes', label: 'Top N genes', type: 'number', default: 2000, description: 'Number of top genes (overrides thresholds if set)' },
          { name: 'min_mean', label: 'Min mean', type: 'number', default: 0.0125, description: 'Minimum mean expression' },
          { name: 'max_mean', label: 'Max mean', type: 'number', default: 3.0, description: 'Maximum mean expression' },
          { name: 'min_disp', label: 'Min dispersion', type: 'number', default: 0.5, description: 'Minimum normalized dispersion' },
          { name: 'flavor', label: 'Method', type: 'select', default: 'cell_ranger', options: ['seurat', 'cell_ranger', 'seurat_v3'], description: 'HVG selection method' },
        ],
      },
    },
  },
  cell_analysis: {
    label: 'Cell Analysis',
    functions: {
      pca: {
        label: 'PCA',
        description: 'Principal component analysis on cells',
        prerequisites: [],
        params: [
          { name: 'n_comps', label: 'Components', type: 'number', default: 50, description: 'Number of principal components' },
          { name: 'svd_solver', label: 'SVD solver', type: 'select', default: 'arpack', options: ['arpack', 'randomized', 'auto'], description: 'SVD algorithm' },
          { name: 'gene_subset', label: 'Gene Subset', type: 'gene_subset', default: null, description: 'Filter genes using boolean columns from .var (default: use highly_variable if available)' },
        ],
      },
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
      leiden: {
        label: 'Leiden',
        description: 'Leiden clustering algorithm',
        prerequisites: ['neighbors'],
        params: [
          { name: 'resolution', label: 'Resolution', type: 'number', default: 0.5, description: 'Higher = more clusters' },
          { name: 'key_added', label: 'Column name', type: 'text', default: 'leiden', description: 'Name for cluster labels' },
        ],
      },
    },
  },
  gene_analysis: {
    label: 'Gene Analysis',
    functions: {
      build_gene_graph: {
        label: 'Build Gene Graph',
        description: 'Build gene-gene similarity graph (runs PCA + neighbors)',
        prerequisites: [],
        params: [
          { name: 'n_pcs', label: 'Num PCs', type: 'number', default: null, description: 'Number of PCs (null = auto-detect with Kneedle)' },
          { name: 'scale', label: 'Scale', type: 'select', default: 'true', options: ['true', 'false'], description: 'Z-score scale genes before PCA' },
          { name: 'n_neighbors', label: 'Neighbors', type: 'number', default: 15, description: 'Number of gene neighbors' },
          { name: 'metric', label: 'Metric', type: 'select', default: 'euclidean', options: ['euclidean', 'cosine'], description: 'Distance metric' },
        ],
      },
      gene_pca: {
        label: 'Gene PCA',
        description: 'PCA on genes (expression patterns across cells)',
        prerequisites: [],
        params: [
          { name: 'n_comps', label: 'Components', type: 'number', default: null, description: 'Number of PCs (null = auto with Kneedle)' },
          { name: 'scale', label: 'Scale', type: 'select', default: 'true', options: ['true', 'false'], description: 'Z-score scale genes' },
          { name: 'use_kneedle', label: 'Auto-detect PCs', type: 'select', default: 'true', options: ['true', 'false'], description: 'Use Kneedle algorithm' },
          { name: 'max_comps', label: 'Max components', type: 'number', default: 100, description: 'Max PCs to compute for Kneedle' },
          { name: 'gene_subset', label: 'Gene Subset', type: 'gene_subset', default: null, description: 'Filter to specific genes using boolean columns from .var' },
        ],
      },
      gene_neighbors: {
        label: 'Gene Neighbors',
        description: 'Compute gene-gene kNN graph',
        prerequisites: [],
        params: [
          { name: 'basis', label: 'Basis', type: 'select', default: 'gene_pca', options: ['gene_pca', 'expression'], description: 'Use gene PCA embedding or raw expression' },
          { name: 'n_neighbors', label: 'Neighbors', type: 'number', default: 15, description: 'Number of neighbors per gene' },
          { name: 'metric', label: 'Metric', type: 'select', default: 'euclidean', options: ['euclidean', 'cosine'], description: 'Distance metric' },
          { name: 'gene_subset', label: 'Gene Subset', type: 'gene_subset', default: null, description: 'Filter to specific genes using boolean columns from .var', visibleWhen: { param: 'basis', value: 'expression' } },
          { name: 'scale', label: 'Scale', type: 'select', default: 'true', options: ['true', 'false'], description: 'Z-score scale genes before computing neighbors', visibleWhen: { param: 'basis', value: 'expression' } },
        ],
      },
      find_similar_genes: {
        label: 'Find Similar Genes',
        description: 'Find genes with similar expression patterns',
        prerequisites: ['gene_neighbors'],
        params: [
          { name: 'gene', label: 'Query gene', type: 'text', default: '', description: 'Gene name to find similar genes for' },
          { name: 'n_neighbors', label: 'Num results', type: 'number', default: 10, description: 'Number of similar genes to return' },
          { name: 'use', label: 'Similarity', type: 'select', default: 'connectivities', options: ['connectivities', 'distances'], description: 'Use connectivity weights or distances' },
        ],
      },
      cluster_genes: {
        label: 'Cluster Genes',
        description: 'Cluster genes into co-expression modules (Leiden)',
        prerequisites: ['gene_neighbors'],
        params: [
          { name: 'resolution', label: 'Resolution', type: 'number', default: 0.5, description: 'Higher = more clusters' },
          { name: 'key_added', label: 'Column name', type: 'text', default: 'gene_cluster', description: 'Name for cluster labels in .var' },
        ],
      },
    },
  },
  spatial_analysis: {
    label: 'Spatial Analysis',
    functions: {
      spatial_neighbors: {
        label: 'Spatial Neighbors',
        description: 'Build spatial neighborhood graph from coordinates',
        prerequisites: ['has_spatial'],
        params: [
          { name: 'n_neighs', label: 'Neighbors', type: 'number', default: 6, description: 'Number of spatial neighbors' },
          { name: 'coord_type', label: 'Coord Type', type: 'select', default: '', options: ['', 'grid', 'generic'], description: 'Coordinate type (empty = auto-detect)' },
          { name: 'delaunay', label: 'Delaunay', type: 'select', default: 'false', options: ['true', 'false'], description: 'Use Delaunay triangulation' },
        ],
      },
      spatial_autocorr: {
        label: 'Spatial Autocorrelation',
        description: "Identify spatially variable genes (Moran's I or Geary's C)",
        prerequisites: ['spatial_neighbors'],
        params: [
          { name: 'mode', label: 'Method', type: 'select', default: 'moran', options: ['moran', 'geary'], description: "Moran's I (clustering) or Geary's C" },
          { name: 'n_perms', label: 'Permutations', type: 'number', default: 100, description: 'Permutations for p-value (0 for analytical only)' },
          { name: 'pval_threshold', label: 'P-value Threshold', type: 'number', default: 0.05, description: 'Threshold for spatially variable' },
          { name: 'n_jobs', label: 'Parallel Jobs', type: 'number', default: 1, description: 'Number of parallel jobs' },
        ],
      },
      contourize: {
        label: 'Contourize',
        description: 'Assign cells to spatial expression contour levels based on selected genes. Select genes in the Gene Panel first.',
        prerequisites: ['has_spatial'],
        params: [
          { name: 'contour_levels', label: 'Contour levels', type: 'number', default: 6, description: 'Number of contour thresholds' },
          { name: 'smooth_sigma', label: 'Smoothing sigma', type: 'number', default: 2.0, description: 'Gaussian smoothing strength' },
          { name: 'grid_res', label: 'Grid resolution', type: 'number', default: 200, description: 'Interpolation grid size' },
          { name: 'log_transform', label: 'Log transform', type: 'select', default: 'true', options: ['true', 'false'], description: 'Apply log1p before contouring' },
          { name: 'annotation_key', label: 'Column name', type: 'text', default: null, description: 'Name for result column (auto-generated if empty)' },
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

// Boolean column info from .var
interface BooleanColumn {
  name: string
  n_true: number
  n_total: number
}

export default function ScanpyModal() {
  const { isScanpyModalOpen, setScanpyModalOpen, schema, setSchema, scanpyActionHistory, addScanpyAction, activeCellMask, resetActiveCells, refreshObsSummaries, setColorBy, setEmbedding, setSelectedEmbedding, selectedGenes } = useStore()

  const [selectedCategory, setSelectedCategory] = useState<CategoryKey>('preprocessing')
  const [selectedFunction, setSelectedFunction] = useState<FunctionKey>('filter_genes')
  const [paramValues, setParamValues] = useState<ParamValues>({})
  const [isRunning, setIsRunning] = useState(false)
  const [result, setResult] = useState<{ success: boolean; message: string } | null>(null)
  const [prereqStatus, setPrereqStatus] = useState<PrerequisiteStatus | null>(null)
  const [varianceData, setVarianceData] = useState<VarianceData | null>(null)
  const [cellVarianceData, setCellVarianceData] = useState<VarianceData | null>(null)

  // Gene subset selection state
  const [booleanColumns, setBooleanColumns] = useState<BooleanColumn[]>([])
  const [selectedGeneColumns, setSelectedGeneColumns] = useState<string[]>([])
  const [geneSubsetOperation, setGeneSubsetOperation] = useState<'intersection' | 'union'>('intersection')

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
      setVarianceData(null)
    }
  }, [selectedFunction, functionDef])

  // Fetch cell PCA variance when Neighbors is selected
  useEffect(() => {
    if (selectedFunction !== 'neighbors') {
      setCellVarianceData(null)
      return
    }
    fetch(appendDataset(`${API_BASE}/scanpy/cell_pca_variance`))
      .then((res) => { if (res.ok) return res.json(); throw new Error() })
      .then(setCellVarianceData)
      .catch(() => setCellVarianceData(null))
  }, [selectedFunction, scanpyActionHistory])

  // Check prerequisites when function changes
  useEffect(() => {
    if (!functionDef || functionDef.prerequisites.length === 0) {
      setPrereqStatus({ satisfied: true, missing: [] })
      return
    }

    fetch(appendDataset(`${API_BASE}/scanpy/prerequisites/${selectedFunction}`))
      .then((res) => res.json())
      .then(setPrereqStatus)
      .catch(() => setPrereqStatus({ satisfied: false, missing: ['unknown'] }))
  }, [selectedFunction, functionDef, scanpyActionHistory])

  // Fetch boolean columns for gene subset selection
  useEffect(() => {
    if (!isScanpyModalOpen) return

    fetch(appendDataset(`${API_BASE}/var/boolean_columns`))
      .then((res) => res.json())
      .then(setBooleanColumns)
      .catch(() => setBooleanColumns([]))
  }, [isScanpyModalOpen, scanpyActionHistory])

  // Reset gene subset selection when function changes
  useEffect(() => {
    setSelectedGeneColumns([])
    setGeneSubsetOperation('intersection')
  }, [selectedFunction])

  // Handle category change
  const handleCategoryChange = useCallback((category: CategoryKey) => {
    setSelectedCategory(category)
    const firstFunction = Object.keys(SCANPY_FUNCTIONS[category].functions)[0]
    setSelectedFunction(firstFunction)
    setResult(null)
    setVarianceData(null)
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
      const response = await fetch(appendDataset(`${API_BASE}/schema`))
      if (response.ok) {
        const newSchema = await response.json()
        setSchema(newSchema)
      }
    } catch (err) {
      console.error('Failed to refresh schema:', err)
    }
  }, [setSchema])

  // Load variance chart data (for viewing existing gene PCA)
  const loadVarianceChart = useCallback(async () => {
    try {
      const response = await fetch(appendDataset(`${API_BASE}/scanpy/gene_pca_variance`))
      if (response.ok) {
        const data = await response.json()
        setVarianceData(data)
      } else {
        const err = await response.json()
        setResult({ success: false, message: err.detail || 'Gene PCA not yet computed' })
      }
    } catch {
      setResult({ success: false, message: 'Failed to load variance data' })
    }
  }, [])

  // Run the selected function
  const handleRun = useCallback(async () => {
    if (!functionDef || isRunning) return
    if (prereqStatus && !prereqStatus.satisfied) return

    setIsRunning(true)
    setResult(null)

    try {
      // Build set of hidden param names (visibleWhen condition not met)
      const hiddenParams = new Set<string>()
      for (const p of functionDef.params) {
        if (p.visibleWhen && String(paramValues[p.visibleWhen.param]) !== p.visibleWhen.value) {
          hiddenParams.add(p.name)
        }
      }

      // Convert 'true'/'false' strings to actual booleans for the request
      const requestParams: Record<string, unknown> = {}
      for (const [key, value] of Object.entries(paramValues)) {
        if (hiddenParams.has(key)) continue
        if (value === 'true') {
          requestParams[key] = true
        } else if (value === 'false') {
          requestParams[key] = false
        } else {
          requestParams[key] = value
        }
      }

      // Handle gene_subset parameter specially (only if visible)
      if (functionDef.params.some(p => p.type === 'gene_subset' && !hiddenParams.has(p.name))) {
        if (selectedGeneColumns.length === 0) {
          requestParams['gene_subset'] = null
        } else if (selectedGeneColumns.length === 1) {
          requestParams['gene_subset'] = selectedGeneColumns[0]
        } else {
          requestParams['gene_subset'] = {
            columns: selectedGeneColumns,
            operation: geneSubsetOperation,
          }
        }
      }

      // Handle exclude_genes: parse textarea into arrays
      if (selectedFunction === 'exclude_genes') {
        const namesStr = String(requestParams['gene_names'] || '').trim()
        const patternsStr = String(requestParams['patterns'] || '').trim()
        requestParams['gene_names'] = namesStr ? namesStr.split(/\n+/).map((s: string) => s.trim()).filter(Boolean) : null
        requestParams['patterns'] = patternsStr ? patternsStr.split(',').map((s: string) => s.trim()).filter(Boolean) : null
        if (!requestParams['gene_names'] && !requestParams['patterns']) {
          setResult({ success: false, message: 'Provide at least one gene name or regex pattern.' })
          setIsRunning(false)
          return
        }
      }

      // Inject selected genes for contourize
      if (selectedFunction === 'contourize') {
        if (selectedGenes.length === 0) {
          setResult({ success: false, message: 'No genes selected. Select genes in the Gene Panel first.' })
          setIsRunning(false)
          return
        }
        requestParams['genes'] = selectedGenes
        // Convert log_transform from string to boolean
        if (requestParams['log_transform'] !== undefined) {
          requestParams['log_transform'] = requestParams['log_transform'] === 'true'
        }
        // Remove empty annotation_key so backend auto-generates
        if (!requestParams['annotation_key']) {
          delete requestParams['annotation_key']
        }
      }

      // Add active cell indices if a cell mask is active
      if (activeCellMask) {
        const activeIndices: number[] = []
        activeCellMask.forEach((active, idx) => { if (active) activeIndices.push(idx) })
        if (activeIndices.length === 0) {
          setResult({ success: false, message: 'No active cells. Reset the cell mask first.' })
          setIsRunning(false)
          return
        }
        requestParams['active_cell_indices'] = activeIndices
      }

      const response = await fetch(appendDataset(`${API_BASE}/scanpy/${selectedFunction}`), {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(requestParams),
      })

      const data = await response.json()

      if (!response.ok) {
        throw new Error(data.detail || `HTTP ${response.status}`)
      }

      // Build result message
      let message = 'Completed successfully'
      if (data.n_genes_removed !== undefined && data.removed_genes) {
        // exclude_genes — show some removed names
        const preview = data.removed_genes.slice(0, 5).join(', ')
        message = `Removed ${data.n_genes_removed} genes (${data.n_genes_before} → ${data.n_genes_after}): ${preview}${data.n_genes_removed > 5 ? '...' : ''}`
      } else if (data.n_genes_removed !== undefined) {
        message = `Removed ${data.n_genes_removed} genes (${data.n_genes_before} → ${data.n_genes_after})`
      } else if (data.n_cells_removed !== undefined) {
        message = `Removed ${data.n_cells_removed} cells (${data.n_cells_before} → ${data.n_cells_after})`
      } else if (data.module_sizes !== undefined && data.n_clusters !== undefined) {
        // cluster_genes result - create folder in gene_clusters category
        const totalModules = Object.keys(data.module_sizes).length
        message = `Created ${totalModules} gene modules (${data.n_genes_clustered} genes clustered)`

        // Fetch the gene modules and add them to the gene_clusters category
        try {
          const modulesResponse = await fetch(appendDataset(`${API_BASE}/scanpy/gene_modules`))
          if (modulesResponse.ok) {
            const modulesData = await modulesResponse.json()
            const timestamp = new Date().toLocaleString('en-US', {
              month: 'short', day: 'numeric', hour: '2-digit', minute: '2-digit'
            })
            const folderName = `Clusters (res=${requestParams.resolution || 0.5}) ${timestamp}`
            const geneSets = Object.entries(modulesData.modules || {}).map(([name, genes]) => ({
              name: name.replace('module_', 'Module '),
              genes: genes as string[],
            }))
            useStore.getState().addFolderToCategory('gene_clusters', folderName, geneSets)
          }
        } catch {
          console.error('Failed to fetch gene modules for folder creation')
        }
      } else if (data.n_clusters !== undefined) {
        // leiden cell clustering result
        message = `Found ${data.n_clusters} clusters`
      } else if (data.embedding_name && data.n_genes_used !== undefined) {
        // PCA with gene subset info
        const subsetInfo = data.gene_subset_type && data.gene_subset_type !== 'default'
          ? ` on ${data.n_genes_used.toLocaleString()} genes (${data.gene_subset_type})`
          : ''
        message = `Created embedding: ${data.embedding_name} (${data.n_comps} PCs)${subsetInfo}`
      } else if (data.embedding_name) {
        message = `Created embedding: ${data.embedding_name}`
      } else if (data.similar_genes !== undefined) {
        // find_similar_genes result
        if (data.similar_genes.length === 0) {
          message = `No similar genes found for ${data.query_gene}`
        } else {
          message = `Similar to ${data.query_gene}: ${data.similar_genes.slice(0, 5).join(', ')}${data.similar_genes.length > 5 ? '...' : ''}`
        }
      } else if (data.pca !== undefined && data.neighbors !== undefined) {
        // build_gene_graph result
        message = `Gene graph built: ${data.pca.n_comps} PCs, ${data.neighbors.n_neighbors} neighbors`
      } else if (data.n_comps !== undefined && data.cumulative_variance !== undefined) {
        // gene_pca result
        const varPct = (data.cumulative_variance * 100).toFixed(1)
        const subsetInfo = data.n_genes_used !== undefined && data.n_genes_used !== data.n_comps
          ? ` on ${data.n_genes_used.toLocaleString()} genes`
          : ''
        message = `Gene PCA: ${data.n_comps} PCs (${varPct}% variance)${subsetInfo}${data.elbow_detected !== null ? `, elbow at PC ${data.elbow_detected + 1}` : ''}`
      } else if (data.n_genes !== undefined && data.n_neighbors !== undefined) {
        // gene_neighbors result
        message = `Gene kNN graph: ${data.n_neighbors} neighbors for ${data.n_genes.toLocaleString()} genes`
      } else if (data.n_edges !== undefined && data.spatial_key !== undefined) {
        // spatial_neighbors result
        message = `Spatial graph: ${data.n_edges.toLocaleString()} edges for ${data.n_cells.toLocaleString()} cells`
      } else if (data.n_spatially_variable !== undefined) {
        // spatial_autocorr result
        const topGenes = data.top_sv_genes?.slice(0, 3).join(', ') || ''
        message = `Found ${data.n_spatially_variable.toLocaleString()} spatially variable genes (${data.mode === 'moran' ? "Moran's I" : "Geary's C"})${topGenes ? `. Top: ${topGenes}...` : ''}`

        // Add top spatially variable genes to the spatial category
        if (data.top_sv_genes && data.top_sv_genes.length > 0) {
          const timestamp = new Date().toLocaleString('en-US', {
            month: 'short', day: 'numeric', hour: '2-digit', minute: '2-digit'
          })
          const setName = `Top SVG (${data.mode === 'moran' ? "Moran's I" : "Geary's C"}) ${timestamp}`
          useStore.getState().addGeneSetToCategory('manual', setName, data.top_sv_genes)
        }
      } else if (data.annotation_key !== undefined && data.contour_levels !== undefined) {
        // contourize result
        message = `Contourize: created "${data.annotation_key}" with ${data.contour_levels} levels from ${data.n_genes} gene${data.n_genes > 1 ? 's' : ''}`
      } else if (data.n_highly_variable !== undefined) {
        // highly_variable_genes result
        message = `Identified ${data.n_highly_variable.toLocaleString()} highly variable genes (${data.flavor} method)`
      }

      setResult({ success: true, message })

      // Add to history
      const actionRecord: ScanpyActionRecord = {
        action: selectedFunction,
        params: requestParams,
        result: data,
        timestamp: new Date().toISOString(),
      }
      addScanpyAction(actionRecord)

      // Refresh schema if data shape may have changed
      if (['filter_genes', 'exclude_genes', 'filter_cells', 'pca', 'umap', 'leiden', 'cluster_genes', 'spatial_autocorr', 'highly_variable_genes', 'contourize'].includes(selectedFunction)) {
        await refreshSchema()
        // Also refresh obs summaries so Cell Manager shows new/updated columns (e.g. leiden clusters, contour levels)
        refreshObsSummaries()
      }

      // Invalidate cached colorBy so it re-fetches (e.g. leiden labels changed)
      if (['leiden', 'filter_cells'].includes(selectedFunction)) {
        setColorBy(null)
      }

      // Invalidate cached embedding so it re-fetches (e.g. new UMAP coordinates)
      if (['umap', 'pca', 'filter_cells'].includes(selectedFunction)) {
        setEmbedding(null)
        // Auto-select the newly created embedding (important when dataset had none initially)
        if (data.embedding_name) {
          setSelectedEmbedding(data.embedding_name)
        }
      }

      // Reset cell mask after filter_cells removes cells (indices become stale)
      if (selectedFunction === 'filter_cells' && data.n_cells_removed > 0) {
        resetActiveCells()
      }

      // Fetch variance data for visualization after gene PCA functions
      if (['gene_pca', 'build_gene_graph'].includes(selectedFunction)) {
        try {
          const varResponse = await fetch(appendDataset(`${API_BASE}/scanpy/gene_pca_variance`))
          if (varResponse.ok) {
            const varData = await varResponse.json()
            setVarianceData(varData)
          }
        } catch {
          // Ignore variance fetch errors
        }
      }
    } catch (err) {
      setResult({ success: false, message: (err as Error).message })
    } finally {
      setIsRunning(false)
    }
  }, [functionDef, isRunning, prereqStatus, selectedFunction, paramValues, selectedGeneColumns, geneSubsetOperation, addScanpyAction, refreshSchema, activeCellMask, resetActiveCells, refreshObsSummaries, setColorBy, setEmbedding, setSelectedEmbedding, selectedGenes])

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
          <div style={styles.description}>
            {functionDef.description}
            {/* Show "View Variance Chart" button for gene analysis functions */}
            {selectedCategory === 'gene_analysis' && !varianceData && (
              <button
                onClick={loadVarianceChart}
                style={{
                  marginLeft: '12px',
                  padding: '3px 8px',
                  fontSize: '11px',
                  backgroundColor: '#1a1a2e',
                  color: '#4ecdc4',
                  border: '1px solid #4ecdc4',
                  borderRadius: '3px',
                  cursor: 'pointer',
                }}
              >
                View Variance Chart
              </button>
            )}
          </div>
        )}

        {/* Prerequisite warning */}
        {prereqStatus && !prereqStatus.satisfied && (
          <div style={styles.prerequisiteWarning}>
            Requires: {prereqStatus.missing.map((p) => p.toUpperCase()).join(', ')} to be computed first
          </div>
        )}

        {/* Cell mask indicator */}
        {activeCellMask && (
          <div style={{ fontSize: '12px', color: '#4ecdc4', backgroundColor: 'rgba(78,205,196,0.1)',
            padding: '8px', borderRadius: '4px', marginBottom: '16px', borderLeft: '3px solid #4ecdc4' }}>
            Cell mask active: operating on {activeCellMask.filter(Boolean).length.toLocaleString()} of {schema?.n_cells.toLocaleString()} cells
          </div>
        )}

        {/* Gene input indicator for contourize */}
        {selectedFunction === 'contourize' && (
          <div style={{ fontSize: '12px', padding: '8px', borderRadius: '4px', marginBottom: '16px',
            borderLeft: `3px solid ${selectedGenes.length > 0 ? '#4ecdc4' : '#e94560'}`,
            backgroundColor: selectedGenes.length > 0 ? 'rgba(78,205,196,0.1)' : 'rgba(233,69,96,0.1)',
            color: selectedGenes.length > 0 ? '#4ecdc4' : '#e94560' }}>
            {selectedGenes.length > 0
              ? `Using ${selectedGenes.length} selected gene${selectedGenes.length > 1 ? 's' : ''}: ${selectedGenes.slice(0, 5).join(', ')}${selectedGenes.length > 5 ? '...' : ''}`
              : 'No genes selected. Select genes in the Gene Panel first.'}
          </div>
        )}

        {/* Cell PCA variance chart for Neighbors */}
        {selectedFunction === 'neighbors' && cellVarianceData && (
          <div style={{ marginBottom: '12px' }}>
            <div style={{ fontSize: '12px', color: '#888', marginBottom: '6px' }}>
              Cell PCA — % Variance Explained
            </div>
            <VarianceChart data={cellVarianceData} width={400} height={140} />
            <div style={{ fontSize: '10px', color: '#666', marginTop: '4px' }}>
              {cellVarianceData.n_comps_computed} PCs computed
              {cellVarianceData.elbow_index !== null && ` · elbow at PC ${cellVarianceData.elbow_index + 1}`}
            </div>
          </div>
        )}

        {/* Parameters */}
        {functionDef && functionDef.params.length > 0 && (
          <div style={styles.paramsSection}>
            {functionDef.params.map((param) => {
              // Skip params whose visibleWhen condition is not met
              if (param.visibleWhen) {
                const refValue = paramValues[param.visibleWhen.param]
                if (String(refValue) !== param.visibleWhen.value) return null
              }
              return (
              <div key={param.name}>
                {param.type === 'gene_subset' ? (
                  /* Special UI for gene subset selection */
                  <div style={{ marginBottom: '12px' }}>
                    <label style={styles.paramLabel}>{param.label}</label>
                    <div style={styles.paramDescription}>{param.description}</div>
                    {booleanColumns.length === 0 ? (
                      <div style={{ fontSize: '12px', color: '#888', padding: '8px', backgroundColor: '#1a1a2e', borderRadius: '4px', marginTop: '8px' }}>
                        No boolean columns available. Run spatial_autocorr or use scanpy's highly_variable_genes to create filterable gene sets.
                      </div>
                    ) : (
                      <div style={{ marginTop: '8px' }}>
                        <div style={{ fontSize: '11px', color: '#888', marginBottom: '6px' }}>
                          Select columns to filter genes (leave empty for all genes):
                        </div>
                        <div style={{ display: 'flex', flexWrap: 'wrap', gap: '6px', marginBottom: '8px' }}>
                          {booleanColumns.map((col) => {
                            const isSelected = selectedGeneColumns.includes(col.name)
                            return (
                              <button
                                key={col.name}
                                onClick={() => {
                                  setSelectedGeneColumns((prev) =>
                                    isSelected
                                      ? prev.filter((c) => c !== col.name)
                                      : [...prev, col.name]
                                  )
                                }}
                                style={{
                                  padding: '4px 10px',
                                  fontSize: '11px',
                                  backgroundColor: isSelected ? '#4ecdc4' : '#1a1a2e',
                                  color: isSelected ? '#000' : '#aaa',
                                  border: `1px solid ${isSelected ? '#4ecdc4' : '#333'}`,
                                  borderRadius: '12px',
                                  cursor: 'pointer',
                                }}
                              >
                                {col.name} ({col.n_true.toLocaleString()})
                              </button>
                            )
                          })}
                        </div>
                        {selectedGeneColumns.length >= 2 && (
                          <div style={{ display: 'flex', alignItems: 'center', gap: '12px', marginTop: '8px' }}>
                            <span style={{ fontSize: '11px', color: '#888' }}>Combine with:</span>
                            <label style={{ fontSize: '12px', color: '#ccc', cursor: 'pointer' }}>
                              <input
                                type="radio"
                                name="geneSubsetOp"
                                checked={geneSubsetOperation === 'intersection'}
                                onChange={() => setGeneSubsetOperation('intersection')}
                                style={{ marginRight: '4px' }}
                              />
                              AND (intersection)
                            </label>
                            <label style={{ fontSize: '12px', color: '#ccc', cursor: 'pointer' }}>
                              <input
                                type="radio"
                                name="geneSubsetOp"
                                checked={geneSubsetOperation === 'union'}
                                onChange={() => setGeneSubsetOperation('union')}
                                style={{ marginRight: '4px' }}
                              />
                              OR (union)
                            </label>
                          </div>
                        )}
                        {selectedGeneColumns.length > 0 && (
                          <div style={{ fontSize: '11px', color: '#4ecdc4', marginTop: '8px' }}>
                            {selectedGeneColumns.length === 1
                              ? `Using genes where ${selectedGeneColumns[0]} = True`
                              : `Using genes where ${selectedGeneColumns.join(geneSubsetOperation === 'intersection' ? ' AND ' : ' OR ')}`}
                          </div>
                        )}
                      </div>
                    )}
                  </div>
                ) : (
                  <>
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
                      ) : param.type === 'textarea' ? (
                        <textarea
                          style={{ ...styles.paramInput, height: '60px', resize: 'vertical' as const, fontFamily: 'monospace', fontSize: '11px' }}
                          value={paramValues[param.name] ?? ''}
                          onChange={(e) => handleParamChange(param.name, e.target.value)}
                          placeholder={param.default === null ? '(optional)' : String(param.default)}
                        />
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
                  </>
                )}
              </div>
            )})}
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

        {/* Variance ratio chart for gene PCA */}
        {varianceData && (
          <div style={{ marginTop: '16px' }}>
            <div style={{ fontSize: '12px', color: '#888', marginBottom: '8px' }}>
              Gene PCA Variance Explained
            </div>
            <VarianceChart data={varianceData} width={400} height={160} />
            <div style={{ fontSize: '10px', color: '#666', marginTop: '4px' }}>
              Using {varianceData.n_comps_used} of {varianceData.n_comps_computed} computed PCs
              {varianceData.elbow_index !== null && ` (elbow detected at PC ${varianceData.elbow_index + 1})`}
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
