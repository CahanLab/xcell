import { useCallback, useEffect, useMemo, useState, useRef } from 'react'
import { useStore, DatasetSlot } from './store'
import { useSchema, useEmbedding, useColorBy, useDataActions, exportAnnotations, useExpressionTransformEffect, useBivariateTransformEffect, runDiffExp, appendDataset } from './hooks/useData'
import ScatterPlot, { BIVARIATE_COLORMAPS, getBivariateColor } from './components/ScatterPlot'
import GenePanel from './components/GenePanel'
import CellPanel from './components/CellPanel'
import DisplaySettings from './components/DisplaySettings'
import DiffExpModal from './components/DiffExpModal'
import LineAssociationModal from './components/LineAssociationModal'
import ScanpyModal from './components/ScanpyModal'
import ShapeManager from './components/ShapeManager'
import HeatmapView from './components/HeatmapView'
import MarkerGenesModal from './components/MarkerGenesModal'
import { MESSAGES } from './messages'

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
    flex: '0 0 auto',
  },
  logoGroup: {
    flex: '0 0 auto',
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
    display: 'flex',
    flexDirection: 'column' as const,
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
    top: '12px',
    left: '50%',
    transform: 'translateX(-50%)',
    padding: '8px 16px',
    backgroundColor: '#4a1c1c',
    border: '1px solid #e94560',
    borderRadius: '6px',
    color: '#e94560',
    fontSize: '12px',
    zIndex: 10,
    cursor: 'pointer',
    display: 'flex',
    alignItems: 'center',
    maxWidth: '80%',
    whiteSpace: 'nowrap' as const,
    overflow: 'hidden',
    textOverflow: 'ellipsis',
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
  tabBar: {
    display: 'flex',
    gap: '0px',
    borderBottom: '1px solid #0f3460',
    flexShrink: 0,
  },
  tab: {
    padding: '6px 16px',
    fontSize: '12px',
    fontWeight: 500,
    color: '#888',
    backgroundColor: 'transparent',
    border: 'none',
    borderBottom: '2px solid transparent',
    cursor: 'pointer',
  },
  tabActive: {
    color: '#e94560',
    borderBottomColor: '#e94560',
  },
  vizContent: {
    flex: 1,
    position: 'relative' as const,
    overflow: 'hidden',
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

// Canvas-based bivariate legend that accurately reflects the bilinear interpolation
function BivariateLegend({
  bivariateData,
  colormap,
  sortReversed,
  onToggleSort,
}: {
  bivariateData: { genes1: string[]; genes2: string[]; transform?: string }
  colormap: import('./store').BivariateColormap
  sortReversed: boolean
  onToggleSort: () => void
}) {
  const canvasRef = useRef<HTMLCanvasElement>(null)
  const size = 80

  useEffect(() => {
    const canvas = canvasRef.current
    if (!canvas) return

    const ctx = canvas.getContext('2d')
    if (!ctx) return

    const imageData = ctx.createImageData(size, size)
    const data = imageData.data

    for (let y = 0; y < size; y++) {
      for (let x = 0; x < size; x++) {
        const u = x / (size - 1)  // gene set 1 (horizontal, 0=left, 1=right)
        const v = 1 - y / (size - 1)  // gene set 2 (vertical, 0=bottom, 1=top, but y increases down)
        const color = getBivariateColor(u, v, colormap)
        const idx = (y * size + x) * 4
        data[idx] = color[0]
        data[idx + 1] = color[1]
        data[idx + 2] = color[2]
        data[idx + 3] = 255
      }
    }

    ctx.putImageData(imageData, 0, 0)
  }, [colormap])

  // Get corner colors for axis labels
  const corners = BIVARIATE_COLORMAPS[colormap]
  const color1 = `rgb(${corners.c10.join(',')})`  // High gene1 color
  const color2 = `rgb(${corners.c01.join(',')})`  // High gene2 color

  return (
    <div style={styles.expressionLegend}>
      <div style={styles.legendTitle}>
        Bivariate Expression
        {bivariateData.transform === 'log1p' && (
          <span style={{ fontSize: '9px', color: '#4ecdc4', marginLeft: '6px' }}>
            (log1p)
          </span>
        )}
      </div>
      <canvas
        ref={canvasRef}
        width={size}
        height={size}
        style={{ borderRadius: '4px', marginBottom: '4px' }}
      />
      <div style={{ fontSize: '10px', color: '#888' }}>
        <div style={{ display: 'flex', alignItems: 'center', gap: '4px' }}>
          <span style={{ color: color1 }}>→</span>
          <span>{bivariateData.genes1.length === 1 ? bivariateData.genes1[0] : `${bivariateData.genes1.length} genes`}</span>
        </div>
        <div style={{ display: 'flex', alignItems: 'center', gap: '4px' }}>
          <span style={{ color: color2 }}>↑</span>
          <span>{bivariateData.genes2.length === 1 ? bivariateData.genes2[0] : `${bivariateData.genes2.length} genes`}</span>
        </div>
      </div>
      {/* Sort order toggle button */}
      <button
        onClick={onToggleSort}
        style={{
          marginTop: '8px',
          padding: '4px 8px',
          fontSize: '10px',
          backgroundColor: sortReversed ? '#4ecdc4' : '#0f3460',
          color: sortReversed ? '#000' : '#aaa',
          border: '1px solid #1a1a2e',
          borderRadius: '4px',
          cursor: 'pointer',
          width: '100%',
        }}
        title={sortReversed
          ? 'Currently: Low expression on top. Click to show high expression on top.'
          : 'Currently: High expression on top. Click to show low expression on top.'}
      >
        {sortReversed ? '↓ Low on Top' : '↑ High on Top'}
      </button>
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
    bivariateData,
    selectedGenes,
    selectedGeneSetName,
    interactionMode,
    selectedCellIndices,
    setInteractionMode,
    setSelectedCellIndices,
    addToSelection,
    clearSelection,
    invertSelection,
    // cellSortOrder, sortCellsByExpression, resetCellOrder - now auto-applied
    displayPreferences,
    bivariateSortReversed,
    toggleBivariateSortOrder,
    drawnLines,
    addLine,
    drawTool,
    setDrawTool,
    setScanpyModalOpen,
    centerPanelView,
    setCenterPanelView,
    setEmbedding,
    comparisonCheckedColumn,
    comparisonCheckedCategories,
    setComparisonGroup1,
    setComparisonGroup2,
    setDiffExpModalOpen,
    setDiffExpResult,
    setDiffExpLoading,
    setMarkerGenesModalOpen,
    setMarkerGenesColumn,
    clearComparisonCategories,
    activeSlot,
    setActiveSlot,
    loadDatasetIntoSlot,
    datasets,
    layoutMode,
    setLayoutMode,
    quiltPhase,
    setQuiltPhase,
    quiltUndoDepth,
    setQuiltUndoDepth,
    setError,
  } = useStore()

  const schema = useSchema()
  const embedding = useEmbedding()
  const colorBy = useColorBy()
  const { selectEmbedding } = useDataActions()

  // Re-fetch expression data when transform setting changes
  useExpressionTransformEffect()
  useBivariateTransformEffect()

  // Auto-dismiss errors after 5 seconds
  useEffect(() => {
    if (!error) return
    const timer = setTimeout(() => setError(null), 5000)
    return () => clearTimeout(timer)
  }, [error, setError])

  // State for line naming dialog
  const [pendingLinePoints, setPendingLinePoints] = useState<[number, number][] | null>(null)
  const [pendingDrawType, setPendingDrawType] = useState<'pencil' | 'polygon' | 'segmented' | 'smooth_curve'>('pencil')
  const [newLineName, setNewLineName] = useState('')
  const [showDrawMenu, setShowDrawMenu] = useState(false)

  // State for export modal
  const [isExportModalOpen, setIsExportModalOpen] = useState(false)
  const [exportLoading, setExportLoading] = useState<string | null>(null)

  // State for compare button
  const [isCompareLoading, setIsCompareLoading] = useState(false)

  // State for load data modal
  const [loadSlot, setLoadSlot] = useState<DatasetSlot>('primary')
  const [isLoadModalOpen, setIsLoadModalOpen] = useState(false)
  const [loadFilePath, setLoadFilePath] = useState('')
  const [loadError, setLoadError] = useState<string | null>(null)
  const [loadLoading, setLoadLoading] = useState(false)

  // State for file browser
  const [browseEntries, setBrowseEntries] = useState<{ name: string; type: string; path: string; size?: number }[]>([])
  const [browseCurrent, setBrowseCurrent] = useState<string | null>(null)
  const [browseLoading, setBrowseLoading] = useState(false)
  const [browseShortcuts, setBrowseShortcuts] = useState<{ name: string; path: string }[]>([])
  const [recentFiles, setRecentFiles] = useState<string[]>(() => {
    try {
      return JSON.parse(localStorage.getItem('xcell_recentFiles') || '[]')
    } catch { return [] }
  })
  const lastBrowseDirRef = useRef<string | null>(localStorage.getItem('xcell_lastBrowseDir'))
  const [leftPanelCollapsed, setLeftPanelCollapsed] = useState(false)
  const [rightPanelCollapsed, setRightPanelCollapsed] = useState(false)

  const handleQuiltUndo = useCallback(async () => {
    if (!selectedEmbedding) return
    try {
      const response = await fetch(appendDataset(`/api/embedding/${selectedEmbedding}/undo`), {
        method: 'POST',
      })
      if (!response.ok) return
      const data = await response.json()
      setEmbedding(data)
      setQuiltUndoDepth(data.undo_depth ?? 0)
      clearSelection()
      setQuiltPhase('lasso')
    } catch (err) {
      console.error('Quilt undo failed:', err)
    }
  }, [selectedEmbedding, setEmbedding, setQuiltUndoDepth, clearSelection, setQuiltPhase])

  // Handle escape key to exit lasso/draw/quilt mode, and Ctrl/Cmd+Z for quilt undo
  useEffect(() => {
    const handleKeyDown = (e: KeyboardEvent) => {
      if ((e.metaKey || e.ctrlKey) && e.key === 'z' && interactionMode === 'quilt' && quiltUndoDepth > 0) {
        e.preventDefault()
        handleQuiltUndo()
        return
      }
      if (e.key === 'Escape') {
        if (interactionMode === 'quilt' && quiltPhase === 'transform') {
          // In quilt transform phase: clear selection, return to lasso phase
          clearSelection()
          setQuiltPhase('lasso')
          return
        }
        setInteractionMode('pan')
        setPendingLinePoints(null)
      }
    }
    window.addEventListener('keydown', handleKeyDown)
    return () => window.removeEventListener('keydown', handleKeyDown)
  }, [setInteractionMode, interactionMode, quiltPhase, clearSelection, setQuiltPhase, quiltUndoDepth, handleQuiltUndo])

  const handleSelectionComplete = useCallback(
    (indices: number[], additive: boolean) => {
      if (additive) {
        addToSelection(indices)
      } else {
        setSelectedCellIndices(indices)
      }
    },
    [setSelectedCellIndices, addToSelection]
  )

  const toggleLassoMode = useCallback(() => {
    setInteractionMode(interactionMode === 'lasso' ? 'pan' : 'lasso')
  }, [interactionMode, setInteractionMode])

  const toggleDrawMode = useCallback(() => {
    setInteractionMode(interactionMode === 'draw' ? 'pan' : 'draw')
    setShowDrawMenu(false)
  }, [interactionMode, setInteractionMode])

  // Close draw menu on outside click
  useEffect(() => {
    if (!showDrawMenu) return
    const close = () => setShowDrawMenu(false)
    document.addEventListener('click', close)
    return () => document.removeEventListener('click', close)
  }, [showDrawMenu])

  const toggleAdjustMode = useCallback(() => {
    setInteractionMode(interactionMode === 'adjust' ? 'pan' : 'adjust')
  }, [interactionMode, setInteractionMode])

  const toggleQuiltMode = useCallback(() => {
    setInteractionMode(interactionMode === 'quilt' ? 'pan' : 'quilt')
  }, [interactionMode, setInteractionMode])

  const handleTransformEmbedding = useCallback(async (opts: { rotation_degrees?: number; reflect_x?: boolean; reflect_y?: boolean }) => {
    if (!selectedEmbedding) return
    try {
      const response = await fetch(appendDataset(`/api/embedding/${selectedEmbedding}/transform`), {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(opts),
      })
      if (!response.ok) {
        const err = await response.json().catch(() => ({ detail: response.statusText }))
        throw new Error(err.detail || `HTTP ${response.status}`)
      }
      const data = await response.json()
      setEmbedding(data)
    } catch (err) {
      console.error('Transform embedding failed:', err)
    }
  }, [selectedEmbedding, setEmbedding])

  const handleRotateEmbedding = useCallback((rotationDegrees: number) => {
    handleTransformEmbedding({ rotation_degrees: rotationDegrees })
  }, [handleTransformEmbedding])

  const handleTransformEmbeddingSubset = useCallback(async (opts: {
    rotation_degrees?: number
    reflect_x?: boolean
    reflect_y?: boolean
    translate_x?: number
    translate_y?: number
    cell_indices: number[]
  }) => {
    if (!selectedEmbedding) return
    try {
      const response = await fetch(appendDataset(`/api/embedding/${selectedEmbedding}/transform`), {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(opts),
      })
      if (!response.ok) {
        const err = await response.json().catch(() => ({ detail: response.statusText }))
        throw new Error(err.detail || `HTTP ${response.status}`)
      }
      const data = await response.json()
      setEmbedding(data)
      setQuiltUndoDepth(data.undo_depth ?? 0)
    } catch (err) {
      console.error('Transform embedding subset failed:', err)
    }
  }, [selectedEmbedding, setEmbedding, setQuiltUndoDepth])

  const handleLineDrawn = useCallback((points: [number, number][]) => {
    setPendingLinePoints(points)
    setPendingDrawType(drawTool)
    setNewLineName(`Line ${drawnLines.length + 1}`)
    setInteractionMode('pan')
  }, [drawnLines.length, setInteractionMode, drawTool])

  const handleSaveLine = useCallback(() => {
    if (pendingLinePoints && newLineName.trim() && selectedEmbedding) {
      const closed = pendingDrawType === 'polygon'
      addLine(newLineName.trim(), pendingLinePoints, selectedEmbedding, pendingDrawType, closed)
      setPendingLinePoints(null)
      setNewLineName('')
    }
  }, [pendingLinePoints, newLineName, selectedEmbedding, addLine, pendingDrawType])

  const handleCancelLine = useCallback(() => {
    setPendingLinePoints(null)
    setNewLineName('')
  }, [])

  // Handle Compare button click (checkbox-based comparison)
  const handleCompare = useCallback(async () => {
    if (!comparisonCheckedColumn || comparisonCheckedCategories.size < 2) return

    const column = comparisonCheckedColumn
    const checkedGroups = [...comparisonCheckedCategories]

    setIsCompareLoading(true)
    try {
      // Fetch the column data to resolve cell indices
      const response = await fetch(appendDataset(`/api/obs/${encodeURIComponent(column)}`))
      if (!response.ok) throw new Error('Failed to fetch column data')
      const data = await response.json()

      // Build category → indices map for checked categories
      const categoryIndices: Record<string, number[]> = {}
      for (const cat of checkedGroups) {
        categoryIndices[cat] = []
      }

      const categories = data.categories || []
      if (data.dtype === 'category' && categories.length > 0) {
        data.values.forEach((val: number, idx: number) => {
          const catName = categories[val]
          if (catName !== undefined && categoryIndices[catName]) {
            categoryIndices[catName].push(idx)
          }
        })
      } else {
        data.values.forEach((val: string, idx: number) => {
          if (categoryIndices[val]) {
            categoryIndices[val].push(idx)
          }
        })
      }

      if (checkedGroups.length === 2) {
        // Pairwise diff exp
        const group1 = categoryIndices[checkedGroups[0]]
        const group2 = categoryIndices[checkedGroups[1]]
        setComparisonGroup1(group1, checkedGroups[0])
        setComparisonGroup2(group2, checkedGroups[1])
        setDiffExpLoading(true)
        setDiffExpModalOpen(true)
        try {
          const result = await runDiffExp(group1, group2, 25)
          setDiffExpResult(result)
        } finally {
          setDiffExpLoading(false)
        }
      } else {
        // 3+ groups → marker gene analysis
        setMarkerGenesColumn(column)
        setMarkerGenesModalOpen(true)
      }
      clearComparisonCategories()
    } catch (err) {
      alert(`Comparison failed: ${(err as Error).message}`)
    } finally {
      setIsCompareLoading(false)
    }
  }, [comparisonCheckedColumn, comparisonCheckedCategories, setComparisonGroup1, setComparisonGroup2, setDiffExpLoading, setDiffExpModalOpen, setDiffExpResult, setMarkerGenesColumn, setMarkerGenesModalOpen, clearComparisonCategories])

  const geneSetCategories = useStore((state) => state.geneSetCategories)

  // Collect all gene sets from all categories (direct + inside folders)
  const allGeneSets = useMemo(() => {
    const result: { name: string; genes: string[]; category: string; folder?: string }[] = []
    for (const cat of Object.values(geneSetCategories)) {
      for (const gs of cat.geneSets) {
        result.push({ name: gs.name, genes: gs.genes, category: cat.name })
      }
      for (const folder of cat.folders) {
        for (const gs of folder.geneSets) {
          result.push({ name: gs.name, genes: gs.genes, category: cat.name, folder: folder.name })
        }
      }
    }
    return result
  }, [geneSetCategories])

  const handleExportH5ad = useCallback(async () => {
    setExportLoading('h5ad')
    try {
      // First, send drawn lines to backend so they can be included in export
      if (drawnLines.length > 0) {
        const linesPayload = drawnLines.map((line) => ({
          name: line.name,
          embeddingName: line.embeddingName,
          points: line.points,
          smoothedPoints: line.smoothedPoints,
        }))
        const linesResponse = await fetch(appendDataset('/api/lines'), {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ lines: linesPayload }),
        })
        if (!linesResponse.ok) {
          console.warn('Failed to send lines to backend, continuing with export')
        }
      }

      // Now export h5ad
      const response = await fetch(appendDataset('/api/export/h5ad'))
      if (!response.ok) {
        const error = await response.json().catch(() => ({ detail: response.statusText }))
        throw new Error(error.detail || `HTTP ${response.status}`)
      }
      const blob = await response.blob()
      const url = URL.createObjectURL(blob)
      const a = document.createElement('a')
      a.href = url
      a.download = 'xcell_export.h5ad'
      document.body.appendChild(a)
      a.click()
      document.body.removeChild(a)
      URL.revokeObjectURL(url)
      setIsExportModalOpen(false)
    } catch (err) {
      alert(`Failed to export h5ad: ${(err as Error).message}`)
    } finally {
      setExportLoading(null)
    }
  }, [drawnLines])

  const handleExportMetadata = useCallback(async () => {
    setExportLoading('metadata')
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
      setIsExportModalOpen(false)
    } catch (err) {
      alert(`Failed to export metadata: ${(err as Error).message}`)
    } finally {
      setExportLoading(null)
    }
  }, [])

  const handleExportGeneSets = useCallback(() => {
    setExportLoading('genesets')
    try {
      if (allGeneSets.length === 0) {
        alert('No gene sets to export')
        return
      }
      // Export as JSON
      const data = JSON.stringify(allGeneSets, null, 2)
      const blob = new Blob([data], { type: 'application/json' })
      const url = URL.createObjectURL(blob)
      const a = document.createElement('a')
      a.href = url
      a.download = 'gene_sets.json'
      document.body.appendChild(a)
      a.click()
      document.body.removeChild(a)
      URL.revokeObjectURL(url)
      setIsExportModalOpen(false)
    } catch (err) {
      alert(`Failed to export gene sets: ${(err as Error).message}`)
    } finally {
      setExportLoading(null)
    }
  }, [allGeneSets])

  const browseDirectory = useCallback(async (dirPath?: string) => {
    setBrowseLoading(true)
    try {
      const target = dirPath ?? lastBrowseDirRef.current
      const url = target ? `/api/browse?path=${encodeURIComponent(target)}` : '/api/browse'
      const response = await fetch(url)
      if (!response.ok) {
        const err = await response.json().catch(() => ({ detail: response.statusText }))
        throw new Error(err.detail || `HTTP ${response.status}`)
      }
      const data = await response.json()
      setBrowseEntries(data.entries)
      setBrowseCurrent(data.current)
      if (data.shortcuts) setBrowseShortcuts(data.shortcuts)
      lastBrowseDirRef.current = data.current
      localStorage.setItem('xcell_lastBrowseDir', data.current)
    } catch (err) {
      setLoadError((err as Error).message)
    } finally {
      setBrowseLoading(false)
    }
  }, [])

  const handleLoadDataset = useCallback(async () => {
    if (!loadFilePath.trim()) return
    setLoadLoading(true)
    setLoadError(null)
    try {
      // Load into the selected slot (backend reads slot from request body)
      const response = await fetch('/api/load', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ file_path: loadFilePath.trim(), slot: loadSlot }),
      })
      if (!response.ok) {
        const err = await response.json().catch(() => ({ detail: response.statusText }))
        throw new Error(err.detail || `HTTP ${response.status}`)
      }
      // Fetch schema for the loaded slot
      const schemaUrl = loadSlot === 'primary'
        ? '/api/schema'
        : `/api/schema?dataset=${loadSlot}`
      const schemaData = await fetch(schemaUrl).then(r => {
        if (!r.ok) throw new Error(`Failed to fetch schema: ${r.statusText}`)
        return r.json()
      })
      // Update store
      loadDatasetIntoSlot(loadSlot, schemaData)
      // Switch to the loaded slot
      if (loadSlot !== activeSlot) {
        setActiveSlot(loadSlot)
      }
      // Track recently loaded files
      const filePath = loadFilePath.trim()
      setRecentFiles(prev => {
        const updated = [filePath, ...prev.filter(f => f !== filePath)].slice(0, 5)
        localStorage.setItem('xcell_recentFiles', JSON.stringify(updated))
        return updated
      })
      setIsLoadModalOpen(false)
    } catch (err) {
      setLoadError((err as Error).message)
    } finally {
      setLoadLoading(false)
    }
  }, [loadFilePath, loadSlot, activeSlot, loadDatasetIntoSlot, setActiveSlot])

  return (
    <div style={styles.container}>
      <header style={styles.header}>
        <div style={styles.titleGroup}>
          <h1 style={styles.title}>xcell</h1>
          {schema && (
            <span style={styles.statsInline}>
              {schema.n_cells.toLocaleString()} cells · {schema.n_genes.toLocaleString()} genes
            </span>
          )}
        </div>

        <div style={styles.controls}>
          {datasets.secondary.schema && (
            <>
              <select
                value={activeSlot}
                onChange={(e) => setActiveSlot(e.target.value as DatasetSlot)}
                style={{ ...styles.embeddingSelect, fontSize: '12px' }}
                title="Switch active dataset"
              >
                <option value="primary">
                  Primary ({datasets.primary.schema?.n_cells.toLocaleString() ?? '\u2014'} cells)
                </option>
                <option value="secondary">
                  Secondary ({datasets.secondary.schema?.n_cells.toLocaleString() ?? '\u2014'} cells)
                </option>
              </select>
              <button
                style={{
                  ...styles.toolButton,
                  ...(layoutMode === 'dual' ? styles.toolButtonActive : {}),
                }}
                onClick={() => setLayoutMode(layoutMode === 'dual' ? 'single' : 'dual')}
                title={layoutMode === 'dual' ? 'Switch to single view' : 'Show both datasets side by side'}
              >
                Split
              </button>
            </>
          )}
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

              <div style={{ position: 'relative', display: 'inline-block' }}>
                <button
                  style={{
                    ...styles.toolButton,
                    ...(interactionMode === 'draw' ? { ...styles.toolButtonActive, backgroundColor: '#4ecdc4' } : {}),
                  }}
                  onClick={toggleDrawMode}
                  title="Draw shapes for trajectory analysis (Escape to exit)"
                >
                  <span>&#9998;</span> Draw
                </button>
                <button
                  style={{
                    padding: '4px 2px',
                    fontSize: '8px',
                    backgroundColor: interactionMode === 'draw' ? '#4ecdc4' : '#0f3460',
                    color: interactionMode === 'draw' ? '#000' : '#aaa',
                    border: 'none',
                    borderLeft: '1px solid #1a1a2e',
                    cursor: 'pointer',
                    borderRadius: '0 4px 4px 0',
                    marginLeft: '-5px',
                  }}
                  onClick={(e) => { e.stopPropagation(); setShowDrawMenu(!showDrawMenu) }}
                  title="Select draw tool"
                >
                  {'\u25BC'}
                </button>
                {showDrawMenu && (
                  <div
                    style={{
                      position: 'absolute',
                      top: '100%',
                      left: 0,
                      backgroundColor: '#16213e',
                      border: '1px solid #0f3460',
                      borderRadius: '4px',
                      zIndex: 100,
                      minWidth: '160px',
                      boxShadow: '0 4px 12px rgba(0,0,0,0.4)',
                    }}
                  >
                    {([
                      ['pencil', '\u270F', 'Pencil', 'Freehand drawing'],
                      ['polygon', '\u2B21', 'Polygon', 'Click vertices, dbl-click to close'],
                      ['segmented', '\u2571', 'Segmented Line', 'Click points, dbl-click to finish'],
                      ['smooth_curve', '\u223F', 'Smooth Curve', 'Click control pts, dbl-click to finish'],
                    ] as const).map(([tool, icon, label, desc]) => (
                      <div
                        key={tool}
                        onClick={() => {
                          setDrawTool(tool)
                          setShowDrawMenu(false)
                          if (interactionMode !== 'draw') setInteractionMode('draw')
                        }}
                        style={{
                          padding: '8px 12px',
                          cursor: 'pointer',
                          fontSize: '13px',
                          color: drawTool === tool ? '#4ecdc4' : '#ccc',
                          backgroundColor: drawTool === tool ? '#0f3460' : 'transparent',
                          display: 'flex',
                          alignItems: 'center',
                          gap: '8px',
                        }}
                      >
                        <span style={{ fontSize: '16px', width: '20px', textAlign: 'center' }}>{icon}</span>
                        <div>
                          <div>{label}</div>
                          <div style={{ fontSize: '10px', color: '#888' }}>{desc}</div>
                        </div>
                      </div>
                    ))}
                  </div>
                )}
              </div>

              <button
                style={{
                  ...styles.toolButton,
                  ...(interactionMode === 'adjust' ? { ...styles.toolButtonActive, backgroundColor: '#ffa500' } : {}),
                }}
                onClick={toggleAdjustMode}
                title="Adjust embedding orientation: flip or shift+drag to rotate (Escape to exit)"
              >
                <span>&#8634;</span> Adjust
              </button>

              <button
                style={{
                  ...styles.toolButton,
                  ...(interactionMode === 'quilt' ? { ...styles.toolButtonActive, backgroundColor: '#9b59b6' } : {}),
                }}
                onClick={toggleQuiltMode}
                title="Quilt mode: lasso cells then drag/rotate/flip to reposition (Escape to exit)"
              >
                <span>&#9638;</span> Quilt
              </button>

              <button
                style={styles.toolButton}
                onClick={() => setScanpyModalOpen(true)}
                title="Run scanpy analysis functions"
              >
                Analyze
              </button>

              <button
                style={{
                  ...styles.toolButton,
                  ...(comparisonCheckedCategories.size >= 2
                    ? { backgroundColor: '#e94560', color: '#fff', borderColor: '#e94560' }
                    : {}),
                  opacity: comparisonCheckedCategories.size < 2 ? 0.5 : 1,
                  position: 'relative' as const,
                }}
                onClick={handleCompare}
                disabled={comparisonCheckedCategories.size < 2 || isCompareLoading}
                title={
                  comparisonCheckedCategories.size < 2
                    ? 'Check 2+ categories in the cell panel to compare'
                    : comparisonCheckedCategories.size === 2
                      ? 'Run pairwise differential expression'
                      : `Run marker gene analysis (${comparisonCheckedCategories.size} groups)`
                }
              >
                {isCompareLoading ? 'Running...' : 'Compare'}
                {comparisonCheckedCategories.size >= 2 && (
                  <span
                    style={{
                      position: 'absolute',
                      top: '-6px',
                      right: '-6px',
                      backgroundColor: '#ffd700',
                      color: '#000',
                      fontSize: '10px',
                      fontWeight: 700,
                      borderRadius: '50%',
                      width: '16px',
                      height: '16px',
                      display: 'flex',
                      alignItems: 'center',
                      justifyContent: 'center',
                    }}
                  >
                    {comparisonCheckedCategories.size}
                  </span>
                )}
              </button>

              <DisplaySettings />

              <button
                style={styles.exportButton}
                onClick={() => { setIsLoadModalOpen(true); setLoadError(null); setLoadFilePath(''); browseDirectory() }}
                title="Load a different dataset file"
              >
                Load
              </button>

              <button
                style={styles.exportButton}
                onClick={() => setIsExportModalOpen(true)}
                title="Export data"
              >
                Export
              </button>

              {selectedCellIndices.length > 0 && (
                <div style={styles.selectionInfo}>
                  {selectedCellIndices.length.toLocaleString()} cells selected
                  <button style={styles.clearButton} onClick={invertSelection}>
                    Invert
                  </button>
                  <button style={styles.clearButton} onClick={clearSelection}>
                    Clear
                  </button>
                </div>
              )}
            </>
          )}
        </div>
        <div style={styles.logoGroup}>
          <a href="https://cahanlab.org/" target="_blank" rel="noopener noreferrer">
            <img src="/logoGlow.png" alt="CahanLab" style={{ height: '32px' }} />
          </a>
        </div>
      </header>

      <div style={styles.body}>
        {leftPanelCollapsed ? (
          <div
            style={{
              width: '28px',
              flexShrink: 0,
              backgroundColor: '#16213e',
              borderRight: '1px solid #0f3460',
              display: 'flex',
              flexDirection: 'column',
              alignItems: 'center',
              cursor: 'pointer',
              userSelect: 'none',
            }}
            onClick={() => setLeftPanelCollapsed(false)}
            title="Expand Cells panel"
          >
            <span style={{ fontSize: '11px', color: '#888', marginTop: '10px' }}>{'\u25B6'}</span>
            <span style={{
              writingMode: 'vertical-rl',
              fontSize: '11px',
              color: '#e94560',
              fontWeight: 600,
              marginTop: '8px',
              letterSpacing: '1px',
            }}>
              Cells
            </span>
          </div>
        ) : (
          <div style={{
            display: 'flex',
            flexDirection: 'column',
            height: '100%',
            overflow: 'hidden',
            flexShrink: 0,
            position: 'relative',
          }}>
            <button
              onClick={() => setLeftPanelCollapsed(true)}
              title="Collapse Cells panel"
              style={{
                position: 'absolute',
                top: '10px',
                right: '8px',
                zIndex: 10,
                background: 'none',
                border: 'none',
                color: '#888',
                fontSize: '10px',
                cursor: 'pointer',
                padding: '2px 4px',
              }}
            >
              {'\u25C0'}
            </button>
            <CellPanel />
            <ShapeManager />
          </div>
        )}

        <main style={styles.main}>
          {/* Tab bar (rollback: remove this block and restore plain <main> content) */}
          <div style={styles.tabBar}>
            <button
              style={{ ...styles.tab, ...(centerPanelView === 'scatter' ? styles.tabActive : {}) }}
              onClick={() => setCenterPanelView('scatter')}
            >
              Scatter Plot
            </button>
            <button
              style={{ ...styles.tab, ...(centerPanelView === 'heatmap' ? styles.tabActive : {}) }}
              onClick={() => setCenterPanelView('heatmap')}
            >
              Heatmap
            </button>
          </div>

          <div style={styles.vizContent}>
            {centerPanelView === 'scatter' && (
              <>
                {isLoading && <div style={styles.loading}>Loading...</div>}

                {error && (
                  <div style={styles.error} onClick={() => setError(null)} title="Click to dismiss">
                    <strong>{MESSAGES.errorPrefix}</strong> {error}
                    <span style={{ marginLeft: '12px', cursor: 'pointer', opacity: 0.7, fontSize: '14px' }}>&times;</span>
                  </div>
                )}

                {layoutMode === 'dual' && datasets.secondary.schema ? (
                  /* Dual scatter layout — side by side */
                  <div style={{ display: 'flex', position: 'absolute', top: 0, left: 0, right: 0, bottom: 0, overflow: 'hidden' }}>
                    {/* Primary plot */}
                    <div
                      style={{
                        flex: 1,
                        position: 'relative',
                        overflow: 'hidden',
                        borderRight: '1px solid #0f3460',
                        outline: activeSlot === 'primary' ? '2px solid #e94560' : '2px solid transparent',
                        outlineOffset: '-2px',
                      }}
                      onPointerDown={() => { if (activeSlot !== 'primary') setActiveSlot('primary') }}
                    >
                      {datasets.primary.embedding ? (
                        <>
                          <ScatterPlot
                            slot="primary"
                            embedding={datasets.primary.embedding}
                            colorBy={datasets.primary.colorBy}
                            expressionData={datasets.primary.expressionData}
                            bivariateData={datasets.primary.bivariateData}
                            colorMode={datasets.primary.colorMode}
                            interactionMode={interactionMode}
                            selectedCellIndices={datasets.primary.selectedCellIndices}
                            onSelectionComplete={handleSelectionComplete}
                            onLineDrawn={handleLineDrawn}
                            onTransformEmbedding={handleRotateEmbedding}
                            onTransformEmbeddingSubset={handleTransformEmbeddingSubset}
                          />
                          {/* Per-plot embedding selector */}
                          {datasets.primary.schema && datasets.primary.schema.embeddings.length > 1 && (
                            <div style={{ ...styles.embeddingSelector }}>
                              <span style={styles.embeddingLabel}>Embedding:</span>
                              <select
                                style={styles.embeddingSelect}
                                value={datasets.primary.selectedEmbedding || ''}
                                onChange={(e) => {
                                  if (activeSlot !== 'primary') setActiveSlot('primary')
                                  selectEmbedding(e.target.value)
                                }}
                              >
                                {datasets.primary.schema.embeddings.map((emb) => (
                                  <option key={emb} value={emb}>{emb}</option>
                                ))}
                              </select>
                            </div>
                          )}
                          {/* Per-plot legend */}
                          {datasets.primary.colorMode === 'metadata' && datasets.primary.colorBy?.dtype === 'category' && (
                            <CategoryLegend colorBy={datasets.primary.colorBy} />
                          )}
                          {datasets.primary.colorMode === 'metadata' && datasets.primary.colorBy?.dtype === 'numeric' && (
                            <ContinuousLegend
                              name={datasets.primary.colorBy.name}
                              min={Math.min(...(datasets.primary.colorBy.values.filter((v) => v !== null) as number[]))}
                              max={Math.max(...(datasets.primary.colorBy.values.filter((v) => v !== null) as number[]))}
                            />
                          )}
                          {datasets.primary.colorMode === 'expression' && datasets.primary.expressionData && (
                            <div style={styles.expressionLegend}>
                              <div style={styles.legendTitle}>
                                {datasets.primary.selectedGenes.length === 1 ? datasets.primary.selectedGenes[0] : datasets.primary.selectedGeneSetName ? `${datasets.primary.selectedGeneSetName} (${datasets.primary.selectedGenes.length})` : `${datasets.primary.selectedGenes.length} genes`}
                                {datasets.primary.expressionData.transform === 'log1p' && (
                                  <span style={{ fontSize: '9px', color: '#4ecdc4', marginLeft: '6px' }}>(log1p)</span>
                                )}
                              </div>
                              <div style={{ ...styles.colorBar, background: COLOR_SCALE_GRADIENTS[datasets.primary.displayPreferences.colorScale] || COLOR_SCALE_GRADIENTS.viridis }} />
                              <div style={styles.colorBarLabels}>
                                <span>{datasets.primary.expressionData.min.toFixed(2)}</span>
                                <span>{datasets.primary.expressionData.max.toFixed(2)}</span>
                              </div>
                            </div>
                          )}
                          {datasets.primary.colorMode === 'bivariate' && datasets.primary.bivariateData && (
                            <BivariateLegend
                              bivariateData={datasets.primary.bivariateData}
                              colormap={datasets.primary.displayPreferences.bivariateColormap}
                              sortReversed={datasets.primary.bivariateSortReversed}
                              onToggleSort={toggleBivariateSortOrder}
                            />
                          )}
                        </>
                      ) : (
                        <div style={{ ...styles.loading, position: 'absolute' }}>No embedding loaded</div>
                      )}
                      <div style={{ position: 'absolute', top: 6, left: 8, fontSize: '11px', color: '#888', pointerEvents: 'none' }}>
                        Primary
                      </div>
                    </div>

                    {/* Secondary plot */}
                    <div
                      style={{
                        flex: 1,
                        position: 'relative',
                        overflow: 'hidden',
                        outline: activeSlot === 'secondary' ? '2px solid #e94560' : '2px solid transparent',
                        outlineOffset: '-2px',
                      }}
                      onPointerDown={() => { if (activeSlot !== 'secondary') setActiveSlot('secondary') }}
                    >
                      {datasets.secondary.embedding ? (
                        <>
                          <ScatterPlot
                            slot="secondary"
                            embedding={datasets.secondary.embedding}
                            colorBy={datasets.secondary.colorBy}
                            expressionData={datasets.secondary.expressionData}
                            bivariateData={datasets.secondary.bivariateData}
                            colorMode={datasets.secondary.colorMode}
                            interactionMode={interactionMode}
                            selectedCellIndices={datasets.secondary.selectedCellIndices}
                            onSelectionComplete={handleSelectionComplete}
                            onLineDrawn={handleLineDrawn}
                            onTransformEmbedding={handleRotateEmbedding}
                            onTransformEmbeddingSubset={handleTransformEmbeddingSubset}
                          />
                          {/* Per-plot embedding selector */}
                          {datasets.secondary.schema && datasets.secondary.schema.embeddings.length > 1 && (
                            <div style={{ ...styles.embeddingSelector }}>
                              <span style={styles.embeddingLabel}>Embedding:</span>
                              <select
                                style={styles.embeddingSelect}
                                value={datasets.secondary.selectedEmbedding || ''}
                                onChange={(e) => {
                                  if (activeSlot !== 'secondary') setActiveSlot('secondary')
                                  selectEmbedding(e.target.value)
                                }}
                              >
                                {datasets.secondary.schema.embeddings.map((emb) => (
                                  <option key={emb} value={emb}>{emb}</option>
                                ))}
                              </select>
                            </div>
                          )}
                          {/* Per-plot legend */}
                          {datasets.secondary.colorMode === 'metadata' && datasets.secondary.colorBy?.dtype === 'category' && (
                            <CategoryLegend colorBy={datasets.secondary.colorBy} />
                          )}
                          {datasets.secondary.colorMode === 'metadata' && datasets.secondary.colorBy?.dtype === 'numeric' && (
                            <ContinuousLegend
                              name={datasets.secondary.colorBy.name}
                              min={Math.min(...(datasets.secondary.colorBy.values.filter((v) => v !== null) as number[]))}
                              max={Math.max(...(datasets.secondary.colorBy.values.filter((v) => v !== null) as number[]))}
                            />
                          )}
                          {datasets.secondary.colorMode === 'expression' && datasets.secondary.expressionData && (
                            <div style={styles.expressionLegend}>
                              <div style={styles.legendTitle}>
                                {datasets.secondary.selectedGenes.length === 1 ? datasets.secondary.selectedGenes[0] : datasets.secondary.selectedGeneSetName ? `${datasets.secondary.selectedGeneSetName} (${datasets.secondary.selectedGenes.length})` : `${datasets.secondary.selectedGenes.length} genes`}
                                {datasets.secondary.expressionData.transform === 'log1p' && (
                                  <span style={{ fontSize: '9px', color: '#4ecdc4', marginLeft: '6px' }}>(log1p)</span>
                                )}
                              </div>
                              <div style={{ ...styles.colorBar, background: COLOR_SCALE_GRADIENTS[datasets.secondary.displayPreferences.colorScale] || COLOR_SCALE_GRADIENTS.viridis }} />
                              <div style={styles.colorBarLabels}>
                                <span>{datasets.secondary.expressionData.min.toFixed(2)}</span>
                                <span>{datasets.secondary.expressionData.max.toFixed(2)}</span>
                              </div>
                            </div>
                          )}
                          {datasets.secondary.colorMode === 'bivariate' && datasets.secondary.bivariateData && (
                            <BivariateLegend
                              bivariateData={datasets.secondary.bivariateData}
                              colormap={datasets.secondary.displayPreferences.bivariateColormap}
                              sortReversed={datasets.secondary.bivariateSortReversed}
                              onToggleSort={toggleBivariateSortOrder}
                            />
                          )}
                        </>
                      ) : (
                        <div style={{ ...styles.loading, position: 'absolute' }}>No embedding loaded</div>
                      )}
                      <div style={{ position: 'absolute', top: 6, left: 8, fontSize: '11px', color: '#888', pointerEvents: 'none' }}>
                        Secondary
                      </div>
                    </div>
                  </div>
                ) : (
                  /* Single scatter layout — existing behavior */
                  <>
                    {embedding && (
                      <>
                        <ScatterPlot
                          embedding={embedding}
                          colorBy={colorBy}
                          expressionData={expressionData}
                          bivariateData={bivariateData}
                          colorMode={colorMode}
                          interactionMode={interactionMode}
                          selectedCellIndices={selectedCellIndices}
                          onSelectionComplete={handleSelectionComplete}
                          onLineDrawn={handleLineDrawn}
                          onTransformEmbedding={handleRotateEmbedding}
                          onTransformEmbeddingSubset={handleTransformEmbeddingSubset}
                        />

                        {/* Embedding selector - bottom left */}
                        {schema && (schema.embeddings.length > 1 || interactionMode === 'adjust' || interactionMode === 'quilt') && (
                          <div style={{ ...styles.embeddingSelector, flexDirection: 'column' as const, alignItems: 'flex-start', gap: '6px' }}>
                            {schema.embeddings.length > 1 && (
                              <div style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
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
                            {interactionMode === 'adjust' && (
                              <div style={{ display: 'flex', alignItems: 'center', gap: '6px' }}>
                                <button
                                  style={{ padding: '4px 8px', fontSize: '11px', backgroundColor: '#0f3460', color: '#aaa', border: '1px solid #1a1a2e', borderRadius: '4px', cursor: 'pointer' }}
                                  onClick={() => handleTransformEmbedding({ reflect_y: true })}
                                  title="Mirror x-coordinates (reflect about y-axis)"
                                >
                                  &#8596; Flip X
                                </button>
                                <button
                                  style={{ padding: '4px 8px', fontSize: '11px', backgroundColor: '#0f3460', color: '#aaa', border: '1px solid #1a1a2e', borderRadius: '4px', cursor: 'pointer' }}
                                  onClick={() => handleTransformEmbedding({ reflect_x: true })}
                                  title="Mirror y-coordinates (reflect about x-axis)"
                                >
                                  &#8597; Flip Y
                                </button>
                                <span style={{ fontSize: '10px', color: '#666' }}>Shift+drag to rotate</span>
                              </div>
                            )}
                            {interactionMode === 'quilt' && quiltPhase === 'transform' && selectedCellIndices.length > 0 && (
                              <div style={{ display: 'flex', alignItems: 'center', gap: '6px' }}>
                                <button
                                  style={{ padding: '4px 8px', fontSize: '11px', backgroundColor: '#0f3460', color: '#aaa', border: '1px solid #1a1a2e', borderRadius: '4px', cursor: 'pointer' }}
                                  onClick={() => handleTransformEmbeddingSubset({ reflect_y: true, cell_indices: selectedCellIndices })}
                                  title="Flip selected cells horizontally"
                                >
                                  &#8596; Flip X
                                </button>
                                <button
                                  style={{ padding: '4px 8px', fontSize: '11px', backgroundColor: '#0f3460', color: '#aaa', border: '1px solid #1a1a2e', borderRadius: '4px', cursor: 'pointer' }}
                                  onClick={() => handleTransformEmbeddingSubset({ reflect_x: true, cell_indices: selectedCellIndices })}
                                  title="Flip selected cells vertically"
                                >
                                  &#8597; Flip Y
                                </button>
                                <span style={{ fontSize: '10px', color: '#9b59b6' }}>Drag to move, Shift+drag to rotate</span>
                              </div>
                            )}
                            {interactionMode === 'quilt' && quiltUndoDepth > 0 && (
                              <div style={{ display: 'flex', alignItems: 'center', gap: '6px' }}>
                                <button
                                  style={{ padding: '4px 8px', fontSize: '11px', backgroundColor: '#0f3460', color: '#aaa', border: '1px solid #1a1a2e', borderRadius: '4px', cursor: 'pointer' }}
                                  onClick={handleQuiltUndo}
                                  title={`Undo last quilt transform (${quiltUndoDepth} remaining) — Ctrl/Cmd+Z`}
                                >
                                  &#8617; Undo
                                </button>
                              </div>
                            )}
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
                              {selectedGenes.length === 1 ? selectedGenes[0] : selectedGeneSetName ? `${selectedGeneSetName} (${selectedGenes.length})` : `${selectedGenes.length} genes`}
                              {expressionData.transform === 'log1p' && (
                                <span style={{ fontSize: '9px', color: '#4ecdc4', marginLeft: '6px' }}>
                                  (log1p)
                                </span>
                              )}
                            </div>
                            <div style={{ ...styles.colorBar, background: COLOR_SCALE_GRADIENTS[displayPreferences.colorScale] || COLOR_SCALE_GRADIENTS.viridis }} />
                            <div style={styles.colorBarLabels}>
                              <span>{expressionData.min.toFixed(2)}</span>
                              <span>{expressionData.max.toFixed(2)}</span>
                            </div>
                          </div>
                        )}
                        {colorMode === 'bivariate' && bivariateData && (
                          <BivariateLegend
                            bivariateData={bivariateData}
                            colormap={displayPreferences.bivariateColormap}
                            sortReversed={bivariateSortReversed}
                            onToggleSort={toggleBivariateSortOrder}
                          />
                        )}
                      </>
                    )}

                    {!embedding && !isLoading && !error && (
                      <div style={styles.loading}>
                        {schema ? MESSAGES.noEmbedding : MESSAGES.noDataLoaded}
                      </div>
                    )}
                  </>
                )}
              </>
            )}

            {centerPanelView === 'heatmap' && (
              <HeatmapView />
            )}
          </div>
        </main>

        {rightPanelCollapsed ? (
          <div
            style={{
              width: '28px',
              flexShrink: 0,
              backgroundColor: '#16213e',
              borderLeft: '1px solid #0f3460',
              display: 'flex',
              flexDirection: 'column',
              alignItems: 'center',
              cursor: 'pointer',
              userSelect: 'none',
            }}
            onClick={() => setRightPanelCollapsed(false)}
            title="Expand Genes panel"
          >
            <span style={{ fontSize: '11px', color: '#888', marginTop: '10px' }}>{'\u25C0'}</span>
            <span style={{
              writingMode: 'vertical-rl',
              fontSize: '11px',
              color: '#e94560',
              fontWeight: 600,
              marginTop: '8px',
              letterSpacing: '1px',
            }}>
              Genes
            </span>
          </div>
        ) : (
          <div style={{ position: 'relative', height: '100%', flexShrink: 0 }}>
            <button
              onClick={() => setRightPanelCollapsed(true)}
              title="Collapse Genes panel"
              style={{
                position: 'absolute',
                top: '10px',
                left: '8px',
                zIndex: 10,
                background: 'none',
                border: 'none',
                color: '#888',
                fontSize: '10px',
                cursor: 'pointer',
                padding: '2px 4px',
              }}
            >
              {'\u25B6'}
            </button>
            <GenePanel />
          </div>
        )}
      </div>

      <DiffExpModal />
      <LineAssociationModal />
      <ScanpyModal />
      <MarkerGenesModal />

      {/* Export modal */}
      {isExportModalOpen && (
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
          onClick={() => setIsExportModalOpen(false)}
        >
          <div
            style={{
              backgroundColor: '#16213e',
              border: '1px solid #0f3460',
              borderRadius: '8px',
              padding: '24px',
              minWidth: '360px',
              maxWidth: '400px',
            }}
            onClick={(e) => e.stopPropagation()}
          >
            <div style={{ fontSize: '16px', fontWeight: 600, color: '#e94560', marginBottom: '20px' }}>
              Export Data
            </div>

            <div style={{ display: 'flex', flexDirection: 'column', gap: '12px' }}>
              {/* Export H5AD */}
              <button
                onClick={handleExportH5ad}
                disabled={exportLoading !== null}
                style={{
                  padding: '12px 16px',
                  fontSize: '14px',
                  backgroundColor: '#0f3460',
                  color: '#eee',
                  border: '1px solid #1a1a2e',
                  borderRadius: '6px',
                  cursor: exportLoading !== null ? 'wait' : 'pointer',
                  textAlign: 'left',
                  opacity: exportLoading !== null && exportLoading !== 'h5ad' ? 0.5 : 1,
                }}
              >
                <div style={{ fontWeight: 500, marginBottom: '4px' }}>
                  {exportLoading === 'h5ad' ? 'Exporting...' : 'AnnData (.h5ad)'}
                </div>
                <div style={{ fontSize: '12px', color: '#888' }}>
                  Full dataset with any new annotation columns
                </div>
              </button>

              {/* Export Cell Metadata */}
              <button
                onClick={handleExportMetadata}
                disabled={exportLoading !== null}
                style={{
                  padding: '12px 16px',
                  fontSize: '14px',
                  backgroundColor: '#0f3460',
                  color: '#eee',
                  border: '1px solid #1a1a2e',
                  borderRadius: '6px',
                  cursor: exportLoading !== null ? 'wait' : 'pointer',
                  textAlign: 'left',
                  opacity: exportLoading !== null && exportLoading !== 'metadata' ? 0.5 : 1,
                }}
              >
                <div style={{ fontWeight: 500, marginBottom: '4px' }}>
                  {exportLoading === 'metadata' ? 'Exporting...' : 'Cell Metadata (.tsv)'}
                </div>
                <div style={{ fontSize: '12px', color: '#888' }}>
                  All cell annotations as tab-separated values
                </div>
              </button>

              {/* Export Gene Sets */}
              <button
                onClick={handleExportGeneSets}
                disabled={exportLoading !== null || allGeneSets.length === 0}
                style={{
                  padding: '12px 16px',
                  fontSize: '14px',
                  backgroundColor: '#0f3460',
                  color: allGeneSets.length === 0 ? '#666' : '#eee',
                  border: '1px solid #1a1a2e',
                  borderRadius: '6px',
                  cursor: exportLoading !== null || allGeneSets.length === 0 ? 'not-allowed' : 'pointer',
                  textAlign: 'left',
                  opacity: exportLoading !== null && exportLoading !== 'genesets' ? 0.5 : 1,
                }}
              >
                <div style={{ fontWeight: 500, marginBottom: '4px' }}>
                  {exportLoading === 'genesets' ? 'Exporting...' : 'Gene Sets (.json)'}
                </div>
                <div style={{ fontSize: '12px', color: '#666' }}>
                  {allGeneSets.length === 0
                    ? 'No gene sets defined'
                    : `${allGeneSets.length} gene set${allGeneSets.length === 1 ? '' : 's'}`
                  }
                </div>
              </button>
            </div>

            <div style={{ marginTop: '20px', display: 'flex', justifyContent: 'flex-end' }}>
              <button
                onClick={() => setIsExportModalOpen(false)}
                disabled={exportLoading !== null}
                style={{
                  padding: '8px 16px',
                  fontSize: '13px',
                  backgroundColor: 'transparent',
                  color: '#aaa',
                  border: '1px solid #0f3460',
                  borderRadius: '4px',
                  cursor: exportLoading !== null ? 'wait' : 'pointer',
                }}
              >
                Cancel
              </button>
            </div>
          </div>
        </div>
      )}

      {/* Load data modal */}
      {isLoadModalOpen && (
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
          onClick={() => !loadLoading && setIsLoadModalOpen(false)}
        >
          <div
            style={{
              backgroundColor: '#16213e',
              border: '1px solid #0f3460',
              borderRadius: '8px',
              padding: '20px',
              width: '640px',
              maxHeight: '80vh',
              display: 'flex',
              flexDirection: 'column',
            }}
            onClick={(e) => e.stopPropagation()}
          >
            {/* Header row: title + slot selector */}
            <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', marginBottom: '12px' }}>
              <div style={{ fontSize: '16px', fontWeight: 600, color: '#e94560' }}>
                Load Dataset
              </div>
              <div style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
                <span style={{ fontSize: '12px', color: '#888' }}>Load into:</span>
                <select
                  value={loadSlot}
                  onChange={(e) => setLoadSlot(e.target.value as DatasetSlot)}
                  style={styles.embeddingSelect}
                >
                  <option value="primary">Primary</option>
                  <option value="secondary">Secondary</option>
                </select>
              </div>
            </div>

            {/* Two-column browser */}
            <div style={{ display: 'flex', gap: '12px', flex: 1, minHeight: 0 }}>

              {/* Sidebar */}
              <div style={{
                width: '130px',
                flexShrink: 0,
                display: 'flex',
                flexDirection: 'column',
                gap: '2px',
                fontSize: '12px',
              }}>
                {browseShortcuts.map((sc) => (
                  <div
                    key={sc.path}
                    onClick={() => browseDirectory(sc.path)}
                    style={{
                      padding: '5px 8px',
                      borderRadius: '4px',
                      cursor: 'pointer',
                      color: browseCurrent === sc.path ? '#e94560' : '#aaa',
                      backgroundColor: browseCurrent === sc.path ? 'rgba(233, 69, 96, 0.1)' : 'transparent',
                    }}
                    onMouseEnter={(e) => { if (browseCurrent !== sc.path) e.currentTarget.style.backgroundColor = 'rgba(255,255,255,0.03)' }}
                    onMouseLeave={(e) => { if (browseCurrent !== sc.path) e.currentTarget.style.backgroundColor = 'transparent' }}
                  >
                    {sc.name}
                  </div>
                ))}

                {recentFiles.length > 0 && (
                  <>
                    <div style={{
                      borderTop: '1px solid #0f3460',
                      marginTop: '6px',
                      paddingTop: '6px',
                      fontSize: '10px',
                      color: '#666',
                      textTransform: 'uppercase',
                      letterSpacing: '0.5px',
                      paddingLeft: '8px',
                    }}>
                      Recent
                    </div>
                    {recentFiles.map((fp) => {
                      const fname = fp.split('/').pop() || fp
                      return (
                        <div
                          key={fp}
                          onClick={() => {
                            setLoadFilePath(fp)
                            setLoadError(null)
                            const parentDir = fp.substring(0, fp.lastIndexOf('/'))
                            if (parentDir) browseDirectory(parentDir)
                          }}
                          title={fp}
                          style={{
                            padding: '4px 8px',
                            borderRadius: '4px',
                            cursor: 'pointer',
                            color: loadFilePath === fp ? '#e94560' : '#888',
                            backgroundColor: loadFilePath === fp ? 'rgba(233, 69, 96, 0.1)' : 'transparent',
                            overflow: 'hidden',
                            textOverflow: 'ellipsis',
                            whiteSpace: 'nowrap',
                            fontSize: '11px',
                          }}
                          onMouseEnter={(e) => { if (loadFilePath !== fp) e.currentTarget.style.backgroundColor = 'rgba(255,255,255,0.03)' }}
                          onMouseLeave={(e) => { if (loadFilePath !== fp) e.currentTarget.style.backgroundColor = 'transparent' }}
                        >
                          {fname}
                        </div>
                      )
                    })}
                  </>
                )}
              </div>

              {/* Main browser area */}
              <div style={{ flex: 1, display: 'flex', flexDirection: 'column', minWidth: 0 }}>

                {/* Breadcrumb path bar */}
                {browseCurrent && (
                  <div style={{
                    display: 'flex',
                    alignItems: 'center',
                    gap: '2px',
                    marginBottom: '6px',
                    fontSize: '11px',
                    overflowX: 'auto',
                    whiteSpace: 'nowrap',
                    padding: '4px 0',
                  }}>
                    {(() => {
                      const parts = browseCurrent.split('/').filter(Boolean)
                      return (
                        <>
                          <span
                            onClick={() => browseDirectory('/')}
                            style={{ cursor: 'pointer', color: '#888', padding: '1px 3px', borderRadius: '3px' }}
                            onMouseEnter={(e) => { e.currentTarget.style.backgroundColor = 'rgba(255,255,255,0.06)' }}
                            onMouseLeave={(e) => { e.currentTarget.style.backgroundColor = 'transparent' }}
                          >/</span>
                          {parts.map((part, i) => {
                            const fullPath = '/' + parts.slice(0, i + 1).join('/')
                            const isLast = i === parts.length - 1
                            return (
                              <span key={fullPath} style={{ display: 'flex', alignItems: 'center', gap: '2px' }}>
                                <span style={{ color: '#555' }}>&rsaquo;</span>
                                <span
                                  onClick={() => !isLast && browseDirectory(fullPath)}
                                  style={{
                                    cursor: isLast ? 'default' : 'pointer',
                                    color: isLast ? '#ccc' : '#888',
                                    fontWeight: isLast ? 600 : 400,
                                    padding: '1px 3px',
                                    borderRadius: '3px',
                                  }}
                                  onMouseEnter={(e) => { if (!isLast) e.currentTarget.style.backgroundColor = 'rgba(255,255,255,0.06)' }}
                                  onMouseLeave={(e) => { e.currentTarget.style.backgroundColor = 'transparent' }}
                                >
                                  {part}
                                </span>
                              </span>
                            )
                          })}
                        </>
                      )
                    })()}
                  </div>
                )}

                {/* File/directory list */}
                <div style={{
                  flex: 1,
                  minHeight: '200px',
                  maxHeight: '300px',
                  overflowY: 'auto',
                  backgroundColor: '#0d1b30',
                  border: '1px solid #1a1a2e',
                  borderRadius: '4px',
                }}>
                  {browseLoading ? (
                    <div style={{ padding: '20px', textAlign: 'center', color: '#888', fontSize: '12px' }}>Loading...</div>
                  ) : browseEntries.length === 0 ? (
                    <div style={{ padding: '20px', textAlign: 'center', color: '#666', fontSize: '12px' }}>No folders or data files here</div>
                  ) : (
                    browseEntries.map((entry) => (
                      <div
                        key={entry.path}
                        onClick={() => {
                          if (entry.type === 'directory') {
                            browseDirectory(entry.path)
                          } else {
                            // 'file' and '10x_mtx' are both loadable
                            setLoadFilePath(entry.path)
                            setLoadError(null)
                          }
                        }}
                        style={{
                          padding: '5px 10px',
                          fontSize: '12px',
                          color: entry.type === 'directory' ? '#aaa' : '#e94560',  // 10x_mtx styled like files
                          cursor: 'pointer',
                          display: 'flex',
                          alignItems: 'center',
                          justifyContent: 'space-between',
                          backgroundColor: entry.path === loadFilePath ? 'rgba(233, 69, 96, 0.15)' : 'transparent',
                          borderBottom: '1px solid #1a1a2e',
                        }}
                        onMouseEnter={(e) => {
                          if (entry.path !== loadFilePath) e.currentTarget.style.backgroundColor = 'rgba(255,255,255,0.03)'
                        }}
                        onMouseLeave={(e) => {
                          if (entry.path !== loadFilePath) e.currentTarget.style.backgroundColor = 'transparent'
                        }}
                      >
                        <span style={{ display: 'flex', alignItems: 'center', gap: '6px', overflow: 'hidden' }}>
                          <span style={{ flexShrink: 0 }}>{entry.type === 'directory' ? '\uD83D\uDCC1' : entry.type === '10x_mtx' ? '\uD83D\uDDC2\uFE0F' : '\uD83D\uDCC4'}</span>
                          <span style={{ overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap' }}>{entry.name}</span>
                        </span>
                        {entry.type === 'file' && entry.size != null && (
                          <span style={{ fontSize: '11px', color: '#666', flexShrink: 0, marginLeft: '8px' }}>
                            {entry.size < 1024 * 1024
                              ? `${(entry.size / 1024).toFixed(0)} KB`
                              : entry.size < 1024 * 1024 * 1024
                                ? `${(entry.size / (1024 * 1024)).toFixed(1)} MB`
                                : `${(entry.size / (1024 * 1024 * 1024)).toFixed(2)} GB`}
                          </span>
                        )}
                      </div>
                    ))
                  )}
                </div>
              </div>
            </div>

            {/* Path input */}
            <div style={{ fontSize: '11px', color: '#666', marginBottom: '4px', marginTop: '12px' }}>Or enter path directly:</div>
            <input
              type="text"
              value={loadFilePath}
              onChange={(e) => setLoadFilePath(e.target.value)}
              onKeyDown={(e) => e.key === 'Enter' && !loadLoading && loadFilePath.trim() && handleLoadDataset()}
              placeholder="/path/to/data.h5ad or .h5"
              disabled={loadLoading}
              style={{
                width: '100%',
                padding: '8px 12px',
                fontSize: '13px',
                backgroundColor: '#0f3460',
                color: '#eee',
                border: '1px solid #1a1a2e',
                borderRadius: '4px',
                marginBottom: '8px',
                outline: 'none',
                boxSizing: 'border-box',
              }}
            />
            {loadError && (
              <div style={{ fontSize: '12px', color: '#e94560', marginBottom: '8px' }}>
                {loadError}
              </div>
            )}
            <div style={{ display: 'flex', gap: '8px', justifyContent: 'flex-end', marginTop: '4px' }}>
              <button
                onClick={() => setIsLoadModalOpen(false)}
                disabled={loadLoading}
                style={{
                  padding: '8px 16px',
                  fontSize: '13px',
                  backgroundColor: 'transparent',
                  color: '#aaa',
                  border: '1px solid #0f3460',
                  borderRadius: '4px',
                  cursor: loadLoading ? 'wait' : 'pointer',
                }}
              >
                Cancel
              </button>
              <button
                onClick={handleLoadDataset}
                disabled={loadLoading || !loadFilePath.trim()}
                style={{
                  padding: '8px 16px',
                  fontSize: '13px',
                  backgroundColor: '#e94560',
                  color: '#fff',
                  border: 'none',
                  borderRadius: '4px',
                  cursor: loadLoading || !loadFilePath.trim() ? 'not-allowed' : 'pointer',
                  opacity: !loadFilePath.trim() ? 0.5 : 1,
                }}
              >
                {loadLoading ? 'Loading...' : 'Load'}
              </button>
            </div>
          </div>
        </div>
      )}

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
