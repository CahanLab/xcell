/**
 * HeatmapView — canvas-based expression heatmap for the center panel.
 *
 * Shows a per-gene (or per-gene-set) expression matrix with cells ordered
 * along a user-chosen axis. Gene labels on the left, optional group
 * separators, viridis colormap, hover tooltip, click-to-color-by-gene.
 *
 * Rollback: delete this file, HeatmapConfigModal.tsx, and remove imports
 * from App.tsx plus the heatmap state from store.ts.
 */

import { useState, useEffect, useRef, useCallback } from 'react'
import { useStore, HeatmapConfig } from '../store'
import { useDataActions } from '../hooks/useData'
import HeatmapConfigModal from './HeatmapConfigModal'

// ---------------------------------------------------------------------------
// Viridis colormap (5-stop approximation)
// ---------------------------------------------------------------------------

const VIRIDIS_STOPS: [number, number, number][] = [
  [68, 1, 84],
  [59, 82, 139],
  [33, 145, 140],
  [94, 201, 98],
  [253, 231, 37],
]

function viridisColor(t: number): [number, number, number] {
  const clamped = Math.max(0, Math.min(1, t))
  const idx = clamped * (VIRIDIS_STOPS.length - 1)
  const lo = Math.floor(idx)
  const hi = Math.min(lo + 1, VIRIDIS_STOPS.length - 1)
  const f = idx - lo
  return [
    Math.round(VIRIDIS_STOPS[lo][0] + (VIRIDIS_STOPS[hi][0] - VIRIDIS_STOPS[lo][0]) * f),
    Math.round(VIRIDIS_STOPS[lo][1] + (VIRIDIS_STOPS[hi][1] - VIRIDIS_STOPS[lo][1]) * f),
    Math.round(VIRIDIS_STOPS[lo][2] + (VIRIDIS_STOPS[hi][2] - VIRIDIS_STOPS[lo][2]) * f),
  ]
}

// Module/group colors (same palette as scatter plot categories)
const GROUP_COLORS = [
  '#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd',
  '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf',
]

// ---------------------------------------------------------------------------
// Types
// ---------------------------------------------------------------------------

interface HeatmapData {
  matrix: number[][]
  row_labels: string[]
  row_groups: (string | null)[]
  column_groups: { name: string; start: number; size: number }[]
  n_bins: number
  n_cells: number
}

// ---------------------------------------------------------------------------
// HeatmapCanvas
// ---------------------------------------------------------------------------

const LABEL_WIDTH = 110
const GROUP_BAR_WIDTH = 8
const COL_LABEL_HEIGHT = 40
const ROW_HEIGHT = 16
const MIN_CELL_WIDTH = 1
const MAX_CELL_WIDTH = 8

function HeatmapCanvas({ data, onGeneClick }: { data: HeatmapData; onGeneClick: (gene: string) => void }) {
  const canvasRef = useRef<HTMLCanvasElement>(null)
  const containerRef = useRef<HTMLDivElement>(null)
  const [hoveredRow, setHoveredRow] = useState<number | null>(null)
  const [tooltip, setTooltip] = useState<{ x: number; y: number; text: string } | null>(null)

  const nRows = data.matrix.length
  const nCols = nRows > 0 ? data.matrix[0].length : 0

  // Compute unique row groups for color bar
  const uniqueGroups: string[] = []
  const groupSet = new Set<string>()
  for (const g of data.row_groups) {
    if (g && !groupSet.has(g)) {
      groupSet.add(g)
      uniqueGroups.push(g)
    }
  }
  const hasGroups = uniqueGroups.length > 0
  const groupBarOffset = LABEL_WIDTH + (hasGroups ? GROUP_BAR_WIDTH + 2 : 0)

  // Canvas sizing
  const containerWidth = containerRef.current?.clientWidth || 800
  const availableWidth = containerWidth - groupBarOffset - 20
  const cellWidth = Math.max(MIN_CELL_WIDTH, Math.min(MAX_CELL_WIDTH, availableWidth / nCols))
  const heatmapWidth = cellWidth * nCols
  const canvasLogicalWidth = groupBarOffset + heatmapWidth
  const canvasLogicalHeight = nRows * ROW_HEIGHT + COL_LABEL_HEIGHT

  // Draw
  useEffect(() => {
    const canvas = canvasRef.current
    if (!canvas || nRows === 0 || nCols === 0) return

    const dpr = window.devicePixelRatio || 1
    canvas.width = canvasLogicalWidth * dpr
    canvas.height = canvasLogicalHeight * dpr
    canvas.style.width = canvasLogicalWidth + 'px'
    canvas.style.height = canvasLogicalHeight + 'px'

    const ctx = canvas.getContext('2d')
    if (!ctx) return
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0)

    // Clear
    ctx.fillStyle = '#1a1a2e'
    ctx.fillRect(0, 0, canvasLogicalWidth, canvasLogicalHeight)

    // Gene labels (y-axis)
    ctx.font = '11px monospace'
    ctx.textBaseline = 'middle'
    for (let r = 0; r < nRows; r++) {
      const y = r * ROW_HEIGHT + ROW_HEIGHT / 2
      const isHovered = r === hoveredRow
      ctx.fillStyle = isHovered ? '#fff' : '#ccc'
      const label = data.row_labels[r]
      const truncated = label.length > 14 ? label.slice(0, 13) + '\u2026' : label
      ctx.fillText(truncated, 4, y)
    }

    // Gene set group color bar
    if (hasGroups) {
      for (let r = 0; r < nRows; r++) {
        const group = data.row_groups[r]
        if (group) {
          const gIdx = uniqueGroups.indexOf(group)
          ctx.fillStyle = GROUP_COLORS[gIdx % GROUP_COLORS.length]
        } else {
          ctx.fillStyle = '#333'
        }
        ctx.fillRect(LABEL_WIDTH, r * ROW_HEIGHT, GROUP_BAR_WIDTH, ROW_HEIGHT)
      }

      // Group boundaries
      let prevGroup: string | null = null
      for (let r = 0; r < nRows; r++) {
        if (r > 0 && data.row_groups[r] !== prevGroup && prevGroup !== null) {
          ctx.strokeStyle = '#555'
          ctx.lineWidth = 1
          ctx.beginPath()
          ctx.moveTo(LABEL_WIDTH, r * ROW_HEIGHT)
          ctx.lineTo(groupBarOffset + heatmapWidth, r * ROW_HEIGHT)
          ctx.stroke()
        }
        prevGroup = data.row_groups[r]
      }
    }

    // Heatmap cells
    for (let r = 0; r < nRows; r++) {
      const row = data.matrix[r]
      for (let c = 0; c < nCols; c++) {
        const val = row[c]
        const [cr, cg, cb] = viridisColor(val)
        ctx.fillStyle = `rgb(${cr},${cg},${cb})`
        ctx.fillRect(groupBarOffset + c * cellWidth, r * ROW_HEIGHT, cellWidth + 0.5, ROW_HEIGHT)
      }
    }

    // Hover highlight
    if (hoveredRow !== null) {
      ctx.strokeStyle = 'rgba(255, 255, 255, 0.6)'
      ctx.lineWidth = 1
      ctx.strokeRect(groupBarOffset, hoveredRow * ROW_HEIGHT, heatmapWidth, ROW_HEIGHT)
    }

    // Column group labels
    if (data.column_groups.length > 0) {
      const labelY = nRows * ROW_HEIGHT + 4
      ctx.font = '10px sans-serif'
      ctx.textBaseline = 'top'

      for (const grp of data.column_groups) {
        const x = groupBarOffset + grp.start * cellWidth
        const w = grp.size * cellWidth

        // Group separator line
        if (grp.start > 0) {
          ctx.strokeStyle = '#888'
          ctx.lineWidth = 1
          ctx.beginPath()
          ctx.moveTo(x, 0)
          ctx.lineTo(x, nRows * ROW_HEIGHT)
          ctx.stroke()
        }

        // Group label
        ctx.fillStyle = '#aaa'
        const labelText = grp.name.length > Math.floor(w / 7) ? grp.name.slice(0, Math.max(1, Math.floor(w / 7) - 1)) + '\u2026' : grp.name
        ctx.save()
        ctx.translate(x + w / 2, labelY)
        ctx.rotate(-Math.PI / 4)
        ctx.fillText(labelText, 0, 0)
        ctx.restore()
      }
    }
  }, [data, hoveredRow, nRows, nCols, canvasLogicalWidth, canvasLogicalHeight, cellWidth, heatmapWidth, groupBarOffset, hasGroups, uniqueGroups])

  const handleMouseMove = useCallback((e: React.MouseEvent) => {
    const canvas = canvasRef.current
    if (!canvas) return
    const rect = canvas.getBoundingClientRect()
    const x = (e.clientX - rect.left)
    const y = (e.clientY - rect.top)

    const row = Math.floor(y / ROW_HEIGHT)
    const col = Math.floor((x - groupBarOffset) / cellWidth)

    if (row >= 0 && row < nRows && col >= 0 && col < nCols) {
      setHoveredRow(row)
            const val = data.matrix[row][col]
      const gene = data.row_labels[row]
      let colInfo = `bin ${col + 1}`
      for (const grp of data.column_groups) {
        if (col >= grp.start && col < grp.start + grp.size) {
          colInfo = grp.name
          break
        }
      }
      setTooltip({
        x: e.clientX,
        y: e.clientY,
        text: `${gene} | ${colInfo} | ${val.toFixed(3)}`,
      })
    } else {
      setHoveredRow(null)
            setTooltip(null)
    }
  }, [data, nRows, nCols, cellWidth, groupBarOffset])

  const handleMouseLeave = useCallback(() => {
    setHoveredRow(null)
        setTooltip(null)
  }, [])

  const handleClick = useCallback((e: React.MouseEvent) => {
    const canvas = canvasRef.current
    if (!canvas) return
    const rect = canvas.getBoundingClientRect()
    const y = e.clientY - rect.top
    const row = Math.floor(y / ROW_HEIGHT)
    if (row >= 0 && row < nRows) {
      onGeneClick(data.row_labels[row])
    }
  }, [data, nRows, onGeneClick])

  return (
    <div ref={containerRef} style={hmStyles.canvasContainer}>
      <canvas
        ref={canvasRef}
        style={{ cursor: 'pointer' }}
        onMouseMove={handleMouseMove}
        onMouseLeave={handleMouseLeave}
        onClick={handleClick}
      />

      {/* Tooltip */}
      {tooltip && (
        <div
          style={{
            position: 'fixed',
            left: tooltip.x + 12,
            top: tooltip.y - 8,
            padding: '4px 8px',
            backgroundColor: 'rgba(0, 0, 0, 0.85)',
            color: '#eee',
            fontSize: '11px',
            fontFamily: 'monospace',
            borderRadius: '4px',
            pointerEvents: 'none',
            zIndex: 1000,
            whiteSpace: 'nowrap',
          }}
        >
          {tooltip.text}
        </div>
      )}

      {/* Color legend */}
      <div style={hmStyles.legend}>
        <div style={hmStyles.legendTitle}>Normalized Expression</div>
        <div style={hmStyles.legendBar} />
        <div style={hmStyles.legendLabels}>
          <span>0</span>
          <span>1</span>
        </div>
      </div>

      {/* Gene set group legend */}
      {hasGroups && (
        <div style={hmStyles.groupLegend}>
          {uniqueGroups.map((g, i) => (
            <div key={g} style={{ display: 'flex', alignItems: 'center', gap: '6px', marginBottom: '2px' }}>
              <div style={{ width: '10px', height: '10px', borderRadius: '2px', backgroundColor: GROUP_COLORS[i % GROUP_COLORS.length] }} />
              <span style={{ fontSize: '10px', color: '#aaa' }}>{g}</span>
            </div>
          ))}
        </div>
      )}
    </div>
  )
}

// ---------------------------------------------------------------------------
// HeatmapView (main component)
// ---------------------------------------------------------------------------

export default function HeatmapView() {
  const heatmapConfig = useStore((s) => s.heatmapConfig)
  const setHeatmapConfig = useStore((s) => s.setHeatmapConfig)
  const drawnLines = useStore((s) => s.drawnLines)
  const displayPreferences = useStore((s) => s.displayPreferences)
  const { colorByGene } = useDataActions()

  const [configOpen, setConfigOpen] = useState(!heatmapConfig)
  const [data, setData] = useState<HeatmapData | null>(null)
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState<string | null>(null)

  // Fetch heatmap data when config changes
  useEffect(() => {
    if (!heatmapConfig) return
    fetchHeatmapData(heatmapConfig)
  }, [heatmapConfig])

  const fetchHeatmapData = async (config: HeatmapConfig) => {
    setLoading(true)
    setError(null)
    try {
      // Sync drawn lines to backend if needed for line-based ordering
      if (config.lineName && drawnLines.length > 0) {
        const linesPayload = drawnLines.map((l) => ({
          name: l.name,
          embeddingName: l.embeddingName,
          points: l.points,
          smoothedPoints: l.smoothedPoints,
        }))
        await fetch('/api/lines', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ lines: linesPayload }),
        })
      }

      // Build gene list and gene_set_groups
      const allGenes: string[] = []
      const geneSetGroups: { name: string; genes: string[] }[] = []
      for (const gs of config.selectedGeneSets) {
        geneSetGroups.push({ name: gs.name, genes: gs.genes })
        allGenes.push(...gs.genes)
      }
      const seen = new Set<string>()
      const uniqueGenes: string[] = []
      for (const g of allGenes) {
        if (!seen.has(g)) {
          seen.add(g)
          uniqueGenes.push(g)
        }
      }

      const transform = displayPreferences.expressionTransform === 'log1p' ? 'log1p' : null

      const response = await fetch('/api/heatmap/data', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          genes: uniqueGenes,
          gene_set_groups: geneSetGroups,
          aggregate_gene_sets: config.aggregateGeneSets,
          cell_ordering: config.cellOrdering,
          obs_column: config.obsColumn,
          line_name: config.lineName,
          gene_ordering: config.geneOrdering,
          n_bins: config.nBins,
          transform,
        }),
      })

      if (!response.ok) {
        const err = await response.json().catch(() => ({ detail: 'Unknown error' }))
        throw new Error(err.detail || `HTTP ${response.status}`)
      }

      const result: HeatmapData = await response.json()
      setData(result)
    } catch (err) {
      setError((err as Error).message)
      setData(null)
    } finally {
      setLoading(false)
    }
  }

  const handleGeneClick = useCallback((gene: string) => {
    colorByGene(gene)
  }, [colorByGene])

  // Show config panel
  if (configOpen || !heatmapConfig) {
    return (
      <HeatmapConfigModal
        config={heatmapConfig}
        onApply={(config) => {
          setHeatmapConfig(config)
          setConfigOpen(false)
        }}
        onCancel={() => setConfigOpen(false)}
      />
    )
  }

  if (loading) {
    return (
      <div style={hmStyles.centered}>
        <div style={{ color: '#aaa', fontSize: '14px' }}>Computing heatmap...</div>
      </div>
    )
  }

  if (error) {
    return (
      <div style={hmStyles.centered}>
        <div style={{ color: '#e94560', fontSize: '13px', marginBottom: '12px' }}>{error}</div>
        <button style={hmStyles.settingsButton} onClick={() => setConfigOpen(true)}>
          Reconfigure
        </button>
      </div>
    )
  }

  if (!data || data.matrix.length === 0) {
    return (
      <div style={hmStyles.centered}>
        <div style={{ color: '#888', fontSize: '13px', marginBottom: '12px' }}>
          No data to display. Check that selected gene sets contain valid genes.
        </div>
        <button style={hmStyles.settingsButton} onClick={() => setConfigOpen(true)}>
          Configure Heatmap
        </button>
      </div>
    )
  }

  return (
    <div style={hmStyles.wrapper}>
      {/* Toolbar */}
      <div style={hmStyles.toolbar}>
        <button style={hmStyles.settingsButton} onClick={() => setConfigOpen(true)}>
          Settings
        </button>
        <span style={hmStyles.info}>
          {data.row_labels.length} genes &times; {data.n_bins} {data.n_bins < data.n_cells ? 'bins' : 'cells'}
          {data.n_bins < data.n_cells && ` (${data.n_cells.toLocaleString()} cells)`}
        </span>
        <span style={hmStyles.hint}>Click a gene row to color scatter plot</span>
      </div>

      {/* Canvas heatmap */}
      <HeatmapCanvas data={data} onGeneClick={handleGeneClick} />
    </div>
  )
}

// ---------------------------------------------------------------------------
// Styles
// ---------------------------------------------------------------------------

const hmStyles: Record<string, React.CSSProperties> = {
  wrapper: {
    display: 'flex',
    flexDirection: 'column',
    height: '100%',
    overflow: 'hidden',
  },
  toolbar: {
    display: 'flex',
    alignItems: 'center',
    gap: '12px',
    padding: '8px 12px',
    borderBottom: '1px solid #0f3460',
    flexShrink: 0,
  },
  settingsButton: {
    padding: '4px 12px',
    fontSize: '12px',
    backgroundColor: '#0f3460',
    color: '#aaa',
    border: '1px solid #1a1a2e',
    borderRadius: '4px',
    cursor: 'pointer',
  },
  info: {
    fontSize: '12px',
    color: '#888',
  },
  hint: {
    fontSize: '11px',
    color: '#555',
    marginLeft: 'auto',
  },
  centered: {
    display: 'flex',
    flexDirection: 'column',
    alignItems: 'center',
    justifyContent: 'center',
    height: '100%',
    padding: '20px',
  },
  canvasContainer: {
    flex: 1,
    overflow: 'auto',
    padding: '8px',
    position: 'relative',
  },
  legend: {
    position: 'absolute',
    bottom: '12px',
    right: '12px',
    padding: '8px',
    backgroundColor: 'rgba(22, 33, 62, 0.9)',
    borderRadius: '6px',
  },
  legendTitle: {
    fontSize: '10px',
    color: '#888',
    marginBottom: '4px',
  },
  legendBar: {
    width: '100px',
    height: '10px',
    borderRadius: '2px',
    background: 'linear-gradient(to right, rgb(68,1,84), rgb(59,82,139), rgb(33,145,140), rgb(94,201,98), rgb(253,231,37))',
  },
  legendLabels: {
    display: 'flex',
    justifyContent: 'space-between',
    fontSize: '9px',
    color: '#888',
    marginTop: '2px',
  },
  groupLegend: {
    position: 'absolute',
    bottom: '12px',
    right: '140px',
    padding: '8px',
    backgroundColor: 'rgba(22, 33, 62, 0.9)',
    borderRadius: '6px',
  },
}
