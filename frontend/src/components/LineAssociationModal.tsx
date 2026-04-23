import { useState, useRef, useEffect, useCallback } from 'react'
import { useStore, LineAssociationGene, LineAssociationModule } from '../store'
import { useDataActions } from '../hooks/useData'

const PATTERN_COLORS: Record<string, string> = {
  increasing: '#4ecdc4',
  decreasing: '#e94560',
  peak: '#f0a500',
  trough: '#7b68ee',
  complex: '#888',
}

const PATTERN_ICONS: Record<string, string> = {
  increasing: '\u2197',   // ↗
  decreasing: '\u2198',   // ↘
  peak: '\u2229',         // ∩
  trough: '\u222A',       // ∪
  complex: '\u223F',      // ∿
}

// Viridis colormap stops for the heatmap
const VIRIDIS_STOPS = [
  { pos: 0, r: 68, g: 1, b: 84 },
  { pos: 0.25, r: 59, g: 82, b: 139 },
  { pos: 0.5, r: 33, g: 145, b: 140 },
  { pos: 0.75, r: 94, g: 201, b: 98 },
  { pos: 1, r: 253, g: 231, b: 37 },
]

function viridisColor(t: number): [number, number, number] {
  const clamped = Math.max(0, Math.min(1, t))
  for (let i = 0; i < VIRIDIS_STOPS.length - 1; i++) {
    if (clamped >= VIRIDIS_STOPS[i].pos && clamped <= VIRIDIS_STOPS[i + 1].pos) {
      const local = (clamped - VIRIDIS_STOPS[i].pos) / (VIRIDIS_STOPS[i + 1].pos - VIRIDIS_STOPS[i].pos)
      return [
        Math.round(VIRIDIS_STOPS[i].r + (VIRIDIS_STOPS[i + 1].r - VIRIDIS_STOPS[i].r) * local),
        Math.round(VIRIDIS_STOPS[i].g + (VIRIDIS_STOPS[i + 1].g - VIRIDIS_STOPS[i].g) * local),
        Math.round(VIRIDIS_STOPS[i].b + (VIRIDIS_STOPS[i + 1].b - VIRIDIS_STOPS[i].b) * local),
      ]
    }
  }
  const last = VIRIDIS_STOPS[VIRIDIS_STOPS.length - 1]
  return [last.r, last.g, last.b]
}

type ViewMode = 'list' | 'heatmap'

const styles = {
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
    width: '900px',
    maxWidth: '95vw',
    maxHeight: '85vh',
    display: 'flex',
    flexDirection: 'column' as const,
    boxShadow: '0 8px 32px rgba(0, 0, 0, 0.4)',
  },
  header: {
    padding: '16px 20px',
    borderBottom: '1px solid #0f3460',
    display: 'flex',
    justifyContent: 'space-between',
    alignItems: 'center',
  },
  title: {
    fontSize: '16px',
    fontWeight: 600,
    color: '#e94560',
    margin: 0,
  },
  closeButton: {
    background: 'none',
    border: 'none',
    color: '#aaa',
    fontSize: '20px',
    cursor: 'pointer',
    padding: '4px 8px',
  },
  content: {
    flex: 1,
    overflow: 'auto',
    padding: '16px 20px',
  },
  summary: {
    display: 'flex',
    gap: '24px',
    marginBottom: '16px',
    flexWrap: 'wrap' as const,
  },
  summaryItem: {
    backgroundColor: '#0f3460',
    padding: '8px 16px',
    borderRadius: '6px',
  },
  summaryLabel: {
    fontSize: '11px',
    color: '#888',
    marginBottom: '2px',
  },
  summaryValue: {
    fontSize: '14px',
    color: '#eee',
    fontWeight: 500,
  },
  geneRow: {
    display: 'flex',
    alignItems: 'center',
    padding: '6px 16px',
    borderBottom: '1px solid #1a1a2e',
    cursor: 'pointer',
    transition: 'background-color 0.1s',
  },
  geneRowHover: {
    backgroundColor: '#16213e',
  },
  geneName: {
    flex: 1,
    fontSize: '13px',
    color: '#eee',
    fontFamily: 'monospace',
  },
  geneStat: {
    fontSize: '11px',
    color: '#888',
    marginLeft: '12px',
    minWidth: '50px',
    textAlign: 'right' as const,
  },
  geneR2: {
    fontSize: '11px',
    color: '#4ecdc4',
    marginLeft: '12px',
    minWidth: '45px',
    textAlign: 'right' as const,
  },
  geneFDR: {
    fontSize: '11px',
    color: '#e94560',
    marginLeft: '12px',
    minWidth: '70px',
    textAlign: 'right' as const,
  },
  footer: {
    padding: '16px 20px',
    borderTop: '1px solid #0f3460',
    display: 'flex',
    justifyContent: 'space-between',
    alignItems: 'center',
    gap: '12px',
  },
  button: {
    padding: '8px 16px',
    fontSize: '13px',
    borderRadius: '6px',
    border: 'none',
    cursor: 'pointer',
    fontWeight: 500,
  },
  primaryButton: {
    backgroundColor: '#e94560',
    color: '#fff',
  },
  secondaryButton: {
    backgroundColor: '#0f3460',
    color: '#eee',
  },
}

function Sparkline({
  profile,
  color,
  width = 120,
  height = 32,
}: {
  profile: number[]
  color: string
  width?: number
  height?: number
}) {
  if (profile.length < 2) return null

  const padding = 2
  const w = width - padding * 2
  const h = height - padding * 2

  const points = profile
    .map((v, i) => {
      const x = padding + (i / (profile.length - 1)) * w
      const y = padding + (1 - v) * h
      return `${x},${y}`
    })
    .join(' ')

  return (
    <svg width={width} height={height} style={{ display: 'block' }}>
      <polyline
        points={points}
        fill="none"
        stroke={color}
        strokeWidth={2}
        strokeLinecap="round"
        strokeLinejoin="round"
      />
    </svg>
  )
}

function GeneRow({
  gene,
  onSelect,
}: {
  gene: LineAssociationGene
  onSelect: (gene: string) => void
}) {
  const [isHovered, setIsHovered] = useState(false)

  return (
    <div
      style={{
        ...styles.geneRow,
        ...(isHovered ? styles.geneRowHover : {}),
      }}
      onMouseEnter={() => setIsHovered(true)}
      onMouseLeave={() => setIsHovered(false)}
      onClick={() => onSelect(gene.gene)}
    >
      <span style={styles.geneName}>{gene.gene}</span>
      <span style={styles.geneR2} title="R-squared (variance explained)">
        R²={gene.r_squared.toFixed(2)}
      </span>
      <span style={styles.geneStat} title="Amplitude (expression range)">
        A={gene.amplitude.toFixed(2)}
      </span>
      <span style={styles.geneFDR} title="FDR-adjusted p-value">
        {gene.fdr < 0.001 ? gene.fdr.toExponential(1) : gene.fdr.toFixed(3)}
      </span>
    </div>
  )
}

function ModuleCard({
  module,
  onGeneSelect,
  defaultExpanded,
}: {
  module: LineAssociationModule
  onGeneSelect: (gene: string) => void
  defaultExpanded: boolean
}) {
  const [expanded, setExpanded] = useState(defaultExpanded)
  const color = PATTERN_COLORS[module.pattern] || PATTERN_COLORS.complex
  const icon = PATTERN_ICONS[module.pattern] || PATTERN_ICONS.complex

  return (
    <div style={{
      backgroundColor: '#0a0f1a',
      borderRadius: '8px',
      overflow: 'hidden',
      marginBottom: '12px',
      border: `1px solid ${expanded ? color + '44' : '#1a1a2e'}`,
    }}>
      <div
        style={{
          padding: '10px 16px',
          display: 'flex',
          alignItems: 'center',
          gap: '12px',
          cursor: 'pointer',
          borderBottom: expanded ? '1px solid #0f3460' : 'none',
        }}
        onClick={() => setExpanded(!expanded)}
      >
        <span style={{
          fontSize: '16px',
          color,
          width: '20px',
          textAlign: 'center',
        }}>
          {icon}
        </span>
        <div style={{ flex: 1 }}>
          <span style={{
            fontSize: '13px',
            fontWeight: 600,
            color: '#eee',
            textTransform: 'capitalize',
          }}>
            {module.pattern}
          </span>
          <span style={{
            fontSize: '11px',
            color: '#888',
            marginLeft: '8px',
          }}>
            {module.n_genes} gene{module.n_genes !== 1 ? 's' : ''}
          </span>
        </div>
        <Sparkline
          profile={module.representative_profile}
          color={color}
        />
        <span style={{
          fontSize: '14px',
          color: '#666',
          marginLeft: '4px',
          transition: 'transform 0.15s',
          transform: expanded ? 'rotate(180deg)' : 'rotate(0deg)',
        }}>
          &#9660;
        </span>
      </div>

      {expanded && (
        <div style={{ maxHeight: '300px', overflow: 'auto' }}>
          {module.genes.map((gene) => (
            <GeneRow
              key={gene.gene}
              gene={gene}
              onSelect={onGeneSelect}
            />
          ))}
        </div>
      )}
    </div>
  )
}

// Simple direction card for the non-clustered path (positive / negative lists).
// Mirrors ModuleCard's expand/collapse + GeneRow body, without module/pattern UI.
function DirectionCard({
  title,
  subtitle,
  genes,
  color,
  onGeneSelect,
}: {
  title: string
  subtitle: string
  genes: LineAssociationGene[]
  color: string
  onGeneSelect: (gene: string) => void
}) {
  const [expanded, setExpanded] = useState(true)
  return (
    <div style={{
      backgroundColor: '#0a0f1a',
      borderRadius: '8px',
      overflow: 'hidden',
      marginBottom: '12px',
      border: `1px solid ${expanded ? color + '44' : '#1a1a2e'}`,
    }}>
      <div
        style={{
          padding: '10px 16px',
          display: 'flex',
          alignItems: 'center',
          gap: '12px',
          cursor: 'pointer',
          borderBottom: expanded ? '1px solid #0f3460' : 'none',
        }}
        onClick={() => setExpanded(!expanded)}
      >
        <div style={{ flex: 1 }}>
          <span style={{ fontSize: '13px', fontWeight: 600, color: '#eee' }}>{title}</span>
          <span style={{ fontSize: '11px', color: '#888', marginLeft: '8px' }}>
            {genes.length} gene{genes.length !== 1 ? 's' : ''}
          </span>
          <div style={{ fontSize: '11px', color: '#666', marginTop: '2px' }}>{subtitle}</div>
        </div>
        <span style={{
          fontSize: '14px',
          color: '#666',
          marginLeft: '4px',
          transition: 'transform 0.15s',
          transform: expanded ? 'rotate(180deg)' : 'rotate(0deg)',
        }}>
          &#9660;
        </span>
      </div>
      {expanded && (
        <div style={{ maxHeight: '300px', overflow: 'auto' }}>
          {genes.map((gene) => (
            <GeneRow key={gene.gene} gene={gene} onSelect={onGeneSelect} />
          ))}
        </div>
      )}
    </div>
  )
}

// =========================================================================
// Heatmap component
// =========================================================================

interface HeatmapGeneEntry {
  gene: LineAssociationGene
  modulePattern: string
  moduleId: number
}

function ProfileHeatmap({
  modules,
  onGeneSelect,
  testVariable,
}: {
  modules: LineAssociationModule[]
  onGeneSelect: (gene: string) => void
  testVariable: string
}) {
  const canvasRef = useRef<HTMLCanvasElement>(null)
  const containerRef = useRef<HTMLDivElement>(null)
  const [hoveredGene, setHoveredGene] = useState<string | null>(null)
  const [tooltipInfo, setTooltipInfo] = useState<{ gene: string; x: number; y: number; value: number } | null>(null)

  // Collect all genes with profiles across modules, preserving module order
  const geneEntries: HeatmapGeneEntry[] = []
  const moduleBoundaries: { startIdx: number; pattern: string; moduleId: number }[] = []

  for (const mod of modules) {
    const startIdx = geneEntries.length
    moduleBoundaries.push({ startIdx, pattern: mod.pattern, moduleId: mod.module_id })
    for (const gene of mod.genes) {
      if (gene.profile && gene.profile.length > 0) {
        geneEntries.push({ gene, modulePattern: mod.pattern, moduleId: mod.module_id })
      }
    }
  }

  const nGenes = geneEntries.length
  const nPositions = nGenes > 0 && geneEntries[0].gene.profile ? geneEntries[0].gene.profile.length : 50

  // Layout constants
  const labelWidth = 90
  const moduleBarWidth = 6
  const cellHeight = 14
  const cellWidth = Math.max(4, Math.min(12, Math.floor(600 / nPositions)))
  const heatmapWidth = cellWidth * nPositions
  const legendHeight = 40
  const axisLabelHeight = 20
  const topPadding = 4
  const totalWidth = labelWidth + moduleBarWidth + heatmapWidth + 20
  const totalHeight = topPadding + nGenes * cellHeight + axisLabelHeight + legendHeight

  // Draw heatmap on canvas
  useEffect(() => {
    const canvas = canvasRef.current
    if (!canvas || nGenes === 0) return

    const dpr = window.devicePixelRatio || 1
    canvas.width = totalWidth * dpr
    canvas.height = totalHeight * dpr
    canvas.style.width = `${totalWidth}px`
    canvas.style.height = `${totalHeight}px`

    const ctx = canvas.getContext('2d')
    if (!ctx) return
    ctx.scale(dpr, dpr)

    // Clear
    ctx.fillStyle = '#0a0f1a'
    ctx.fillRect(0, 0, totalWidth, totalHeight)

    const heatmapLeft = labelWidth + moduleBarWidth

    // Draw module color bar
    for (let i = 0; i < moduleBoundaries.length; i++) {
      const start = moduleBoundaries[i].startIdx
      const end = i + 1 < moduleBoundaries.length ? moduleBoundaries[i + 1].startIdx : nGenes
      const color = PATTERN_COLORS[moduleBoundaries[i].pattern] || PATTERN_COLORS.complex
      ctx.fillStyle = color
      ctx.fillRect(
        labelWidth,
        topPadding + start * cellHeight,
        moduleBarWidth,
        (end - start) * cellHeight
      )
    }

    // Draw gene labels
    ctx.font = '10px monospace'
    ctx.textAlign = 'right'
    ctx.textBaseline = 'middle'
    for (let g = 0; g < nGenes; g++) {
      const entry = geneEntries[g]
      const y = topPadding + g * cellHeight + cellHeight / 2
      ctx.fillStyle = hoveredGene === entry.gene.gene ? '#fff' : '#ccc'
      // Truncate long names
      let name = entry.gene.gene
      if (name.length > 12) name = name.slice(0, 11) + '\u2026'
      ctx.fillText(name, labelWidth - 4, y)
    }

    // Draw heatmap cells
    for (let g = 0; g < nGenes; g++) {
      const profile = geneEntries[g].gene.profile!
      for (let p = 0; p < nPositions; p++) {
        const [r, gVal, b] = viridisColor(profile[p])
        ctx.fillStyle = `rgb(${r},${gVal},${b})`
        ctx.fillRect(
          heatmapLeft + p * cellWidth,
          topPadding + g * cellHeight,
          cellWidth,
          cellHeight - 1  // 1px gap between rows
        )
      }
    }

    // Highlight hovered gene row
    if (hoveredGene) {
      const idx = geneEntries.findIndex((e) => e.gene.gene === hoveredGene)
      if (idx >= 0) {
        ctx.strokeStyle = '#fff'
        ctx.lineWidth = 1
        ctx.strokeRect(
          heatmapLeft - 0.5,
          topPadding + idx * cellHeight - 0.5,
          heatmapWidth + 1,
          cellHeight
        )
      }
    }

    // Draw module boundary lines
    ctx.strokeStyle = '#333'
    ctx.lineWidth = 1
    for (let i = 1; i < moduleBoundaries.length; i++) {
      const y = topPadding + moduleBoundaries[i].startIdx * cellHeight
      ctx.beginPath()
      ctx.moveTo(heatmapLeft, y)
      ctx.lineTo(heatmapLeft + heatmapWidth, y)
      ctx.stroke()
    }

    // Draw position axis
    const axisY = topPadding + nGenes * cellHeight + 4
    ctx.fillStyle = '#888'
    ctx.font = '10px sans-serif'
    ctx.textAlign = 'center'
    ctx.textBaseline = 'top'
    const axisLabel = testVariable === 'distance' ? 'Distance from line' : 'Position along line'
    ctx.fillText(axisLabel, heatmapLeft + heatmapWidth / 2, axisY + 10)

    // Tick marks
    const ticks = [0, 0.25, 0.5, 0.75, 1.0]
    ctx.fillStyle = '#666'
    ctx.font = '9px sans-serif'
    for (const t of ticks) {
      const x = heatmapLeft + t * (heatmapWidth - cellWidth) + cellWidth / 2
      ctx.fillText(t.toFixed(2), x, axisY)
    }

    // Draw color legend
    const legendY = topPadding + nGenes * cellHeight + axisLabelHeight + 12
    const legendBarWidth = 150
    const legendBarHeight = 8
    const legendX = heatmapLeft + (heatmapWidth - legendBarWidth) / 2

    ctx.fillStyle = '#888'
    ctx.font = '9px sans-serif'
    ctx.textAlign = 'center'
    ctx.fillText('Expression (normalized)', legendX + legendBarWidth / 2, legendY - 4)

    // Draw gradient bar
    for (let i = 0; i < legendBarWidth; i++) {
      const t = i / (legendBarWidth - 1)
      const [r, gVal, b] = viridisColor(t)
      ctx.fillStyle = `rgb(${r},${gVal},${b})`
      ctx.fillRect(legendX + i, legendY, 1, legendBarHeight)
    }

    // Legend labels
    ctx.fillStyle = '#888'
    ctx.textAlign = 'left'
    ctx.fillText('Low', legendX, legendY + legendBarHeight + 10)
    ctx.textAlign = 'right'
    ctx.fillText('High', legendX + legendBarWidth, legendY + legendBarHeight + 10)

  }, [nGenes, nPositions, hoveredGene, geneEntries, moduleBoundaries, totalWidth, totalHeight, cellWidth, cellHeight, heatmapWidth, testVariable])

  // Mouse interaction
  const handleMouseMove = useCallback((e: React.MouseEvent<HTMLCanvasElement>) => {
    const canvas = canvasRef.current
    if (!canvas || nGenes === 0) return

    const rect = canvas.getBoundingClientRect()
    const x = e.clientX - rect.left
    const y = e.clientY - rect.top

    const heatmapLeft = labelWidth + moduleBarWidth

    // Check if over heatmap area
    const geneIdx = Math.floor((y - topPadding) / cellHeight)
    const posIdx = Math.floor((x - heatmapLeft) / cellWidth)

    if (geneIdx >= 0 && geneIdx < nGenes && posIdx >= 0 && posIdx < nPositions && x >= heatmapLeft) {
      const entry = geneEntries[geneIdx]
      setHoveredGene(entry.gene.gene)
      const value = entry.gene.profile ? entry.gene.profile[posIdx] : 0
      setTooltipInfo({
        gene: entry.gene.gene,
        x: e.clientX,
        y: e.clientY,
        value,
      })
    } else if (geneIdx >= 0 && geneIdx < nGenes && x < heatmapLeft) {
      // Over label area
      setHoveredGene(geneEntries[geneIdx].gene.gene)
      setTooltipInfo(null)
    } else {
      setHoveredGene(null)
      setTooltipInfo(null)
    }
  }, [nGenes, nPositions, geneEntries, cellWidth, cellHeight])

  const handleMouseLeave = useCallback(() => {
    setHoveredGene(null)
    setTooltipInfo(null)
  }, [])

  const handleClick = useCallback((e: React.MouseEvent<HTMLCanvasElement>) => {
    const canvas = canvasRef.current
    if (!canvas || nGenes === 0) return

    const rect = canvas.getBoundingClientRect()
    const y = e.clientY - rect.top
    const geneIdx = Math.floor((y - topPadding) / cellHeight)

    if (geneIdx >= 0 && geneIdx < nGenes) {
      onGeneSelect(geneEntries[geneIdx].gene.gene)
    }
  }, [nGenes, geneEntries, cellHeight, onGeneSelect])

  if (nGenes === 0) {
    return (
      <div style={{ padding: '24px', color: '#666', textAlign: 'center' }}>
        No genes with profile data available
      </div>
    )
  }

  return (
    <div ref={containerRef} style={{ position: 'relative' }}>
      <canvas
        ref={canvasRef}
        style={{ display: 'block', cursor: 'pointer' }}
        onMouseMove={handleMouseMove}
        onMouseLeave={handleMouseLeave}
        onClick={handleClick}
      />
      {/* Tooltip */}
      {tooltipInfo && (
        <div
          style={{
            position: 'fixed',
            left: tooltipInfo.x + 12,
            top: tooltipInfo.y - 30,
            backgroundColor: '#0a0f1a',
            border: '1px solid #0f3460',
            borderRadius: '4px',
            padding: '4px 8px',
            fontSize: '11px',
            color: '#eee',
            pointerEvents: 'none',
            zIndex: 3000,
            whiteSpace: 'nowrap',
          }}
        >
          <span style={{ fontFamily: 'monospace', color: '#4ecdc4' }}>{tooltipInfo.gene}</span>
          {' '}
          <span style={{ color: '#888' }}>{tooltipInfo.value.toFixed(2)}</span>
        </div>
      )}
    </div>
  )
}


// =========================================================================
// Main modal
// =========================================================================

const ALL_PATTERNS = ['increasing', 'decreasing', 'peak', 'trough', 'complex'] as const

export default function LineAssociationModal() {
  const {
    isLineAssociationModalOpen,
    lineAssociationResult,
    setLineAssociationModalOpen,
    addFolderToCategory,
  } = useStore()

  const { colorByGene } = useDataActions()

  const [viewMode, setViewMode] = useState<ViewMode>('heatmap')
  const [filterMinR2, setFilterMinR2] = useState(0)
  const [filterMinAmplitude, setFilterMinAmplitude] = useState(0)
  const [filterMaxFDR, setFilterMaxFDR] = useState(1)
  const [filterPatterns, setFilterPatterns] = useState<Set<string>>(new Set(ALL_PATTERNS))

  const handleClose = () => {
    setLineAssociationModalOpen(false)
  }

  const handleGeneSelect = (gene: string) => {
    colorByGene(gene)
  }

  // Reset filters when new results arrive
  useEffect(() => {
    if (lineAssociationResult) {
      setFilterMinR2(0)
      setFilterMinAmplitude(0)
      setFilterMaxFDR(lineAssociationResult.fdr_threshold)
      setFilterPatterns(new Set(ALL_PATTERNS))
    }
  }, [lineAssociationResult])

  const togglePattern = useCallback((pattern: string) => {
    setFilterPatterns((prev) => {
      const next = new Set(prev)
      if (next.has(pattern)) {
        next.delete(pattern)
      } else {
        next.add(pattern)
      }
      return next
    })
  }, [])

  if (!isLineAssociationModalOpen || !lineAssociationResult) {
    return null
  }

  const { n_cells, n_significant, line_name, test_variable, fdr_threshold, diagnostics, modules, n_lines, lines_used } = lineAssociationResult
  const hasModules = modules && modules.length > 0

  // Apply client-side filters to modules
  const filteredModules: LineAssociationModule[] = hasModules
    ? modules
        .filter((m) => filterPatterns.has(m.pattern))
        .map((m) => ({
          ...m,
          genes: m.genes.filter(
            (g) =>
              g.r_squared >= filterMinR2 &&
              g.amplitude >= filterMinAmplitude &&
              g.fdr <= filterMaxFDR
          ),
        }))
        .map((m) => ({ ...m, n_genes: m.genes.length }))
        .filter((m) => m.genes.length > 0)
    : []

  // Apply the same filters to positive/negative lists (used when clustering is off).
  const geneFilterPredicate = (g: LineAssociationGene) =>
    g.r_squared >= filterMinR2 &&
    g.amplitude >= filterMinAmplitude &&
    g.fdr <= filterMaxFDR
  const filteredPositive = lineAssociationResult.positive.filter(geneFilterPredicate)
  const filteredNegative = lineAssociationResult.negative.filter(geneFilterPredicate)

  const totalModuleGenes = filteredModules.reduce((sum, m) => sum + m.n_genes, 0)
  const unfilteredModuleTotal = hasModules ? modules.reduce((sum, m) => sum + m.genes.length, 0) : 0
  const directionTotal = filteredPositive.length + filteredNegative.length
  const unfilteredDirectionTotal = lineAssociationResult.positive.length + lineAssociationResult.negative.length

  // Genes that "Add to Gene Sets" would save, and the counts for UI state.
  const totalAddableGenes = hasModules ? totalModuleGenes : directionTotal
  const unfilteredTotal = hasModules ? unfilteredModuleTotal : unfilteredDirectionTotal
  const isFiltered = totalAddableGenes !== unfilteredTotal
  const hasProfiles = filteredModules.length > 0 && filteredModules.some((m) => m.genes.some((g) => g.profile && g.profile.length > 0))

  const handleAddToGeneSets = () => {
    const lineName = lineAssociationResult.line_name
    const timestamp = new Date().toLocaleString('en-US', {
      month: 'short', day: 'numeric', hour: '2-digit', minute: '2-digit',
    })
    const folderName = `${lineName} ${timestamp}`

    const geneSets: { name: string; genes: string[] }[] = []

    if (hasModules) {
      for (const mod of filteredModules) {
        const genes = mod.genes.map((g) => g.gene)
        if (genes.length > 0) {
          const label = `${mod.pattern.charAt(0).toUpperCase() + mod.pattern.slice(1)} (${genes.length})`
          geneSets.push({ name: label, genes })
        }
      }
    } else {
      // When clustering is off, save a single combined set of all passing genes
      // (both directions). Deduplicate while preserving order.
      const seen = new Set<string>()
      const allGenes: string[] = []
      for (const g of filteredPositive) {
        if (!seen.has(g.gene)) { seen.add(g.gene); allGenes.push(g.gene) }
      }
      for (const g of filteredNegative) {
        if (!seen.has(g.gene)) { seen.add(g.gene); allGenes.push(g.gene) }
      }
      if (allGenes.length > 0) {
        geneSets.push({ name: `Associated genes (${allGenes.length})`, genes: allGenes })
      }
    }

    if (geneSets.length > 0) {
      addFolderToCategory('line_association', folderName, geneSets)
    }

    handleClose()
  }

  const handleDownloadCsv = () => {
    const rows = lineAssociationResult.all_genes ?? []
    if (rows.length === 0) return

    // RFC 4180: wrap each field in quotes to be robust to commas in gene names.
    const quote = (v: string) => `"${v.replace(/"/g, '""')}"`
    const header = ['gene', 'f_stat', 'pval', 'fdr', 'r_squared', 'amplitude', 'direction']
    const lines: string[] = [header.join(',')]
    for (const r of rows) {
      lines.push([
        quote(r.gene),
        r.f_stat,
        r.pval,
        r.fdr,
        r.r_squared,
        r.amplitude,
        r.direction,
      ].join(','))
    }
    const csv = lines.join('\n') + '\n'

    const safeLineName = (lineAssociationResult.line_name || 'line').replace(/[^A-Za-z0-9_.-]+/g, '_')
    const filename = `${safeLineName}_line_association.csv`

    const blob = new Blob([csv], { type: 'text/csv;charset=utf-8' })
    const url = URL.createObjectURL(blob)
    const a = document.createElement('a')
    a.href = url
    a.download = filename
    document.body.appendChild(a)
    a.click()
    document.body.removeChild(a)
    URL.revokeObjectURL(url)
  }

  return (
    <div style={styles.overlay} onClick={handleClose}>
      <div style={styles.modal} onClick={(e) => e.stopPropagation()}>
        <div style={styles.header}>
          <h2 style={styles.title}>
            Line Association: {lines_used && lines_used.length > 1 ? lines_used.join(' + ') : line_name}
            {test_variable === 'distance' && (
              <span style={{ fontSize: '12px', color: '#888', fontWeight: 400, marginLeft: '8px' }}>
                (distance from line)
              </span>
            )}
          </h2>
          <div style={{ display: 'flex', alignItems: 'center', gap: '12px' }}>
            {/* View mode toggle */}
            {hasProfiles && (
              <div style={{
                display: 'flex',
                backgroundColor: '#0a0f1a',
                borderRadius: '4px',
                overflow: 'hidden',
              }}>
                <button
                  onClick={() => setViewMode('heatmap')}
                  style={{
                    padding: '4px 10px',
                    fontSize: '11px',
                    border: 'none',
                    cursor: 'pointer',
                    backgroundColor: viewMode === 'heatmap' ? '#0f3460' : 'transparent',
                    color: viewMode === 'heatmap' ? '#eee' : '#666',
                    fontWeight: viewMode === 'heatmap' ? 600 : 400,
                  }}
                >
                  Heatmap
                </button>
                <button
                  onClick={() => setViewMode('list')}
                  style={{
                    padding: '4px 10px',
                    fontSize: '11px',
                    border: 'none',
                    cursor: 'pointer',
                    backgroundColor: viewMode === 'list' ? '#0f3460' : 'transparent',
                    color: viewMode === 'list' ? '#eee' : '#666',
                    fontWeight: viewMode === 'list' ? 600 : 400,
                  }}
                >
                  List
                </button>
              </div>
            )}
            <button style={styles.closeButton} onClick={handleClose}>
              &times;
            </button>
          </div>
        </div>

        <div style={styles.content}>
          <div style={styles.summary}>
            <div style={styles.summaryItem}>
              <div style={styles.summaryLabel}>Cells Tested</div>
              <div style={styles.summaryValue}>{n_cells.toLocaleString()}</div>
            </div>
            {n_lines != null && n_lines > 1 && (
              <div style={styles.summaryItem}>
                <div style={styles.summaryLabel}>Lines</div>
                <div style={styles.summaryValue}>{n_lines}</div>
              </div>
            )}
            <div style={styles.summaryItem}>
              <div style={styles.summaryLabel}>Significant Genes</div>
              <div style={styles.summaryValue}>{n_significant.toLocaleString()}</div>
            </div>
            <div style={styles.summaryItem}>
              <div style={styles.summaryLabel}>FDR Threshold</div>
              <div style={styles.summaryValue}>{fdr_threshold}</div>
            </div>
            {hasModules && (
              <div style={styles.summaryItem}>
                <div style={styles.summaryLabel}>Modules</div>
                <div style={styles.summaryValue}>{modules.length}</div>
              </div>
            )}
          </div>

          {/* Diagnostics section */}
          {diagnostics && (
            <div style={{
              backgroundColor: '#0a0f1a',
              borderRadius: '6px',
              padding: '10px 14px',
              marginBottom: '16px',
              fontSize: '11px',
              color: '#888',
            }}>
              <div style={{ marginBottom: '6px', color: '#aaa', fontWeight: 500 }}>Diagnostics</div>
              <div style={{ display: 'grid', gridTemplateColumns: 'repeat(3, 1fr)', gap: '8px' }}>
                <div>
                  <span style={{ color: '#666' }}>Genes tested: </span>
                  <span style={{ color: '#ddd' }}>{diagnostics.n_genes_tested.toLocaleString()}</span>
                </div>
                <div>
                  <span style={{ color: '#666' }}>p {'<'} 0.05 (raw): </span>
                  <span style={{ color: '#ddd' }}>{diagnostics.n_pval_below_05.toLocaleString()}</span>
                </div>
                <div>
                  <span style={{ color: '#666' }}>p {'<'} 0.01 (raw): </span>
                  <span style={{ color: '#ddd' }}>{diagnostics.n_pval_below_01.toLocaleString()}</span>
                </div>
                <div>
                  <span style={{ color: '#666' }}>Position range: </span>
                  <span style={{ color: '#ddd' }}>[{diagnostics.position_range[0].toFixed(2)}, {diagnostics.position_range[1].toFixed(2)}]</span>
                </div>
                <div>
                  <span style={{ color: '#666' }}>Position std: </span>
                  <span style={{ color: '#ddd' }}>{diagnostics.position_std.toFixed(3)}</span>
                </div>
                <div>
                  <span style={{ color: '#666' }}>Spline df: </span>
                  <span style={{ color: '#ddd' }}>{diagnostics.spline_df}</span>
                </div>
                <div>
                  <span style={{ color: '#666' }}>Expr range: </span>
                  <span style={{ color: '#ddd' }}>[{diagnostics.expression_range[0].toFixed(1)}, {diagnostics.expression_range[1].toFixed(1)}]</span>
                </div>
                <div>
                  <span style={{ color: '#666' }}>Expr mean: </span>
                  <span style={{ color: '#ddd' }}>{diagnostics.expression_mean.toFixed(2)}</span>
                </div>
                <div>
                  <span style={{ color: '#666' }}>Zero genes: </span>
                  <span style={{ color: '#ddd' }}>{diagnostics.n_zero_genes.toLocaleString()}</span>
                </div>
              </div>
              <div style={{ marginTop: '6px', color: '#666', fontStyle: 'italic' }}>
                Uses current expression values (user-preprocessed)
              </div>
            </div>
          )}

          {/* Filter controls */}
          {(hasModules || unfilteredDirectionTotal > 0) && (
            <div style={{
              backgroundColor: '#0a0f1a',
              borderRadius: '6px',
              padding: '10px 14px',
              marginBottom: '16px',
            }}>
              <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', marginBottom: '8px' }}>
                <span style={{ fontSize: '11px', color: '#aaa', fontWeight: 500 }}>Filters</span>
                {isFiltered && (
                  <span style={{ fontSize: '11px', color: '#4ecdc4' }}>
                    Showing {totalAddableGenes} of {unfilteredTotal} genes
                  </span>
                )}
              </div>

              <div style={{ display: 'flex', flexWrap: 'wrap', gap: '12px', alignItems: 'center', fontSize: '11px', color: '#ccc' }}>
                <div style={{ display: 'flex', alignItems: 'center', gap: '4px' }}>
                  <span style={{ color: '#888' }}>Min R²</span>
                  <input
                    type="number"
                    min="0"
                    max="1"
                    step="0.05"
                    value={filterMinR2}
                    onChange={(e) => setFilterMinR2(Math.max(0, Math.min(1, parseFloat(e.target.value) || 0)))}
                    style={{
                      width: '52px',
                      padding: '3px 5px',
                      fontSize: '11px',
                      backgroundColor: '#0f3460',
                      color: '#eee',
                      border: '1px solid #1a1a2e',
                      borderRadius: '4px',
                      textAlign: 'center' as const,
                    }}
                    title="Minimum R-squared (variance explained)"
                  />
                </div>
                <div style={{ display: 'flex', alignItems: 'center', gap: '4px' }}>
                  <span style={{ color: '#888' }}>Min amplitude</span>
                  <input
                    type="number"
                    min="0"
                    step="0.1"
                    value={filterMinAmplitude}
                    onChange={(e) => setFilterMinAmplitude(Math.max(0, parseFloat(e.target.value) || 0))}
                    style={{
                      width: '52px',
                      padding: '3px 5px',
                      fontSize: '11px',
                      backgroundColor: '#0f3460',
                      color: '#eee',
                      border: '1px solid #1a1a2e',
                      borderRadius: '4px',
                      textAlign: 'center' as const,
                    }}
                    title="Minimum amplitude (predicted expression range)"
                  />
                </div>
                <div style={{ display: 'flex', alignItems: 'center', gap: '4px' }}>
                  <span style={{ color: '#888' }}>Max FDR</span>
                  <input
                    type="number"
                    min="0.0001"
                    max={fdr_threshold}
                    step="0.005"
                    value={filterMaxFDR}
                    onChange={(e) => setFilterMaxFDR(Math.max(0.0001, Math.min(fdr_threshold, parseFloat(e.target.value) || fdr_threshold)))}
                    style={{
                      width: '62px',
                      padding: '3px 5px',
                      fontSize: '11px',
                      backgroundColor: '#0f3460',
                      color: '#eee',
                      border: '1px solid #1a1a2e',
                      borderRadius: '4px',
                      textAlign: 'center' as const,
                    }}
                    title={`Maximum FDR (up to the analysis threshold of ${fdr_threshold})`}
                  />
                </div>
              </div>

              {/* Pattern toggles — only meaningful when clustering produced modules */}
              {hasModules && (
              <div style={{ display: 'flex', gap: '4px', marginTop: '8px', flexWrap: 'wrap' }}>
                {ALL_PATTERNS.map((pattern) => {
                  const active = filterPatterns.has(pattern)
                  const color = PATTERN_COLORS[pattern] || '#888'
                  const icon = PATTERN_ICONS[pattern] || ''
                  return (
                    <button
                      key={pattern}
                      onClick={() => togglePattern(pattern)}
                      style={{
                        padding: '2px 8px',
                        fontSize: '10px',
                        borderRadius: '10px',
                        cursor: 'pointer',
                        border: `1px solid ${active ? color + '88' : '#1a1a2e'}`,
                        backgroundColor: active ? color + '22' : '#0f3460',
                        color: active ? color : '#555',
                        fontWeight: active ? 600 : 400,
                        textTransform: 'capitalize',
                      }}
                    >
                      {icon} {pattern}
                    </button>
                  )
                })}
              </div>
              )}
            </div>
          )}

          {/* Heatmap or List view */}
          {viewMode === 'heatmap' && hasProfiles ? (
            <ProfileHeatmap
              modules={filteredModules}
              onGeneSelect={handleGeneSelect}
              testVariable={test_variable}
            />
          ) : filteredModules.length > 0 ? (
            <div>
              {filteredModules.map((mod, idx) => (
                <ModuleCard
                  key={mod.module_id}
                  module={mod}
                  onGeneSelect={handleGeneSelect}
                  defaultExpanded={idx < 3}
                />
              ))}
              {totalModuleGenes === 0 && (
                <div style={{ padding: '24px', color: '#666', textAlign: 'center' }}>
                  No genes match current filters
                </div>
              )}
            </div>
          ) : directionTotal > 0 ? (
            /* Non-clustered results: render positive and negative lists as simple cards */
            <div>
              {filteredPositive.length > 0 && (
                <DirectionCard
                  title="Increasing"
                  subtitle={test_variable === 'distance' ? 'Expression rises with distance from line' : 'Expression rises along the line'}
                  genes={filteredPositive}
                  color="#4ecdc4"
                  onGeneSelect={handleGeneSelect}
                />
              )}
              {filteredNegative.length > 0 && (
                <DirectionCard
                  title="Decreasing"
                  subtitle={test_variable === 'distance' ? 'Expression falls with distance from line' : 'Expression falls along the line'}
                  genes={filteredNegative}
                  color="#ff7f7f"
                  onGeneSelect={handleGeneSelect}
                />
              )}
            </div>
          ) : (
            <div style={{ padding: '24px', color: '#666', textAlign: 'center' }}>
              {isFiltered ? 'No genes match current filters' : 'No significant genes found'}
            </div>
          )}
        </div>

        <div style={styles.footer}>
          <div style={{ fontSize: '11px', color: '#666' }}>
            Click a gene to color cells by its expression
          </div>
          <div style={{ display: 'flex', gap: '8px' }}>
            <button
              style={{ ...styles.button, ...styles.secondaryButton }}
              onClick={handleClose}
            >
              Close
            </button>
            {(() => {
              const allRows = lineAssociationResult.all_genes ?? []
              const hasAllGenes = allRows.length > 0
              return (
                <button
                  style={{ ...styles.button, ...styles.secondaryButton }}
                  onClick={handleDownloadCsv}
                  disabled={!hasAllGenes}
                  title={hasAllGenes
                    ? `Download per-gene stats for all ${allRows.length.toLocaleString()} tested genes as CSV (ranked list for GSEA / external analysis)`
                    : 'No per-gene table available for this result'}
                >
                  Download CSV
                </button>
              )
            })()}
            <button
              style={{ ...styles.button, ...styles.primaryButton }}
              onClick={handleAddToGeneSets}
              disabled={totalAddableGenes === 0}
              title={totalAddableGenes === 0
                ? 'No genes to add (filters may be too strict)'
                : `Add ${totalAddableGenes} gene${totalAddableGenes === 1 ? '' : 's'} to the Line Association category`}
            >
              Add to Gene Sets
            </button>
          </div>
        </div>
      </div>
    </div>
  )
}
