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

import { useState, useEffect, useCallback } from 'react'
import { useStore, HeatmapConfig } from '../store'
import { useDataActions, appendDataset, createFigure } from '../hooks/useData'
import { heatmapFigureFromConfig } from '../lib/figures'
import HeatmapConfigModal from './HeatmapConfigModal'
import ExpressionHeatmapFigure from './figures/ExpressionHeatmapFigure'
import type { ExpressionHeatmapData as HeatmapData } from '../lib/figures'

// ---------------------------------------------------------------------------
// HeatmapView (main component)
// ---------------------------------------------------------------------------

export default function HeatmapView() {
  const heatmapConfig = useStore((s) => s.heatmapConfig)
  const setHeatmapConfig = useStore((s) => s.setHeatmapConfig)
  const drawnLines = useStore((s) => s.drawnLines)
  const displayPreferences = useStore((s) => s.displayPreferences)
  const activeSlot = useStore((s) => s.activeSlot)
  const refreshFigures = useStore((s) => s.refreshFigures)
  const setActiveFigureId = useStore((s) => s.setActiveFigureId)
  const setCenterPanelView = useStore((s) => s.setCenterPanelView)
  const saveAsFigure = useCallback(async () => {
    if (!heatmapConfig) return
    try {
      // What the tab drew: only config.cellIndices scope it (never the active subset), under the display transform.
      const transform = displayPreferences.expressionTransform === 'log1p' ? 'log1p' as const : null
      const rec = await createFigure(heatmapFigureFromConfig(heatmapConfig, null, transform), activeSlot)
      refreshFigures(); setActiveFigureId(rec.id); setCenterPanelView('figures')
    } catch (e) {
      setError((e as Error).message)
    }
  }, [heatmapConfig, displayPreferences.expressionTransform, activeSlot, refreshFigures, setActiveFigureId, setCenterPanelView])
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
        await fetch(appendDataset('/api/lines'), {
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

      const response = await fetch(appendDataset('/api/heatmap/data'), {
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
          cell_indices: config.cellIndices ?? null,
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
          {data?.n_genes_hidden
            ? `${data.n_genes_hidden === 1 ? 'The selected gene is' : `All ${data.n_genes_hidden} selected genes are`} hidden by the active gene mask (Genes ⋯ → Gene mask…).`
            : 'No data to display. Check that selected gene sets contain valid genes.'}
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
        <button style={hmStyles.settingsButton} onClick={saveAsFigure} title="Keep this heatmap as a figure record (Figures tab): editable, exportable as PNG, with provenance">
          Save as figure
        </button>
        <span style={hmStyles.info}>
          {data.row_labels.length} genes &times; {data.n_bins} {data.n_bins < data.n_cells ? 'bins' : 'cells'}
          {data.n_bins < data.n_cells && ` (${data.n_cells.toLocaleString()} cells)`}
          {heatmapConfig?.cellLabel && ` · ${heatmapConfig.cellLabel}`}
          {!!data.n_genes_hidden && (
            <span
              style={{ color: '#e9a23b' }}
              title="These genes are in the chosen sets but hidden by the active gene mask (Genes ⋯ → Gene mask…)."
            >
              {` · ${data.n_genes_hidden} hidden by gene mask`}
            </span>
          )}
        </span>
        <span style={hmStyles.hint}>Click a gene row to color scatter plot</span>
      </div>

      {/* Canvas heatmap */}
      <ExpressionHeatmapFigure data={data} onGeneClick={handleGeneClick} legends="floating" />
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
}
