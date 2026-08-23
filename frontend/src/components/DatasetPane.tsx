import type { ComponentProps } from 'react'
import { useStore, type DatasetSlot } from '../store'
import EmbeddingPlot from './EmbeddingPlot'
import { CategoryLegend, ContinuousLegend, ExpressionLegend, BivariateLegend } from './PlotLegends'
import { umPerUnitForSlot, expressionLegendTitle } from '../lib/datasetPanes'

// One dataset's plot, with the chrome that belongs to that dataset rather than
// to the app: its own embedding picker, its own legend, its own scale bar.
// Everything it shows is read from `datasets[slot]`, never from the flat
// top-level mirrors — those track the *active* slot, so a pane that read them
// would draw the active dataset's colors onto its neighbour's cells.

type PlotProps = ComponentProps<typeof EmbeddingPlot>

const styles = {
  loading: {
    position: 'absolute' as const,
    top: '50%',
    left: '50%',
    transform: 'translate(-50%, -50%)',
    fontSize: '18px',
    color: '#aaa',
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
  slotLabel: {
    position: 'absolute' as const,
    top: 6,
    left: 8,
    // Filenames are long and the pane's own buttons sit at the top right;
    // without a ceiling the two run into each other in a narrow pane.
    maxWidth: '45%',
    overflow: 'hidden',
    textOverflow: 'ellipsis',
    whiteSpace: 'nowrap' as const,
    fontSize: '11px',
    color: '#888',
    pointerEvents: 'none' as const,
  },
}

export default function DatasetPane({
  slot,
  label,
  onSelectionComplete,
  onLineDrawn,
  onTransformEmbedding,
  onTransformEmbeddingSubset,
  onSelectEmbedding,
  onToggleBivariateSort,
}: {
  slot: DatasetSlot
  label: string
  onSelectionComplete: PlotProps['onSelectionComplete']
  onLineDrawn: PlotProps['onLineDrawn']
  onTransformEmbedding: PlotProps['onTransformEmbedding']
  onTransformEmbeddingSubset: PlotProps['onTransformEmbeddingSubset']
  onSelectEmbedding: (name: string) => void
  onToggleBivariateSort: () => void
}) {
  const ds = useStore((s) => s.datasets[slot])
  const activeSlot = useStore((s) => s.activeSlot)
  const setActiveSlot = useStore((s) => s.setActiveSlot)
  const interactionMode = useStore((s) => s.interactionMode)

  // The slot was unloaded out from under us. No pane, rather than an empty one.
  if (!ds) return null

  const isActive = activeSlot === slot
  // Acting on a pane makes it the active dataset first, so the panels down the
  // left and the modals all follow the plot the user just touched.
  const activate = () => { if (!isActive) setActiveSlot(slot) }

  return (
    <div
      style={{
        // The grid cell decides the size; the pane fills whatever it is given.
        width: '100%',
        height: '100%',
        position: 'relative',
        overflow: 'hidden',
        outline: isActive ? '2px solid #e94560' : '2px solid transparent',
        outlineOffset: '-2px',
      }}
      onPointerDown={activate}
    >
      {ds.embedding ? (
        <>
          <EmbeddingPlot
            slot={slot}
            umPerUnit={umPerUnitForSlot(ds)}
            embedding={ds.embedding}
            colorBy={ds.colorBy}
            expressionData={ds.expressionData}
            bivariateData={ds.bivariateData}
            highlightLayers={ds.highlightLayers}
            colorMode={ds.colorMode}
            interactionMode={interactionMode}
            selectedCellIndices={ds.selectedCellIndices}
            onSelectionComplete={onSelectionComplete}
            onLineDrawn={onLineDrawn}
            onTransformEmbedding={onTransformEmbedding}
            onTransformEmbeddingSubset={onTransformEmbeddingSubset}
          />
          {/* Per-plot embedding selector */}
          {ds.schema && ds.schema.embeddings.length > 1 && (
            <div style={{ ...styles.embeddingSelector }}>
              <span style={styles.embeddingLabel}>Embedding:</span>
              <select
                style={styles.embeddingSelect}
                value={ds.selectedEmbedding || ''}
                onChange={(e) => {
                  activate()
                  onSelectEmbedding(e.target.value)
                }}
              >
                {ds.schema.embeddings.map((emb) => (
                  <option key={emb} value={emb}>{emb}</option>
                ))}
              </select>
            </div>
          )}
          {/* Per-plot legend */}
          {ds.colorMode === 'metadata' && ds.colorBy?.dtype === 'category' && (
            <CategoryLegend colorBy={ds.colorBy} panelId={`${slot}-category`} />
          )}
          {ds.colorMode === 'metadata' && ds.colorBy?.dtype === 'numeric' && (
            <ContinuousLegend
              panelId={`${slot}-continuous`}
              colorScale={ds.displayPreferences.colorScale}
              name={ds.colorBy.name}
              min={Math.min(...(ds.colorBy.values.filter((v) => v !== null) as number[]))}
              max={Math.max(...(ds.colorBy.values.filter((v) => v !== null) as number[]))}
            />
          )}
          {ds.colorMode === 'expression' && ds.expressionData && (
            <ExpressionLegend
              panelId={`${slot}-expression`}
              title={expressionLegendTitle(ds.selectedGenes, ds.selectedGeneSetName)}
              transform={ds.expressionData.transform}
              min={ds.expressionData.min}
              max={ds.expressionData.max}
              colorScale={ds.displayPreferences.colorScale}
            />
          )}
          {ds.colorMode === 'bivariate' && ds.bivariateData && (
            <BivariateLegend
              panelId={`${slot}-bivariate`}
              bivariateData={ds.bivariateData}
              colormap={ds.displayPreferences.bivariateColormap}
              sortReversed={ds.bivariateSortReversed}
              onToggleSort={onToggleBivariateSort}
            />
          )}
        </>
      ) : (
        <div style={{ ...styles.loading, position: 'absolute' }}>No embedding loaded</div>
      )}
      <div style={styles.slotLabel}>{label}</div>
    </div>
  )
}
