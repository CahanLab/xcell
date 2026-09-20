import type { CSSProperties } from 'react'
import type { EmbeddingData } from '../store'
import type { CellSubsetInfo } from '../lib/cellSubsets'
import { cellsWithoutCoordinates, supersetEmbedding } from '../lib/embeddingCoverage'
import { MESSAGES } from '../messages'

// Shown over a plot whose embedding has no coordinates for some cells — a
// subset's UMAP or PCA. deck.gl draws nothing for those cells, so without
// this the plot is silently missing half the dataset and the only way back
// is to know to change the embedding picker. The button goes back to where
// the subset was drawn (see supersetEmbedding).

const styles: Record<string, CSSProperties> = {
  notice: {
    position: 'absolute',
    top: '12px',
    left: '20px',
    // Leave the plot's own top-right toolbar (snapshot, add to record) clear.
    maxWidth: 'min(640px, calc(100% - 200px))',
    padding: '6px 10px',
    backgroundColor: 'rgba(22, 33, 62, 0.92)',
    border: '1px solid rgba(233, 162, 59, 0.5)',
    borderRadius: '6px',
    display: 'flex',
    alignItems: 'center',
    gap: '10px',
    fontSize: '11px',
    color: '#e9a23b',
    zIndex: 5,
    pointerEvents: 'auto',
  },
  button: {
    padding: '3px 8px',
    fontSize: '11px',
    backgroundColor: '#e9a23b',
    color: '#0f1625',
    border: 'none',
    borderRadius: '4px',
    cursor: 'pointer',
    whiteSpace: 'nowrap',
  },
}

export default function CoverageNotice({ embedding, embeddings, subsets, onSelectEmbedding }: {
  embedding: EmbeddingData | null
  embeddings: readonly string[] | undefined
  subsets: readonly CellSubsetInfo[]
  onSelectEmbedding: (name: string) => void
}) {
  if (!embedding || !embeddings) return null
  const hidden = cellsWithoutCoordinates(embedding.coordinates)
  if (hidden === 0) return null
  const target = supersetEmbedding(embedding.name, embeddings, subsets)
  const text = MESSAGES.cellSubsets.hiddenCells(hidden, embedding.coordinates.length, embedding.name)
  return (
    <div style={styles.notice} title={text} data-testid="coverage-notice">
      <span style={{ flex: 1 }}>{text}</span>
      {target && (
        <button
          style={styles.button}
          onClick={(e) => { e.stopPropagation(); onSelectEmbedding(target) }}
        >
          {MESSAGES.cellSubsets.showAllOn(target)}
        </button>
      )}
    </div>
  )
}
