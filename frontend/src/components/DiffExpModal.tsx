import { useCallback, useState } from 'react'
import { useStore, DiffExpGene } from '../store'
import { useDiffExp } from '../hooks/useData'

const styles = {
  backdrop: {
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
    boxShadow: '0 8px 32px rgba(0, 0, 0, 0.4)',
    width: '700px',
    maxWidth: '90vw',
    maxHeight: '80vh',
    display: 'flex',
    flexDirection: 'column' as const,
    overflow: 'hidden',
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
    lineHeight: 1,
  },
  content: {
    padding: '20px',
    overflowY: 'auto' as const,
    flex: 1,
  },
  groupInfo: {
    display: 'flex',
    gap: '20px',
    marginBottom: '20px',
  },
  groupBox: {
    flex: 1,
    padding: '12px 16px',
    backgroundColor: '#0f3460',
    borderRadius: '6px',
  },
  groupLabel: {
    fontSize: '12px',
    color: '#aaa',
    marginBottom: '4px',
  },
  groupName: {
    fontSize: '14px',
    color: '#eee',
    fontWeight: 500,
  },
  groupCount: {
    fontSize: '12px',
    color: '#888',
    marginTop: '2px',
  },
  resultsContainer: {
    display: 'flex',
    gap: '20px',
  },
  resultsColumn: {
    flex: 1,
  },
  columnHeader: {
    fontSize: '13px',
    fontWeight: 600,
    color: '#e94560',
    marginBottom: '12px',
    paddingBottom: '8px',
    borderBottom: '1px solid #0f3460',
  },
  geneRow: {
    display: 'flex',
    justifyContent: 'space-between',
    alignItems: 'center',
    padding: '8px 10px',
    backgroundColor: '#0a0f1a',
    borderRadius: '4px',
    marginBottom: '6px',
  },
  geneName: {
    fontSize: '13px',
    color: '#eee',
    fontWeight: 500,
  },
  geneStats: {
    display: 'flex',
    gap: '12px',
    fontSize: '11px',
    color: '#888',
  },
  statLabel: {
    color: '#666',
  },
  statValue: {
    color: '#aaa',
  },
  positiveFC: {
    color: '#4ecdc4',
  },
  negativeFC: {
    color: '#ff6b6b',
  },
  footer: {
    padding: '16px 20px',
    borderTop: '1px solid #0f3460',
    display: 'flex',
    justifyContent: 'flex-end',
    gap: '12px',
  },
  button: {
    padding: '8px 16px',
    fontSize: '13px',
    borderRadius: '4px',
    cursor: 'pointer',
    border: 'none',
  },
  primaryButton: {
    backgroundColor: '#e94560',
    color: '#fff',
  },
  secondaryButton: {
    backgroundColor: '#0f3460',
    color: '#aaa',
    border: '1px solid #1a1a2e',
  },
  emptyState: {
    textAlign: 'center' as const,
    padding: '40px 20px',
    color: '#666',
    fontSize: '14px',
  },
  loadingState: {
    textAlign: 'center' as const,
    padding: '40px 20px',
    color: '#aaa',
    fontSize: '14px',
  },
  settingRow: {
    display: 'flex',
    alignItems: 'center',
    gap: '12px',
    marginBottom: '16px',
    padding: '8px 12px',
    backgroundColor: '#0f3460',
    borderRadius: '6px',
  },
  settingLabel: {
    fontSize: '13px',
    color: '#aaa',
  },
  settingInput: {
    width: '70px',
    padding: '4px 8px',
    fontSize: '13px',
    backgroundColor: '#0a0f1a',
    color: '#eee',
    border: '1px solid #1a1a2e',
    borderRadius: '4px',
    textAlign: 'center' as const,
  },
}

function formatPValue(pval: number): string {
  if (pval < 0.001) {
    return pval.toExponential(2)
  }
  return pval.toFixed(4)
}

function formatTimestamp(): string {
  const now = new Date()
  const months = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
  const month = months[now.getMonth()]
  const day = String(now.getDate()).padStart(2, '0')
  const hours = String(now.getHours()).padStart(2, '0')
  const mins = String(now.getMinutes()).padStart(2, '0')
  return `${month}${day} ${hours}${mins}`
}

function GeneList({ genes, isPositive }: { genes: DiffExpGene[]; isPositive: boolean }) {
  if (genes.length === 0) {
    return <div style={{ color: '#666', fontSize: '13px', padding: '8px' }}>No significant genes</div>
  }

  return (
    <>
      {genes.map((gene) => (
        <div key={gene.gene} style={styles.geneRow}>
          <span style={styles.geneName}>{gene.gene}</span>
          <div style={styles.geneStats}>
            <span>
              <span style={styles.statLabel}>log2FC: </span>
              <span style={isPositive ? styles.positiveFC : styles.negativeFC}>
                {gene.log2fc.toFixed(2)}
              </span>
            </span>
            <span>
              <span style={styles.statLabel}>padj: </span>
              <span style={styles.statValue}>{formatPValue(gene.pval_adj)}</span>
            </span>
          </div>
        </div>
      ))}
    </>
  )
}

export default function DiffExpModal() {
  const {
    isDiffExpModalOpen,
    setDiffExpModalOpen,
    comparison,
    diffExpResult,
    isDiffExpLoading,
    addGeneSet,
    clearComparison,
  } = useStore()
  const { runComparison } = useDiffExp()
  const [topN, setTopN] = useState(25)

  const handleClose = useCallback(() => {
    setDiffExpModalOpen(false)
  }, [setDiffExpModalOpen])

  const handleAddToGeneSets = useCallback(() => {
    if (!diffExpResult) return

    const timestamp = formatTimestamp()

    // Add upregulated in group1 - name format: "Feb02 1430 grp1 MS4A1"
    if (diffExpResult.positive.length > 0) {
      const genes = diffExpResult.positive.map((g) => g.gene)
      const topGene = diffExpResult.positive[0].gene
      addGeneSet(`${timestamp} grp1 ${topGene}`, genes)
    }

    // Add upregulated in group2 - name format: "Feb02 1430 grp2 CD3D"
    if (diffExpResult.negative.length > 0) {
      const genes = diffExpResult.negative.map((g) => g.gene)
      const topGene = diffExpResult.negative[0].gene
      addGeneSet(`${timestamp} grp2 ${topGene}`, genes)
    }

    handleClose()
  }, [diffExpResult, addGeneSet, handleClose])

  const handleClearAndClose = useCallback(() => {
    clearComparison()
    handleClose()
  }, [clearComparison, handleClose])

  const handleRunComparison = useCallback(async () => {
    try {
      await runComparison(topN)
    } catch (err) {
      alert(`Differential expression failed: ${(err as Error).message}`)
    }
  }, [runComparison, topN])

  if (!isDiffExpModalOpen) return null

  const hasGroups = comparison.group1 && comparison.group2
  const hasResults = diffExpResult !== null

  return (
    <div style={styles.backdrop} onClick={handleClose}>
      <div style={styles.modal} onClick={(e) => e.stopPropagation()}>
        <div style={styles.header}>
          <h2 style={styles.title}>Differential Expression</h2>
          <button style={styles.closeButton} onClick={handleClose}>
            x
          </button>
        </div>

        <div style={styles.content}>
          {/* Group info */}
          <div style={styles.groupInfo}>
            <div style={styles.groupBox}>
              <div style={styles.groupLabel}>Group 1</div>
              <div style={styles.groupName}>
                {comparison.group1Label || 'Not set'}
              </div>
              {comparison.group1 && (
                <div style={styles.groupCount}>{comparison.group1.length} cells</div>
              )}
            </div>
            <div style={styles.groupBox}>
              <div style={styles.groupLabel}>Group 2</div>
              <div style={styles.groupName}>
                {comparison.group2Label || 'Not set'}
              </div>
              {comparison.group2 && (
                <div style={styles.groupCount}>{comparison.group2.length} cells</div>
              )}
            </div>
          </div>

          {/* Top N setting */}
          {!hasResults && (
            <div style={styles.settingRow}>
              <span style={styles.settingLabel}>Top genes per group:</span>
              <input
                type="number"
                min="1"
                max="500"
                value={topN}
                onChange={(e) => setTopN(Math.max(1, Math.min(500, parseInt(e.target.value) || 50)))}
                style={styles.settingInput}
              />
            </div>
          )}

          {/* Loading state */}
          {isDiffExpLoading && (
            <div style={styles.loadingState}>Running differential expression analysis...</div>
          )}

          {/* Results */}
          {!isDiffExpLoading && hasResults && (
            <div style={styles.resultsContainer}>
              <div style={styles.resultsColumn}>
                <div style={styles.columnHeader}>
                  Upregulated in {comparison.group1Label || 'Group 1'}
                </div>
                <GeneList genes={diffExpResult.positive} isPositive={true} />
              </div>
              <div style={styles.resultsColumn}>
                <div style={styles.columnHeader}>
                  Upregulated in {comparison.group2Label || 'Group 2'}
                </div>
                <GeneList genes={diffExpResult.negative} isPositive={false} />
              </div>
            </div>
          )}

          {/* Empty state - no groups set */}
          {!isDiffExpLoading && !hasResults && !hasGroups && (
            <div style={styles.emptyState}>
              Select two groups to compare using the lasso tool or category buttons in the Cells panel.
            </div>
          )}

          {/* Groups set but no results yet */}
          {!isDiffExpLoading && !hasResults && hasGroups && (
            <div style={styles.emptyState}>
              Click "Run Comparison" to identify differentially expressed genes.
            </div>
          )}
        </div>

        <div style={styles.footer}>
          <button
            style={{ ...styles.button, ...styles.secondaryButton }}
            onClick={handleClearAndClose}
          >
            Clear & Close
          </button>
          {hasGroups && !hasResults && (
            <button
              style={{ ...styles.button, ...styles.primaryButton }}
              onClick={handleRunComparison}
              disabled={isDiffExpLoading}
            >
              Run Comparison
            </button>
          )}
          {hasResults && (
            <button
              style={{ ...styles.button, ...styles.primaryButton }}
              onClick={handleAddToGeneSets}
            >
              Add to Gene Sets
            </button>
          )}
        </div>
      </div>
    </div>
  )
}
