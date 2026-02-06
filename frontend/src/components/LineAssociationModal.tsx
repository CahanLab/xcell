import { useState } from 'react'
import { useStore, LineAssociationGene } from '../store'

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
  columnsContainer: {
    display: 'grid',
    gridTemplateColumns: '1fr 1fr',
    gap: '20px',
  },
  column: {
    backgroundColor: '#0a0f1a',
    borderRadius: '8px',
    overflow: 'hidden',
  },
  columnHeader: {
    padding: '12px 16px',
    borderBottom: '1px solid #0f3460',
    display: 'flex',
    alignItems: 'center',
    gap: '8px',
  },
  columnTitle: {
    fontSize: '13px',
    fontWeight: 600,
    color: '#eee',
  },
  columnCount: {
    fontSize: '11px',
    color: '#888',
    backgroundColor: '#0f3460',
    padding: '2px 8px',
    borderRadius: '10px',
  },
  geneList: {
    maxHeight: '400px',
    overflow: 'auto',
  },
  geneRow: {
    display: 'flex',
    alignItems: 'center',
    padding: '8px 16px',
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
  arrowUp: {
    color: '#4ecdc4',
    fontSize: '14px',
    marginRight: '4px',
  },
  arrowDown: {
    color: '#e94560',
    fontSize: '14px',
    marginRight: '4px',
  },
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

export default function LineAssociationModal() {
  const {
    isLineAssociationModalOpen,
    lineAssociationResult,
    setLineAssociationModalOpen,
    addGeneSetToCategory,
  } = useStore()

  const handleClose = () => {
    setLineAssociationModalOpen(false)
  }

  const handleGeneSelect = (gene: string) => {
    // Could trigger expression coloring here
    console.log('Selected gene:', gene)
  }

  const handleAddToGeneSets = () => {
    if (!lineAssociationResult) return

    const lineName = lineAssociationResult.line_name

    // Add positive genes (increasing along line)
    if (lineAssociationResult.positive.length > 0) {
      const posGenes = lineAssociationResult.positive.map((g) => g.gene)
      addGeneSetToCategory('manual', `${lineName} - Increasing`, posGenes)
    }

    // Add negative genes (decreasing along line)
    if (lineAssociationResult.negative.length > 0) {
      const negGenes = lineAssociationResult.negative.map((g) => g.gene)
      addGeneSetToCategory('manual', `${lineName} - Decreasing`, negGenes)
    }

    handleClose()
  }

  if (!isLineAssociationModalOpen || !lineAssociationResult) {
    return null
  }

  const { positive, negative, n_cells, n_significant, line_name, fdr_threshold, diagnostics } = lineAssociationResult

  return (
    <div style={styles.overlay} onClick={handleClose}>
      <div style={styles.modal} onClick={(e) => e.stopPropagation()}>
        <div style={styles.header}>
          <h2 style={styles.title}>Line Association: {line_name}</h2>
          <button style={styles.closeButton} onClick={handleClose}>
            &times;
          </button>
        </div>

        <div style={styles.content}>
          <div style={styles.summary}>
            <div style={styles.summaryItem}>
              <div style={styles.summaryLabel}>Cells Tested</div>
              <div style={styles.summaryValue}>{n_cells.toLocaleString()}</div>
            </div>
            <div style={styles.summaryItem}>
              <div style={styles.summaryLabel}>Significant Genes</div>
              <div style={styles.summaryValue}>{n_significant.toLocaleString()}</div>
            </div>
            <div style={styles.summaryItem}>
              <div style={styles.summaryLabel}>FDR Threshold</div>
              <div style={styles.summaryValue}>{fdr_threshold}</div>
            </div>
            <div style={styles.summaryItem}>
              <div style={styles.summaryLabel}>Increasing</div>
              <div style={styles.summaryValue}>{lineAssociationResult.n_positive}</div>
            </div>
            <div style={styles.summaryItem}>
              <div style={styles.summaryLabel}>Decreasing</div>
              <div style={styles.summaryValue}>{lineAssociationResult.n_negative}</div>
            </div>
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
                {diagnostics.used_log1p ? 'Applied normalize_total + log1p' : 'Used raw expression values'}
              </div>
            </div>
          )}

          <div style={styles.columnsContainer}>
            {/* Positive (increasing) genes */}
            <div style={styles.column}>
              <div style={styles.columnHeader}>
                <span style={styles.arrowUp}>&#8593;</span>
                <span style={styles.columnTitle}>Increasing Along Line</span>
                <span style={styles.columnCount}>{positive.length} shown</span>
              </div>
              <div style={styles.geneList}>
                {positive.map((gene) => (
                  <GeneRow
                    key={gene.gene}
                    gene={gene}
                    onSelect={handleGeneSelect}
                  />
                ))}
                {positive.length === 0 && (
                  <div style={{ padding: '16px', color: '#666', textAlign: 'center' }}>
                    No significant genes
                  </div>
                )}
              </div>
            </div>

            {/* Negative (decreasing) genes */}
            <div style={styles.column}>
              <div style={styles.columnHeader}>
                <span style={styles.arrowDown}>&#8595;</span>
                <span style={styles.columnTitle}>Decreasing Along Line</span>
                <span style={styles.columnCount}>{negative.length} shown</span>
              </div>
              <div style={styles.geneList}>
                {negative.map((gene) => (
                  <GeneRow
                    key={gene.gene}
                    gene={gene}
                    onSelect={handleGeneSelect}
                  />
                ))}
                {negative.length === 0 && (
                  <div style={{ padding: '16px', color: '#666', textAlign: 'center' }}>
                    No significant genes
                  </div>
                )}
              </div>
            </div>
          </div>
        </div>

        <div style={styles.footer}>
          <div style={{ fontSize: '11px', color: '#666' }}>
            Click a gene to view its expression pattern
          </div>
          <div style={{ display: 'flex', gap: '8px' }}>
            <button
              style={{ ...styles.button, ...styles.secondaryButton }}
              onClick={handleClose}
            >
              Close
            </button>
            <button
              style={{ ...styles.button, ...styles.primaryButton }}
              onClick={handleAddToGeneSets}
              disabled={positive.length === 0 && negative.length === 0}
            >
              Add to Gene Sets
            </button>
          </div>
        </div>
      </div>
    </div>
  )
}
