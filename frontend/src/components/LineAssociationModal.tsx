import { useState } from 'react'
import { useStore, LineAssociationGene, LineAssociationModule } from '../store'

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
    console.log('Selected gene:', gene)
  }

  const handleAddToGeneSets = () => {
    if (!lineAssociationResult) return

    const lineName = lineAssociationResult.line_name

    if (lineAssociationResult.modules && lineAssociationResult.modules.length > 0) {
      // Add one gene set per module
      for (const mod of lineAssociationResult.modules) {
        const genes = mod.genes.map((g) => g.gene)
        if (genes.length > 0) {
          const label = `${lineName} - ${mod.pattern.charAt(0).toUpperCase() + mod.pattern.slice(1)}`
          addGeneSetToCategory('manual', label, genes)
        }
      }
    } else {
      // Fallback to positive/negative (backward compatibility)
      if (lineAssociationResult.positive.length > 0) {
        const posGenes = lineAssociationResult.positive.map((g) => g.gene)
        addGeneSetToCategory('manual', `${lineName} - Increasing`, posGenes)
      }
      if (lineAssociationResult.negative.length > 0) {
        const negGenes = lineAssociationResult.negative.map((g) => g.gene)
        addGeneSetToCategory('manual', `${lineName} - Decreasing`, negGenes)
      }
    }

    handleClose()
  }

  if (!isLineAssociationModalOpen || !lineAssociationResult) {
    return null
  }

  const { n_cells, n_significant, line_name, fdr_threshold, diagnostics, modules } = lineAssociationResult
  const hasModules = modules && modules.length > 0
  const totalModuleGenes = hasModules ? modules.reduce((sum, m) => sum + m.n_genes, 0) : 0

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

          {/* Module-based display */}
          {hasModules ? (
            <div>
              {modules.map((mod, idx) => (
                <ModuleCard
                  key={mod.module_id}
                  module={mod}
                  onGeneSelect={handleGeneSelect}
                  defaultExpanded={idx < 3}
                />
              ))}
              {totalModuleGenes === 0 && (
                <div style={{ padding: '24px', color: '#666', textAlign: 'center' }}>
                  No significant genes found
                </div>
              )}
            </div>
          ) : (
            <div style={{ padding: '24px', color: '#666', textAlign: 'center' }}>
              No significant genes found
            </div>
          )}
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
              disabled={!hasModules || totalModuleGenes === 0}
            >
              Add to Gene Sets
            </button>
          </div>
        </div>
      </div>
    </div>
  )
}
