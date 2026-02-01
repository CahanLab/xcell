import { useState, useCallback } from 'react'
import { useStore } from '../store'
import {
  useObsSummaries,
  ObsSummary,
  CategoryValue,
  useDataActions,
  createAnnotation,
  addLabelToAnnotation,
  labelCells,
  deleteAnnotation,
  exportAnnotations,
} from '../hooks/useData'

const styles = {
  panel: {
    width: '280px',
    height: '100%',
    backgroundColor: '#16213e',
    borderRight: '1px solid #0f3460',
    display: 'flex',
    flexDirection: 'column' as const,
    overflow: 'hidden',
  },
  header: {
    padding: '12px 16px',
    borderBottom: '1px solid #0f3460',
  },
  title: {
    fontSize: '14px',
    fontWeight: 600,
    color: '#e94560',
  },
  content: {
    flex: 1,
    overflowY: 'auto' as const,
    padding: '8px 0',
  },
  section: {
    marginBottom: '8px',
  },
  sectionHeader: {
    padding: '8px 16px',
    fontSize: '11px',
    fontWeight: 600,
    color: '#888',
    textTransform: 'uppercase' as const,
    letterSpacing: '0.5px',
    display: 'flex',
    alignItems: 'center',
    cursor: 'pointer',
    userSelect: 'none' as const,
  },
  sectionToggle: {
    marginRight: '6px',
    fontSize: '10px',
  },
  column: {
    backgroundColor: 'transparent',
    marginBottom: '1px',
  },
  columnHeader: {
    display: 'flex',
    alignItems: 'center',
    padding: '8px 16px',
    cursor: 'pointer',
    transition: 'background-color 0.1s',
  },
  columnHeaderHover: {
    backgroundColor: '#0f3460',
  },
  columnHeaderActive: {
    backgroundColor: '#0f3460',
    borderLeft: '3px solid #e94560',
    paddingLeft: '13px',
  },
  expandIcon: {
    width: '16px',
    fontSize: '10px',
    color: '#888',
    marginRight: '8px',
  },
  columnName: {
    flex: 1,
    fontSize: '13px',
    color: '#ddd',
    overflow: 'hidden',
    textOverflow: 'ellipsis',
    whiteSpace: 'nowrap' as const,
  },
  colorButton: {
    padding: '2px 6px',
    fontSize: '11px',
    backgroundColor: 'transparent',
    color: '#888',
    border: '1px solid #444',
    borderRadius: '3px',
    cursor: 'pointer',
    marginLeft: '8px',
  },
  colorButtonActive: {
    backgroundColor: '#e94560',
    color: '#fff',
    borderColor: '#e94560',
  },
  columnMeta: {
    fontSize: '11px',
    color: '#666',
    marginLeft: '8px',
  },
  categoryList: {
    padding: '4px 16px 8px 40px',
    backgroundColor: '#0a0f1a',
  },
  categoryItem: {
    display: 'flex',
    alignItems: 'center',
    justifyContent: 'space-between',
    padding: '4px 0',
    fontSize: '12px',
  },
  categoryName: {
    color: '#bbb',
    overflow: 'hidden',
    textOverflow: 'ellipsis',
    whiteSpace: 'nowrap' as const,
    flex: 1,
  },
  categoryCount: {
    color: '#666',
    marginLeft: '8px',
    fontSize: '11px',
  },
  continuousInfo: {
    padding: '4px 16px 8px 40px',
    fontSize: '11px',
    color: '#888',
  },
  rangeLabel: {
    marginRight: '12px',
  },
  loading: {
    padding: '20px',
    textAlign: 'center' as const,
    color: '#666',
    fontSize: '13px',
  },
  emptyState: {
    padding: '20px',
    textAlign: 'center' as const,
    color: '#666',
    fontSize: '13px',
  },
  createForm: {
    padding: '8px 16px',
  },
  inputRow: {
    display: 'flex',
    gap: '8px',
    marginBottom: '8px',
  },
  input: {
    flex: 1,
    padding: '6px 10px',
    fontSize: '12px',
    backgroundColor: '#0f3460',
    color: '#eee',
    border: '1px solid #1a1a2e',
    borderRadius: '4px',
    outline: 'none',
  },
  smallButton: {
    padding: '6px 10px',
    fontSize: '11px',
    backgroundColor: '#0f3460',
    color: '#aaa',
    border: '1px solid #1a1a2e',
    borderRadius: '4px',
    cursor: 'pointer',
    whiteSpace: 'nowrap' as const,
  },
  primaryButton: {
    backgroundColor: '#e94560',
    color: '#fff',
    borderColor: '#e94560',
  },
  dangerButton: {
    backgroundColor: 'transparent',
    color: '#e94560',
    borderColor: '#e94560',
  },
  selectionActions: {
    padding: '12px 16px',
    backgroundColor: '#0a0f1a',
    borderTop: '1px solid #0f3460',
  },
  selectionHeader: {
    fontSize: '12px',
    color: '#ffd700',
    marginBottom: '8px',
    fontWeight: 500,
  },
  labelInput: {
    display: 'flex',
    gap: '8px',
    marginBottom: '8px',
  },
  annotationItem: {
    padding: '6px 16px',
    backgroundColor: '#0a0f1a',
    marginBottom: '1px',
    display: 'flex',
    alignItems: 'center',
    justifyContent: 'space-between',
  },
  annotationName: {
    fontSize: '12px',
    color: '#bbb',
  },
  annotationActions: {
    display: 'flex',
    gap: '4px',
  },
  iconButton: {
    padding: '2px 6px',
    fontSize: '10px',
    backgroundColor: 'transparent',
    color: '#888',
    border: '1px solid #444',
    borderRadius: '3px',
    cursor: 'pointer',
  },
  exportButton: {
    width: '100%',
    padding: '8px',
    fontSize: '12px',
    backgroundColor: '#0f3460',
    color: '#aaa',
    border: '1px solid #1a1a2e',
    borderRadius: '4px',
    cursor: 'pointer',
    marginTop: '8px',
  },
}

interface CategoryColumnProps {
  summary: ObsSummary
  isActive: boolean
  onColorBy: () => void
}

function CategoryColumn({ summary, isActive, onColorBy }: CategoryColumnProps) {
  const [expanded, setExpanded] = useState(false)
  const [hovered, setHovered] = useState(false)

  const categories = summary.categories || []
  const totalCount = categories.reduce((sum, c) => sum + c.count, 0)

  return (
    <div style={styles.column}>
      <div
        style={{
          ...styles.columnHeader,
          ...(hovered ? styles.columnHeaderHover : {}),
          ...(isActive ? styles.columnHeaderActive : {}),
        }}
        onMouseEnter={() => setHovered(true)}
        onMouseLeave={() => setHovered(false)}
        onClick={() => setExpanded(!expanded)}
      >
        <span style={styles.expandIcon}>{expanded ? '▼' : '▶'}</span>
        <span style={styles.columnName} title={summary.name}>
          {summary.name}
        </span>
        <span style={styles.columnMeta}>({categories.length})</span>
        <button
          style={{
            ...styles.colorButton,
            ...(isActive ? styles.colorButtonActive : {}),
          }}
          onClick={(e) => {
            e.stopPropagation()
            onColorBy()
          }}
          title="Color cells by this column"
        >
          Color
        </button>
      </div>
      {expanded && (
        <div style={styles.categoryList}>
          {categories.map((cat: CategoryValue) => (
            <div key={cat.value} style={styles.categoryItem}>
              <span style={styles.categoryName} title={cat.value}>
                {cat.value}
              </span>
              <span style={styles.categoryCount}>
                {cat.count.toLocaleString()} ({((cat.count / totalCount) * 100).toFixed(1)}%)
              </span>
            </div>
          ))}
        </div>
      )}
    </div>
  )
}

interface ContinuousColumnProps {
  summary: ObsSummary
  isActive: boolean
  onColorBy: () => void
}

function ContinuousColumn({ summary, isActive, onColorBy }: ContinuousColumnProps) {
  const [hovered, setHovered] = useState(false)

  return (
    <div style={styles.column}>
      <div
        style={{
          ...styles.columnHeader,
          ...(hovered ? styles.columnHeaderHover : {}),
          ...(isActive ? styles.columnHeaderActive : {}),
        }}
        onMouseEnter={() => setHovered(true)}
        onMouseLeave={() => setHovered(false)}
      >
        <span style={styles.expandIcon}></span>
        <span style={styles.columnName} title={summary.name}>
          {summary.name}
        </span>
        <button
          style={{
            ...styles.colorButton,
            ...(isActive ? styles.colorButtonActive : {}),
          }}
          onClick={(e) => {
            e.stopPropagation()
            onColorBy()
          }}
          title="Color cells by this column"
        >
          Color
        </button>
      </div>
      <div style={styles.continuousInfo}>
        <span style={styles.rangeLabel}>
          Range: {summary.min?.toFixed(2)} - {summary.max?.toFixed(2)}
        </span>
        {summary.mean !== undefined && (
          <span>Mean: {summary.mean.toFixed(2)}</span>
        )}
      </div>
    </div>
  )
}

export default function CellPanel() {
  const { colorMode, selectedColorColumn, selectedCellIndices, clearSelection } = useStore()
  const { summaries, isLoading, error, refresh } = useObsSummaries()
  const { selectColorColumn } = useDataActions()

  // State for creating new annotation
  const [newAnnotationName, setNewAnnotationName] = useState('')
  const [isCreating, setIsCreating] = useState(false)

  // State for labeling cells
  const [selectedAnnotation, setSelectedAnnotation] = useState<string | null>(null)
  const [newLabel, setNewLabel] = useState('')
  const [isLabeling, setIsLabeling] = useState(false)

  // State for collapsible sections
  const [categoricalExpanded, setCategoricalExpanded] = useState(true)
  const [continuousExpanded, setContinuousExpanded] = useState(true)

  // Separate categorical and continuous columns
  const categoricalColumns = summaries.filter((s) => s.dtype === 'category' || s.dtype === 'string')
  const continuousColumns = summaries.filter((s) => s.dtype === 'numeric')

  // Get user-created annotations (for now, show all categorical)
  const userAnnotations = categoricalColumns

  const handleColorBy = (columnName: string) => {
    if (colorMode === 'metadata' && selectedColorColumn === columnName) {
      selectColorColumn(null)
    } else {
      selectColorColumn(columnName)
    }
  }

  const handleCreateAnnotation = useCallback(async () => {
    if (!newAnnotationName.trim()) return
    setIsCreating(true)
    try {
      await createAnnotation(newAnnotationName.trim())
      setNewAnnotationName('')
      refresh()
    } catch (err) {
      console.error('Failed to create annotation:', err)
      alert(`Failed to create annotation: ${(err as Error).message}`)
    } finally {
      setIsCreating(false)
    }
  }, [newAnnotationName, refresh])

  const handleDeleteAnnotation = useCallback(
    async (name: string) => {
      if (!confirm(`Delete annotation "${name}"?`)) return
      try {
        await deleteAnnotation(name)
        refresh()
      } catch (err) {
        console.error('Failed to delete annotation:', err)
        alert(`Failed to delete annotation: ${(err as Error).message}`)
      }
    },
    [refresh]
  )

  const handleLabelCells = useCallback(async () => {
    if (!selectedAnnotation || !newLabel.trim() || selectedCellIndices.length === 0) return
    setIsLabeling(true)
    try {
      await labelCells(selectedAnnotation, newLabel.trim(), selectedCellIndices)
      setNewLabel('')
      clearSelection()
      refresh()
    } catch (err) {
      console.error('Failed to label cells:', err)
      alert(`Failed to label cells: ${(err as Error).message}`)
    } finally {
      setIsLabeling(false)
    }
  }, [selectedAnnotation, newLabel, selectedCellIndices, clearSelection, refresh])

  const handleExport = useCallback(async () => {
    try {
      const tsv = await exportAnnotations()
      // Create download
      const blob = new Blob([tsv], { type: 'text/tab-separated-values' })
      const url = URL.createObjectURL(blob)
      const a = document.createElement('a')
      a.href = url
      a.download = 'cell_annotations.tsv'
      document.body.appendChild(a)
      a.click()
      document.body.removeChild(a)
      URL.revokeObjectURL(url)
    } catch (err) {
      console.error('Failed to export annotations:', err)
      alert(`Failed to export: ${(err as Error).message}`)
    }
  }, [])

  if (isLoading) {
    return (
      <div style={styles.panel}>
        <div style={styles.header}>
          <div style={styles.title}>Cells</div>
        </div>
        <div style={styles.loading}>Loading metadata...</div>
      </div>
    )
  }

  if (error) {
    return (
      <div style={styles.panel}>
        <div style={styles.header}>
          <div style={styles.title}>Cells</div>
        </div>
        <div style={styles.loading}>Error: {error}</div>
      </div>
    )
  }

  return (
    <div style={styles.panel}>
      <div style={styles.header}>
        <div style={styles.title}>Cells</div>
      </div>

      <div style={styles.content}>
        {/* Add Annotation - at the top */}
        <div style={styles.section}>
          <div style={{ ...styles.sectionHeader, cursor: 'default' }}>Add Annotation</div>
          <div style={styles.createForm}>
            <div style={styles.inputRow}>
              <input
                type="text"
                style={styles.input}
                placeholder="New annotation name..."
                value={newAnnotationName}
                onChange={(e) => setNewAnnotationName(e.target.value)}
                onKeyDown={(e) => e.key === 'Enter' && handleCreateAnnotation()}
              />
              <button
                style={{ ...styles.smallButton, ...styles.primaryButton }}
                onClick={handleCreateAnnotation}
                disabled={isCreating || !newAnnotationName.trim()}
              >
                {isCreating ? '...' : 'Create'}
              </button>
            </div>
          </div>
        </div>

        {/* Categorical Columns - collapsible */}
        {categoricalColumns.length > 0 && (
          <div style={styles.section}>
            <div
              style={styles.sectionHeader}
              onClick={() => setCategoricalExpanded(!categoricalExpanded)}
            >
              <span style={styles.sectionToggle}>{categoricalExpanded ? '▼' : '▶'}</span>
              Categorical ({categoricalColumns.length})
            </div>
            {categoricalExpanded &&
              categoricalColumns.map((summary) => (
                <CategoryColumn
                  key={summary.name}
                  summary={summary}
                  isActive={colorMode === 'metadata' && selectedColorColumn === summary.name}
                  onColorBy={() => handleColorBy(summary.name)}
                />
              ))}
          </div>
        )}

        {/* Continuous Columns - collapsible */}
        {continuousColumns.length > 0 && (
          <div style={styles.section}>
            <div
              style={styles.sectionHeader}
              onClick={() => setContinuousExpanded(!continuousExpanded)}
            >
              <span style={styles.sectionToggle}>{continuousExpanded ? '▼' : '▶'}</span>
              Continuous ({continuousColumns.length})
            </div>
            {continuousExpanded &&
              continuousColumns.map((summary) => (
                <ContinuousColumn
                  key={summary.name}
                  summary={summary}
                  isActive={colorMode === 'metadata' && selectedColorColumn === summary.name}
                  onColorBy={() => handleColorBy(summary.name)}
                />
              ))}
          </div>
        )}

        {summaries.length === 0 && (
          <div style={styles.emptyState}>No cell metadata available</div>
        )}

        {/* Export button */}
        <div style={{ padding: '8px 16px', borderTop: '1px solid #0f3460', marginTop: '8px' }}>
          <button style={styles.exportButton} onClick={handleExport}>
            Export All Annotations (TSV)
          </button>
        </div>
      </div>

      {/* Selection Actions - shown when cells are selected */}
      {selectedCellIndices.length > 0 && (
        <div style={styles.selectionActions}>
          <div style={styles.selectionHeader}>
            Label {selectedCellIndices.length.toLocaleString()} selected cells
          </div>

          {/* Select annotation to label */}
          <div style={styles.inputRow}>
            <select
              style={{ ...styles.input, cursor: 'pointer' }}
              value={selectedAnnotation || ''}
              onChange={(e) => setSelectedAnnotation(e.target.value || null)}
            >
              <option value="">Select annotation...</option>
              {userAnnotations.map((ann) => (
                <option key={ann.name} value={ann.name}>
                  {ann.name}
                </option>
              ))}
            </select>
          </div>

          {/* Enter label and apply */}
          {selectedAnnotation && (
            <div style={styles.labelInput}>
              <input
                type="text"
                style={styles.input}
                placeholder="Enter label..."
                value={newLabel}
                onChange={(e) => setNewLabel(e.target.value)}
                onKeyDown={(e) => e.key === 'Enter' && handleLabelCells()}
              />
              <button
                style={{ ...styles.smallButton, ...styles.primaryButton }}
                onClick={handleLabelCells}
                disabled={isLabeling || !newLabel.trim()}
              >
                {isLabeling ? '...' : 'Apply'}
              </button>
            </div>
          )}
        </div>
      )}
    </div>
  )
}
