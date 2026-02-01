import { useState } from 'react'
import { useStore } from '../store'
import { useObsSummaries, ObsSummary, CategoryValue } from '../hooks/useData'
import { useDataActions } from '../hooks/useData'

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
  const { colorMode, selectedColorColumn } = useStore()
  const { summaries, isLoading, error } = useObsSummaries()
  const { selectColorColumn } = useDataActions()

  // Separate categorical and continuous columns
  const categoricalColumns = summaries.filter((s) => s.dtype === 'category' || s.dtype === 'string')
  const continuousColumns = summaries.filter((s) => s.dtype === 'numeric')

  const handleColorBy = (columnName: string) => {
    if (colorMode === 'metadata' && selectedColorColumn === columnName) {
      // Toggle off if already selected
      selectColorColumn(null)
    } else {
      selectColorColumn(columnName)
    }
  }

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
        {/* Categorical Columns */}
        {categoricalColumns.length > 0 && (
          <div style={styles.section}>
            <div style={styles.sectionHeader}>Categorical</div>
            {categoricalColumns.map((summary) => (
              <CategoryColumn
                key={summary.name}
                summary={summary}
                isActive={colorMode === 'metadata' && selectedColorColumn === summary.name}
                onColorBy={() => handleColorBy(summary.name)}
              />
            ))}
          </div>
        )}

        {/* Continuous Columns */}
        {continuousColumns.length > 0 && (
          <div style={styles.section}>
            <div style={styles.sectionHeader}>Continuous</div>
            {continuousColumns.map((summary) => (
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
      </div>
    </div>
  )
}
