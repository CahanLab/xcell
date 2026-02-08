import { useState, useCallback, useRef, useEffect } from 'react'
import { useStore } from '../store'
import {
  useObsSummaries,
  ObsSummary,
  CategoryValue,
  useDataActions,
  createAnnotation,
  labelCells,
  useDiffExp,
} from '../hooks/useData'

const API_BASE = '/api'

const styles = {
  panel: {
    width: '280px',
    flex: '1 1 auto',
    minHeight: '200px',
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
  groupButton: {
    padding: '2px 5px',
    fontSize: '10px',
    backgroundColor: 'transparent',
    color: '#666',
    border: '1px solid #444',
    borderRadius: '3px',
    cursor: 'pointer',
    marginLeft: '4px',
  },
  groupButtonActive1: {
    backgroundColor: '#4ecdc4',
    color: '#000',
    borderColor: '#4ecdc4',
  },
  groupButtonActive2: {
    backgroundColor: '#ff6b6b',
    color: '#fff',
    borderColor: '#ff6b6b',
  },
  comparisonBar: {
    padding: '12px 16px',
    backgroundColor: '#1a2744',
    borderTop: '1px solid #0f3460',
  },
  comparisonTitle: {
    fontSize: '11px',
    fontWeight: 600,
    color: '#e94560',
    textTransform: 'uppercase' as const,
    letterSpacing: '0.5px',
    marginBottom: '8px',
  },
  comparisonGroups: {
    display: 'flex',
    gap: '12px',
    marginBottom: '10px',
  },
  comparisonGroup: {
    flex: 1,
    fontSize: '12px',
  },
  comparisonLabel: {
    color: '#888',
    fontSize: '10px',
    marginBottom: '2px',
  },
  comparisonValue: {
    color: '#ddd',
    overflow: 'hidden',
    textOverflow: 'ellipsis',
    whiteSpace: 'nowrap' as const,
  },
  comparisonCount: {
    color: '#666',
    fontSize: '11px',
  },
  comparisonButtons: {
    display: 'flex',
    gap: '8px',
  },
  selectionGroupButtons: {
    display: 'flex',
    gap: '8px',
    marginBottom: '10px',
  },
  columnActions: {
    display: 'flex',
    gap: '2px',
    marginLeft: '4px',
  },
  columnActionButton: {
    padding: '2px 4px',
    fontSize: '10px',
    backgroundColor: 'transparent',
    color: '#666',
    border: 'none',
    borderRadius: '2px',
    cursor: 'pointer',
    opacity: 0.6,
  },
  columnNameInput: {
    padding: '2px 6px',
    fontSize: '13px',
    backgroundColor: '#0f3460',
    color: '#eee',
    border: '1px solid #1a1a2e',
    borderRadius: '4px',
    outline: 'none',
    flex: 1,
    minWidth: 0,
  },
  hiddenSection: {
    padding: '8px 16px',
    borderTop: '1px solid #0f3460',
  },
  hiddenTitle: {
    fontSize: '11px',
    fontWeight: 600,
    color: '#666',
    textTransform: 'uppercase' as const,
    letterSpacing: '0.5px',
    marginBottom: '8px',
    cursor: 'pointer',
  },
  hiddenItem: {
    display: 'flex',
    alignItems: 'center',
    justifyContent: 'space-between',
    padding: '4px 8px',
    fontSize: '12px',
    color: '#888',
    backgroundColor: '#0a0f1a',
    borderRadius: '4px',
    marginBottom: '4px',
  },
  maskBar: {
    padding: '12px 16px',
    backgroundColor: '#1a2744',
    borderTop: '1px solid #0f3460',
  },
  maskTitle: {
    fontSize: '11px',
    fontWeight: 600,
    color: '#e94560',
    textTransform: 'uppercase' as const,
    letterSpacing: '0.5px',
    marginBottom: '8px',
    display: 'flex',
    justifyContent: 'space-between',
    alignItems: 'center',
  },
  maskInfo: {
    fontSize: '12px',
    color: '#aaa',
    marginBottom: '8px',
  },
  maskButtons: {
    display: 'flex',
    gap: '8px',
  },
  toggleRow: {
    display: 'flex',
    alignItems: 'center',
    gap: '8px',
    marginTop: '8px',
    fontSize: '12px',
    color: '#aaa',
  },
  toggle: {
    width: '32px',
    height: '18px',
    backgroundColor: '#0f3460',
    borderRadius: '9px',
    position: 'relative' as const,
    cursor: 'pointer',
    transition: 'background-color 0.2s',
  },
  toggleActive: {
    backgroundColor: '#4ecdc4',
  },
  toggleKnob: {
    position: 'absolute' as const,
    top: '2px',
    left: '2px',
    width: '14px',
    height: '14px',
    backgroundColor: '#fff',
    borderRadius: '50%',
    transition: 'left 0.2s',
  },
  toggleKnobActive: {
    left: '16px',
  },
  maskActionButton: {
    padding: '4px 8px',
    fontSize: '10px',
    backgroundColor: '#0f3460',
    color: '#aaa',
    border: '1px solid #1a1a2e',
    borderRadius: '3px',
    cursor: 'pointer',
  },
}

interface CategoryColumnProps {
  summary: ObsSummary
  displayName: string
  isActive: boolean
  onColorBy: () => void
  onSetGroup: (categoryValue: string, groupNumber: 1 | 2) => void
  onSelectCells: (categoryValue: string) => void
  group1Categories: Set<string>
  group2Categories: Set<string>
  onHide: () => void
  onRename: (newName: string) => void
}

function CategoryColumn({ summary, displayName, isActive, onColorBy, onSetGroup, onSelectCells, group1Categories, group2Categories, onHide, onRename }: CategoryColumnProps) {
  const [expanded, setExpanded] = useState(false)
  const [hovered, setHovered] = useState(false)
  const [isEditing, setIsEditing] = useState(false)
  const [editName, setEditName] = useState(displayName)
  const editInputRef = useRef<HTMLInputElement>(null)

  const categories = summary.categories || []
  const totalCount = categories.reduce((sum, c) => sum + c.count, 0)

  // Create a key for this category column to track group assignments
  const getCategoryKey = (catValue: string) => `${summary.name}:${catValue}`

  useEffect(() => {
    if (isEditing && editInputRef.current) {
      editInputRef.current.focus()
      editInputRef.current.select()
    }
  }, [isEditing])

  const handleDoubleClick = (e: React.MouseEvent) => {
    e.stopPropagation()
    setEditName(displayName)
    setIsEditing(true)
  }

  const handleEditSubmit = () => {
    const trimmedName = editName.trim()
    if (trimmedName && trimmedName !== displayName) {
      onRename(trimmedName)
    }
    setIsEditing(false)
  }

  const handleEditKeyDown = (e: React.KeyboardEvent) => {
    if (e.key === 'Enter') {
      handleEditSubmit()
    } else if (e.key === 'Escape') {
      setIsEditing(false)
      setEditName(displayName)
    }
  }

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
        onClick={() => !isEditing && setExpanded(!expanded)}
      >
        <span style={styles.expandIcon}>{expanded ? '▼' : '▶'}</span>
        {isEditing ? (
          <input
            ref={editInputRef}
            type="text"
            value={editName}
            onChange={(e) => setEditName(e.target.value)}
            onBlur={handleEditSubmit}
            onKeyDown={handleEditKeyDown}
            onClick={(e) => e.stopPropagation()}
            style={styles.columnNameInput}
          />
        ) : (
          <span
            style={styles.columnName}
            title={`${displayName}${displayName !== summary.name ? ` (${summary.name})` : ''}\nDouble-click to rename`}
            onDoubleClick={handleDoubleClick}
          >
            {displayName}
          </span>
        )}
        <span style={styles.columnMeta}>({categories.length})</span>
        {hovered && !isEditing && (
          <div style={styles.columnActions}>
            <button
              style={styles.columnActionButton}
              onClick={(e) => {
                e.stopPropagation()
                onHide()
              }}
              title="Hide this column"
            >
              Hide
            </button>
          </div>
        )}
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
          {categories.map((cat: CategoryValue) => {
            const catKey = getCategoryKey(cat.value)
            const isGroup1 = group1Categories.has(catKey)
            const isGroup2 = group2Categories.has(catKey)
            return (
              <div key={cat.value} style={styles.categoryItem}>
                <span
                  style={{ ...styles.categoryName, cursor: 'pointer' }}
                  title={`${cat.value}\nClick to select these cells`}
                  onClick={() => onSelectCells(cat.value)}
                >
                  {cat.value}
                </span>
                <div style={{ display: 'flex', alignItems: 'center' }}>
                  <button
                    style={{
                      ...styles.groupButton,
                      ...(isGroup1 ? styles.groupButtonActive1 : {}),
                    }}
                    onClick={(e) => {
                      e.stopPropagation()
                      onSetGroup(cat.value, 1)
                    }}
                    title="Set as Group 1 for comparison"
                  >
                    G1
                  </button>
                  <button
                    style={{
                      ...styles.groupButton,
                      ...(isGroup2 ? styles.groupButtonActive2 : {}),
                    }}
                    onClick={(e) => {
                      e.stopPropagation()
                      onSetGroup(cat.value, 2)
                    }}
                    title="Set as Group 2 for comparison"
                  >
                    G2
                  </button>
                  <span style={styles.categoryCount}>
                    {cat.count.toLocaleString()} ({((cat.count / totalCount) * 100).toFixed(1)}%)
                  </span>
                </div>
              </div>
            )
          })}
        </div>
      )}
    </div>
  )
}

interface ContinuousColumnProps {
  summary: ObsSummary
  displayName: string
  isActive: boolean
  onColorBy: () => void
  onHide: () => void
  onRename: (newName: string) => void
}

function ContinuousColumn({ summary, displayName, isActive, onColorBy, onHide, onRename }: ContinuousColumnProps) {
  const [hovered, setHovered] = useState(false)
  const [isEditing, setIsEditing] = useState(false)
  const [editName, setEditName] = useState(displayName)
  const editInputRef = useRef<HTMLInputElement>(null)

  useEffect(() => {
    if (isEditing && editInputRef.current) {
      editInputRef.current.focus()
      editInputRef.current.select()
    }
  }, [isEditing])

  const handleDoubleClick = (e: React.MouseEvent) => {
    e.stopPropagation()
    setEditName(displayName)
    setIsEditing(true)
  }

  const handleEditSubmit = () => {
    const trimmedName = editName.trim()
    if (trimmedName && trimmedName !== displayName) {
      onRename(trimmedName)
    }
    setIsEditing(false)
  }

  const handleEditKeyDown = (e: React.KeyboardEvent) => {
    if (e.key === 'Enter') {
      handleEditSubmit()
    } else if (e.key === 'Escape') {
      setIsEditing(false)
      setEditName(displayName)
    }
  }

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
        {isEditing ? (
          <input
            ref={editInputRef}
            type="text"
            value={editName}
            onChange={(e) => setEditName(e.target.value)}
            onBlur={handleEditSubmit}
            onKeyDown={handleEditKeyDown}
            onClick={(e) => e.stopPropagation()}
            style={styles.columnNameInput}
          />
        ) : (
          <span
            style={styles.columnName}
            title={`${displayName}${displayName !== summary.name ? ` (${summary.name})` : ''}\nDouble-click to rename`}
            onDoubleClick={handleDoubleClick}
          >
            {displayName}
          </span>
        )}
        {hovered && !isEditing && (
          <div style={styles.columnActions}>
            <button
              style={styles.columnActionButton}
              onClick={(e) => {
                e.stopPropagation()
                onHide()
              }}
              title="Hide this column"
            >
              Hide
            </button>
          </div>
        )}
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
  const {
    colorMode,
    selectedColorColumn,
    selectedCellIndices,
    setSelectedCellIndices,
    clearSelection,
    comparison,
    setComparisonGroup1,
    setComparisonGroup2,
    clearComparison,
    setDiffExpModalOpen,
    hiddenColumns,
    columnDisplayNames,
    hideColumn,
    showColumn,
    setColumnDisplayName,
    schema,
    activeCellMask,
    showMaskedCells,
    setActiveCellsFromSelection,
    addSelectionToActive,
    removeSelectionFromActive,
    resetActiveCells,
    setShowMaskedCells,
    setSchema,
  } = useStore()
  const { summaries, isLoading, error, refresh } = useObsSummaries()
  const { selectColorColumn } = useDataActions()
  const { runComparison, isDiffExpLoading } = useDiffExp()

  // State for creating new annotation
  const [newAnnotationName, setNewAnnotationName] = useState('')
  const [isCreating, setIsCreating] = useState(false)

  // State for labeling cells
  const [selectedAnnotation, setSelectedAnnotation] = useState<string | null>(null)
  const [newLabel, setNewLabel] = useState('')
  const [isLabeling, setIsLabeling] = useState(false)

  // State for delete confirmation
  const [showDeleteConfirm, setShowDeleteConfirm] = useState(false)
  const [isDeleting, setIsDeleting] = useState(false)

  // State for collapsible sections
  const [categoricalExpanded, setCategoricalExpanded] = useState(true)
  const [continuousExpanded, setContinuousExpanded] = useState(true)
  const [hiddenExpanded, setHiddenExpanded] = useState(false)

  // Track which categories are set as groups (for highlighting buttons)
  const [group1Categories, setGroup1Categories] = useState<Set<string>>(new Set())
  const [group2Categories, setGroup2Categories] = useState<Set<string>>(new Set())

  // Reset delete confirmation when selection changes
  useEffect(() => {
    setShowDeleteConfirm(false)
  }, [selectedCellIndices])

  // Get display name for a column
  const getDisplayName = useCallback((originalName: string) => {
    return columnDisplayNames[originalName] || originalName
  }, [columnDisplayNames])

  // Filter out hidden columns
  const visibleSummaries = summaries.filter((s) => !hiddenColumns.has(s.name))
  const hiddenSummaries = summaries.filter((s) => hiddenColumns.has(s.name))

  // Separate categorical and continuous columns (from visible only)
  const categoricalColumns = visibleSummaries.filter((s) => s.dtype === 'category' || s.dtype === 'string')
  const continuousColumns = visibleSummaries.filter((s) => s.dtype === 'numeric')

  // Get user-created annotations (for now, show all categorical)
  const userAnnotations = categoricalColumns

  // Check if comparison groups are set
  const hasGroup1 = comparison.group1 !== null && comparison.group1.length > 0
  const hasGroup2 = comparison.group2 !== null && comparison.group2.length > 0
  const canCompare = hasGroup1 && hasGroup2

  const handleColorBy = (columnName: string) => {
    if (colorMode === 'metadata' && selectedColorColumn === columnName) {
      selectColorColumn(null)
    } else {
      selectColorColumn(columnName)
    }
  }

  // Set selected cells as a comparison group
  const handleSetSelectionAsGroup = useCallback(
    (groupNumber: 1 | 2) => {
      if (selectedCellIndices.length === 0) return
      const label = `Selection (${selectedCellIndices.length} cells)`
      if (groupNumber === 1) {
        setComparisonGroup1([...selectedCellIndices], label)
        setGroup1Categories(new Set())
      } else {
        setComparisonGroup2([...selectedCellIndices], label)
        setGroup2Categories(new Set())
      }
    },
    [selectedCellIndices, setComparisonGroup1, setComparisonGroup2]
  )

  // Set a category's cells as a comparison group
  const handleSetCategoryAsGroup = useCallback(
    (columnName: string, categoryValue: string, groupNumber: 1 | 2) => {
      // Fetch the column data to get cell indices for this category
      fetch(`/api/obs/${encodeURIComponent(columnName)}`)
        .then((res) => res.json())
        .then((data) => {
          const indices: number[] = []
          const categories = data.categories || []
          const categoryIndex = categories.indexOf(categoryValue)

          if (data.dtype === 'category' && categoryIndex >= 0) {
            // Values are category codes (indices into categories array)
            data.values.forEach((val: number, idx: number) => {
              if (val === categoryIndex) {
                indices.push(idx)
              }
            })
          } else {
            // Values are direct strings
            data.values.forEach((val: string, idx: number) => {
              if (val === categoryValue) {
                indices.push(idx)
              }
            })
          }

          const label = `${categoryValue}`
          const catKey = `${columnName}:${categoryValue}`

          if (groupNumber === 1) {
            setComparisonGroup1(indices, label)
            setGroup1Categories(new Set([catKey]))
            setGroup2Categories((prev) => {
              const next = new Set(prev)
              next.delete(catKey)
              return next
            })
          } else {
            setComparisonGroup2(indices, label)
            setGroup2Categories(new Set([catKey]))
            setGroup1Categories((prev) => {
              const next = new Set(prev)
              next.delete(catKey)
              return next
            })
          }
        })
        .catch((err) => {
          console.error('Failed to fetch category data:', err)
          alert('Failed to set group from category')
        })
    },
    [setComparisonGroup1, setComparisonGroup2]
  )

  // Select cells by category value
  const handleSelectCellsByCategory = useCallback(
    (columnName: string, categoryValue: string) => {
      fetch(`/api/obs/${encodeURIComponent(columnName)}`)
        .then((res) => res.json())
        .then((data) => {
          const indices: number[] = []
          const categories = data.categories || []
          const categoryIndex = categories.indexOf(categoryValue)

          if (data.dtype === 'category' && categoryIndex >= 0) {
            data.values.forEach((val: number, idx: number) => {
              if (val === categoryIndex) {
                indices.push(idx)
              }
            })
          } else {
            data.values.forEach((val: string, idx: number) => {
              if (val === categoryValue) {
                indices.push(idx)
              }
            })
          }

          setSelectedCellIndices(indices)
        })
        .catch((err) => {
          console.error('Failed to fetch category data:', err)
          alert('Failed to select cells from category')
        })
    },
    [setSelectedCellIndices]
  )

  // Handle running comparison
  const handleRunComparison = useCallback(async () => {
    try {
      await runComparison(25)
    } catch (err) {
      alert(`Differential expression failed: ${(err as Error).message}`)
    }
  }, [runComparison])

  // Clear comparison and reset category highlights
  const handleClearComparison = useCallback(() => {
    clearComparison()
    setGroup1Categories(new Set())
    setGroup2Categories(new Set())
  }, [clearComparison])

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

  const handleDeleteCells = useCallback(async () => {
    if (selectedCellIndices.length === 0) return
    setIsDeleting(true)
    try {
      const response = await fetch(`${API_BASE}/cells/delete`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ cell_indices: selectedCellIndices }),
      })
      if (!response.ok) {
        const err = await response.json()
        throw new Error(err.detail || 'Delete failed')
      }
      // Refresh schema (cell count changed)
      const schemaRes = await fetch(`${API_BASE}/schema`)
      if (schemaRes.ok) {
        setSchema(await schemaRes.json())
      }
      // Reset mask (indices are now stale) and clear selection
      resetActiveCells()
      clearSelection()
      refresh()
    } catch (err) {
      console.error('Failed to delete cells:', err)
      alert(`Failed to delete cells: ${(err as Error).message}`)
    } finally {
      setIsDeleting(false)
      setShowDeleteConfirm(false)
    }
  }, [selectedCellIndices, setSchema, resetActiveCells, clearSelection, refresh])

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
                  displayName={getDisplayName(summary.name)}
                  isActive={colorMode === 'metadata' && selectedColorColumn === summary.name}
                  onColorBy={() => handleColorBy(summary.name)}
                  onSetGroup={(categoryValue, groupNumber) =>
                    handleSetCategoryAsGroup(summary.name, categoryValue, groupNumber)
                  }
                  onSelectCells={(categoryValue) =>
                    handleSelectCellsByCategory(summary.name, categoryValue)
                  }
                  group1Categories={group1Categories}
                  group2Categories={group2Categories}
                  onHide={() => hideColumn(summary.name)}
                  onRename={(newName) => setColumnDisplayName(summary.name, newName)}
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
                  displayName={getDisplayName(summary.name)}
                  isActive={colorMode === 'metadata' && selectedColorColumn === summary.name}
                  onColorBy={() => handleColorBy(summary.name)}
                  onHide={() => hideColumn(summary.name)}
                  onRename={(newName) => setColumnDisplayName(summary.name, newName)}
                />
              ))}
          </div>
        )}

        {/* Hidden Columns - collapsible */}
        {hiddenSummaries.length > 0 && (
          <div style={styles.hiddenSection}>
            <div
              style={styles.hiddenTitle}
              onClick={() => setHiddenExpanded(!hiddenExpanded)}
            >
              <span style={styles.sectionToggle}>{hiddenExpanded ? '▼' : '▶'}</span>
              Hidden ({hiddenSummaries.length})
            </div>
            {hiddenExpanded &&
              hiddenSummaries.map((summary) => (
                <div key={summary.name} style={styles.hiddenItem}>
                  <span>{getDisplayName(summary.name)}</span>
                  <button
                    style={styles.smallButton}
                    onClick={() => showColumn(summary.name)}
                  >
                    Show
                  </button>
                </div>
              ))}
          </div>
        )}

        {visibleSummaries.length === 0 && hiddenSummaries.length === 0 && (
          <div style={styles.emptyState}>No cell metadata available</div>
        )}
      </div>

      {/* Cell Mask Status Bar */}
      {activeCellMask && (
        <div style={styles.maskBar}>
          <div style={styles.maskTitle}>
            <span>Cell Mask</span>
          </div>
          <div style={styles.maskInfo}>
            {activeCellMask.filter(Boolean).length.toLocaleString()} of{' '}
            {schema?.n_cells.toLocaleString()} cells active
          </div>
          <div style={styles.toggleRow}>
            <div
              style={{
                ...styles.toggle,
                ...(showMaskedCells ? styles.toggleActive : {}),
              }}
              onClick={() => setShowMaskedCells(!showMaskedCells)}
            >
              <div
                style={{
                  ...styles.toggleKnob,
                  ...(showMaskedCells ? styles.toggleKnobActive : {}),
                }}
              />
            </div>
            <span>Show masked cells</span>
          </div>
          <div style={{ ...styles.maskButtons, marginTop: '8px' }}>
            <button
              style={{ ...styles.smallButton, ...styles.dangerButton, flex: 1 }}
              onClick={resetActiveCells}
            >
              Reset Mask
            </button>
          </div>
        </div>
      )}

      {/* Comparison Status Bar */}
      {(hasGroup1 || hasGroup2) && (
        <div style={styles.comparisonBar}>
          <div style={styles.comparisonTitle}>Comparison</div>
          <div style={styles.comparisonGroups}>
            <div style={styles.comparisonGroup}>
              <div style={{ ...styles.comparisonLabel, color: '#4ecdc4' }}>Group 1</div>
              <div style={styles.comparisonValue}>
                {comparison.group1Label || 'Not set'}
              </div>
              {hasGroup1 && (
                <div style={styles.comparisonCount}>
                  {comparison.group1?.length.toLocaleString()} cells
                </div>
              )}
            </div>
            <div style={styles.comparisonGroup}>
              <div style={{ ...styles.comparisonLabel, color: '#ff6b6b' }}>Group 2</div>
              <div style={styles.comparisonValue}>
                {comparison.group2Label || 'Not set'}
              </div>
              {hasGroup2 && (
                <div style={styles.comparisonCount}>
                  {comparison.group2?.length.toLocaleString()} cells
                </div>
              )}
            </div>
          </div>
          <div style={styles.comparisonButtons}>
            <button
              style={{ ...styles.smallButton, ...styles.primaryButton, flex: 1 }}
              onClick={handleRunComparison}
              disabled={!canCompare || isDiffExpLoading}
            >
              {isDiffExpLoading ? 'Running...' : 'Compare'}
            </button>
            <button
              style={{ ...styles.smallButton, flex: 1 }}
              onClick={() => setDiffExpModalOpen(true)}
            >
              Options
            </button>
            <button
              style={{ ...styles.smallButton, ...styles.dangerButton }}
              onClick={handleClearComparison}
            >
              Clear
            </button>
          </div>
        </div>
      )}

      {/* Selection Actions - shown when cells are selected */}
      {selectedCellIndices.length > 0 && (
        <div style={styles.selectionActions}>
          <div style={styles.selectionHeader}>
            {selectedCellIndices.length.toLocaleString()} cells selected
          </div>

          {/* Set as comparison group buttons */}
          <div style={styles.selectionGroupButtons}>
            <button
              style={{
                ...styles.smallButton,
                flex: 1,
                ...(comparison.group1?.length === selectedCellIndices.length &&
                comparison.group1Label?.includes('Selection')
                  ? styles.groupButtonActive1
                  : {}),
              }}
              onClick={() => handleSetSelectionAsGroup(1)}
            >
              Set as Group 1
            </button>
            <button
              style={{
                ...styles.smallButton,
                flex: 1,
                ...(comparison.group2?.length === selectedCellIndices.length &&
                comparison.group2Label?.includes('Selection')
                  ? styles.groupButtonActive2
                  : {}),
              }}
              onClick={() => handleSetSelectionAsGroup(2)}
            >
              Set as Group 2
            </button>
          </div>

          {/* Cell masking buttons */}
          <div style={{ ...styles.selectionGroupButtons, borderTop: '1px solid #0f3460', paddingTop: '10px' }}>
            <button
              style={{ ...styles.maskActionButton, flex: 1 }}
              onClick={setActiveCellsFromSelection}
              title="Set selection as the only active cells"
            >
              Set Active
            </button>
            <button
              style={{ ...styles.maskActionButton, flex: 1 }}
              onClick={addSelectionToActive}
              title="Add selection to active cells"
            >
              Add to Active
            </button>
            <button
              style={{ ...styles.maskActionButton, flex: 1 }}
              onClick={removeSelectionFromActive}
              title="Mask selection — hide from analysis but keep in data"
            >
              Mask
            </button>
            <button
              style={{
                ...styles.maskActionButton,
                flex: 1,
                backgroundColor: showDeleteConfirm ? '#e94560' : '#3a1020',
                color: showDeleteConfirm ? '#fff' : '#e94560',
                border: '1px solid #e94560',
              }}
              onClick={() => {
                if (showDeleteConfirm) {
                  handleDeleteCells()
                } else {
                  setShowDeleteConfirm(true)
                }
              }}
              disabled={isDeleting}
              title="Permanently remove selected cells from the dataset"
            >
              {isDeleting ? '...' : showDeleteConfirm ? `Delete ${selectedCellIndices.length}?` : 'Delete'}
            </button>
          </div>
          {showDeleteConfirm && (
            <div style={{ fontSize: '9px', color: '#e94560', marginTop: '4px', display: 'flex', alignItems: 'center', gap: '4px' }}>
              <span>This permanently removes {selectedCellIndices.length.toLocaleString()} cells.</span>
              <button
                onClick={() => setShowDeleteConfirm(false)}
                style={{
                  background: 'none',
                  border: 'none',
                  color: '#888',
                  cursor: 'pointer',
                  fontSize: '9px',
                  textDecoration: 'underline',
                }}
              >
                Cancel
              </button>
            </div>
          )}

          {/* Select annotation to label */}
          <div style={{ ...styles.inputRow, marginTop: '8px', paddingTop: '8px', borderTop: '1px solid #0f3460' }}>
            <select
              style={{ ...styles.input, cursor: 'pointer' }}
              value={selectedAnnotation || ''}
              onChange={(e) => setSelectedAnnotation(e.target.value || null)}
            >
              <option value="">Label with annotation...</option>
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
