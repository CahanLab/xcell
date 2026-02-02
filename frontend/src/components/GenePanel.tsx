import { useState, useRef, useEffect, useCallback } from 'react'
import { useStore, GeneSet } from '../store'
import { useGeneSearch, useDataActions } from '../hooks/useData'

// Drag data type for genes
const GENE_DRAG_TYPE = 'application/x-gene'

const styles = {
  panel: {
    width: '300px',
    height: '100%',
    backgroundColor: '#16213e',
    borderLeft: '1px solid #0f3460',
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
    marginBottom: '12px',
  },
  searchContainer: {
    position: 'relative' as const,
  },
  searchInput: {
    width: '100%',
    padding: '8px 12px',
    fontSize: '13px',
    backgroundColor: '#0f3460',
    color: '#eee',
    border: '1px solid #1a1a2e',
    borderRadius: '4px',
    outline: 'none',
  },
  dropdown: {
    position: 'absolute' as const,
    top: '100%',
    left: 0,
    right: 0,
    backgroundColor: '#0f3460',
    border: '1px solid #1a1a2e',
    borderRadius: '4px',
    marginTop: '4px',
    maxHeight: '200px',
    overflowY: 'auto' as const,
    zIndex: 100,
  },
  dropdownItem: {
    padding: '8px 12px',
    fontSize: '13px',
    cursor: 'pointer',
    borderBottom: '1px solid #1a1a2e',
  },
  dropdownItemHover: {
    backgroundColor: '#1a1a2e',
  },
  content: {
    flex: 1,
    overflowY: 'auto' as const,
    padding: '12px 16px',
  },
  section: {
    marginBottom: '16px',
  },
  sectionHeader: {
    display: 'flex',
    alignItems: 'center',
    justifyContent: 'space-between',
    marginBottom: '8px',
  },
  sectionTitle: {
    fontSize: '12px',
    fontWeight: 600,
    color: '#aaa',
    textTransform: 'uppercase' as const,
  },
  addButton: {
    padding: '4px 8px',
    fontSize: '11px',
    backgroundColor: '#0f3460',
    color: '#aaa',
    border: 'none',
    borderRadius: '4px',
    cursor: 'pointer',
  },
  geneSet: {
    backgroundColor: '#0f3460',
    borderRadius: '4px',
    marginBottom: '8px',
    overflow: 'hidden',
  },
  geneSetHeader: {
    display: 'flex',
    alignItems: 'center',
    justifyContent: 'space-between',
    padding: '8px 12px',
    cursor: 'pointer',
  },
  geneSetName: {
    fontSize: '13px',
    fontWeight: 500,
    color: '#eee',
  },
  geneSetCount: {
    fontSize: '11px',
    color: '#888',
    marginLeft: '8px',
  },
  geneSetActions: {
    display: 'flex',
    gap: '4px',
  },
  iconButton: {
    padding: '4px',
    backgroundColor: 'transparent',
    color: '#888',
    border: 'none',
    borderRadius: '4px',
    cursor: 'pointer',
    fontSize: '12px',
  },
  geneList: {
    padding: '4px 12px 8px',
    borderTop: '1px solid #1a1a2e',
  },
  gene: {
    display: 'flex',
    alignItems: 'center',
    justifyContent: 'space-between',
    padding: '4px 8px',
    marginBottom: '2px',
    borderRadius: '4px',
    fontSize: '12px',
    color: '#ccc',
  },
  geneActive: {
    backgroundColor: '#e94560',
    color: '#fff',
  },
  geneHover: {
    backgroundColor: '#1a1a2e',
  },
  geneName: {
    flex: 1,
    cursor: 'pointer',
  },
  colorIndicator: {
    fontSize: '11px',
    color: '#e94560',
    marginLeft: '8px',
  },
  emptyState: {
    padding: '20px',
    textAlign: 'center' as const,
    color: '#666',
    fontSize: '13px',
  },
  selectedGenesBar: {
    padding: '8px 16px',
    backgroundColor: '#0f3460',
    borderBottom: '1px solid #1a1a2e',
    display: 'flex',
    alignItems: 'center',
    justifyContent: 'space-between',
  },
  selectedGenesText: {
    fontSize: '12px',
    color: '#e94560',
  },
  clearButton: {
    padding: '4px 8px',
    fontSize: '11px',
    backgroundColor: 'transparent',
    color: '#888',
    border: '1px solid #888',
    borderRadius: '4px',
    cursor: 'pointer',
  },
  searchResults: {
    marginTop: '8px',
    backgroundColor: '#0a0f1a',
    borderRadius: '4px',
    maxHeight: '180px',
    overflowY: 'auto' as const,
  },
  searchResultsHeader: {
    display: 'flex',
    alignItems: 'center',
    justifyContent: 'space-between',
    padding: '6px 10px',
    borderBottom: '1px solid #1a1a2e',
    fontSize: '11px',
    color: '#888',
  },
  searchResultItem: {
    display: 'flex',
    alignItems: 'center',
    padding: '6px 10px',
    fontSize: '12px',
    color: '#ccc',
    cursor: 'grab',
    borderBottom: '1px solid #0f3460',
    transition: 'background-color 0.1s',
  },
  searchResultItemSelected: {
    backgroundColor: '#1a3a5c',
    color: '#fff',
  },
  searchResultItemHover: {
    backgroundColor: '#0f3460',
  },
  searchResultCheckbox: {
    marginRight: '8px',
    cursor: 'pointer',
  },
  dragHint: {
    fontSize: '10px',
    color: '#666',
    padding: '6px 10px',
    textAlign: 'center' as const,
  },
  dropZone: {
    border: '2px dashed transparent',
    transition: 'all 0.2s',
  },
  dropZoneActive: {
    border: '2px dashed #e94560',
    backgroundColor: 'rgba(233, 69, 96, 0.1)',
  },
  selectionActions: {
    display: 'flex',
    gap: '4px',
    padding: '6px 10px',
    borderTop: '1px solid #1a1a2e',
  },
  smallActionButton: {
    padding: '4px 8px',
    fontSize: '10px',
    backgroundColor: '#0f3460',
    color: '#aaa',
    border: 'none',
    borderRadius: '3px',
    cursor: 'pointer',
  },
}

interface GeneSearchProps {
  onColorByGene: (gene: string) => void
  selectedSearchGenes: Set<string>
  setSelectedSearchGenes: React.Dispatch<React.SetStateAction<Set<string>>>
}

function GeneSearch({ onColorByGene, selectedSearchGenes, setSelectedSearchGenes }: GeneSearchProps) {
  const [query, setQuery] = useState('')
  const [hoveredGene, setHoveredGene] = useState<string | null>(null)
  const inputRef = useRef<HTMLInputElement>(null)
  const { results, searchGenes, clearResults } = useGeneSearch()

  useEffect(() => {
    const timer = setTimeout(() => {
      searchGenes(query)
    }, 150) // Debounce
    return () => clearTimeout(timer)
  }, [query, searchGenes])

  const handleClearSearch = () => {
    setQuery('')
    clearResults()
    setSelectedSearchGenes(new Set())
  }

  const toggleGeneSelection = (gene: string) => {
    setSelectedSearchGenes((prev) => {
      const next = new Set(prev)
      if (next.has(gene)) {
        next.delete(gene)
      } else {
        next.add(gene)
      }
      return next
    })
  }

  const selectAll = () => {
    setSelectedSearchGenes(new Set(results))
  }

  const selectNone = () => {
    setSelectedSearchGenes(new Set())
  }

  const handleDragStart = (e: React.DragEvent, gene: string) => {
    // If dragging a non-selected gene, just drag that one
    // If dragging a selected gene, drag all selected genes
    const genesToDrag = selectedSearchGenes.has(gene)
      ? Array.from(selectedSearchGenes)
      : [gene]

    e.dataTransfer.setData(GENE_DRAG_TYPE, JSON.stringify(genesToDrag))
    e.dataTransfer.effectAllowed = 'copy'
  }

  return (
    <div>
      <div style={styles.searchContainer}>
        <input
          ref={inputRef}
          type="text"
          placeholder="Search genes..."
          value={query}
          onChange={(e) => setQuery(e.target.value)}
          style={styles.searchInput}
        />
      </div>

      {results.length > 0 && (
        <div style={styles.searchResults}>
          <div style={styles.searchResultsHeader}>
            <span>{results.length} results</span>
            <div style={{ display: 'flex', gap: '8px' }}>
              <button style={styles.smallActionButton} onClick={selectAll}>
                All
              </button>
              <button style={styles.smallActionButton} onClick={selectNone}>
                None
              </button>
              <button style={styles.smallActionButton} onClick={handleClearSearch}>
                Clear
              </button>
            </div>
          </div>

          {results.map((gene) => {
            const isSelected = selectedSearchGenes.has(gene)
            const isHovered = hoveredGene === gene
            return (
              <div
                key={gene}
                draggable
                onDragStart={(e) => handleDragStart(e, gene)}
                style={{
                  ...styles.searchResultItem,
                  ...(isSelected ? styles.searchResultItemSelected : {}),
                  ...(isHovered && !isSelected ? styles.searchResultItemHover : {}),
                }}
                onMouseEnter={() => setHoveredGene(gene)}
                onMouseLeave={() => setHoveredGene(null)}
              >
                <input
                  type="checkbox"
                  checked={isSelected}
                  onChange={() => toggleGeneSelection(gene)}
                  style={styles.searchResultCheckbox}
                />
                <span
                  style={{ flex: 1, cursor: 'pointer' }}
                  onClick={() => onColorByGene(gene)}
                  title="Click to color by expression"
                >
                  {gene}
                </span>
              </div>
            )
          })}

          {selectedSearchGenes.size > 0 && (
            <div style={styles.dragHint}>
              Drag {selectedSearchGenes.size} selected gene{selectedSearchGenes.size > 1 ? 's' : ''} to a gene set
            </div>
          )}
        </div>
      )}
    </div>
  )
}

function GeneSetComponent({
  geneSet,
  onColorByGene,
  onColorBySet,
  activeGenes,
  onAddGenes,
}: {
  geneSet: GeneSet
  onColorByGene: (gene: string) => void
  onColorBySet: (genes: string[]) => void
  activeGenes: string[]
  onAddGenes: (setName: string, genes: string[]) => void
}) {
  const [expanded, setExpanded] = useState(true)
  const [hoveredGene, setHoveredGene] = useState<string | null>(null)
  const [isDragOver, setIsDragOver] = useState(false)
  const [isEditing, setIsEditing] = useState(false)
  const [editName, setEditName] = useState(geneSet.name)
  const editInputRef = useRef<HTMLInputElement>(null)
  const { removeGeneSet, removeGenesFromSet, renameGeneSet } = useStore()

  // Focus input when editing starts
  useEffect(() => {
    if (isEditing && editInputRef.current) {
      editInputRef.current.focus()
      editInputRef.current.select()
    }
  }, [isEditing])

  const handleDoubleClick = (e: React.MouseEvent) => {
    e.stopPropagation()
    setEditName(geneSet.name)
    setIsEditing(true)
  }

  const handleEditSubmit = () => {
    const trimmedName = editName.trim()
    if (trimmedName && trimmedName !== geneSet.name) {
      renameGeneSet(geneSet.name, trimmedName)
    }
    setIsEditing(false)
  }

  const handleEditKeyDown = (e: React.KeyboardEvent) => {
    if (e.key === 'Enter') {
      handleEditSubmit()
    } else if (e.key === 'Escape') {
      setIsEditing(false)
      setEditName(geneSet.name)
    }
  }

  const isActive = (gene: string) => activeGenes.includes(gene)

  const handleDragOver = useCallback((e: React.DragEvent) => {
    if (e.dataTransfer.types.includes(GENE_DRAG_TYPE)) {
      e.preventDefault()
      e.dataTransfer.dropEffect = 'copy'
      setIsDragOver(true)
    }
  }, [])

  const handleDragLeave = useCallback(() => {
    setIsDragOver(false)
  }, [])

  const handleDrop = useCallback(
    (e: React.DragEvent) => {
      e.preventDefault()
      setIsDragOver(false)

      const data = e.dataTransfer.getData(GENE_DRAG_TYPE)
      if (data) {
        try {
          const genes = JSON.parse(data) as string[]
          onAddGenes(geneSet.name, genes)
        } catch (err) {
          console.error('Failed to parse dropped genes:', err)
        }
      }
    },
    [geneSet.name, onAddGenes]
  )

  // Make genes within the set draggable too
  const handleGeneDragStart = (e: React.DragEvent, gene: string) => {
    e.dataTransfer.setData(GENE_DRAG_TYPE, JSON.stringify([gene]))
    e.dataTransfer.effectAllowed = 'copy'
  }

  return (
    <div
      style={{
        ...styles.geneSet,
        ...styles.dropZone,
        ...(isDragOver ? styles.dropZoneActive : {}),
      }}
      onDragOver={handleDragOver}
      onDragLeave={handleDragLeave}
      onDrop={handleDrop}
    >
      <div style={styles.geneSetHeader} onClick={() => !isEditing && setExpanded(!expanded)}>
        <div style={{ flex: 1, minWidth: 0 }}>
          {isEditing ? (
            <input
              ref={editInputRef}
              type="text"
              value={editName}
              onChange={(e) => setEditName(e.target.value)}
              onBlur={handleEditSubmit}
              onKeyDown={handleEditKeyDown}
              onClick={(e) => e.stopPropagation()}
              style={{
                ...styles.searchInput,
                padding: '2px 6px',
                fontSize: '13px',
                width: '100%',
              }}
            />
          ) : (
            <>
              <span
                style={styles.geneSetName}
                onDoubleClick={handleDoubleClick}
                title="Double-click to rename"
              >
                {geneSet.name}
              </span>
              <span style={styles.geneSetCount}>({geneSet.genes.length})</span>
            </>
          )}
        </div>
        <div style={styles.geneSetActions}>
          <button
            style={styles.iconButton}
            onClick={(e) => {
              e.stopPropagation()
              onColorBySet(geneSet.genes)
            }}
            title="Color by mean expression"
          >
            🎨
          </button>
          <button
            style={styles.iconButton}
            onClick={(e) => {
              e.stopPropagation()
              removeGeneSet(geneSet.name)
            }}
            title="Delete gene set"
          >
            ✕
          </button>
          <span style={{ color: '#888', fontSize: '12px' }}>{expanded ? '▼' : '▶'}</span>
        </div>
      </div>
      {expanded && geneSet.genes.length > 0 && (
        <div style={styles.geneList}>
          {geneSet.genes.map((gene) => (
            <div
              key={gene}
              draggable
              onDragStart={(e) => handleGeneDragStart(e, gene)}
              style={{
                ...styles.gene,
                ...(isActive(gene) ? styles.geneActive : {}),
                ...(hoveredGene === gene && !isActive(gene) ? styles.geneHover : {}),
                cursor: 'grab',
              }}
              onMouseEnter={() => setHoveredGene(gene)}
              onMouseLeave={() => setHoveredGene(null)}
            >
              <span
                style={styles.geneName}
                onClick={() => onColorByGene(gene)}
                title="Click to color by expression"
              >
                {gene}
              </span>
              <button
                style={{ ...styles.iconButton, fontSize: '10px' }}
                onClick={() => removeGenesFromSet(geneSet.name, [gene])}
                title="Remove from set"
              >
                ✕
              </button>
            </div>
          ))}
        </div>
      )}
      {expanded && geneSet.genes.length === 0 && (
        <div style={{ ...styles.dragHint, padding: '12px 10px' }}>
          Drag genes here to add
        </div>
      )}
    </div>
  )
}

export default function GenePanel() {
  const { geneSets, selectedGenes, addGeneSet, addGenesToSet } = useStore()
  const { colorByGene, colorByGenes, clearExpressionColor } = useDataActions()
  const [newSetName, setNewSetName] = useState('')
  const [showNewSetInput, setShowNewSetInput] = useState(false)
  const [selectedSearchGenes, setSelectedSearchGenes] = useState<Set<string>>(new Set())

  const handleAddGenesToSet = useCallback(
    (setName: string, genes: string[]) => {
      addGenesToSet(setName, genes)
      // Clear selection after adding
      setSelectedSearchGenes((prev) => {
        const next = new Set(prev)
        genes.forEach((g) => next.delete(g))
        return next
      })
    },
    [addGenesToSet]
  )

  const handleCreateSet = () => {
    if (newSetName.trim()) {
      addGeneSet(newSetName.trim(), [])
      setNewSetName('')
      setShowNewSetInput(false)
    }
  }

  return (
    <div style={styles.panel}>
      <div style={styles.header}>
        <div style={styles.title}>Genes</div>
        <GeneSearch
          onColorByGene={colorByGene}
          selectedSearchGenes={selectedSearchGenes}
          setSelectedSearchGenes={setSelectedSearchGenes}
        />
      </div>

      {selectedGenes.length > 0 && (
        <div style={styles.selectedGenesBar}>
          <span style={styles.selectedGenesText}>
            Coloring by: {selectedGenes.length === 1 ? selectedGenes[0] : `${selectedGenes.length} genes (mean)`}
          </span>
          <button style={styles.clearButton} onClick={clearExpressionColor}>
            Clear
          </button>
        </div>
      )}

      <div style={styles.content}>
        <div style={styles.section}>
          <div style={styles.sectionHeader}>
            <span style={styles.sectionTitle}>Gene Sets</span>
            <button
              style={styles.addButton}
              onClick={() => setShowNewSetInput(true)}
            >
              + New Set
            </button>
          </div>

          {showNewSetInput && (
            <div style={{ marginBottom: '8px', display: 'flex', gap: '4px' }}>
              <input
                type="text"
                placeholder="Set name..."
                value={newSetName}
                onChange={(e) => setNewSetName(e.target.value)}
                onKeyDown={(e) => {
                  if (e.key === 'Enter') handleCreateSet()
                  if (e.key === 'Escape') setShowNewSetInput(false)
                }}
                style={{ ...styles.searchInput, flex: 1 }}
                autoFocus
              />
              <button style={styles.addButton} onClick={handleCreateSet}>
                Add
              </button>
            </div>
          )}

          {geneSets.length === 0 ? (
            <div style={styles.emptyState}>
              Create a gene set, then drag genes from search results to add them
            </div>
          ) : (
            geneSets.map((gs) => (
              <GeneSetComponent
                key={gs.name}
                geneSet={gs}
                onColorByGene={colorByGene}
                onColorBySet={colorByGenes}
                activeGenes={selectedGenes}
                onAddGenes={handleAddGenesToSet}
              />
            ))
          )}
        </div>
      </div>
    </div>
  )
}
