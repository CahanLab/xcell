import { useState, useRef, useEffect, useCallback } from 'react'
import { useStore, GeneSet, GeneSetCategory, GeneSetFolder, GeneSetCategoryType } from '../store'
import { useGeneSearch, useGeneBrowse, useDataActions } from '../hooks/useData'
import ImportModal from './ImportModal'

const API_BASE = '/api'

// Drag data type for genes
const GENE_DRAG_TYPE = 'application/x-gene'

// Category display order and icons
const CATEGORY_ORDER: GeneSetCategoryType[] = ['manual', 'gene_clusters', 'similar_genes', 'diff_exp', 'marker_genes']
const CATEGORY_ICONS: Record<GeneSetCategoryType, string> = {
  manual: '📁',
  gene_clusters: '🧬',
  similar_genes: '🔗',
  diff_exp: '📊',
  spatial: '🗺️',
  marker_genes: '🏷️',
}

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
  // Category styles
  category: {
    marginBottom: '12px',
  },
  categoryHeader: {
    display: 'flex',
    alignItems: 'center',
    padding: '6px 8px',
    backgroundColor: '#0a0f1a',
    borderRadius: '4px',
    cursor: 'pointer',
    marginBottom: '4px',
  },
  categoryIcon: {
    marginRight: '6px',
    fontSize: '12px',
  },
  categoryName: {
    flex: 1,
    fontSize: '12px',
    fontWeight: 600,
    color: '#aaa',
    textTransform: 'uppercase' as const,
  },
  categoryCount: {
    fontSize: '10px',
    color: '#666',
    marginRight: '8px',
  },
  categoryExpander: {
    color: '#666',
    fontSize: '10px',
  },
  categoryContent: {
    paddingLeft: '8px',
  },
  // Folder styles
  folder: {
    marginBottom: '6px',
    marginLeft: '8px',
  },
  folderHeader: {
    display: 'flex',
    alignItems: 'center',
    padding: '4px 8px',
    backgroundColor: '#0f1a2e',
    borderRadius: '4px',
    cursor: 'pointer',
    marginBottom: '4px',
  },
  folderIcon: {
    marginRight: '6px',
    fontSize: '11px',
    color: '#888',
  },
  folderName: {
    flex: 1,
    fontSize: '11px',
    fontWeight: 500,
    color: '#bbb',
  },
  folderCount: {
    fontSize: '10px',
    color: '#666',
    marginRight: '6px',
  },
  folderActions: {
    display: 'flex',
    gap: '2px',
  },
  folderContent: {
    paddingLeft: '12px',
  },
  emptyCategory: {
    fontSize: '11px',
    color: '#555',
    padding: '8px 12px',
    fontStyle: 'italic',
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
  const [showBrowse, setShowBrowse] = useState(false)
  const inputRef = useRef<HTMLInputElement>(null)
  const { results, searchGenes, clearResults } = useGeneSearch()
  const { page, isLoading: isBrowseLoading, fetchPage } = useGeneBrowse(50)

  useEffect(() => {
    const timer = setTimeout(() => {
      searchGenes(query)
    }, 150) // Debounce
    return () => clearTimeout(timer)
  }, [query, searchGenes])

  // When browse is opened for the first time, fetch the first page
  useEffect(() => {
    if (showBrowse && !page) {
      fetchPage(0)
    }
  }, [showBrowse, page, fetchPage])

  // Hide browse when user starts typing a search query
  useEffect(() => {
    if (query.length > 0) {
      setShowBrowse(false)
    }
  }, [query])

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
    const genes = query ? results : (page?.genes || [])
    setSelectedSearchGenes(new Set(genes))
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

  // Determine which genes to display: search results or browse page
  const isSearchMode = query.length > 0
  const displayGenes = isSearchMode ? results : (showBrowse && page ? page.genes : [])
  const showGeneList = displayGenes.length > 0

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

      {/* Browse toggle - shown when not searching */}
      {!isSearchMode && (
        <button
          onClick={() => setShowBrowse(!showBrowse)}
          style={{
            width: '100%',
            marginTop: '6px',
            padding: '5px 8px',
            fontSize: '11px',
            color: showBrowse ? '#4ecdc4' : '#888',
            backgroundColor: 'transparent',
            border: '1px solid ' + (showBrowse ? '#4ecdc4' : '#1a1a2e'),
            borderRadius: '4px',
            cursor: 'pointer',
            textAlign: 'center',
          }}
        >
          {showBrowse ? 'Hide gene browser' : 'Browse all genes'}
        </button>
      )}

      {showGeneList && (
        <div style={styles.searchResults}>
          <div style={styles.searchResultsHeader}>
            <span>
              {isSearchMode
                ? `${results.length} results`
                : page
                  ? `${page.offset + 1}\u2013${Math.min(page.offset + page.limit, page.total)} of ${page.total.toLocaleString()}`
                  : ''
              }
            </span>
            <div style={{ display: 'flex', gap: '8px' }}>
              <button style={styles.smallActionButton} onClick={selectAll}>
                All
              </button>
              <button style={styles.smallActionButton} onClick={selectNone}>
                None
              </button>
              {isSearchMode && (
                <button style={styles.smallActionButton} onClick={handleClearSearch}>
                  Clear
                </button>
              )}
            </div>
          </div>

          {displayGenes.map((gene) => {
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

          {/* Pagination controls for browse mode */}
          {!isSearchMode && page && page.total > page.limit && (
            <div style={{
              display: 'flex',
              justifyContent: 'space-between',
              alignItems: 'center',
              padding: '6px 10px',
              borderTop: '1px solid #1a1a2e',
            }}>
              <button
                onClick={() => fetchPage(Math.max(0, page.offset - page.limit))}
                disabled={page.offset === 0 || isBrowseLoading}
                style={{
                  ...styles.smallActionButton,
                  opacity: page.offset === 0 ? 0.4 : 1,
                  cursor: page.offset === 0 ? 'default' : 'pointer',
                }}
              >
                Prev
              </button>
              <span style={{ fontSize: '10px', color: '#666' }}>
                {isBrowseLoading ? 'Loading...' : `page ${Math.floor(page.offset / page.limit) + 1} of ${Math.ceil(page.total / page.limit)}`}
              </span>
              <button
                onClick={() => fetchPage(page.offset + page.limit)}
                disabled={page.offset + page.limit >= page.total || isBrowseLoading}
                style={{
                  ...styles.smallActionButton,
                  opacity: page.offset + page.limit >= page.total ? 0.4 : 1,
                  cursor: page.offset + page.limit >= page.total ? 'default' : 'pointer',
                }}
              >
                Next
              </button>
            </div>
          )}
        </div>
      )}
    </div>
  )
}

// New component for rendering gene sets within the category system
function CategoryGeneSetComponent({
  geneSet,
  categoryType,
  folderId,
  onColorByGene,
  onColorBySet,
  activeGenes,
}: {
  geneSet: GeneSet
  categoryType: GeneSetCategoryType
  folderId?: string
  onColorByGene: (gene: string) => void
  onColorBySet: (genes: string[]) => void
  activeGenes: string[]
}) {
  const [expanded, setExpanded] = useState(false)
  const [hoveredGene, setHoveredGene] = useState<string | null>(null)
  const [isDragOver, setIsDragOver] = useState(false)
  const [isEditing, setIsEditing] = useState(false)
  const [editName, setEditName] = useState(geneSet.name)
  const editInputRef = useRef<HTMLInputElement>(null)
  const {
    removeGeneSetFromCategory,
    removeGeneSetFromFolder,
    addGenesToCategorySet,
    removeGenesFromCategorySet,
    renameCategoryGeneSet,
  } = useStore()

  useEffect(() => {
    if (isEditing && editInputRef.current) {
      editInputRef.current.focus()
      editInputRef.current.select()
    }
  }, [isEditing])

  const handleRemove = () => {
    if (folderId) {
      removeGeneSetFromFolder(categoryType, folderId, geneSet.id)
    } else {
      removeGeneSetFromCategory(categoryType, geneSet.id)
    }
  }

  const handleAddGenes = (genes: string[]) => {
    addGenesToCategorySet(categoryType, geneSet.id, genes)
  }

  const handleRemoveGene = (gene: string) => {
    removeGenesFromCategorySet(categoryType, geneSet.id, [gene])
  }

  const handleRename = () => {
    const trimmedName = editName.trim()
    if (trimmedName && trimmedName !== geneSet.name) {
      renameCategoryGeneSet(categoryType, geneSet.id, trimmedName)
    }
    setIsEditing(false)
  }

  const handleDragOver = useCallback((e: React.DragEvent) => {
    if (e.dataTransfer.types.includes(GENE_DRAG_TYPE)) {
      e.preventDefault()
      e.dataTransfer.dropEffect = 'copy'
      setIsDragOver(true)
    }
  }, [])

  const handleDragLeave = useCallback(() => setIsDragOver(false), [])

  const handleDrop = useCallback(
    (e: React.DragEvent) => {
      e.preventDefault()
      setIsDragOver(false)
      const data = e.dataTransfer.getData(GENE_DRAG_TYPE)
      if (data) {
        try {
          const genes = JSON.parse(data) as string[]
          handleAddGenes(genes)
        } catch (err) {
          console.error('Failed to parse dropped genes:', err)
        }
      }
    },
    [handleAddGenes]
  )

  return (
    <div
      style={{
        ...styles.geneSet,
        ...styles.dropZone,
        ...(isDragOver ? styles.dropZoneActive : {}),
        marginBottom: '4px',
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
              onBlur={handleRename}
              onKeyDown={(e) => {
                if (e.key === 'Enter') handleRename()
                else if (e.key === 'Escape') {
                  setIsEditing(false)
                  setEditName(geneSet.name)
                }
              }}
              onClick={(e) => e.stopPropagation()}
              style={{ ...styles.searchInput, padding: '2px 6px', fontSize: '12px', width: '100%' }}
            />
          ) : (
            <>
              <span
                style={{ ...styles.geneSetName, fontSize: '12px' }}
                onDoubleClick={(e) => {
                  e.stopPropagation()
                  setEditName(geneSet.name)
                  setIsEditing(true)
                }}
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
            onClick={(e) => { e.stopPropagation(); onColorBySet(geneSet.genes) }}
            title="Color by mean expression"
          >
            🎨
          </button>
          <button
            style={styles.iconButton}
            onClick={(e) => { e.stopPropagation(); handleRemove() }}
            title="Delete gene set"
          >
            ✕
          </button>
          <span style={{ color: '#888', fontSize: '10px' }}>{expanded ? '▼' : '▶'}</span>
        </div>
      </div>
      {expanded && geneSet.genes.length > 0 && (
        <div style={styles.geneList}>
          {geneSet.genes.map((gene) => (
            <div
              key={gene}
              draggable
              onDragStart={(e) => {
                e.dataTransfer.setData(GENE_DRAG_TYPE, JSON.stringify([gene]))
                e.dataTransfer.effectAllowed = 'copy'
              }}
              style={{
                ...styles.gene,
                ...(activeGenes.includes(gene) ? styles.geneActive : {}),
                ...(hoveredGene === gene && !activeGenes.includes(gene) ? styles.geneHover : {}),
                cursor: 'grab',
              }}
              onMouseEnter={() => setHoveredGene(gene)}
              onMouseLeave={() => setHoveredGene(null)}
            >
              <span style={styles.geneName} onClick={() => onColorByGene(gene)} title="Click to color by expression">
                {gene}
              </span>
              <button
                style={{ ...styles.iconButton, fontSize: '10px' }}
                onClick={() => handleRemoveGene(gene)}
                title="Remove from set"
              >
                ✕
              </button>
            </div>
          ))}
        </div>
      )}
      {expanded && geneSet.genes.length === 0 && (
        <div style={{ ...styles.dragHint, padding: '8px 10px' }}>Drag genes here</div>
      )}
    </div>
  )
}

// Folder component
function GeneSetFolderComponent({
  folder,
  categoryType,
  onColorByGene,
  onColorBySet,
  activeGenes,
}: {
  folder: GeneSetFolder
  categoryType: GeneSetCategoryType
  onColorByGene: (gene: string) => void
  onColorBySet: (genes: string[]) => void
  activeGenes: string[]
}) {
  const { toggleFolderExpanded, removeFolder } = useStore()

  const totalGenes = folder.geneSets.reduce((sum, gs) => sum + gs.genes.length, 0)

  return (
    <div style={styles.folder}>
      <div
        style={styles.folderHeader}
        onClick={() => toggleFolderExpanded(categoryType, folder.id)}
      >
        <span style={styles.folderIcon}>{folder.expanded ? '📂' : '📁'}</span>
        <span style={styles.folderName}>{folder.name}</span>
        <span style={styles.folderCount}>
          {folder.geneSets.length} sets, {totalGenes} genes
        </span>
        <div style={styles.folderActions}>
          <button
            style={{ ...styles.iconButton, fontSize: '10px' }}
            onClick={(e) => {
              e.stopPropagation()
              removeFolder(categoryType, folder.id)
            }}
            title="Delete folder"
          >
            ✕
          </button>
        </div>
        <span style={styles.categoryExpander}>{folder.expanded ? '▼' : '▶'}</span>
      </div>
      {folder.expanded && (
        <div style={styles.folderContent}>
          {folder.geneSets.map((gs) => (
            <CategoryGeneSetComponent
              key={gs.id}
              geneSet={gs}
              categoryType={categoryType}
              folderId={folder.id}
              onColorByGene={onColorByGene}
              onColorBySet={onColorBySet}
              activeGenes={activeGenes}
            />
          ))}
        </div>
      )}
    </div>
  )
}

// Category component
function GeneSetCategoryComponent({
  category,
  onColorByGene,
  onColorBySet,
  activeGenes,
  onAddNewSet,
}: {
  category: GeneSetCategory
  onColorByGene: (gene: string) => void
  onColorBySet: (genes: string[]) => void
  activeGenes: string[]
  onAddNewSet?: () => void
}) {
  const { toggleCategoryExpanded } = useStore()

  const totalSets = category.geneSets.length + category.folders.reduce((sum, f) => sum + f.geneSets.length, 0)
  const totalGenes = category.geneSets.reduce((sum, gs) => sum + gs.genes.length, 0) +
    category.folders.reduce((sum, f) => f.geneSets.reduce((s, gs) => s + gs.genes.length, 0) + sum, 0)

  const isEmpty = totalSets === 0

  return (
    <div style={styles.category}>
      <div
        style={styles.categoryHeader}
        onClick={() => toggleCategoryExpanded(category.type)}
      >
        <span style={styles.categoryIcon}>{CATEGORY_ICONS[category.type]}</span>
        <span style={styles.categoryName}>{category.name}</span>
        {!isEmpty && (
          <span style={styles.categoryCount}>
            {totalSets} sets, {totalGenes} genes
          </span>
        )}
        {category.type === 'manual' && onAddNewSet && (
          <button
            style={{ ...styles.addButton, marginRight: '8px', padding: '2px 6px', fontSize: '10px' }}
            onClick={(e) => {
              e.stopPropagation()
              onAddNewSet()
            }}
          >
            +
          </button>
        )}
        <span style={styles.categoryExpander}>{category.expanded ? '▼' : '▶'}</span>
      </div>
      {category.expanded && (
        <div style={styles.categoryContent}>
          {/* Folders */}
          {category.folders.map((folder) => (
            <GeneSetFolderComponent
              key={folder.id}
              folder={folder}
              categoryType={category.type}
              onColorByGene={onColorByGene}
              onColorBySet={onColorBySet}
              activeGenes={activeGenes}
            />
          ))}
          {/* Direct gene sets */}
          {category.geneSets.map((gs) => (
            <CategoryGeneSetComponent
              key={gs.id}
              geneSet={gs}
              categoryType={category.type}
              onColorByGene={onColorByGene}
              onColorBySet={onColorBySet}
              activeGenes={activeGenes}
            />
          ))}
          {isEmpty && (
            <div style={styles.emptyCategory}>
              {category.type === 'manual' ? 'Click + to create a gene set' : 'No gene sets yet'}
            </div>
          )}
        </div>
      )}
    </div>
  )
}

// Helper to flatten all gene sets from hierarchical categories
function flattenGeneSets(categories: Record<GeneSetCategoryType, GeneSetCategory>): GeneSet[] {
  const allSets: GeneSet[] = []
  for (const catType of CATEGORY_ORDER) {
    const cat = categories[catType]
    // Add direct gene sets in category
    allSets.push(...cat.geneSets)
    // Add gene sets from folders
    for (const folder of cat.folders) {
      allSets.push(...folder.geneSets)
    }
  }
  return allSets
}

export default function GenePanel() {
  const { geneSetCategories, selectedGenes, bivariateData, colorMode, addGeneSet, addGeneSetToCategory, setImportModalOpen } = useStore()
  const { colorByGene, colorByGenes, clearExpressionColor, colorByBivariate, clearBivariateColor } = useDataActions()

  // Flatten gene sets for bivariate selection
  const allGeneSets = flattenGeneSets(geneSetCategories)
  const [newSetName, setNewSetName] = useState('')
  const [showNewSetInput, setShowNewSetInput] = useState(false)
  const [selectedSearchGenes, setSelectedSearchGenes] = useState<Set<string>>(new Set())
  const [bivariateSet1, setBivariateSet1] = useState<string | null>(null)
  const [bivariateSet2, setBivariateSet2] = useState<string | null>(null)

  // Find Similar Genes state
  const [hasGeneNeighbors, setHasGeneNeighbors] = useState(false)
  const [similarGenesSeed, setSimilarGenesSeed] = useState('')
  const [numSimilarGenes, setNumSimilarGenes] = useState(10)
  const [findingSimilarGenes, setFindingSimilarGenes] = useState(false)
  const [similarGenesError, setSimilarGenesError] = useState<string | null>(null)

  // Check prerequisites for find_similar_genes on mount and periodically
  useEffect(() => {
    const checkPrerequisites = async () => {
      try {
        const response = await fetch(`${API_BASE}/scanpy/prerequisites/find_similar_genes`)
        if (response.ok) {
          const data = await response.json()
          setHasGeneNeighbors(data.satisfied)
        }
      } catch {
        // Silently fail - feature just won't be shown
      }
    }

    checkPrerequisites()
    // Re-check every 5 seconds in case user runs gene_neighbors from ScanpyModal
    const interval = setInterval(checkPrerequisites, 5000)
    return () => clearInterval(interval)
  }, [])

  const handleFindSimilarGenes = async () => {
    if (!similarGenesSeed.trim()) return

    setFindingSimilarGenes(true)
    setSimilarGenesError(null)

    try {
      const response = await fetch(`${API_BASE}/scanpy/find_similar_genes`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          gene: similarGenesSeed.trim(),
          n_neighbors: numSimilarGenes,
        }),
      })

      if (!response.ok) {
        const errorData = await response.json()
        throw new Error(errorData.detail || 'Failed to find similar genes')
      }

      const data = await response.json()
      // data.similar_genes is an array of gene names
      const similarGenes = data.similar_genes as string[]

      // Create gene set with seed gene + similar genes in the similar_genes category
      const setName = `Similar to ${similarGenesSeed.trim()}`
      addGeneSetToCategory('similar_genes', setName, [similarGenesSeed.trim(), ...similarGenes])

      // Clear input after success
      setSimilarGenesSeed('')
    } catch (err) {
      setSimilarGenesError(err instanceof Error ? err.message : 'Unknown error')
    } finally {
      setFindingSimilarGenes(false)
    }
  }

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
        <div style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
          <div style={styles.title}>Genes</div>
          <button
            onClick={() => setImportModalOpen(true)}
            style={{
              padding: '2px 8px',
              fontSize: '10px',
              backgroundColor: '#0f3460',
              color: '#aaa',
              border: '1px solid #1a1a2e',
              borderRadius: '3px',
              cursor: 'pointer',
            }}
            title="Import gene lists from file (.gmt, .csv, .txt)"
          >
            Import
          </button>
        </div>
        <GeneSearch
          onColorByGene={colorByGene}
          selectedSearchGenes={selectedSearchGenes}
          setSelectedSearchGenes={setSelectedSearchGenes}
        />
      </div>
      <ImportModal />

      {selectedGenes.length > 0 && colorMode === 'expression' && (
        <div style={styles.selectedGenesBar}>
          <span style={styles.selectedGenesText}>
            Coloring by: {selectedGenes.length === 1 ? selectedGenes[0] : `${selectedGenes.length} genes (mean)`}
          </span>
          <button style={styles.clearButton} onClick={clearExpressionColor}>
            Clear
          </button>
        </div>
      )}

      {colorMode === 'bivariate' && bivariateData && (
        <div style={styles.selectedGenesBar}>
          <span style={styles.selectedGenesText}>
            Bivariate: {bivariateData.genes1.length} × {bivariateData.genes2.length} genes
          </span>
          <button style={styles.clearButton} onClick={clearBivariateColor}>
            Clear
          </button>
        </div>
      )}

      <div style={styles.content}>
        {/* New Set Input (for manual category) */}
        {showNewSetInput && (
          <div style={{ marginBottom: '12px', display: 'flex', gap: '4px' }}>
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

        {/* Gene Set Categories */}
        {CATEGORY_ORDER.map((catType) => (
          <GeneSetCategoryComponent
            key={catType}
            category={geneSetCategories[catType]}
            onColorByGene={colorByGene}
            onColorBySet={colorByGenes}
            activeGenes={selectedGenes}
            onAddNewSet={catType === 'manual' ? () => setShowNewSetInput(true) : undefined}
          />
        ))}

        {/* Find Similar Genes Section - only shown when gene_neighbors exists */}
        {hasGeneNeighbors && (
          <div style={styles.section}>
            <div style={styles.sectionHeader}>
              <span style={styles.sectionTitle}>Find Similar Genes</span>
            </div>
            <div style={{
              backgroundColor: '#0f3460',
              borderRadius: '4px',
              padding: '12px',
            }}>
              <div style={{ marginBottom: '8px' }}>
                <label style={{ fontSize: '11px', color: '#888', display: 'block', marginBottom: '4px' }}>
                  Seed Gene
                </label>
                <input
                  type="text"
                  placeholder="Enter gene name..."
                  value={similarGenesSeed}
                  onChange={(e) => setSimilarGenesSeed(e.target.value)}
                  onKeyDown={(e) => {
                    if (e.key === 'Enter') handleFindSimilarGenes()
                  }}
                  style={{
                    ...styles.searchInput,
                    width: '100%',
                  }}
                />
              </div>
              <div style={{ marginBottom: '12px' }}>
                <label style={{ fontSize: '11px', color: '#888', display: 'block', marginBottom: '4px' }}>
                  Number of Similar Genes
                </label>
                <input
                  type="number"
                  min={1}
                  max={100}
                  value={numSimilarGenes}
                  onChange={(e) => setNumSimilarGenes(Math.max(1, parseInt(e.target.value) || 10))}
                  style={{
                    ...styles.searchInput,
                    width: '80px',
                  }}
                />
              </div>
              {similarGenesError && (
                <div style={{
                  fontSize: '11px',
                  color: '#e94560',
                  marginBottom: '8px',
                  padding: '6px 8px',
                  backgroundColor: 'rgba(233, 69, 96, 0.1)',
                  borderRadius: '4px',
                }}>
                  {similarGenesError}
                </div>
              )}
              <button
                onClick={handleFindSimilarGenes}
                disabled={!similarGenesSeed.trim() || findingSimilarGenes}
                style={{
                  width: '100%',
                  padding: '8px',
                  fontSize: '12px',
                  fontWeight: 500,
                  backgroundColor: similarGenesSeed.trim() && !findingSimilarGenes ? '#4ecdc4' : '#1a1a2e',
                  color: similarGenesSeed.trim() && !findingSimilarGenes ? '#000' : '#666',
                  border: 'none',
                  borderRadius: '4px',
                  cursor: similarGenesSeed.trim() && !findingSimilarGenes ? 'pointer' : 'not-allowed',
                }}
              >
                {findingSimilarGenes ? 'Finding...' : 'Find Similar Genes'}
              </button>
              <div style={{ fontSize: '10px', color: '#666', marginTop: '8px', textAlign: 'center' }}>
                Finds genes with similar expression patterns based on gene embeddings
              </div>
            </div>
          </div>
        )}

        {/* Bivariate Mode Section */}
        {allGeneSets.filter(gs => gs.genes.length > 0).length >= 2 && (
          <div style={styles.section}>
            <div style={styles.sectionHeader}>
              <span style={styles.sectionTitle}>Bivariate Coloring</span>
            </div>
            <div style={{
              backgroundColor: '#0f3460',
              borderRadius: '4px',
              padding: '12px',
            }}>
              <div style={{ marginBottom: '8px' }}>
                <label style={{ fontSize: '11px', color: '#888', display: 'block', marginBottom: '4px' }}>
                  Set 1 <span style={{ color: '#e31a1c' }}>(→ Red)</span>
                </label>
                <select
                  value={bivariateSet1 || ''}
                  onChange={(e) => setBivariateSet1(e.target.value || null)}
                  style={{
                    width: '100%',
                    padding: '6px 8px',
                    fontSize: '12px',
                    backgroundColor: '#1a1a2e',
                    color: '#eee',
                    border: '1px solid #0f3460',
                    borderRadius: '4px',
                    cursor: 'pointer',
                  }}
                >
                  <option value="">Select gene set...</option>
                  {allGeneSets.filter(gs => gs.genes.length > 0 && gs.id !== bivariateSet2).map((gs) => (
                    <option key={gs.id} value={gs.id}>
                      {gs.name} ({gs.genes.length} genes)
                    </option>
                  ))}
                </select>
              </div>
              <div style={{ marginBottom: '12px' }}>
                <label style={{ fontSize: '11px', color: '#888', display: 'block', marginBottom: '4px' }}>
                  Set 2 <span style={{ color: '#1f78b4' }}>(↑ Blue)</span>
                </label>
                <select
                  value={bivariateSet2 || ''}
                  onChange={(e) => setBivariateSet2(e.target.value || null)}
                  style={{
                    width: '100%',
                    padding: '6px 8px',
                    fontSize: '12px',
                    backgroundColor: '#1a1a2e',
                    color: '#eee',
                    border: '1px solid #0f3460',
                    borderRadius: '4px',
                    cursor: 'pointer',
                  }}
                >
                  <option value="">Select gene set...</option>
                  {allGeneSets.filter(gs => gs.genes.length > 0 && gs.id !== bivariateSet1).map((gs) => (
                    <option key={gs.id} value={gs.id}>
                      {gs.name} ({gs.genes.length} genes)
                    </option>
                  ))}
                </select>
              </div>
              <button
                onClick={() => {
                  const set1 = allGeneSets.find(gs => gs.id === bivariateSet1)
                  const set2 = allGeneSets.find(gs => gs.id === bivariateSet2)
                  if (set1 && set2) {
                    colorByBivariate(set1.genes, set2.genes)
                  }
                }}
                disabled={!bivariateSet1 || !bivariateSet2}
                style={{
                  width: '100%',
                  padding: '8px',
                  fontSize: '12px',
                  fontWeight: 500,
                  backgroundColor: bivariateSet1 && bivariateSet2 ? '#4ecdc4' : '#1a1a2e',
                  color: bivariateSet1 && bivariateSet2 ? '#000' : '#666',
                  border: 'none',
                  borderRadius: '4px',
                  cursor: bivariateSet1 && bivariateSet2 ? 'pointer' : 'not-allowed',
                }}
              >
                Apply Bivariate Coloring
              </button>
              <div style={{ fontSize: '10px', color: '#666', marginTop: '8px', textAlign: 'center' }}>
                Colors cells by expression of both gene sets simultaneously
              </div>
            </div>
          </div>
        )}
      </div>
    </div>
  )
}
