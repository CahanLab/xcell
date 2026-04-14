import React, { useState, useRef, useEffect, useCallback } from 'react'
import { useStore, GeneSet, GeneSetCategory, GeneSetFolder, GeneSetCategoryType } from '../store'
import { useGeneSearch, useGeneBrowse, useDataActions, appendDataset, fetchVarIdentifierColumns, swapVarIndex } from '../hooks/useData'
import { exportFolderAsJson, exportFolderAsGmt, exportFolderAsCsv } from '../utils/exportGeneSets'
import ImportModal from './ImportModal'

const API_BASE = '/api'

// Drag data type for genes
const GENE_DRAG_TYPE = 'application/x-gene'
const GENE_SET_DRAG_TYPE = 'application/x-gene-set'
const GENE_SET_FOLDER_DRAG_TYPE = 'application/x-gene-set-folder'

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

// Sort helper: pinned items first (stable), unpinned items after (stable).
// Used for gene sets and folders alike — any item with a `pinned?: boolean`
// property can pass through this.
function pinSort<T extends { pinned?: boolean }>(items: T[]): T[] {
  const pinned: T[] = []
  const rest: T[] = []
  for (const item of items) {
    if (item.pinned) pinned.push(item)
    else rest.push(item)
  }
  return [...pinned, ...rest]
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
    padding: '8px 12px',
    borderBottom: '1px solid #0f3460',
  },
  title: {
    fontSize: '14px',
    fontWeight: 600,
    color: '#e94560',
    marginBottom: '0',
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
    padding: '8px 12px',
  },
  section: {
    marginBottom: '10px',
  },
  sectionHeader: {
    display: 'flex',
    alignItems: 'center',
    justifyContent: 'space-between',
    marginBottom: '4px',
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
    marginBottom: '4px',
    overflow: 'hidden',
  },
  geneSetHeader: {
    display: 'flex',
    alignItems: 'center',
    justifyContent: 'space-between',
    padding: '5px 10px',
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
    padding: '2px 10px 4px',
    borderTop: '1px solid #1a1a2e',
  },
  gene: {
    display: 'flex',
    alignItems: 'center',
    justifyContent: 'space-between',
    padding: '2px 6px',
    marginBottom: '1px',
    borderRadius: '3px',
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
    padding: '5px 12px',
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
    padding: '3px 10px',
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
    marginBottom: '6px',
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
    padding: '4px 10px',
    fontStyle: 'italic',
  },
}

// ---------------------------------------------------------------------------
// OverflowMenu — a "⋯" button that opens a dropdown with action items.
// Supports optional nested children (rendered inline with extra indent).
// Used by gene-set rows (Tasks 6 & 7) to consolidate pin / export actions.
// ---------------------------------------------------------------------------
interface OverflowMenuItem {
  label: string
  onClick?: () => void
  disabled?: boolean
  tooltip?: string
  children?: OverflowMenuItem[]
}

export function OverflowMenu({ items }: { items: OverflowMenuItem[] }) {
  const [open, setOpen] = useState(false)
  const [hoveredParent, setHoveredParent] = useState<string | null>(null)
  // Viewport coordinates for the popup — populated when the menu opens.
  // Using fixed positioning (instead of absolute) lets the popup escape
  // ancestors with `overflow: hidden` (e.g. the gene set container).
  const [popupPos, setPopupPos] = useState<{ top: number; right: number } | null>(null)
  const menuRef = useRef<HTMLDivElement>(null)
  const buttonRef = useRef<HTMLButtonElement>(null)

  useEffect(() => {
    if (!open) return
    const onDocClick = (e: MouseEvent) => {
      if (menuRef.current && !menuRef.current.contains(e.target as Node)) {
        setOpen(false)
      }
    }
    const onKey = (e: KeyboardEvent) => {
      if (e.key === 'Escape') setOpen(false)
    }
    // Close on scroll so the popup doesn't float away from its trigger.
    const onScroll = () => setOpen(false)
    document.addEventListener('mousedown', onDocClick)
    document.addEventListener('keydown', onKey)
    window.addEventListener('scroll', onScroll, true)
    return () => {
      document.removeEventListener('mousedown', onDocClick)
      document.removeEventListener('keydown', onKey)
      window.removeEventListener('scroll', onScroll, true)
    }
  }, [open])

  const renderItem = (item: OverflowMenuItem, depth: number = 0): React.ReactElement => {
    const hasChildren = !!item.children && item.children.length > 0
    const isDisabled = !!item.disabled
    return (
      <div
        key={item.label}
        onMouseEnter={() => hasChildren && setHoveredParent(item.label)}
        onMouseLeave={() => hasChildren && setHoveredParent(null)}
        style={{ position: 'relative' }}
      >
        <button
          style={{
            width: '100%',
            padding: `4px 8px`,
            paddingLeft: `${8 + depth * 8}px`,
            fontSize: '11px',
            backgroundColor: 'transparent',
            color: isDisabled ? '#555' : '#ccc',
            border: 'none',
            cursor: isDisabled ? 'not-allowed' : 'pointer',
            textAlign: 'left',
            display: 'flex',
            alignItems: 'center',
            justifyContent: 'space-between',
          }}
          disabled={isDisabled}
          title={item.tooltip}
          onClick={(e) => {
            e.stopPropagation()
            if (isDisabled) return
            if (hasChildren) return
            item.onClick?.()
            setOpen(false)
          }}
        >
          <span>{item.label}</span>
          {hasChildren && <span style={{ marginLeft: '8px', color: '#666' }}>▸</span>}
        </button>
        {hasChildren && hoveredParent === item.label && (
          <div
            style={{
              backgroundColor: '#0f1625',
              borderTop: '1px solid #0f3460',
              borderBottom: '1px solid #0f3460',
            }}
          >
            {item.children!.map((child) => renderItem(child, depth + 1))}
          </div>
        )}
      </div>
    )
  }

  return (
    <div ref={menuRef} style={{ position: 'relative' }}>
      <button
        ref={buttonRef}
        style={{ ...styles.iconButton, color: '#888' }}
        onClick={(e) => {
          e.stopPropagation()
          setOpen((v) => {
            const next = !v
            if (next && buttonRef.current) {
              const rect = buttonRef.current.getBoundingClientRect()
              setPopupPos({
                top: rect.bottom + 2,
                right: window.innerWidth - rect.right,
              })
            } else {
              setPopupPos(null)
            }
            return next
          })
        }}
        title="More actions"
      >
        ⋯
      </button>
      {open && popupPos && (
        <div
          style={{
            position: 'fixed',
            top: popupPos.top,
            right: popupPos.right,
            backgroundColor: '#1a1a2e',
            border: '1px solid #0f3460',
            borderRadius: '4px',
            padding: '4px',
            display: 'flex',
            flexDirection: 'column',
            gap: '2px',
            zIndex: 1000,
            minWidth: '120px',
          }}
          onClick={(e) => e.stopPropagation()}
        >
          {items.map((item) => renderItem(item))}
        </div>
      )}
    </div>
  )
}

interface GeneSearchProps {
  onColorByGene: (gene: string) => void
  selectedSearchGenes: Set<string>
  setSelectedSearchGenes: React.Dispatch<React.SetStateAction<Set<string>>>
  showBrowse: boolean
  setShowBrowse: React.Dispatch<React.SetStateAction<boolean>>
}

function GeneSearch({ onColorByGene, selectedSearchGenes, setSelectedSearchGenes, showBrowse, setShowBrowse }: GeneSearchProps) {
  const [query, setQuery] = useState('')
  const [hoveredGene, setHoveredGene] = useState<string | null>(null)
  const inputRef = useRef<HTMLInputElement>(null)
  const { results, searchGenes, clearResults } = useGeneSearch()
  const { page, isLoading: isBrowseLoading, fetchPage } = useGeneBrowse(50)
  const setSelectByExpressionSource = useStore((s) => s.setSelectByExpressionSource)

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
                <OverflowMenu
                  items={[
                    {
                      label: 'Select cells…',
                      onClick: () => setSelectByExpressionSource({ type: 'gene', gene }),
                    },
                  ]}
                />
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
  onColorBySet: (genes: string[], geneSetName?: string) => void
  activeGenes: string[]
}) {
  const [expanded, setExpanded] = useState(false)
  const [hoveredGene, setHoveredGene] = useState<string | null>(null)
  const [isDragOver, setIsDragOver] = useState(false)
  const [isEditing, setIsEditing] = useState(false)
  const [editName, setEditName] = useState(geneSet.name)
  const [dropIndicator, setDropIndicator] = useState<'above' | 'below' | null>(null)
  const editInputRef = useRef<HTMLInputElement>(null)
  const rowRef = useRef<HTMLDivElement>(null)
  const {
    removeGeneSetFromCategory,
    removeGeneSetFromFolder,
    addGenesToCategorySet,
    removeGenesFromCategorySet,
    renameCategoryGeneSet,
    toggleSetPinned,
    moveGeneSetToFolder,
    reorderGeneSet,
    setClusterModalSourceSet,
  } = useStore()
  const setSelectByExpressionSource = useStore((s) => s.setSelectByExpressionSource)

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

  const handleDragOver = useCallback(
    (e: React.DragEvent) => {
      if (e.dataTransfer.types.includes(GENE_DRAG_TYPE)) {
        e.preventDefault()
        e.dataTransfer.dropEffect = 'copy'
        setIsDragOver(true)
        return
      }
      if (
        categoryType === 'manual' &&
        e.dataTransfer.types.includes(GENE_SET_DRAG_TYPE)
      ) {
        e.preventDefault()
        e.dataTransfer.dropEffect = 'move'
        const rect = rowRef.current?.getBoundingClientRect()
        if (rect) {
          const midY = rect.top + rect.height / 2
          setDropIndicator(e.clientY < midY ? 'above' : 'below')
        }
      }
    },
    [categoryType]
  )

  const handleDragLeave = useCallback(() => {
    setIsDragOver(false)
    setDropIndicator(null)
  }, [])

  const handleDrop = useCallback(
    (e: React.DragEvent) => {
      // Gene drop (into set) — existing behavior
      const geneData = e.dataTransfer.getData(GENE_DRAG_TYPE)
      if (geneData) {
        e.preventDefault()
        setIsDragOver(false)
        try {
          const genes = JSON.parse(geneData) as string[]
          handleAddGenes(genes)
        } catch (err) {
          console.error('Failed to parse dropped genes:', err)
        }
        return
      }
      // Gene set drop (reorder or cross-container move)
      if (categoryType !== 'manual') return
      const setData = e.dataTransfer.getData(GENE_SET_DRAG_TYPE)
      if (!setData) return
      e.preventDefault()
      e.stopPropagation()
      const indicator = dropIndicator
      setDropIndicator(null)
      try {
        const { setId: draggedId, sourceFolderId } = JSON.parse(setData) as {
          setId: string
          sourceFolderId: string | null
        }
        if (draggedId === geneSet.id) return
        const targetFolderId = folderId ?? null
        if (sourceFolderId === targetFolderId) {
          // Same container → reorder.
          // Compute insertion index in POST-removal coordinates:
          //   1. Look up source idx and target idx in the CURRENT stored array.
          //   2. The "visible" slot is targetIdx (above) or targetIdx + 1 (below).
          //   3. If the source is earlier than that slot, subtract 1 to account
          //      for the source being removed before insertion.
          //   4. The store action splices the result directly.
          const store = useStore.getState()
          const container =
            targetFolderId === null
              ? store.geneSetCategories.manual.geneSets
              : store.geneSetCategories.manual.folders.find((f) => f.id === targetFolderId)?.geneSets ?? []
          const targetIdx = container.findIndex((gs) => gs.id === geneSet.id)
          const sourceIdx = container.findIndex((gs) => gs.id === draggedId)
          if (targetIdx === -1 || sourceIdx === -1) return
          // Respect pin boundary: don't cross between pinned and unpinned
          const dragged = container[sourceIdx]
          if (!!dragged.pinned !== !!geneSet.pinned) return
          let insertAt = indicator === 'below' ? targetIdx + 1 : targetIdx
          if (sourceIdx < insertAt) insertAt -= 1
          if (insertAt === sourceIdx) return // No-op
          reorderGeneSet(categoryType, targetFolderId, draggedId, insertAt)
        } else {
          // Cross-container → move (indicator is ignored; move appends)
          moveGeneSetToFolder(categoryType, draggedId, targetFolderId, sourceFolderId)
        }
      } catch (err) {
        console.error('Failed to parse gene set drop:', err)
      }
    },
    [categoryType, folderId, geneSet.id, geneSet.pinned, handleAddGenes, dropIndicator, moveGeneSetToFolder, reorderGeneSet]
  )

  return (
    <div style={{ position: 'relative' }}>
      {dropIndicator === 'above' && (
        <div
          style={{
            position: 'absolute',
            top: -1,
            left: 0,
            right: 0,
            height: '2px',
            backgroundColor: '#4ecdc4',
            pointerEvents: 'none',
            zIndex: 2,
          }}
        />
      )}
      <div
        ref={rowRef}
        style={{
          ...styles.geneSet,
          ...styles.dropZone,
          ...(isDragOver ? styles.dropZoneActive : {}),
          marginBottom: '4px',
        }}
        draggable={categoryType === 'manual'}
        onDragStart={(e) => {
          if (categoryType !== 'manual') return
          const payload = JSON.stringify({
            setId: geneSet.id,
            sourceFolderId: folderId ?? null,
          })
          e.dataTransfer.setData(GENE_SET_DRAG_TYPE, payload)
          e.dataTransfer.effectAllowed = 'move'
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
            onClick={(e) => { e.stopPropagation(); onColorBySet(geneSet.genes, geneSet.name) }}
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
          <OverflowMenu
            items={[
              {
                label: geneSet.pinned ? 'Unpin' : 'Pin to top',
                onClick: () => toggleSetPinned(categoryType, folderId ?? null, geneSet.id),
              },
              {
                label: 'Cluster genes…',
                onClick: () =>
                  setClusterModalSourceSet({
                    name: geneSet.name,
                    genes: geneSet.genes,
                    categoryType,
                    folderId: folderId ?? null,
                  }),
                disabled: geneSet.genes.length < 4,
                tooltip: geneSet.genes.length < 4 ? 'Need at least 4 genes to cluster' : undefined,
              },
              {
                label: 'Select cells…',
                onClick: () =>
                  setSelectByExpressionSource({
                    type: 'geneSet',
                    name: geneSet.name,
                    genes: geneSet.genes,
                  }),
                disabled: geneSet.genes.length === 0,
                tooltip: geneSet.genes.length === 0 ? 'Gene set is empty' : undefined,
              },
            ]}
          />
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
              <OverflowMenu
                items={[
                  {
                    label: 'Select cells…',
                    onClick: () => setSelectByExpressionSource({ type: 'gene', gene }),
                  },
                ]}
              />
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
      {dropIndicator === 'below' && (
        <div
          style={{
            position: 'absolute',
            bottom: -1,
            left: 0,
            right: 0,
            height: '2px',
            backgroundColor: '#4ecdc4',
            pointerEvents: 'none',
            zIndex: 2,
          }}
        />
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
  allowAddSet,
}: {
  folder: GeneSetFolder
  categoryType: GeneSetCategoryType
  onColorByGene: (gene: string) => void
  onColorBySet: (genes: string[], geneSetName?: string) => void
  activeGenes: string[]
  allowAddSet?: boolean
}) {
  const { toggleFolderExpanded, removeFolder, toggleFolderPinned, renameFolder, addGeneSetToFolder, moveGeneSetToFolder, reorderFolder } = useStore()

  const [isEditing, setIsEditing] = useState(false)
  const [editName, setEditName] = useState(folder.name)
  const editInputRef = useRef<HTMLInputElement>(null)
  const [showAddSet, setShowAddSet] = useState(false)
  const [newSetNameInFolder, setNewSetNameInFolder] = useState('')
  const [isSetDragOver, setIsSetDragOver] = useState(false)
  const [folderDropIndicator, setFolderDropIndicator] = useState<'above' | 'below' | null>(null)
  const folderRowRef = useRef<HTMLDivElement>(null)

  const handleSetDragOver = (e: React.DragEvent) => {
    if (categoryType !== 'manual') return
    if (!e.dataTransfer.types.includes(GENE_SET_DRAG_TYPE)) return
    e.preventDefault()
    e.dataTransfer.dropEffect = 'move'
    setIsSetDragOver(true)
  }

  const handleSetDragLeave = () => setIsSetDragOver(false)

  const handleSetDrop = (e: React.DragEvent) => {
    if (categoryType !== 'manual') return
    const raw = e.dataTransfer.getData(GENE_SET_DRAG_TYPE)
    if (!raw) return
    e.preventDefault()
    setIsSetDragOver(false)
    try {
      const { setId, sourceFolderId } = JSON.parse(raw) as {
        setId: string
        sourceFolderId: string | null
      }
      if (sourceFolderId === folder.id) return // No-op — already here.
      moveGeneSetToFolder(categoryType, setId, folder.id, sourceFolderId)
    } catch (err) {
      console.error('Failed to parse gene set drop:', err)
    }
  }

  const handleFolderDragOver = (e: React.DragEvent) => {
    if (!e.dataTransfer.types.includes(GENE_SET_FOLDER_DRAG_TYPE)) return
    e.preventDefault()
    e.dataTransfer.dropEffect = 'move'
    const rect = folderRowRef.current?.getBoundingClientRect()
    if (rect) {
      const midY = rect.top + rect.height / 2
      setFolderDropIndicator(e.clientY < midY ? 'above' : 'below')
    }
  }

  const handleFolderDrop = (e: React.DragEvent) => {
    const raw = e.dataTransfer.getData(GENE_SET_FOLDER_DRAG_TYPE)
    if (!raw) return
    e.preventDefault()
    const indicator = folderDropIndicator
    setFolderDropIndicator(null)
    try {
      const { folderId: draggedId } = JSON.parse(raw) as { folderId: string }
      if (draggedId === folder.id) return
      const store = useStore.getState()
      const siblings = store.geneSetCategories[categoryType].folders
      const sourceIdx = siblings.findIndex((f) => f.id === draggedId)
      if (sourceIdx === -1) return // Cross-category drop — reject.
      const targetIdx = siblings.findIndex((f) => f.id === folder.id)
      if (targetIdx === -1) return
      // Respect pin boundary.
      const draggedSibling = siblings[sourceIdx]
      if (!!draggedSibling.pinned !== !!folder.pinned) return
      // Compute post-removal insertion index.
      let insertAt = indicator === 'below' ? targetIdx + 1 : targetIdx
      if (sourceIdx < insertAt) insertAt -= 1
      if (insertAt === sourceIdx) return
      reorderFolder(categoryType, draggedId, insertAt)
    } catch (err) {
      console.error('Failed to parse folder drop:', err)
    }
  }

  useEffect(() => {
    if (isEditing && editInputRef.current) {
      editInputRef.current.focus()
      editInputRef.current.select()
    }
  }, [isEditing])

  const handleRename = () => {
    const trimmed = editName.trim()
    if (trimmed && trimmed !== folder.name) {
      renameFolder(categoryType, folder.id, trimmed)
    }
    setIsEditing(false)
  }

  const handleCreateSetInFolder = () => {
    const trimmed = newSetNameInFolder.trim()
    if (trimmed) {
      addGeneSetToFolder(categoryType, folder.id, trimmed, [])
      setNewSetNameInFolder('')
      setShowAddSet(false)
    }
  }

  const totalGenes = folder.geneSets.reduce((sum, gs) => sum + gs.genes.length, 0)

  return (
    <div
      style={{
        ...styles.folder,
        position: 'relative',
        border: isSetDragOver ? '2px dashed #4ecdc4' : '2px dashed transparent',
        borderRadius: '4px',
      }}
      draggable
      onDragStart={(e) => {
        e.stopPropagation()
        const payload = JSON.stringify({ folderId: folder.id })
        e.dataTransfer.setData(GENE_SET_FOLDER_DRAG_TYPE, payload)
        e.dataTransfer.effectAllowed = 'move'
      }}
      onDragOver={(e) => {
        handleSetDragOver(e)
        handleFolderDragOver(e)
      }}
      onDragLeave={() => {
        handleSetDragLeave()
        setFolderDropIndicator(null)
      }}
      onDrop={(e) => {
        handleSetDrop(e)
        handleFolderDrop(e)
      }}
    >
      {folderDropIndicator === 'above' && (
        <div
          style={{
            position: 'absolute',
            top: -1,
            left: 0,
            right: 0,
            height: '2px',
            backgroundColor: '#4ecdc4',
            pointerEvents: 'none',
            zIndex: 2,
          }}
        />
      )}
      <div
        ref={folderRowRef}
        style={styles.folderHeader}
        onClick={() => {
          if (isEditing) return
          toggleFolderExpanded(categoryType, folder.id)
        }}
      >
        <span style={styles.folderIcon}>{folder.expanded ? '📂' : '📁'}</span>
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
                setEditName(folder.name)
              }
            }}
            onClick={(e) => e.stopPropagation()}
            style={{ ...styles.searchInput, padding: '2px 6px', fontSize: '11px', flex: 1, marginRight: '6px' }}
          />
        ) : (
          <span
            style={styles.folderName}
            onDoubleClick={(e) => {
              if (categoryType !== 'manual') return
              e.stopPropagation()
              setEditName(folder.name)
              setIsEditing(true)
            }}
            title={categoryType === 'manual' ? 'Double-click to rename' : undefined}
          >
            {folder.name}
          </span>
        )}
        <span style={styles.folderCount}>
          {folder.geneSets.length} sets, {totalGenes} genes
        </span>
        <div style={styles.folderActions}>
          {allowAddSet && categoryType === 'manual' && (
            <button
              style={{ ...styles.iconButton, fontSize: '11px', color: '#888' }}
              onClick={(e) => {
                e.stopPropagation()
                setShowAddSet(true)
              }}
              title="New gene set in this folder"
            >
              +
            </button>
          )}
          <button
            style={{ ...styles.iconButton, fontSize: '10px' }}
            onClick={(e) => {
              e.stopPropagation()
              const hasContents = folder.geneSets.length > 0
              if (categoryType === 'manual' && hasContents) {
                const n = folder.geneSets.length
                const plural = n === 1 ? 'gene set' : 'gene sets'
                const ok = window.confirm(
                  `Delete folder '${folder.name}' and its ${n} ${plural}? This cannot be undone.`
                )
                if (!ok) return
              }
              removeFolder(categoryType, folder.id)
            }}
            title="Delete folder"
          >
            ✕
          </button>
          <OverflowMenu
            items={
              categoryType === 'manual'
                ? [
                    {
                      label: folder.pinned ? 'Unpin folder' : 'Pin folder to top',
                      onClick: () => toggleFolderPinned(categoryType, folder.id),
                    },
                    {
                      label: 'Export ▸',
                      children: [
                        {
                          label: 'JSON',
                          onClick: () => exportFolderAsJson(folder.name, pinSort(folder.geneSets)),
                        },
                        {
                          label: 'GMT',
                          onClick: () => exportFolderAsGmt(folder.name, pinSort(folder.geneSets)),
                        },
                        {
                          label: 'CSV',
                          onClick: () => exportFolderAsCsv(folder.name, pinSort(folder.geneSets)),
                        },
                      ],
                    },
                  ]
                : [
                    {
                      label: folder.pinned ? 'Unpin folder' : 'Pin folder to top',
                      onClick: () => toggleFolderPinned(categoryType, folder.id),
                    },
                  ]
            }
          />
        </div>
        <span style={styles.categoryExpander}>{folder.expanded ? '▼' : '▶'}</span>
      </div>
      {folder.expanded && (
        <div style={styles.folderContent}>
          {showAddSet && (
            <div style={{ marginBottom: '4px', display: 'flex', gap: '4px' }}>
              <input
                type="text"
                placeholder="Set name..."
                value={newSetNameInFolder}
                onChange={(e) => setNewSetNameInFolder(e.target.value)}
                onKeyDown={(e) => {
                  if (e.key === 'Enter') handleCreateSetInFolder()
                  if (e.key === 'Escape') {
                    setNewSetNameInFolder('')
                    setShowAddSet(false)
                  }
                }}
                style={{ ...styles.searchInput, flex: 1, padding: '3px 6px', fontSize: '11px' }}
                autoFocus
              />
              <button
                style={{ ...styles.addButton, padding: '3px 8px', fontSize: '10px' }}
                onClick={handleCreateSetInFolder}
              >
                Add
              </button>
            </div>
          )}
          {pinSort(folder.geneSets).map((gs) => (
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
      {folderDropIndicator === 'below' && (
        <div
          style={{
            position: 'absolute',
            bottom: -1,
            left: 0,
            right: 0,
            height: '2px',
            backgroundColor: '#4ecdc4',
            pointerEvents: 'none',
            zIndex: 2,
          }}
        />
      )}
    </div>
  )
}

// Drop strip for moving gene sets to the top level of the manual category
function ManualTopLevelDropStrip() {
  const [isDragOver, setIsDragOver] = useState(false)
  const { moveGeneSetToFolder } = useStore()

  return (
    <div
      onDragOver={(e) => {
        if (!e.dataTransfer.types.includes(GENE_SET_DRAG_TYPE)) return
        e.preventDefault()
        e.dataTransfer.dropEffect = 'move'
        setIsDragOver(true)
      }}
      onDragLeave={() => setIsDragOver(false)}
      onDrop={(e) => {
        const raw = e.dataTransfer.getData(GENE_SET_DRAG_TYPE)
        if (!raw) return
        e.preventDefault()
        setIsDragOver(false)
        try {
          const { setId, sourceFolderId } = JSON.parse(raw) as {
            setId: string
            sourceFolderId: string | null
          }
          if (sourceFolderId === null) return // Already top-level.
          moveGeneSetToFolder('manual', setId, null, sourceFolderId)
        } catch (err) {
          console.error('Failed to parse top-level drop:', err)
        }
      }}
      style={{
        height: '12px',
        marginBottom: '4px',
        borderRadius: '4px',
        backgroundColor: isDragOver ? 'rgba(78, 205, 196, 0.2)' : 'transparent',
        border: isDragOver ? '2px dashed #4ecdc4' : '2px dashed transparent',
        transition: 'all 0.1s',
      }}
    />
  )
}

// Category component
function HiddenCategoriesFooter() {
  const [expanded, setExpanded] = useState(false)
  const geneSetCategories = useStore((s) => s.geneSetCategories)
  const toggleCategoryVisible = useStore((s) => s.toggleCategoryVisible)

  const hidden = CATEGORY_ORDER.filter((t) => geneSetCategories[t].visible === false)
  if (hidden.length === 0) return null

  return (
    <div
      style={{
        marginTop: '8px',
        padding: '6px 10px',
        backgroundColor: '#0a0f1a',
        borderRadius: '4px',
        fontSize: '11px',
        color: '#888',
      }}
    >
      <div
        style={{ cursor: 'pointer', userSelect: 'none' }}
        onClick={() => setExpanded((v) => !v)}
      >
        {hidden.length} hidden {expanded ? '▾' : '▸'}
      </div>
      {expanded && (
        <div style={{ marginTop: '6px', display: 'flex', flexDirection: 'column', gap: '4px' }}>
          {hidden.map((catType) => (
            <div
              key={catType}
              style={{
                display: 'flex',
                alignItems: 'center',
                justifyContent: 'space-between',
                gap: '8px',
              }}
            >
              <span>
                {CATEGORY_ICONS[catType]} {geneSetCategories[catType].name}
              </span>
              <button
                style={{
                  ...styles.addButton,
                  padding: '2px 6px',
                  fontSize: '10px',
                }}
                onClick={() => toggleCategoryVisible(catType)}
              >
                Unhide
              </button>
            </div>
          ))}
        </div>
      )}
    </div>
  )
}

function GeneSetCategoryComponent({
  category,
  onColorByGene,
  onColorBySet,
  activeGenes,
  onAddNewSet,
  onAddNewFolder,
}: {
  category: GeneSetCategory
  onColorByGene: (gene: string) => void
  onColorBySet: (genes: string[], geneSetName?: string) => void
  activeGenes: string[]
  onAddNewSet?: () => void
  onAddNewFolder?: () => void
}) {
  const { toggleCategoryExpanded, toggleCategoryVisible } = useStore()

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
            style={{ ...styles.addButton, marginRight: '4px', padding: '2px 6px', fontSize: '10px' }}
            onClick={(e) => {
              e.stopPropagation()
              onAddNewSet()
            }}
            title="New gene set"
          >
            +
          </button>
        )}
        {category.type === 'manual' && onAddNewFolder && (
          <button
            style={{ ...styles.addButton, marginRight: '8px', padding: '2px 6px', fontSize: '10px' }}
            onClick={(e) => {
              e.stopPropagation()
              onAddNewFolder()
            }}
            title="New folder"
          >
            + 📁
          </button>
        )}
        <button
          style={{ ...styles.iconButton, fontSize: '11px', marginRight: '4px', color: '#666' }}
          onClick={(e) => {
            e.stopPropagation()
            toggleCategoryVisible(category.type)
          }}
          title="Hide category"
        >
          👁
        </button>
        <span style={styles.categoryExpander}>{category.expanded ? '▼' : '▶'}</span>
      </div>
      {category.expanded && (
        <div style={styles.categoryContent}>
          {category.type === 'manual' && <ManualTopLevelDropStrip />}
          {/* Folders */}
          {pinSort(category.folders).map((folder) => (
            <GeneSetFolderComponent
              key={folder.id}
              folder={folder}
              categoryType={category.type}
              onColorByGene={onColorByGene}
              onColorBySet={onColorBySet}
              activeGenes={activeGenes}
              allowAddSet={category.type === 'manual'}
            />
          ))}
          {/* Direct gene sets */}
          {pinSort(category.geneSets).map((gs) => (
            <CategoryGeneSetComponent
              key={gs.id}
              geneSet={gs}
              categoryType={category.type}
              onColorByGene={onColorByGene}
              onColorBySet={onColorBySet}
              activeGenes={activeGenes}
            />
          ))}
          {isEmpty && category.type !== 'manual' && (
            <div style={styles.emptyCategory}>
              No gene sets yet
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
  const { geneSetCategories, selectedGenes, bivariateData, colorMode, addGeneSet, addGeneSetToCategory, addFolderToCategory, setImportModalOpen } = useStore()
  const { colorByGene, colorByGenes, clearExpressionColor, colorByBivariate, clearBivariateColor } = useDataActions()

  const handleColorBySet = useCallback((genes: string[], geneSetName?: string) => {
    colorByGenes(genes, undefined, geneSetName)
  }, [colorByGenes])

  // Flatten gene sets for bivariate selection
  const allGeneSets = flattenGeneSets(geneSetCategories)
  const [newSetName, setNewSetName] = useState('')
  const [showNewSetInput, setShowNewSetInput] = useState(false)
  const [newFolderName, setNewFolderName] = useState('')
  const [showNewFolderInput, setShowNewFolderInput] = useState(false)
  const [selectedSearchGenes, setSelectedSearchGenes] = useState<Set<string>>(new Set())
  const [bivariateSet1, setBivariateSet1] = useState<string | null>(null)
  const [bivariateSet2, setBivariateSet2] = useState<string | null>(null)

  // Find Similar Genes state
  const [hasGeneNeighbors, setHasGeneNeighbors] = useState(false)
  const [similarGenesSeed, setSimilarGenesSeed] = useState('')
  const [numSimilarGenes, setNumSimilarGenes] = useState(10)
  const [findingSimilarGenes, setFindingSimilarGenes] = useState(false)
  const [similarGenesError, setSimilarGenesError] = useState<string | null>(null)

  // Var identifier column switching
  const varIdentifierColumns = useStore((s) => s.varIdentifierColumns)
  const currentVarIndex = useStore((s) => s.currentVarIndex)
  const [isSwapping, setIsSwapping] = useState(false)
  const [showBrowse, setShowBrowse] = useState(false)

  // Fetch var identifier columns on mount
  useEffect(() => {
    fetchVarIdentifierColumns()
  }, [])

  const handleSwapVarIndex = async (column: string) => {
    if (column === currentVarIndex) return
    setIsSwapping(true)
    try {
      await swapVarIndex(column)
    } catch (err) {
      console.error('Failed to swap var index:', err)
    } finally {
      setIsSwapping(false)
    }
  }

  // Check prerequisites for find_similar_genes on mount and periodically
  useEffect(() => {
    const checkPrerequisites = async () => {
      try {
        const response = await fetch(appendDataset(`${API_BASE}/scanpy/prerequisites/find_similar_genes`))
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
      const response = await fetch(appendDataset(`${API_BASE}/scanpy/find_similar_genes`), {
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

  const handleCreateFolder = () => {
    const trimmed = newFolderName.trim()
    if (trimmed) {
      addFolderToCategory('manual', trimmed, [])
      setNewFolderName('')
      setShowNewFolderInput(false)
    }
  }

  return (
    <div style={styles.panel}>
      <div style={styles.header}>
        <div style={{ display: 'flex', alignItems: 'center', gap: '6px', marginBottom: '6px' }}>
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
          <button
            onClick={() => setShowBrowse(!showBrowse)}
            style={{
              padding: '2px 8px',
              fontSize: '10px',
              backgroundColor: showBrowse ? 'transparent' : '#0f3460',
              color: showBrowse ? '#4ecdc4' : '#aaa',
              border: '1px solid ' + (showBrowse ? '#4ecdc4' : '#1a1a2e'),
              borderRadius: '3px',
              cursor: 'pointer',
            }}
            title="Browse all genes alphabetically"
          >
            Browse
          </button>
        </div>
        {varIdentifierColumns.length > 1 && (
          <div style={{ marginBottom: '6px' }}>
            <div style={{ display: 'flex', alignItems: 'center', gap: '6px' }}>
              <label style={{ fontSize: '11px', color: '#888', whiteSpace: 'nowrap' }}>
                Gene IDs:
              </label>
              <select
                value={currentVarIndex}
                onChange={(e) => handleSwapVarIndex(e.target.value)}
                disabled={isSwapping}
                style={{
                  flex: 1,
                  padding: '3px 6px',
                  fontSize: '11px',
                  backgroundColor: '#0f3460',
                  color: '#ccc',
                  border: '1px solid #1a1a2e',
                  borderRadius: '3px',
                  cursor: isSwapping ? 'wait' : 'pointer',
                  opacity: isSwapping ? 0.6 : 1,
                }}
              >
                {varIdentifierColumns.map((col) => (
                  <option key={col} value={col}>
                    {col === '_index' ? '(current index)' : col}
                  </option>
                ))}
              </select>
            </div>
          </div>
        )}
        <GeneSearch
          onColorByGene={colorByGene}
          selectedSearchGenes={selectedSearchGenes}
          setSelectedSearchGenes={setSelectedSearchGenes}
          showBrowse={showBrowse}
          setShowBrowse={setShowBrowse}
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

        {/* New Folder Input (for manual category) */}
        {showNewFolderInput && (
          <div style={{ marginBottom: '12px', display: 'flex', gap: '4px' }}>
            <input
              type="text"
              placeholder="Folder name..."
              value={newFolderName}
              onChange={(e) => setNewFolderName(e.target.value)}
              onKeyDown={(e) => {
                if (e.key === 'Enter') handleCreateFolder()
                if (e.key === 'Escape') {
                  setNewFolderName('')
                  setShowNewFolderInput(false)
                }
              }}
              style={{ ...styles.searchInput, flex: 1 }}
              autoFocus
            />
            <button style={styles.addButton} onClick={handleCreateFolder}>
              Add
            </button>
          </div>
        )}

        {/* Gene Set Categories — only show non-manual categories when they have content */}
        {CATEGORY_ORDER.map((catType) => {
          const cat = geneSetCategories[catType]
          if (cat.visible === false) return null  // Hidden by user
          const totalSets = cat.geneSets.length + cat.folders.reduce((sum, f) => sum + f.geneSets.length, 0)
          // Hide empty non-manual categories to save space
          if (catType !== 'manual' && totalSets === 0) return null
          return (
            <GeneSetCategoryComponent
              key={catType}
              category={cat}
              onColorByGene={colorByGene}
              onColorBySet={handleColorBySet}
              activeGenes={selectedGenes}
              onAddNewSet={catType === 'manual' ? () => setShowNewSetInput(true) : undefined}
              onAddNewFolder={catType === 'manual' ? () => setShowNewFolderInput(true) : undefined}
            />
          )
        })}

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

        <HiddenCategoriesFooter />
      </div>
    </div>
  )
}
