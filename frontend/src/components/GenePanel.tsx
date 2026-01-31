import { useState, useRef, useEffect } from 'react'
import { useStore, GeneSet } from '../store'
import { useGeneSearch, useDataActions } from '../hooks/useData'

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
}

function GeneSearch({ onSelectGene }: { onSelectGene: (gene: string) => void }) {
  const [query, setQuery] = useState('')
  const [showDropdown, setShowDropdown] = useState(false)
  const [hoveredIndex, setHoveredIndex] = useState(-1)
  const inputRef = useRef<HTMLInputElement>(null)
  const { results, searchGenes, clearResults } = useGeneSearch()

  useEffect(() => {
    const timer = setTimeout(() => {
      searchGenes(query)
    }, 150) // Debounce
    return () => clearTimeout(timer)
  }, [query, searchGenes])

  const handleSelect = (gene: string) => {
    onSelectGene(gene)
    setQuery('')
    clearResults()
    setShowDropdown(false)
  }

  const handleKeyDown = (e: React.KeyboardEvent) => {
    if (e.key === 'ArrowDown') {
      e.preventDefault()
      setHoveredIndex((i) => Math.min(i + 1, results.length - 1))
    } else if (e.key === 'ArrowUp') {
      e.preventDefault()
      setHoveredIndex((i) => Math.max(i - 1, 0))
    } else if (e.key === 'Enter' && hoveredIndex >= 0 && results[hoveredIndex]) {
      handleSelect(results[hoveredIndex])
    } else if (e.key === 'Escape') {
      setShowDropdown(false)
      clearResults()
    }
  }

  return (
    <div style={styles.searchContainer}>
      <input
        ref={inputRef}
        type="text"
        placeholder="Search genes..."
        value={query}
        onChange={(e) => {
          setQuery(e.target.value)
          setShowDropdown(true)
          setHoveredIndex(-1)
        }}
        onFocus={() => setShowDropdown(true)}
        onBlur={() => setTimeout(() => setShowDropdown(false), 200)}
        onKeyDown={handleKeyDown}
        style={styles.searchInput}
      />
      {showDropdown && results.length > 0 && (
        <div style={styles.dropdown}>
          {results.map((gene, i) => (
            <div
              key={gene}
              style={{
                ...styles.dropdownItem,
                ...(i === hoveredIndex ? styles.dropdownItemHover : {}),
              }}
              onMouseEnter={() => setHoveredIndex(i)}
              onMouseDown={() => handleSelect(gene)}
            >
              {gene}
            </div>
          ))}
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
}: {
  geneSet: GeneSet
  onColorByGene: (gene: string) => void
  onColorBySet: (genes: string[]) => void
  activeGenes: string[]
}) {
  const [expanded, setExpanded] = useState(true)
  const [hoveredGene, setHoveredGene] = useState<string | null>(null)
  const { removeGeneSet, removeGenesFromSet } = useStore()

  const isActive = (gene: string) => activeGenes.includes(gene)

  return (
    <div style={styles.geneSet}>
      <div style={styles.geneSetHeader} onClick={() => setExpanded(!expanded)}>
        <div>
          <span style={styles.geneSetName}>{geneSet.name}</span>
          <span style={styles.geneSetCount}>({geneSet.genes.length})</span>
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
              style={{
                ...styles.gene,
                ...(isActive(gene) ? styles.geneActive : {}),
                ...(hoveredGene === gene && !isActive(gene) ? styles.geneHover : {}),
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
    </div>
  )
}

export default function GenePanel() {
  const { geneSets, selectedGenes, addGeneSet, addGenesToSet } = useStore()
  const { colorByGene, colorByGenes, clearExpressionColor } = useDataActions()
  const [newSetName, setNewSetName] = useState('')
  const [showNewSetInput, setShowNewSetInput] = useState(false)

  const handleAddGene = (gene: string) => {
    // If there's a "Search Results" set, add to it, otherwise create one
    const searchResultsSet = geneSets.find((gs) => gs.name === 'Search Results')
    if (searchResultsSet) {
      addGenesToSet('Search Results', [gene])
    } else {
      addGeneSet('Search Results', [gene])
    }
    // Also color by this gene
    colorByGene(gene)
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
        <div style={styles.title}>Genes</div>
        <GeneSearch onSelectGene={handleAddGene} />
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
              Search for genes above to add them to a set
            </div>
          ) : (
            geneSets.map((gs) => (
              <GeneSetComponent
                key={gs.name}
                geneSet={gs}
                onColorByGene={colorByGene}
                onColorBySet={colorByGenes}
                activeGenes={selectedGenes}
              />
            ))
          )}
        </div>
      </div>
    </div>
  )
}
