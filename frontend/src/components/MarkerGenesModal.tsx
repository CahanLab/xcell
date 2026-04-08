import { useState, useCallback, useEffect, useRef } from 'react'
import { useStore } from '../store'
import { useObsSummaries, runMarkerGenes, MarkerGenesGroupResult, appendDataset } from '../hooks/useData'

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
    width: '560px',
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
  body: {
    flex: 1,
    overflowY: 'auto' as const,
    padding: '16px 20px',
  },
  label: {
    fontSize: '12px',
    color: '#aaa',
    marginBottom: '6px',
    display: 'block',
    fontWeight: 500,
  },
  columnName: {
    fontSize: '14px',
    color: '#eee',
    fontWeight: 600,
    marginBottom: '16px',
  },
  groupList: {
    maxHeight: '200px',
    overflowY: 'auto' as const,
    backgroundColor: '#0a0f1a',
    borderRadius: '4px',
    padding: '8px',
    marginBottom: '16px',
  },
  groupItem: {
    display: 'flex',
    alignItems: 'center',
    padding: '4px 8px',
    fontSize: '13px',
    color: '#ccc',
    cursor: 'pointer',
    borderRadius: '4px',
    marginBottom: '2px',
  },
  groupItemHover: {
    backgroundColor: '#0f3460',
  },
  checkbox: {
    marginRight: '8px',
    cursor: 'pointer',
  },
  groupCount: {
    color: '#666',
    fontSize: '11px',
    marginLeft: 'auto',
  },
  selectAll: {
    display: 'flex',
    alignItems: 'center',
    padding: '4px 8px',
    fontSize: '12px',
    color: '#888',
    cursor: 'pointer',
    borderBottom: '1px solid #1a1a2e',
    marginBottom: '4px',
    paddingBottom: '8px',
  },
  inputRow: {
    display: 'flex',
    alignItems: 'center',
    gap: '8px',
    marginBottom: '16px',
  },
  input: {
    padding: '6px 10px',
    fontSize: '13px',
    backgroundColor: '#0f3460',
    color: '#eee',
    border: '1px solid #1a1a2e',
    borderRadius: '4px',
    outline: 'none',
    width: '80px',
  },
  advancedToggle: {
    fontSize: '12px',
    color: '#888',
    cursor: 'pointer',
    userSelect: 'none' as const,
    marginBottom: '12px',
  },
  advancedContent: {
    backgroundColor: '#0a0f1a',
    borderRadius: '4px',
    padding: '12px',
    marginBottom: '16px',
  },
  footer: {
    padding: '12px 20px',
    borderTop: '1px solid #0f3460',
    display: 'flex',
    justifyContent: 'space-between',
    alignItems: 'center',
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
  successButton: {
    backgroundColor: '#4ecdc4',
    color: '#000',
  },
  disabledButton: {
    opacity: 0.5,
    cursor: 'not-allowed',
  },
  resultsSummary: {
    marginBottom: '12px',
    fontSize: '13px',
    color: '#aaa',
  },
  resultGroup: {
    backgroundColor: '#0a0f1a',
    borderRadius: '4px',
    marginBottom: '8px',
    overflow: 'hidden',
  },
  resultGroupHeader: {
    display: 'flex',
    alignItems: 'center',
    justifyContent: 'space-between',
    padding: '8px 12px',
    cursor: 'pointer',
    fontSize: '13px',
    color: '#eee',
    fontWeight: 500,
  },
  resultGeneList: {
    padding: '4px 12px 8px',
    borderTop: '1px solid #1a1a2e',
  },
  resultGene: {
    display: 'flex',
    alignItems: 'center',
    justifyContent: 'space-between',
    padding: '3px 0',
    fontSize: '12px',
    color: '#bbb',
  },
  resultGeneStats: {
    fontSize: '11px',
    color: '#666',
  },
  error: {
    fontSize: '12px',
    color: '#e94560',
    padding: '8px 12px',
    backgroundColor: 'rgba(233, 69, 96, 0.1)',
    borderRadius: '4px',
    marginBottom: '12px',
  },
}

export default function MarkerGenesModal() {
  const {
    isMarkerGenesModalOpen,
    markerGenesColumn,
    setMarkerGenesModalOpen,
    addFolderToCategory,
    comparisonCheckedColumn,
    comparisonCheckedCategories,
  } = useStore()

  const { summaries } = useObsSummaries()

  // Config state
  const [selectedGroups, setSelectedGroups] = useState<Set<string>>(new Set())
  const [topN, setTopN] = useState(100)
  const [showAdvanced, setShowAdvanced] = useState(false)
  const [minInGroupFraction, setMinInGroupFraction] = useState('')
  const [maxOutGroupFraction, setMaxOutGroupFraction] = useState('')
  const [minFoldChange, setMinFoldChange] = useState('')

  // Gene subset state
  const [booleanColumns, setBooleanColumns] = useState<{ name: string; n_true: number; n_total: number }[]>([])
  const [geneSubset, setGeneSubset] = useState<string>('')

  // Fetch boolean columns when modal opens
  useEffect(() => {
    if (!isMarkerGenesModalOpen) return
    fetch(appendDataset('/api/var/boolean_columns'))
      .then((res) => res.json())
      .then((cols: { name: string; n_true: number; n_total: number }[]) => {
        setBooleanColumns(cols)
        const hvg = cols.find((c) => c.name === 'highly_variable')
        setGeneSubset(hvg ? 'highly_variable' : '')
      })
      .catch(() => setBooleanColumns([]))
  }, [isMarkerGenesModalOpen])

  // Results state
  const [results, setResults] = useState<MarkerGenesGroupResult[] | null>(null)
  const [isRunning, setIsRunning] = useState(false)
  const [errorMsg, setErrorMsg] = useState<string | null>(null)
  const [expandedGroups, setExpandedGroups] = useState<Set<string>>(new Set())
  const [addedToSets, setAddedToSets] = useState(false)

  // Get the summary for the selected column
  const columnSummary = summaries.find((s) => s.name === markerGenesColumn)
  const categories = columnSummary?.categories || []

  // Snapshot the pre-checked categories when modal opens (to avoid re-triggering on every toggle)
  const preCheckedRef = useRef<Set<string> | null>(null)
  useEffect(() => {
    if (isMarkerGenesModalOpen && markerGenesColumn) {
      preCheckedRef.current =
        comparisonCheckedColumn === markerGenesColumn && comparisonCheckedCategories.size > 0
          ? new Set(comparisonCheckedCategories)
          : null
    }
  // Only capture on open
  // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [isMarkerGenesModalOpen, markerGenesColumn])

  // Initialize selected groups when column changes or categories load
  useEffect(() => {
    if (markerGenesColumn && categories.length > 0) {
      if (preCheckedRef.current && preCheckedRef.current.size > 0) {
        setSelectedGroups(new Set(preCheckedRef.current))
      } else {
        setSelectedGroups(new Set(categories.map((c) => c.value)))
      }
      setResults(null)
      setErrorMsg(null)
      setAddedToSets(false)
    }
  }, [markerGenesColumn, categories.length])

  const toggleGroup = useCallback((group: string) => {
    setSelectedGroups((prev) => {
      const next = new Set(prev)
      if (next.has(group)) {
        next.delete(group)
      } else {
        next.add(group)
      }
      return next
    })
  }, [])

  const toggleSelectAll = useCallback(() => {
    if (selectedGroups.size === categories.length) {
      setSelectedGroups(new Set())
    } else {
      setSelectedGroups(new Set(categories.map((c) => c.value)))
    }
  }, [categories, selectedGroups.size])

  const handleRun = useCallback(async () => {
    if (!markerGenesColumn || selectedGroups.size < 2) return

    setIsRunning(true)
    setErrorMsg(null)
    setResults(null)
    setAddedToSets(false)

    try {
      const params: {
        obs_column: string
        groups?: string[]
        top_n: number
        min_in_group_fraction?: number
        max_out_group_fraction?: number
        min_fold_change?: number
        gene_subset?: string | null
      } = {
        obs_column: markerGenesColumn,
        top_n: topN,
      }

      if (geneSubset) params.gene_subset = geneSubset

      // Only send groups if not all selected
      if (selectedGroups.size < categories.length) {
        params.groups = Array.from(selectedGroups)
      }

      const minIGF = parseFloat(minInGroupFraction)
      if (!isNaN(minIGF)) params.min_in_group_fraction = minIGF

      const maxOGF = parseFloat(maxOutGroupFraction)
      if (!isNaN(maxOGF)) params.max_out_group_fraction = maxOGF

      const minFC = parseFloat(minFoldChange)
      if (!isNaN(minFC)) params.min_fold_change = minFC

      const response = await runMarkerGenes(params)
      setResults(response.results)
      // Auto-expand first group
      if (response.results.length > 0) {
        setExpandedGroups(new Set([response.results[0].group]))
      }
    } catch (err) {
      setErrorMsg((err as Error).message)
    } finally {
      setIsRunning(false)
    }
  }, [markerGenesColumn, selectedGroups, categories.length, topN, minInGroupFraction, maxOutGroupFraction, minFoldChange])

  const handleAddToGeneSets = useCallback(() => {
    if (!results || !markerGenesColumn) return

    const geneSets = results
      .filter((r) => r.genes.length > 0)
      .map((r) => ({
        name: `${r.group} markers`,
        genes: r.genes.map((g) => g.gene),
      }))

    if (geneSets.length === 0) return

    const folderName = `${markerGenesColumn} markers`
    addFolderToCategory('marker_genes', folderName, geneSets)
    setAddedToSets(true)
  }, [results, markerGenesColumn, addFolderToCategory])

  const toggleResultGroup = useCallback((group: string) => {
    setExpandedGroups((prev) => {
      const next = new Set(prev)
      if (next.has(group)) {
        next.delete(group)
      } else {
        next.add(group)
      }
      return next
    })
  }, [])

  const handleClose = useCallback(() => {
    setMarkerGenesModalOpen(false)
  }, [setMarkerGenesModalOpen])

  if (!isMarkerGenesModalOpen || !markerGenesColumn) return null

  const canRun = selectedGroups.size >= 2 && !isRunning

  return (
    <div style={styles.backdrop} onClick={handleClose}>
      <div style={styles.modal} onClick={(e) => e.stopPropagation()}>
        <div style={styles.header}>
          <h2 style={styles.title}>Find Marker Genes</h2>
          <button style={styles.closeButton} onClick={handleClose}>
            &times;
          </button>
        </div>

        <div style={styles.body}>
          <div style={styles.label}>Column</div>
          <div style={styles.columnName}>{markerGenesColumn}</div>

          {/* Group selection */}
          <div style={styles.label}>Select Groups (min 2)</div>
          <div style={styles.groupList}>
            <div style={styles.selectAll} onClick={toggleSelectAll}>
              <input
                type="checkbox"
                checked={selectedGroups.size === categories.length}
                onChange={toggleSelectAll}
                style={styles.checkbox}
              />
              <span>Select All</span>
              <span style={styles.groupCount}>{selectedGroups.size}/{categories.length}</span>
            </div>
            {categories.map((cat) => (
              <div
                key={cat.value}
                style={styles.groupItem}
                onClick={() => toggleGroup(cat.value)}
              >
                <input
                  type="checkbox"
                  checked={selectedGroups.has(cat.value)}
                  onChange={() => toggleGroup(cat.value)}
                  style={styles.checkbox}
                />
                <span>{cat.value}</span>
                <span style={styles.groupCount}>
                  {cat.count.toLocaleString()} cells
                </span>
              </div>
            ))}
          </div>

          {/* Gene subset */}
          <div style={styles.inputRow}>
            <span style={{ ...styles.label, marginBottom: 0, flex: 1 }}>Genes:</span>
            <select
              value={geneSubset}
              onChange={(e) => setGeneSubset(e.target.value)}
              style={{ ...styles.input, width: 'auto' }}
            >
              <option value="">All genes</option>
              {booleanColumns.map((col) => (
                <option key={col.name} value={col.name}>
                  {col.name} ({col.n_true.toLocaleString()})
                </option>
              ))}
            </select>
          </div>

          {/* Top N */}
          <div style={styles.inputRow}>
            <span style={{ ...styles.label, marginBottom: 0 }}>Top N genes per group:</span>
            <input
              type="number"
              min={1}
              max={500}
              value={topN}
              onChange={(e) => setTopN(Math.max(1, parseInt(e.target.value) || 25))}
              style={styles.input}
            />
          </div>

          {/* Advanced filters */}
          <div
            style={styles.advancedToggle}
            onClick={() => setShowAdvanced(!showAdvanced)}
          >
            {showAdvanced ? '\u25BC' : '\u25B6'} Advanced Filters
          </div>
          {showAdvanced && (
            <div style={styles.advancedContent}>
              <div style={styles.inputRow}>
                <span style={{ ...styles.label, marginBottom: 0, flex: 1 }}>Min % in group:</span>
                <input
                  type="number"
                  min={0}
                  max={1}
                  step={0.05}
                  placeholder="e.g. 0.25"
                  value={minInGroupFraction}
                  onChange={(e) => setMinInGroupFraction(e.target.value)}
                  style={{ ...styles.input, width: '100px' }}
                />
              </div>
              <div style={styles.inputRow}>
                <span style={{ ...styles.label, marginBottom: 0, flex: 1 }}>Max % in others:</span>
                <input
                  type="number"
                  min={0}
                  max={1}
                  step={0.05}
                  placeholder="e.g. 0.75"
                  value={maxOutGroupFraction}
                  onChange={(e) => setMaxOutGroupFraction(e.target.value)}
                  style={{ ...styles.input, width: '100px' }}
                />
              </div>
              <div style={{ ...styles.inputRow, marginBottom: 0 }}>
                <span style={{ ...styles.label, marginBottom: 0, flex: 1 }}>Min fold change:</span>
                <input
                  type="number"
                  min={0}
                  step={0.5}
                  placeholder="e.g. 2.0"
                  value={minFoldChange}
                  onChange={(e) => setMinFoldChange(e.target.value)}
                  style={{ ...styles.input, width: '100px' }}
                />
              </div>
            </div>
          )}

          {/* Error */}
          {errorMsg && <div style={styles.error}>{errorMsg}</div>}

          {/* Results */}
          {results && (
            <>
              <div style={styles.resultsSummary}>
                {results.length} group{results.length !== 1 ? 's' : ''},{' '}
                {results.reduce((sum, r) => sum + r.genes.length, 0)} total marker genes
              </div>
              {results.map((groupResult) => (
                <div key={groupResult.group} style={styles.resultGroup}>
                  <div
                    style={styles.resultGroupHeader}
                    onClick={() => toggleResultGroup(groupResult.group)}
                  >
                    <span>
                      {expandedGroups.has(groupResult.group) ? '\u25BC' : '\u25B6'}{' '}
                      {groupResult.group}
                    </span>
                    <span style={{ fontSize: '11px', color: '#888' }}>
                      {groupResult.genes.length} genes
                    </span>
                  </div>
                  {expandedGroups.has(groupResult.group) && groupResult.genes.length > 0 && (
                    <div style={styles.resultGeneList}>
                      <div style={{ ...styles.resultGene, fontWeight: 600, color: '#888', fontSize: '11px', borderBottom: '1px solid #1a1a2e', paddingBottom: '4px', marginBottom: '4px' }}>
                        <span style={{ flex: 1 }}>Gene</span>
                        <span style={{ width: '70px', textAlign: 'right' }}>log2FC</span>
                        <span style={{ width: '80px', textAlign: 'right' }}>p-adj</span>
                      </div>
                      {groupResult.genes.map((gene) => (
                        <div key={gene.gene} style={styles.resultGene}>
                          <span style={{ flex: 1 }}>{gene.gene}</span>
                          <span style={{ ...styles.resultGeneStats, width: '70px', textAlign: 'right' }}>
                            {gene.log2fc.toFixed(2)}
                          </span>
                          <span style={{ ...styles.resultGeneStats, width: '80px', textAlign: 'right' }}>
                            {gene.pval_adj < 0.001 ? gene.pval_adj.toExponential(1) : gene.pval_adj.toFixed(3)}
                          </span>
                        </div>
                      ))}
                    </div>
                  )}
                  {expandedGroups.has(groupResult.group) && groupResult.genes.length === 0 && (
                    <div style={{ padding: '8px 12px', fontSize: '12px', color: '#666', fontStyle: 'italic' }}>
                      No marker genes found
                    </div>
                  )}
                </div>
              ))}
            </>
          )}
        </div>

        <div style={styles.footer}>
          <button
            style={{ ...styles.button, ...styles.secondaryButton }}
            onClick={handleClose}
          >
            Close
          </button>
          <div style={{ display: 'flex', gap: '8px' }}>
            {results && !addedToSets && (
              <button
                style={{
                  ...styles.button,
                  ...styles.successButton,
                  ...(results.every((r) => r.genes.length === 0) ? styles.disabledButton : {}),
                }}
                onClick={handleAddToGeneSets}
                disabled={results.every((r) => r.genes.length === 0)}
              >
                Add to Gene Sets
              </button>
            )}
            {addedToSets && (
              <span style={{ fontSize: '12px', color: '#4ecdc4', alignSelf: 'center' }}>
                Added to Marker Genes
              </span>
            )}
            <button
              style={{
                ...styles.button,
                ...styles.primaryButton,
                ...(!canRun ? styles.disabledButton : {}),
              }}
              onClick={handleRun}
              disabled={!canRun}
            >
              {isRunning ? 'Running...' : 'Run'}
            </button>
          </div>
        </div>
      </div>
    </div>
  )
}
