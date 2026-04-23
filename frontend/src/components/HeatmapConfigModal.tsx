/**
 * HeatmapConfigModal — inline configuration panel for the heatmap view.
 *
 * Rendered inside the center panel when the user first switches to the
 * Heatmap tab or clicks the Settings button. Lets the user choose:
 *   - Gene sets to display
 *   - Cell ordering (category, line position, etc.)
 *   - Gene ordering (as provided, by peak position)
 *   - Aggregate gene sets toggle
 *   - Number of cell bins
 *
 * Rollback: delete this file and remove imports from HeatmapView.tsx.
 */

import { useState, useEffect } from 'react'
import { useStore, HeatmapConfig, GeneSet, GeneSetCategoryType } from '../store'

const CATEGORY_ORDER: GeneSetCategoryType[] = ['manual', 'gene_clusters', 'similar_genes', 'diff_exp', 'spatial', 'marker_genes', 'line_association']
const CATEGORY_NAMES: Record<GeneSetCategoryType, string> = {
  manual: 'Manual',
  gene_clusters: 'Gene Clusters',
  similar_genes: 'Similar Genes',
  diff_exp: 'Diff. Expression',
  spatial: 'Spatial',
  marker_genes: 'Marker Genes',
  line_association: 'Line Association',
}

interface FlatGeneSet {
  id: string
  name: string
  genes: string[]
  category: GeneSetCategoryType
  folderName?: string
}

function getAllGeneSets(categories: Record<GeneSetCategoryType, { geneSets: GeneSet[]; folders: { name: string; geneSets: GeneSet[] }[] }>): FlatGeneSet[] {
  const all: FlatGeneSet[] = []
  for (const catType of CATEGORY_ORDER) {
    const cat = categories[catType]
    for (const gs of cat.geneSets) {
      all.push({ ...gs, category: catType })
    }
    for (const folder of cat.folders) {
      for (const gs of folder.geneSets) {
        all.push({ ...gs, category: catType, folderName: folder.name })
      }
    }
  }
  return all
}

interface Props {
  config: HeatmapConfig | null
  onApply: (config: HeatmapConfig) => void
  onCancel: () => void
}

export default function HeatmapConfigModal({ config, onApply, onCancel }: Props) {
  const geneSetCategories = useStore((s) => s.geneSetCategories)
  const schema = useStore((s) => s.schema)
  const drawnLines = useStore((s) => s.drawnLines)

  const allGeneSets = getAllGeneSets(geneSetCategories)
  const nonEmptySets = allGeneSets.filter((gs) => gs.genes.length > 0)

  // Initialize from existing config or defaults
  const [selectedIds, setSelectedIds] = useState<Set<string>>(
    () => new Set(config?.selectedGeneSets.map((gs) => nonEmptySets.find((s) => s.name === gs.name && s.genes.length === gs.genes.length)?.id || '').filter(Boolean))
  )
  const [cellOrdering, setCellOrdering] = useState<HeatmapConfig['cellOrdering']>(config?.cellOrdering ?? 'none')
  const [obsColumn, setObsColumn] = useState<string | null>(config?.obsColumn ?? null)
  const [lineName, setLineName] = useState<string | null>(config?.lineName ?? null)
  const [geneOrdering, setGeneOrdering] = useState<HeatmapConfig['geneOrdering']>(config?.geneOrdering ?? 'as_provided')
  const [aggregateGeneSets, setAggregateGeneSets] = useState(config?.aggregateGeneSets ?? false)
  const [nBins, setNBins] = useState(config?.nBins ?? 300)

  // Get categorical obs columns for ordering
  const categoricalColumns = schema
    ? schema.obs_columns.filter((c) => schema.obs_dtypes[c] === 'category')
    : []

  // Set defaults if needed
  useEffect(() => {
    if (cellOrdering === 'category' && !obsColumn && categoricalColumns.length > 0) {
      setObsColumn(categoricalColumns[0])
    }
    if ((cellOrdering === 'line_position' || cellOrdering === 'line_distance' || cellOrdering === 'category_then_position') && !lineName && drawnLines.length > 0) {
      setLineName(drawnLines[0].name)
    }
  }, [cellOrdering, obsColumn, lineName, categoricalColumns, drawnLines])

  const toggleGeneSet = (id: string) => {
    setSelectedIds((prev) => {
      const next = new Set(prev)
      if (next.has(id)) next.delete(id)
      else next.add(id)
      return next
    })
  }

  const selectAllGeneSets = () => {
    setSelectedIds(new Set(nonEmptySets.map((gs) => gs.id)))
  }

  const selectNoneGeneSets = () => {
    setSelectedIds(new Set())
  }

  const handleApply = () => {
    const selected = nonEmptySets.filter((gs) => selectedIds.has(gs.id))
    if (selected.length === 0) return

    onApply({
      selectedGeneSets: selected.map((gs) => ({ name: gs.name, genes: gs.genes })),
      cellOrdering,
      obsColumn: (cellOrdering === 'category' || cellOrdering === 'category_then_position') ? obsColumn : null,
      lineName: (cellOrdering === 'line_position' || cellOrdering === 'line_distance' || cellOrdering === 'category_then_position') ? lineName : null,
      geneOrdering,
      aggregateGeneSets,
      nBins,
    })
  }

  const needsCategory = cellOrdering === 'category' || cellOrdering === 'category_then_position'
  const needsLine = cellOrdering === 'line_position' || cellOrdering === 'line_distance' || cellOrdering === 'category_then_position'

  return (
    <div style={styles.container}>
      <div style={styles.panel}>
        <h3 style={styles.title}>Configure Heatmap</h3>

        {/* Gene set selection */}
        <div style={styles.section}>
          <div style={styles.sectionHeader}>
            <span style={styles.sectionTitle}>Gene Sets</span>
            <div style={{ display: 'flex', gap: '6px' }}>
              <button style={styles.smallButton} onClick={selectAllGeneSets}>All</button>
              <button style={styles.smallButton} onClick={selectNoneGeneSets}>None</button>
            </div>
          </div>
          {nonEmptySets.length === 0 ? (
            <div style={styles.empty}>
              No gene sets with genes defined. Create gene sets in the Gene panel first.
            </div>
          ) : (
            <div style={styles.geneSetList}>
              {CATEGORY_ORDER.map((catType) => {
                const catSets = nonEmptySets.filter((gs) => gs.category === catType)
                if (catSets.length === 0) return null
                return (
                  <div key={catType}>
                    <div style={styles.categoryLabel}>{CATEGORY_NAMES[catType]}</div>
                    {catSets.map((gs) => (
                      <label key={gs.id} style={styles.checkboxRow}>
                        <input
                          type="checkbox"
                          checked={selectedIds.has(gs.id)}
                          onChange={() => toggleGeneSet(gs.id)}
                        />
                        <span style={styles.geneSetName}>
                          {gs.folderName ? `${gs.folderName} / ` : ''}{gs.name}
                        </span>
                        <span style={styles.geneSetCount}>({gs.genes.length})</span>
                      </label>
                    ))}
                  </div>
                )
              })}
            </div>
          )}
        </div>

        {/* Cell ordering */}
        <div style={styles.section}>
          <div style={styles.sectionTitle}>Cell Ordering</div>
          <div style={styles.radioGroup}>
            <label style={styles.radioRow}>
              <input type="radio" name="cellOrder" checked={cellOrdering === 'none'} onChange={() => setCellOrdering('none')} />
              <span>None (original order)</span>
            </label>
            <label style={styles.radioRow}>
              <input type="radio" name="cellOrder" checked={cellOrdering === 'category'} onChange={() => setCellOrdering('category')} disabled={categoricalColumns.length === 0} />
              <span>Group by category</span>
            </label>
            {needsCategory && cellOrdering !== 'category_then_position' && (
              <select style={styles.inlineSelect} value={obsColumn || ''} onChange={(e) => setObsColumn(e.target.value || null)}>
                {categoricalColumns.map((c) => <option key={c} value={c}>{c}</option>)}
              </select>
            )}
            <label style={styles.radioRow}>
              <input type="radio" name="cellOrder" checked={cellOrdering === 'line_position'} onChange={() => setCellOrdering('line_position')} disabled={drawnLines.length === 0} />
              <span>By line position</span>
            </label>
            <label style={styles.radioRow}>
              <input type="radio" name="cellOrder" checked={cellOrdering === 'line_distance'} onChange={() => setCellOrdering('line_distance')} disabled={drawnLines.length === 0} />
              <span>By distance from line</span>
            </label>
            <label style={styles.radioRow}>
              <input type="radio" name="cellOrder" checked={cellOrdering === 'category_then_position'} onChange={() => setCellOrdering('category_then_position')} disabled={categoricalColumns.length === 0 || drawnLines.length === 0} />
              <span>Category, then position</span>
            </label>
            {cellOrdering === 'category_then_position' && (
              <div style={{ marginLeft: '24px', display: 'flex', gap: '8px', flexWrap: 'wrap' }}>
                <select style={styles.inlineSelect} value={obsColumn || ''} onChange={(e) => setObsColumn(e.target.value || null)}>
                  {categoricalColumns.map((c) => <option key={c} value={c}>{c}</option>)}
                </select>
              </div>
            )}
            {needsLine && (
              <select style={{ ...styles.inlineSelect, marginLeft: '24px' }} value={lineName || ''} onChange={(e) => setLineName(e.target.value || null)}>
                {drawnLines.map((l) => <option key={l.id} value={l.name}>{l.name}</option>)}
              </select>
            )}
          </div>
        </div>

        {/* Gene ordering */}
        <div style={styles.section}>
          <div style={styles.sectionTitle}>Gene Ordering</div>
          <div style={styles.radioGroup}>
            <label style={styles.radioRow}>
              <input type="radio" name="geneOrder" checked={geneOrdering === 'as_provided'} onChange={() => setGeneOrdering('as_provided')} />
              <span>By gene set</span>
            </label>
            <label style={styles.radioRow}>
              <input type="radio" name="geneOrder" checked={geneOrdering === 'peak_position'} onChange={() => setGeneOrdering('peak_position')} />
              <span>By peak position</span>
            </label>
          </div>
        </div>

        {/* Aggregate toggle */}
        <div style={styles.section}>
          <label style={styles.checkboxRow}>
            <input type="checkbox" checked={aggregateGeneSets} onChange={(e) => setAggregateGeneSets(e.target.checked)} />
            <span>Show gene set aggregates (mean expression per set)</span>
          </label>
        </div>

        {/* Bin count */}
        <div style={styles.section}>
          <div style={styles.sectionTitle}>Cell Bins</div>
          <div style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
            <input
              type="number"
              min={0}
              max={2000}
              value={nBins}
              onChange={(e) => setNBins(Math.max(0, parseInt(e.target.value) || 0))}
              style={styles.numberInput}
            />
            <span style={styles.hint}>0 = no binning (show all cells)</span>
          </div>
        </div>

        {/* Actions */}
        <div style={styles.actions}>
          {config && (
            <button style={styles.cancelButton} onClick={onCancel}>Cancel</button>
          )}
          <button
            style={{
              ...styles.applyButton,
              opacity: selectedIds.size === 0 ? 0.4 : 1,
              cursor: selectedIds.size === 0 ? 'not-allowed' : 'pointer',
            }}
            onClick={handleApply}
            disabled={selectedIds.size === 0}
          >
            Generate Heatmap
          </button>
        </div>
      </div>
    </div>
  )
}

const styles: Record<string, React.CSSProperties> = {
  container: {
    display: 'flex',
    alignItems: 'center',
    justifyContent: 'center',
    height: '100%',
    padding: '20px',
  },
  panel: {
    backgroundColor: '#16213e',
    border: '1px solid #0f3460',
    borderRadius: '8px',
    padding: '24px',
    width: '100%',
    maxWidth: '500px',
    maxHeight: '80vh',
    overflowY: 'auto',
  },
  title: {
    fontSize: '16px',
    fontWeight: 600,
    color: '#e94560',
    marginTop: 0,
    marginBottom: '20px',
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
    textTransform: 'uppercase',
    marginBottom: '8px',
  },
  categoryLabel: {
    fontSize: '10px',
    fontWeight: 600,
    color: '#666',
    textTransform: 'uppercase',
    padding: '4px 0',
    marginTop: '4px',
  },
  geneSetList: {
    maxHeight: '200px',
    overflowY: 'auto',
    backgroundColor: '#0f3460',
    borderRadius: '4px',
    padding: '8px',
  },
  checkboxRow: {
    display: 'flex',
    alignItems: 'center',
    gap: '8px',
    padding: '4px 0',
    fontSize: '13px',
    color: '#ccc',
    cursor: 'pointer',
  },
  geneSetName: {
    flex: 1,
    overflow: 'hidden',
    textOverflow: 'ellipsis',
    whiteSpace: 'nowrap',
  },
  geneSetCount: {
    fontSize: '11px',
    color: '#888',
  },
  radioGroup: {
    display: 'flex',
    flexDirection: 'column',
    gap: '4px',
  },
  radioRow: {
    display: 'flex',
    alignItems: 'center',
    gap: '8px',
    fontSize: '13px',
    color: '#ccc',
    cursor: 'pointer',
  },
  inlineSelect: {
    marginLeft: '24px',
    padding: '4px 8px',
    fontSize: '12px',
    backgroundColor: '#0f3460',
    color: '#eee',
    border: '1px solid #1a1a2e',
    borderRadius: '4px',
    cursor: 'pointer',
  },
  numberInput: {
    width: '80px',
    padding: '6px 8px',
    fontSize: '13px',
    backgroundColor: '#0f3460',
    color: '#eee',
    border: '1px solid #1a1a2e',
    borderRadius: '4px',
  },
  hint: {
    fontSize: '11px',
    color: '#666',
  },
  smallButton: {
    padding: '2px 8px',
    fontSize: '10px',
    backgroundColor: '#0f3460',
    color: '#aaa',
    border: '1px solid #1a1a2e',
    borderRadius: '3px',
    cursor: 'pointer',
  },
  empty: {
    fontSize: '13px',
    color: '#666',
    padding: '12px',
    textAlign: 'center',
    fontStyle: 'italic',
  },
  actions: {
    display: 'flex',
    justifyContent: 'flex-end',
    gap: '8px',
    marginTop: '20px',
  },
  cancelButton: {
    padding: '8px 16px',
    fontSize: '13px',
    backgroundColor: 'transparent',
    color: '#aaa',
    border: '1px solid #0f3460',
    borderRadius: '4px',
    cursor: 'pointer',
  },
  applyButton: {
    padding: '8px 20px',
    fontSize: '13px',
    fontWeight: 500,
    backgroundColor: '#4ecdc4',
    color: '#000',
    border: 'none',
    borderRadius: '4px',
    cursor: 'pointer',
  },
}
