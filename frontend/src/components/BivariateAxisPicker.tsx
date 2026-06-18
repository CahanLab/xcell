/**
 * BivariateAxisPicker — choose one bivariate axis as either a saved gene set
 * OR a single gene. Used by the Embedding view (GenePanel) and the Figure
 * builder so both axes can be a gene or a set (e.g. Sox9 × Scx, or
 * cartilage × tendon, or a mix).
 *
 * Presentational: the parent supplies the available gene sets and the current
 * {kind, value}, and handles resolution via `resolveBivariateAxis`.
 */

import { useState } from 'react'
import { useGeneSearch } from '../hooks/useData'

export type AxisKind = 'set' | 'gene'

export interface AxisGeneSet {
  id: string
  name: string
  genes: string[]
}

/** Resolve an axis selection to its gene list + a short display label. */
export function resolveBivariateAxis(
  kind: AxisKind,
  value: string | null,
  geneSets: AxisGeneSet[],
): { genes: string[]; label: string } {
  if (!value) return { genes: [], label: '' }
  if (kind === 'gene') return { genes: [value], label: value }
  const gs = geneSets.find((g) => g.id === value)
  return { genes: gs?.genes ?? [], label: gs?.name ?? '' }
}

interface Props {
  kind: AxisKind
  value: string | null
  geneSets: AxisGeneSet[]
  onChange: (kind: AxisKind, value: string | null) => void
  /** Optional gene-set id to hide from the dropdown (the other axis's set). */
  excludeSetId?: string | null
}

export default function BivariateAxisPicker({ kind, value, geneSets, onChange, excludeSetId }: Props) {
  const { results, searchGenes, clearResults } = useGeneSearch()
  const [query, setQuery] = useState('')

  const setMode = (k: AxisKind) => {
    if (k !== kind) {
      onChange(k, null)
      setQuery('')
      clearResults()
    }
  }

  return (
    <div>
      <div style={styles.toggle}>
        <button
          type="button"
          style={{ ...styles.toggleBtn, ...(kind === 'set' ? styles.toggleActive : {}) }}
          onClick={() => setMode('set')}
        >
          Gene set
        </button>
        <button
          type="button"
          style={{ ...styles.toggleBtn, ...(kind === 'gene' ? styles.toggleActive : {}) }}
          onClick={() => setMode('gene')}
        >
          Gene
        </button>
      </div>

      {kind === 'set' ? (
        <select
          value={value ?? ''}
          onChange={(e) => onChange('set', e.target.value || null)}
          style={styles.input}
        >
          <option value="">Select gene set…</option>
          {geneSets
            .filter((gs) => gs.genes.length > 0 && gs.id !== excludeSetId)
            .map((gs) => (
              <option key={gs.id} value={gs.id}>
                {gs.name} ({gs.genes.length} genes)
              </option>
            ))}
        </select>
      ) : value ? (
        <div style={styles.chipRow}>
          <span style={styles.chip}>{value}</span>
          <button type="button" style={styles.changeBtn} onClick={() => onChange('gene', null)}>
            change
          </button>
        </div>
      ) : (
        <div style={styles.searchWrap}>
          <input
            type="text"
            value={query}
            placeholder="Search a gene…"
            style={styles.input}
            onChange={(e) => { setQuery(e.target.value); searchGenes(e.target.value) }}
          />
          {results.length > 0 && (
            <div style={styles.results}>
              {results.map((g) => (
                <div
                  key={g}
                  style={styles.resultRow}
                  onClick={() => { onChange('gene', g); setQuery(''); clearResults() }}
                >
                  {g}
                </div>
              ))}
            </div>
          )}
        </div>
      )}
    </div>
  )
}

const styles: Record<string, React.CSSProperties> = {
  toggle: { display: 'flex', gap: 0, marginBottom: 6 },
  toggleBtn: {
    flex: 1, padding: '4px 6px', fontSize: 11, backgroundColor: '#1a1a2e', color: '#aaa',
    border: '1px solid #0f3460', cursor: 'pointer',
  },
  toggleActive: { backgroundColor: '#4ecdc4', color: '#000', borderColor: '#4ecdc4', fontWeight: 600 },
  input: {
    width: '100%', boxSizing: 'border-box', padding: '6px 8px', fontSize: 12,
    backgroundColor: '#1a1a2e', color: '#eee', border: '1px solid #0f3460', borderRadius: 4,
  },
  searchWrap: { position: 'relative' },
  results: {
    position: 'absolute', top: '100%', left: 0, right: 0, zIndex: 10, maxHeight: 160,
    overflowY: 'auto', backgroundColor: '#1a1a2e', border: '1px solid #0f3460', borderRadius: 4,
  },
  resultRow: { padding: '5px 8px', fontSize: 12, color: '#eee', cursor: 'pointer' },
  chipRow: { display: 'flex', alignItems: 'center', gap: 8 },
  chip: { flex: 1, padding: '6px 8px', fontSize: 12, backgroundColor: '#1a1a2e', color: '#4ecdc4', border: '1px solid #0f3460', borderRadius: 4 },
  changeBtn: { padding: '4px 8px', fontSize: 11, backgroundColor: '#0f3460', color: '#eee', border: 'none', borderRadius: 4, cursor: 'pointer' },
}
