import { useEffect, useState } from 'react'
import { useStore } from '../store'
import { fetchVarBooleanColumns, fetchVarColumnGenes, type VarBooleanColumn } from '../hooks/useData'

/** Lists boolean .var columns; each can be materialized as a frozen gene set. */
export default function VarColumnsSection() {
  const addGeneSetToCategory = useStore((s) => s.addGeneSetToCategory)
  const [columns, setColumns] = useState<VarBooleanColumn[]>([])
  const [expanded, setExpanded] = useState(true)
  const [error, setError] = useState<string | null>(null)
  const [busy, setBusy] = useState<string | null>(null)

  useEffect(() => {
    fetchVarBooleanColumns()
      .then(setColumns)
      .catch(() => setColumns([]))
  }, [])

  if (columns.length === 0) return null

  const addColumn = async (name: string) => {
    setBusy(name)
    setError(null)
    try {
      const genes = await fetchVarColumnGenes(name)
      addGeneSetToCategory('manual', name, genes)
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e))
    } finally {
      setBusy(null)
    }
  }

  return (
    <div style={{ marginBottom: '12px' }}>
      <div
        onClick={() => setExpanded(!expanded)}
        style={{ display: 'flex', alignItems: 'center', gap: '4px', cursor: 'pointer', fontSize: '11px', color: '#888', marginBottom: '4px' }}
      >
        <span>{expanded ? '▼' : '▶'}</span>
        <span>.var columns ({columns.length})</span>
      </div>
      {expanded && (
        <div style={{ backgroundColor: '#0f3460', borderRadius: '4px', padding: '6px 8px' }}>
          {columns.map((c) => (
            <div key={c.name} style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', fontSize: '11px', color: '#ccc', padding: '2px 0' }}>
              <span title={`${c.n_true} of ${c.n_total} genes`}>
                {c.name} <span style={{ color: '#888' }}>({c.n_true})</span>
              </span>
              <button
                onClick={() => addColumn(c.name)}
                disabled={busy === c.name}
                title="Add as a gene set in Manual"
                style={{ padding: '1px 8px', fontSize: '11px', backgroundColor: '#16213e', color: '#4ecdc4', border: '1px solid #4ecdc4', borderRadius: '3px', cursor: busy === c.name ? 'wait' : 'pointer' }}
              >
                {busy === c.name ? '…' : '+'}
              </button>
            </div>
          ))}
          {error && <div style={{ color: '#e94560', fontSize: '10px', marginTop: '4px' }}>{error}</div>}
        </div>
      )}
    </div>
  )
}
