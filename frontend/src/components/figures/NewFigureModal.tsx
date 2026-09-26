import { useEffect, useMemo, useState } from 'react'
import { useStore } from '../../store'
import { createFigure, fetchEnrichmentResults } from '../../hooks/useData'
import { defaultTitle, FIGURE_KIND_LABELS, type FigureKind, type FigureRecord } from '../../lib/figures'
import type { EnrichmentSummary } from '../../lib/enrichment'

/** Pick a kind and its inputs, name it, create. Only the enrichment kinds
 *  are offered here; the barplot and expression heatmap are saved from their
 *  own tabs ("Save as figure"). */

const dark = {
  backdrop: { position: 'fixed' as const, inset: 0, backgroundColor: 'rgba(0,0,0,0.6)', zIndex: 1100, display: 'flex', alignItems: 'center', justifyContent: 'center' },
  card: { backgroundColor: '#16213e', border: '1px solid #0f3460', borderRadius: '8px', width: '520px', maxWidth: '92vw', padding: '16px 20px', color: '#eee' },
  title: { margin: '0 0 12px', fontSize: '15px', color: '#e94560', fontWeight: 600 },
  label: { display: 'block', fontSize: '12px', color: '#aaa', margin: '10px 0 4px' },
  select: { width: '100%', padding: '6px 8px', fontSize: '13px', backgroundColor: '#0f3460', color: '#eee', border: '1px solid #1a1a2e', borderRadius: '4px' },
  input: { width: '100%', padding: '6px 8px', fontSize: '13px', backgroundColor: '#0f3460', color: '#eee', border: '1px solid #1a1a2e', borderRadius: '4px', boxSizing: 'border-box' as const },
  row: { display: 'flex', justifyContent: 'flex-end', gap: '8px', marginTop: '16px' },
  button: { padding: '7px 14px', fontSize: '13px', borderRadius: '4px', cursor: 'pointer', border: 'none' },
  muted: { fontSize: '11px', color: '#888' },
  error: { fontSize: '12px', color: '#e94560', marginTop: '8px' },
}

const KINDS: FigureKind[] = ['enrichment_heatmap', 'enrichment_network']

export default function NewFigureModal({ onClose, onCreated, initialKey }: {
  onClose: () => void
  onCreated: (record: FigureRecord) => void
  initialKey?: string | null
}) {
  const activeSlot = useStore((s) => s.activeSlot)
  const [kind, setKind] = useState<FigureKind>('enrichment_heatmap')
  const [results, setResults] = useState<EnrichmentSummary[]>([])
  const [key, setKey] = useState(initialKey ?? '')
  const [title, setTitle] = useState('')
  const [busy, setBusy] = useState(false)
  const [error, setError] = useState<string | null>(null)

  useEffect(() => {
    fetchEnrichmentResults(activeSlot)
      .then((r) => {
        // collections first: they are what a heatmap wants
        const sorted = [...r].sort((a, b) => Number(Boolean(b.n_groups)) - Number(Boolean(a.n_groups)))
        setResults(sorted)
        setKey((k) => k || sorted[0]?.key || '')
      })
      .catch((e) => setError((e as Error).message))
  }, [activeSlot])

  useEffect(() => {
    const onKey = (e: KeyboardEvent) => { if (e.key === 'Escape') onClose() }
    window.addEventListener('keydown', onKey)
    return () => window.removeEventListener('keydown', onKey)
  }, [onClose])

  const labels = useMemo(() => Object.fromEntries(results.map((r) => [r.key, r.label])), [results])
  const placeholder = key ? defaultTitle(kind, { enrichment_keys: [key] }, labels) : ''

  const create = async () => {
    if (!key) return
    setBusy(true); setError(null)
    try {
      const rec = await createFigure({ kind, title: title.trim() || null, inputs: { enrichment_keys: [key] } }, activeSlot)
      onCreated(rec)
    } catch (e) {
      setError((e as Error).message)
    } finally {
      setBusy(false)
    }
  }

  return (
    <div style={dark.backdrop} onClick={onClose}>
      <div style={dark.card} onClick={(e) => e.stopPropagation()}>
        <h3 style={dark.title}>New figure</h3>
        <label style={dark.label}>Kind</label>
        <select style={dark.select} value={kind} onChange={(e) => setKind(e.target.value as FigureKind)}>
          {KINDS.map((k) => <option key={k} value={k}>{FIGURE_KIND_LABELS[k]}</option>)}
        </select>
        <label style={dark.label}>Enrichment result</label>
        {results.length === 0 ? (
          <div style={dark.muted}>No enrichment results stored yet. Run one from Analyze → Genes → Gene-set Enrichment (a batch over a column gives the best heatmap).</div>
        ) : (
          <select style={dark.select} value={key} onChange={(e) => setKey(e.target.value)}>
            {results.map((r) => (
              <option key={r.key} value={r.key}>{r.label}{r.n_groups ? ` (${r.n_groups} groups)` : ''} — {r.n_significant}/{r.n_sets_tested}</option>
            ))}
          </select>
        )}
        <label style={dark.label}>Title</label>
        <input style={dark.input} value={title} onChange={(e) => setTitle(e.target.value)} placeholder={placeholder} />
        {error && <div style={dark.error}>{error}</div>}
        <div style={dark.row}>
          <button style={{ ...dark.button, backgroundColor: '#0f3460', color: '#aaa' }} onClick={onClose}>Cancel</button>
          <button style={{ ...dark.button, backgroundColor: '#e94560', color: '#fff', opacity: key && !busy ? 1 : 0.5 }} disabled={!key || busy} onClick={create}>
            {busy ? 'Creating…' : 'Create'}
          </button>
        </div>
      </div>
    </div>
  )
}
