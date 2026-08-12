import { useCallback, useEffect, useState } from 'react'
import { useStore } from '../store'
import { appendDataset, fetchVarIdentifierColumns, refreshSchema } from '../hooks/useData'

/**
 * Add gene symbols — for a dataset whose .var holds only Ensembl ids.
 *
 * The panel shows what the mapping will cost before it happens: which species
 * was detected from the ids themselves, how many map, how many do not, and how
 * many symbols collide. A dataset that does not need this is told so rather
 * than offered a button that would do nothing useful.
 */

const dark = {
  panel: '#16213e',
  border: '#0f3460',
  inset: '#0f1625',
  accent: '#4ecdc4',
  alert: '#e94560',
  warn: '#e9a23b',
  text: '#ccc',
  sub: '#aaa',
  faint: '#888',
}

interface Preview {
  species: string | null
  n_genes: number
  n_recognized: number
  existing_symbol_column: string | null
  n_mapped: number
  n_unmapped: number
  n_duplicate_symbols: number
  examples: { id: string; symbol: string }[]
  applicable: boolean
  reason: string
}

interface Result {
  species: string
  column: string
  n_genes: number
  n_mapped: number
  n_unmapped: number
  n_duplicate_symbols: number
  set_as_index: boolean
}

const n = (v: number) => v.toLocaleString()

export default function GeneSymbolModal() {
  const open = useStore((s) => s.geneSymbolModalOpen)
  const setOpen = useStore((s) => s.setGeneSymbolModalOpen)

  const [preview, setPreview] = useState<Preview | null>(null)
  const [loading, setLoading] = useState(false)
  const [busy, setBusy] = useState(false)
  const [asIndex, setAsIndex] = useState(true)
  const [error, setError] = useState<string | null>(null)
  const [result, setResult] = useState<Result | null>(null)

  const close = useCallback(() => setOpen(false), [setOpen])

  useEffect(() => {
    if (!open) return
    setPreview(null); setResult(null); setError(null); setAsIndex(true)
    setLoading(true)
    fetch(appendDataset('/api/var/symbol_mapping_preview'))
      .then(async (r) => {
        if (!r.ok) throw new Error((await r.json().catch(() => ({}))).detail || `HTTP ${r.status}`)
        return r.json()
      })
      .then(setPreview)
      .catch((e) => setError((e as Error).message))
      .finally(() => setLoading(false))
  }, [open])

  useEffect(() => {
    if (!open) return
    const onKey = (e: KeyboardEvent) => { if (e.key === 'Escape') close() }
    window.addEventListener('keydown', onKey)
    return () => window.removeEventListener('keydown', onKey)
  }, [open, close])

  if (!open) return null

  const apply = async () => {
    setBusy(true); setError(null)
    try {
      const r = await fetch(appendDataset('/api/var/map_symbols'), {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ set_as_index: asIndex }),
      })
      if (!r.ok) {
        const err = await r.json().catch(() => ({ detail: r.statusText }))
        throw new Error(err.detail || `HTTP ${r.status}`)
      }
      setResult(await r.json())
      // Two independent channels, and this needs both: the schema so the gene
      // index and counts are current, the identifier columns so the Gene IDs
      // picker offers the new column straight away.
      await refreshSchema()
      await fetchVarIdentifierColumns()
    } catch (e) {
      setError((e as Error).message)
    } finally {
      setBusy(false)
    }
  }

  const label = { fontSize: 11, color: dark.sub }
  const value = { fontSize: 12, color: dark.text }

  return (
    <div
      onClick={close}
      style={{
        position: 'fixed', inset: 0, backgroundColor: 'rgba(0,0,0,0.6)',
        display: 'flex', alignItems: 'center', justifyContent: 'center', zIndex: 1000,
      }}
    >
      <div
        onClick={(e) => e.stopPropagation()}
        style={{
          backgroundColor: dark.panel, border: `1px solid ${dark.border}`,
          borderRadius: 8, padding: 18, width: 520, maxHeight: '80vh',
          overflowY: 'auto', boxShadow: '0 8px 32px rgba(0,0,0,0.5)',
        }}
      >
        <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
          <div style={{ fontSize: 14, fontWeight: 600, color: dark.alert }}>
            Add gene symbols
          </div>
          <button
            onClick={close}
            style={{
              background: 'transparent', border: 'none', color: dark.sub,
              fontSize: 16, cursor: 'pointer',
            }}
          >
            ×
          </button>
        </div>
        <div style={{ fontSize: 11, color: dark.faint, marginTop: 2, marginBottom: 12 }}>
          Map Ensembl gene ids to official symbols, from tables shipped with xcell
        </div>

        {loading && <div style={value}>Checking this dataset…</div>}

        {preview && !result && (
          <>
            {!preview.applicable ? (
              <div style={{
                backgroundColor: dark.inset, border: `1px solid ${dark.border}`,
                borderRadius: 4, padding: 10, fontSize: 12, color: dark.warn,
              }}>
                {preview.reason}
              </div>
            ) : (
              <>
                <div style={{
                  backgroundColor: dark.inset, border: `1px solid ${dark.border}`,
                  borderRadius: 4, padding: 10, display: 'grid', gap: 4,
                }}>
                  <div style={{ display: 'flex', justifyContent: 'space-between' }}>
                    <span style={label}>Species (from the ids)</span>
                    <span style={{ ...value, color: dark.accent }}>{preview.species}</span>
                  </div>
                  <div style={{ display: 'flex', justifyContent: 'space-between' }}>
                    <span style={label}>Genes with a symbol</span>
                    <span style={value}>
                      {n(preview.n_mapped)} of {n(preview.n_genes)}
                      {' '}({((preview.n_mapped / Math.max(preview.n_genes, 1)) * 100).toFixed(1)}%)
                    </span>
                  </div>
                  {preview.n_unmapped > 0 && (
                    <div style={{ display: 'flex', justifyContent: 'space-between' }}>
                      <span style={label}>Not in the tables (keep their id)</span>
                      <span style={value}>{n(preview.n_unmapped)}</span>
                    </div>
                  )}
                  {preview.n_duplicate_symbols > 0 && (
                    <div style={{ display: 'flex', justifyContent: 'space-between' }}>
                      <span style={label}>Symbols used by more than one id</span>
                      <span style={value}>{n(preview.n_duplicate_symbols)}</span>
                    </div>
                  )}
                </div>

                {preview.examples.length > 0 && (
                  <div style={{ marginTop: 8, fontSize: 11, color: dark.faint }}>
                    {preview.examples.map((e) => (
                      <div key={e.id}>
                        <code>{e.id}</code> → <code style={{ color: dark.text }}>{e.symbol}</code>
                      </div>
                    ))}
                  </div>
                )}

                {preview.reason && (
                  <div style={{ marginTop: 8, fontSize: 11, color: dark.warn }}>
                    {preview.reason}
                  </div>
                )}

                <label style={{
                  display: 'flex', alignItems: 'center', gap: 6,
                  marginTop: 12, fontSize: 12, color: dark.text, cursor: 'pointer',
                }}>
                  <input
                    type="checkbox"
                    checked={asIndex}
                    onChange={(e) => setAsIndex(e.target.checked)}
                  />
                  Use symbols as gene names
                </label>
                <div style={{ fontSize: 11, color: dark.faint, marginTop: 2 }}>
                  {asIndex
                    ? 'Genes will be named by symbol everywhere. The ids stay in .var, and the Gene IDs picker can switch back.'
                    : 'Symbols are written to .var only. Switch to them later with the Gene IDs picker.'}
                </div>
              </>
            )}
          </>
        )}

        {result && (
          <div style={{
            backgroundColor: dark.inset, border: `1px solid ${dark.border}`,
            borderRadius: 4, padding: 10, fontSize: 12, color: dark.text,
          }}>
            Mapped <b>{n(result.n_mapped)}</b> of {n(result.n_genes)} {result.species} ids
            to symbols → <code>.var['{result.column}']</code>
            {result.set_as_index ? ', and made them the gene names.' : '.'}
            {result.n_unmapped > 0 && (
              <div style={{ color: dark.faint, marginTop: 4 }}>
                {n(result.n_unmapped)} gene{result.n_unmapped === 1 ? '' : 's'} had no
                symbol in the tables and kept their Ensembl id.
              </div>
            )}
          </div>
        )}

        {error && (
          <div style={{ marginTop: 10, fontSize: 12, color: dark.alert }}>{error}</div>
        )}

        <div style={{ display: 'flex', justifyContent: 'flex-end', gap: 8, marginTop: 14 }}>
          <button
            onClick={close}
            style={{
              padding: '5px 12px', fontSize: 12, backgroundColor: 'transparent',
              color: dark.sub, border: `1px solid ${dark.border}`,
              borderRadius: 4, cursor: 'pointer',
            }}
          >
            {result ? 'Close' : 'Cancel'}
          </button>
          {!result && (
            <button
              onClick={apply}
              disabled={busy || loading || !preview?.applicable}
              style={{
                padding: '5px 12px', fontSize: 12,
                backgroundColor: preview?.applicable ? dark.accent : dark.border,
                color: preview?.applicable ? '#12203a' : dark.faint,
                border: 'none', borderRadius: 4,
                cursor: preview?.applicable && !busy ? 'pointer' : 'default',
                opacity: busy ? 0.6 : 1,
              }}
            >
              {busy ? 'Mapping…' : 'Add symbols'}
            </button>
          )}
        </div>
      </div>
    </div>
  )
}
