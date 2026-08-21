/**
 * AssignTerritoriesModal — Analyze → Spatial → Assign Territories.
 *
 * Turns saved territory geometry into .obs columns: one per territory type,
 * plus an optional combined column crossing two or more of them (a P-D × A-P
 * grid, written as `proximal|anterior`). The combined column is text only —
 * there is no grid geometry, because nothing needs one.
 *
 * Rollback: delete this file, remove its mount in App.tsx, the Spatial
 * "Assign Territories" launcher in ScanpyModal, and isAssignTerritoriesOpen.
 */
import { useEffect, useState } from 'react'
import { useStore } from '../store'
import { appendDataset, refreshSchema } from '../hooks/useData'

const API = '/api'

interface StoredType {
  embedding: string
  section_col: string | null
  source: string
  created?: string
  sections: Record<string, { anchors: { name: string }[]; cuts: unknown[] }>
}

interface AssignResult {
  columns: string[]
  combined_column?: string
  types: Record<string, { column: string; counts: Record<string, number>; n_unplaced: number }>
}

const dark = {
  panel: '#16213e', border: '#0f3460', inset: '#0f1625',
  accent: '#4ecdc4', alert: '#e94560', text: '#eee', sub: '#aaa', faint: '#888',
}

const ghost: React.CSSProperties = {
  padding: '6px 14px', fontSize: 12, backgroundColor: 'transparent',
  color: dark.sub, border: `1px solid ${dark.border}`, borderRadius: 4, cursor: 'pointer',
}

function primary(enabled: boolean): React.CSSProperties {
  return {
    padding: '6px 14px', fontSize: 12, fontWeight: 600,
    backgroundColor: enabled ? dark.accent : '#1a1a2e',
    color: enabled ? dark.panel : '#555', border: 'none', borderRadius: 4,
    cursor: enabled ? 'pointer' : 'not-allowed',
  }
}

export default function AssignTerritoriesModal() {
  const isOpen = useStore((s) => s.isAssignTerritoriesOpen)
  const setOpen = useStore((s) => s.setAssignTerritoriesOpen)
  const refreshObsSummaries = useStore((s) => s.refreshObsSummaries)
  const addScanpyAction = useStore((s) => s.addScanpyAction)

  const [types, setTypes] = useState<Record<string, StoredType>>({})
  const [picked, setPicked] = useState<string[]>([])
  const [combine, setCombine] = useState(false)
  const [busy, setBusy] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const [result, setResult] = useState<AssignResult | null>(null)

  useEffect(() => {
    if (!isOpen) return
    setError(null); setResult(null)
    fetch(appendDataset(`${API}/territories`))
      .then((r) => (r.ok ? r.json() : null))
      .then((d) => {
        if (!d) return
        setTypes(d.territories || {})
        setPicked(Object.keys(d.territories || {}))
      })
      .catch(() => setError('Could not read this dataset’s territories.'))
  }, [isOpen])

  useEffect(() => {
    if (!isOpen) return
    const onKey = (e: KeyboardEvent) => { if (e.key === 'Escape') setOpen(false) }
    document.addEventListener('keydown', onKey)
    return () => document.removeEventListener('keydown', onKey)
  }, [isOpen, setOpen])

  if (!isOpen) return null

  const names = Object.keys(types)
  const canCombine = picked.length >= 2
  const combinedColumn = `territory_${picked.join('__')}`

  const toggle = (name: string) =>
    setPicked((prev) => (prev.includes(name) ? prev.filter((n) => n !== name) : [...prev, name]))

  const run = async () => {
    setBusy(true); setError(null)
    try {
      const resp = await fetch(appendDataset(`${API}/territories/assign`), {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ types: picked, combine: combine && canCombine }),
      })
      const body = await resp.json()
      if (!resp.ok) throw new Error(body.detail || `HTTP ${resp.status}`)
      setResult(body as AssignResult)
      // A new .obs column needs both refresh channels: the schema for anything
      // reading obs_dtypes, and the summaries for the Cell Manager's lists.
      await refreshSchema()
      refreshObsSummaries()
      addScanpyAction({
        action: 'assign_territories',
        params: { types: picked, combine: combine && canCombine },
        result: { columns: body.columns },
        timestamp: new Date().toISOString(),
      })
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e))
    } finally {
      setBusy(false)
    }
  }

  return (
    <div
      onClick={() => setOpen(false)}
      style={{
        position: 'fixed', inset: 0, backgroundColor: 'rgba(0,0,0,0.6)',
        display: 'flex', alignItems: 'center', justifyContent: 'center', zIndex: 100,
      }}
    >
      <div
        onClick={(e) => e.stopPropagation()}
        style={{
          backgroundColor: dark.panel, border: `1px solid ${dark.border}`, borderRadius: 8,
          padding: '20px 24px', width: 520, maxWidth: '94vw', maxHeight: '90vh',
          overflowY: 'auto', color: dark.text,
        }}
      >
        <div style={{ fontSize: 16, fontWeight: 600, marginBottom: 4 }}>Assign territories</div>
        <div style={{ fontSize: 12, color: dark.sub, marginBottom: 16 }}>
          Annotate every cell by the territory it occupies. One .obs column per type.
        </div>

        {names.length === 0 ? (
          <div style={{ fontSize: 12, color: dark.faint, padding: '12px 0' }}>
            No territories saved on this dataset yet — draw some with
            Analyze → Spatial → Define Territories, or import them from a
            spatial reference when you localize.
          </div>
        ) : (
          <>
            {names.map((name) => {
              const spec = types[name]
              const regions = [...new Set(
                Object.values(spec.sections).flatMap((s) => s.anchors.map((a) => a.name)),
              )]
              return (
                <label key={name} style={{
                  display: 'flex', gap: 8, alignItems: 'flex-start', padding: '8px 10px',
                  marginBottom: 6, backgroundColor: dark.inset, borderRadius: 4, cursor: 'pointer',
                }}>
                  <input type="checkbox" checked={picked.includes(name)}
                         onChange={() => toggle(name)} style={{ marginTop: 2 }} />
                  <div>
                    <div style={{ fontSize: 12 }}>
                      {name}
                      <span style={{ color: dark.faint }}> → territory_{name}</span>
                    </div>
                    <div style={{ fontSize: 10.5, color: dark.faint, marginTop: 2 }}>
                      {Object.keys(spec.sections).length} section(s) ·{' '}
                      {regions.length ? regions.join(', ') : 'no named regions'}
                      {spec.source.startsWith('imported') && ' · imported'}
                    </div>
                  </div>
                </label>
              )
            })}

            <label style={{
              display: 'flex', gap: 8, alignItems: 'flex-start', marginTop: 10,
              opacity: canCombine ? 1 : 0.5, cursor: canCombine ? 'pointer' : 'not-allowed',
            }}>
              <input type="checkbox" checked={combine && canCombine} disabled={!canCombine}
                     onChange={(e) => setCombine(e.target.checked)} style={{ marginTop: 2 }} />
              <div>
                <div style={{ fontSize: 12 }}>Also write a combined column</div>
                <div style={{ fontSize: 10.5, color: dark.faint, marginTop: 2 }}>
                  {canCombine
                    ? <>Crosses the ticked types into <code>{combinedColumn}</code>, with values like <code>proximal|anterior</code>.</>
                    : 'Tick two or more types to cross them.'}
                </div>
              </div>
            </label>

            {error && (
              <div style={{
                fontSize: 11, color: dark.alert, backgroundColor: 'rgba(233,69,96,0.15)',
                padding: 8, borderRadius: 4, marginTop: 12,
              }}>{error}</div>
            )}

            {result && (
              <div style={{
                marginTop: 12, padding: '8px 10px', backgroundColor: dark.inset,
                border: `1px solid ${dark.border}`, borderRadius: 4, fontSize: 11.5,
              }}>
                {Object.entries(result.types).map(([name, info]) => (
                  <div key={name} style={{ marginBottom: 6 }}>
                    <div style={{ color: dark.accent }}>{info.column}</div>
                    <div style={{ color: dark.sub }}>
                      {Object.entries(info.counts)
                        .sort((a, b) => b[1] - a[1])
                        .map(([label, n]) => `${label} ${n.toLocaleString()}`)
                        .join(' · ')}
                      {info.n_unplaced > 0 && (
                        <span style={{ color: dark.faint }}>
                          {' '}· {info.n_unplaced.toLocaleString()} with no coordinate (blank)
                        </span>
                      )}
                    </div>
                  </div>
                ))}
                {result.combined_column && (
                  <div style={{ color: dark.accent }}>{result.combined_column}</div>
                )}
              </div>
            )}

            <div style={{ display: 'flex', justifyContent: 'flex-end', gap: 8, marginTop: 16 }}>
              <button onClick={() => setOpen(false)} style={ghost}>
                {result ? 'Done' : 'Cancel'}
              </button>
              <button onClick={run} disabled={busy || picked.length === 0}
                      style={primary(!busy && picked.length > 0)}>
                {busy ? 'Assigning…' : 'Assign cells'}
              </button>
            </div>
          </>
        )}
      </div>
    </div>
  )
}
