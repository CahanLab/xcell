/**
 * TerritoryPanel — Analyze → Spatial → Define Territories.
 *
 * A floating, NON-blocking panel (no backdrop) so the spatial plot stays usable
 * while drawing. Flow:
 *   1. Name a territory type ("prox-dist") and pick the section column.
 *   2. Start → computes a ring per section from that section's cells, arms the
 *      draw tool, and switches to the spatial embedding.
 *   3. Draw cuts across the tissue. The regions between them are derived by the
 *      backend on every edit and drawn under the cuts.
 *   4. Name each region in the face list. The name attaches to a point inside
 *      the face, so moving a cut later does not orphan it.
 *   5. Save → writes into the live adata's uns.
 *
 * The cuts are the stored object and the regions are derived, so neighbouring
 * territories share a boundary exactly: overlaps and gaps cannot be drawn.
 *
 * Rollback: delete this file, remove its mount in App.tsx, the Spatial
 * "Territories" launcher in ScanpyModal, and isTerritoryPanelOpen in store.ts.
 */
import { useEffect, useMemo, useState } from 'react'
import { useStore, type TerritoryDraft } from '../store'
import { appendDataset, refreshSchema, type ObsSummary } from '../hooks/useData'
import { ringFromCells, faceColor, sectionLabels } from '../lib/territoryGeometry'

const API = '/api'

interface Face {
  polygon: [number, number][]
  name: string | null
  area: number
  anchor: [number, number]
}

const dark = {
  panel: '#16213e', border: '#0f3460', inset: '#0f1625',
  accent: '#4ecdc4', alert: '#e94560', warn: '#e9a23b',
  text: '#eee', sub: '#aaa', faint: '#888',
}

const input: React.CSSProperties = {
  width: '100%', padding: '6px 8px', fontSize: 12, backgroundColor: dark.border,
  color: dark.text, border: '1px solid #1a1a2e', borderRadius: 4,
}

const ghost: React.CSSProperties = {
  padding: '5px 12px', fontSize: 12, backgroundColor: 'transparent',
  color: dark.sub, border: `1px solid ${dark.border}`, borderRadius: 4, cursor: 'pointer',
}

function primary(enabled: boolean): React.CSSProperties {
  return {
    padding: '5px 12px', fontSize: 12, fontWeight: 600,
    backgroundColor: enabled ? dark.accent : '#1a1a2e',
    color: enabled ? dark.panel : '#555', border: 'none', borderRadius: 4,
    cursor: enabled ? 'pointer' : 'not-allowed',
  }
}

export default function TerritoryPanel() {
  const isOpen = useStore((s) => s.isTerritoryPanelOpen)
  const setOpen = useStore((s) => s.setTerritoryPanelOpen)
  const draft = useStore((s) => s.territoryDraft)
  const setDraft = useStore((s) => s.setTerritoryDraft)
  const removeCut = useStore((s) => s.removeTerritoryCut)
  const setSection = useStore((s) => s.setTerritorySection)
  const setAnchor = useStore((s) => s.setTerritoryAnchor)
  const embedding = useStore((s) => s.embedding)
  const setSelectedEmbedding = useStore((s) => s.setSelectedEmbedding)
  const setInteractionMode = useStore((s) => s.setInteractionMode)
  const setDrawTool = useStore((s) => s.setDrawTool)

  const [typeName, setTypeName] = useState('prox-dist')
  const [sectionCol, setSectionCol] = useState('')
  const [summaries, setSummaries] = useState<ObsSummary[]>([])
  const [faces, setFaces] = useState<Face[]>([])
  const [busy, setBusy] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const [saved, setSaved] = useState<string | null>(null)

  useEffect(() => {
    if (!isOpen) return
    fetch(appendDataset(`${API}/obs/summaries`))
      .then((r) => (r.ok ? r.json() : null))
      .then((d) => {
        if (!Array.isArray(d)) return
        const cats = (d as ObsSummary[]).filter((s) => s.dtype === 'category')
        setSummaries(cats)
        setSectionCol((prev) => prev || (cats.find((c) => /section/i.test(c.name))?.name ?? ''))
      })
      .catch(() => {})
  }, [isOpen])

  // The face list mirrors what the plot draws — same endpoint, same debounce.
  useEffect(() => {
    const section = draft?.sections[draft.activeSection]
    if (!section) { setFaces([]); return }
    let cancelled = false
    const timer = setTimeout(() => {
      fetch(`${API}/territories/faces`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ ring: section.ring, cuts: section.cuts, anchors: section.anchors }),
      })
        .then((r) => (r.ok ? r.json() : null))
        .then((d) => { if (!cancelled && d) setFaces(d.faces) })
        .catch(() => {})
    }, 130)
    return () => { cancelled = true; clearTimeout(timer) }
  }, [draft])

  const start = async () => {
    setError(null); setBusy(true)
    try {
      if (!embedding?.coordinates?.length) {
        throw new Error('Open a spatial embedding first — territories are drawn on coordinates.')
      }
      const coords = embedding.coordinates as [number, number][]

      // One ring per section, from that section's own cells. Frozen here: a
      // later cell filter must not silently move a saved boundary.
      let sections: TerritoryDraft['sections']
      if (sectionCol) {
        const resp = await fetch(appendDataset(`${API}/obs/${encodeURIComponent(sectionCol)}`))
        if (!resp.ok) throw new Error(`Could not read '${sectionCol}'`)
        const body = await resp.json()
        // Codes, not labels — see sectionLabels. Keying by the code produces
        // sections that match no cell and assign everything to unassigned.
        const labels = sectionLabels(body.values, body.categories)
        const groups = new Map<string, [number, number][]>()
        labels.forEach((key, i) => {
          if (!groups.has(key)) groups.set(key, [])
          groups.get(key)!.push(coords[i])
        })
        sections = {}
        for (const [name, pts] of groups) {
          const ring = ringFromCells(pts)
          if (ring.length) sections[name] = { ring, cuts: [], anchors: [] }
        }
      } else {
        sections = { all: { ring: ringFromCells(coords), cuts: [], anchors: [] } }
      }
      if (!Object.keys(sections).length) throw new Error('No cells to bound')

      setDraft({
        typeName: typeName.trim() || 'territory',
        embedding: embedding.name,
        sectionCol: sectionCol || null,
        activeSection: Object.keys(sections)[0],
        sections,
      })
      setSelectedEmbedding(embedding.name)
      setInteractionMode('draw')
      setDrawTool('smooth_curve')
      setSaved(null)
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e))
    } finally {
      setBusy(false)
    }
  }

  const cancel = () => {
    setDraft(null)
    setInteractionMode('pan')
    setFaces([])
  }

  const save = async () => {
    if (!draft) return
    setBusy(true); setError(null)
    try {
      const resp = await fetch(
        appendDataset(`${API}/territories/${encodeURIComponent(draft.typeName)}`),
        {
          method: 'PUT',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({
            embedding: draft.embedding,
            section_col: draft.sectionCol,
            source: 'drawn',
            sections: draft.sections,
          }),
        },
      )
      const body = await resp.json()
      if (!resp.ok) throw new Error(body.detail || `HTTP ${resp.status}`)
      setSaved(`Saved “${draft.typeName}” — ${body.n_named} named region(s) across ${body.n_sections} section(s).`)
      setDraft(null)
      setInteractionMode('pan')
      setFaces([])
      await refreshSchema()
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e))
    } finally {
      setBusy(false)
    }
  }

  const section = draft?.sections[draft.activeSection]
  const namedCount = useMemo(
    () => faces.filter((f) => f.name).length,
    [faces],
  )

  if (!isOpen) return null

  return (
    <div style={{
      position: 'fixed', right: 16, top: 92, width: 300, zIndex: 60,
      backgroundColor: dark.panel, border: `1px solid ${dark.border}`,
      borderRadius: 8, color: dark.text, boxShadow: '0 6px 24px rgba(0,0,0,0.45)',
      maxHeight: 'calc(100vh - 120px)', overflowY: 'auto',
    }}>
      <div style={{
        display: 'flex', alignItems: 'center', justifyContent: 'space-between',
        padding: '10px 12px', borderBottom: `1px solid ${dark.border}`,
      }}>
        <div style={{ fontSize: 13, fontWeight: 600, color: dark.accent }}>Territories</div>
        <button onClick={() => { cancel(); setOpen(false) }} style={{ ...ghost, padding: '2px 8px' }}>×</button>
      </div>

      <div style={{ padding: 12 }}>
        {error && (
          <div style={{
            fontSize: 11, color: dark.alert, backgroundColor: 'rgba(233,69,96,0.15)',
            padding: 8, borderRadius: 4, marginBottom: 10,
          }}>{error}</div>
        )}
        {saved && (
          <div style={{
            fontSize: 11, color: dark.accent, backgroundColor: 'rgba(78,205,196,0.12)',
            padding: 8, borderRadius: 4, marginBottom: 10,
          }}>{saved}</div>
        )}

        {!draft ? (
          <>
            <label style={{ display: 'block', fontSize: 11, color: dark.sub, marginBottom: 4 }}>
              Territory type
            </label>
            <input value={typeName} onChange={(e) => setTypeName(e.target.value)}
                   placeholder="prox-dist" style={{ ...input, marginBottom: 4 }} />
            <div style={{ fontSize: 10, color: dark.faint, marginBottom: 10, lineHeight: 1.4 }}>
              One axis of anatomy per type. Each becomes an .obs column
              (<code>territory_{typeName || 'name'}</code>).
            </div>

            <label style={{ display: 'block', fontSize: 11, color: dark.sub, marginBottom: 4 }}>
              Section column
            </label>
            <select value={sectionCol} onChange={(e) => setSectionCol(e.target.value)}
                    style={{ ...input, marginBottom: 4 }}>
              <option value="">— treat as one tissue —</option>
              {summaries.map((s) => <option key={s.name} value={s.name}>{s.name}</option>)}
            </select>
            <div style={{ fontSize: 10, color: dark.faint, marginBottom: 12, lineHeight: 1.4 }}>
              Sections sit side by side in one coordinate space, so each gets its
              own boundary. Draw the same region in each and name it the same —
              it stays one label.
            </div>

            <button onClick={start} disabled={busy} style={primary(!busy)}>Start drawing</button>
          </>
        ) : (
          <>
            <div style={{ fontSize: 11, color: dark.sub, marginBottom: 8 }}>
              <strong style={{ color: dark.text }}>{draft.typeName}</strong>
              {draft.sectionCol ? ` · by ${draft.sectionCol}` : ' · one tissue'}
            </div>

            {Object.keys(draft.sections).length > 1 && (
              <>
                <div style={{ fontSize: 11, color: dark.sub, marginBottom: 4 }}>Section</div>
                <div style={{ display: 'flex', flexWrap: 'wrap', gap: 4, marginBottom: 10 }}>
                  {Object.entries(draft.sections).map(([name, s]) => (
                    <button key={name} onClick={() => setSection(name)}
                            style={{
                              padding: '3px 8px', fontSize: 11, borderRadius: 4, cursor: 'pointer',
                              backgroundColor: name === draft.activeSection ? dark.accent : 'transparent',
                              color: name === draft.activeSection ? dark.panel : dark.sub,
                              border: `1px solid ${name === draft.activeSection ? dark.accent : dark.border}`,
                              fontWeight: name === draft.activeSection ? 600 : 400,
                            }}>
                      {name} ({s.cuts.length})
                    </button>
                  ))}
                </div>
              </>
            )}

            <div style={{
              fontSize: 10.5, color: dark.faint, marginBottom: 10, lineHeight: 1.45,
              padding: '6px 8px', backgroundColor: dark.inset, borderRadius: 4,
            }}>
              Draw a line across the tissue to divide it. Ends are extended to the
              boundary automatically, so a cut that stops short still divides. A
              closed loop makes an island.
            </div>

            <div style={{ fontSize: 11, color: dark.sub, marginBottom: 4 }}>
              Cuts ({section?.cuts.length ?? 0})
            </div>
            {section && section.cuts.length === 0 && (
              <div style={{ fontSize: 11, color: dark.faint, marginBottom: 10 }}>
                None yet — draw one on the plot.
              </div>
            )}
            {section?.cuts.map((c, i) => (
              <div key={c.id} style={{
                display: 'flex', alignItems: 'center', justifyContent: 'space-between',
                padding: '3px 6px', marginBottom: 3, backgroundColor: dark.inset, borderRadius: 3,
              }}>
                <span style={{ fontSize: 11, color: dark.sub }}>
                  {c.closed ? 'loop' : 'cut'} {i + 1} · {c.points.length} pts
                </span>
                <button onClick={() => removeCut(c.id)}
                        style={{ ...ghost, padding: '0 6px', border: 'none', color: dark.alert }}>×</button>
              </div>
            ))}

            <div style={{ fontSize: 11, color: dark.sub, margin: '10px 0 4px' }}>
              Regions ({namedCount}/{faces.length} named)
            </div>
            {faces.map((f, i) => (
              <div key={`${f.anchor[0]},${f.anchor[1]}`} style={{
                display: 'flex', alignItems: 'center', gap: 6, marginBottom: 4,
              }}>
                <span style={{
                  width: 12, height: 12, borderRadius: 2, flex: '0 0 auto',
                  backgroundColor: faceColor(i, faces.length),
                  border: `1px solid ${dark.border}`,
                }} />
                <input
                  value={f.name ?? ''}
                  placeholder="name this region…"
                  onChange={(e) => setAnchor(e.target.value, f.anchor[0], f.anchor[1])}
                  style={{ ...input, padding: '3px 6px', fontSize: 11 }}
                />
              </div>
            ))}

            <div style={{ display: 'flex', justifyContent: 'space-between', gap: 8, marginTop: 12 }}>
              <button onClick={cancel} style={ghost}>Discard</button>
              <button onClick={save} disabled={busy || namedCount === 0}
                      style={primary(!busy && namedCount > 0)}>
                Save territories
              </button>
            </div>
            {namedCount === 0 && (
              <div style={{ fontSize: 10, color: dark.faint, marginTop: 6 }}>
                Name at least one region before saving.
              </div>
            )}
          </>
        )}
      </div>
    </div>
  )
}
