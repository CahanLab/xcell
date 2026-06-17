/**
 * DefineSectionsPanel — Analyze → Spatial → Define Sections.
 *
 * A floating, NON-blocking panel (pinned to a corner, no backdrop) so the user
 * can use the polygon selection tool on the spatial plot while the panel guides
 * them. Flow:
 *   1. Name a new categorical .obs column (default "section").
 *   2. Start → creates the column, switches to the spatial embedding, and arms
 *      the polygon selection tool.
 *   3. For each section: polygon-select a region on the plot, type its name, and
 *      click "Add region" — the selected cells are labeled with that name.
 *   4. Finish → color by the new section column.
 *
 * Reuses the annotation API (createAnnotation / addLabelToAnnotation / labelCells).
 *
 * Rollback: delete this file, remove its mount in App.tsx, the Spatial "Define
 * Sections" launcher in ScanpyModal, and isDefineSectionsOpen in store.ts.
 */

import { useState } from 'react'
import { useStore } from '../store'
import { createAnnotation, addLabelToAnnotation, labelCells, refreshSchema } from '../hooks/useData'

export default function DefineSectionsPanel() {
  const isOpen = useStore((s) => s.isDefineSectionsOpen)
  const setOpen = useStore((s) => s.setDefineSectionsOpen)
  const schema = useStore((s) => s.schema)
  const selectedCellIndices = useStore((s) => s.selectedCellIndices)
  const clearSelection = useStore((s) => s.clearSelection)
  const setInteractionMode = useStore((s) => s.setInteractionMode)
  const setSelectionTool = useStore((s) => s.setSelectionTool)
  const setSelectedEmbedding = useStore((s) => s.setSelectedEmbedding)
  const setSelectedColorColumn = useStore((s) => s.setSelectedColorColumn)
  const refreshObsSummaries = useStore((s) => s.refreshObsSummaries)

  const [phase, setPhase] = useState<'setup' | 'drawing'>('setup')
  const [columnName, setColumnName] = useState('section')
  const [regionName, setRegionName] = useState('')
  const [added, setAdded] = useState<{ name: string; count: number }[]>([])
  const [busy, setBusy] = useState(false)
  const [error, setError] = useState<string | null>(null)

  if (!isOpen) return null

  const hasSpatial = !!schema?.embeddings?.includes('spatial')

  const reset = () => {
    setPhase('setup'); setColumnName('section'); setRegionName('')
    setAdded([]); setBusy(false); setError(null)
  }
  const close = () => { reset(); setOpen(false) }

  const armPolygon = () => {
    if (hasSpatial) setSelectedEmbedding('spatial')
    setInteractionMode('lasso')
    setSelectionTool('polygon')
    clearSelection()
  }

  const start = async () => {
    const name = columnName.trim()
    if (!name) { setError('Name the section column first.'); return }
    setError(null); setBusy(true)
    try {
      await createAnnotation(name)
      armPolygon()
      setPhase('drawing')
    } catch (e) {
      setError((e as Error).message)
    } finally {
      setBusy(false)
    }
  }

  const addRegion = async () => {
    const name = regionName.trim()
    if (!name) { setError('Name this region.'); return }
    if (selectedCellIndices.length === 0) { setError('Polygon-select a region first.'); return }
    setError(null); setBusy(true)
    try {
      await addLabelToAnnotation(columnName.trim(), name)
      await labelCells(columnName.trim(), name, selectedCellIndices)
      setAdded((prev) => [...prev, { name, count: selectedCellIndices.length }])
      setRegionName('')
      clearSelection()
    } catch (e) {
      setError((e as Error).message)
    } finally {
      setBusy(false)
    }
  }

  const finish = async () => {
    setBusy(true)
    try {
      await refreshSchema()
      refreshObsSummaries()
      if (added.length > 0) setSelectedColorColumn(columnName.trim())
    } finally {
      close()
    }
  }

  return (
    <div style={styles.panel}>
      <div style={styles.header}>
        <span style={styles.title}>Define sections</span>
        <button style={styles.close} onClick={close}>×</button>
      </div>

      {error && <div style={styles.error}>{error}</div>}

      {phase === 'setup' && (
        <div>
          <div style={styles.note}>
            Create a section column, then polygon-select each section on the spatial
            plot and name it.
          </div>
          {!hasSpatial && (
            <div style={styles.warn}>No "spatial" embedding found — sections will be
              drawn on the current view.</div>
          )}
          <label style={styles.label}>Section column name</label>
          <input
            style={styles.input}
            value={columnName}
            onChange={(e) => setColumnName(e.target.value)}
            placeholder="section"
          />
          <button style={styles.primary} disabled={busy} onClick={start}>
            {busy ? 'Starting…' : 'Start defining'}
          </button>
        </div>
      )}

      {phase === 'drawing' && (
        <div>
          <div style={styles.note}>
            <strong>Polygon tool armed.</strong> Click points on the plot to enclose a
            region, then name it and add it. Repeat for each section.
          </div>
          <div style={styles.added}>
            {added.length === 0 && <span style={styles.dim}>No sections added yet.</span>}
            {added.map((a, i) => (
              <div key={i} style={styles.addedRow}>
                <span>{a.name}</span><span style={styles.dim}>{a.count} cells</span>
              </div>
            ))}
          </div>
          <label style={styles.label}>
            Region name <span style={styles.dim}>({selectedCellIndices.length} cells selected)</span>
          </label>
          <input
            style={styles.input}
            value={regionName}
            onChange={(e) => setRegionName(e.target.value)}
            placeholder="e.g. section_1"
            onKeyDown={(e) => { if (e.key === 'Enter') addRegion() }}
          />
          <div style={styles.row}>
            <button
              style={styles.secondary}
              disabled={busy || selectedCellIndices.length === 0}
              onClick={addRegion}
            >
              {busy ? 'Adding…' : 'Add region'}
            </button>
            <button style={styles.secondary} onClick={armPolygon} title="Clear selection and re-arm the polygon tool">
              Re-arm tool
            </button>
          </div>
          <button style={styles.primary} disabled={busy} onClick={finish}>
            Finish ({added.length})
          </button>
        </div>
      )}
    </div>
  )
}

const styles: Record<string, React.CSSProperties> = {
  panel: {
    position: 'fixed', top: 70, right: 16, width: 280, zIndex: 1200,
    backgroundColor: '#16213e', color: '#eee', borderRadius: 8, padding: 14,
    border: '1px solid #0f3460', boxShadow: '0 8px 32px rgba(0,0,0,0.5)',
  },
  header: { display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: 8 },
  title: { fontSize: 14, fontWeight: 600 },
  close: { background: 'none', border: 'none', color: '#aaa', fontSize: 20, cursor: 'pointer', lineHeight: 1 },
  note: { fontSize: 12, color: '#bbb', backgroundColor: 'rgba(78,205,196,0.08)', padding: 8, borderRadius: 4, borderLeft: '3px solid #4ecdc4', marginBottom: 10, lineHeight: 1.45 },
  warn: { fontSize: 11.5, color: '#ffcf6a', marginBottom: 8 },
  error: { backgroundColor: 'rgba(233,69,96,0.15)', color: '#ff7a90', padding: '6px 8px', borderRadius: 4, marginBottom: 8, fontSize: 12 },
  label: { display: 'block', fontSize: 11, color: '#888', margin: '6px 0 4px' },
  input: { width: '100%', boxSizing: 'border-box', padding: '6px 8px', fontSize: 13, backgroundColor: '#0f3460', color: '#eee', border: '1px solid #1a1a2e', borderRadius: 4, marginBottom: 8 },
  row: { display: 'flex', gap: 8, marginBottom: 8 },
  primary: { width: '100%', padding: '8px 12px', fontSize: 13, backgroundColor: '#4ecdc4', color: '#0a0a1a', border: 'none', borderRadius: 4, cursor: 'pointer', fontWeight: 600 },
  secondary: { flex: 1, padding: '7px 10px', fontSize: 12, backgroundColor: '#0f3460', color: '#eee', border: '1px solid #1a1a2e', borderRadius: 4, cursor: 'pointer' },
  added: { maxHeight: 120, overflowY: 'auto', marginBottom: 6 },
  addedRow: { display: 'flex', justifyContent: 'space-between', fontSize: 12, padding: '2px 0' },
  dim: { color: '#888', fontSize: 11 },
}
