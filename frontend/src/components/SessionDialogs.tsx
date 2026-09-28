import { useState } from 'react'
import { useStore } from '../store'
import ConfirmDialog from './ConfirmDialog'
import { loadedSlots, datasetLabel } from '../lib/datasetSlots'
import { clearWorkspace } from '../lib/workspaceLayout'
import { countGeneSets } from '../lib/geneSetManager'
import { downloadGeneSetsJson, flattenGeneSetsForExport } from '../utils/exportGeneSets'

const plural = (n: number, one: string, many = `${one}s`) => `${n.toLocaleString()} ${n === 1 ? one : many}`

function describeGeneSets(c: { sets: number; folders: number }): string {
  if (c.folders === 0) return plural(c.sets, 'gene set')
  return `${plural(c.sets, 'gene set')} in ${plural(c.folders, 'folder')}`
}

/** File → New session. Unloads every dataset on the server, then reloads the
 *  page: a reload is the only fresh start that is actually fresh — global
 *  modals never unmount and many panels keep local state and refs, so an
 *  in-place store reset would leave pieces of the old analysis behind. */
export function NewSessionDialog({
  open,
  onClose,
  onExport,
}: {
  open: boolean
  onClose: () => void
  onExport: () => void
}) {
  const datasets = useStore((s) => s.datasets)
  const geneSetCategories = useStore((s) => s.geneSetCategories)
  const [alsoGeneSets, setAlsoGeneSets] = useState(false)
  const [busy, setBusy] = useState(false)
  const [error, setError] = useState<string | null>(null)

  // Every open starts unticked: wiping gene sets is never carried over from
  // a previous visit to this dialog.
  const [prevOpen, setPrevOpen] = useState(open)
  if (open !== prevOpen) {
    setPrevOpen(open)
    if (open) { setAlsoGeneSets(false); setError(null); setBusy(false) }
  }

  const slots = loadedSlots(datasets)
  const counts = countGeneSets(geneSetCategories)

  const confirm = async () => {
    setBusy(true); setError(null)
    try {
      const r = await fetch('/api/session/reset', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ clear_gene_sets: alsoGeneSets }),
      })
      if (!r.ok) {
        const detail = await r.json().catch(() => ({}))
        throw new Error(detail.detail || `HTTP ${r.status}`)
      }
      clearWorkspace()
      window.location.reload()
      // Stay busy: the page is going away.
    } catch (e) {
      setError(`Could not reset: ${e instanceof Error ? e.message : String(e)}`)
      setBusy(false)
    }
  }

  return (
    <ConfirmDialog
      open={open}
      title="Start a new session?"
      confirmLabel="Start new session"
      busy={busy}
      error={error}
      onConfirm={confirm}
      onCancel={onClose}
    >
      {slots.length > 0 ? (
        <>
          <div>These datasets will be unloaded:</div>
          <ul style={{ margin: '6px 0 8px', paddingLeft: '18px', maxHeight: '120px', overflowY: 'auto' }}>
            {slots.map((slot) => {
              const ds = datasets[slot]
              return (
                <li key={slot}>
                  <span style={{ color: '#eee' }}>{datasetLabel(slot, ds?.schema?.filename, ds?.displayName)}</span>
                  {ds?.schema && <span style={{ color: '#888' }}> · {ds.schema.n_cells.toLocaleString()} cells</span>}
                </li>
              )
            })}
          </ul>
          <div>
            Clusterings, embeddings, figures and the analysis record live only in memory until they are exported.{' '}
            <button
              onClick={() => { onClose(); onExport() }}
              disabled={busy}
              style={{ background: 'none', border: 'none', padding: 0, color: '#4ecdc4', cursor: 'pointer', fontSize: '12px', textDecoration: 'underline' }}
            >
              Export first…
            </button>
          </div>
        </>
      ) : (
        <div>No datasets are loaded.</div>
      )}
      <label style={{ display: 'flex', alignItems: 'center', gap: '8px', marginTop: '12px', cursor: counts.sets + counts.folders > 0 ? 'pointer' : 'default', color: counts.sets + counts.folders > 0 ? '#ddd' : '#666' }}>
        <input
          type="checkbox"
          checked={alsoGeneSets}
          disabled={busy || counts.sets + counts.folders === 0}
          onChange={(e) => setAlsoGeneSets(e.target.checked)}
        />
        Also delete all gene sets
        {counts.sets + counts.folders > 0 && <span style={{ color: '#888' }}>({describeGeneSets(counts)})</span>}
      </label>
      {!alsoGeneSets && counts.sets > 0 && (
        <div style={{ fontSize: '11px', color: '#888', marginTop: '4px', marginLeft: '24px' }}>
          Gene sets are kept — they are not tied to a dataset.
        </div>
      )}
      <div style={{ fontSize: '11px', color: '#888', marginTop: '10px' }}>xcell reloads when this is done.</div>
    </ConfirmDialog>
  )
}

/** Clear all gene sets — one global instance, opened from the File menu, the
 *  Genes pane's ⋯ and the Gene Set Manager. */
export function ClearGeneSetsDialog() {
  const open = useStore((s) => s.isClearGeneSetsOpen)
  const setOpen = useStore((s) => s.setClearGeneSetsOpen)
  const geneSetCategories = useStore((s) => s.geneSetCategories)
  const clearAllGeneSets = useStore((s) => s.clearAllGeneSets)
  const [backedUp, setBackedUp] = useState(false)

  const [prevOpen, setPrevOpen] = useState(open)
  if (open !== prevOpen) {
    setPrevOpen(open)
    if (open) setBackedUp(false)
  }

  const counts = countGeneSets(geneSetCategories)
  const empty = counts.sets + counts.folders === 0

  const backup = () => {
    // Local time, as the user reads a clock: 20260928-0940, not UTC's 1340.
    const d = new Date()
    const two = (n: number) => String(n).padStart(2, '0')
    const stamp = `${d.getFullYear()}${two(d.getMonth() + 1)}${two(d.getDate())}-${two(d.getHours())}${two(d.getMinutes())}`
    downloadGeneSetsJson(`xcell_gene_sets_${stamp}.json`, flattenGeneSetsForExport(geneSetCategories))
    setBackedUp(true)
  }

  return (
    <ConfirmDialog
      open={open}
      title="Clear all gene sets?"
      confirmLabel="Delete all gene sets"
      confirmDisabled={empty}
      onConfirm={() => { clearAllGeneSets(); setOpen(false) }}
      onCancel={() => setOpen(false)}
    >
      {empty ? (
        <div>There are no gene sets to clear.</div>
      ) : (
        <>
          <div>
            Deletes {describeGeneSets(counts)}
            {counts.categories > 1 && <> across {counts.categories} categories</>}. The Genes pane starts
            empty; this cannot be undone.
          </div>
          <div style={{ marginTop: '10px' }}>
            <button
              onClick={backup}
              style={{
                padding: '4px 10px', fontSize: '11px', backgroundColor: '#0f3460', color: '#4ecdc4',
                border: '1px solid #1a1a2e', borderRadius: '4px', cursor: 'pointer',
              }}
              title="A JSON file the Import button reads back, folders and down genes included"
            >
              {backedUp ? '✓ Backup downloaded' : 'Download a backup (JSON) first'}
            </button>
          </div>
        </>
      )}
    </ConfirmDialog>
  )
}
