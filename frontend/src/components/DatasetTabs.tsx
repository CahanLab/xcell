import { useState } from 'react'
import { useStore } from '../store'
import { loadedSlots, datasetLabel } from '../lib/datasetSlots'

// One tab per loaded dataset: which one the panels act on, and — with Split —
// whether they are all on screen at once. Replaces a two-option <select> in the
// header that could only ever say "Primary" and "Secondary".
//
// Hidden until a second dataset is loaded. With one there is nothing to switch
// between, and the header already names the file.

const dark = {
  bar: {
    display: 'flex',
    alignItems: 'stretch',
    gap: '8px',
    padding: '4px 8px 0',
    backgroundColor: '#16213e',
    borderBottom: '1px solid #0f3460',
  },
  // Only the tabs scroll. Split sits outside that, so it stays reachable no
  // matter how many datasets are open.
  tabScroller: {
    display: 'flex',
    alignItems: 'stretch',
    gap: '2px',
    flex: 1,
    minWidth: 0,
    overflowX: 'auto' as const,
  },
  tab: {
    display: 'flex',
    alignItems: 'center',
    gap: '6px',
    maxWidth: '220px',
    padding: '5px 8px',
    fontSize: '11px',
    color: '#aaa',
    backgroundColor: '#0f1625',
    // Longhand throughout: the active tab overrides borderColor, and mixing
    // that with a `border` shorthand makes React drop one on every re-render.
    borderWidth: '1px 1px 0 1px',
    borderStyle: 'solid',
    borderColor: '#0f3460',
    borderRadius: '4px 4px 0 0',
    cursor: 'pointer',
    whiteSpace: 'nowrap' as const,
  },
  tabActive: {
    color: '#eee',
    backgroundColor: '#1a1a2e',
    borderColor: '#e94560',
  },
  name: {
    overflow: 'hidden',
    textOverflow: 'ellipsis',
    whiteSpace: 'nowrap' as const,
  },
  count: { color: '#888', fontSize: '10px', flex: '0 0 auto' },
  close: {
    flex: '0 0 auto',
    background: 'transparent',
    border: 'none',
    color: '#888',
    cursor: 'pointer',
    fontSize: '13px',
    lineHeight: 1,
    padding: '0 2px',
  },
  add: {
    padding: '5px 9px',
    fontSize: '12px',
    color: '#aaa',
    backgroundColor: 'transparent',
    border: '1px dashed #0f3460',
    borderRadius: '4px 4px 0 0',
    cursor: 'pointer',
  },
  split: {
    alignSelf: 'center',
    padding: '4px 10px',
    fontSize: '11px',
    color: '#aaa',
    backgroundColor: '#0f3460',
    border: '1px solid #1a1a2e',
    borderRadius: '4px',
    cursor: 'pointer',
  },
  splitActive: { backgroundColor: '#4ecdc4', color: '#000' },
}

export default function DatasetTabs({ onAddDataset }: { onAddDataset: () => void }) {
  const datasets = useStore((s) => s.datasets)
  const activeSlot = useStore((s) => s.activeSlot)
  const setActiveSlot = useStore((s) => s.setActiveSlot)
  const unloadDataset = useStore((s) => s.unloadDataset)
  const layoutMode = useStore((s) => s.layoutMode)
  const setLayoutMode = useStore((s) => s.setLayoutMode)
  const setError = useStore((s) => s.setError)
  const [closing, setClosing] = useState<string | null>(null)

  const slots = loadedSlots(datasets)
  if (slots.length < 2) return null

  const close = async (slot: string, label: string) => {
    const ok = window.confirm(
      `Close ${label}?\n\nEverything computed on it in this session is discarded — `
      + 'reloading the file starts over.'
    )
    if (!ok) return
    setClosing(slot)
    try {
      const r = await fetch(`/api/datasets/${encodeURIComponent(slot)}`, { method: 'DELETE' })
      // 404 means the backend already dropped it; the UI still has to catch up.
      if (!r.ok && r.status !== 404) {
        const detail = await r.json().catch(() => ({}))
        // Leave it in place rather than showing a dataset the backend still
        // holds as gone — the two would silently disagree from here on.
        setError(detail.detail || `Could not close ${label} (HTTP ${r.status})`)
        return
      }
      unloadDataset(slot)
    } catch (e) {
      setError(`Could not close ${label}: ${(e as Error).message}`)
    } finally {
      setClosing(null)
    }
  }

  return (
    <div style={dark.bar}>
      <div style={dark.tabScroller}>
      {slots.map((slot) => {
        const label = datasetLabel(slot, datasets[slot]?.schema?.filename)
        const isActive = slot === activeSlot
        return (
          <div
            key={slot}
            onClick={() => setActiveSlot(slot)}
            title={datasets[slot]?.schema?.filename ?? slot}
            style={{ ...dark.tab, ...(isActive ? dark.tabActive : {}) }}
          >
            <span style={dark.name}>{label}</span>
            <span style={dark.count}>
              {datasets[slot]?.schema?.n_cells.toLocaleString()}
            </span>
            <button
              style={dark.close}
              disabled={closing === slot}
              onClick={(e) => { e.stopPropagation(); close(slot, label) }}
              title={`Close ${label}`}
            >
              &times;
            </button>
          </div>
        )
      })}
      <button style={dark.add} onClick={onAddDataset} title="Load another dataset">
        +
      </button>
      </div>
      <button
        style={{ ...dark.split, ...(layoutMode === 'tiled' ? dark.splitActive : {}) }}
        onClick={() => setLayoutMode(layoutMode === 'tiled' ? 'single' : 'tiled')}
        title={layoutMode === 'tiled'
          ? 'Show only the active dataset'
          : 'Show every loaded dataset side by side'}
      >
        Split
      </button>
    </div>
  )
}
