import { useCallback, useEffect, useRef, useState } from 'react'
import { assertJsonResponse } from '../lib/foreignServer'

/**
 * Pick a server-side file by browsing the filesystem.
 *
 * Typing an absolute path from memory is the worst part of any "where is your
 * file?" field, so anything asking for one should offer this instead.
 *
 * `kind` selects what the backend surfaces — `data` for loadable datasets,
 * `classifier` for pickled PySingleCellNet classifiers — while navigation,
 * shortcuts and the remembered last directory stay identical either way.
 *
 * The Load dialog in App.tsx still has its own copy of this UI. It carries
 * combine-mode multi-select and dataset-specific entry types (10x folders and
 * file trios) that this component deliberately does not model; folding the two
 * together means generalizing selection semantics, which is worth doing
 * separately rather than as a side effect of adding a second caller.
 */

export type BrowseKind = 'data' | 'classifier'

export interface BrowseEntry {
  name: string
  type: string
  path: string
  size?: number
}

/** Remembered per kind — where you keep classifiers is not where you keep data. */
const lastDirKey = (kind: BrowseKind) => `xcell_lastBrowseDir_${kind}`

export function formatSize(bytes: number): string {
  const KB = 1024, MB = KB * 1024, GB = MB * 1024
  if (bytes < MB) return `${(bytes / KB).toFixed(0)} KB`
  if (bytes < GB) return `${(bytes / MB).toFixed(1)} MB`
  return `${(bytes / GB).toFixed(2)} GB`
}

export default function FileBrowser({
  kind,
  selectedPath,
  onSelect,
  onError,
  onNavigate,
  emptyMessage = 'No folders or matching files here',
  height = 240,
}: {
  kind: BrowseKind
  selectedPath?: string
  onSelect: (path: string) => void
  onError?: (message: string | null) => void
  /**
   * Fires on every successful navigation with the directory now shown and its
   * contents. A "save as" caller needs the directory (that is what the user is
   * choosing) and the listing (to warn before clobbering an existing file).
   */
  onNavigate?: (dir: string, entries: BrowseEntry[]) => void
  emptyMessage?: string
  height?: number
}) {
  const [entries, setEntries] = useState<BrowseEntry[]>([])
  const [current, setCurrent] = useState<string | null>(null)
  const [shortcuts, setShortcuts] = useState<{ name: string; path: string }[]>([])
  const [loading, setLoading] = useState(false)
  const lastDir = useRef<string | null>(localStorage.getItem(lastDirKey(kind)))
  // Held in a ref so a caller passing an inline callback doesn't restart
  // navigation on every render.
  const onNavigateRef = useRef(onNavigate)
  onNavigateRef.current = onNavigate

  const browse = useCallback(async (dirPath?: string) => {
    setLoading(true)
    try {
      const target = dirPath ?? lastDir.current
      const params = new URLSearchParams({ kind })
      if (target) params.set('path', target)
      const resp = await fetch(`/api/browse?${params}`)
      if (!resp.ok) {
        const err = await resp.json().catch(() => ({ detail: resp.statusText }))
        throw new Error(err.detail || `HTTP ${resp.status}`)
      }
      await assertJsonResponse(resp)
      const data = await resp.json()
      setEntries(data.entries)
      setCurrent(data.current)
      if (data.shortcuts) setShortcuts(data.shortcuts)
      lastDir.current = data.current
      localStorage.setItem(lastDirKey(kind), data.current)
      onError?.(null)
      onNavigateRef.current?.(data.current, data.entries)
    } catch (err) {
      // A directory that vanished or is unreadable shouldn't strand the user
      // in an empty pane with no way back.
      onError?.((err as Error).message)
      if (dirPath) {
        lastDir.current = null
        localStorage.removeItem(lastDirKey(kind))
      }
    } finally {
      setLoading(false)
    }
    // onError identity changes per render in most callers; re-creating browse
    // on it would restart navigation.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [kind])

  useEffect(() => { browse() }, [browse])

  const crumbs = (current ?? '').split('/').filter(Boolean)

  return (
    <div style={{ display: 'flex', gap: 10, minHeight: 0 }}>
      <div style={{ width: 110, flexShrink: 0, display: 'flex', flexDirection: 'column', gap: 2 }}>
        {shortcuts.map((sc) => (
          <div
            key={sc.path}
            onClick={() => browse(sc.path)}
            style={{
              padding: '4px 8px', fontSize: 11, borderRadius: 4, cursor: 'pointer',
              color: current === sc.path ? '#e94560' : '#aaa',
              backgroundColor: current === sc.path ? 'rgba(233,69,96,0.1)' : 'transparent',
              whiteSpace: 'nowrap', overflow: 'hidden', textOverflow: 'ellipsis',
            }}
          >
            {sc.name}
          </div>
        ))}
      </div>

      <div style={{ flex: 1, minWidth: 0, display: 'flex', flexDirection: 'column', gap: 4 }}>
        {/* Breadcrumb — every segment navigable, so you can jump back up. */}
        <div style={{ fontSize: 10, color: '#888', display: 'flex', flexWrap: 'wrap', gap: 2 }}>
          <span onClick={() => browse('/')} style={{ cursor: 'pointer' }}>/</span>
          {crumbs.map((part, i) => (
            <span key={i}>
              <span
                onClick={() => browse('/' + crumbs.slice(0, i + 1).join('/'))}
                style={{ cursor: 'pointer', color: i === crumbs.length - 1 ? '#ddd' : '#888' }}
              >
                {part}
              </span>
              {i < crumbs.length - 1 && <span style={{ color: '#555' }}>{' / '}</span>}
            </span>
          ))}
        </div>

        <div style={{
          height, overflowY: 'auto', backgroundColor: '#0d1b30',
          border: '1px solid #1a1a2e', borderRadius: 4,
        }}>
          {loading ? (
            <div style={centeredNote}>Loading…</div>
          ) : entries.length === 0 ? (
            <div style={{ ...centeredNote, color: '#666' }}>{emptyMessage}</div>
          ) : (
            entries.map((entry) => {
              const isDir = entry.type === 'directory'
              const isSelected = entry.path === selectedPath
              return (
                <div
                  key={entry.path}
                  onClick={() => (isDir ? browse(entry.path) : onSelect(entry.path))}
                  style={{
                    padding: '4px 9px', fontSize: 12, cursor: 'pointer',
                    color: isDir ? '#aaa' : '#e94560',
                    display: 'flex', alignItems: 'center', justifyContent: 'space-between',
                    backgroundColor: isSelected ? 'rgba(233,69,96,0.15)' : 'transparent',
                    borderBottom: '1px solid #1a1a2e',
                  }}
                  onMouseEnter={(e) => {
                    if (!isSelected) e.currentTarget.style.backgroundColor = 'rgba(255,255,255,0.03)'
                  }}
                  onMouseLeave={(e) => {
                    if (!isSelected) e.currentTarget.style.backgroundColor = 'transparent'
                  }}
                >
                  <span style={{ display: 'flex', alignItems: 'center', gap: 6, overflow: 'hidden' }}>
                    <span style={{ flexShrink: 0 }}>{isDir ? '📁' : '📄'}</span>
                    <span style={{ overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap' }}>
                      {entry.name}
                    </span>
                  </span>
                  {entry.size != null && (
                    <span style={{ fontSize: 10, color: '#666', flexShrink: 0, marginLeft: 8 }}>
                      {formatSize(entry.size)}
                    </span>
                  )}
                </div>
              )
            })
          )}
        </div>
      </div>
    </div>
  )
}

const centeredNote: React.CSSProperties = {
  padding: 20, textAlign: 'center', color: '#888', fontSize: 12,
}
