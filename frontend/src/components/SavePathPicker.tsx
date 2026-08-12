import { useEffect, useState } from 'react'
import FileBrowser, { type BrowseEntry, type BrowseKind } from './FileBrowser'
import { composeSavePath, splitSavePath } from '../lib/savePath'

/**
 * Choosing where to write a file: browse to a directory, type a name.
 *
 * The two halves are joined by {@link composeSavePath}, so the absolute path
 * shown underneath is the single source of truth and is exactly what will be
 * written. Existing files of the same kind are listed — partly so you can see
 * what is already there, partly so clicking one means "overwrite that".
 *
 * The overwrite warning is the point of tracking the directory's entries: this
 * writes to the user's own filesystem, so a collision has to be visible before
 * the button is pressed rather than discovered afterwards.
 */

const dark = {
  border: '#0f3460',
  inset: '#0f1625',
  warn: '#f0c987',
  text: '#ccc',
  sub: '#aaa',
  faint: '#888',
}

interface SavePathPickerProps {
  /** Which files to surface, and which directory to remember. */
  kind: BrowseKind
  /** Appended to a bare name. A list means "append the first, accept any". */
  ext: string | string[]
  /** Suggested filename, used until the user types their own. */
  defaultName: string
  /** Called with the composed absolute path, or '' when it is incomplete. */
  onChange: (path: string) => void
  height?: number
  onError?: (message: string | null) => void
}

export default function SavePathPicker({
  kind, ext, defaultName, onChange, height = 190, onError,
}: SavePathPickerProps) {
  const [dir, setDir] = useState('')
  const [name, setName] = useState(defaultName)
  const [entries, setEntries] = useState<BrowseEntry[]>([])

  // A new default (a different dataset, or a different export format) replaces
  // a name the user has not touched.
  useEffect(() => { setName(defaultName) }, [defaultName])

  const path = composeSavePath(dir, name, ext)
  useEffect(() => { onChange(path) }, [path, onChange])

  const basename = path ? path.slice(path.lastIndexOf('/') + 1) : ''
  const willOverwrite = Boolean(
    basename && entries.some((e) => e.type !== 'directory' && e.name === basename),
  )

  return (
    <div style={{ padding: 8, background: dark.inset, borderRadius: 4 }}>
      <FileBrowser
        kind={kind}
        height={height}
        emptyMessage="No folders or existing files here"
        onError={(m) => onError?.(m)}
        onNavigate={(d, list) => { setDir(d); setEntries(list) }}
        onSelect={(p) => {
          // Clicking an existing file means "write over that one".
          const split = splitSavePath(p)
          setDir(split.dir)
          setName(split.name)
        }}
      />
      <div style={{ display: 'flex', alignItems: 'center', gap: 6, marginTop: 8 }}>
        <span style={{ fontSize: 11, color: dark.sub, whiteSpace: 'nowrap' }}>
          File name
        </span>
        <input
          value={name}
          onChange={(e) => setName(e.target.value)}
          placeholder={defaultName}
          style={{
            flex: 1, padding: '4px 6px', fontSize: 12,
            backgroundColor: '#0d1b30', color: dark.text,
            border: `1px solid ${dark.border}`, borderRadius: 3,
          }}
        />
      </div>
      <div style={{
        marginTop: 6, fontSize: 10, color: dark.faint,
        fontFamily: 'ui-monospace, monospace', wordBreak: 'break-all',
      }}>
        {path || 'Pick a folder and type a name'}
      </div>
      {willOverwrite && (
        <div style={{ marginTop: 6, fontSize: 11, color: dark.warn }}>
          <code>{basename}</code> already exists here and will be overwritten.
        </div>
      )}
    </div>
  )
}
