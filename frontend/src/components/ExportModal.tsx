import { useCallback, useEffect, useMemo, useState } from 'react'
import SavePathPicker from './SavePathPicker'
import { appendDataset } from '../hooks/useData'
import type { BrowseKind } from './FileBrowser'

/**
 * File → Export.
 *
 * Exports used to stream a browser download under a fixed name, so every export
 * of every dataset landed in Downloads as `xcell_export.h5ad` and three
 * variants were indistinguishable. Here you pick the folder and the name, and
 * the backend writes it — which for a multi-gigabyte h5ad also skips a pointless
 * round trip through the browser.
 *
 * Downloading is still offered per format: it is the right thing when the
 * backend is not on the machine you are sitting at.
 */

const dark = {
  panel: '#16213e',
  border: '#0f3460',
  inset: '#0f1625',
  accent: '#4ecdc4',
  alert: '#e94560',
  text: '#eee',
  sub: '#aaa',
  faint: '#888',
}

type FormatKey = 'h5ad' | 'metadata' | 'gene_sets'

interface FormatSpec {
  label: string
  blurb: string
  kind: BrowseKind
  ext: string
  suffix: string
}

const FORMATS: Record<FormatKey, FormatSpec> = {
  h5ad: {
    label: 'AnnData (.h5ad)',
    blurb: 'Full dataset with any new annotation columns',
    kind: 'data', ext: '.h5ad', suffix: '_xcell.h5ad',
  },
  metadata: {
    label: 'Cell metadata (.tsv)',
    blurb: 'All cell annotations as tab-separated values',
    kind: 'tabular', ext: '.tsv', suffix: '_annotations.tsv',
  },
  gene_sets: {
    label: 'Gene sets (.json)',
    blurb: 'The gene sets curated in the Genes panel',
    kind: 'geneset', ext: '.json', suffix: '_gene_sets.json',
  },
}

const ORDER: FormatKey[] = ['h5ad', 'metadata', 'gene_sets']

const humanBytes = (n: number) =>
  n >= 1e9 ? `${(n / 1e9).toFixed(2)} GB`
    : n >= 1e6 ? `${(n / 1e6).toFixed(1)} MB`
      : n >= 1e3 ? `${(n / 1e3).toFixed(1)} kB`
        : `${n} bytes`

/** `E11.5_best_102424.h5ad` → `E11.5_best_102424`. */
function stemOf(filename: string | undefined): string {
  if (!filename) return 'xcell_export'
  const base = filename.slice(filename.lastIndexOf('/') + 1)
  const dot = base.lastIndexOf('.')
  return (dot > 0 ? base.slice(0, dot) : base) || 'xcell_export'
}

interface ExportModalProps {
  open: boolean
  onClose: () => void
  /** Loaded dataset's filename, used to propose an export name. */
  datasetName?: string
  /** Exactly the JSON written today, so the file's shape does not change. */
  geneSetsJson: string
  nGeneSets: number
  onDownload: (format: FormatKey) => void
}

export default function ExportModal({
  open, onClose, datasetName, geneSetsJson, nGeneSets, onDownload,
}: ExportModalProps) {
  const [format, setFormat] = useState<FormatKey>('h5ad')
  const [path, setPath] = useState('')
  const [busy, setBusy] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const [written, setWritten] = useState<{ path: string; n_bytes: number } | null>(null)

  const spec = FORMATS[format]
  const defaultName = useMemo(
    () => `${stemOf(datasetName)}${spec.suffix}`, [datasetName, spec.suffix],
  )

  useEffect(() => {
    if (!open) return
    setFormat('h5ad'); setError(null); setWritten(null)
  }, [open])

  useEffect(() => { setError(null); setWritten(null) }, [format])

  useEffect(() => {
    if (!open) return
    const onKey = (e: KeyboardEvent) => { if (e.key === 'Escape') onClose() }
    window.addEventListener('keydown', onKey)
    return () => window.removeEventListener('keydown', onKey)
  }, [open, onClose])

  const handleChange = useCallback((p: string) => setPath(p), [])

  if (!open) return null

  const emptyGeneSets = format === 'gene_sets' && nGeneSets === 0

  const save = async () => {
    setBusy(true); setError(null); setWritten(null)
    try {
      const body: Record<string, unknown> = { path, kind: format }
      // Only gene sets carry content: they are assembled in the browser.
      if (format === 'gene_sets') body.content = geneSetsJson
      const r = await fetch(appendDataset('/api/export/save'), {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(body),
      })
      if (!r.ok) {
        const err = await r.json().catch(() => ({ detail: r.statusText }))
        throw new Error(err.detail || `HTTP ${r.status}`)
      }
      setWritten(await r.json())
    } catch (e) {
      setError((e as Error).message)
    } finally {
      setBusy(false)
    }
  }

  return (
    <div
      onClick={onClose}
      style={{
        position: 'fixed', inset: 0, backgroundColor: 'rgba(0,0,0,0.6)',
        zIndex: 1000, display: 'flex', alignItems: 'center', justifyContent: 'center',
      }}
    >
      <div
        onClick={(e) => e.stopPropagation()}
        style={{
          backgroundColor: dark.panel, border: `1px solid ${dark.border}`,
          borderRadius: 8, padding: 20, width: 560, maxHeight: '86vh',
          overflowY: 'auto',
        }}
      >
        <div style={{ fontSize: 16, fontWeight: 600, color: dark.alert }}>
          Export data
        </div>
        <div style={{ fontSize: 11, color: dark.faint, marginTop: 2, marginBottom: 14 }}>
          Choose a folder and a name — xcell writes the file there
        </div>

        <div style={{ display: 'flex', gap: 6, marginBottom: 12 }}>
          {ORDER.map((key) => (
            <button
              key={key}
              onClick={() => setFormat(key)}
              style={{
                flex: 1, padding: '8px 10px', fontSize: 12, textAlign: 'left',
                backgroundColor: format === key ? dark.border : 'transparent',
                color: format === key ? dark.text : dark.sub,
                border: `1px solid ${format === key ? dark.accent : '#1a1a2e'}`,
                borderRadius: 6, cursor: 'pointer',
              }}
            >
              <div style={{ fontWeight: 500 }}>{FORMATS[key].label}</div>
              <div style={{ fontSize: 10, color: dark.faint, marginTop: 2 }}>
                {key === 'gene_sets'
                  ? (nGeneSets === 0 ? 'No gene sets defined'
                    : `${nGeneSets} gene set${nGeneSets === 1 ? '' : 's'}`)
                  : FORMATS[key].blurb}
              </div>
            </button>
          ))}
        </div>

        {emptyGeneSets ? (
          <div style={{
            padding: 10, backgroundColor: dark.inset, borderRadius: 4,
            fontSize: 12, color: dark.sub,
          }}>
            There are no gene sets to export yet. Curate some in the Genes panel
            first.
          </div>
        ) : (
          <SavePathPicker
            kind={spec.kind}
            ext={spec.ext}
            defaultName={defaultName}
            onChange={handleChange}
            onError={setError}
          />
        )}

        {written && (
          <div style={{
            marginTop: 10, padding: 10, backgroundColor: dark.inset,
            borderRadius: 4, fontSize: 12, color: dark.text,
          }}>
            Wrote <code style={{ color: dark.accent }}>{written.path}</code>
            {' '}({humanBytes(written.n_bytes)}).
          </div>
        )}

        {error && (
          <div style={{ marginTop: 10, fontSize: 12, color: dark.alert }}>{error}</div>
        )}

        <div style={{
          display: 'flex', alignItems: 'center', justifyContent: 'space-between',
          marginTop: 16,
        }}>
          <button
            onClick={() => onDownload(format)}
            style={{
              background: 'transparent', border: 'none', color: dark.faint,
              fontSize: 11, cursor: 'pointer', textDecoration: 'underline',
              padding: 0,
            }}
            title="Send it to your browser's downloads instead — useful when the backend is on another machine"
          >
            download to browser instead
          </button>
          <div style={{ display: 'flex', gap: 8 }}>
            <button
              onClick={onClose}
              style={{
                padding: '6px 14px', fontSize: 13, backgroundColor: 'transparent',
                color: dark.sub, border: `1px solid ${dark.border}`,
                borderRadius: 4, cursor: 'pointer',
              }}
            >
              {written ? 'Close' : 'Cancel'}
            </button>
            <button
              onClick={save}
              disabled={busy || !path || emptyGeneSets}
              style={{
                padding: '6px 14px', fontSize: 13,
                backgroundColor: path && !emptyGeneSets ? dark.accent : dark.border,
                color: path && !emptyGeneSets ? '#12203a' : dark.faint,
                border: 'none', borderRadius: 4,
                cursor: path && !busy && !emptyGeneSets ? 'pointer' : 'default',
                opacity: busy ? 0.6 : 1,
              }}
            >
              {busy ? 'Writing…' : 'Save'}
            </button>
          </div>
        </div>
      </div>
    </div>
  )
}
