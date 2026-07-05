import { useState, useCallback, useRef, useEffect } from 'react'
import { useStore } from '../store'

const styles = {
  overlay: {
    position: 'fixed' as const,
    top: 0,
    left: 0,
    right: 0,
    bottom: 0,
    backgroundColor: 'rgba(0, 0, 0, 0.7)',
    display: 'flex',
    alignItems: 'center',
    justifyContent: 'center',
    zIndex: 2000,
  },
  modal: {
    backgroundColor: '#16213e',
    borderRadius: '12px',
    border: '1px solid #0f3460',
    width: '600px',
    maxWidth: '95vw',
    maxHeight: '80vh',
    display: 'flex',
    flexDirection: 'column' as const,
    boxShadow: '0 8px 32px rgba(0, 0, 0, 0.4)',
  },
  header: {
    padding: '16px 20px',
    borderBottom: '1px solid #0f3460',
    display: 'flex',
    justifyContent: 'space-between',
    alignItems: 'center',
  },
  title: {
    fontSize: '16px',
    fontWeight: 600,
    color: '#e94560',
    margin: 0,
  },
  closeButton: {
    background: 'none',
    border: 'none',
    color: '#aaa',
    fontSize: '20px',
    cursor: 'pointer',
    padding: '4px 8px',
  },
  content: {
    flex: 1,
    overflow: 'auto',
    padding: '20px',
  },
  footer: {
    padding: '16px 20px',
    borderTop: '1px solid #0f3460',
    display: 'flex',
    justifyContent: 'flex-end',
    gap: '8px',
  },
  button: {
    padding: '8px 16px',
    fontSize: '13px',
    borderRadius: '6px',
    border: 'none',
    cursor: 'pointer',
    fontWeight: 500,
  },
  primaryButton: {
    backgroundColor: '#e94560',
    color: '#fff',
  },
  secondaryButton: {
    backgroundColor: '#0f3460',
    color: '#eee',
  },
  dropZone: {
    border: '2px dashed #0f3460',
    borderRadius: '8px',
    padding: '32px',
    textAlign: 'center' as const,
    cursor: 'pointer',
    transition: 'border-color 0.2s, background-color 0.2s',
  },
  dropZoneActive: {
    borderColor: '#4ecdc4',
    backgroundColor: 'rgba(78, 205, 196, 0.05)',
  },
}

interface ParsedGeneList {
  name: string
  genes: string[]          // up / positive
  genesDown?: string[]     // down / negative
  folder?: string          // optional sub-group (JSON only)
}

// A curated gene-set bundle shipped with xcell (read-only library).
interface LibrarySet {
  name: string
  genes: string[]
  genesDown?: string[]
  folder?: string
}
interface LibraryBundle {
  id: string
  name: string
  description: string
  count: number
  sets: LibrarySet[]
}

// Split a flat list of sets into per-folder groups plus the ungrouped
// remainder. Used when materialising a JSON file / bundle into folders: sets
// tagged with the same `folder` land together, matching the file's intent.
function groupByFolder<T extends { folder?: string }>(
  sets: T[],
): { ungrouped: T[]; folders: { name: string; sets: T[] }[] } {
  const ungrouped: T[] = []
  const order: string[] = []
  const byFolder = new Map<string, T[]>()
  for (const s of sets) {
    const f = s.folder
    if (f) {
      if (!byFolder.has(f)) { byFolder.set(f, []); order.push(f) }
      byFolder.get(f)!.push(s)
    } else {
      ungrouped.push(s)
    }
  }
  return { ungrouped, folders: order.map((name) => ({ name, sets: byFolder.get(name)! })) }
}

interface ParsedFile {
  filename: string
  geneLists: ParsedGeneList[]
}

// Split a flat token list by the UCell suffix convention: trailing '-' -> down,
// trailing '+' (or none) -> up. Returns cleaned symbols.
function splitSuffixDirection(tokens: string[]): { up: string[]; down: string[] } {
  const up: string[] = []
  const down: string[] = []
  for (const raw of tokens) {
    const t = raw.trim()
    if (!t) continue
    if (t.endsWith('-')) down.push(t.slice(0, -1))
    else if (t.endsWith('+')) up.push(t.slice(0, -1))
    else up.push(t)
  }
  return { up, down }
}

function parseGMT(text: string): ParsedGeneList[] {
  const lists: ParsedGeneList[] = []
  const lines = text.split(/\r?\n/).filter((l) => l.trim())
  for (const line of lines) {
    const parts = line.split('\t')
    if (parts.length < 3) continue
    const name = parts[0].trim()
    // parts[1] is description (skip), rest are genes
    const { up, down } = splitSuffixDirection(parts.slice(2))
    if (name && up.length + down.length > 0) {
      lists.push({ name, genes: up, genesDown: down.length ? down : undefined })
    }
  }
  return lists
}

function parseCSV(text: string, filename: string): ParsedGeneList[] {
  const lines = text.split(/\r?\n/).filter((l) => l.trim())
  if (lines.length === 0) return []

  // Detect delimiter: tab or comma
  const firstLine = lines[0]
  const delimiter = firstLine.includes('\t') ? '\t' : ','

  const headerParts = firstLine.split(delimiter).map((h) => h.trim().replace(/^["']|["']$/g, ''))

  // Check if first row looks like a header (non-gene-like values or column names)
  // Heuristic: if all values in first row appear again in column, it's not a header
  const hasHeader = headerParts.some((h) => /[^A-Za-z0-9._-]/.test(h) || h.length > 30) ||
    headerParts.some((h) => h.toLowerCase() === 'gene' || h.toLowerCase() === 'genes' || h.toLowerCase() === 'name')

  if (headerParts.length === 1) {
    // Single column: one gene per line
    const startIdx = hasHeader ? 1 : 0
    const tokens = lines.slice(startIdx)
      .map((l) => l.trim().replace(/^["']|["']$/g, ''))
      .filter(Boolean)
    const name = hasHeader ? headerParts[0] : filename.replace(/\.\w+$/, '')
    const { up, down } = splitSuffixDirection(tokens)
    return up.length + down.length > 0
      ? [{ name, genes: up, genesDown: down.length ? down : undefined }]
      : []
  }

  // Multiple columns: each column is a gene list
  const lists: ParsedGeneList[] = []
  const startIdx = hasHeader ? 1 : 0
  const names = hasHeader ? headerParts : headerParts.map((_, i) => `List ${i + 1}`)

  for (let col = 0; col < names.length; col++) {
    const genes: string[] = []
    for (let row = startIdx; row < lines.length; row++) {
      const parts = lines[row].split(delimiter)
      if (col < parts.length) {
        const gene = parts[col].trim().replace(/^["']|["']$/g, '')
        if (gene) genes.push(gene)
      }
    }
    if (genes.length > 0) {
      const { up, down } = splitSuffixDirection(genes)
      lists.push({ name: names[col], genes: up, genesDown: down.length ? down : undefined })
    }
  }
  return lists
}

function parseJSON(text: string): ParsedGeneList[] {
  const data = JSON.parse(text)
  // Accepted shapes:
  //   1. legacy array:           [{name, genes:[...]}]
  //   2. geneset-builder export: {sets:[{name, genes:[...]}]}
  //   3. .gsb.json:              genes may be [{symbol, from}]
  //   4. directional:            {name, up:[...], down:[...]} (synonyms below)
  //   Plus the '-'/'+' suffix convention inside any flat list.
  let rawSets: unknown[] | null = null
  if (Array.isArray(data)) rawSets = data
  else if (data && typeof data === 'object' && Array.isArray((data as { sets?: unknown }).sets)) {
    rawSets = (data as { sets: unknown[] }).sets
  }
  if (!rawSets) return []

  const toSymbols = (val: unknown): string[] => {
    if (!Array.isArray(val)) return []
    return val
      .map((g) => {
        if (typeof g === 'string') return g
        if (g && typeof g === 'object' && typeof (g as { symbol?: unknown }).symbol === 'string') {
          return (g as { symbol: string }).symbol
        }
        return ''
      })
      .filter((s) => s.length > 0)
  }

  const lists: ParsedGeneList[] = []
  for (const item of rawSets) {
    if (!item || typeof item !== 'object') continue
    const obj = item as Record<string, unknown>
    if (typeof obj.name !== 'string') continue
    const upRaw = toSymbols(obj.up ?? obj.positive ?? obj.genesUp ?? obj.genes)
    const downRaw = toSymbols(obj.down ?? obj.negative ?? obj.genesDown)
    // honor suffix convention inside the up list (e.g. ["CD8A","CCR7-"])
    const splitUp = splitSuffixDirection(upRaw)
    const up = splitUp.up
    const down = [...splitUp.down, ...downRaw.map((g) => g.replace(/[-+]$/, ''))]
    if (up.length + down.length > 0) {
      const folder = typeof obj.folder === 'string' && obj.folder.trim() ? obj.folder.trim() : undefined
      lists.push({ name: obj.name, genes: up, genesDown: down.length ? down : undefined, folder })
    }
  }
  return lists
}

function parseFile(text: string, filename: string): ParsedGeneList[] {
  const ext = filename.toLowerCase().split('.').pop() || ''

  if (ext === 'gmt') {
    return parseGMT(text)
  }

  if (ext === 'json') {
    return parseJSON(text)
  }

  // CSV and TXT: use generic parser
  return parseCSV(text, filename)
}

function PreviewTable({ parsedFile }: { parsedFile: ParsedFile }) {
  return (
    <div style={{
      backgroundColor: '#0a0f1a',
      borderRadius: '8px',
      overflow: 'hidden',
      marginTop: '16px',
    }}>
      <div style={{
        padding: '10px 14px',
        borderBottom: '1px solid #0f3460',
        display: 'flex',
        justifyContent: 'space-between',
        alignItems: 'center',
      }}>
        <span style={{ fontSize: '13px', fontWeight: 600, color: '#eee' }}>
          {parsedFile.filename}
        </span>
        <span style={{ fontSize: '11px', color: '#888' }}>
          {parsedFile.geneLists.length} gene list{parsedFile.geneLists.length !== 1 ? 's' : ''}
        </span>
      </div>
      <div style={{ maxHeight: '300px', overflow: 'auto' }}>
        {parsedFile.geneLists.map((gl, idx) => (
          <div
            key={idx}
            style={{
              padding: '8px 14px',
              borderBottom: '1px solid #1a1a2e',
              display: 'flex',
              justifyContent: 'space-between',
              alignItems: 'center',
            }}
          >
            <span style={{ fontSize: '12px', color: '#eee', fontFamily: 'monospace' }}>
              {gl.name}
            </span>
            <span style={{ fontSize: '11px', color: '#888' }}>
              {gl.genes.length} genes
            </span>
          </div>
        ))}
      </div>
    </div>
  )
}

export default function ImportModal() {
  const { isImportModalOpen, setImportModalOpen, addFolderToCategory, addGeneSetToCategory } = useStore()
  const [parsedFiles, setParsedFiles] = useState<ParsedFile[]>([])
  const [isDragOver, setIsDragOver] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const [bundles, setBundles] = useState<LibraryBundle[]>([])
  const [loadedBundleIds, setLoadedBundleIds] = useState<Set<string>>(new Set())
  const fileInputRef = useRef<HTMLInputElement>(null)

  // Fetch the shipped gene-set library when the modal opens (lazy — no cost
  // until the user actually opens Import). Failure is silent: the section
  // simply doesn't render.
  useEffect(() => {
    if (!isImportModalOpen) return
    let cancelled = false
    fetch('/api/gene_sets/library')
      .then((res) => (res.ok ? res.json() : Promise.reject(new Error(`HTTP ${res.status}`))))
      .then((data) => {
        if (!cancelled && Array.isArray(data?.bundles)) setBundles(data.bundles as LibraryBundle[])
      })
      .catch(() => { /* no library / offline → hide the section */ })
    return () => { cancelled = true }
  }, [isImportModalOpen])

  const handleClose = useCallback(() => {
    setImportModalOpen(false)
    setParsedFiles([])
    setError(null)
    setLoadedBundleIds(new Set())
  }, [setImportModalOpen])

  // Load a shipped bundle into the user's editable Manual sets. Sets tagged
  // with a `folder` group into their own folder; the rest go into a folder
  // named after the bundle. The modal stays open so several can be loaded.
  const handleLoadBundle = useCallback((bundle: LibraryBundle) => {
    const { ungrouped, folders } = groupByFolder(bundle.sets)
    if (ungrouped.length > 0) {
      addFolderToCategory('manual', bundle.name, ungrouped)
    }
    for (const grp of folders) {
      addFolderToCategory('manual', grp.name, grp.sets)
    }
    setLoadedBundleIds((prev) => new Set(prev).add(bundle.id))
  }, [addFolderToCategory])

  const processFiles = useCallback((files: FileList) => {
    setError(null)
    const newParsed: ParsedFile[] = []
    let remaining = files.length

    for (let i = 0; i < files.length; i++) {
      const file = files[i]
      const ext = file.name.toLowerCase().split('.').pop() || ''
      if (!['gmt', 'csv', 'txt', 'tsv', 'json'].includes(ext)) {
        setError(`Unsupported file type: .${ext}. Use .gmt, .csv, .txt, or .json`)
        remaining--
        if (remaining === 0) setParsedFiles((prev) => [...prev, ...newParsed])
        continue
      }

      const reader = new FileReader()
      reader.onload = (e) => {
        const text = e.target?.result as string
        const geneLists = parseFile(text, file.name)

        if (geneLists.length === 0) {
          setError(`No gene lists found in ${file.name}`)
        } else {
          newParsed.push({ filename: file.name, geneLists })
        }
        remaining--
        if (remaining === 0) {
          setParsedFiles((prev) => [...prev, ...newParsed])
        }
      }
      reader.readAsText(file)
    }
  }, [])

  const handleDrop = useCallback((e: React.DragEvent) => {
    e.preventDefault()
    setIsDragOver(false)
    if (e.dataTransfer.files.length > 0) {
      processFiles(e.dataTransfer.files)
    }
  }, [processFiles])

  const handleFileSelect = useCallback((e: React.ChangeEvent<HTMLInputElement>) => {
    if (e.target.files && e.target.files.length > 0) {
      processFiles(e.target.files)
    }
  }, [processFiles])

  const handleImport = useCallback(() => {
    for (const pf of parsedFiles) {
      const fileName = pf.filename.replace(/\.\w+$/, '')
      const { ungrouped, folders } = groupByFolder(pf.geneLists)
      // Ungrouped sets: a lone one is added directly; several wrap in a
      // folder named after the file (unchanged legacy behavior). Any sets
      // tagged with a `folder` become their own folders alongside.
      if (ungrouped.length === 1 && folders.length === 0) {
        const gl = ungrouped[0]
        addGeneSetToCategory('manual', gl.name, gl.genes, gl.genesDown)
      } else if (ungrouped.length > 0) {
        addFolderToCategory('manual', fileName, ungrouped)
      }
      for (const grp of folders) {
        addFolderToCategory('manual', grp.name, grp.sets)
      }
    }
    handleClose()
  }, [parsedFiles, addFolderToCategory, addGeneSetToCategory, handleClose])

  if (!isImportModalOpen) return null

  const totalLists = parsedFiles.reduce((sum, pf) => sum + pf.geneLists.length, 0)

  return (
    <div style={styles.overlay} onClick={handleClose}>
      <div style={styles.modal} onClick={(e) => e.stopPropagation()}>
        <div style={styles.header}>
          <h2 style={styles.title}>Import Gene Lists</h2>
          <button style={styles.closeButton} onClick={handleClose}>
            &times;
          </button>
        </div>

        <div style={styles.content}>
          {/* Drop zone */}
          <div
            style={{
              ...styles.dropZone,
              ...(isDragOver ? styles.dropZoneActive : {}),
            }}
            onDragOver={(e) => { e.preventDefault(); setIsDragOver(true) }}
            onDragLeave={() => setIsDragOver(false)}
            onDrop={handleDrop}
            onClick={() => fileInputRef.current?.click()}
          >
            <div style={{ fontSize: '24px', marginBottom: '8px', color: '#4ecdc4' }}>
              +
            </div>
            <div style={{ fontSize: '13px', color: '#aaa', marginBottom: '4px' }}>
              Drag and drop files here, or click to browse
            </div>
            <div style={{ fontSize: '11px', color: '#666' }}>
              Supported formats: .gmt, .csv, .txt, .json
            </div>
          </div>

          <input
            ref={fileInputRef}
            type="file"
            multiple
            accept=".gmt,.csv,.txt,.tsv,.json"
            onChange={handleFileSelect}
            style={{ display: 'none' }}
          />

          {error && (
            <div style={{ marginTop: '12px', fontSize: '12px', color: '#e94560' }}>
              {error}
            </div>
          )}

          {/* Bundled gene-set libraries shipped with xcell. Loaded on demand
              into the user's editable Manual sets. */}
          {bundles.length > 0 && (
            <div style={{ marginTop: '20px' }}>
              <div style={{
                display: 'flex', alignItems: 'center', gap: '10px', marginBottom: '10px',
              }}>
                <div style={{ flex: 1, height: 1, backgroundColor: '#0f3460' }} />
                <span style={{ fontSize: '11px', color: '#888', textTransform: 'uppercase', letterSpacing: '0.5px' }}>
                  or load a bundled library
                </span>
                <div style={{ flex: 1, height: 1, backgroundColor: '#0f3460' }} />
              </div>
              <div style={{ display: 'flex', flexDirection: 'column', gap: '8px' }}>
                {bundles.map((b) => {
                  const loaded = loadedBundleIds.has(b.id)
                  return (
                    <div
                      key={b.id}
                      style={{
                        display: 'flex', alignItems: 'center', gap: '12px',
                        padding: '10px 14px',
                        backgroundColor: '#0a0f1a',
                        border: '1px solid #0f3460',
                        borderRadius: '8px',
                      }}
                    >
                      <div style={{ flex: 1, minWidth: 0 }}>
                        <div style={{ fontSize: '13px', fontWeight: 600, color: '#eee' }}>
                          {b.name}
                          <span style={{ fontSize: '11px', fontWeight: 400, color: '#888', marginLeft: '8px' }}>
                            {b.count} set{b.count !== 1 ? 's' : ''}
                          </span>
                        </div>
                        {b.description && (
                          <div style={{ fontSize: '11px', color: '#888', marginTop: '2px' }}>
                            {b.description}
                          </div>
                        )}
                      </div>
                      <button
                        onClick={() => handleLoadBundle(b)}
                        style={{
                          ...styles.button,
                          padding: '6px 14px',
                          fontSize: '12px',
                          backgroundColor: loaded ? 'transparent' : '#0f3460',
                          color: loaded ? '#4ecdc4' : '#eee',
                          border: '1px solid ' + (loaded ? '#4ecdc4' : '#0f3460'),
                          flexShrink: 0,
                        }}
                        title={loaded ? 'Loaded into Manual — click to add again' : 'Load this library into your Manual gene sets'}
                      >
                        {loaded ? '✓ Loaded' : 'Load'}
                      </button>
                    </div>
                  )
                })}
              </div>
            </div>
          )}

          {/* Preview parsed files */}
          {parsedFiles.map((pf, idx) => (
            <PreviewTable key={idx} parsedFile={pf} />
          ))}

        </div>

        <div style={styles.footer}>
          <button
            style={{ ...styles.button, ...styles.secondaryButton }}
            onClick={handleClose}
          >
            Cancel
          </button>
          <button
            style={{
              ...styles.button,
              ...styles.primaryButton,
              opacity: totalLists === 0 ? 0.5 : 1,
            }}
            onClick={handleImport}
            disabled={totalLists === 0}
          >
            Import {totalLists > 0 ? `${totalLists} gene list${totalLists !== 1 ? 's' : ''}` : ''}
          </button>
        </div>
      </div>
    </div>
  )
}
