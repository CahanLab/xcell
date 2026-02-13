import { useState, useCallback, useRef } from 'react'
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
  genes: string[]
}

interface ParsedFile {
  filename: string
  geneLists: ParsedGeneList[]
}

function parseGMT(text: string): ParsedGeneList[] {
  const lists: ParsedGeneList[] = []
  const lines = text.split(/\r?\n/).filter((l) => l.trim())
  for (const line of lines) {
    const parts = line.split('\t')
    if (parts.length < 3) continue
    const name = parts[0].trim()
    // parts[1] is description (skip), rest are genes
    const genes = parts.slice(2).map((g) => g.trim()).filter(Boolean)
    if (name && genes.length > 0) {
      lists.push({ name, genes })
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
    const genes = lines.slice(startIdx)
      .map((l) => l.trim().replace(/^["']|["']$/g, ''))
      .filter(Boolean)
    const name = hasHeader ? headerParts[0] : filename.replace(/\.\w+$/, '')
    return genes.length > 0 ? [{ name, genes }] : []
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
      lists.push({ name: names[col], genes })
    }
  }
  return lists
}

function parseJSON(text: string): ParsedGeneList[] {
  const data = JSON.parse(text)
  if (!Array.isArray(data)) return []

  const lists: ParsedGeneList[] = []
  for (const item of data) {
    if (item && typeof item.name === 'string' && Array.isArray(item.genes) && item.genes.length > 0) {
      lists.push({ name: item.name, genes: item.genes.filter((g: unknown) => typeof g === 'string' && g) })
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
  const fileInputRef = useRef<HTMLInputElement>(null)

  const handleClose = useCallback(() => {
    setImportModalOpen(false)
    setParsedFiles([])
    setError(null)
  }, [setImportModalOpen])

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
      const folderName = pf.filename.replace(/\.\w+$/, '')
      if (pf.geneLists.length === 1) {
        // Single list: add directly without folder
        addGeneSetToCategory('manual', pf.geneLists[0].name, pf.geneLists[0].genes)
      } else {
        // Multiple lists: wrap in folder
        addFolderToCategory('manual', folderName, pf.geneLists)
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
