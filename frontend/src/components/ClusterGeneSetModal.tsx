import { useState, useEffect, useMemo } from 'react'
import { useStore } from '../store'
import { runClusterGeneSet, useObsSummaries } from '../hooks/useData'

// NOTE on useObsSummaries: it returns `{ summaries, isLoading, error, refresh }`
// where `summaries: ObsSummary[]` and each ObsSummary has:
//   { name: string; dtype: 'category' | 'numeric' | 'string'; categories?: { value: string; count: number }[] }

type Method = 'hierarchical' | 'kmeans'
type CellContext = 'all' | 'selection' | 'annotation'

export default function ClusterGeneSetModal() {
  const source = useStore((s) => s.clusterModalSourceSet)
  const setSource = useStore((s) => s.setClusterModalSourceSet)
  const selectedCellIndices = useStore((s) => s.selectedCellIndices)
  const addFolderToCategory = useStore((s) => s.addFolderToCategory)
  const { summaries } = useObsSummaries()

  const [method, setMethod] = useState<Method>('hierarchical')
  const [k, setK] = useState(3)
  const [cellContext, setCellContext] = useState<CellContext>('all')
  const [annotationColumn, setAnnotationColumn] = useState<string>('')
  const [annotationValues, setAnnotationValues] = useState<Set<string>>(new Set())
  const [isRunning, setIsRunning] = useState(false)
  const [error, setError] = useState<string | null>(null)

  // Categorical obs columns usable for the annotation picker.
  const categoricalColumns = useMemo(() => {
    return summaries.filter((s) => s.dtype === 'category').map((s) => s.name)
  }, [summaries])

  // When the modal opens, reset form state and pick a default annotation column.
  // Full form reset fires only when the modal opens (source transitions
  // from null to a set). Depending on categoricalColumns here would clobber
  // user changes if summaries arrive after the modal is already open.
  useEffect(() => {
    if (!source) return
    setMethod('hierarchical')
    setK(3)
    setCellContext('all')
    setAnnotationValues(new Set())
    setError(null)
    setIsRunning(false)
  }, [source])

  // Pick a default annotation column once summaries have loaded, but only
  // if nothing is chosen yet — don't overwrite a user's selection.
  useEffect(() => {
    if (!source) return
    if (annotationColumn === '' && categoricalColumns.length > 0) {
      setAnnotationColumn(categoricalColumns[0])
    }
  }, [source, categoricalColumns, annotationColumn])

  // Find the current column's category values (from obs summaries).
  // ObsSummary.categories is { value: string; count: number }[] — map to raw strings.
  const currentColumnCategories: string[] = useMemo(() => {
    if (!annotationColumn) return []
    const summary = summaries.find((s) => s.name === annotationColumn)
    if (!summary || !summary.categories) return []
    return summary.categories.map((c) => c.value)
  }, [annotationColumn, summaries])

  // Close on Escape.
  useEffect(() => {
    if (!source) return
    const onKey = (e: KeyboardEvent) => {
      if (e.key === 'Escape') setSource(null)
    }
    document.addEventListener('keydown', onKey)
    return () => document.removeEventListener('keydown', onKey)
  }, [source, setSource])

  if (!source) return null

  const maxK = Math.min(source.genes.length - 1, 20)

  const isFormValid =
    k >= 2 &&
    k <= maxK &&
    (cellContext !== 'annotation' ||
      (annotationColumn !== '' && annotationValues.size > 0))

  const toggleAnnotationValue = (value: string) => {
    setAnnotationValues((prev) => {
      const next = new Set(prev)
      if (next.has(value)) next.delete(value)
      else next.add(value)
      return next
    })
  }

  const handleRun = async () => {
    setIsRunning(true)
    setError(null)
    try {
      const payload = {
        geneNames: source.genes,
        method,
        k,
        cellContext,
        ...(cellContext === 'selection' ? { cellIndices: selectedCellIndices } : {}),
        ...(cellContext === 'annotation' ? { annotationColumn, annotationValues: Array.from(annotationValues) } : {}),
      }
      const { clusters } = await runClusterGeneSet(payload)
      const now = new Date()
      const stamp = `${now.getFullYear()}-${String(now.getMonth() + 1).padStart(2, '0')}-${String(
        now.getDate(),
      ).padStart(2, '0')} ${String(now.getHours()).padStart(2, '0')}:${String(
        now.getMinutes(),
      ).padStart(2, '0')}`
      const folderName = `${source.name} sub-clusters (${stamp})`
      const resultSets = clusters.map((genes, i) => ({
        name: `${source.name} cluster ${i + 1}`,
        genes,
      }))
      addFolderToCategory('gene_clusters', folderName, resultSets)
      setSource(null)
    } catch (err) {
      setError(err instanceof Error ? err.message : String(err))
    } finally {
      setIsRunning(false)
    }
  }

  const selectionDisabled = selectedCellIndices.length === 0
  const annotationDisabled = categoricalColumns.length === 0

  return (
    <div
      onClick={() => setSource(null)}
      style={{
        position: 'fixed',
        inset: 0,
        backgroundColor: 'rgba(0, 0, 0, 0.6)',
        display: 'flex',
        alignItems: 'center',
        justifyContent: 'center',
        zIndex: 100,
      }}
    >
      <div
        onClick={(e) => e.stopPropagation()}
        style={{
          backgroundColor: '#16213e',
          border: '1px solid #0f3460',
          borderRadius: '8px',
          padding: '20px 24px',
          minWidth: '380px',
          maxWidth: '480px',
          color: '#eee',
        }}
      >
        <div style={{ fontSize: '15px', fontWeight: 600, marginBottom: '12px' }}>
          Cluster genes
        </div>
        <div style={{ fontSize: '12px', color: '#aaa', marginBottom: '16px' }}>
          Clustering: <span style={{ color: '#eee' }}>{source.name}</span> ({source.genes.length} genes)
        </div>

        <div style={{ marginBottom: '12px' }}>
          <label style={{ display: 'block', fontSize: '11px', color: '#888', marginBottom: '4px' }}>
            Method
          </label>
          <select
            value={method}
            onChange={(e) => setMethod(e.target.value as Method)}
            style={{
              width: '100%',
              padding: '6px 8px',
              fontSize: '12px',
              backgroundColor: '#0f3460',
              color: '#eee',
              border: '1px solid #1a1a2e',
              borderRadius: '4px',
            }}
          >
            <option value="hierarchical">Hierarchical (Ward linkage)</option>
            <option value="kmeans">K-means</option>
          </select>
        </div>

        <div style={{ marginBottom: '12px' }}>
          <label style={{ display: 'block', fontSize: '11px', color: '#888', marginBottom: '4px' }}>
            Number of clusters (K)
          </label>
          <input
            type="number"
            min={2}
            max={maxK}
            value={k}
            onChange={(e) => setK(Math.max(2, Math.min(maxK, parseInt(e.target.value) || 2)))}
            style={{
              width: '80px',
              padding: '6px 8px',
              fontSize: '12px',
              backgroundColor: '#0f3460',
              color: '#eee',
              border: '1px solid #1a1a2e',
              borderRadius: '4px',
            }}
          />
          <span style={{ fontSize: '10px', color: '#666', marginLeft: '8px' }}>
            (max {maxK})
          </span>
        </div>

        <div style={{ marginBottom: '12px' }}>
          <label style={{ display: 'block', fontSize: '11px', color: '#888', marginBottom: '4px' }}>
            Cell context
          </label>
          <div style={{ display: 'flex', flexDirection: 'column', gap: '4px' }}>
            <label style={{ fontSize: '12px', color: '#ccc' }}>
              <input
                type="radio"
                checked={cellContext === 'all'}
                onChange={() => setCellContext('all')}
                style={{ marginRight: '6px' }}
              />
              All cells
            </label>
            <label
              style={{
                fontSize: '12px',
                color: selectionDisabled ? '#555' : '#ccc',
                cursor: selectionDisabled ? 'not-allowed' : 'pointer',
              }}
              title={selectionDisabled ? 'Lasso-select cells first' : undefined}
            >
              <input
                type="radio"
                checked={cellContext === 'selection'}
                disabled={selectionDisabled}
                onChange={() => setCellContext('selection')}
                style={{ marginRight: '6px' }}
              />
              Current selection ({selectedCellIndices.length} cells)
            </label>
            <label
              style={{
                fontSize: '12px',
                color: annotationDisabled ? '#555' : '#ccc',
                cursor: annotationDisabled ? 'not-allowed' : 'pointer',
              }}
              title={annotationDisabled ? 'No categorical obs columns available' : undefined}
            >
              <input
                type="radio"
                checked={cellContext === 'annotation'}
                disabled={annotationDisabled}
                onChange={() => setCellContext('annotation')}
                style={{ marginRight: '6px' }}
              />
              Annotation category
            </label>
          </div>
        </div>

        {cellContext === 'annotation' && (
          <div style={{ marginBottom: '12px', paddingLeft: '20px' }}>
            <label style={{ display: 'block', fontSize: '11px', color: '#888', marginBottom: '4px' }}>
              Column
            </label>
            <select
              value={annotationColumn}
              onChange={(e) => {
                setAnnotationColumn(e.target.value)
                setAnnotationValues(new Set())
              }}
              style={{
                width: '100%',
                padding: '6px 8px',
                fontSize: '12px',
                backgroundColor: '#0f3460',
                color: '#eee',
                border: '1px solid #1a1a2e',
                borderRadius: '4px',
                marginBottom: '8px',
              }}
            >
              {categoricalColumns.map((col) => (
                <option key={col} value={col}>
                  {col}
                </option>
              ))}
            </select>
            <label style={{ display: 'block', fontSize: '11px', color: '#888', marginBottom: '4px' }}>
              Values ({annotationValues.size} selected)
            </label>
            <div
              style={{
                maxHeight: '160px',
                overflowY: 'auto',
                backgroundColor: '#0f3460',
                borderRadius: '4px',
                padding: '6px 8px',
              }}
            >
              {currentColumnCategories.length === 0 ? (
                <div style={{ fontSize: '11px', color: '#666' }}>No values available</div>
              ) : (
                currentColumnCategories.map((v) => (
                  <label
                    key={v}
                    style={{
                      display: 'block',
                      fontSize: '11px',
                      color: '#ccc',
                      marginBottom: '2px',
                      cursor: 'pointer',
                    }}
                  >
                    <input
                      type="checkbox"
                      checked={annotationValues.has(v)}
                      onChange={() => toggleAnnotationValue(v)}
                      style={{ marginRight: '6px' }}
                    />
                    {v}
                  </label>
                ))
              )}
            </div>
          </div>
        )}

        {error && (
          <div
            style={{
              padding: '8px 10px',
              marginBottom: '12px',
              backgroundColor: 'rgba(233, 69, 96, 0.15)',
              border: '1px solid #e94560',
              borderRadius: '4px',
              color: '#f4a0ac',
              fontSize: '11px',
            }}
          >
            {error}
          </div>
        )}

        <div style={{ display: 'flex', justifyContent: 'flex-end', gap: '8px', marginTop: '16px' }}>
          <button
            onClick={() => setSource(null)}
            disabled={isRunning}
            style={{
              padding: '6px 12px',
              fontSize: '12px',
              backgroundColor: 'transparent',
              color: '#888',
              border: '1px solid #888',
              borderRadius: '4px',
              cursor: isRunning ? 'not-allowed' : 'pointer',
            }}
          >
            Cancel
          </button>
          <button
            onClick={handleRun}
            disabled={!isFormValid || isRunning}
            style={{
              padding: '6px 14px',
              fontSize: '12px',
              fontWeight: 500,
              backgroundColor: isFormValid && !isRunning ? '#4ecdc4' : '#1a1a2e',
              color: isFormValid && !isRunning ? '#000' : '#666',
              border: 'none',
              borderRadius: '4px',
              cursor: isFormValid && !isRunning ? 'pointer' : 'not-allowed',
            }}
          >
            {isRunning ? 'Running…' : 'Run'}
          </button>
        </div>
      </div>
    </div>
  )
}
