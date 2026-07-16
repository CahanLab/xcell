import { useEffect, useState } from 'react'
import { appendDataset, refreshSchema } from '../hooks/useData'
import { useStore, ScoreGeneSetsSource, GeneSetPerGeneNorm, GeneSetAggregation } from '../store'

interface Props {
  source: ScoreGeneSetsSource | null
  onClose: () => void
  onScored: (msg: string) => void  // caller toasts
}

const NORM_OPTIONS: [GeneSetPerGeneNorm, string][] = [
  ['none', 'None'],
  ['zscore_mad', 'Z-score (MAD)'],
  ['zscore_sd', 'Z-score (SD)'],
  ['minmax', 'Min–max'],
  ['rank', 'Rank'],
]
const AGG_OPTIONS: [GeneSetAggregation, string][] = [
  ['mean', 'Mean'],
  ['median', 'Median'],
  ['sum', 'Sum'],
  ['max', 'Max'],
]

interface ScoreResult {
  obsm_name: string
  columns: string[]
  n_sets: number
  skipped: { name: string; reason: string }[]
}

const sanitizeName = (folder: string) =>
  ('geneset_scores_' + folder).replace(/[^A-Za-z0-9_]+/g, '_').replace(/_+/g, '_').replace(/^_|_$/g, '')

export function ScoreGeneSetsModal({ source, onClose, onScored }: Props) {
  const dp = useStore((s) => s.displayPreferences)
  const displayLayer = useStore((s) => s.displayLayer)
  const setSelectedEmbedding = useStore((s) => s.setSelectedEmbedding)
  const setEmbedding = useStore((s) => s.setEmbedding)
  const setEmbeddingDims = useStore((s) => s.setEmbeddingDims)

  const [perGeneNorm, setPerGeneNorm] = useState<GeneSetPerGeneNorm>(dp.geneSetPerGeneNorm)
  const [perGeneClip, setPerGeneClip] = useState<number>(dp.geneSetPerGeneClip)
  const [aggregation, setAggregation] = useState<GeneSetAggregation>(dp.geneSetAggregation)
  const [obsmName, setObsmName] = useState('')
  const [overwrite, setOverwrite] = useState(false)
  const [busy, setBusy] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const [result, setResult] = useState<ScoreResult | null>(null)

  // View-two-scores picker (success step).
  const [colX, setColX] = useState('')
  const [colY, setColY] = useState('')

  useEffect(() => {
    if (!source) return
    setPerGeneNorm(dp.geneSetPerGeneNorm)
    setPerGeneClip(dp.geneSetPerGeneClip)
    setAggregation(dp.geneSetAggregation)
    setObsmName(sanitizeName(source.folderName))
    setOverwrite(false)
    setError(null)
    setResult(null)
    setColX(''); setColY('')
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [source])

  if (!source) return null

  const nSets = source.sets.length

  const run = async () => {
    if (!obsmName.trim()) { setError('Enter a name for the .obsm slot'); return }
    setBusy(true); setError(null)
    try {
      const resp = await fetch(appendDataset('/api/scanpy/score_gene_sets_matrix'), {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          sets: source.sets,
          per_gene_norm: perGeneNorm,
          per_gene_clip: perGeneClip,
          aggregation,
          obsm_name: obsmName.trim(),
          layer: displayLayer && displayLayer !== 'X' ? displayLayer : null,
          transform: dp.expressionTransform === 'log1p' ? 'log1p' : null,
          overwrite,
        }),
      })
      if (!resp.ok) throw new Error((await resp.json()).detail || 'Scoring failed')
      const data: ScoreResult = await resp.json()
      await refreshSchema()
      setResult(data)
      setColX(data.columns[0] ?? '')
      setColY(data.columns[1] ?? data.columns[0] ?? '')
      let msg = `Scored ${data.n_sets} set${data.n_sets === 1 ? '' : 's'} → .obsm['${data.obsm_name}']`
      if (data.skipped?.length) msg += ` — skipped ${data.skipped.length}`
      onScored(msg)
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e))
    } finally { setBusy(false) }
  }

  // View the score matrix with the two chosen columns as X/Y. No new .obsm slot —
  // the matrix itself is the embedding; the column picker (beside the embedding
  // selector) also lets the user switch which two scores are shown afterward.
  const viewScores = () => {
    if (!result || !colX || !colY) return
    const xi = result.columns.indexOf(colX)
    const yi = result.columns.indexOf(colY)
    setEmbeddingDims(result.obsm_name, xi >= 0 ? xi : 0, yi >= 0 ? yi : 1)
    setEmbedding(null)
    setSelectedEmbedding(result.obsm_name)
    onScored(`Viewing ${colX} × ${colY} — draw a line to find correlated genes`)
    onClose()
  }

  const label: React.CSSProperties = { display: 'block', margin: '10px 0 4px', color: '#bbb', fontSize: 12 }
  const control: React.CSSProperties = { width: '100%', padding: '5px 6px', background: '#0f3460', color: '#eee', border: '1px solid #1a1a2e', borderRadius: 4 }
  const primaryBtn: React.CSSProperties = { background: '#4ecdc4', color: '#000', border: 'none', padding: '6px 14px', borderRadius: 4, cursor: 'pointer' }

  return (
    <div style={{ position: 'fixed', inset: 0, background: 'rgba(0,0,0,0.5)', zIndex: 1000, display: 'flex', alignItems: 'center', justifyContent: 'center' }} onClick={onClose}>
      <div style={{ background: '#1e1e2e', color: '#eee', padding: 20, borderRadius: 8, minWidth: 400, maxWidth: 460, fontSize: 13 }} onClick={(e) => e.stopPropagation()}>
        <h3 style={{ marginTop: 0 }}>Score gene sets</h3>
        <p style={{ color: '#aaa', marginTop: 0 }}>
          Folder <strong>{source.folderName}</strong> — {nSets} set{nSets === 1 ? '' : 's'} → one <code>.obsm</code> score matrix (cells × sets). Mean pipeline (no UCell). Respects the active gene mask.
        </p>

        {!result && (
          <>
            <label style={label}>Per-gene normalization (across cells)
              <select style={control} value={perGeneNorm} onChange={(e) => setPerGeneNorm(e.target.value as GeneSetPerGeneNorm)}>
                {NORM_OPTIONS.map(([v, l]) => <option key={v} value={v}>{l}</option>)}
              </select>
            </label>
            {perGeneNorm === 'minmax' && (
              <label style={label}>Per-gene clip percentile
                <input type="number" style={control} step="0.5" min={0} max={50} value={perGeneClip}
                  onChange={(e) => setPerGeneClip(parseFloat(e.target.value) || 0)} />
              </label>
            )}
            <label style={label}>Aggregation across genes (per cell)
              <select style={control} value={aggregation} onChange={(e) => setAggregation(e.target.value as GeneSetAggregation)}>
                {AGG_OPTIONS.map(([v, l]) => <option key={v} value={v}>{l}</option>)}
              </select>
            </label>
            <label style={label}>.obsm slot name
              <input type="text" style={control} value={obsmName} onChange={(e) => setObsmName(e.target.value)} />
            </label>
            <label style={{ display: 'flex', alignItems: 'center', gap: 6, margin: '10px 0 0', color: '#bbb', fontSize: 12 }}>
              <input type="checkbox" checked={overwrite} onChange={(e) => setOverwrite(e.target.checked)} />
              Overwrite if a slot with this name already exists
            </label>
            {error && <div style={{ color: '#ff6b6b', margin: '10px 0 0' }}>{error}</div>}
            <div style={{ display: 'flex', gap: 8, justifyContent: 'flex-end', marginTop: 16 }}>
              <button onClick={onClose} disabled={busy}>Cancel</button>
              <button onClick={run} disabled={busy} style={primaryBtn}>{busy ? 'Scoring…' : 'Score sets'}</button>
            </div>
          </>
        )}

        {result && (
          <>
            <div style={{ background: '#16213e', borderRadius: 4, padding: '8px 10px', margin: '4px 0 10px' }}>
              Wrote <code>.obsm['{result.obsm_name}']</code> with {result.columns.length} column{result.columns.length === 1 ? '' : 's'}.
              {result.skipped?.length > 0 && (
                <div style={{ color: '#e0a458', marginTop: 4, fontSize: 12 }}>
                  Skipped (no usable genes): {result.skipped.map((s) => s.name).join(', ')}
                </div>
              )}
            </div>
            <p style={{ color: '#bbb', margin: '0 0 4px' }}>Create a 2-D embedding from two scores (optional):</p>
            <div style={{ display: 'flex', gap: 8 }}>
              <label style={{ ...label, flex: 1, margin: '2px 0' }}>X
                <select style={control} value={colX} onChange={(e) => setColX(e.target.value)}>
                  {result.columns.map((c) => <option key={c} value={c}>{c}</option>)}
                </select>
              </label>
              <label style={{ ...label, flex: 1, margin: '2px 0' }}>Y
                <select style={control} value={colY} onChange={(e) => setColY(e.target.value)}>
                  {result.columns.map((c) => <option key={c} value={c}>{c}</option>)}
                </select>
              </label>
            </div>
            <p style={{ color: '#888', fontSize: 11, margin: '6px 0 0' }}>
              Views these two score columns as the embedding. Use the “Axes” picker beside the
              embedding selector to switch columns later.
            </p>
            {error && <div style={{ color: '#ff6b6b', margin: '10px 0 0' }}>{error}</div>}
            <div style={{ display: 'flex', gap: 8, justifyContent: 'flex-end', marginTop: 16 }}>
              <button onClick={onClose}>Done</button>
              <button onClick={viewScores} disabled={!colX || !colY} style={primaryBtn}>
                View scores
              </button>
            </div>
          </>
        )}
      </div>
    </div>
  )
}
