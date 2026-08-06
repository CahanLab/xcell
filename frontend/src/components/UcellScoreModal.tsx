import { useEffect, useState } from 'react'
import { appendDataset } from '../hooks/useData'
import { LayerScaleBadge, layerOptionLabel, type LayerInfo } from './LayerScaleInfo'

export interface UcellTarget {
  sets: { name: string; up: string[]; down: string[] }[]
}

interface Props {
  target: UcellTarget | null
  onClose: () => void
  onScored: (msg: string) => void   // caller refreshes schema + toasts
}

export function UcellScoreModal({ target, onClose, onScored }: Props) {
  const [layers, setLayers] = useState<LayerInfo[]>([])
  const [layer, setLayer] = useState('counts')
  const [maxRank, setMaxRank] = useState(1500)
  const [wNeg, setWNeg] = useState(1.0)
  const [showAdvanced, setShowAdvanced] = useState(false)
  const [busy, setBusy] = useState(false)
  const [error, setError] = useState<string | null>(null)

  useEffect(() => {
    if (!target) return
    fetch(appendDataset('/api/scanpy/layers'))
      .then((r) => r.json())
      .then((d) => {
        const list: LayerInfo[] = d.layers ?? []
        setLayers(list)
        const names = list.map((l) => l.name)
        setLayer(names.includes('counts') ? 'counts' : 'X')
      })
      .catch(() => setLayers([{ name: 'X', density: 0 }]))
  }, [target])

  if (!target) return null

  const selectedLayer = layers.find((l) => l.name === layer)

  const run = async () => {
    setBusy(true); setError(null)
    try {
      const resp = await fetch(appendDataset('/api/scanpy/score_genes_ucell'), {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ sets: target.sets, layer, max_rank: maxRank, w_neg: wNeg }),
      })
      if (!resp.ok) throw new Error((await resp.json()).detail || 'Scoring failed')
      const data = await resp.json()
      const written = data.results.filter((r: { obs_column?: string }) => r.obs_column)
      const skipped = data.results.filter((r: { skipped?: string }) => r.skipped)
      const cols = written.map((r: { obs_column: string }) => r.obs_column).join(', ')
      let msg = written.length ? `Wrote ${written.length} UCell column(s): ${cols}` : 'No columns written'
      if (skipped.length) msg += ` — skipped ${skipped.length} (no up-genes)`
      onScored(msg)
      onClose()
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e))
    } finally {
      setBusy(false)
    }
  }

  return (
    <div style={{ position: 'fixed', inset: 0, background: 'rgba(0,0,0,0.5)', zIndex: 1000,
      display: 'flex', alignItems: 'center', justifyContent: 'center' }} onClick={onClose}>
      <div style={{ background: '#1e1e1e', color: '#eee', padding: 20, borderRadius: 8,
        minWidth: 360, fontSize: 13 }} onClick={(e) => e.stopPropagation()}>
        <h3 style={{ marginTop: 0 }}>Score with UCell</h3>
        <p style={{ color: '#aaa' }}>
          {target.sets.length} set{target.sets.length > 1 ? 's' : ''} →
          writes <code>UCell_&lt;name&gt;</code> obs column(s).
        </p>
        <div style={{ margin: '8px 0' }}>
          <div style={{ display: 'flex', alignItems: 'center', gap: 6, marginBottom: 4 }}>
            <span>Source layer</span>
            <LayerScaleBadge layer={selectedLayer} />
          </div>
          <select value={layer} onChange={(e) => setLayer(e.target.value)}
            style={{ width: '100%' }}>
            {layers.map((l) => <option key={l.name} value={l.name}>{layerOptionLabel(l)}</option>)}
          </select>
          {selectedLayer?.scale?.verdict === 'z_scored' && (
            <div style={{ marginTop: 6, padding: '6px 8px', fontSize: 11, lineHeight: 1.4,
              color: '#ffb3bd', background: 'rgba(233,69,96,0.15)', borderRadius: 4 }}>
              This layer looks scaled / z-scored. UCell ranks genes within each cell,
              so centered values give scores that are not comparable to the usual
              0–1 range. Pick a counts or log-normalized layer instead.
            </div>
          )}
        </div>
        <label style={{ display: 'block', margin: '8px 0' }}>maxRank
          <input type="number" value={maxRank} min={1}
            onChange={(e) => setMaxRank(Math.max(1, parseInt(e.target.value) || 1500))}
            style={{ width: '100%', marginTop: 4 }} />
        </label>
        <button onClick={() => setShowAdvanced((v) => !v)}
          style={{ background: 'none', color: '#4ecdc4', border: 'none', cursor: 'pointer', padding: 0 }}>
          {showAdvanced ? '▼' : '▶'} Advanced
        </button>
        {showAdvanced && (
          <label style={{ display: 'block', margin: '8px 0' }}>w_neg (down-set weight)
            <input type="number" step="0.1" value={wNeg}
              onChange={(e) => setWNeg(parseFloat(e.target.value) || 0)}
              style={{ width: '100%', marginTop: 4 }} />
          </label>
        )}
        {error && <div style={{ color: '#ff6b6b', margin: '8px 0' }}>{error}</div>}
        <div style={{ display: 'flex', gap: 8, justifyContent: 'flex-end', marginTop: 12 }}>
          <button onClick={onClose} disabled={busy}>Cancel</button>
          <button onClick={run} disabled={busy}
            style={{ background: '#4ecdc4', color: '#000', border: 'none', padding: '6px 14px',
              borderRadius: 4, cursor: 'pointer' }}>
            {busy ? 'Scoring…' : 'Score'}
          </button>
        </div>
      </div>
    </div>
  )
}
