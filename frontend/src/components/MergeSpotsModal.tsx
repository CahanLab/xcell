/** Merge Spots — putting a large cell back together from the spots it covers.
 *
 *  Preview first, commit second. `min_correlation` is the parameter that needs
 *  real data to set, so the default run is a dry one: same route, same code
 *  path, no dataset created, and enough numbers to tell whether the veto is
 *  earning its place.
 */

import { useCallback, useEffect, useState } from 'react'
import { useStore } from '../store'
import { appendDataset } from '../hooks/useData'
import { nextSlotKey } from '../lib/datasetSlots'
import GeneSubsetPicker, { useGeneSubset } from './GeneSubsetPicker'
import { formatSizeHistogram, vetoVerdict, type MergePreview } from '../lib/spotMerge'

const dark = {
  overlay: {
    position: 'fixed' as const, inset: 0, background: 'rgba(0,0,0,0.6)',
    display: 'flex', alignItems: 'center', justifyContent: 'center', zIndex: 1000,
  },
  panel: {
    background: '#16213e', border: '1px solid #0f3460', borderRadius: 8,
    width: 'min(560px, 94vw)', maxHeight: '90vh', display: 'flex',
    flexDirection: 'column' as const, boxShadow: '0 8px 32px rgba(0,0,0,0.5)',
  },
  header: {
    display: 'flex', alignItems: 'center', justifyContent: 'space-between',
    padding: '12px 16px', borderBottom: '1px solid #0f3460',
  },
  title: { margin: 0, fontSize: 15, color: '#e94560', fontWeight: 600 },
  close: {
    background: 'transparent', border: 'none', color: '#888',
    fontSize: 20, cursor: 'pointer', lineHeight: 1,
  },
  body: { padding: '14px 16px', overflowY: 'auto' as const },
  label: {
    fontSize: 10, textTransform: 'uppercase' as const, letterSpacing: 0.6,
    color: '#666', marginBottom: 6,
  },
  row: { display: 'flex', alignItems: 'center', gap: 8, marginBottom: 8, fontSize: 12, color: '#ccc' },
  input: {
    background: '#0f1625', border: '1px solid #0f3460', borderRadius: 4,
    color: '#ddd', fontSize: 12, padding: '5px 8px', outline: 'none', width: 90,
  },
  select: {
    background: '#0f1625', border: '1px solid #0f3460', borderRadius: 4,
    color: '#ddd', fontSize: 12, padding: '5px 8px', outline: 'none',
  },
  notice: { marginTop: 8, padding: '8px 10px', borderRadius: 4, fontSize: 11.5 },
  pre: {
    background: '#0a0f1a', borderRadius: 6, padding: '10px 12px',
    fontSize: 11.5, color: '#bbb', margin: 0, whiteSpace: 'pre-wrap' as const,
  },
  actions: {
    display: 'flex', gap: 8, padding: '12px 16px', borderTop: '1px solid #0f3460',
  },
  primary: {
    flex: 1, padding: '8px 12px', borderRadius: 4, border: '1px solid #4ecdc4',
    background: '#4ecdc4', color: '#000', fontSize: 12, fontWeight: 600, cursor: 'pointer',
  },
  ghost: {
    flex: 1, padding: '8px 12px', borderRadius: 4, border: '1px solid #0f3460',
    background: 'transparent', color: '#aaa', fontSize: 12, cursor: 'pointer',
  },
}

const num = (v: number) => v.toLocaleString()

export default function MergeSpotsModal() {
  const isOpen = useStore((s) => s.isMergeSpotsModalOpen)
  const setOpen = useStore((s) => s.setMergeSpotsModalOpen)
  const selected = useStore((s) => s.selectedCellIndices)
  const datasets = useStore((s) => s.datasets)
  const loadDatasetIntoSlot = useStore((s) => s.loadDatasetIntoSlot)
  const setActiveSlot = useStore((s) => s.setActiveSlot)

  const [maxDiameter, setMaxDiameter] = useState('40')
  const [minCorrelation, setMinCorrelation] = useState('0.15')
  const [maxSpots, setMaxSpots] = useState('')
  const [eligibility, setEligibility] = useState<'all' | 'min_counts' | 'quantile'>('all')
  const [minCounts, setMinCounts] = useState('500')
  const [quantile, setQuantile] = useState('0.5')

  const [busy, setBusy] = useState<'' | 'preview' | 'commit'>('')
  const [error, setError] = useState<string | null>(null)
  const [preview, setPreview] = useState<MergePreview | null>(null)

  const geneSubset = useGeneSubset()

  const close = useCallback(() => {
    setOpen(false)
    setError(null)
    setPreview(null)
    setBusy('')
  }, [setOpen])

  useEffect(() => {
    if (!isOpen) return
    const onKey = (e: KeyboardEvent) => { if (e.key === 'Escape') close() }
    window.addEventListener('keydown', onKey)
    return () => window.removeEventListener('keydown', onKey)
  }, [isOpen, close])

  // A parameter change invalidates the numbers on screen; showing stale ones
  // next to new inputs is how a threshold gets judged on the wrong evidence.
  useEffect(() => {
    setPreview(null)
  }, [maxDiameter, minCorrelation, maxSpots, eligibility, minCounts, quantile, geneSubset.spec])

  const body = useCallback((dryRun: boolean, slot?: string) => ({
    region_indices: selected,
    max_diameter_um: Number(maxDiameter) || 40,
    max_spots: maxSpots ? Number(maxSpots) : null,
    min_correlation: Number(minCorrelation),
    eligibility,
    min_counts: eligibility === 'min_counts' ? Number(minCounts) : null,
    quantile: eligibility === 'quantile' ? Number(quantile) : null,
    gene_subset: geneSubset.spec,
    dry_run: dryRun,
    slot: slot ?? null,
  }), [selected, maxDiameter, maxSpots, minCorrelation, eligibility, minCounts, quantile, geneSubset.spec])

  const post = useCallback(async (dryRun: boolean, slot?: string) => {
    const res = await fetch(appendDataset('/api/spots/merge'), {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(body(dryRun, slot)),
    })
    const data = await res.json()
    if (!res.ok) throw new Error(data.detail || `HTTP ${res.status}`)
    return data
  }, [body])

  const runPreview = useCallback(async () => {
    setBusy('preview'); setError(null)
    try {
      setPreview(await post(true))
    } catch (e) {
      setPreview(null)
      setError((e as Error).message)
    } finally {
      setBusy('')
    }
  }, [post])

  const commit = useCallback(async () => {
    setBusy('commit'); setError(null)
    try {
      const slot = nextSlotKey(Object.keys(datasets))
      const data = await post(false, slot)
      loadDatasetIntoSlot(slot, data)
      setActiveSlot(slot)
      close()
    } catch (e) {
      setError((e as Error).message)
    } finally {
      setBusy('')
    }
  }, [post, datasets, loadDatasetIntoSlot, setActiveSlot, close])

  if (!isOpen) return null

  const tooFew = selected.length < 2

  return (
    <div style={dark.overlay} onClick={close}>
      <div style={dark.panel} onClick={(e) => e.stopPropagation()}>
        <div style={dark.header}>
          <h2 style={dark.title}>Merge Spots</h2>
          <button style={dark.close} onClick={close}>&times;</button>
        </div>

        <div style={dark.body}>
          <div style={dark.label}>Region</div>
          <div style={{ ...dark.notice, background: tooFew ? 'rgba(233,162,59,0.14)' : 'rgba(78,205,196,0.12)', color: tooFew ? '#f0c987' : '#bfeae6' }}>
            {tooFew
              ? 'Lasso at least two spots first — the region is what says "these are large cells".'
              : `${num(selected.length)} spots selected.`}
          </div>

          <div style={{ ...dark.label, marginTop: 14 }}>Cell size</div>
          <div style={dark.row}>
            <span style={{ width: 120 }}>Max diameter (µm)</span>
            <input style={dark.input} value={maxDiameter}
                   onChange={(e) => setMaxDiameter(e.target.value)}
                   title="Spots stop merging once their footprint reaches this across. The one parameter with biological meaning." />
            <span style={{ width: 78, color: '#888' }}>Max spots</span>
            <input style={dark.input} value={maxSpots} placeholder="auto"
                   onChange={(e) => setMaxSpots(e.target.value)}
                   title="Blank derives it from the diameter and the measured spot pitch." />
          </div>

          <div style={{ ...dark.label, marginTop: 6 }}>Veto</div>
          <div style={dark.row}>
            <span style={{ width: 120 }}>Min correlation</span>
            <input style={dark.input} value={minCorrelation}
                   onChange={(e) => setMinCorrelation(e.target.value)}
                   title="Blocks a merge between spots whose profiles are clearly different. A floor, not a ranking." />
            <span style={{ fontSize: 11, color: '#666' }}>blocks clearly different neighbours</span>
          </div>
          <GeneSubsetPicker control={geneSubset} />
          <div style={{ fontSize: 10, color: '#666', marginTop: -8, marginBottom: 10 }}>
            Correlation is computed on these genes' expression, not a set score.
          </div>

          <div style={{ ...dark.label, marginTop: 6 }}>Eligibility</div>
          <div style={dark.row}>
            <select style={dark.select} value={eligibility}
                    onChange={(e) => setEligibility(e.target.value as typeof eligibility)}>
              <option value="all">Every spot in the region</option>
              <option value="min_counts">Only spots under a count threshold</option>
              <option value="quantile">Only the shallowest fraction</option>
            </select>
            {eligibility === 'min_counts' && (
              <input style={dark.input} value={minCounts}
                     onChange={(e) => setMinCounts(e.target.value)} title="Counts below this are eligible." />
            )}
            {eligibility === 'quantile' && (
              <input style={dark.input} value={quantile}
                     onChange={(e) => setQuantile(e.target.value)} title="Fraction of the region, shallowest first." />
            )}
          </div>

          {error && (
            <div style={{ ...dark.notice, background: 'rgba(233,69,96,0.15)', color: '#ff8fa3' }}>
              {error}
            </div>
          )}

          {preview && (
            <>
              <div style={{ ...dark.label, marginTop: 14 }}>Preview</div>
              <pre style={dark.pre}>
{`region          ${num(preview.n_region_spots)} spots
estimated pitch ${preview.pitch_um.toFixed(1)} µm  →  up to ${preview.max_spots} spots per merge
result          ${num(preview.n_region_spots)} → ${num(preview.n_merged_spots)} merged spots
merge sizes     ${formatSizeHistogram(preview.size_histogram)}
median counts   ${num(Math.round(preview.median_counts_before))} per spot before
veto blocked    ${num(preview.n_vetoed)} candidate pairs`}
              </pre>
              <div style={{ ...dark.notice, background: 'rgba(78,205,196,0.10)', color: '#bfeae6' }}>
                {vetoVerdict(preview.n_vetoed, preview.correlation_quantiles, Number(minCorrelation))}
              </div>
            </>
          )}
        </div>

        <div style={dark.actions}>
          <button style={{ ...dark.ghost, opacity: busy || tooFew ? 0.6 : 1 }}
                  onClick={runPreview} disabled={!!busy || tooFew}>
            {busy === 'preview' ? 'Previewing…' : 'Preview'}
          </button>
          <button style={{ ...dark.primary, opacity: busy || tooFew || !preview ? 0.5 : 1 }}
                  onClick={commit} disabled={!!busy || tooFew || !preview}
                  title={preview ? 'Create the merged dataset in a new tab' : 'Preview first'}>
            {busy === 'commit' ? 'Merging…' : 'Create merged dataset'}
          </button>
        </div>
      </div>
    </div>
  )
}
