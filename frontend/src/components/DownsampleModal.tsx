/** Downsample — thin a group of cells to a random subset and select it.
 *
 *  The result is a selection, nothing more: from there the same Mask / Delete /
 *  Invert / Set Active buttons apply as to any lasso. The usual move is to
 *  lasso a dominant cluster, thin it, Invert, then Delete or Mask — so the pool
 *  defaults to the current selection, and falls back to the active cells and
 *  then the whole dataset when there is none.
 */

import { useCallback, useEffect, useMemo, useState } from 'react'
import { useStore } from '../store'
import {
  availablePools, drawSeed, parseSeed, poolIndices, resolveTarget,
  sampleWithoutReplacement, seededRng, type Pool, type TargetMode,
} from '../lib/downsample'

const dark = {
  overlay: {
    position: 'fixed' as const, inset: 0, background: 'rgba(0,0,0,0.6)',
    display: 'flex', alignItems: 'center', justifyContent: 'center', zIndex: 1000,
  },
  panel: {
    background: '#16213e', border: '1px solid #0f3460', borderRadius: 8,
    width: 'min(460px, 94vw)', maxHeight: '90vh', display: 'flex',
    flexDirection: 'column' as const, boxShadow: '0 8px 32px rgba(0,0,0,0.5)',
  },
  header: {
    display: 'flex', alignItems: 'center', justifyContent: 'space-between',
    padding: '12px 16px', borderBottom: '1px solid #0f3460',
  },
  title: { margin: 0, fontSize: 15, color: '#e94560', fontWeight: 600 },
  close: { background: 'transparent', border: 'none', color: '#888', fontSize: 20, cursor: 'pointer', lineHeight: 1 },
  body: { padding: '14px 16px', overflowY: 'auto' as const },
  label: { fontSize: 10, textTransform: 'uppercase' as const, letterSpacing: 0.6, color: '#666', marginBottom: 6 },
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
  hint: { fontSize: 10.5, color: '#666', marginTop: 4, marginBottom: 10, lineHeight: 1.4 },
  actions: { display: 'flex', gap: 8, padding: '12px 16px', borderTop: '1px solid #0f3460' },
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

const POOL_LABEL: Record<Pool, string> = {
  selection: 'Selected cells',
  active: 'Active (unmasked) cells',
  all: 'All cells',
}

export default function DownsampleModal() {
  const isOpen = useStore((s) => s.isDownsampleModalOpen)
  const setOpen = useStore((s) => s.setDownsampleModalOpen)
  const selected = useStore((s) => s.selectedCellIndices)
  const activeCellMask = useStore((s) => s.activeCellMask)
  const nCells = useStore((s) => s.schema?.n_cells ?? 0)
  const setSelectedCellIndices = useStore((s) => s.setSelectedCellIndices)

  const pools = useMemo(() => availablePools(selected.length, !!activeCellMask), [selected.length, activeCellMask])
  const [pool, setPool] = useState<Pool>('all')
  const [mode, setMode] = useState<TargetMode>('count')
  const [target, setTarget] = useState('')
  const [seed, setSeed] = useState('')
  const [lastSeed, setLastSeed] = useState<number | null>(null)

  // Re-pick the default pool each time the modal opens: the selection that
  // existed last time is not the one that exists now.
  useEffect(() => {
    if (isOpen) { setPool(pools[0]); setLastSeed(null) }
  }, [isOpen])  // eslint-disable-line react-hooks/exhaustive-deps

  // If the selection is cleared while open, the pool must not keep naming it.
  useEffect(() => {
    if (!pools.includes(pool)) setPool(pools[0])
  }, [pools, pool])

  const poolSize = useMemo(() => {
    if (pool === 'selection') return selected.length
    if (pool === 'active' && activeCellMask) return activeCellMask.reduce((n, b) => n + (b ? 1 : 0), 0)
    return nCells
  }, [pool, selected.length, activeCellMask, nCells])

  const resolved = resolveTarget(mode, target, poolSize)
  const seedInvalid = seed.trim() !== '' && parseSeed(seed) === null

  const close = useCallback(() => setOpen(false), [setOpen])

  useEffect(() => {
    if (!isOpen) return
    const onKey = (e: KeyboardEvent) => { if (e.key === 'Escape') close() }
    window.addEventListener('keydown', onKey)
    return () => window.removeEventListener('keydown', onKey)
  }, [isOpen, close])

  const run = useCallback(() => {
    if (resolved.n === null || seedInvalid) return
    const used = parseSeed(seed) ?? drawSeed()
    const indices = poolIndices(pool, selected, activeCellMask, nCells)
    setSelectedCellIndices(sampleWithoutReplacement(indices, resolved.n, seededRng(used)))
    setLastSeed(used)
    close()
  }, [resolved.n, seedInvalid, seed, pool, selected, activeCellMask, nCells, setSelectedCellIndices, close])

  if (!isOpen) return null

  const canRun = resolved.n !== null && !seedInvalid

  return (
    <div style={dark.overlay} onClick={close}>
      <div style={dark.panel} onClick={(e) => e.stopPropagation()}>
        <div style={dark.header}>
          <h2 style={dark.title}>Downsample Cells</h2>
          <button style={dark.close} onClick={close}>&times;</button>
        </div>

        <div style={dark.body}>
          <div style={dark.label}>Sample from</div>
          <div style={dark.row}>
            <select style={{ ...dark.select, flex: 1 }} value={pool}
                    onChange={(e) => setPool(e.target.value as Pool)}>
              {pools.map((p) => (
                <option key={p} value={p}>{POOL_LABEL[p]} ({num(
                  p === 'selection' ? selected.length
                    : p === 'active' && activeCellMask ? activeCellMask.filter(Boolean).length
                    : nCells)})</option>
              ))}
            </select>
          </div>
          <div style={dark.hint}>
            {pools.includes('selection')
              ? 'Thinning the selection leaves every other cell alone — lasso a dominant cluster, downsample, then Invert and Delete or Mask.'
              : 'Nothing is selected, so this samples the whole dataset. Lasso a cluster first to thin only that cluster.'}
          </div>

          <div style={dark.label}>Keep</div>
          <div style={dark.row}>
            <input style={dark.input} value={target} autoFocus
                   onChange={(e) => setTarget(e.target.value)}
                   onKeyDown={(e) => { if (e.key === 'Enter' && canRun) run() }}
                   placeholder={mode === 'count' ? 'e.g. 500' : 'e.g. 25'} />
            <select style={dark.select} value={mode} onChange={(e) => setMode(e.target.value as TargetMode)}>
              <option value="count">cells</option>
              <option value="percent">% of pool</option>
            </select>
            <span style={{ fontSize: 11, color: resolved.n === null ? '#888' : '#bfeae6' }}>
              {resolved.n === null
                ? (target === '' ? '' : resolved.reason)
                : `→ ${num(resolved.n)} of ${num(poolSize)} cells`}
            </span>
          </div>

          <div style={{ ...dark.label, marginTop: 6 }}>Seed</div>
          <div style={dark.row}>
            <input style={{ ...dark.input, border: `1px solid ${seedInvalid ? '#e94560' : '#0f3460'}` }}
                   value={seed} placeholder="random"
                   onChange={(e) => setSeed(e.target.value)}
                   title="Integer. Blank draws one; the seed used is reported so the sample can be reproduced." />
            <span style={{ fontSize: 11, color: '#666' }}>
              {seedInvalid ? 'Seed must be a whole number' : 'same seed, same cells'}
            </span>
          </div>

          {lastSeed !== null && (
            <div style={{ ...dark.notice, background: 'rgba(78,205,196,0.10)', color: '#bfeae6' }}>
              Last sample used seed {lastSeed}.
            </div>
          )}
        </div>

        <div style={dark.actions}>
          <button style={dark.ghost} onClick={close}>Cancel</button>
          <button style={{ ...dark.primary, opacity: canRun ? 1 : 0.5 }}
                  onClick={run} disabled={!canRun}
                  title={canRun ? 'Replace the selection with the sampled cells' : resolved.reason ?? ''}>
            {resolved.n === null ? 'Select sample' : `Select ${num(resolved.n)} cells`}
          </button>
        </div>
      </div>
    </div>
  )
}
