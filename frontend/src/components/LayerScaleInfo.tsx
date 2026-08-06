import { useEffect, useRef, useState } from 'react'

/**
 * "Is this matrix raw counts or already normalized?"
 *
 * The backend (xcell/layer_scale.py) classifies every readable matrix — .X and
 * each adata.layers key — from its own values, and ships the evidence along
 * with the verdict. This module renders that as an unobtrusive badge: a short
 * label inline wherever a layer is named, and a click-to-open popover with the
 * reasoning for anyone who wants to check the call rather than trust it.
 */

export type ScaleVerdict =
  | 'raw_counts'
  | 'normalized_linear'
  | 'log_normalized'
  | 'log_transformed'
  | 'z_scored'
  | 'binary'
  | 'empty'
  | 'unknown'

export interface LayerScale {
  verdict: ScaleVerdict
  /** Short human name, e.g. "log-normalized". Safe to inline in a dropdown. */
  label: string
  /** One line on what the scale means for downstream analysis. */
  note: string
  confidence: 'high' | 'medium' | 'low'
  /** Why the classifier landed here, in plain language. */
  reasons: string[]
  /** Facts recorded by scanpy/xcell, as opposed to inferred from values. */
  provenance: string[]
  stats: {
    n_cells_sampled: number
    n_genes: number
    min: number
    max: number
    integer_valued: boolean
    has_negative: boolean
    nonzero_frac: number
    nonzero_mean: number
    row_sum_median: number
    row_sum_cv: number | null
    expm1_row_sum_median: number
    expm1_row_sum_cv: number | null
  }
}

export interface LayerInfo {
  name: string
  shape?: number[]
  nnz?: number
  density: number
  sparse?: boolean
  is_default?: boolean
  scale?: LayerScale
}

/** Verdict → accent colour. Red is reserved for scales that break things. */
export const VERDICT_COLOR: Record<ScaleVerdict, string> = {
  raw_counts: '#4ecdc4',
  normalized_linear: '#e9a23b',
  log_normalized: '#7ec9f0',
  log_transformed: '#9b8ce0',
  z_scored: '#e94560',
  binary: '#9aa5b1',
  empty: '#888',
  unknown: '#888',
}

/** Label for a layer inside a <option>, e.g. `.X (default) — log-normalized`. */
export function layerOptionLabel(L: LayerInfo): string {
  const base = L.name === 'X' ? '.X (default)' : L.name
  return L.scale ? `${base} — ${L.scale.label}` : base
}

const fmt = (v: number): string => {
  if (!isFinite(v)) return '—'
  if (v === 0) return '0'
  const a = Math.abs(v)
  if (a >= 1e5 || a < 1e-3) return v.toExponential(2)
  if (a >= 100) return v.toLocaleString(undefined, { maximumFractionDigits: 0 })
  return v.toFixed(a >= 1 ? 2 : 3)
}

function Row({ k, v }: { k: string; v: string }) {
  return (
    <div style={{ display: 'flex', justifyContent: 'space-between', gap: 12 }}>
      <span style={{ color: '#888' }}>{k}</span>
      <span style={{ color: '#ddd', fontVariantNumeric: 'tabular-nums' }}>{v}</span>
    </div>
  )
}

/**
 * A one-character affordance that opens the full scale report for `layer`.
 *
 * Deliberately tiny: the answer most users need is already in the label next
 * to it, and this is only for when they want to see why.
 */
export function LayerScaleBadge({
  layer,
  align = 'left',
}: {
  layer: LayerInfo | undefined
  /** Which edge of the button the popover hangs off. */
  align?: 'left' | 'right'
}) {
  const [open, setOpen] = useState(false)
  const wrap = useRef<HTMLSpanElement>(null)

  // Close on outside click / Escape, like the app's other transient popovers.
  useEffect(() => {
    if (!open) return
    const onDown = (e: MouseEvent) => {
      if (wrap.current && !wrap.current.contains(e.target as Node)) setOpen(false)
    }
    const onKey = (e: KeyboardEvent) => {
      if (e.key === 'Escape') setOpen(false)
    }
    window.addEventListener('mousedown', onDown)
    window.addEventListener('keydown', onKey)
    return () => {
      window.removeEventListener('mousedown', onDown)
      window.removeEventListener('keydown', onKey)
    }
  }, [open])

  const scale = layer?.scale
  if (!layer || !scale) return null

  const color = VERDICT_COLOR[scale.verdict] ?? '#888'
  const s = scale.stats

  return (
    <span ref={wrap} style={{ position: 'relative', display: 'inline-flex' }}>
      <button
        onClick={() => setOpen((o) => !o)}
        style={{
          width: 16,
          height: 16,
          lineHeight: '14px',
          padding: 0,
          fontSize: 10,
          fontWeight: 700,
          borderRadius: '50%',
          border: `1px solid ${color}`,
          background: open ? color : 'transparent',
          color: open ? '#16213e' : color,
          cursor: 'pointer',
          flexShrink: 0,
        }}
        title={`${layer.name === 'X' ? '.X' : layer.name}: ${scale.label} (${scale.confidence} confidence) — click for the evidence`}
      >
        i
      </button>

      {open && (
        <div
          style={{
            position: 'absolute',
            bottom: 22,
            [align]: 0,
            width: 320,
            maxHeight: '50vh',
            overflowY: 'auto',
            padding: 12,
            background: '#16213e',
            border: '1px solid #0f3460',
            borderRadius: 6,
            boxShadow: '0 4px 16px rgba(0,0,0,0.5)',
            fontSize: 11,
            lineHeight: 1.5,
            color: '#ccc',
            zIndex: 60,
            textAlign: 'left',
          }}
        >
          <div style={{ display: 'flex', alignItems: 'baseline', gap: 6, marginBottom: 6 }}>
            <code style={{ color: '#eee', fontSize: 12 }}>
              {layer.name === 'X' ? 'adata.X' : `adata.layers['${layer.name}']`}
            </code>
          </div>

          <div style={{ display: 'flex', alignItems: 'center', gap: 6, marginBottom: 8 }}>
            <span
              style={{
                padding: '2px 7px',
                borderRadius: 10,
                background: `${color}22`,
                border: `1px solid ${color}`,
                color,
                fontWeight: 600,
                fontSize: 11,
              }}
            >
              {scale.label}
            </span>
            <span style={{ color: '#888', fontSize: 10 }}>{scale.confidence} confidence</span>
          </div>

          <div style={{ color: '#bbb', marginBottom: 10 }}>{scale.note}</div>

          {scale.provenance.length > 0 && (
            <>
              <div style={sectionTitle}>Recorded in the file</div>
              <ul style={listStyle}>
                {scale.provenance.map((p, i) => (
                  <li key={i} style={{ color: '#9be7d8' }}>{p}</li>
                ))}
              </ul>
            </>
          )}

          <div style={sectionTitle}>Inferred from the values</div>
          <ul style={listStyle}>
            {scale.reasons.map((r, i) => (
              <li key={i}>{r}</li>
            ))}
          </ul>

          <div style={sectionTitle}>Evidence</div>
          <div style={{ display: 'flex', flexDirection: 'column', gap: 2 }}>
            <Row k="range" v={`${fmt(s.min)} … ${fmt(s.max)}`} />
            <Row k="integer-valued" v={s.integer_valued ? 'yes' : 'no'} />
            <Row k="has negatives" v={s.has_negative ? 'yes' : 'no'} />
            <Row k="non-zero" v={`${(s.nonzero_frac * 100).toFixed(1)}%`} />
            <Row k="median row sum" v={fmt(s.row_sum_median)} />
            <Row
              k="row-sum CV"
              v={s.row_sum_cv == null ? 'n/a' : fmt(s.row_sum_cv)}
            />
            {/* Only meaningful once the data is off the integer scale — on
                raw counts the expm1 statistic is arithmetic noise. */}
            {s.expm1_row_sum_cv != null && !s.integer_valued && (
              <Row k="expm1 row-sum CV" v={fmt(s.expm1_row_sum_cv)} />
            )}
            <Row
              k="sampled"
              v={`${s.n_cells_sampled.toLocaleString()} cells × ${s.n_genes.toLocaleString()} genes`}
            />
          </div>
        </div>
      )}
    </span>
  )
}

const sectionTitle: React.CSSProperties = {
  color: '#888',
  fontSize: 10,
  textTransform: 'uppercase',
  letterSpacing: 0.5,
  margin: '8px 0 3px',
}

const listStyle: React.CSSProperties = {
  margin: '0 0 2px',
  paddingLeft: 16,
}
