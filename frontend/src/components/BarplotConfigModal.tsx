/**
 * BarplotConfigModal — choose the two .obs columns a stacked barplot compares,
 * and how the bars are ordered.
 *
 * Only categorical columns are offered. Grouping a continuous column would
 * mean binning it, which is a decision the user has to make somewhere they can
 * see it — the backend refuses it too.
 */

import { useEffect, useState } from 'react'
import { BarplotConfig, Schema } from '../store'

const DEFAULTS: BarplotConfig = {
  columnA: '', columnB: '',
  order: 'category', shareOf: null,
  normalize: true, minCells: 0, showValues: false,
}

const ORDERS: { value: BarplotConfig['order']; label: string; hint: string }[] = [
  { value: 'category', label: "The dataset's order", hint: 'As the column stores its categories' },
  { value: 'alphabetical', label: 'By name', hint: 'Numeric-aware, so 9 comes before 10' },
  { value: 'total', label: 'By size', hint: 'Largest group first' },
  { value: 'share', label: 'By composition', hint: 'Most of a chosen category first' },
]

export default function BarplotConfigModal({
  schema,
  initial,
  categoriesOfB = [],
  onClose,
  onApply,
}: {
  schema: Schema | null
  initial: BarplotConfig | null
  categoriesOfB?: string[]
  onClose: () => void
  onApply: (config: BarplotConfig) => void
}) {
  const [draft, setDraft] = useState<BarplotConfig>(initial ?? DEFAULTS)

  useEffect(() => {
    const onKey = (e: KeyboardEvent) => { if (e.key === 'Escape') onClose() }
    window.addEventListener('keydown', onKey)
    return () => window.removeEventListener('keydown', onKey)
  }, [onClose])

  // Categorical columns only — a continuous one has no categories to stack.
  const columns = (schema?.obs_columns ?? []).filter((c) => {
    const dtype = schema?.obs_dtypes?.[c]
    return dtype === 'category' || dtype === 'string' || dtype === 'bool'
  })

  const set = (patch: Partial<BarplotConfig>) => setDraft((d) => ({ ...d, ...patch }))
  const sameColumn = !!draft.columnA && draft.columnA === draft.columnB
  const ready = !!draft.columnA && !!draft.columnB && !sameColumn

  return (
    <div style={styles.backdrop} onClick={onClose}>
      <div style={styles.card} onClick={(e) => e.stopPropagation()}>
        <div style={styles.header}>Stacked barplot</div>

        <div style={styles.row}>
          <label style={styles.label}>One bar per</label>
          <select style={styles.select} value={draft.columnA}
                  onChange={(e) => set({ columnA: e.target.value })}>
            <option value="">Choose a column…</option>
            {columns.map((c) => <option key={c} value={c}>{c}</option>)}
          </select>
        </div>

        <div style={styles.row}>
          <label style={styles.label}>Split each bar by</label>
          <select style={styles.select} value={draft.columnB}
                  onChange={(e) => set({ columnB: e.target.value, shareOf: null })}>
            <option value="">Choose a column…</option>
            {columns.map((c) => <option key={c} value={c}>{c}</option>)}
          </select>
        </div>

        {sameColumn && (
          <div style={styles.warn}>
            Splitting a column by itself gives one block per bar. Choose two
            different columns.
          </div>
        )}

        {columns.length === 0 && (
          <div style={styles.warn}>
            This dataset has no categorical .obs columns to group by. Cluster it,
            or bin a continuous column, first.
          </div>
        )}

        <div style={styles.divider} />

        <div style={styles.row}>
          <label style={styles.label}>Bar height</label>
          <div style={styles.segmented}>
            {[
              { v: true, t: 'Proportion' },
              { v: false, t: 'Cell count' },
            ].map(({ v, t }) => (
              <button key={t}
                      style={{ ...styles.segment, ...(draft.normalize === v ? styles.segmentOn : {}) }}
                      onClick={() => set({ normalize: v })}>
                {t}
              </button>
            ))}
          </div>
        </div>
        <div style={styles.hint}>
          Proportion makes every bar the same height, so composition is
          comparable across groups of very different sizes. Cell count keeps the
          sizes visible instead.
        </div>

        <div style={styles.row}>
          <label style={styles.label}>Order bars</label>
          <select style={styles.select} value={draft.order}
                  onChange={(e) => set({ order: e.target.value as BarplotConfig['order'] })}>
            {ORDERS.map((o) => <option key={o.value} value={o.value}>{o.label}</option>)}
          </select>
        </div>
        <div style={styles.hint}>{ORDERS.find((o) => o.value === draft.order)?.hint}</div>

        {draft.order === 'share' && (
          <div style={styles.row}>
            <label style={styles.label}>Most of</label>
            <select style={styles.select} value={draft.shareOf ?? ''}
                    onChange={(e) => set({ shareOf: e.target.value || null })}>
              <option value="">Choose a category…</option>
              {categoriesOfB.map((c) => <option key={c} value={c}>{c}</option>)}
            </select>
          </div>
        )}
        {draft.order === 'share' && categoriesOfB.length === 0 && (
          <div style={styles.hint}>
            Apply the two columns first — the categories to sort by come from the
            data.
          </div>
        )}

        <div style={styles.row}>
          <label style={styles.label}>Hide bars under</label>
          <input type="number" min={0} style={styles.number} value={draft.minCells}
                 onChange={(e) => set({ minCells: Math.max(0, Number(e.target.value) || 0) })} />
          <span style={styles.unit}>cells</span>
        </div>
        <div style={styles.hint}>
          A group of three cells has a composition, but not one worth reading.
        </div>

        <div style={styles.row}>
          <label style={styles.label}>
            <input type="checkbox" checked={draft.showValues}
                   onChange={(e) => set({ showValues: e.target.checked })} />
            {' '}Label each bar with its cell count
          </label>
        </div>

        <div style={styles.actions}>
          <button style={styles.btn} onClick={onClose}>Cancel</button>
          <button style={{ ...styles.btn, ...styles.primary, ...(ready ? {} : styles.disabled) }}
                  disabled={!ready}
                  onClick={() => onApply(draft)}>
            Plot
          </button>
        </div>
      </div>
    </div>
  )
}

const styles = {
  backdrop: {
    position: 'fixed' as const, inset: 0, backgroundColor: 'rgba(0,0,0,0.55)',
    display: 'flex', alignItems: 'center', justifyContent: 'center', zIndex: 200,
  },
  card: {
    width: '440px', maxHeight: '84vh', overflowY: 'auto' as const,
    padding: '18px 20px', backgroundColor: '#16213e',
    border: '1px solid #0f3460', borderRadius: '8px',
  },
  header: { fontSize: '14px', color: '#eee', marginBottom: '14px' },
  row: { display: 'flex', alignItems: 'center', gap: '10px', marginBottom: '6px' },
  label: { flex: '0 0 130px', fontSize: '12px', color: '#aaa' },
  select: {
    flex: 1, padding: '5px 8px', fontSize: '12px', color: '#eee',
    backgroundColor: '#0f3460', border: '1px solid #1a1a2e', borderRadius: '4px',
  },
  number: {
    width: '80px', padding: '5px 8px', fontSize: '12px', color: '#eee',
    backgroundColor: '#0f3460', border: '1px solid #1a1a2e', borderRadius: '4px',
  },
  unit: { fontSize: '11px', color: '#888' },
  hint: { margin: '0 0 12px 140px', fontSize: '10px', color: '#777', lineHeight: 1.5 },
  warn: {
    padding: '8px 10px', margin: '4px 0 10px', fontSize: '11px',
    color: '#e9a23b', backgroundColor: '#0f1625',
    border: '1px solid #0f3460', borderRadius: '4px', lineHeight: 1.5,
  },
  divider: { height: '1px', backgroundColor: '#0f3460', margin: '10px 0 14px' },
  segmented: { display: 'flex', gap: '4px' },
  segment: {
    padding: '5px 10px', fontSize: '11px', color: '#aaa',
    backgroundColor: '#0f3460', border: '1px solid #1a1a2e',
    borderRadius: '4px', cursor: 'pointer',
  },
  segmentOn: { backgroundColor: '#4ecdc4', color: '#000' },
  actions: { display: 'flex', justifyContent: 'flex-end', gap: '8px', marginTop: '16px' },
  btn: {
    padding: '6px 14px', fontSize: '12px', color: '#aaa',
    backgroundColor: '#0f3460', border: '1px solid #1a1a2e',
    borderRadius: '4px', cursor: 'pointer',
  },
  primary: { backgroundColor: '#4ecdc4', color: '#000' },
  disabled: { opacity: 0.45, cursor: 'not-allowed' },
}
