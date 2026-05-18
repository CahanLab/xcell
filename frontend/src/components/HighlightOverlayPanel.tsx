import { useMemo, useState } from 'react'
import { GeneSet, HighlightLayer, HighlightSource, HighlightThresholdMode } from '../store'
import { ObsSummary } from '../hooks/useData'
import { HistogramChart, computeHistogram, defaultThresholds } from '../utils/histogram'

interface Props {
  highlightLayers: HighlightLayer[]
  allGeneSets: GeneSet[]
  obsSummaries: ObsSummary[]
  selectedCellIndices: number[]
  addGeneSetHighlight: (
    genes: string[],
    label: string,
    opts: { color: string; intensity: number; thresholdMode?: HighlightThresholdMode }
  ) => Promise<string | null>
  addCellSetHighlight: (
    indices: number[],
    label: string,
    opts: { color: string; intensity: number }
  ) => string | null
  removeHighlightLayer: (id: string) => void
  updateHighlightLayer: (
    id: string,
    patch: Partial<Omit<HighlightLayer, 'id' | 'source'>> & { source?: Partial<HighlightSource> }
  ) => void
  clearHighlightOverlay: () => void
}

const HIGHLIGHT_PALETTE = ['#22c55e', '#06b6d4', '#f59e0b', '#ec4899', '#a855f7', '#84cc16', '#ef4444', '#0ea5e9']

function pickNextColor(n: number): string {
  return HIGHLIGHT_PALETTE[n % HIGHLIGHT_PALETTE.length]
}

const labelStyle = { fontSize: '11px', color: '#888' } as const

export default function HighlightOverlayPanel({
  highlightLayers,
  allGeneSets,
  obsSummaries,
  selectedCellIndices,
  addGeneSetHighlight,
  addCellSetHighlight,
  removeHighlightLayer,
  updateHighlightLayer,
  clearHighlightOverlay,
}: Props) {
  // Picker UI state — which add-source picker is currently open, plus
  // per-picker form fields. All collapse back to null after Apply.
  const [picker, setPicker] = useState<null | 'geneset' | 'selection' | 'category'>(null)
  const [pickGeneSetId, setPickGeneSetId] = useState<string>('')
  const [pickColumn, setPickColumn] = useState<string>('')
  const [pickValue, setPickValue] = useState<string>('')
  const [adding, setAdding] = useState(false)
  // Whole-section collapse — defaults to expanded. Local state, not persisted.
  const [expanded, setExpanded] = useState<boolean>(true)

  const usableGeneSets = useMemo(
    () => allGeneSets.filter((gs) => gs.genes.length > 0),
    [allGeneSets]
  )
  const categoricalColumns = useMemo(
    () => obsSummaries.filter((s) => (s.dtype === 'category' || s.dtype === 'string') && s.categories && s.categories.length > 0),
    [obsSummaries]
  )
  const categoryValuesFor = (column: string) => {
    const col = categoricalColumns.find((c) => c.name === column)
    return col?.categories ?? []
  }

  // Show the section if either there's a usable source OR there are already
  // layers (so the user can still clear them after gene sets get filtered out).
  const hasUsableSources = usableGeneSets.length > 0 || categoricalColumns.length > 0
  if (!hasUsableSources && highlightLayers.length === 0) return null

  const handleAddGeneSet = async () => {
    const gs = usableGeneSets.find((s) => s.id === pickGeneSetId)
    if (!gs) return
    setAdding(true)
    try {
      await addGeneSetHighlight(gs.genes, gs.name, {
        color: pickNextColor(highlightLayers.length),
        intensity: 0.85,
        thresholdMode: 'above',
      })
      setPicker(null)
      setPickGeneSetId('')
    } finally {
      setAdding(false)
    }
  }

  const handleAddSelection = () => {
    if (selectedCellIndices.length === 0) return
    const label = `Selection (${selectedCellIndices.length.toLocaleString()} cells)`
    addCellSetHighlight([...selectedCellIndices], label, {
      color: pickNextColor(highlightLayers.length),
      intensity: 0.85,
    })
    setPicker(null)
  }

  const handleAddCategory = async () => {
    if (!pickColumn || !pickValue) return
    setAdding(true)
    try {
      // Fetch the obs column to get per-cell category assignments
      const url = `/api/obs/${encodeURIComponent(pickColumn)}`
      const resp = await fetch(url)
      if (!resp.ok) throw new Error('Failed to fetch obs column')
      const data = await resp.json()
      const indices: number[] = []
      const cats: string[] = data.categories ?? []
      const catIdx = cats.indexOf(pickValue)
      if (data.dtype === 'category' && catIdx >= 0) {
        for (let i = 0; i < data.values.length; i++) {
          if (data.values[i] === catIdx) indices.push(i)
        }
      } else {
        for (let i = 0; i < data.values.length; i++) {
          if (data.values[i] === pickValue) indices.push(i)
        }
      }
      if (indices.length === 0) return
      addCellSetHighlight(indices, `${pickColumn}: ${pickValue}`, {
        color: pickNextColor(highlightLayers.length),
        intensity: 0.85,
      })
      setPicker(null)
      setPickColumn('')
      setPickValue('')
    } finally {
      setAdding(false)
    }
  }

  return (
    <div style={{ marginBottom: '10px' }}>
      <div
        style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', marginBottom: '4px', cursor: 'pointer', userSelect: 'none' }}
        onClick={() => setExpanded((v) => !v)}
        title={expanded ? 'Collapse Highlight panel' : 'Expand Highlight panel'}
      >
        <span style={{ display: 'flex', alignItems: 'center', gap: '6px' }}>
          <span style={{ fontSize: '10px', color: '#888', width: '10px' }}>{expanded ? '▼' : '▶'}</span>
          <span style={{ fontSize: '12px', fontWeight: 600, color: '#aaa', textTransform: 'uppercase' }}>
            Highlight (overlay)
          </span>
          {highlightLayers.length > 0 && (
            <span style={{ fontSize: '10px', color: '#888' }}>· {highlightLayers.length} active</span>
          )}
        </span>
        {expanded && highlightLayers.length > 0 && (
          <button
            onClick={(e) => { e.stopPropagation(); clearHighlightOverlay() }}
            style={{ padding: '2px 8px', fontSize: '11px', backgroundColor: 'transparent', color: '#888', border: '1px solid #444', borderRadius: '3px', cursor: 'pointer' }}
            title="Remove all highlight layers"
          >
            Clear all
          </button>
        )}
      </div>

      {expanded && (
      <div style={{ backgroundColor: '#0f3460', borderRadius: '4px', padding: '8px' }}>
        {highlightLayers.length === 0 && (
          <div style={{ fontSize: '10px', color: '#888', padding: '4px 6px 8px' }}>
            No highlights. Add gene-set, selection, or category-based layers below.
          </div>
        )}

        {highlightLayers.map((layer) => (
          <LayerRow
            key={layer.id}
            layer={layer}
            onRemove={() => removeHighlightLayer(layer.id)}
            onUpdate={(patch) => updateHighlightLayer(layer.id, patch)}
          />
        ))}

        {/* Add highlight chooser */}
        {picker === null && hasUsableSources && (
          <div style={{ display: 'flex', gap: '4px', flexWrap: 'wrap', marginTop: highlightLayers.length > 0 ? '8px' : 0 }}>
            {usableGeneSets.length > 0 && (
              <button
                onClick={() => setPicker('geneset')}
                style={pickerButtonStyle}
                title="Highlight cells above a gene-set expression threshold"
              >
                + Gene set
              </button>
            )}
            <button
              onClick={() => setPicker('selection')}
              disabled={selectedCellIndices.length === 0}
              style={{ ...pickerButtonStyle, opacity: selectedCellIndices.length === 0 ? 0.4 : 1, cursor: selectedCellIndices.length === 0 ? 'not-allowed' : 'pointer' }}
              title={selectedCellIndices.length === 0 ? 'No cells selected' : `Snapshot ${selectedCellIndices.length} selected cells as a highlight layer`}
            >
              + Selection {selectedCellIndices.length > 0 ? `(${selectedCellIndices.length})` : ''}
            </button>
            {categoricalColumns.length > 0 && (
              <button onClick={() => setPicker('category')} style={pickerButtonStyle} title="Highlight all cells in a category">
                + Category…
              </button>
            )}
          </div>
        )}

        {picker === 'geneset' && (
          <div style={pickerFormStyle}>
            <label style={labelStyle}>Gene set</label>
            <select
              value={pickGeneSetId}
              onChange={(e) => setPickGeneSetId(e.target.value)}
              style={inputStyle}
            >
              <option value="">Select gene set...</option>
              {usableGeneSets.map((gs) => (
                <option key={gs.id} value={gs.id}>{gs.name} ({gs.genes.length} genes)</option>
              ))}
            </select>
            <div style={{ display: 'flex', gap: '6px', marginTop: '6px' }}>
              <button onClick={handleAddGeneSet} disabled={!pickGeneSetId || adding} style={applyButtonStyle(!!pickGeneSetId && !adding)}>
                {adding ? 'Adding…' : 'Add'}
              </button>
              <button onClick={() => { setPicker(null); setPickGeneSetId('') }} style={cancelButtonStyle}>Cancel</button>
            </div>
          </div>
        )}

        {picker === 'selection' && (
          <div style={pickerFormStyle}>
            <div style={{ fontSize: '11px', color: '#ccc' }}>
              Snapshot the current selection ({selectedCellIndices.length.toLocaleString()} cells) as a highlight layer.
              Later selection changes will not update this layer.
            </div>
            <div style={{ display: 'flex', gap: '6px', marginTop: '6px' }}>
              <button onClick={handleAddSelection} disabled={selectedCellIndices.length === 0} style={applyButtonStyle(selectedCellIndices.length > 0)}>
                Add
              </button>
              <button onClick={() => setPicker(null)} style={cancelButtonStyle}>Cancel</button>
            </div>
          </div>
        )}

        {picker === 'category' && (
          <div style={pickerFormStyle}>
            <label style={labelStyle}>Column</label>
            <select
              value={pickColumn}
              onChange={(e) => { setPickColumn(e.target.value); setPickValue('') }}
              style={inputStyle}
            >
              <option value="">Select column...</option>
              {categoricalColumns.map((c) => (
                <option key={c.name} value={c.name}>{c.name}</option>
              ))}
            </select>
            {pickColumn && (
              <>
                <label style={{ ...labelStyle, marginTop: '6px', display: 'block' }}>Value</label>
                <select
                  value={pickValue}
                  onChange={(e) => setPickValue(e.target.value)}
                  style={inputStyle}
                >
                  <option value="">Select value...</option>
                  {categoryValuesFor(pickColumn).map((v) => (
                    <option key={v.value} value={v.value}>{v.value} ({v.count.toLocaleString()})</option>
                  ))}
                </select>
              </>
            )}
            <div style={{ display: 'flex', gap: '6px', marginTop: '6px' }}>
              <button onClick={handleAddCategory} disabled={!pickColumn || !pickValue || adding} style={applyButtonStyle(!!pickColumn && !!pickValue && !adding)}>
                {adding ? 'Adding…' : 'Add'}
              </button>
              <button onClick={() => { setPicker(null); setPickColumn(''); setPickValue('') }} style={cancelButtonStyle}>Cancel</button>
            </div>
          </div>
        )}
      </div>
      )}
    </div>
  )
}

function LayerRow({
  layer,
  onRemove,
  onUpdate,
}: {
  layer: HighlightLayer
  onRemove: () => void
  onUpdate: (patch: Partial<Omit<HighlightLayer, 'id' | 'source'>> & { source?: Partial<HighlightSource> }) => void
}) {
  const src = layer.source
  // Compute histogram for geneset layers (memoized on values identity).
  const histogram = useMemo(() => {
    if (src.kind !== 'geneset') return null
    return computeHistogram(src.values)
  }, [src])

  const handleModeChange = (mode: HighlightThresholdMode) => {
    if (src.kind !== 'geneset' || !histogram) return
    const d = defaultThresholds(src.values, mode, histogram.min, histogram.max)
    onUpdate({ source: { thresholdMode: mode, lo: d.lo, hi: d.hi } })
  }

  const isGeneset = src.kind === 'geneset'
  const hasHistogram = isGeneset && histogram && !histogram.zeroVariance

  return (
    <div style={layerRowStyle}>
      {/* Row 1 — color | label | (mode dropdown if geneset) | × */}
      <div style={{ display: 'flex', alignItems: 'center', gap: '6px' }}>
        <input
          type="color"
          value={layer.color}
          onChange={(e) => onUpdate({ color: e.target.value })}
          style={{ width: '20px', height: '18px', padding: 0, border: '1px solid #444', borderRadius: '3px', backgroundColor: '#1a1a2e', cursor: 'pointer', flexShrink: 0 }}
          title="Layer color"
        />
        <span style={{ fontSize: '12px', color: '#eee', flex: 1, overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap' }} title={src.label}>
          {src.label}
        </span>
        {hasHistogram && (
          <select
            value={src.thresholdMode}
            onChange={(e) => handleModeChange(e.target.value as HighlightThresholdMode)}
            style={{ padding: '1px 2px', fontSize: '11px', backgroundColor: '#1a1a2e', color: '#eee', border: '1px solid #0f3460', borderRadius: '3px', cursor: 'pointer' }}
            title="Threshold mode"
          >
            <option value="above">≥</option>
            <option value="below">≤</option>
            <option value="between">⇔</option>
          </select>
        )}
        <button
          onClick={onRemove}
          style={{ padding: '0 4px', fontSize: '14px', backgroundColor: 'transparent', color: '#888', border: 'none', cursor: 'pointer', lineHeight: 1 }}
          title="Remove this layer"
        >
          ×
        </button>
      </div>

      {/* Row 2 (geneset only) — full-width histogram with draggable cutoff */}
      {hasHistogram && (
        <div style={{ marginTop: '4px' }}>
          <HistogramChart
            histogram={histogram}
            mode={src.thresholdMode}
            lo={src.lo}
            hi={src.hi}
            onChangeLo={(v) => onUpdate({ source: { lo: v } })}
            onChangeHi={(v) => onUpdate({ source: { hi: v } })}
            width={252}
            height={44}
            barColor={layer.color}
          />
        </div>
      )}

      {isGeneset && histogram && histogram.zeroVariance && (
        <div style={{ fontSize: '10px', color: '#888', marginTop: '4px' }}>
          (no variance in this score — nothing to threshold)
        </div>
      )}

      {/* Row 3 — combined threshold-value (geneset) + intensity slider */}
      <div style={{ display: 'flex', alignItems: 'center', gap: '6px', marginTop: '4px' }}>
        {hasHistogram && (
          <span style={{ ...labelStyle, fontVariantNumeric: 'tabular-nums', minWidth: src.thresholdMode === 'between' ? '70px' : '40px' }}>
            {src.thresholdMode === 'between'
              ? `${src.lo.toFixed(2)}–${src.hi.toFixed(2)}`
              : src.lo.toFixed(2)}
          </span>
        )}
        <input
          type="range"
          min={0}
          max={1}
          step={0.05}
          value={layer.intensity}
          onChange={(e) => onUpdate({ intensity: parseFloat(e.target.value) })}
          style={{ flex: 1 }}
          title="Intensity"
        />
        <span style={{ fontSize: '10px', color: '#888', width: '30px', textAlign: 'right', fontVariantNumeric: 'tabular-nums' }}>
          {Math.round(layer.intensity * 100)}%
        </span>
      </div>
    </div>
  )
}

const inputStyle = {
  width: '100%',
  padding: '4px 6px',
  fontSize: '11px',
  backgroundColor: '#1a1a2e',
  color: '#eee',
  border: '1px solid #0f3460',
  borderRadius: '3px',
  cursor: 'pointer',
} as const

const pickerButtonStyle = {
  padding: '4px 8px',
  fontSize: '11px',
  backgroundColor: '#1a1a2e',
  color: '#ccc',
  border: '1px solid #444',
  borderRadius: '3px',
  cursor: 'pointer',
} as const

const pickerFormStyle = {
  backgroundColor: '#1a1a2e',
  borderRadius: '3px',
  padding: '8px',
  marginTop: '8px',
} as const

const layerRowStyle = {
  backgroundColor: '#1a1a2e',
  borderRadius: '3px',
  padding: '6px 8px',
  marginBottom: '6px',
} as const

const applyButtonStyle = (enabled: boolean) => ({
  padding: '4px 10px',
  fontSize: '11px',
  fontWeight: 500,
  backgroundColor: enabled ? '#4ecdc4' : '#333',
  color: enabled ? '#000' : '#666',
  border: 'none',
  borderRadius: '3px',
  cursor: enabled ? 'pointer' : 'not-allowed',
})

const cancelButtonStyle = {
  padding: '4px 10px',
  fontSize: '11px',
  backgroundColor: 'transparent',
  color: '#888',
  border: '1px solid #444',
  borderRadius: '3px',
  cursor: 'pointer',
} as const
