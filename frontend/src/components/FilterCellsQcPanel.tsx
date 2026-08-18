import { useMemo, useState } from 'react'
import { computeHistogram, HistogramChart } from '../utils/histogram'
import {
  toChartSpace,
  fromChartSpace,
  formatCount,
  passingCount,
} from '../lib/filterCellsQc'

export interface FilterCellsDistributions {
  counts: Array<number | null>
  genes: Array<number | null>
}

const dark = {
  label: { fontSize: '11px', color: '#aaa', marginBottom: '3px' } as const,
  hint: { fontSize: '11px', color: '#888' } as const,
  passLine: { fontSize: '12px', color: '#4ecdc4', margin: '6px 0 2px' } as const,
  toggleRow: {
    display: 'flex', alignItems: 'center', gap: '6px',
    fontSize: '11px', color: '#aaa', margin: '2px 0 8px', cursor: 'pointer',
  } as const,
}

/** One histogram with draggable min/max cutoffs bound to two thresholds. */
function ThresholdChart({
  title,
  values,
  lo,
  hi,
  log,
  onChange,
}: {
  title: string
  values: Array<number | null>
  lo: number | null
  hi: number | null
  log: boolean
  onChange: (which: 'lo' | 'hi', value: number | null) => void
}) {
  const chartValues = useMemo(
    () => (log ? values.map((v) => (v == null ? null : toChartSpace(v, true))) : values),
    [values, log],
  )
  const histogram = useMemo(() => computeHistogram(chartValues, 60), [chartValues])
  // Raw-space extent, for edge-clearing: a cutoff dragged to the axis end
  // means "no threshold", shown as the empty input beside it.
  const rawMin = useMemo(() => fromChartSpace(histogram.min, log), [histogram.min, log])
  const rawMax = useMemo(() => fromChartSpace(histogram.max, log), [histogram.max, log])

  if (histogram.zeroVariance) {
    return (
      <div>
        <div style={dark.label}>{title}</div>
        <div style={dark.hint}>Every cell has the same value — nothing to threshold.</div>
      </div>
    )
  }

  const handle = (which: 'lo' | 'hi') => (chartValue: number) => {
    const raw = fromChartSpace(chartValue, log)
    if (which === 'lo') onChange('lo', raw <= rawMin ? null : raw)
    else onChange('hi', raw >= rawMax ? null : raw)
  }

  return (
    <div>
      <div style={dark.label}>{title}</div>
      <HistogramChart
        histogram={histogram}
        mode="between"
        lo={lo != null ? toChartSpace(lo, log) : histogram.min}
        hi={hi != null ? toChartSpace(hi, log) : histogram.max}
        onChangeLo={handle('lo')}
        onChangeHi={handle('hi')}
        width={460}
        height={104}
        formatValue={(v) => formatCount(fromChartSpace(v, log))}
      />
    </div>
  )
}

/** The Filter Cells preamble: counts/genes histograms whose draggable
 * cutoffs set the same four thresholds as the inputs below them, plus a
 * live count of what the current thresholds would keep. */
export function FilterCellsQcPanel({
  dist,
  error,
  minCounts,
  maxCounts,
  minGenes,
  maxGenes,
  onChangeThreshold,
}: {
  dist: FilterCellsDistributions | null
  error: string | null
  minCounts: number | null
  maxCounts: number | null
  minGenes: number | null
  maxGenes: number | null
  onChangeThreshold: (name: string, value: number | null) => void
}) {
  const [log, setLog] = useState(true)

  const passing = useMemo(
    () => (dist
      ? passingCount(dist.counts, dist.genes, {
          minCounts, maxCounts, minGenes, maxGenes,
        })
      : null),
    [dist, minCounts, maxCounts, minGenes, maxGenes],
  )

  if (error) return <div style={dark.hint}>Distributions unavailable: {error}</div>
  if (!dist) return <div style={dark.hint}>Loading distributions…</div>

  return (
    <div style={{ marginBottom: '10px' }}>
      <label style={dark.toggleRow}>
        <input
          type="checkbox"
          checked={log}
          onChange={(e) => setLog(e.target.checked)}
        />
        log₁₀(1+x) axis
      </label>
      <div style={{ display: 'flex', flexDirection: 'column', gap: '10px' }}>
        <ThresholdChart
          title="Total counts per cell — drag the red lines to set Min/Max counts"
          values={dist.counts}
          lo={minCounts}
          hi={maxCounts}
          log={log}
          onChange={(which, v) =>
            onChangeThreshold(which === 'lo' ? 'min_counts' : 'max_counts', v)}
        />
        <ThresholdChart
          title="Genes per cell — drag the red lines to set Min/Max genes"
          values={dist.genes}
          lo={minGenes}
          hi={maxGenes}
          log={log}
          onChange={(which, v) =>
            onChangeThreshold(which === 'lo' ? 'min_genes' : 'max_genes', v)}
        />
      </div>
      {passing != null && (
        <div style={dark.passLine}>
          {passing.toLocaleString()} of {dist.counts.length.toLocaleString()} cells
          pass the current thresholds
        </div>
      )}
    </div>
  )
}
