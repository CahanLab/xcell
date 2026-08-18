/** Pure pieces of the Filter Cells histograms: the log-axis mapping between
 * threshold values and chart coordinates, and the live pass count that must
 * mirror the backend's mask exactly (inclusive bounds, AND of all set bounds,
 * a null metric fails any bound on it). */

export function toChartSpace(value: number, log: boolean): number {
  return log ? Math.log10(1 + value) : value
}

export function fromChartSpace(chartValue: number, log: boolean): number {
  return log ? Math.round(Math.pow(10, chartValue) - 1) : Math.round(chartValue)
}

/** Compact count for axis labels: 950, 1.2k, 48k. */
export function formatCount(v: number): string {
  if (v < 1000) return String(Math.round(v))
  const k = v / 1000
  return `${k >= 100 ? Math.round(k) : Math.round(k * 10) / 10}k`
}

export interface FilterBounds {
  minCounts?: number | null
  maxCounts?: number | null
  minGenes?: number | null
  maxGenes?: number | null
}

export function passingCount(
  counts: Array<number | null>,
  genes: Array<number | null>,
  bounds: FilterBounds,
): number {
  const { minCounts, maxCounts, minGenes, maxGenes } = bounds
  let n = 0
  for (let i = 0; i < counts.length; i++) {
    const c = counts[i]
    const g = genes[i]
    if (minCounts != null && !(c != null && c >= minCounts)) continue
    if (maxCounts != null && !(c != null && c <= maxCounts)) continue
    if (minGenes != null && !(g != null && g >= minGenes)) continue
    if (maxGenes != null && !(g != null && g <= maxGenes)) continue
    n++
  }
  return n
}
