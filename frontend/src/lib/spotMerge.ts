/** Reading a spot-merge preview: the numbers that decide `min_correlation`. */

export interface MergePreview {
  n_region_spots: number
  n_merged_spots: number
  n_spots_before: number
  n_spots_after: number
  pitch_um: number
  max_spots: number
  size_histogram: Record<string, number>
  n_vetoed: number
  correlation_quantiles: { p10: number; p50: number; p90: number } | null
  median_counts_before: number
  median_counts_after?: number
  gene_subset?: { type: string; n_genes: number }
}

/** "1:12  2:31  3:44" — how many groups came out at each size. */
export function formatSizeHistogram(h: Record<string, number>): string {
  const entries = Object.entries(h).sort((a, b) => Number(a[0]) - Number(b[0]))
  if (entries.length === 0) return '—'
  return entries.map(([size, n]) => `${size}:${n}`).join('  ')
}

/** Whether the veto threshold is earning its place, in one sentence.
 *
 *  The tuning question is not how many merges were blocked. It is whether the
 *  candidate correlations sit near the threshold at all: a veto far below every
 *  observed correlation is inert however many pairs there were, and one far
 *  above would block everything. What matters is that the cut falls inside the
 *  distribution.
 */
export function vetoVerdict(
  nVetoed: number,
  q: { p10: number; p50: number; p90: number } | null,
  threshold: number,
): string {
  if (!q) return 'No candidate pairs were tested.'
  if (nVetoed === 0) {
    return `The veto blocked nothing — every candidate pair scored above ${threshold}. `
      + 'Raise it if merges are crossing boundaries you can see.'
  }
  if (q.p10 < threshold && threshold < q.p90) {
    return `Candidate correlations straddle ${threshold} (p10 ${q.p10.toFixed(2)}, `
      + `p90 ${q.p90.toFixed(2)}), so the veto is doing real work — ${nVetoed} pairs blocked.`
  }
  return `${nVetoed} pairs blocked, but ${threshold} sits outside the bulk of the `
    + `distribution (p10 ${q.p10.toFixed(2)}, p90 ${q.p90.toFixed(2)}), so it fires rarely.`
}
