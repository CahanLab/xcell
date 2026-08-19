/**
 * Layout and color helpers for the meta-program similarity heatmap.
 *
 * The heatmap is programs x programs, laid out in the dendrogram's leaf order
 * so each meta-program appears as a bright block on the diagonal — that block
 * structure is the whole diagnostic: a meta-program whose block is dim, or
 * bleeds into its neighbours, is not a real consensus.
 */

export interface ClusterBand {
  cluster: number
  start: number   // position in the leaf order where the run begins
  size: number
}

/**
 * Contiguous runs of one meta-program along the dendrogram leaf order.
 *
 * `clusters` is indexed by program; `order` lists program indices in leaf
 * order. A maxclust cut normally makes each meta-program one run, but this
 * never merges non-adjacent runs — a band drawn across programs that aren't
 * neighbours would claim a block the matrix doesn't show. Programs whose
 * meta-program was dropped carry cluster -1 and band as themselves.
 */
export function clusterBands(clusters: number[], order: number[]): ClusterBand[] {
  const bands: ClusterBand[] = []
  for (let pos = 0; pos < order.length; pos++) {
    const cluster = clusters[order[pos]]
    const last = bands[bands.length - 1]
    if (last && last.cluster === cluster && last.start + last.size === pos) {
      last.size += 1
    } else {
      bands.push({ cluster, start: pos, size: 1 })
    }
  }
  return bands
}

/** Similarity in [0, 1] to a color on xcell's dark inset → accent ramp. */
export function similarityColor(value: number): string {
  const v = Number.isFinite(value) ? Math.min(1, Math.max(0, value)) : 0
  // #0f1625 (inset) → #4ecdc4 (accent), eased so mid similarities stay legible
  const t = Math.sqrt(v)
  const r = Math.round(15 + (78 - 15) * t)
  const g = Math.round(22 + (205 - 22) * t)
  const b = Math.round(37 + (196 - 37) * t)
  return `rgb(${r}, ${g}, ${b})`
}

/** "2/3 samples" — coverage is a fraction, but people count samples. */
export function coverageLabel(coverage: number, nSamples: number): string {
  const n = Math.round(coverage * nSamples)
  return `${n}/${nSamples} samples`
}
