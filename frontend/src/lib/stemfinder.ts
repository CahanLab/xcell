/**
 * The PyStemFinder request (Analyze → Cells → Differentiation), built from the
 * modal's state. Kept outside the component so the request shape is pinned by
 * a test, and so Run's callback cannot send a stale value it forgot to list
 * as a dependency — it hands the whole state to this function.
 */

export type StemMetric = 'stemfinder' | 'diffometer' | 'cc_mean' | 'n_tfs'

/** Table order; also the order the backend writes columns in. */
export const STEM_METRICS: { key: StemMetric; label: string; hint: string }[] = [
  { key: 'stemfinder', label: 'stemFinder', hint: 'Cell-cycle heterogeneity among each cell’s neighbours. Writes stemfinder (lower = less differentiated, like pseudotime) and stemfinder_raw.' },
  { key: 'diffometer', label: 'diffOmeter', hint: 'Mean Gini impurity of the binarized markers across the neighbourhood. Higher = less differentiated.' },
  { key: 'cc_mean', label: 'Cell-cycle expression', hint: 'Baseline: mean log expression of the markers (stemfinder_cc_mean). Higher = more proliferative.' },
  { key: 'n_tfs', label: 'Expressed TFs', hint: 'Baseline: how many of the species’ transcription factors each cell expresses (stemfinder_n_TFs).' },
]

export interface StemFinderInputs {
  metrics: ReadonlySet<StemMetric>
  markerSource: 'cell_cycle' | 'gene_set'
  species: string
  geneSetGenes: string[] | null
  method: 'gini' | 'stdev' | 'variance'
  threshold: string
  binarizeOn: 'scaled' | 'log_normalized'
  weightBy: 'equal' | 'expression' | 'presence'
  includeSelf: boolean
  graph: 'build' | 'existing'
  useRep: string
  nPcs: string
  nNeighbors: string
  graphKey: string
  /** '' = adata.X */
  layer: string
  suffix: string
  /** '' = no per-group summary */
  summaryBy: string
  activeCellIndices: number[] | null
}

const needsGraph = (m: ReadonlySet<StemMetric>) => m.has('stemfinder') || m.has('diffometer')
const needsMarkers = (m: ReadonlySet<StemMetric>) => needsGraph(m) || m.has('cc_mean')

export function stemFinderParams(i: StemFinderInputs): Record<string, unknown> {
  const p: Record<string, unknown> = {
    metrics: STEM_METRICS.map((m) => m.key).filter((k) => i.metrics.has(k)),
    species: i.species,
    method: i.method,
    threshold: Number.isFinite(parseFloat(i.threshold)) ? parseFloat(i.threshold) : 0,
    binarize_on: i.binarizeOn,
    weight_by: i.weightBy,
    include_self: i.includeSelf,
    graph: i.graph,
    suffix: i.suffix.trim(),
  }
  if (i.markerSource === 'gene_set' && i.geneSetGenes) p.markers = i.geneSetGenes
  if (i.graph === 'build') {
    p.use_rep = i.useRep
    const pcs = parseInt(i.nPcs, 10)
    p.n_pcs = Number.isFinite(pcs) && pcs > 0 ? pcs : null
    const k = parseInt(i.nNeighbors, 10)
    if (Number.isFinite(k)) p.n_neighbors = k
  } else {
    p.graph_key = i.graphKey
  }
  if (i.layer) p.layer = i.layer
  if (i.summaryBy) p.summary_by = i.summaryBy
  if (i.activeCellIndices) p.active_cell_indices = i.activeCellIndices
  return p
}

/** Why Run is disabled, or null when it can run. */
export function stemFinderBlocker(i: StemFinderInputs): string | null {
  if (i.metrics.size === 0) return 'Pick at least one metric.'
  if (needsMarkers(i.metrics) && i.markerSource === 'gene_set' && !(i.geneSetGenes && i.geneSetGenes.length)) {
    return 'Pick a gene set with genes in it, or use the cell-cycle markers.'
  }
  if (needsGraph(i.metrics)) {
    if (i.graph === 'build' && i.nNeighbors.trim() !== '') {
      const k = Number(i.nNeighbors)
      if (!Number.isInteger(k) || k < 2) return 'k must be a whole number of at least 2 (blank = √n).'
    }
    if (i.graph === 'build' && !i.useRep) return 'Pick a PC embedding to build the kNN graph on.'
    if (i.graph === 'existing' && !i.graphKey) return 'Pick a kNN graph.'
    if (i.metrics.has('diffometer') && i.weightBy === 'expression' && i.binarizeOn === 'scaled') {
      return 'Weighting by expression needs log-normalized binarizing (scaled expression averages ~0).'
    }
  }
  return null
}
