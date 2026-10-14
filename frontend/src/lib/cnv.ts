/**
 * The CNV inference request (Analyze → Cells → CNV inference), built from the
 * modal's state. Outside the component so the request shape is pinned by a
 * test and Run cannot send a stale value it forgot to list as a dependency.
 */

export type CnvCluster = 'leiden' | 'hierarchical' | 'none'

export interface CnvInputs {
  key: string
  /** Where gene positions come from. */
  positions: 'table' | 'columns'
  species: string
  chromosomeColumn: string
  startColumn: string
  /** '' = no reference: the all-cell mean stands in. */
  referenceColumn: string
  referenceCategories: ReadonlySet<string>
  includeXY: boolean
  cluster: CnvCluster
  resolution: string
  nClusters: string
  nPcs: string
  window: string
  step: string
  minMean: string
  lfcClip: string
  noiseThreshold: string
  centerCells: boolean
  /** Drop immunoglobulin / TCR / MHC genes before smoothing. */
  excludeLineageGenes: boolean
  /** A reference category needs this many cells to set the bounds. */
  minReferenceCells: string
  /** '' = adata.X */
  layer: string
  activeCellIndices: number[] | null
  /** Species with a shipped position table; absent = unknown. */
  tableSpecies?: string[]
  nCells?: number
  hierarchicalMaxCells?: number
}

// Mirrors DataAdaptor.prepare_cnv; the benchmark (docs/measurements/2026-10-13-cnv-benchmark.md)
// set minMean 0.02 and resolution 0.5.
export const CNV_DEFAULTS = {
  window: '100', step: '10', minMean: '0.02', lfcClip: '3', noiseThreshold: '1.5', resolution: '0.5', nClusters: '4', nPcs: '30',
  minReferenceCells: '50',
}

/** The key as the backend writes it: runs of other characters become '_'. */
export const cleanKey = (s: string) => s.trim().replace(/[^A-Za-z0-9_]+/g, '_').replace(/^_+|_+$/g, '')

const num = (s: string, fallback: number) => {
  const n = parseFloat(s)
  return Number.isFinite(n) ? n : fallback
}
const int = (s: string, fallback: number) => Math.round(num(s, fallback))

export function cnvParams(i: CnvInputs): Record<string, unknown> {
  const p: Record<string, unknown> = {
    key: cleanKey(i.key) || 'cnv',
    include_xy: i.includeXY,
    min_mean: num(i.minMean, 0.1),
    lfc_clip: num(i.lfcClip, 3),
    window: int(i.window, 100),
    step: int(i.step, 10),
    center_cells: i.centerCells,
    noise_threshold: num(i.noiseThreshold, 1.5),
    cluster: i.cluster,
    resolution: num(i.resolution, 1),
    n_clusters: int(i.nClusters, 4),
    n_pcs: int(i.nPcs, 30),
    min_reference_cells: int(i.minReferenceCells, 50),
    exclude_lineage_genes: i.excludeLineageGenes,
  }
  if (i.positions === 'columns') {
    p.chromosome_column = i.chromosomeColumn
    p.start_column = i.startColumn
    if (i.species) p.species = i.species
  } else {
    p.species = i.species
  }
  if (i.referenceColumn) {
    p.reference_column = i.referenceColumn
    p.reference_categories = [...i.referenceCategories]
  }
  if (i.layer) p.layer = i.layer
  if (i.activeCellIndices) p.active_cell_indices = i.activeCellIndices
  return p
}

/** Why Run is disabled, or null when the request can be sent. */
export function cnvBlocker(i: CnvInputs): string | null {
  if (i.positions === 'table') {
    if (!i.species) return 'Pick the species whose gene positions to use.'
    if (i.tableSpecies && !i.tableSpecies.includes(i.species)) return `No position table is shipped for ${i.species}.`
  } else if (!i.chromosomeColumn || !i.startColumn) {
    return 'Pick the .var columns that hold each gene’s chromosome and start.'
  }
  if (i.referenceColumn && i.referenceCategories.size === 0) return 'Tick at least one reference category, or clear the reference column.'
  const w = int(i.window, 0), s = int(i.step, 0)
  if (w < 2) return 'The window must be at least 2 genes.'
  if (s < 1 || s > w) return 'The step must be between 1 and the window.'
  if (i.cluster === 'hierarchical' && i.hierarchicalMaxCells != null && (i.nCells ?? 0) > i.hierarchicalMaxCells) {
    return `Hierarchical clustering is limited to ${i.hierarchicalMaxCells.toLocaleString()} cells; use Leiden.`
  }
  if (i.cluster === 'hierarchical' && int(i.nClusters, 0) < 1) return 'Ask for at least one clone.'
  if (!(num(i.lfcClip, 0) > 0)) return 'The clip must be positive.'
  if (int(i.minReferenceCells, 0) < 1) return 'A reference category needs at least one cell to set the bounds.'
  return null
}
