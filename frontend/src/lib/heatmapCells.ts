/**
 * Which cells the Heatmap tab draws.
 *
 * A config can carry its own restriction (`cellIndices`: the cells a gene
 * clustering ran on). The cell mask narrows that like it narrows every other
 * view — the tab used to ignore the mask entirely and draw cells the user had
 * put out of scope. `subset` names the saved subset the drawn cells *are*, so
 * a saved figure records `cell_subset` rather than an index list; it is null
 * whenever a restriction makes the drawn cells something narrower.
 */
import { indicesFromMask } from './cellSubsets'

export interface HeatmapCellScope {
  /** null = every cell. */
  indices: number[] | null
  subset: string | null
}

export function heatmapCellScope(
  configIndices: readonly number[] | null | undefined,
  mask: readonly boolean[] | null,
  maskSubsetName: string | null,
): HeatmapCellScope {
  const restricted = configIndices && configIndices.length > 0 ? configIndices : null
  if (!mask) return { indices: restricted ? [...restricted] : null, subset: null }
  if (!restricted) return { indices: indicesFromMask(mask), subset: maskSubsetName }
  return { indices: restricted.filter((i) => mask[i]), subset: null }
}
