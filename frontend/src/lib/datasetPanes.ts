import type { DatasetState } from '../store'

/** Micron-per-unit for a slot's scale bar, or null for no bar.
 *
 * A dataset can carry a spatial scale and still be showing something that has
 * no physical extent, so the embedding on screen — not the dataset — decides
 * whether a bar means anything. Ten microns across a UMAP is nonsense.
 */
export function umPerUnitForSlot(ds: DatasetState | undefined): number | null {
  if (!ds) return null
  const sc = ds.spatialScale
  if (!sc || sc.umPerUnit == null) return null
  if (!ds.embedding || ds.embedding.name !== sc.spatialKey) return null
  return ds.displayPreferences.showScaleBar ? sc.umPerUnit : null
}

/** Heading for the expression legend.
 *
 * One gene is its own best label, even when it arrived as a one-gene set —
 * the set's name would hide which gene is actually on screen.
 */
export function expressionLegendTitle(
  selectedGenes: string[],
  selectedGeneSetName: string | null,
): string {
  if (selectedGenes.length === 1) return selectedGenes[0]
  if (selectedGeneSetName) return `${selectedGeneSetName} (${selectedGenes.length})`
  return `${selectedGenes.length} genes`
}
