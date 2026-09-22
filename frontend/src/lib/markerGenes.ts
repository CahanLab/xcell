/**
 * The one-vs-rest marker-genes request, built from the modal's state.
 *
 * Kept outside the component so the request shape is pinned by a test — in
 * particular that `gene_subset` travels only when a column is chosen. The
 * modal defaults the choice to `highly_variable` when the dataset has that
 * column and to "all genes" otherwise; sending a stale column name to a
 * dataset without it is a 400 the user cannot get past.
 */

export interface MarkerGenesInputs {
  obsColumn: string
  selectedGroups: ReadonlySet<string>
  nCategories: number
  topN: number
  geneSubset: string
  minInGroupFraction: string
  maxOutGroupFraction: string
  minFoldChange: string
}

export interface MarkerGenesParams {
  obs_column: string
  groups?: string[]
  top_n: number
  min_in_group_fraction?: number
  max_out_group_fraction?: number
  min_fold_change?: number
  gene_subset?: string
}

export function markerGenesParams(i: MarkerGenesInputs): MarkerGenesParams {
  const params: MarkerGenesParams = { obs_column: i.obsColumn, top_n: i.topN }
  if (i.geneSubset) params.gene_subset = i.geneSubset
  // Only name the groups when not every category is checked.
  if (i.selectedGroups.size < i.nCategories) params.groups = Array.from(i.selectedGroups)
  const minIGF = parseFloat(i.minInGroupFraction)
  if (!Number.isNaN(minIGF)) params.min_in_group_fraction = minIGF
  const maxOGF = parseFloat(i.maxOutGroupFraction)
  if (!Number.isNaN(maxOGF)) params.max_out_group_fraction = maxOGF
  const minFC = parseFloat(i.minFoldChange)
  if (!Number.isNaN(minFC)) params.min_fold_change = minFC
  return params
}
