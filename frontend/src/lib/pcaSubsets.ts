/**
 * PC subsets — which ones the store holds and which the Neighbors picker offers.
 *
 * A PC subset drops chosen components from a PCA. The dataset's live at
 * `X_pca_<suffix>`; a saved subset's, made from its own PCA, at
 * `X_pca_<name>_<suffix>` and the backend lists them separately. The store
 * keeps one combined list, each entry tagged with its owner, so the PCA
 * Loadings list can show only the current scope's while the Neighbors picker
 * offers everything a run on the current cells could use.
 */

export interface PCASubsetSummary {
  obsmKey: string       // e.g. 'X_pca_noPC2_5' or 'X_pca_chondro_noPC2_5'
  suffix: string        // e.g. 'noPC2_5'
  droppedPcs: number[]  // 1-indexed
  nPcsKept: number
  cellSubset: string | null  // the saved subset whose PCA it was made from; null = the dataset's
}

/** The subset's own PC subsets first, then the dataset's, each tagged. */
export function mergePcaSubsetLists(
  subsetList: readonly PCASubsetSummary[],
  datasetList: readonly PCASubsetSummary[],
  scope: string | null,
): PCASubsetSummary[] {
  return [
    ...(scope ? subsetList.map((s) => ({ ...s, cellSubset: scope })) : []),
    ...datasetList.map((s) => ({ ...s, cellSubset: null })),
  ]
}

export interface PcSourceOption {
  value: string
  label: string
}

/** What the Neighbors PC-source picker offers, in order: the active subset's
 * own PCA and PC subsets, then the dataset's PCA and PC subsets. */
export function pcSourceOptions(
  subsetPca: string | null, list: readonly PCASubsetSummary[],
): PcSourceOption[] {
  const out: PcSourceOption[] = []
  if (subsetPca) out.push({ value: subsetPca, label: `${subsetPca} (this subset's PCA)` })
  for (const s of list) {
    if (s.cellSubset) out.push({ value: s.obsmKey, label: `${s.obsmKey} (this subset, ${s.nPcsKept} kept)` })
  }
  out.push({ value: 'X_pca', label: 'X_pca (all PCs)' })
  for (const s of list) {
    if (!s.cellSubset) out.push({ value: s.obsmKey, label: `${s.obsmKey} (${s.nPcsKept} kept)` })
  }
  return out
}
