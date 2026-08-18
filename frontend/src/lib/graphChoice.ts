/** Which kNN graph the UMAP/Leiden selects should preselect. */

/** Only the key matters here; callers pass their own richer graph rows. */
export interface NeighborGraphInfo {
  key: string
  label?: string
  n_edges?: number
  suffix?: string
}

/** The expression kNN graph (`obsp['connectivities']`, from Cell Neighbors)
 * is what UMAP/Leiden should run on unless the user says otherwise — the
 * implicit empty option answers the same question but hides *which* graph is
 * used, and fails outright on datasets that carry the graph without a
 * `uns['neighbors']` entry.
 *
 * Returns the key to preselect, or null to leave the choice as it is.
 * Never overrides a non-empty choice, and never promotes a non-expression
 * graph: silently clustering the spatial graph would answer a different
 * question.
 */
export function defaultGraphKey(
  graphs: NeighborGraphInfo[],
  current: string,
): string | null {
  if (current !== '') return null
  return graphs.some((g) => g.key === 'connectivities') ? 'connectivities' : null
}
