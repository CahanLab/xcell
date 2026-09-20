/**
 * Named cell subsets — the persisted, scoped form of the active cell mask.
 *
 * A mask is a browser-only boolean array; a subset is the same cells saved on
 * the backend as `obs['subset_<name>']`. The clustering chain (HVG, PCA,
 * Neighbors, UMAP, Leiden) run on a subset writes to keys suffixed with its
 * name, so the dataset's own results survive a sub-clustering and the masked
 * cells can always be brought back. The naming here mirrors the backend's;
 * both sides derive the same keys from the same name.
 */

export interface CellSubsetDerived {
  hvg: string | null
  pca: string | null
  graph: string | null
  umap: string[]          // every UMAP the subset owns (one per graph it was run over)
  leiden: string[]
  pca_subsets: string[]   // X_pca_<name>_<suffix> — PCs dropped from its own PCA
}

/** How a subset came to be; the embedding is the one on screen when it was
 * saved, which is what the tree can act on ("view where it was drawn"). */
export interface CellSubsetOrigin {
  kind: string
  embedding?: string
}

export interface CellSubsetInfo {
  name: string
  obs_key: string
  n_cells: number
  n_total: number
  created_at: string | null
  description: string | null
  // The tree: parent by containment, children in creation order, depth from a root.
  parent: string | null
  children: string[]
  depth: number
  origin: CellSubsetOrigin | null
  derived: CellSubsetDerived
  // The parameters each scoped step ran with, keyed as the backend records them.
  steps: Record<string, unknown>
  // Every .obsm key the subset owns that still exists: its PCA, PC subsets, UMAPs.
  embeddings: string[]
}

/** Operations whose results a subset scopes to suffixed keys. Everything else
 * (normalize, log1p, filter, QC…) acts on the cells in place and keeps sending
 * the mask as ad-hoc indices. */
export const SUBSET_SCOPED_OPS: ReadonlySet<string> = new Set([
  'highly_variable_genes', 'pca', 'neighbors', 'umap', 'leiden',
])

/** The backend's rule: runs of anything but [A-Za-z0-9_] become one '_',
 * leading/trailing '_' are dropped. Empty means "not a usable name". */
export function sanitizeSubsetName(raw: string): string {
  return String(raw ?? '').replace(/[^A-Za-z0-9_]+/g, '_').replace(/^_+|_+$/g, '')
}

const RESERVED = new Set(['unassigned', 'nan', 'none', 'null'])

export function isUsableSubsetName(raw: string): boolean {
  const s = sanitizeSubsetName(raw)
  return s.length > 0 && !RESERVED.has(s.toLowerCase())
}

/** 'sub1', 'sub2', … — the first not already taken. */
export function suggestSubsetName(existing: readonly string[]): string {
  const taken = new Set(existing)
  for (let i = 1; ; i++) {
    const candidate = `sub${i}`
    if (!taken.has(candidate)) return candidate
  }
}

export function indicesFromMask(mask: readonly boolean[]): number[] {
  const out: number[] = []
  for (let i = 0; i < mask.length; i++) if (mask[i]) out.push(i)
  return out
}

export function maskFromIndices(indices: readonly number[], nCells: number): boolean[] {
  const mask = new Array<boolean>(nCells).fill(false)
  for (const i of indices) if (i >= 0 && i < nCells) mask[i] = true
  return mask
}

/** True when the mask is exactly the subset's cells — the mask has not been
 * edited since the subset was saved or activated. */
export function maskMatchesIndices(mask: readonly boolean[], indices: readonly number[]): boolean {
  let n = 0
  for (const i of indices) {
    if (i < 0 || i >= mask.length || !mask[i]) return false
    n++
  }
  for (let i = 0; i < mask.length; i++) if (mask[i]) n--
  return n === 0
}

export const subsetPcaKey = (name: string) => `X_pca_${name}`
export const subsetGraphKey = (name: string) => `${name}_connectivities`
export const subsetUmapKey = (name: string) => `X_umap_${name}`
export const subsetLeidenKey = (name: string) => `leiden_${name}`
export const subsetHvgColumn = (name: string) => `highly_variable__${name}`

/** What a scoped operation writes, and what it would have overwritten. */
export function scopedOutput(op: string, name: string): { writes: string; spares: string } | null {
  switch (op) {
    case 'highly_variable_genes':
      return { writes: `.var["${subsetHvgColumn(name)}"]`, spares: '.var["highly_variable"]' }
    case 'pca':
      return { writes: `.obsm["${subsetPcaKey(name)}"]`, spares: '.obsm["X_pca"]' }
    case 'neighbors':
      return { writes: `.obsp["${subsetGraphKey(name)}"]`, spares: '.obsp["connectivities"]' }
    case 'umap':
      return { writes: `.obsm["${subsetUmapKey(name)}"]`, spares: '.obsm["X_umap"]' }
    case 'leiden':
      return { writes: `.obs["${subsetLeidenKey(name)}"]`, spares: '.obs["leiden"]' }
    default:
      return null
  }
}

/** Short badges for a subset row: which of the chain has been run on it. */
export function derivedBadges(d: CellSubsetDerived): string[] {
  const out: string[] = []
  if (d.hvg) out.push('HVG')
  if (d.pca) out.push('PCA')
  if (d.graph) out.push('kNN')
  for (const key of d.pca_subsets) out.push(key)
  for (const key of d.umap) out.push(key)
  for (const col of d.leiden) out.push(col)
  return out
}
