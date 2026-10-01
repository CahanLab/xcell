/**
 * The Cells panel's subset tree — pure helpers over the backend's listing.
 *
 * A subset row shows chips for everything the subset owns: the steps run on
 * it, its embeddings (own PCA, PC subsets, UMAPs), its clusterings, and what
 * was drawn on those embeddings. Chips are typed so the panel knows what a
 * click does (switch embedding, colour by a column, open the refine modal).
 */
import type { CellSubsetInfo } from './cellSubsets'

export type SubsetChipKind = 'step' | 'embedding' | 'leiden' | 'pseudotime' | 'lines' | 'territories'

export interface SubsetChip {
  kind: SubsetChipKind
  label: string
  /** The key a click acts on: an .obsm name for an embedding, an .obs column for leiden / pseudotime. */
  key?: string
  /** For decoration chips: the embedding to switch to so the drawing is visible. */
  embedding?: string
}

const count = (n: number, one: string, many: string) => `${n} ${n === 1 ? one : many}`

/** Where a subset's decorations most likely are: its UMAP if it has one,
 * otherwise its last embedding. The listing names the shapes, not their
 * embedding, and a subset's drawings are almost always on its UMAP. */
function decorationEmbedding(s: CellSubsetInfo): string | undefined {
  return s.derived.umap[0] ?? s.embeddings[s.embeddings.length - 1]
}

export function subsetChips(s: CellSubsetInfo): SubsetChip[] {
  const d = s.derived
  const out: SubsetChip[] = []
  if (d.hvg) out.push({ kind: 'step', label: 'HVG', key: d.hvg })
  if (d.pca) out.push({ kind: 'embedding', label: d.pca, key: d.pca })
  for (const k of d.pca_subsets) out.push({ kind: 'embedding', label: k, key: k })
  if (d.graph) out.push({ kind: 'step', label: 'kNN', key: d.graph })
  for (const k of d.umap) out.push({ kind: 'embedding', label: k, key: k })
  for (const c of d.leiden) out.push({ kind: 'leiden', label: c, key: c })
  for (const k of d.diffmap) out.push({ kind: 'embedding', label: k, key: k })
  for (const c of d.dpt) out.push({ kind: 'pseudotime', label: c, key: c })
  const where = decorationEmbedding(s)
  const nLines = s.decorations.lines.length
  const nTerr = s.decorations.territories.length
  if (nLines > 0) out.push({ kind: 'lines', label: count(nLines, 'shape', 'shapes'), embedding: where })
  if (nTerr > 0) out.push({ kind: 'territories', label: count(nTerr, 'territory', 'territories'), embedding: where })
  return out
}

/** What "delete with results" removes, for the confirm row. Empty when nothing. */
export function dropSummary(s: CellSubsetInfo): string {
  const d = s.derived
  const parts: string[] = []
  if (d.hvg) parts.push(d.hvg)
  if (d.pca) parts.push(d.pca)
  parts.push(...d.pca_subsets)
  if (d.graph) parts.push(d.graph)
  parts.push(...d.umap, ...d.leiden, ...d.diffmap, ...d.dpt)
  const nLines = s.decorations.lines.length
  const nTerr = s.decorations.territories.length
  if (nLines > 0) parts.push(count(nLines, 'shape', 'shapes'))
  if (nTerr > 0) parts.push(count(nTerr, 'territory type', 'territory types'))
  return parts.join(', ')
}

/** The column a subset's Leiden result would refine: the parent's first
 * Leiden column, else the dataset's own `leiden`, else nothing. */
export function refineTarget(
  s: CellSubsetInfo, all: readonly CellSubsetInfo[], obsColumns: readonly string[],
): string | null {
  const parent = s.parent ? all.find((p) => p.name === s.parent) : undefined
  if (parent?.derived.leiden[0]) return parent.derived.leiden[0]
  if (obsColumns.includes('leiden')) return 'leiden'
  return null
}
