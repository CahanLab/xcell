/**
 * Which cells an embedding can show, and where to go to see the rest.
 *
 * A subset's UMAP / PCA has NaN rows for the cells outside it (by design —
 * the dataset's own keys are never overwritten), which reach the browser as
 * null coordinates and are simply not drawn. That is fine while the subset is
 * the point; it is a trap once the user wants the whole dataset back, because
 * nothing but the embedding picker changes it. These helpers let the pane say
 * how many cells are hidden and offer the way back.
 */
import type { CellSubsetInfo } from './cellSubsets'
import { pickPreferredEmbedding } from './pickSecondEmbedding'

type Coord = readonly (number | null)[] | null | undefined

export function cellsWithoutCoordinates(coords: readonly Coord[]): number {
  let n = 0
  for (const c of coords) {
    if (!c || c[0] == null || Number.isNaN(c[0])) n++
  }
  return n
}

const at = (c: Coord, k: 0 | 1): number => {
  const v = c?.[k]
  return v == null ? NaN : v
}

/** Positions for deck.gl's binary attribute, NaN for a cell with no
 * coordinates so it is not drawn. A Float32Array stores null as 0, which drew
 * every such cell as one dot at the origin. */
export function packPositions(coords: readonly Coord[]): Float32Array {
  const buf = new Float32Array(coords.length * 2)
  for (let i = 0; i < coords.length; i++) {
    buf[i * 2] = at(coords[i], 0)
    buf[i * 2 + 1] = at(coords[i], 1)
  }
  return buf
}

/** Indices of the cells inside a ray-cast polygon. A cell with no coordinates
 * is inside none: `null < x` reads null as 0, so a lasso around the origin
 * used to take every one of them. */
export function cellsInPolygon(coords: readonly Coord[], polygon: readonly (readonly [number, number])[]): number[] {
  const out: number[] = []
  for (let k = 0; k < coords.length; k++) {
    const x = at(coords[k], 0), y = at(coords[k], 1)
    if (!Number.isFinite(x) || !Number.isFinite(y)) continue
    let inside = false
    for (let i = 0, j = polygon.length - 1; i < polygon.length; j = i++) {
      const [xi, yi] = polygon[i]
      const [xj, yj] = polygon[j]
      if ((yi > y) !== (yj > y) && x < (xj - xi) * (y - yi) / (yj - yi) + xi) inside = !inside
    }
    if (inside) out.push(k)
  }
  return out
}

/** The embedding to show the superset on, leaving `name`:
 *  1. where the owning subset was drawn (its origin) — for a child drawn on
 *     its parent's UMAP that is one level up, which is what "back" means;
 *  2. the dataset's counterpart with the subset's name removed
 *     (`X_umap_chondro` → `X_umap`, `X_umap_chondro_alt` → `X_umap_alt`);
 *  3. the preferred embedding among the rest.
 * Null when `name` is the only embedding there is. */
export function supersetEmbedding(
  name: string, embeddings: readonly string[], subsets: readonly CellSubsetInfo[],
): string | null {
  const listed = (e: string | null | undefined): e is string =>
    !!e && e !== name && embeddings.includes(e)
  const owner = subsets.find((s) => s.embeddings.includes(name))
  if (owner) {
    if (listed(owner.origin?.embedding)) return owner.origin!.embedding!
    const counterpart = name.replace(`_${owner.name}`, '')
    if (listed(counterpart)) return counterpart
  }
  return pickPreferredEmbedding(embeddings.filter((e) => e !== name))
}
