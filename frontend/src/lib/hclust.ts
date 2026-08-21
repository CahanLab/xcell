/** Agglomerative hierarchical clustering for reordering small heatmaps.
 *
 * A cell-type × cell-type matrix in input order (usually alphabetical, or
 * Leiden's numeric order) hides its own block structure: types that share a
 * neighbourhood land wherever the alphabet puts them. Reordering by similarity
 * is what turns the matrix into a readable picture, and it is the convention
 * both histoCAT and squidpy's `nhood_enrichment` plot follow.
 *
 * Sizes here are tens of rows, so the naive O(n³) Lance-Williams loop is far
 * below the noise floor of a React render — no need for the nearest-neighbour
 * chain algorithm scipy uses.
 */

export type Linkage = 'average' | 'complete' | 'single' | 'ward'
export type DistanceMetric = 'correlation' | 'euclidean' | 'cosine'

export interface Merge {
  /** Cluster ids being joined: leaves are 0..n-1, internal nodes n..2n-2. */
  a: number
  b: number
  height: number
  size: number
}

export interface ClusterResult {
  /** Leaf indices, left to right — the row/column order to draw. */
  order: number[]
  merges: Merge[]
}

const dot = (a: number[], b: number[]) => a.reduce((s, v, i) => s + v * b[i], 0)
const norm2 = (a: number[]) => Math.sqrt(dot(a, a))

/** Pairwise distances between the rows of `rows`.
 *
 * Correlation distance is 1 − Pearson r, so it ranges 0 (same shape) to 2
 * (opposite shape) and ignores magnitude — the right default when rows are
 * enrichment profiles whose overall scale differs by cell-type abundance.
 * A constant row has no shape to correlate, so it is placed at distance 1
 * (uncorrelated) from everything rather than propagating NaN through the tree.
 */
export function pairwiseDistance(rows: number[][], metric: DistanceMetric): number[][] {
  const n = rows.length
  const d: number[][] = Array.from({ length: n }, () => new Array(n).fill(0))

  const prepared = rows.map((r) => {
    const clean = r.map((v) => (Number.isFinite(v) ? v : 0))
    if (metric === 'correlation') {
      const mean = clean.reduce((s, v) => s + v, 0) / (clean.length || 1)
      const centered = clean.map((v) => v - mean)
      return { vec: centered, len: norm2(centered) }
    }
    return { vec: clean, len: norm2(clean) }
  })

  for (let i = 0; i < n; i++) {
    for (let j = i + 1; j < n; j++) {
      const A = prepared[i]
      const B = prepared[j]
      let dist: number
      if (metric === 'euclidean') {
        let s = 0
        for (let c = 0; c < A.vec.length; c++) s += (A.vec[c] - B.vec[c]) ** 2
        dist = Math.sqrt(s)
      } else {
        // correlation and cosine are the same computation; correlation just
        // centers first (done above).
        dist = A.len === 0 || B.len === 0 ? 1 : 1 - dot(A.vec, B.vec) / (A.len * B.len)
      }
      d[i][j] = dist
      d[j][i] = dist
    }
  }
  return d
}

/** Lance-Williams update for the distance between a freshly merged cluster
 *  (a ∪ b) and some other cluster k. */
function updated(
  linkage: Linkage,
  dAK: number, dBK: number, dAB: number,
  nA: number, nB: number, nK: number,
): number {
  switch (linkage) {
    case 'single':
      return Math.min(dAK, dBK)
    case 'complete':
      return Math.max(dAK, dBK)
    case 'average':
      return (nA * dAK + nB * dBK) / (nA + nB)
    case 'ward': {
      const t = nA + nB + nK
      return Math.sqrt(
        Math.max(
          0,
          ((nA + nK) * dAK * dAK + (nB + nK) * dBK * dBK - nK * dAB * dAB) / t,
        ),
      )
    }
  }
}

/** Cluster the rows of `matrix` and return the leaf order plus the merge list.
 *
 * Leaves are ordered by walking the tree and, at every merge, flipping the two
 * subtrees into whichever of the four end-to-end arrangements puts the most
 * similar pair at the join. That is the cheap half of optimal leaf ordering
 * (Bar-Joseph et al. 2001): it fixes the arrangement that makes a clean block
 * structure look scattered, without the O(n³) dynamic program.
 */
export function hierarchicalOrder(
  matrix: number[][],
  opts: { metric: DistanceMetric; linkage: Linkage },
): ClusterResult {
  const n = matrix.length
  if (n <= 1) return { order: n === 1 ? [0] : [], merges: [] }

  const dist = pairwiseDistance(matrix, opts.metric)

  // Active clusters, each holding its leaves in current left-to-right order.
  const leaves = new Map<number, number[]>()
  const size = new Map<number, number>()
  for (let i = 0; i < n; i++) {
    leaves.set(i, [i])
    size.set(i, 1)
  }
  // d[i][j] over cluster ids, grown as clusters are created.
  const d = new Map<number, Map<number, number>>()
  const setD = (i: number, j: number, v: number) => {
    if (!d.has(i)) d.set(i, new Map())
    if (!d.has(j)) d.set(j, new Map())
    d.get(i)!.set(j, v)
    d.get(j)!.set(i, v)
  }
  const getD = (i: number, j: number) => d.get(i)?.get(j) ?? Infinity
  for (let i = 0; i < n; i++) for (let j = i + 1; j < n; j++) setD(i, j, dist[i][j])

  const merges: Merge[] = []
  let nextId = n

  while (leaves.size > 1) {
    // Closest pair among active clusters.
    let best = Infinity
    let bi = -1
    let bj = -1
    const ids = [...leaves.keys()]
    for (let x = 0; x < ids.length; x++) {
      for (let y = x + 1; y < ids.length; y++) {
        const v = getD(ids[x], ids[y])
        if (v < best) {
          best = v
          bi = ids[x]
          bj = ids[y]
        }
      }
    }

    const la = leaves.get(bi)!
    const lb = leaves.get(bj)!
    // Orient the two subtrees so the closest ends meet at the join.
    const ends = [
      { a: la, b: lb, cost: dist[la[la.length - 1]][lb[0]] },
      { a: [...la].reverse(), b: lb, cost: dist[la[0]][lb[0]] },
      { a: la, b: [...lb].reverse(), cost: dist[la[la.length - 1]][lb[lb.length - 1]] },
      { a: [...la].reverse(), b: [...lb].reverse(), cost: dist[la[0]][lb[lb.length - 1]] },
    ].reduce((m, c) => (c.cost < m.cost ? c : m))

    const nA = size.get(bi)!
    const nB = size.get(bj)!
    merges.push({ a: bi, b: bj, height: best, size: nA + nB })

    // Distances from the new cluster to every survivor, then retire a and b.
    for (const k of leaves.keys()) {
      if (k === bi || k === bj) continue
      setD(nextId, k, updated(
        opts.linkage, getD(bi, k), getD(bj, k), best, nA, nB, size.get(k)!,
      ))
    }
    leaves.delete(bi)
    leaves.delete(bj)
    d.delete(bi)
    d.delete(bj)
    leaves.set(nextId, [...ends.a, ...ends.b])
    size.set(nextId, nA + nB)
    nextId++
  }

  return { order: [...leaves.values()][0], merges }
}

export interface DendrogramSegment {
  x1: number
  y1: number
  x2: number
  y2: number
}

/** The line segments of a dendrogram drawn against a heatmap's leaf order.
 *
 * Coordinates are in matrix-cell units — `x` is the leaf slot (0..n-1, at cell
 * centres) and `y` is the merge height — so the caller scales both to pixels
 * and decides which way is up. Each merge contributes two uprights and a
 * crossbar. `maxHeight` is floored above zero: a dataset where everything
 * merges at distance 0 would otherwise scale the drawing by 1/0.
 */
export function dendrogramSegments(result: ClusterResult): {
  segments: DendrogramSegment[]
  maxHeight: number
} {
  const { order, merges } = result
  const pos = new Map<number, number>()
  const height = new Map<number, number>()
  order.forEach((leaf, slot) => {
    pos.set(leaf, slot)
    height.set(leaf, 0)
  })

  const segments: DendrogramSegment[] = []
  let nextId = order.length
  for (const m of merges) {
    const xa = pos.get(m.a) ?? 0
    const xb = pos.get(m.b) ?? 0
    segments.push({ x1: xa, y1: height.get(m.a) ?? 0, x2: xa, y2: m.height })
    segments.push({ x1: xb, y1: height.get(m.b) ?? 0, x2: xb, y2: m.height })
    segments.push({ x1: xa, y1: m.height, x2: xb, y2: m.height })
    pos.set(nextId, (xa + xb) / 2)
    height.set(nextId, m.height)
    nextId++
  }

  const root = merges.length ? merges[merges.length - 1].height : 0
  return { segments, maxHeight: root > 0 ? root : 1 }
}
