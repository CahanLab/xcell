/** Downsample: pick a random subset of a pool of cells and make it the
 *  selection. Everything here is pure so the sampling contract — uniform,
 *  without replacement, reproducible from a seed — is pinned by tests rather
 *  than by reading the modal.
 */

export type Pool = 'selection' | 'active' | 'all'
export type TargetMode = 'count' | 'percent'

export interface ResolvedTarget {
  n: number | null
  reason?: string
}

/** Turn the user's number into a cell count against the pool size. A target
 *  that would not thin the pool is refused rather than silently selecting
 *  everything, since the point is to end up with fewer cells. */
export function resolveTarget(mode: TargetMode, raw: unknown, poolSize: number): ResolvedTarget {
  if (poolSize <= 0) return { n: null, reason: 'Nothing to sample from' }
  if (raw === '' || raw === null || raw === undefined) return { n: null, reason: 'Enter a target' }
  const value = typeof raw === 'number' ? raw : Number(String(raw).trim())
  if (!Number.isFinite(value) || value <= 0) return { n: null, reason: 'Enter a positive number' }

  let n: number
  if (mode === 'count') {
    n = Math.floor(value)
    if (n < 1) return { n: null, reason: 'Enter at least 1 cell' }
  } else {
    if (value >= 100) return { n: null, reason: 'Percent must be below 100' }
    // Round, but never to zero: asking for 0.01% of a small pool still means
    // "give me something", not "give me nothing".
    n = Math.max(1, Math.round((value / 100) * poolSize))
  }
  if (n >= poolSize) {
    return { n: null, reason: `Target must be smaller than the pool (${poolSize.toLocaleString()} cells)` }
  }
  return { n }
}

/** mulberry32 — a small, decent 32-bit PRNG. Math.random cannot be seeded,
 *  and a downsample that cannot be reproduced cannot be written up. */
export function seededRng(seed: number): () => number {
  let a = seed >>> 0
  return () => {
    a = (a + 0x6d2b79f5) >>> 0
    let t = a
    t = Math.imul(t ^ (t >>> 15), t | 1)
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61)
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296
  }
}

/** Partial Fisher–Yates: the first n positions of a shuffled copy. Result is
 *  sorted ascending so the selection reads like any other index list. */
export function sampleWithoutReplacement(pool: readonly number[], n: number, rng: () => number): number[] {
  const k = Math.max(0, Math.min(n, pool.length))
  const arr = [...pool]
  for (let i = 0; i < k; i++) {
    const j = i + Math.floor(rng() * (arr.length - i))
    const tmp = arr[i]; arr[i] = arr[j]; arr[j] = tmp
  }
  return arr.slice(0, k).sort((a, b) => a - b)
}

export function poolIndices(kind: Pool, selected: readonly number[], mask: readonly boolean[] | null, nCells: number): number[] {
  if (kind === 'selection') return [...selected]
  if (kind === 'active' && mask) {
    const out: number[] = []
    for (let i = 0; i < mask.length; i++) if (mask[i]) out.push(i)
    return out
  }
  return Array.from({ length: nCells }, (_, i) => i)
}

/** Which pools make sense right now, most specific first — the first entry
 *  is the default, so a lassoed cluster is what gets thinned, not the dataset. */
export function availablePools(selectedCount: number, hasMask: boolean): Pool[] {
  const out: Pool[] = []
  if (selectedCount > 0) out.push('selection')
  if (hasMask) out.push('active')
  out.push('all')
  return out
}

export function parseSeed(raw: string): number | null {
  const s = raw.trim()
  if (!/^\d+$/.test(s)) return null
  return Number(s)
}

export function drawSeed(): number {
  return Math.floor(Math.random() * 1_000_000_000)
}
