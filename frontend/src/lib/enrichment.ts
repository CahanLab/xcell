/** Pure helpers for the Enrichment modal: formatting, SVG geometry, filters,
 *  and turning result rows into Gene Panel sets. Kept free of React so the
 *  numerical/formatting contract is unit-testable. */

export interface OraRow {
  name: string
  library: string
  description: string
  url: string
  n_set: number
  n_overlap: number
  expected: number
  fold_enrichment: number
  odds_ratio: number
  pval: number
  padj: number
  genes: string[]
  below_min_overlap: boolean
}

export interface GseaRow {
  name: string
  library: string
  description: string
  url: string
  n_set: number
  es: number
  nes: number
  pval: number
  padj: number
  leading_edge: string[]
  n_leading_edge: number
  curve: [number, number][] | null
}

interface ResultCommon {
  key: string
  label: string
  created_at: string
  universe_size: number
  gene_subset_type: string
  n_sets_input: number
  n_sets_tested: number
  n_significant: number
  params: Record<string, unknown>
}

export interface OraResult extends ResultCommon {
  kind: 'ora'
  query: { name: string; n_input: number; n_in_universe: number; genes_missing: string[] }
  results: OraRow[]
}

export interface GseaResult extends ResultCommon {
  kind: 'gsea'
  ranking: { kind: string; label: string; n_ranked: number; genes: string[]; scores: number[] }
  n_perm: number
  results: GseaRow[]
}

export type EnrichmentResult = OraResult | GseaResult

export interface EnrichmentSummary {
  key: string
  kind: 'ora' | 'gsea'
  label: string
  n_sets_tested: number
  n_significant: number
  created_at: string
}

export interface CachedLibrary {
  source: string
  id: string
  name: string
  species: string
  n_sets: number
  version?: string | null
}

export type EnrichmentSource = { kind: 'ora'; name?: string; genes?: string[] } | { kind: 'gsea' }

/** `0.032`, `1.2e-7`, `1.0`; values at or below `floor` (the permutation
 *  floor) print as `<floor` so a wall of identical minimums reads honestly. */
export function formatP(p: number, floor?: number): string {
  if (floor !== undefined && p <= floor) return `<${floor.toPrecision(2)}`
  if (p >= 1) return '1.0'
  if (p >= 0.001) return String(Number(p.toPrecision(2)))
  return p.toExponential(1).replace('.0e', 'e')
}

/** SVG path for a running-score curve. y is symmetric around the midline so
 *  positive and negative ES read at the same scale. */
export function curvePath(curve: [number, number][], width: number, height: number, nRanked: number) {
  const yMax = Math.max(1e-9, ...curve.map(([, y]) => Math.abs(y)))
  const zeroY = height / 2
  const sx = (x: number) => (nRanked > 1 ? (x / (nRanked - 1)) * width : 0)
  const sy = (y: number) => zeroY - (y / yMax) * (height / 2)
  const d = curve.map(([x, y], i) => `${i === 0 ? 'M' : 'L'}${sx(x)},${sy(y)}`).join(' ')
  return { d, zeroY, yMax }
}

/** One x per hit: interior vertices come in (pre, post) pairs at the same x. */
export function hitTicks(curve: [number, number][]): number[] {
  const xs: number[] = []
  for (let i = 1; i + 1 < curve.length; i += 2) xs.push(curve[i][0])
  return xs
}

export function filterRows<T extends { name: string; library: string; padj: number }>(
  rows: T[],
  opts: { padjMax: number | null; query: string; hideBelowMinOverlap?: boolean },
): T[] {
  const q = opts.query.trim().toLowerCase()
  return rows.filter((r) => {
    if (opts.padjMax !== null && r.padj > opts.padjMax) return false
    if (opts.hideBelowMinOverlap && (r as { below_min_overlap?: boolean }).below_min_overlap) return false
    if (q && !r.name.toLowerCase().includes(q) && !r.library.toLowerCase().includes(q)) return false
    return true
  })
}

/** ORA rows become their overlap genes, GSEA rows their leading edge. */
export function resultsToGeneSets(
  result: EnrichmentResult,
  opts: { padjMax: number | null; topN: number },
): { name: string; genes: string[] }[] {
  if (result.kind === 'ora') {
    return filterRows(result.results, { padjMax: opts.padjMax, query: '', hideBelowMinOverlap: true })
      .slice(0, opts.topN)
      .filter((r) => r.genes.length > 0)
      .map((r) => ({ name: `${r.name} (${r.n_overlap}/${r.n_set})`, genes: r.genes }))
  }
  return filterRows(result.results, { padjMax: opts.padjMax, query: '' })
    .slice(0, opts.topN)
    .filter((r) => r.leading_edge.length > 0)
    .map((r) => ({ name: `${r.name} (NES ${r.nes.toFixed(1)})`, genes: r.leading_edge }))
}

export function libraryGroups(cached: CachedLibrary[]): { source: string; libraries: CachedLibrary[] }[] {
  const order: string[] = []
  const by = new Map<string, CachedLibrary[]>()
  for (const lib of cached) {
    if (!by.has(lib.source)) {
      by.set(lib.source, [])
      order.push(lib.source)
    }
    by.get(lib.source)!.push(lib)
  }
  return order.map((source) => ({ source, libraries: by.get(source)! }))
}

export function rowsToTsv(result: EnrichmentResult): string {
  const table: (string | number)[][] = []
  if (result.kind === 'ora') {
    table.push(['name', 'library', 'n_set', 'n_overlap', 'expected', 'fold_enrichment', 'odds_ratio', 'pval', 'padj', 'genes'])
    for (const r of result.results) {
      table.push([r.name, r.library, r.n_set, r.n_overlap, r.expected.toFixed(3), r.fold_enrichment.toFixed(3),
        r.odds_ratio.toFixed(3), r.pval.toExponential(3), r.padj.toExponential(3), r.genes.join(',')])
    }
  } else {
    table.push(['name', 'library', 'n_set', 'es', 'nes', 'pval', 'padj', 'leading_edge'])
    for (const r of result.results) {
      table.push([r.name, r.library, r.n_set, r.es.toFixed(4), r.nes.toFixed(3), r.pval.toExponential(3),
        r.padj.toExponential(3), r.leading_edge.join(',')])
    }
  }
  return table.map((row) => row.join('\t')).join('\n') + '\n'
}

/** Mean ranking score per bin, for the metric strip under a curve. */
export function metricStripBins(scores: number[], nBins: number): number[] {
  const out = new Array<number>(Math.max(0, nBins)).fill(0)
  if (scores.length === 0 || nBins <= 0) return out
  const per = scores.length / nBins
  for (let b = 0; b < nBins; b++) {
    const lo = Math.floor(b * per)
    const hi = Math.max(lo + 1, Math.floor((b + 1) * per))
    const slice = scores.slice(lo, hi)
    out[b] = slice.reduce((s, v) => s + v, 0) / slice.length
  }
  return out
}
