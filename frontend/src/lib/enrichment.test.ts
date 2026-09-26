import { describe, it, expect } from 'vitest'
import {
  formatP, curvePath, hitTicks, filterRows, resultsToGeneSets, libraryGroups, rowsToTsv, metricStripBins, reconcileChoice,
  isBatch, batchGroupsSorted, batchToGeneSets,
  type OraResult, type GseaResult, type BatchCollection,
} from './enrichment'

const ora = (over: Partial<OraResult> = {}): OraResult => ({
  key: 'ora_q', kind: 'ora', label: 'Overlap: q', created_at: 't',
  query: { name: 'q', n_input: 3, n_in_universe: 3, genes_missing: [] },
  universe_size: 100, gene_subset_type: 'all', n_sets_input: 2, n_sets_tested: 2, n_significant: 1,
  results: [
    { name: 'A', library: 'L', description: '', url: '', n_set: 10, n_overlap: 3, expected: 0.3, fold_enrichment: 10, odds_ratio: 20, pval: 1e-4, padj: 2e-4, genes: ['x', 'y', 'z'], below_min_overlap: false },
    { name: 'B', library: 'M', description: '', url: '', n_set: 10, n_overlap: 1, expected: 0.3, fold_enrichment: 3, odds_ratio: 3, pval: 0.3, padj: 0.3, genes: ['x'], below_min_overlap: true },
  ],
  params: {},
  ...over,
})

const gsea = (): GseaResult => ({
  key: 'gsea_k', kind: 'gsea', label: 'GSEA: k', created_at: 't',
  ranking: { kind: 'diffexp', label: 'grp: a vs rest', n_ranked: 10, genes: ['g0', 'g1', 'g2', 'g3', 'g4', 'g5', 'g6', 'g7', 'g8', 'g9'], scores: [5, 4, 3, 2, 1, -1, -2, -3, -4, -5] },
  universe_size: 10, gene_subset_type: 'all', n_sets_input: 1, n_sets_tested: 2, n_significant: 1, n_perm: 100,
  results: [
    { name: 'TOP', library: 'L', description: '', url: '', n_set: 2, es: 0.8, nes: 1.9, pval: 1 / 101, padj: 0.02, leading_edge: ['g0', 'g1'], n_leading_edge: 2, curve: [[0, 0], [0, -0.0], [0, 0.5], [1, 0.5], [1, 1.0], [9, 0]] },
    { name: 'DOWN', library: 'L', description: '', url: '', n_set: 2, es: -0.7, nes: -1.5, pval: 0.2, padj: 0.2, leading_edge: ['g8', 'g9'], n_leading_edge: 2, curve: null },
  ],
  params: {},
})

describe('formatP', () => {
  it('formats large, small and floored values', () => {
    expect(formatP(0.032)).toBe('0.032')
    expect(formatP(1.234e-7)).toBe('1.2e-7')
    expect(formatP(1 / 101, 1 / 101)).toBe('<0.0099')
    expect(formatP(1)).toBe('1.0')
  })
})

describe('curvePath', () => {
  it('maps hits to x and running score to y around a zero line', () => {
    const { d, zeroY, yMax } = curvePath(gsea().results[0].curve!, 100, 50, 10)
    expect(yMax).toBe(1)
    expect(zeroY).toBe(25)
    expect(d.startsWith('M0,25')).toBe(true)
    expect(d).toContain('L100,25')      // last vertex at n-1 maps to full width
  })
  it('hitTicks lists one x per hit', () => {
    expect(hitTicks(gsea().results[0].curve!)).toEqual([0, 1])
  })
})

describe('filterRows', () => {
  it('applies padj, text and min-overlap filters', () => {
    const rows = ora().results
    expect(filterRows(rows, { padjMax: 0.05, query: '' }).map((r) => r.name)).toEqual(['A'])
    expect(filterRows(rows, { padjMax: null, query: 'm' }).map((r) => r.name)).toEqual(['B'])
    expect(filterRows(rows, { padjMax: null, query: '', hideBelowMinOverlap: true }).map((r) => r.name)).toEqual(['A'])
  })
})

describe('resultsToGeneSets', () => {
  it('uses overlap genes for ORA and leading edge for GSEA, capped and filtered', () => {
    expect(resultsToGeneSets(ora(), { padjMax: 0.05, topN: 50 })).toEqual([{ name: 'A (3/10)', genes: ['x', 'y', 'z'] }])
    expect(resultsToGeneSets(gsea(), { padjMax: null, topN: 1 })).toEqual([{ name: 'TOP (NES 1.9)', genes: ['g0', 'g1'] }])
  })
})

describe('libraryGroups + rowsToTsv + metricStripBins', () => {
  it('groups cached libraries by source keeping order', () => {
    const g = libraryGroups([
      { source: 'msigdb', id: 'a', name: 'A', species: 'mouse', n_sets: 1 },
      { source: 'go', id: 'b', name: 'B', species: 'mouse', n_sets: 2 },
      { source: 'msigdb', id: 'c', name: 'C', species: 'human', n_sets: 3 },
    ])
    expect(g.map((x) => x.source)).toEqual(['msigdb', 'go'])
    expect(g[0].libraries.map((l) => l.id)).toEqual(['a', 'c'])
  })
  it('writes a header and one row per result', () => {
    const tsv = rowsToTsv(ora())
    const lines = tsv.trim().split('\n')
    expect(lines[0].split('\t')).toEqual(['name', 'library', 'n_set', 'n_overlap', 'expected', 'fold_enrichment', 'odds_ratio', 'pval', 'padj', 'genes'])
    expect(lines).toHaveLength(3)
    expect(rowsToTsv(gsea()).split('\n')[0].split('\t')).toEqual(['name', 'library', 'n_set', 'es', 'nes', 'pval', 'padj', 'leading_edge'])
  })
  it('bins scores by mean', () => {
    expect(metricStripBins([4, 2, -2, -4], 2)).toEqual([3, -3])
    expect(metricStripBins([], 3)).toEqual([0, 0, 0])
  })
})

describe('reconcileChoice', () => {
  it('keeps a value that is still offered and falls back otherwise', () => {
    expect(reconcileChoice('leiden', ['cell_type', 'leiden'], 'cell_type')).toBe('leiden')
    expect(reconcileChoice('gone', ['cell_type', 'leiden'], 'cell_type')).toBe('cell_type')
    expect(reconcileChoice('', ['a'], '')).toBe('')            // '' is the "none" sentinel, always valid
    expect(reconcileChoice('x', [], '')).toBe('')
  })
})

describe('batch collections', () => {
  const member = (nSig: number): GseaResult => ({ ...gsea(), n_significant: nSig })
  const col: BatchCollection = {
    key: 'gsea_grp_batch', kind: 'gsea_batch', label: 'GSEA: grp (2 groups vs rest)', created_at: 't',
    obs_column: 'grp', reference: 'rest', groups: ['a', 'b'], members: { a: 'gsea_grp_a_vs_rest', b: 'gsea_grp_b_vs_rest' },
    skipped: { c: 'fewer than 2 cells in group' }, n_perm: 100, universe_size: 10, gene_subset_type: 'all',
    n_sets_tested: 2, n_significant: 3, params: {},
    member_results: { a: member(1), b: member(2) },
  }
  it('isBatch distinguishes collections from single results', () => {
    expect(isBatch(col)).toBe(true)
    expect(isBatch(gsea())).toBe(false)
    expect(isBatch(ora())).toBe(false)
  })
  it('batchGroupsSorted orders groups by significant hits, most first', () => {
    expect(batchGroupsSorted(col)).toEqual(['b', 'a'])
    expect(batchGroupsSorted({ ...col, member_results: undefined })).toEqual(['a', 'b'])
  })
  it('batchToGeneSets makes one folder per group from each member', () => {
    const folders = batchToGeneSets(col, { padjMax: null, topN: 1 })
    expect(folders.map((f) => f.folder)).toEqual(['gsea_grp_a_vs_rest', 'gsea_grp_b_vs_rest'])
    expect(folders[0].sets).toEqual([{ name: 'TOP (NES 1.9)', genes: ['g0', 'g1'] }])
  })
})
