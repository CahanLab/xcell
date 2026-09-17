import { describe, it, expect } from 'vitest'
import { programsToGeneSets, coherenceVerdict, programWeight, type Program } from './geneSetDecomposition'

const prog = (over: Partial<Program>): Program => ({
  name: 'PC1', index: 0, genes: ['Col1a1'], weights: [0.5], genes_down: [], weights_down: [], ...over,
})

describe('programsToGeneSets', () => {
  it('names sets after the source and keeps the down list only when present', () => {
    const out = programsToGeneSets('NABA_COLLAGENS', 'pca', [
      prog({ name: 'PC1', genes: ['Col1a1', 'Col1a2'], genes_down: ['Col2a1'] }),
      prog({ name: 'PC2', index: 1, genes: ['Col2a1'] }),
    ])
    expect(out).toEqual([
      { name: 'NABA_COLLAGENS PC1', genes: ['Col1a1', 'Col1a2'], genesDown: ['Col2a1'] },
      { name: 'NABA_COLLAGENS PC2', genes: ['Col2a1'] },
    ])
  })
  it('drops a program with no genes on either side', () => {
    expect(programsToGeneSets('S', 'nmf', [prog({ genes: [] })])).toEqual([])
  })
})

describe('coherenceVerdict', () => {
  it('calls a tight set one pattern and a loose set several', () => {
    expect(coherenceVerdict({ eigengene_pve: 0.82, suggested_k: 1 }).kind).toBe('one')
    expect(coherenceVerdict({ eigengene_pve: 0.35, suggested_k: 3 }).kind).toBe('several')
    expect(coherenceVerdict({ eigengene_pve: 0.35, suggested_k: 3 }).text).toMatch(/3 patterns/)
  })
  it('treats the border as several, since a score would hide the rest', () => {
    expect(coherenceVerdict({ eigengene_pve: 0.6, suggested_k: 2 }).kind).toBe('several')
  })
  it('lets the backend noise-edge verdict override the variance-share rule', () => {
    // sparse single-cell data: 5% of variance in one pattern, but it is the only one above noise
    const v = coherenceVerdict({ eigengene_pve: 0.05, suggested_k: 1, one_pattern: true, n_significant: 1 })
    expect(v.kind).toBe('one')
    const w = coherenceVerdict({ eigengene_pve: 0.05, suggested_k: 3, one_pattern: false, n_significant: 3, signal_share_top: 0.41 })
    expect(w.kind).toBe('several')
    expect(w.text).toMatch(/3 patterns stand above the noise floor; the strongest carries only 41%/)
  })
})

describe('programWeight', () => {
  it('uses variance for PCA and the factor weight for NMF, normalised to the largest', () => {
    const ps = [prog({ variance_ratio: 0.4 }), prog({ name: 'PC2', variance_ratio: 0.2 })]
    expect(programWeight(ps[1], ps)).toBeCloseTo(0.5)
    const ns = [prog({ name: 'F1', factor_weight: 10 }), prog({ name: 'F2', factor_weight: 2.5 })]
    expect(programWeight(ns[1], ns)).toBeCloseTo(0.25)
    expect(programWeight(prog({}), [prog({})])).toBe(1)
  })
})
