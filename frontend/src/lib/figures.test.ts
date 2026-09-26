import { describe, it, expect } from 'vitest'
import { defaultTitle, divergingColor, truncate, figureFilename, type FigureRecord } from './figures'

const rec = (over: Partial<FigureRecord> = {}): FigureRecord => ({
  id: 'fig_3', kind: 'enrichment_heatmap', title: 'Programs by cluster', caption: '', created_at: 't', updated_at: 't',
  inputs: { enrichment_keys: ['gsea_leiden_batch'] }, params: { padj_max: 0.05 },
  provenance: { steps: [4], created_step: 5, source: '' }, ...over,
})

describe('defaultTitle', () => {
  it('names enrichment figures after their inputs and barplots after their columns', () => {
    expect(defaultTitle('enrichment_heatmap', { enrichment_keys: ['gsea_leiden_batch'] }, { gsea_leiden_batch: 'GSEA: leiden (5 groups vs rest)' }))
      .toBe('Enrichment heatmap: GSEA: leiden (5 groups vs rest)')
    expect(defaultTitle('enrichment_network', { enrichment_keys: ['k'] }, {})).toBe('Enrichment network: k')
    expect(defaultTitle('composition_barplot', { column_a: 'leiden', column_b: 'sample' }, {})).toBe('Composition of leiden by sample')
  })
})

describe('divergingColor', () => {
  it('is white at zero, red-ish positive and blue-ish negative for rdbu, clamped at vmax', () => {
    expect(divergingColor(0, 2, 'rdbu')).toBe('rgb(247,247,247)')
    const pos = divergingColor(2, 2, 'rdbu')
    const neg = divergingColor(-2, 2, 'rdbu')
    const [pr, , pb] = pos.match(/\d+/g)!.map(Number)
    const [nr, , nb] = neg.match(/\d+/g)!.map(Number)
    expect(pr).toBeGreaterThan(pb)
    expect(nb).toBeGreaterThan(nr)
    expect(divergingColor(99, 2, 'rdbu')).toBe(pos)
    expect(divergingColor(1, 0, 'rdbu')).toBe('rgb(247,247,247)')   // vmax 0 never divides by zero
  })
})

describe('truncate + figureFilename', () => {
  it('truncates with an ellipsis and makes a safe filename', () => {
    expect(truncate('HALLMARK_EPITHELIAL_MESENCHYMAL_TRANSITION', 20)).toBe('HALLMARK_EPITHELIAL…')
    expect(truncate('short', 20)).toBe('short')
    expect(figureFilename(rec(), 'svg')).toBe('fig_3_Programs_by_cluster.svg')
    expect(figureFilename(rec({ title: '  a/b:c  ' }), 'png')).toBe('fig_3_a_b_c.png')
  })
})
