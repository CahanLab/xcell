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

describe('tab configs become figure records', () => {
  it('maps a barplot config, preferring a named subset over indices', async () => {
    const { barplotFigureFromConfig } = await import('./figures')
    const cfg = { columnA: 'leiden', columnB: 'sample', order: 'share' as const, shareOf: 'A', normalize: false, minCells: 5, showValues: true }
    expect(barplotFigureFromConfig(cfg, 'chondro', [1, 2])).toEqual({
      kind: 'composition_barplot',
      inputs: { column_a: 'leiden', column_b: 'sample', cell_subset: 'chondro', cell_indices: null },
      params: { order: 'share', share_of: 'A', normalize: false, min_cells: 5, show_values: true },
    })
    expect(barplotFigureFromConfig({ ...cfg, shareOf: null }, null, [1, 2]).inputs).toEqual({ column_a: 'leiden', column_b: 'sample', cell_subset: null, cell_indices: [1, 2] })
    expect(barplotFigureFromConfig(cfg, null, null).params.share_of).toBe('A')
  })
  it('maps a heatmap config to gene sets {name, genes} and display params', async () => {
    const { heatmapFigureFromConfig } = await import('./figures')
    const cfg = {
      selectedGeneSets: [{ id: 'x', name: 'collagen', genes: ['Col1a1', 'Col1a2'] }], cellOrdering: 'line_position' as const,
      obsColumn: 'leiden', lineName: 'L1', geneOrdering: 'peak_position' as const, aggregateGeneSets: true, nBins: 20, cellIndices: [3, 4],
    }
    expect(heatmapFigureFromConfig(cfg, null, 'log1p')).toEqual({
      kind: 'expression_heatmap',
      inputs: { gene_sets: [{ name: 'collagen', genes: ['Col1a1', 'Col1a2'] }], obs_column: 'leiden', line_name: 'L1', transform: 'log1p', cell_subset: null, cell_indices: [3, 4] },
      params: { cell_ordering: 'line_position', gene_ordering: 'peak_position', aggregate_gene_sets: true, n_bins: 20 },
    })
    // the tab never draws with the active subset, so callers pass null; the helper still honours one if given
    expect(heatmapFigureFromConfig({ ...cfg, cellIndices: null }, 'chondro').inputs.cell_subset).toBe('chondro')
    expect(heatmapFigureFromConfig(cfg, null).inputs.transform).toBeNull()
  })
})
