import { describe, it, expect } from 'vitest'
import { FIGURE_SCHEMAS, coerceParams } from './figureSchemas'

// A literal copy of backend/xcell/config.yaml `figures:` — the schema must
// name exactly the params the backend accepts, or a form field would 400.
const BACKEND_DEFAULTS: Record<string, Record<string, unknown>> = {
  enrichment_heatmap: {
    value: 'nes', padj_max: 0.05, top_n: 5, direction: 'both', collapse_jaccard: 0.5, row_order: 'peak',
    col_order: 'input', colormap: 'rdbu', vmax: null, show_values: false, label_max_chars: 40, cell_size: 14,
  },
  enrichment_network: {
    value: 'nes', padj_max: 0.05, top_n: 5, direction: 'both', set_edge_jaccard: 0.3, layout: 'force', seed: 0,
    node_size_by: 'degree', edge_width_by: 'value', label_max_chars: 30, colormap: 'rdbu',
  },
  composition_barplot: { order: 'category', share_of: null, normalize: true, min_cells: 0, show_values: false },
  expression_heatmap: { cell_ordering: 'category', gene_ordering: 'as_provided', aggregate_gene_sets: false, n_bins: 50 },
}

describe('FIGURE_SCHEMAS', () => {
  it('covers every backend param for every kind, and nothing else', () => {
    for (const [kind, defaults] of Object.entries(BACKEND_DEFAULTS)) {
      const names = FIGURE_SCHEMAS[kind as keyof typeof FIGURE_SCHEMAS].map((f) => f.name).sort()
      expect(names).toEqual(Object.keys(defaults).sort())
    }
  })
  it('every select field lists its default among its options', () => {
    for (const [kind, defaults] of Object.entries(BACKEND_DEFAULTS)) {
      for (const f of FIGURE_SCHEMAS[kind as keyof typeof FIGURE_SCHEMAS]) {
        // share_of's options are the live categories of column B; its default is null
        if (f.type === 'select' && defaults[f.name] !== null) expect(f.options!.map((o) => o.value)).toContain(defaults[f.name])
      }
    }
  })
})

describe('coerceParams', () => {
  it('rounds integer fields, nulls blank nullable numbers, keeps booleans', () => {
    const out = coerceParams('enrichment_heatmap', { top_n: '3.7', padj_max: '0.1', vmax: '', show_values: true, cell_size: 12.4, value: 'nes' })
    expect(out).toEqual({ top_n: 4, padj_max: 0.1, vmax: null, show_values: true, cell_size: 12, value: 'nes' })
    expect(coerceParams('enrichment_network', { seed: -3, set_edge_jaccard: '0.4' })).toEqual({ seed: 0, set_edge_jaccard: 0.4 })
  })
})
