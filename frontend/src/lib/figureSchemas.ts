/** Per-kind parameter schemas that drive the Figures tab's params form.
 *  The field names are the backend's `figures.<kind>` config keys; a test
 *  pins the two lists together so a form field can never 400. */

import type { FigureKind } from './figures'

export interface ParamOption { value: string; label: string }

export interface ParamField {
  name: string
  label: string
  type: 'number' | 'int' | 'select' | 'bool' | 'nullable-number'
  min?: number
  max?: number
  step?: number
  options?: ParamOption[]
  help?: string
}

const VALUE_OPTIONS: ParamOption[] = [
  { value: 'nes', label: 'NES' },
  { value: 'signed_logp', label: 'signed −log10 padj' },
  { value: 'es', label: 'ES' },
  { value: 'fold_enrichment', label: 'log2 fold enrichment' },
]
const DIRECTION_OPTIONS: ParamOption[] = [
  { value: 'both', label: 'up and down' }, { value: 'up', label: 'up only' }, { value: 'down', label: 'down only' },
]
const COLORMAP_OPTIONS: ParamOption[] = [
  { value: 'rdbu', label: 'blue – white – red' }, { value: 'prgn', label: 'purple – white – green' }, { value: 'roma', label: 'roma' },
]

const SHARED_ENRICHMENT: ParamField[] = [
  { name: 'value', label: 'Value', type: 'select', options: VALUE_OPTIONS, help: 'What a cell / edge carries. ORA results have no NES: use signed −log10 padj.' },
  { name: 'padj_max', label: 'padj ≤', type: 'number', min: 0, max: 1, step: 0.01, help: 'Cells above this are drawn as 0.' },
  { name: 'top_n', label: 'Top N per column', type: 'int', min: 1, max: 100, help: 'Positive and negative separately; rows are the union.' },
  { name: 'direction', label: 'Direction', type: 'select', options: DIRECTION_OPTIONS },
]

export const FIGURE_SCHEMAS: Record<FigureKind, ParamField[]> = {
  enrichment_heatmap: [
    ...SHARED_ENRICHMENT,
    { name: 'collapse_jaccard', label: 'Collapse rows (Jaccard ≥)', type: 'nullable-number', min: 0, max: 1, step: 0.05, help: 'Merge sets whose member genes overlap at least this much; blank = off.' },
    { name: 'row_order', label: 'Row order', type: 'select', options: [{ value: 'peak', label: 'by peak column' }, { value: 'cluster', label: 'hierarchical' }, { value: 'input', label: 'as selected' }] },
    { name: 'col_order', label: 'Column order', type: 'select', options: [{ value: 'input', label: 'as given' }, { value: 'cluster', label: 'hierarchical' }] },
    { name: 'colormap', label: 'Colours', type: 'select', options: COLORMAP_OPTIONS },
    { name: 'vmax', label: 'Colour limit (±)', type: 'nullable-number', min: 0, step: 0.5, help: 'Blank = the data maximum.' },
    { name: 'show_values', label: 'Show values', type: 'bool' },
    { name: 'label_max_chars', label: 'Label length', type: 'int', min: 8, max: 120 },
    { name: 'cell_size', label: 'Cell size (px)', type: 'int', min: 6, max: 40 },
  ],
  enrichment_network: [
    ...SHARED_ENRICHMENT,
    { name: 'set_edge_jaccard', label: 'Set–set edges (Jaccard ≥)', type: 'nullable-number', min: 0, max: 1, step: 0.05, help: 'Link two gene sets whose members overlap at least this much; blank = none.' },
    { name: 'layout', label: 'Layout', type: 'select', options: [{ value: 'force', label: 'force-directed' }, { value: 'bipartite', label: 'two columns' }] },
    { name: 'seed', label: 'Layout seed', type: 'int', min: 0 },
    { name: 'node_size_by', label: 'Node size', type: 'select', options: [{ value: 'degree', label: 'by degree' }, { value: 'n_set', label: 'by set size' }, { value: 'constant', label: 'constant' }] },
    { name: 'edge_width_by', label: 'Edge width', type: 'select', options: [{ value: 'value', label: 'by value' }, { value: 'constant', label: 'constant' }] },
    { name: 'label_max_chars', label: 'Label length', type: 'int', min: 8, max: 120 },
    { name: 'colormap', label: 'Colours', type: 'select', options: COLORMAP_OPTIONS },
  ],
  composition_barplot: [
    { name: 'order', label: 'Bar order', type: 'select', options: [{ value: 'category', label: 'category order' }, { value: 'alphabetical', label: 'alphabetical' }, { value: 'total', label: 'by total cells' }, { value: 'share', label: 'by share of…' }] },
    { name: 'share_of', label: 'Share of', type: 'select', options: [] },
    { name: 'normalize', label: 'Normalise to 100%', type: 'bool' },
    { name: 'min_cells', label: 'Min cells per bar', type: 'int', min: 0 },
    { name: 'show_values', label: 'Show values', type: 'bool' },
  ],
  expression_heatmap: [
    { name: 'cell_ordering', label: 'Cell order', type: 'select', options: [{ value: 'none', label: 'none' }, { value: 'category', label: 'by category' }, { value: 'line_position', label: 'by line position' }, { value: 'line_distance', label: 'by line distance' }, { value: 'category_then_position', label: 'category, then position' }] },
    { name: 'gene_ordering', label: 'Gene order', type: 'select', options: [{ value: 'as_provided', label: 'as provided' }, { value: 'peak_position', label: 'by peak position' }] },
    { name: 'aggregate_gene_sets', label: 'Aggregate gene sets', type: 'bool' },
    { name: 'n_bins', label: 'Bins', type: 'int', min: 0, max: 1000 },
  ],
}

/** Turn form strings into what the backend expects: ints rounded and
 *  clamped at their minimum, blank nullable numbers → null, booleans kept. */
export function coerceParams(kind: FigureKind, raw: Record<string, unknown>): Record<string, unknown> {
  const fields = new Map(FIGURE_SCHEMAS[kind].map((f) => [f.name, f]))
  const out: Record<string, unknown> = {}
  for (const [name, value] of Object.entries(raw)) {
    const f = fields.get(name)
    if (!f) { out[name] = value; continue }
    if (f.type === 'bool') { out[name] = Boolean(value); continue }
    if (f.type === 'select') { out[name] = value == null ? null : String(value); continue }
    if (f.type === 'nullable-number') {
      const s = value == null ? '' : String(value).trim()
      const n = Number(s)
      out[name] = s === '' || !Number.isFinite(n) ? null : n
      continue
    }
    let n = Number(value)
    if (!Number.isFinite(n)) n = f.min ?? 0
    if (f.type === 'int') n = Math.round(n)
    if (f.min !== undefined) n = Math.max(f.min, n)
    if (f.max !== undefined) n = Math.min(f.max, n)
    out[name] = n
  }
  return out
}
