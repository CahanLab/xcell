/** Types and pure helpers for the Figures tab: the figure record, the data
 *  shapes each renderer draws, colour ramps, filenames and PNG rasterising.
 *  Kept free of React so the contract is unit-testable. */

export type FigureKind = 'enrichment_heatmap' | 'enrichment_network' | 'composition_barplot' | 'expression_heatmap'

export const FIGURE_KIND_LABELS: Record<FigureKind, string> = {
  enrichment_heatmap: 'Enrichment heatmap',
  enrichment_network: 'Enrichment network',
  composition_barplot: 'Composition barplot',
  expression_heatmap: 'Expression heatmap',
}

export interface FigureProvenance {
  steps: number[]
  created_step: number | null
  source: string
}

export interface FigureRecord {
  id: string
  kind: FigureKind
  title: string
  caption: string
  created_at: string
  updated_at: string
  inputs: Record<string, unknown>
  params: Record<string, unknown>
  provenance: FigureProvenance
}

export interface FigureSummary {
  id: string
  kind: FigureKind
  title: string
  created_at: string
  updated_at: string
  inputs: Record<string, unknown>
  n_provenance_steps: number
}

export interface EnrichmentHeatmapData {
  rows: { name: string; library: string; members: string[]; n_set: number }[]
  cols: { label: string; key?: string }[]
  values: number[][]
  padj: number[][]
  value_label: string
  n_rows_total: number
  n_collapsed: number
  note: string | null
}

export interface EnrichmentNetworkData {
  nodes: { id: string; kind: 'group' | 'set'; label: string; x: number; y: number; size: number; degree: number; library: string | null }[]
  edges: { source: string; target: string; kind: 'enrichment' | 'overlap'; value: number; padj: number | null }[]
  bounds: [number, number, number, number]
  note: string | null
}

/** `/api/obs/crosstab` plus the params echoed back (composition_barplot). */
export interface CrosstabData {
  a: string
  b: string
  a_categories: string[]
  b_categories: string[]
  a_colors: (string | null)[] | null
  b_colors: (string | null)[] | null
  counts: number[][]
  n_cells: number
  n_total?: number
  params?: Record<string, unknown>
}

/** `/api/heatmap/data` (expression_heatmap). */
export interface ExpressionHeatmapData {
  matrix: number[][]
  row_labels: string[]
  row_groups: (string | null)[]
  column_groups: { name: string; start: number; size: number }[]
  n_bins: number
  n_cells: number
  n_genes_hidden?: number
}

/** Every data payload names the figure and kind it was computed for, so the
 *  view never has to guess what it is drawing. */
export interface FigureDataTag { figure_id?: string; kind?: FigureKind }

export type FigureData = (EnrichmentHeatmapData | EnrichmentNetworkData | CrosstabData | ExpressionHeatmapData) & FigureDataTag

export interface BarplotConfigLike {
  columnA: string
  columnB: string
  order: 'category' | 'alphabetical' | 'total' | 'share'
  shareOf: string | null
  normalize: boolean
  minCells: number
  showValues: boolean
}

export interface HeatmapConfigLike {
  selectedGeneSets: { id?: string; name: string; genes: string[] }[]
  cellOrdering: string
  obsColumn: string | null
  lineName: string | null
  geneOrdering: string
  aggregateGeneSets: boolean
  nBins: number
  cellIndices?: number[] | null
}

/** The cells a tab is showing, as figure inputs: a named subset when one is
 *  active (it follows edits), else the frozen selection, else every cell. */
function cellInputs(cellSubset: string | null, cellIndices: number[] | null | undefined) {
  if (cellSubset) return { cell_subset: cellSubset, cell_indices: null }
  return { cell_subset: null, cell_indices: cellIndices && cellIndices.length ? cellIndices : null }
}

export function barplotFigureFromConfig(cfg: BarplotConfigLike, cellSubset: string | null, cellIndices: number[] | null) {
  return {
    kind: 'composition_barplot' as const,
    inputs: { column_a: cfg.columnA, column_b: cfg.columnB, ...cellInputs(cellSubset, cellIndices) },
    params: { order: cfg.order, share_of: cfg.shareOf, normalize: cfg.normalize, min_cells: cfg.minCells, show_values: cfg.showValues },
  }
}

/** The Heatmap tab draws only the cells in `config.cellIndices` (never the
 *  active subset or mask), so callers pass `null` for the subset; `transform`
 *  is the display transform the tab drew with, part of what the figure is. */
export function heatmapFigureFromConfig(cfg: HeatmapConfigLike, cellSubset: string | null, transform: 'log1p' | null = null) {
  return {
    kind: 'expression_heatmap' as const,
    inputs: {
      gene_sets: cfg.selectedGeneSets.map((g) => ({ name: g.name, genes: g.genes })),
      obs_column: cfg.obsColumn, line_name: cfg.lineName, transform,
      ...cellInputs(cellSubset, cfg.cellIndices ?? null),
    },
    params: { cell_ordering: cfg.cellOrdering, gene_ordering: cfg.geneOrdering, aggregate_gene_sets: cfg.aggregateGeneSets, n_bins: cfg.nBins },
  }
}

/** Params → the BarplotView config shape the shared renderer draws from. */
export function paramsToBarplotConfig(params: Record<string, unknown>): Omit<BarplotConfigLike, 'columnA' | 'columnB'> {
  return {
    order: (params.order as BarplotConfigLike['order']) ?? 'category',
    shareOf: (params.share_of as string | null) ?? null,
    normalize: params.normalize !== false,
    minCells: Number(params.min_cells ?? 0),
    showValues: Boolean(params.show_values),
  }
}

export interface CanvasLegend {
  /** Gradient stops, left to right, as CSS colours. */
  gradient: string[]
  gradientLabel: string
  groups: { name: string; color: string }[]
}

/** Compose a canvas figure into a PNG at `scale`× with a title and legend
 *  drawn in, so the file stands on its own like the SVG exports do. The
 *  source canvas is a DPR-scaled backing store; it is drawn at its CSS size. */
export function canvasToPngBlob(canvas: HTMLCanvasElement, opts: { scale?: number; title?: string; legend?: CanvasLegend; background?: string } = {}): Promise<Blob> {
  const scale = Math.max(1, opts.scale ?? 1)
  const cssW = canvas.clientWidth || canvas.width
  const cssH = canvas.clientHeight || canvas.height
  const titleH = opts.title ? 22 : 0
  const legendH = opts.legend ? 18 + 14 * Math.max(1, opts.legend.groups.length) : 0
  const out = document.createElement('canvas')
  out.width = Math.round(cssW * scale)
  out.height = Math.round((cssH + titleH + legendH + 8) * scale)
  const ctx = out.getContext('2d')
  if (!ctx) return Promise.reject(new Error('No 2D canvas context'))
  ctx.scale(scale, scale)
  ctx.fillStyle = opts.background ?? '#1a1a2e'
  ctx.fillRect(0, 0, cssW, cssH + titleH + legendH + 8)
  if (opts.title) {
    ctx.fillStyle = '#eee'
    ctx.font = '600 13px sans-serif'
    ctx.fillText(opts.title, 8, 15)
  }
  ctx.drawImage(canvas, 0, titleH, cssW, cssH)
  if (opts.legend) {
    const y0 = titleH + cssH + 8
    const g = ctx.createLinearGradient(8, 0, 108, 0)
    opts.legend.gradient.forEach((c, i) => g.addColorStop(i / Math.max(1, opts.legend!.gradient.length - 1), c))
    ctx.fillStyle = g
    ctx.fillRect(8, y0, 100, 10)
    ctx.fillStyle = '#aaa'
    ctx.font = '10px sans-serif'
    ctx.fillText(`${opts.legend.gradientLabel}  0 … 1`, 114, y0 + 9)
    opts.legend.groups.forEach((grp, i) => {
      const y = y0 + 16 + i * 14
      ctx.fillStyle = grp.color
      ctx.fillRect(8, y, 10, 10)
      ctx.fillStyle = '#aaa'
      ctx.fillText(grp.name, 22, y + 9)
    })
  }
  return new Promise((resolve, reject) => {
    out.toBlob((blob) => (blob ? resolve(blob) : reject(new Error('PNG encoding failed (too large for this browser?)'))), 'image/png')
  })
}

export function defaultTitle(kind: FigureKind, inputs: Record<string, unknown>, labelsByKey: Record<string, string>): string {
  if (kind === 'enrichment_heatmap' || kind === 'enrichment_network') {
    const keys = (inputs.enrichment_keys as string[] | undefined) ?? []
    const what = kind === 'enrichment_heatmap' ? 'heatmap' : 'network'
    return `Enrichment ${what}: ${keys.map((k) => labelsByKey[k] ?? k).join('; ')}`
  }
  if (kind === 'composition_barplot') return `Composition of ${String(inputs.column_a)} by ${String(inputs.column_b)}`
  const sets = (inputs.gene_sets as { name: string }[] | undefined) ?? []
  return `Expression heatmap: ${sets.slice(0, 3).map((s) => s.name).join(', ')}`
}

/** Five-stop diverging ramps, negative → zero → positive. */
const RAMPS: Record<string, [number, number, number][]> = {
  rdbu: [[5, 48, 97], [103, 169, 207], [247, 247, 247], [239, 138, 98], [103, 0, 31]],
  prgn: [[64, 0, 75], [175, 141, 195], [247, 247, 247], [127, 191, 123], [0, 68, 27]],
  roma: [[126, 23, 0], [219, 165, 95], [247, 247, 247], [96, 176, 176], [1, 51, 80]],
}

export function divergingColor(v: number, vmax: number, colormap: string): string {
  const ramp = RAMPS[colormap] ?? RAMPS.rdbu
  if (!(vmax > 0) || !Number.isFinite(v)) return `rgb(${ramp[2].join(',')})`
  const t = Math.max(-1, Math.min(1, v / vmax))   // −1..1
  const pos = (t + 1) * 2                            // 0..4 across five stops
  const i = Math.min(3, Math.floor(pos))
  const f = pos - i
  const a = ramp[i], b = ramp[i + 1]
  const mix = a.map((c, k) => Math.round(c + (b[k] - c) * f))
  return `rgb(${mix.join(',')})`
}

export function truncate(label: string, n: number): string {
  return label.length > n ? `${label.slice(0, Math.max(1, n - 1))}…` : label
}

export function figureFilename(record: Pick<FigureRecord, 'id' | 'title'>, ext: string): string {
  const safe = record.title.trim().replace(/[^A-Za-z0-9_-]+/g, '_').replace(/^_+|_+$/g, '')
  return `${record.id}_${safe || 'figure'}.${ext}`
}

/** Rasterise an on-screen SVG at `scale`× onto a canvas and return a PNG blob.
 *  The SVG is serialised with its xmlns and painted onto a background so the
 *  file is not white-on-transparent in viewers. */
export function svgToPngBlob(svg: SVGSVGElement, scale: number, background: string): Promise<Blob> {
  return new Promise((resolve, reject) => {
    const w = svg.width.baseVal.value || svg.getBoundingClientRect().width
    const h = svg.height.baseVal.value || svg.getBoundingClientRect().height
    if (!(w > 0 && h > 0)) return reject(new Error('The figure has no size yet'))
    let markup = new XMLSerializer().serializeToString(svg)
    if (!markup.includes('xmlns=')) markup = markup.replace('<svg', '<svg xmlns="http://www.w3.org/2000/svg"')
    const url = URL.createObjectURL(new Blob([markup], { type: 'image/svg+xml;charset=utf-8' }))
    const img = new Image()
    img.onload = () => {
      const canvas = document.createElement('canvas')
      canvas.width = Math.round(w * scale)
      canvas.height = Math.round(h * scale)
      const ctx = canvas.getContext('2d')
      if (!ctx) { URL.revokeObjectURL(url); return reject(new Error('No 2D canvas context')) }
      ctx.fillStyle = background
      ctx.fillRect(0, 0, canvas.width, canvas.height)
      ctx.drawImage(img, 0, 0, canvas.width, canvas.height)
      URL.revokeObjectURL(url)
      canvas.toBlob((blob) => (blob ? resolve(blob) : reject(new Error('PNG encoding failed'))), 'image/png')
    }
    img.onerror = () => { URL.revokeObjectURL(url); reject(new Error('The SVG could not be rasterised')) }
    img.src = url
  })
}

export function blobToBase64(blob: Blob): Promise<string> {
  return new Promise((resolve, reject) => {
    const reader = new FileReader()
    reader.onload = () => resolve(String(reader.result))
    reader.onerror = () => reject(new Error('Could not read the PNG'))
    reader.readAsDataURL(blob)
  })
}

export function downloadBlob(filename: string, blob: Blob): void {
  const url = URL.createObjectURL(blob)
  const a = document.createElement('a')
  a.href = url
  a.download = filename
  document.body.appendChild(a)
  a.click()
  document.body.removeChild(a)
  setTimeout(() => URL.revokeObjectURL(url), 1000)
}
