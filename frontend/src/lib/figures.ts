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
  cols: { label: string }[]
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

export type FigureData = EnrichmentHeatmapData | EnrichmentNetworkData | Record<string, unknown>

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
