/**
 * Highlight-overlay helpers shared by the Color tab's Highlight section and
 * the gene-set row's one-click toggle, so both add layers the same way.
 */
import type { HighlightLayer } from '../store'

export const HIGHLIGHT_PALETTE = ['#22c55e', '#06b6d4', '#f59e0b', '#ec4899', '#a855f7', '#84cc16', '#ef4444', '#0ea5e9']

/** Settings a new gene-set layer starts with (the Color tab's defaults). */
export const GENE_SET_LAYER_DEFAULTS = { intensity: 0.85, thresholdMode: 'above' as const }

/** The first palette colour no layer uses; by count once all are taken.
 *  Counting alone repeats a colour still on screen after a removal. */
export function nextHighlightColor(layers: readonly HighlightLayer[]): string {
  const used = new Set(layers.map((l) => l.color.toLowerCase()))
  return HIGHLIGHT_PALETTE.find((c) => !used.has(c)) ?? HIGHLIGHT_PALETTE[layers.length % HIGHLIGHT_PALETTE.length]
}

/** The gene-set layer made from this set: same label and the same genes in
 *  any order. A set edited since has other genes, so it reads as not shown. */
export function findGeneSetLayer(
  layers: readonly HighlightLayer[],
  set: { name: string; genes: readonly string[] },
): HighlightLayer | null {
  const want = new Set(set.genes)
  for (const l of layers) {
    if (l.source.kind !== 'geneset' || l.source.label !== set.name) continue
    const genes = l.source.genes
    if (genes.length === want.size && genes.every((g) => want.has(g))) return l
  }
  return null
}
