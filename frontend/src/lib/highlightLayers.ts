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

/** Every gene-set layer made from this set: same label and the same genes,
 *  as sets (imports can list a gene twice). Duplicates are possible — the
 *  Color tab's "+ Gene set" does not dedupe — so a toggle removes them all.
 *  A set edited since has other genes, so it reads as not shown. */
export function findGeneSetLayers(
  layers: readonly HighlightLayer[],
  set: { name: string; genes: readonly string[] },
): HighlightLayer[] {
  // Rows call this on every highlightLayers change (a threshold drag is one
  // per tick), so do no per-row work unless a layer could match.
  const candidates = layers.filter((l) => l.source.kind === 'geneset' && l.source.label === set.name)
  if (candidates.length === 0) return []
  const want = new Set(set.genes)
  return candidates.filter((l) => {
    const have = new Set(l.source.kind === 'geneset' ? l.source.genes : [])
    if (have.size !== want.size) return false
    for (const g of have) if (!want.has(g)) return false
    return true
  })
}

export function findGeneSetLayer(
  layers: readonly HighlightLayer[],
  set: { name: string; genes: readonly string[] },
): HighlightLayer | null {
  return findGeneSetLayers(layers, set)[0] ?? null
}

/** Identity of a set for "an add is already in flight": name + genes as a set. */
export function geneSetLayerKey(set: { name: string; genes: readonly string[] }): string {
  return `${set.name}\u0000${[...new Set(set.genes)].sort().join('\u0001')}`
}

/** Why a gene-set score should not become a highlight layer, or null.
 *  An "above" threshold on a score that is the same in every cell passes
 *  every cell — including the all-zero score the backend returns when none of
 *  the set's genes are in the dataset — and tints the whole plot. */
export function highlightSkipReason(data: {
  genes?: readonly string[]
  min: number
  max: number
  n_masked_excluded?: number
}): string | null {
  if ((data.genes?.length ?? 0) === 0) {
    return (data.n_masked_excluded ?? 0) > 0
      ? 'Every gene in this set is hidden by the gene mask, so there is nothing to highlight.'
      : 'None of this set\u2019s genes are in this dataset, so there is nothing to highlight.'
  }
  if (!(data.max > data.min)) {
    return 'This set\u2019s score is the same in every cell, so there is nothing to highlight.'
  }
  return null
}
