import { describe, it, expect } from 'vitest'
import { HIGHLIGHT_PALETTE, findGeneSetLayer, findGeneSetLayers, geneSetLayerKey, highlightButtonAction, highlightSkipReason, nextHighlightColor } from './highlightLayers'
import type { HighlightLayer } from '../store'

// The gene-set row's highlight toggle: it must find the layer it added (to
// show it as on and to remove it), and a new layer should not reuse a colour
// already on screen.

function geneLayer(id: string, label: string, genes: string[], color = '#22c55e'): HighlightLayer {
  return {
    id, color, intensity: 0.85,
    source: { kind: 'geneset', label, genes, values: [], min: 0, max: 1, thresholdMode: 'above', lo: 0.5, hi: 1 },
  }
}

function cellLayer(id: string, label: string, color = '#06b6d4'): HighlightLayer {
  return { id, color, intensity: 0.85, source: { kind: 'cellset', label, mask: new Uint8Array(0) } }
}

describe('findGeneSetLayer', () => {
  const set = { name: 'Wnt targets', genes: ['Axin2', 'Lef1', 'Sp5'] }

  it('finds the layer made from the same set', () => {
    const layers = [cellLayer('a', 'Wnt targets'), geneLayer('b', 'Wnt targets', ['Sp5', 'Axin2', 'Lef1'])]
    expect(findGeneSetLayer(layers, set)?.id).toBe('b')
  })

  it('does not match a same-named set with other genes', () => {
    expect(findGeneSetLayer([geneLayer('b', 'Wnt targets', ['Axin2'])], set)).toBeNull()
  })

  it('does not match a cell-set layer with the set name', () => {
    expect(findGeneSetLayer([cellLayer('a', 'Wnt targets')], set)).toBeNull()
  })

  it('does not match another set with the same genes', () => {
    expect(findGeneSetLayer([geneLayer('b', 'Other', ['Axin2', 'Lef1', 'Sp5'])], set)).toBeNull()
  })

  it('matches a set listing a gene twice', () => {
    // Imports do not dedupe; a length check would never match, and every
    // click would stack another layer.
    const dup = { name: 'Wnt targets', genes: ['Axin2', 'Axin2', 'Lef1', 'Sp5'] }
    expect(findGeneSetLayer([geneLayer('b', 'Wnt targets', ['Axin2', 'Axin2', 'Lef1', 'Sp5'])], dup)?.id).toBe('b')
    expect(findGeneSetLayer([geneLayer('b', 'Wnt targets', ['Axin2', 'Lef1', 'Sp5'])], dup)?.id).toBe('b')
  })

  it('finds every layer of the set, so a toggle can clear duplicates', () => {
    const layers = [geneLayer('a', 'Wnt targets', set.genes), cellLayer('x', 'y'), geneLayer('b', 'Wnt targets', set.genes)]
    expect(findGeneSetLayers(layers, set).map((l) => l.id)).toEqual(['a', 'b'])
  })

  it('finds nothing among no layers', () => {
    expect(findGeneSetLayers([], set)).toEqual([])
  })
})

describe('geneSetLayerKey', () => {
  it('ignores gene order and repeats but not the name', () => {
    expect(geneSetLayerKey({ name: 'A', genes: ['x', 'y', 'x'] })).toBe(geneSetLayerKey({ name: 'A', genes: ['y', 'x'] }))
    expect(geneSetLayerKey({ name: 'A', genes: ['x'] })).not.toBe(geneSetLayerKey({ name: 'B', genes: ['x'] }))
  })
})

describe('highlightSkipReason', () => {
  it('refuses a set with no gene in the dataset', () => {
    // The backend returns all-zero values with min = max = 0, and "above 0"
    // then highlights every cell.
    expect(highlightSkipReason({ genes: [], min: 0, max: 0 })).toMatch(/none of/i)
  })

  it('says so when the gene mask hid them all', () => {
    expect(highlightSkipReason({ genes: [], min: 0, max: 0, n_masked_excluded: 4 })).toMatch(/mask/i)
  })

  it('refuses a score that is the same in every cell', () => {
    expect(highlightSkipReason({ genes: ['a'], min: 0, max: 0 })).toMatch(/same/i)
  })

  it('accepts a score that varies', () => {
    expect(highlightSkipReason({ genes: ['a'], min: 0, max: 2.5 })).toBeNull()
  })
})

describe('nextHighlightColor', () => {
  it('starts at the first palette colour', () => {
    expect(nextHighlightColor([])).toBe(HIGHLIGHT_PALETTE[0])
  })

  it('skips colours already in use', () => {
    const layers = [geneLayer('a', 'x', ['g'], HIGHLIGHT_PALETTE[0]), cellLayer('b', 'y', HIGHLIGHT_PALETTE[2])]
    expect(nextHighlightColor(layers)).toBe(HIGHLIGHT_PALETTE[1])
  })

  it('reuses a freed colour after a layer is removed', () => {
    const layers = [geneLayer('b', 'y', ['g'], HIGHLIGHT_PALETTE[1])]
    expect(nextHighlightColor(layers)).toBe(HIGHLIGHT_PALETTE[0])
  })

  it('cycles by count once every colour is taken', () => {
    const layers = HIGHLIGHT_PALETTE.map((c, i) => geneLayer(String(i), String(i), ['g'], c))
    expect(nextHighlightColor(layers)).toBe(HIGHLIGHT_PALETTE[0])
  })

  it('compares colours case-insensitively', () => {
    const layers = [geneLayer('a', 'x', ['g'], HIGHLIGHT_PALETTE[0].toUpperCase())]
    expect(nextHighlightColor(layers)).toBe(HIGHLIGHT_PALETTE[1])
  })
})

describe('highlightButtonAction', () => {
  // The row's 🖍: add the layer and open its tuning strip; once shown, the
  // button opens and closes the strip, whose × removes the layer.
  it('adds a set that is not highlighted', () => {
    expect(highlightButtonAction(false, false)).toBe('add')
    expect(highlightButtonAction(false, true)).toBe('add')
  })

  it('opens the tuning strip of a highlighted set', () => {
    expect(highlightButtonAction(true, false)).toBe('open')
  })

  it('closes an open strip rather than removing the layer', () => {
    expect(highlightButtonAction(true, true)).toBe('close')
  })
})
