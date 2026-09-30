import { describe, it, expect } from 'vitest'
import { HIGHLIGHT_PALETTE, findGeneSetLayer, nextHighlightColor } from './highlightLayers'
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

  it('cycles once every colour is taken', () => {
    const layers = HIGHLIGHT_PALETTE.map((c, i) => geneLayer(String(i), String(i), ['g'], c))
    expect(HIGHLIGHT_PALETTE).toContain(nextHighlightColor(layers))
  })
})
