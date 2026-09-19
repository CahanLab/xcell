import { describe, it, expect } from 'vitest'
import { defaultGraphKey, NeighborGraphInfo } from './graphChoice'

const expression: NeighborGraphInfo = {
  key: 'connectivities', label: 'Expression neighbors', n_edges: 12000, suffix: '',
}
const spatial: NeighborGraphInfo = {
  key: 'spatial_connectivities', label: 'Spatial neighbors', n_edges: 8000, suffix: 'spatial',
}

describe('defaultGraphKey', () => {
  it('preselects the expression kNN graph when nothing is chosen', () => {
    expect(defaultGraphKey([spatial, expression], '')).toBe('connectivities')
  })

  it('never overrides an existing choice', () => {
    expect(defaultGraphKey([spatial, expression], 'spatial_connectivities')).toBeNull()
    expect(defaultGraphKey([expression], 'connectivities')).toBeNull()
  })

  it('suggests nothing when only non-expression graphs exist', () => {
    // A spatial-only dataset must keep its implicit-default option: silently
    // clustering the spatial graph would answer a different question.
    expect(defaultGraphKey([spatial], '')).toBeNull()
  })

  it("prefers a subset's own graph when it exists, else the expression graph", () => {
    const sub = { key: 'chondro_connectivities' }
    expect(defaultGraphKey([expression, sub], '', 'chondro_connectivities')).toBe('chondro_connectivities')
    expect(defaultGraphKey([expression], '', 'chondro_connectivities')).toBe('connectivities')
    expect(defaultGraphKey([expression, sub], 'connectivities', 'chondro_connectivities')).toBeNull()
  })

  it('suggests nothing when no graphs exist', () => {
    expect(defaultGraphKey([], '')).toBeNull()
  })
})
