import { describe, it, expect } from 'vitest'
import { flattenGeneSetsForExport } from './exportGeneSets'
import { createDefaultCategories } from '../store'

describe('flattenGeneSetsForExport', () => {
  it('lists every set with its category and folder, keeping down genes', () => {
    const c = createDefaultCategories()
    c.manual.geneSets = [{ id: 'a', name: 'chondro', genes: ['Sox9'] }]
    c.manual.folders = [{
      id: 'f', name: 'Wnt', expanded: true, createdAt: '',
      geneSets: [{ id: 'b', name: 'targets', genes: ['Axin2'], genesDown: ['Dkk1'] }],
    }]
    expect(flattenGeneSetsForExport(c)).toEqual([
      { name: 'chondro', genes: ['Sox9'], category: 'Manual' },
      { name: 'targets', genes: ['Axin2'], genesDown: ['Dkk1'], category: 'Manual', folder: 'Wnt' },
    ])
  })

  it('is empty for a fresh set of categories', () => {
    expect(flattenGeneSetsForExport(createDefaultCategories())).toEqual([])
  })
})
