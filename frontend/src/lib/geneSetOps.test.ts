import { describe, it, expect } from 'vitest'
import { sortGenes, sortGeneSetInCategory } from './geneSetOps'

describe('sortGenes', () => {
  it('orders case-insensitively so lowercase names do not sort last', () => {
    expect(sortGenes(['Sox9', 'actb', 'Meis1'])).toEqual(['actb', 'Meis1', 'Sox9'])
  })

  it('orders embedded numbers numerically, not lexicographically', () => {
    expect(sortGenes(['Hoxd10', 'Hoxd2', 'Hoxd1', 'Hoxd13'])).toEqual(
      ['Hoxd1', 'Hoxd2', 'Hoxd10', 'Hoxd13'],
    )
  })

  it('preserves membership exactly — a reorder, never a filter', () => {
    const genes = ['Shh', 'Gli3', 'Shh', 'Ptch1']
    const sorted = sortGenes(genes)
    expect(sorted).toHaveLength(genes.length)
    expect([...sorted].sort()).toEqual([...genes].sort())
  })

  it('does not mutate the input array', () => {
    const genes = ['Sox9', 'Actb']
    sortGenes(genes)
    expect(genes).toEqual(['Sox9', 'Actb'])
  })
})

describe('sortGeneSetInCategory', () => {
  const category = () => ({
    geneSets: [
      { id: 'a', genes: ['Sox9', 'Actb'] },
      { id: 'b', genes: ['Wnt5a', 'Bmp4'] },
    ],
    folders: [
      { id: 'f1', geneSets: [{ id: 'c', genes: ['Hoxd13', 'Hoxd9'], genesDown: ['Meis2', 'Irx3'] }] },
    ],
  })

  it('sorts a gene set held directly by the category', () => {
    const out = sortGeneSetInCategory(category(), 'a')
    expect(out.geneSets[0].genes).toEqual(['Actb', 'Sox9'])
  })

  it('sorts a gene set nested inside a folder', () => {
    const out = sortGeneSetInCategory(category(), 'c')
    expect(out.folders[0].geneSets[0].genes).toEqual(['Hoxd9', 'Hoxd13'])
  })

  it('sorts the DOWN list too, so a directional set stays coherent', () => {
    const out = sortGeneSetInCategory(category(), 'c')
    expect(out.folders[0].geneSets[0].genesDown).toEqual(['Irx3', 'Meis2'])
  })

  it('leaves sibling sets referentially untouched so React skips their re-render', () => {
    const before = category()
    const out = sortGeneSetInCategory(before, 'a')
    expect(out.geneSets[1]).toBe(before.geneSets[1])
  })

  it('returns the category unchanged when the id matches nothing', () => {
    const before = category()
    expect(sortGeneSetInCategory(before, 'nope')).toBe(before)
  })
})
