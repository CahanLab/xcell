import { describe, it, expect } from 'vitest'
import { loadedSlots, mirrorTargets } from './datasetSlots'

const loaded = { schema: { n_cells: 10 } }
const empty = { schema: null }

describe('loadedSlots', () => {
  it('names the slots that actually hold a dataset', () => {
    expect(loadedSlots({ primary: loaded, secondary: empty })).toEqual(['primary'])
  })

  it('keeps them in the order the slots were created', () => {
    expect(loadedSlots({ primary: loaded, secondary: loaded, tertiary: loaded }))
      .toEqual(['primary', 'secondary', 'tertiary'])
  })

  it('reports nothing when no dataset is loaded', () => {
    expect(loadedSlots({ primary: empty })).toEqual([])
  })
})

describe('mirrorTargets', () => {
  it('mirrors onto the one other dataset', () => {
    expect(mirrorTargets('primary', { primary: loaded, secondary: loaded }))
      .toEqual(['secondary'])
  })

  it('never mirrors a dataset onto itself', () => {
    expect(mirrorTargets('secondary', { primary: loaded, secondary: loaded }))
      .toEqual(['primary'])
  })

  // The reason otherSlot() had to go: "the one that isn't this one" is not a
  // question with an answer once a third dataset is loaded.
  it('mirrors onto every other dataset, not just one of them', () => {
    expect(mirrorTargets('primary', { primary: loaded, secondary: loaded, tertiary: loaded }))
      .toEqual(['secondary', 'tertiary'])
  })

  it('does not mirror onto a slot with nothing in it', () => {
    expect(mirrorTargets('primary', { primary: loaded, secondary: empty })).toEqual([])
  })

  it('has nothing to mirror onto when only one dataset is loaded', () => {
    expect(mirrorTargets('primary', { primary: loaded })).toEqual([])
  })
})
