import { describe, it, expect } from 'vitest'
import { mergeHydratedCategories } from './geneSetHydration'

const defaults = () => ({
  manual: { type: 'manual', name: 'Manual', expanded: true, folders: [], geneSets: [] },
  spatial: { type: 'spatial', name: 'Spatially Variable', expanded: true, folders: [], geneSets: [] },
})

const stored = { type: 'manual', name: 'Manual', expanded: true, folders: [], geneSets: [{ id: 'a', name: 'S', genes: ['Sox9'] }] }

describe('mergeHydratedCategories', () => {
  it('keeps a stored category', () => {
    const out = mergeHydratedCategories(defaults(), { manual: stored })
    expect(out.manual.geneSets).toHaveLength(1)
  })

  it('fills in a category the stored payload has never heard of', () => {
    // The upgrade case: sets saved before `spatial` existed must not leave the
    // panel reading properties of undefined.
    const out = mergeHydratedCategories(defaults(), { manual: stored })
    expect(out.spatial).toBeDefined()
    expect(out.spatial.geneSets).toEqual([])
  })

  it('drops a key the current version does not know', () => {
    const out = mergeHydratedCategories(defaults(), { manual: stored, from_the_future: stored })
    expect(Object.keys(out).sort()).toEqual(['manual', 'spatial'])
  })

  it('repairs a stored category missing its arrays rather than dropping its content', () => {
    const out = mergeHydratedCategories(defaults(), { manual: { name: 'Manual' } })
    expect(out.manual.geneSets).toEqual([])
    expect(out.manual.folders).toEqual([])
    expect(out.manual.name).toBe('Manual')
  })

  it('ignores an entry that is not an object at all', () => {
    const out = mergeHydratedCategories(defaults(), { manual: 'nonsense' })
    expect(out.manual.geneSets).toEqual([])
  })

  it('returns the defaults untouched when the payload is not a mapping', () => {
    expect(mergeHydratedCategories(defaults(), null)).toEqual(defaults())
    expect(mergeHydratedCategories(defaults(), [1, 2])).toEqual(defaults())
  })
})
