import { describe, it, expect } from 'vitest'
import { buildCrumbs } from './pathCrumbs'

const SHORTCUTS = [
  { name: 'Home', path: '/Users/pcahan' },
  { name: 'Documents', path: '/Users/pcahan/Documents' },
  { name: 'Downloads', path: '/Users/pcahan/Downloads' },
]

// The path this was written for, from a real session: 11 segments, 140 chars.
const DEEP =
  '/Users/pcahan/CahanLab Dropbox/Patrick Cahan/PC/PC/projects/SJD/data/spatial/xcell/h5ad_empty_removed/best_per_stage'

const labels = (t: { crumbs: { label: string }[] }) => t.crumbs.map((c) => c.label)

describe('buildCrumbs', () => {
  it('keeps the folder you are in, which is what scrolling used to hide', () => {
    const ls = labels(buildCrumbs(DEEP, SHORTCUTS))
    expect(ls[ls.length - 1]).toBe('best_per_stage')
  })

  it('keeps a way back out', () => {
    expect(labels(buildCrumbs(DEEP, SHORTCUTS))[0]).toBe('Home')
  })

  it('collapses the middle rather than growing', () => {
    const t = buildCrumbs(DEEP, SHORTCUTS)
    expect(t.crumbs).toHaveLength(4)
    expect(t.hidden.length).toBeGreaterThan(0)
    // Nothing is lost — every segment is either shown or reachable via the "…".
    expect(t.crumbs.length + t.hidden.length).toBe(12)
  })

  it('names a location by its shortcut instead of spelling out the home path', () => {
    const t = buildCrumbs('/Users/pcahan/Documents/atlas', SHORTCUTS)
    expect(labels(t)).toEqual(['Documents', 'atlas'])
    expect(t.crumbs[0].path).toBe('/Users/pcahan/Documents')
  })

  it('prefers the most specific shortcut', () => {
    // Both Home and Documents are prefixes; Documents says more.
    expect(labels(buildCrumbs('/Users/pcahan/Documents', SHORTCUTS))).toEqual(['Documents'])
  })

  it('does not match a shortcut on a partial folder name', () => {
    const t = buildCrumbs('/Users/pcahanson/data', [{ name: 'Home', path: '/Users/pcahan' }])
    expect(labels(t)[0]).toBe('/')
  })

  it('falls back to the root when no shortcut applies', () => {
    expect(labels(buildCrumbs('/mnt/scratch', SHORTCUTS))).toEqual(['/', 'mnt', 'scratch'])
  })

  it('leaves a short path alone', () => {
    const t = buildCrumbs('/Users/pcahan/data', SHORTCUTS)
    expect(labels(t)).toEqual(['Home', 'data'])
    expect(t.hidden).toEqual([])
  })

  it('handles the root itself', () => {
    expect(labels(buildCrumbs('/', SHORTCUTS))).toEqual(['/'])
  })

  it('has nothing to say about no path', () => {
    expect(buildCrumbs(null, SHORTCUTS)).toEqual({ crumbs: [], hidden: [] })
    expect(buildCrumbs('', SHORTCUTS)).toEqual({ crumbs: [], hidden: [] })
  })

  it('gives every crumb a path that actually leads there', () => {
    const t = buildCrumbs(DEEP, SHORTCUTS)
    for (const c of [...t.crumbs, ...t.hidden]) {
      expect(DEEP.startsWith(c.path)).toBe(true)
    }
    expect(t.crumbs[t.crumbs.length - 1].path).toBe(DEEP)
  })

  it('works without any shortcuts at all', () => {
    const t = buildCrumbs('/a/b/c/d/e/f', [])
    expect(labels(t)[0]).toBe('/')
    const l2 = labels(t)
    expect(l2[l2.length - 1]).toBe('f')
  })
})
