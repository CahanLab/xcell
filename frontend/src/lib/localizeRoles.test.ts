import { describe, it, expect } from 'vitest'
import { assignRoles, type RefSlot } from './localizeRoles'

const slot = (
  s: string, filename: string, hasSpatial: boolean, spatialKey: string | null = null,
): RefSlot => ({
  slot: s, filename, n_cells: 1000, n_genes: 12000,
  has_spatial: hasSpatial, spatial_key: spatialKey, is_query: false,
})

// The configuration this bug was reported against: the spatial reference sits
// in primary (the slot the UI treats as active), the dissociated scRNA-seq in
// secondary.
const LIMB = [
  slot('primary', 'E11.5_best_102424.h5ad', true, 'X_spatial'),
  slot('secondary', 'GSM4227224_E11_112125.h5ad', false),
]

describe('assignRoles', () => {
  it('makes the dataset holding coordinates the reference, whichever slot it is in', () => {
    const r = assignRoles(LIMB)
    expect(r.referenceSlot).toBe('primary')
    expect(r.querySlot).toBe('secondary')
    expect(r.problem).toBeNull()
  })

  it('does not depend on which slot is active', () => {
    // The old behaviour tied the query to the active slot, so the answer moved
    // when the user switched datasets. Roles come from the data now.
    const asPrimaryActive = assignRoles(LIMB.map((s) => ({ ...s, is_query: s.slot === 'primary' })))
    const asSecondaryActive = assignRoles(LIMB.map((s) => ({ ...s, is_query: s.slot === 'secondary' })))
    expect(asPrimaryActive).toEqual(asSecondaryActive)
  })

  it('works with the roles reversed', () => {
    const r = assignRoles([
      slot('primary', 'scrna.h5ad', false),
      slot('secondary', 'visium.h5ad', true, 'spatial'),
    ])
    expect(r.referenceSlot).toBe('secondary')
    expect(r.querySlot).toBe('primary')
  })

  it('is not swappable when only one dataset has coordinates', () => {
    expect(assignRoles(LIMB).swappable).toBe(false)
  })

  it('is swappable when both have coordinates, and honours an explicit choice', () => {
    const both = [
      slot('primary', 'a.h5ad', true, 'spatial'),
      slot('secondary', 'b.h5ad', true, 'X_spatial'),
    ]
    const auto = assignRoles(both)
    expect(auto.swappable).toBe(true)
    expect(auto.referenceSlot).toBe('primary')
    expect(auto.querySlot).toBe('secondary')

    const forced = assignRoles(both, 'secondary')
    expect(forced.referenceSlot).toBe('secondary')
    expect(forced.querySlot).toBe('primary')
  })

  it('explains that no dataset has coordinates when that is actually true', () => {
    const r = assignRoles([
      slot('primary', 'a.h5ad', false),
      slot('secondary', 'b.h5ad', false),
    ])
    expect(r.referenceSlot).toBeNull()
    expect(r.problem).toMatch(/no .*spatial coordinates/i)
  })

  it('asks for a second dataset rather than blaming coordinates', () => {
    // One slot loaded, and it is the spatial one. The old message claimed
    // nothing had coordinates, which was the opposite of the truth.
    const r = assignRoles([slot('primary', 'E11.5_best_102424.h5ad', true, 'X_spatial')])
    expect(r.referenceSlot).toBeNull()
    expect(r.problem).toMatch(/second dataset|another dataset/i)
    expect(r.problem).not.toMatch(/no .*spatial coordinates/i)
    // and it should name the one it found, so the user knows it was seen
    expect(r.problem).toContain('E11.5_best_102424.h5ad')
  })

  it('handles an empty list without inventing roles', () => {
    const r = assignRoles([])
    expect(r.referenceSlot).toBeNull()
    expect(r.querySlot).toBeNull()
    expect(r.problem).toBeTruthy()
  })

  it('ignores an override naming a slot with no coordinates', () => {
    const r = assignRoles(LIMB, 'secondary')
    expect(r.referenceSlot).toBe('primary')
  })
})
