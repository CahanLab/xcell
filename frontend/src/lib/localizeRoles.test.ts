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
    const roleOf = (r: ReturnType<typeof assignRoles>) =>
      ({ reference: r.referenceSlot, query: r.querySlot, problem: r.problem })
    const asPrimaryActive = assignRoles(LIMB.map((s) => ({ ...s, is_query: s.slot === 'primary' })))
    const asSecondaryActive = assignRoles(LIMB.map((s) => ({ ...s, is_query: s.slot === 'secondary' })))
    // Compare the decision, not the option lists: those echo back each
    // dataset's is_query flag, which is precisely the field this asserts is
    // never consulted.
    expect(roleOf(asPrimaryActive)).toEqual(roleOf(asSecondaryActive))
  })

  it('works with the roles reversed', () => {
    const r = assignRoles([
      slot('primary', 'scrna.h5ad', false),
      slot('secondary', 'visium.h5ad', true, 'spatial'),
    ])
    expect(r.referenceSlot).toBe('secondary')
    expect(r.querySlot).toBe('primary')
  })

  it('offers only the datasets that could actually be a reference', () => {
    const r = assignRoles(LIMB)
    expect(r.referenceOptions.map((o) => o.slot)).toEqual(['primary'])
  })

  it('honours an explicit reference when both have coordinates', () => {
    const both = [
      slot('primary', 'a.h5ad', true, 'spatial'),
      slot('secondary', 'b.h5ad', true, 'X_spatial'),
    ]
    const auto = assignRoles(both)
    expect(auto.referenceSlot).toBe('primary')
    expect(auto.querySlot).toBe('secondary')
    expect(auto.referenceOptions.map((o) => o.slot)).toEqual(['primary', 'secondary'])

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

// ---------------------------------------------------------------------------
// Four datasets — two spatial, two dissociated. Reported 2026-08-25: the modal
// pinned the reference to the first ST and offered no way to reach the second
// pair. Roles were derived for a two-slot world and never revisited when
// N datasets shipped.
// ---------------------------------------------------------------------------

const TWO_PAIRS = [
  slot('primary', 'ST_A.h5ad', true, 'X_spatial'),
  slot('secondary', 'SC_A.h5ad', false),
  slot('slot3', 'ST_B.h5ad', true, 'X_spatial'),
  slot('slot4', 'SC_B.h5ad', false),
]

describe('assignRoles with more than two datasets', () => {
  it('can name the second spatial dataset as the reference', () => {
    const r = assignRoles(TWO_PAIRS, 'slot3')
    expect(r.referenceSlot).toBe('slot3')
  })

  it('can name the query, not just the reference', () => {
    const r = assignRoles(TWO_PAIRS, 'slot3', 'slot4')
    expect(r.referenceSlot).toBe('slot3')
    expect(r.querySlot).toBe('slot4')
    expect(r.problem).toBeNull()
  })

  it('offers every spatial dataset as a reference', () => {
    expect(assignRoles(TWO_PAIRS).referenceOptions.map((o) => o.slot))
      .toEqual(['primary', 'slot3'])
  })

  it('offers every other dataset as a query, and never the reference itself', () => {
    const r = assignRoles(TWO_PAIRS, 'slot3')
    expect(r.queryOptions.map((o) => o.slot)).toEqual(['primary', 'secondary', 'slot4'])
    expect(r.queryOptions.some((o) => o.slot === 'slot3')).toBe(false)
  })

  it('defaults the query to dissociated cells rather than the other tissue', () => {
    // Loading ST, ST, SC, SC used to make the *other* spatial dataset the
    // query — the one arrangement where the default is never what was meant.
    const stFirst = [
      slot('primary', 'ST_A.h5ad', true, 'X_spatial'),
      slot('secondary', 'ST_B.h5ad', true, 'X_spatial'),
      slot('slot3', 'SC_A.h5ad', false),
      slot('slot4', 'SC_B.h5ad', false),
    ]
    const r = assignRoles(stFirst)
    expect(r.referenceSlot).toBe('primary')
    expect(r.querySlot).toBe('slot3')
  })

  it('still allows tissue onto tissue when that is what was asked for', () => {
    const r = assignRoles(TWO_PAIRS, 'primary', 'slot3')
    expect(r.referenceSlot).toBe('primary')
    expect(r.querySlot).toBe('slot3')
  })

  it('ignores a query override naming the chosen reference', () => {
    const r = assignRoles(TWO_PAIRS, 'slot3', 'slot3')
    expect(r.referenceSlot).toBe('slot3')
    expect(r.querySlot).not.toBe('slot3')
    expect(r.problem).toBeNull()
  })

  it('treats an empty override as no preference, not as a slot', () => {
    // A cleared <select> yields '', which must not be mistaken for a choice.
    const r = assignRoles(TWO_PAIRS, '', '')
    expect(r.referenceSlot).toBe('primary')
    expect(r.querySlot).toBe('secondary')
  })

  it('ignores a query override naming a dataset that is not loaded', () => {
    // Falls back to the default — the first dissociated dataset that is not
    // the reference. Nothing pairs ST_B with SC_B; they are just four slots.
    const r = assignRoles(TWO_PAIRS, 'slot3', 'slot99')
    expect(r.querySlot).toBe('secondary')
  })
})
