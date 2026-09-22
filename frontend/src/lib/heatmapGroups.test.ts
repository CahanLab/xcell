import { describe, it, expect } from 'vitest'
import { orderSelectedSets, groupLabel, folderSelectionState, restoreSelectedIds } from './heatmapGroups'

// A pathway is a folder of role sets in a fixed order (ligands, receptors,
// …). The heatmap should show a picked pathway in that order, and when sets
// from several pathways are mixed, each row group should say which pathway
// it belongs to — "Wnt (canonical) · ligands" — since every pathway has a
// "ligands" set.

type S = { id: string; name: string; genes: string[]; category: string; folderName?: string }
const set = (id: string, name: string, folderName?: string): S =>
  ({ id, name, genes: [id], category: 'manual', folderName })

const all: S[] = [
  set('w1', 'ligands', 'Wnt (canonical)'),
  set('w2', 'receptors', 'Wnt (canonical)'),
  set('w3', 'effectors', 'Wnt (canonical)'),
  set('b1', 'ligands', 'BMP'),
  set('b2', 'receptors', 'BMP'),
  set('m1', 'my markers'),
]

describe('orderSelectedSets', () => {
  it('keeps folder order and each folder\'s own set order whatever the click order', () => {
    const picked = new Set(['b2', 'w3', 'w1', 'b1'])
    expect(orderSelectedSets(all, picked).map((s) => s.id)).toEqual(['w1', 'w3', 'b1', 'b2'])
  })
  it('leaves loose sets where they are', () => {
    expect(orderSelectedSets(all, new Set(['m1', 'w2'])).map((s) => s.id)).toEqual(['w2', 'm1'])
  })
})

describe('groupLabel', () => {
  it('is just the role when every picked set comes from one folder', () => {
    const picked = orderSelectedSets(all, new Set(['w1', 'w2']))
    expect(picked.map((s) => groupLabel(s, picked))).toEqual(['ligands', 'receptors'])
  })
  it('carries the pathway when folders are mixed, so "ligands" is not ambiguous', () => {
    const picked = orderSelectedSets(all, new Set(['w1', 'b1']))
    expect(picked.map((s) => groupLabel(s, picked))).toEqual(['Wnt (canonical) · ligands', 'BMP · ligands'])
  })
  it('a loose set among folders keeps its own name', () => {
    const picked = orderSelectedSets(all, new Set(['w1', 'm1']))
    expect(picked.map((s) => groupLabel(s, picked))).toEqual(['Wnt (canonical) · ligands', 'my markers'])
  })
})

describe('folderSelectionState', () => {
  it('reports none, some or all of a folder\'s sets picked', () => {
    expect(folderSelectionState(all, 'Wnt (canonical)', new Set())).toBe('none')
    expect(folderSelectionState(all, 'Wnt (canonical)', new Set(['w1']))).toBe('some')
    expect(folderSelectionState(all, 'Wnt (canonical)', new Set(['w1', 'w2', 'w3']))).toBe('all')
  })
})

describe('restoreSelectedIds', () => {
  const sets = [set('w1', 'ligands', 'Wnt (canonical)'), set('b1', 'ligands', 'BMP'), set('m1', 'my markers')]
  it('restores by id when the applied config carries one', () => {
    const applied = [{ id: 'b1', name: 'BMP · ligands', genes: ['b1'] }]
    expect(Array.from(restoreSelectedIds(applied, sets))).toEqual(['b1'])
  })
  it('falls back to name and gene count for a config saved without ids', () => {
    const applied = [{ name: 'my markers', genes: ['m1'] }]
    expect(Array.from(restoreSelectedIds(applied, sets))).toEqual(['m1'])
  })
  it('does not guess between same-named sets without an id', () => {
    const applied = [{ name: 'ligands', genes: ['x'] }]
    expect(restoreSelectedIds(applied, sets).size).toBe(0)
  })
})
