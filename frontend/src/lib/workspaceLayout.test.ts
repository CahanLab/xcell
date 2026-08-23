import { describe, it, expect } from 'vitest'
import { decodeWorkspace, encodeWorkspace, type Workspace } from './workspaceLayout'

const full: Workspace = {
  order: ['primary', 'slot3', 'secondary'],
  names: { slot3: 'E11.5 rep2' },
  cols: [1.4, 0.6],
  rows: [1],
  tiled: true,
}

describe('encodeWorkspace / decodeWorkspace', () => {
  it('brings back what it was given', () => {
    expect(decodeWorkspace(JSON.stringify(encodeWorkspace(full)))).toEqual(full)
  })

  // Everything below is a stored value that has to be survivable: the workspace
  // is a convenience, and none of it is worth a blank screen on startup.
  it('has an answer for nothing stored yet', () => {
    expect(decodeWorkspace(null)).toBeNull()
  })

  it('shrugs off a value that is not JSON at all', () => {
    expect(decodeWorkspace('{oh no')).toBeNull()
  })

  it('shrugs off JSON that is not a workspace', () => {
    expect(decodeWorkspace('42')).toBeNull()
    expect(decodeWorkspace('[1,2,3]')).toBeNull()
  })

  it('drops a layout written by a version that did not agree with this one', () => {
    expect(decodeWorkspace(JSON.stringify({ ...encodeWorkspace(full), version: 99 }))).toBeNull()
  })

  it('keeps the parts that survived and defaults the rest', () => {
    const partial = JSON.stringify({ version: 1, order: ['primary', 'secondary'] })
    expect(decodeWorkspace(partial)).toEqual({
      order: ['primary', 'secondary'], names: {}, cols: [], rows: [], tiled: false,
    })
  })

  it('only comes back tiled when it was stored that way', () => {
    const single = JSON.stringify({ ...encodeWorkspace(full), tiled: 'yes please' })
    expect(decodeWorkspace(single)?.tiled).toBe(false)
  })

  it('refuses track weights that would collapse a pane', () => {
    const bad = JSON.stringify({ ...encodeWorkspace(full), cols: [1, 0, -3, NaN] })
    expect(decodeWorkspace(bad)?.cols).toEqual([])
  })

  it('keeps only the entries that are the shape they claim to be', () => {
    const mixed = JSON.stringify({
      ...encodeWorkspace(full),
      order: ['primary', 7, null, 'secondary'],
      names: { primary: 'ok', secondary: 12 },
    })
    const out = decodeWorkspace(mixed)
    expect(out?.order).toEqual(['primary', 'secondary'])
    expect(out?.names).toEqual({ primary: 'ok' })
  })
})
