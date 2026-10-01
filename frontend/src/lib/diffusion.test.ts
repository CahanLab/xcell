import { describe, it, expect } from 'vitest'
import {
  isDiffmapKey, diffmapAxisLabel, defaultDiffmapDims, dptOutputName, dptParams, dptBlocker,
  type DptInputs,
} from './diffusion'

describe('diffusion map keys', () => {
  it('recognises diffusion maps by name', () => {
    expect(isDiffmapKey('X_diffmap')).toBe(true)
    expect(isDiffmapKey('X_diffmap_spatial')).toBe(true)
    expect(isDiffmapKey('X_diffmap_chondro_spatial')).toBe(true)
    expect(isDiffmapKey('X_diffmapper')).toBe(false)
    expect(isDiffmapKey('X_umap')).toBe(false)
  })

  it("labels columns with scanpy's numbering, so DC1 is X_diffmap[:, 1]", () => {
    expect(diffmapAxisLabel(0)).toBe('DC0')
    expect(diffmapAxisLabel(1)).toBe('DC1')
  })

  it('opens a map on DC1 × DC2, past the stationary column', () => {
    expect(defaultDiffmapDims(15)).toEqual({ x: 1, y: 2 })
    expect(defaultDiffmapDims(3)).toEqual({ x: 1, y: 2 })
  })

  it('leaves a map without a third column alone', () => {
    // The backend clamps out-of-range dims, and useEmbedding would refetch
    // forever on the mismatch.
    expect(defaultDiffmapDims(2)).toBeNull()
    expect(defaultDiffmapDims(0)).toBeNull()
  })

  it('names a pseudotime after its map, like the backend', () => {
    expect(dptOutputName('X_diffmap')).toBe('dpt_pseudotime')
    expect(dptOutputName('X_diffmap_spatial')).toBe('dpt_pseudotime_spatial')
    expect(dptOutputName('X_diffmap_chondro')).toBe('dpt_pseudotime_chondro')
    expect(dptOutputName('X_custom')).toBe('dpt_pseudotime_custom')
    expect(dptOutputName('mine')).toBe('dpt_pseudotime_mine')
  })
})

const base: DptInputs = {
  diffmapKey: 'X_diffmap',
  nDcs: '10',
  rootMode: 'potency',
  potencyColumn: 'stemfinder',
  potencyDirection: 'min',
  column: '',
  groupValue: '',
  component: '1',
  selection: [],
  keyAdded: '',
}

describe('dptParams', () => {
  it('roots the stemFinder preset at the lowest stemfinder cell', () => {
    expect(dptParams(base)).toEqual({
      diffmap_key: 'X_diffmap', n_dcs: 10, root_mode: 'obs_min', root_column: 'stemfinder',
    })
  })

  it('roots a diffOmeter preset at the highest value', () => {
    expect(dptParams({ ...base, potencyColumn: 'diffometer', potencyDirection: 'max' }))
      .toMatchObject({ root_mode: 'obs_max', root_column: 'diffometer' })
  })

  it('sends the chosen column for lowest / highest', () => {
    expect(dptParams({ ...base, rootMode: 'obs_max', column: 'potency' }))
      .toMatchObject({ root_mode: 'obs_max', root_column: 'potency' })
  })

  it('sends a group as its column and value', () => {
    expect(dptParams({ ...base, rootMode: 'group', column: 'stage', groupValue: 'early' }))
      .toMatchObject({ root_mode: 'group', root_column: 'stage', root_value: 'early' })
  })

  it('sends the selection as root cells', () => {
    const p = dptParams({ ...base, rootMode: 'cells', selection: [3, 4] })
    expect(p).toMatchObject({ root_mode: 'cells', root_cells: [3, 4] })
    expect(p).not.toHaveProperty('root_column')
  })

  it('sends a component tip', () => {
    expect(dptParams({ ...base, rootMode: 'dc_max', component: '2' }))
      .toMatchObject({ root_mode: 'dc_max', root_component: 2 })
  })

  it('sends a custom output name, trimmed, and nothing when blank', () => {
    expect(dptParams({ ...base, keyAdded: ' pt_a ' }).key_added).toBe('pt_a')
    expect(dptParams(base)).not.toHaveProperty('key_added')
  })

  it('leaves n_dcs to the backend when it is not a number', () => {
    expect(dptParams({ ...base, nDcs: '' })).not.toHaveProperty('n_dcs')
  })
})

describe('dptBlocker', () => {
  it('needs a diffusion map', () => {
    expect(dptBlocker({ ...base, diffmapKey: '' })).toMatch(/Diffusion map/)
  })

  it('needs what each root mode reads', () => {
    expect(dptBlocker({ ...base, potencyColumn: '' })).toMatch(/stemFinder/)
    expect(dptBlocker({ ...base, rootMode: 'obs_min', column: '' })).toMatch(/column/)
    expect(dptBlocker({ ...base, rootMode: 'group', column: 'stage', groupValue: '' })).toMatch(/group/)
    expect(dptBlocker({ ...base, rootMode: 'cells', selection: [] })).toMatch(/[Ss]elect/)
    expect(dptBlocker({ ...base, rootMode: 'dc_min', component: 'x' })).toMatch(/component/)
  })

  it('is null when it can run', () => {
    expect(dptBlocker(base)).toBeNull()
    expect(dptBlocker({ ...base, rootMode: 'cells', selection: [1] })).toBeNull()
  })
})
