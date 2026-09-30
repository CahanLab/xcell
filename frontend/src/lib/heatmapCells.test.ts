import { describe, it, expect } from 'vitest'
import { heatmapCellScope } from './heatmapCells'

// The cells the Heatmap tab draws: the config's own restriction (the subset a
// gene clustering ran on) narrowed by the cell mask, like every other view.

const mask = [true, false, true, true, false, true]

describe('heatmapCellScope', () => {
  it('is every cell with neither a restriction nor a mask', () => {
    expect(heatmapCellScope(null, null, null)).toEqual({ indices: null, subset: null })
  })

  it('is the masked cells under a mask', () => {
    expect(heatmapCellScope(null, mask, null)).toEqual({ indices: [0, 2, 3, 5], subset: null })
  })

  it('names the saved subset the mask is, for the figure record', () => {
    expect(heatmapCellScope(null, mask, 'chondro')).toEqual({ indices: [0, 2, 3, 5], subset: 'chondro' })
  })

  it('keeps the config restriction without a mask', () => {
    expect(heatmapCellScope([5, 1, 3], null, null)).toEqual({ indices: [5, 1, 3], subset: null })
  })

  it('intersects the restriction with the mask, in the restriction order', () => {
    expect(heatmapCellScope([5, 1, 3], mask, 'chondro')).toEqual({ indices: [5, 3], subset: null })
  })

  it('is empty when the mask excludes every restricted cell', () => {
    expect(heatmapCellScope([1, 4], mask, null)).toEqual({ indices: [], subset: null })
  })

  it('treats an empty restriction as none', () => {
    expect(heatmapCellScope([], null, null)).toEqual({ indices: null, subset: null })
  })
})
