import { describe, it, expect } from 'vitest'
import { loadedSlots, mirrorTargets, nextSlotKey, datasetLabel, slotAfterUnload, resizePanes, paneGrid, moveItem, activeSlotFrom } from './datasetSlots'

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

describe('nextSlotKey', () => {
  it('calls the first dataset primary', () => {
    expect(nextSlotKey([])).toBe('primary')
  })

  it('calls the second one secondary', () => {
    expect(nextSlotKey(['primary'])).toBe('secondary')
  })

  it('numbers the ones after that', () => {
    expect(nextSlotKey(['primary', 'secondary'])).toBe('slot3')
    expect(nextSlotKey(['primary', 'secondary', 'slot3'])).toBe('slot4')
  })

  // Otherwise load-unload-load climbs forever and the keys record nothing but
  // how many times you changed your mind.
  it('reuses a name that has been freed', () => {
    expect(nextSlotKey(['primary', 'slot3'])).toBe('secondary')
  })

  it('steps over a number already in use', () => {
    expect(nextSlotKey(['primary', 'secondary', 'slot4'])).toBe('slot3')
  })
})

describe('datasetLabel', () => {
  it('names a dataset after its file', () => {
    expect(datasetLabel('secondary', 'E11.5_hindlimbs_xcell_qced.h5ad'))
      .toBe('E11.5_hindlimbs_xcell_qced')
  })

  it('drops the directory a file came from', () => {
    expect(datasetLabel('primary', '/data/spatial/xcell/E11.5.h5ad')).toBe('E11.5')
  })

  it('strips the extensions xcell actually loads', () => {
    expect(datasetLabel('primary', 'sample.h5')).toBe('sample')
    expect(datasetLabel('primary', 'seurat_obj.rds')).toBe('seurat_obj')
  })

  it('leaves a 10x matrix directory name alone', () => {
    expect(datasetLabel('primary', 'GSM8155797_filtered_feature_bc_matrix'))
      .toBe('GSM8155797_filtered_feature_bc_matrix')
  })

  // Real filenames here carry version dots ('E11.5') that are part of the name,
  // so only a known extension is allowed to end one.
  it('keeps dots that belong to the name', () => {
    expect(datasetLabel('primary', 'E11.5_hindlimbs_102424.h5ad')).toBe('E11.5_hindlimbs_102424')
  })

  it('falls back to the slot when there is no file', () => {
    expect(datasetLabel('primary', undefined)).toBe('Primary')
    expect(datasetLabel('slot3', undefined)).toBe('Slot3')
  })

  it('lets a name the user typed win over the filename', () => {
    expect(datasetLabel('primary', 'GSM8155797_scMm_Shox2trac.h5', 'E11.5 rep2')).toBe('E11.5 rep2')
  })

  it('treats a name cleared back to nothing as no name at all', () => {
    expect(datasetLabel('primary', 'sample.h5ad', '')).toBe('sample')
    expect(datasetLabel('primary', 'sample.h5ad', '   ')).toBe('sample')
  })
})

describe('slotAfterUnload', () => {
  const order = ['primary', 'secondary', 'slot3']

  it('leaves the active dataset alone when another one is closed', () => {
    expect(slotAfterUnload('slot3', order, 'primary')).toBe('primary')
  })

  it('falls back to the dataset on the left', () => {
    expect(slotAfterUnload('slot3', order, 'slot3')).toBe('secondary')
  })

  it('falls forward when the first one is closed', () => {
    expect(slotAfterUnload('primary', order, 'primary')).toBe('secondary')
  })

  it('has nothing to fall back to when the last one goes', () => {
    expect(slotAfterUnload('primary', ['primary'], 'primary')).toBeNull()
  })
})

describe('resizePanes', () => {
  // Two 300px panes in a 600px row, equal weights.
  const two = [1, 1]

  it('gives what it takes: dragging right widens the left pane', () => {
    const [a, b] = resizePanes(two, 0, 60, 600)
    expect(a).toBeCloseTo(1.2)
    expect(b).toBeCloseTo(0.8)
  })

  it('leaves the total alone, so panes beyond the drag do not move', () => {
    const w = resizePanes([1, 1, 1], 0, 50, 900)
    expect(w[0] + w[1] + w[2]).toBeCloseTo(3)
    expect(w[2]).toBe(1)
  })

  it('drags the other way too', () => {
    const [a, b] = resizePanes(two, 0, -60, 600)
    expect(a).toBeCloseTo(0.8)
    expect(b).toBeCloseTo(1.2)
  })

  it('refuses to squeeze a pane out of existence', () => {
    // 120px minimum: from 300px the left pane can give up at most 180.
    const [a, b] = resizePanes(two, 0, -1000, 600, 120)
    expect(a).toBeCloseTo(0.4)   // 120/600 * 2
    expect(b).toBeCloseTo(1.6)   // 480/600 * 2
  })

  it('protects the neighbour just as hard', () => {
    const [a, b] = resizePanes(two, 0, 1000, 600, 120)
    expect(a).toBeCloseTo(1.6)
    expect(b).toBeCloseTo(0.4)
  })

  it('does nothing when nothing moved', () => {
    expect(resizePanes(two, 0, 0, 600)).toEqual(two)
  })
})


describe('paneGrid', () => {
  // Two and three datasets stay in one row — that is what a wide screen is for,
  // and it is what the side-by-side comparison has always looked like.
  it('keeps a few datasets in a single row', () => {
    expect(paneGrid(1)).toEqual({ rows: 1, cols: 1 })
    expect(paneGrid(2)).toEqual({ rows: 1, cols: 2 })
    expect(paneGrid(3)).toEqual({ rows: 1, cols: 3 })
  })

  // Past that a row makes every pane a sliver; a grid keeps them roughly square,
  // which is what spatial data wants.
  it('squares off once a row would be too thin', () => {
    expect(paneGrid(4)).toEqual({ rows: 2, cols: 2 })
    expect(paneGrid(6)).toEqual({ rows: 2, cols: 3 })
    expect(paneGrid(9)).toEqual({ rows: 3, cols: 3 })
  })

  it('leaves a gap rather than an uneven row when the count is awkward', () => {
    expect(paneGrid(5)).toEqual({ rows: 2, cols: 3 })
    expect(paneGrid(7)).toEqual({ rows: 3, cols: 3 })
  })

  it('always has room for every pane', () => {
    for (let n = 1; n <= 16; n++) {
      const { rows, cols } = paneGrid(n)
      expect(rows * cols).toBeGreaterThanOrEqual(n)
    }
  })

  it('has nothing to lay out for nothing', () => {
    expect(paneGrid(0)).toEqual({ rows: 0, cols: 0 })
  })
})

describe('moveItem', () => {
  const abc = ['a', 'b', 'c', 'd']

  it('drags a tab to the right', () => {
    expect(moveItem(abc, 0, 2)).toEqual(['b', 'c', 'a', 'd'])
  })

  it('drags a tab to the left', () => {
    expect(moveItem(abc, 3, 1)).toEqual(['a', 'd', 'b', 'c'])
  })

  it('leaves the order alone when a tab is dropped where it started', () => {
    expect(moveItem(abc, 1, 1)).toEqual(abc)
  })

  it('ignores a drop outside the strip', () => {
    expect(moveItem(abc, 0, 9)).toEqual(abc)
    expect(moveItem(abc, -1, 2)).toEqual(abc)
  })

  it('does not mutate the array it was given', () => {
    const original = [...abc]
    moveItem(abc, 0, 2)
    expect(abc).toEqual(original)
  })
})

// ---------------------------------------------------------------------------
// Reported 2026-08-25: after closing the dataset loaded first and refreshing,
// every request failed with "No data loaded for slot 'primary'" while two
// datasets sat healthy in the backend. activeSlot initialises to the literal
// 'primary', the restore never sets it, and appendDataset() omits ?dataset=
// for primary — so the whole app addressed a slot that was not there.
// ---------------------------------------------------------------------------

describe('activeSlotFrom', () => {
  it('keeps the active slot when it is actually loaded', () => {
    expect(activeSlotFrom('slot3', ['slot3', 'slot4'], ['slot3', 'slot4'])).toBe('slot3')
  })

  it('moves off a slot that is not loaded', () => {
    expect(activeSlotFrom('primary', ['slot3', 'slot4'], ['slot3', 'slot4'])).toBe('slot3')
  })

  it('respects the restored tab order when it has to choose', () => {
    // The user dragged slot4 to the front; landing on slot3 would put them on a
    // dataset that is not the one they left in the first position.
    expect(activeSlotFrom('primary', ['slot3', 'slot4'], ['slot4', 'slot3'])).toBe('slot4')
  })

  it('ignores an order naming datasets that are not loaded', () => {
    expect(activeSlotFrom('primary', ['slot4'], ['secondary', 'slot3', 'slot4'])).toBe('slot4')
  })

  it('leaves the active slot alone when nothing is loaded', () => {
    // Startup before any fetch resolves. Repointing at nothing helps no one,
    // and the empty state renders from the default dataset either way.
    expect(activeSlotFrom('primary', [], [])).toBe('primary')
  })

  it('does not move just because the active slot is not first', () => {
    expect(activeSlotFrom('slot4', ['slot3', 'slot4'], ['slot3', 'slot4'])).toBe('slot4')
  })
})
