import { describe, it, expect } from 'vitest'
import {
  loadedSlots,
  mirrorTargets,
  nextSlotKey,
  datasetLabel,
  slotAfterUnload,
  resizePanes,
} from './datasetSlots'

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
