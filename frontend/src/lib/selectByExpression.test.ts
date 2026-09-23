import { describe, it, expect } from 'vitest'
import { runSignature, isOutcomeCurrent, type RunInputs } from './selectByExpression'

const base: RunInputs = {
  mode: 'above',
  lo: 1.5,
  hi: 3,
  action: 'labelCells',
  labelContext: 'all',
  annotationName: 'Proximal_above',
  highLabel: 'high',
  lowLabel: 'low',
}

const proximal = { type: 'geneSet', name: 'Proximal', genes: ['Meis1'] }
const distal = { type: 'geneSet', name: 'Distal', genes: ['Hoxd13'] }

describe('runSignature', () => {
  it('is stable for the same inputs', () => {
    expect(runSignature(base)).toBe(runSignature({ ...base }))
  })

  it.each([
    ['mode', { mode: 'below' as const }],
    ['threshold', { lo: 2.0 }],
    ['upper threshold', { hi: 4.0 }],
    ['action', { action: 'updateSelection' as const }],
    ['label context', { labelContext: 'selection' as const }],
    ['annotation name', { annotationName: 'Proximal_v2' }],
    ['high label', { highLabel: 'Meis1+' }],
    ['low label', { lowLabel: 'Meis1-' }],
  ])('changes when the %s changes', (_what, patch) => {
    expect(runSignature({ ...base, ...patch })).not.toBe(runSignature(base))
  })
})

describe('isOutcomeCurrent', () => {
  const sig = runSignature(base)

  it('holds while the source and the inputs are unchanged', () => {
    expect(isOutcomeCurrent({ source: proximal, signature: sig }, proximal, sig)).toBe(true)
  })

  it('goes stale for a different gene set', () => {
    // The bug this exists for: the modal kept showing the previous set's
    // "Labeled N cells" footer, and with it only Open Diff Exp / Close.
    expect(isOutcomeCurrent({ source: proximal, signature: sig }, distal, sig)).toBe(false)
  })

  it('goes stale when the same gene set is reopened', () => {
    // Every "Select cells…" click builds a fresh source object, so a second
    // run on the same set must start from a clean slate too.
    const reopened = { ...proximal }
    expect(isOutcomeCurrent({ source: proximal, signature: sig }, reopened, sig)).toBe(false)
  })

  it('goes stale when the threshold moves under the same source', () => {
    const moved = runSignature({ ...base, lo: 2.5 })
    expect(isOutcomeCurrent({ source: proximal, signature: sig }, proximal, moved)).toBe(false)
  })

  it('is not current when nothing has been applied yet', () => {
    expect(isOutcomeCurrent(null, proximal, sig)).toBe(false)
  })
})
