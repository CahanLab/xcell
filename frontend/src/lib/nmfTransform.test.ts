import { describe, it, expect } from 'vitest'
import { defaultTransformFor, transformNoteFor } from './nmfTransform'

describe('defaultTransformFor', () => {
  it('normalizes and logs raw counts', () => {
    expect(defaultTransformFor('raw_counts')).toBe('log1p')
  })

  it('logs library-size-normalized values that are still linear', () => {
    expect(defaultTransformFor('normalized_linear')).toBe('log1p')
  })

  it('leaves already-log data alone', () => {
    // The trap this exists for: xcell's default log1p on top of a matrix that
    // is already on a log scale silently factorizes log(1 + log(1 + counts)).
    expect(defaultTransformFor('log_normalized')).toBe('none')
    expect(defaultTransformFor('log_transformed')).toBe('none')
  })

  it('leaves binary data alone', () => {
    expect(defaultTransformFor('binary')).toBe('none')
  })

  it('falls back to log1p when the scale could not be determined', () => {
    // Unknown is the common case for a hand-built layer; log1p matches what
    // every other xcell gene-expression path does, so it is the safe default.
    expect(defaultTransformFor('unknown')).toBe('log1p')
    expect(defaultTransformFor(undefined)).toBe('log1p')
  })

  it('leaves z-scored data alone rather than pretending it is fixable', () => {
    // NMF rejects negatives outright; transforming would not help and would
    // hide why. The run should fail with the backend's explanation.
    expect(defaultTransformFor('z_scored')).toBe('none')
  })
})

describe('transformNoteFor', () => {
  it('explains why an already-log matrix is not transformed again', () => {
    const note = transformNoteFor('log_normalized')
    expect(note).toBeTruthy()
    expect(note!.toLowerCase()).toContain('already')
  })

  it('warns that z-scored data will be rejected', () => {
    expect(transformNoteFor('z_scored')!.toLowerCase()).toContain('negative')
  })

  it('says nothing when the default needs no explanation', () => {
    expect(transformNoteFor('raw_counts')).toBeNull()
    expect(transformNoteFor(undefined)).toBeNull()
  })
})
