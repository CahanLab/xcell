import { describe, it, expect } from 'vitest'
import { umPerUnitForSlot, expressionLegendTitle } from './datasetPanes'
import { createDefaultDatasetState, type DatasetState, type EmbeddingData } from '../store'

function dsWith(over: Partial<DatasetState>): DatasetState {
  return { ...createDefaultDatasetState(), ...over }
}

function shown(name: string): EmbeddingData {
  return { name, coordinates: [[0, 0]] }
}

const SPATIAL = { spatialKey: 'X_spatial', umPerUnit: 2.5 }

describe('umPerUnitForSlot', () => {
  it('gives a spatial plot its physical scale', () => {
    const ds = dsWith({ spatialScale: SPATIAL, embedding: shown('X_spatial') })
    expect(umPerUnitForSlot(ds)).toBe(2.5)
  })

  it('refuses to put a micron scale on a UMAP', () => {
    const ds = dsWith({ spatialScale: SPATIAL, embedding: shown('X_umap') })
    expect(umPerUnitForSlot(ds)).toBeNull()
  })

  it('stays silent when the dataset has no spatial identity', () => {
    expect(umPerUnitForSlot(dsWith({ embedding: shown('X_spatial') }))).toBeNull()
  })

  it('stays silent when the scale is unknown', () => {
    const ds = dsWith({
      spatialScale: { spatialKey: 'X_spatial', umPerUnit: null },
      embedding: shown('X_spatial'),
    })
    expect(umPerUnitForSlot(ds)).toBeNull()
  })

  it('honours the display preference to hide the scale bar', () => {
    const base = createDefaultDatasetState()
    const ds = dsWith({
      spatialScale: SPATIAL,
      embedding: shown('X_spatial'),
      displayPreferences: { ...base.displayPreferences, showScaleBar: false },
    })
    expect(umPerUnitForSlot(ds)).toBeNull()
  })

  it('tolerates a slot that holds no dataset', () => {
    expect(umPerUnitForSlot(undefined)).toBeNull()
  })
})

describe('expressionLegendTitle', () => {
  it('names the gene when only one is shown', () => {
    expect(expressionLegendTitle(['Sox9'], null)).toBe('Sox9')
  })

  it('prefers the gene name even when that gene came from a set', () => {
    expect(expressionLegendTitle(['Sox9'], 'Chondrocyte')).toBe('Sox9')
  })

  it('names the set and how many genes it scored', () => {
    expect(expressionLegendTitle(['A', 'B', 'C'], 'Chondrocyte')).toBe('Chondrocyte (3)')
  })

  it('falls back to a count for an ad-hoc list of genes', () => {
    expect(expressionLegendTitle(['A', 'B'], null)).toBe('2 genes')
  })
})
