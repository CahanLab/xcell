import { describe, it, expect } from 'vitest'
import { datasetIdentity } from './datasetIdentity'
import type { Schema } from '../store'

const schema = (over: Partial<Schema> = {}): Schema => ({
  n_cells: 320,
  n_genes: 500,
  embeddings: ['X_umap'],
  obs_columns: ['cell_type'],
  obs_dtypes: { cell_type: 'category' },
  filename: 'queryA.h5ad',
  ...over,
})

describe('datasetIdentity', () => {
  it('is stable for the same dataset in the same slot', () => {
    expect(datasetIdentity('secondary', schema()))
      .toBe(datasetIdentity('secondary', schema()))
  })

  it('changes when a different file is loaded into the same slot', () => {
    // The reported bug: swapping the dataset inside a slot left a modal
    // showing results computed from the previous one.
    expect(datasetIdentity('secondary', schema({ filename: 'queryB.h5ad', n_cells: 160 })))
      .not.toBe(datasetIdentity('secondary', schema()))
  })

  it('changes when the active slot changes', () => {
    expect(datasetIdentity('primary', schema()))
      .not.toBe(datasetIdentity('secondary', schema()))
  })

  it('changes when cells are removed', () => {
    // A classification result enumerates every cell; drop some and it is stale.
    expect(datasetIdentity('primary', schema({ n_cells: 300 })))
      .not.toBe(datasetIdentity('primary', schema()))
  })

  it('changes when genes are removed', () => {
    // Gene coverage against a classifier is the guard the feature rests on,
    // so it has to be recomputed when the gene set changes.
    expect(datasetIdentity('primary', schema({ n_genes: 400 })))
      .not.toBe(datasetIdentity('primary', schema()))
  })

  it('distinguishes two files that coincidentally have the same shape', () => {
    expect(datasetIdentity('primary', schema({ filename: 'other.h5ad' })))
      .not.toBe(datasetIdentity('primary', schema()))
  })

  it('is defined before a schema has loaded, and differs from any loaded one', () => {
    const empty = datasetIdentity('primary', null)
    expect(typeof empty).toBe('string')
    expect(empty).not.toBe(datasetIdentity('primary', schema()))
  })

  it('treats a missing filename as its own identity rather than throwing', () => {
    expect(() => datasetIdentity('primary', schema({ filename: undefined }))).not.toThrow()
  })
})
