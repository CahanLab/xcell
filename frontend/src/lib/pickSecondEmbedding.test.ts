import { describe, it, expect } from 'vitest'
import { pickSecondEmbedding } from './pickSecondEmbedding'

describe('pickSecondEmbedding', () => {
  it('offers the complementary view: expression map beside a spatial one', () => {
    expect(pickSecondEmbedding(['spatial', 'X_umap', 'X_pca'], 'spatial')).toBe('X_umap')
  })

  it('offers the tissue beside an expression map', () => {
    expect(pickSecondEmbedding(['spatial', 'X_umap', 'X_pca'], 'X_umap')).toBe('spatial')
  })

  it('falls back to any other embedding when no preferred kind exists', () => {
    expect(pickSecondEmbedding(['custom_a', 'custom_b'], 'custom_a')).toBe('custom_b')
  })

  it('returns null when there is nothing else to show', () => {
    expect(pickSecondEmbedding(['spatial'], 'spatial')).toBeNull()
    expect(pickSecondEmbedding([], null)).toBeNull()
  })
})
