import { describe, it, expect } from 'vitest'
import { clusterBands, similarityColor, coverageLabel } from './metaPrograms'

describe('clusterBands', () => {
  it('turns a dendrogram-ordered cluster assignment into contiguous runs', () => {
    // order is the leaf order; clusters are indexed by program, not by leaf
    const clusters = [1, 0, 0, 1, 2]
    const order = [1, 2, 0, 3, 4]
    expect(clusterBands(clusters, order)).toEqual([
      { cluster: 0, start: 0, size: 2 },
      { cluster: 1, start: 2, size: 2 },
      { cluster: 2, start: 4, size: 1 },
    ])
  })

  it('splits a cluster that the leaf order does not keep together', () => {
    // Shouldn't happen with a maxclust cut, but a band must never silently
    // span programs that aren't adjacent — that would mislabel the heatmap.
    const clusters = [0, 1, 0]
    const order = [0, 1, 2]
    expect(clusterBands(clusters, order)).toEqual([
      { cluster: 0, start: 0, size: 1 },
      { cluster: 1, start: 1, size: 1 },
      { cluster: 0, start: 2, size: 1 },
    ])
  })

  it('is empty for no programs', () => {
    expect(clusterBands([], [])).toEqual([])
  })

  it('ignores programs whose meta-program was dropped', () => {
    // -1 marks a program in a meta-program that had no genes left.
    expect(clusterBands([0, -1, 0], [0, 1, 2])).toEqual([
      { cluster: 0, start: 0, size: 1 },
      { cluster: -1, start: 1, size: 1 },
      { cluster: 0, start: 2, size: 1 },
    ])
  })
})

describe('similarityColor', () => {
  it('is dark at no similarity and bright at full similarity', () => {
    expect(similarityColor(0)).not.toEqual(similarityColor(1))
    expect(similarityColor(0.5)).toMatch(/^rgb\(/)
  })

  it('clamps out-of-range values rather than producing invalid colors', () => {
    expect(similarityColor(-3)).toEqual(similarityColor(0))
    expect(similarityColor(9)).toEqual(similarityColor(1))
  })

  it('handles a non-finite similarity without emitting NaN', () => {
    expect(similarityColor(Number.NaN)).not.toMatch(/NaN/)
  })
})

describe('coverageLabel', () => {
  it('reads as a fraction of samples', () => {
    expect(coverageLabel(1, 3)).toBe('3/3 samples')
    expect(coverageLabel(1 / 3, 3)).toBe('1/3 samples')
  })

  it('rounds to the nearest whole sample', () => {
    expect(coverageLabel(0.66, 3)).toBe('2/3 samples')
  })
})
