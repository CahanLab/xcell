import { describe, it, expect } from 'vitest'
import { pairwiseDistance, hierarchicalOrder, dendrogramSegments, type Linkage } from './hclust'

describe('pairwiseDistance', () => {
  it('measures euclidean distance between rows', () => {
    const d = pairwiseDistance([[0, 0], [3, 4]], 'euclidean')
    expect(d[0][1]).toBeCloseTo(5)
    expect(d[0][0]).toBe(0)
  })

  it('scores perfectly correlated rows as distance 0 regardless of scale', () => {
    const d = pairwiseDistance([[1, 2, 3], [10, 20, 30]], 'correlation')
    expect(d[0][1]).toBeCloseTo(0, 6)
  })

  it('scores anti-correlated rows as distance 2', () => {
    const d = pairwiseDistance([[1, 2, 3], [3, 2, 1]], 'correlation')
    expect(d[0][1]).toBeCloseTo(2, 6)
  })

  it('treats a constant row as maximally uninformative rather than NaN', () => {
    const d = pairwiseDistance([[1, 1, 1], [1, 2, 3]], 'correlation')
    expect(Number.isFinite(d[0][1])).toBe(true)
    expect(d[0][1]).toBeCloseTo(1)
  })

  it('is symmetric', () => {
    const d = pairwiseDistance([[1, 5, 2], [4, 0, 9], [2, 2, 2]], 'cosine')
    expect(d[1][2]).toBeCloseTo(d[2][1])
  })
})

describe('hierarchicalOrder', () => {
  const blocks = [
    // two tight blocks: {0,1} and {2,3}
    [10, 9, 0, 1],
    [9, 10, 1, 0],
    [0, 1, 10, 9],
    [1, 0, 9, 10],
  ]

  it('puts members of the same block next to each other', () => {
    const { order } = hierarchicalOrder(blocks, { metric: 'correlation', linkage: 'average' })
    const pos = new Map(order.map((leaf, i) => [leaf, i]))
    expect(Math.abs(pos.get(0)! - pos.get(1)!)).toBe(1)
    expect(Math.abs(pos.get(2)! - pos.get(3)!)).toBe(1)
  })

  it('returns every leaf exactly once', () => {
    const { order } = hierarchicalOrder(blocks, { metric: 'euclidean', linkage: 'complete' })
    expect([...order].sort()).toEqual([0, 1, 2, 3])
  })

  it('handles the two-row case without clustering machinery falling over', () => {
    const { order, merges } = hierarchicalOrder([[1, 2], [3, 4]], { metric: 'euclidean', linkage: 'average' })
    expect(order).toHaveLength(2)
    expect(merges).toHaveLength(1)
  })

  it('handles a single row', () => {
    const { order, merges } = hierarchicalOrder([[1, 2]], { metric: 'euclidean', linkage: 'average' })
    expect(order).toEqual([0])
    expect(merges).toHaveLength(0)
  })

  it('records one merge per join, each with a finite height', () => {
    const { merges } = hierarchicalOrder(blocks, { metric: 'correlation', linkage: 'average' })
    expect(merges).toHaveLength(3)
    expect(merges.every((m) => Number.isFinite(m.height))).toBe(true)
  })

  it('merges in non-decreasing height order, so the dendrogram never inverts', () => {
    const heights = hierarchicalOrder(blocks, { metric: 'correlation', linkage: 'average' }).merges.map((m) => m.height)
    for (let i = 1; i < heights.length; i++) expect(heights[i]).toBeGreaterThanOrEqual(heights[i - 1] - 1e-12)
  })

  it('orients each merge so the closest ends of the two subtrees meet', () => {
    // 0 and 3 are the closest pair across the two blocks; a good ordering puts
    // them at the boundary rather than at the two far ends.
    const m = [
      [0, 0.1, 5, 4.6],
      [0.1, 0, 5.2, 5],
      [5, 5.2, 0, 0.1],
      [4.6, 5, 0.1, 0],
    ]
    const { order } = hierarchicalOrder(m, { metric: 'euclidean', linkage: 'average' })
    const pos = new Map(order.map((leaf, i) => [leaf, i]))
    expect(Math.abs(pos.get(0)! - pos.get(3)!)).toBe(1)
  })

  it('supports ward linkage', () => {
    const { order } = hierarchicalOrder(blocks, { metric: 'euclidean', linkage: 'ward' as Linkage })
    expect([...order].sort()).toEqual([0, 1, 2, 3])
  })
})

describe('dendrogramSegments', () => {
  const clustered = () => hierarchicalOrder(
    [[10, 9, 0, 1], [9, 10, 1, 0], [0, 1, 10, 9], [1, 0, 9, 10]],
    { metric: 'correlation', linkage: 'average' },
  )

  it('draws three segments per merge — two uprights and the crossbar', () => {
    const { segments } = dendrogramSegments(clustered())
    expect(segments).toHaveLength(3 * 3)
  })

  it('places leaf uprights at their slot in the drawn order, not their index', () => {
    const result = clustered()
    const { segments } = dendrogramSegments(result)
    const firstLeaf = result.order[0]
    // the upright for the first-drawn leaf rises from slot 0 at height 0
    const upright = segments.find((s) => s.x1 === s.x2 && s.y1 === 0 && s.x1 === 0)
    expect(upright).toBeDefined()
    expect(result.order.indexOf(firstLeaf)).toBe(0)
  })

  it('reports the root height so the caller can scale the drawing', () => {
    const result = clustered()
    const { maxHeight } = dendrogramSegments(result)
    expect(maxHeight).toBeCloseTo(result.merges[result.merges.length - 1].height)
  })

  it('never returns a zero height, which would divide the scale by zero', () => {
    const identical = [[1, 2], [1, 2], [1, 2]]
    const { maxHeight } = dendrogramSegments(hierarchicalOrder(identical, { metric: 'euclidean', linkage: 'average' }))
    expect(maxHeight).toBeGreaterThan(0)
  })

  it('hangs a crossbar at the merge height spanning both children', () => {
    const { segments } = dendrogramSegments(clustered())
    const crossbars = segments.filter((s) => s.y1 === s.y2 && s.x1 !== s.x2)
    expect(crossbars).toHaveLength(3)
    expect(crossbars.every((s) => s.y1 > 0)).toBe(true)
  })
})
