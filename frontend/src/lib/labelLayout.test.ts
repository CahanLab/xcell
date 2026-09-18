import { describe, it, expect } from 'vitest'
import { measureLabel, layoutLabels, fitFontSize, boxesOverlap, type LabelInput } from './labelLayout'

const bounds = { w: 400, h: 300 }
const L = (id: string, x: number, y: number, weight = 1): LabelInput => ({ id, text: id, x, y, weight })

function assertNoOverlap(placed: ReturnType<typeof layoutLabels>, fontPx: number) {
  const shown = placed.filter((p) => !p.hidden)
  for (let i = 0; i < shown.length; i++) {
    for (let j = i + 1; j < shown.length; j++) {
      const a = shown[i], b = shown[j]
      const ba = measureLabel(a.text, fontPx), bb = measureLabel(b.text, fontPx)
      expect(boxesOverlap({ x: a.x, y: a.y, ...ba }, { x: b.x, y: b.y, ...bb }, 0)).toBe(false)
    }
  }
}

describe('measureLabel', () => {
  it('scales with text length and font size', () => {
    const a = measureLabel('ab', 12), b = measureLabel('abcdefgh', 12), c = measureLabel('ab', 24)
    expect(b.w).toBeGreaterThan(a.w)
    expect(c.w).toBeGreaterThan(a.w)
    expect(c.h).toBeGreaterThan(a.h)
  })
})

describe('layoutLabels', () => {
  it('leaves non-colliding labels at their anchors', () => {
    const out = layoutLabels([L('A', 50, 50), L('B', 300, 200)], { fontPx: 13, bounds })
    expect(out.map((p) => [p.id, p.x, p.y, p.hidden])).toEqual([['A', 50, 50, false], ['B', 300, 200, false]])
  })

  it('nudges the lighter of two coincident labels and keeps the heavier at its anchor', () => {
    const out = layoutLabels([L('small', 200, 150, 10), L('big', 200, 150, 100)], { fontPx: 13, bounds })
    const big = out.find((p) => p.id === 'big')!, small = out.find((p) => p.id === 'small')!
    expect([big.x, big.y]).toEqual([200, 150])
    expect(small.hidden).toBe(false)
    expect(Math.hypot(small.x - 200, small.y - 150)).toBeGreaterThan(0)
    expect(small.ax).toBe(200)  // the anchor is remembered for a leader line
    assertNoOverlap(out, 13)
  })

  it('keeps every shown label inside the bounds', () => {
    const out = layoutLabels([L('corner', 2, 2), L('edge', 398, 150), L('bottom', 200, 299)], { fontPx: 13, bounds })
    for (const p of out.filter((q) => !q.hidden)) {
      const b = measureLabel(p.text, 13)
      expect(p.x - b.w / 2).toBeGreaterThanOrEqual(0)
      expect(p.x + b.w / 2).toBeLessThanOrEqual(bounds.w)
      expect(p.y - b.h / 2).toBeGreaterThanOrEqual(0)
      expect(p.y + b.h / 2).toBeLessThanOrEqual(bounds.h)
    }
  })

  it('hides labels that cannot be placed within the shift limit, lightest first', () => {
    // 30 labels piled on one point in a small box: not all can fit
    const labels = Array.from({ length: 30 }, (_, i) => L(`cluster ${i}`, 60, 40, 30 - i))
    const out = layoutLabels(labels, { fontPx: 13, bounds: { w: 120, h: 80 }, maxShift: 30 })
    const hidden = out.filter((p) => p.hidden)
    expect(hidden.length).toBeGreaterThan(0)
    expect(out.find((p) => p.id === 'cluster 0')!.hidden).toBe(false)  // the heaviest survives
    assertNoOverlap(out, 13)
    // output keeps the input order so callers can zip it with their data
    expect(out.map((p) => p.id)).toEqual(labels.map((l) => l.id))
  })

  it('is deterministic', () => {
    const labels = Array.from({ length: 12 }, (_, i) => L(`c${i}`, 100 + (i % 3) * 5, 100 + Math.floor(i / 3) * 4, i))
    const a = layoutLabels(labels, { fontPx: 12, bounds })
    const b = layoutLabels(labels, { fontPx: 12, bounds })
    expect(a).toEqual(b)
  })
})

describe('fitFontSize', () => {
  it('returns the largest font at which nothing is hidden, and the floor when nothing fits', () => {
    const spread = [L('A', 40, 40), L('B', 360, 40), L('C', 40, 260), L('D', 360, 260)]
    expect(fitFontSize(spread, { bounds, max: 13, min: 7 })).toBe(13)
    const pile = Array.from({ length: 12 }, (_, i) => L(`cluster ${i}`, 100 + i, 100, 12 - i))
    const f = fitFontSize(pile, { bounds: { w: 200, h: 120 }, max: 13, min: 7, maxShift: 30 })
    expect(f).toBeGreaterThanOrEqual(7)
    expect(f).toBeLessThan(13)
    const impossible = Array.from({ length: 200 }, (_, i) => L(`cluster ${i}`, 50, 50, 200 - i))
    expect(fitFontSize(impossible, { bounds: { w: 100, h: 60 }, max: 13, min: 7 })).toBe(7)
  })
})
