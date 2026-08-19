import { describe, it, expect } from 'vitest'
import {
  zDomain, divergingColor, seqColor, textColorFor, formatZ, formatPct,
  Z_MID, Z_COOL, Z_WARM,
} from './neighborhoodViz'

const rgb = (c: [number, number, number]) => `rgb(${c[0]},${c[1]},${c[2]})`

describe('zDomain', () => {
  it('never collapses below 3 so weak results do not look saturated', () => {
    expect(zDomain([[0.2, -1.1], [0.4, 0.9]])).toBe(3)
  })
  it('uses a robust quantile so one huge z does not flatten the rest', () => {
    const z = Array.from({ length: 10 }, () => Array.from({ length: 10 }, () => 2))
    z[0][1] = 50
    expect(zDomain(z)).toBeLessThan(10)
    expect(zDomain(z)).toBeGreaterThanOrEqual(3)
  })
  it('tracks the bulk when enrichment is strong overall', () => {
    const z = [[8, -8], [8, 8]]
    expect(zDomain(z)).toBe(8)
  })
})

describe('divergingColor', () => {
  it('hits the neutral midpoint at z = 0', () => {
    expect(divergingColor(0, 5)).toBe(rgb(Z_MID))
  })
  it('hits the poles at ±domain and clamps beyond', () => {
    expect(divergingColor(5, 5)).toBe(rgb(Z_WARM))
    expect(divergingColor(-5, 5)).toBe(rgb(Z_COOL))
    expect(divergingColor(12, 5)).toBe(rgb(Z_WARM))
  })
  it('is monotone within an arm', () => {
    const chan = (s: string) => Number(s.slice(4).split(',')[0])
    const r0 = chan(divergingColor(0, 5))
    const r1 = chan(divergingColor(2.5, 5))
    const r2 = chan(divergingColor(5, 5))
    expect(r1).toBeGreaterThan(r0)
    expect(r2).toBeGreaterThan(r1)
  })
})

describe('seqColor', () => {
  it('spans low to high and clamps', () => {
    const lo = seqColor(0)
    expect(seqColor(-0.5)).toBe(lo)
    expect(seqColor(1)).toBe(seqColor(2))
    expect(lo).not.toBe(seqColor(1))
  })
})

describe('textColorFor', () => {
  it('uses dark ink on light cells and light ink on dark cells', () => {
    expect(textColorFor([240, 240, 240])).toBe('#101418')
    expect(textColorFor([20, 25, 40])).toBe('#eee')
  })
})

describe('formatZ', () => {
  it('signs and rounds to one decimal', () => {
    expect(formatZ(3.14)).toBe('+3.1')
    expect(formatZ(-2.06)).toBe('-2.1')
  })
  it('never prints negative zero', () => {
    expect(formatZ(-0.04)).toBe('0.0')
  })
})

describe('formatPct', () => {
  it('keeps a decimal only below 10%', () => {
    expect(formatPct(0.253)).toBe('25%')
    expect(formatPct(0.083)).toBe('8.3%')
    expect(formatPct(0)).toBe('0%')
  })
})
