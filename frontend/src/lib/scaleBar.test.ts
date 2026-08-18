import { describe, it, expect } from 'vitest'
import { pickScaleBar } from './scaleBar'

describe('pickScaleBar', () => {
  it('picks the largest 1/2/5 x 10^k length that fits the target width', () => {
    // 1 µm per screen px, 120 px target -> a 100 µm bar drawn 100 px wide
    expect(pickScaleBar(1)).toEqual({ um: 100, px: 100, label: '100 µm' })
    // raw 60 µm -> nice 50
    expect(pickScaleBar(0.5, 120)).toEqual({ um: 50, px: 100, label: '50 µm' })
    // raw 3000 µm -> nice 2000
    expect(pickScaleBar(25)).toEqual({ um: 2000, px: 80, label: '2 mm' })
  })

  it('labels millimetres from 1000 µm up', () => {
    expect(pickScaleBar(10)).toEqual({ um: 1000, px: 100, label: '1 mm' })
    expect(pickScaleBar(50)).toEqual({ um: 5000, px: 100, label: '5 mm' })
  })

  it('handles sub-micron zoom levels', () => {
    expect(pickScaleBar(0.004)).toEqual({ um: 0.2, px: 50, label: '0.2 µm' })
  })

  it('returns null for degenerate input', () => {
    expect(pickScaleBar(0)).toBeNull()
    expect(pickScaleBar(-1)).toBeNull()
    expect(pickScaleBar(NaN)).toBeNull()
    expect(pickScaleBar(Infinity)).toBeNull()
  })
})
