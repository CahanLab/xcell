import { describe, it, expect } from 'vitest'
import { standaloneSvg } from './svgExport'

const svg = '<svg width="100" height="50"><rect x="1" y="1" width="8" height="8"/></svg>'

describe('standaloneSvg', () => {
  it('adds the xmlns a file needs to open outside the browser', () => {
    expect(standaloneSvg(svg, { background: '#16213e' })).toContain('xmlns="http://www.w3.org/2000/svg"')
  })

  it('does not add a second xmlns when the markup already carries one', () => {
    const already = '<svg xmlns="http://www.w3.org/2000/svg" width="100" height="50"></svg>'
    const out = standaloneSvg(already, { background: '#16213e' })
    expect(out.match(/xmlns="http:\/\/www\.w3\.org\/2000\/svg"/g)).toHaveLength(1)
  })

  it('paints the background, since the figure is drawn for a dark surface', () => {
    const out = standaloneSvg(svg, { background: '#16213e' })
    expect(out).toContain('<rect width="100%" height="100%" fill="#16213e"/>')
  })

  it('puts the background behind the content, not over it', () => {
    const out = standaloneSvg(svg, { background: '#16213e' })
    expect(out.indexOf('fill="#16213e"')).toBeLessThan(out.indexOf('<rect x="1"'))
  })

  it('starts with an XML declaration', () => {
    expect(standaloneSvg(svg, { background: '#16213e' }).startsWith('<?xml version="1.0"')).toBe(true)
  })

  it('leaves the background out when none is asked for', () => {
    expect(standaloneSvg(svg, {})).not.toContain('width="100%"')
  })
})
