import { describe, it, expect } from 'vitest'
import { adviseContour, type ContourSettings, type ContourGeometry } from './contourAdvice'

// 400 spots on a 20x20 grid at spacing 2, so extent = 38. These are the real
// numbers the backend produces for that layout: suggest_grid_res clamps to a
// floor of 50 (hence 50 rather than sqrt(400)=20), and suggest_smooth_sigma
// solves for a radius of 2 spot spacings, giving 2*(2/(38/50)) = 5.26.
const GEOM: ContourGeometry = {
  nSpots: 400, medianSpacing: 2, extent: 38,
  suggestedGridRes: 50, suggestedSigma: 5.26,
}

const OK: ContourSettings = {
  gridRes: 50, smoothSigma: 5.26, contourLevels: 3, logTransform: false,
  sectionCol: '', sectionCandidates: [], nSources: 2, smallestSourceSize: 12,
  hasPca: true, matrixScale: 'log_normalized', matrixMax: 6.2,
}

const texts = (a: { text: string }[]) => a.map((x) => x.text).join(' | ')

describe('adviseContour', () => {
  it('says nothing when the settings match the data', () => {
    // The suggested grid already puts 2.6 pixels between adjacent spots
    // because of the floor of 50. Advice that fires on its own defaults is
    // noise, so every threshold sits clear of this point.
    expect(adviseContour(OK, GEOM)).toEqual([])
  })

  it('returns nothing at all when the geometry is unknown', () => {
    // A dataset with one spot has no spacing to compare against; silence beats
    // a comparison against null.
    expect(adviseContour(OK, null)).toEqual([])
  })

  it('warns when the grid is far finer than the spot spacing', () => {
    // grid 600 -> 31 pixels between adjacent spots. Everything between them is
    // interpolated, not measured.
    const a = adviseContour({ ...OK, gridRes: 600 }, GEOM)
    expect(texts(a)).toMatch(/finer than/i)
    expect(a.some((x) => x.level === 'warn')).toBe(true)
  })

  it('warns when the grid is coarser than the tissue', () => {
    // grid 5 -> one pixel spans 3.8 spot spacings.
    const a = adviseContour({ ...OK, gridRes: 5 }, GEOM)
    expect(texts(a)).toMatch(/coarser|spans/i)
  })

  it('warns when the smoothing does not reach the next spot', () => {
    // radius = 0.3 * 0.76 = 0.23 data units = 0.11 spacings.
    const a = adviseContour({ ...OK, smoothSigma: 0.3 }, GEOM)
    expect(texts(a)).toMatch(/speckle|noise/i)
  })

  it('warns when the smoothing spans a whole zone', () => {
    // radius = 30 * 0.76 = 22.8 data units = 11.4 spacings.
    const a = adviseContour({ ...OK, smoothSigma: 30 }, GEOM)
    expect(texts(a)).toMatch(/merge/i)
  })

  it('catches the grid-times-sigma coupling', () => {
    // Raising the grid to 80 without touching sigma moves the radius from 2.00
    // to 1.25 spot spacings -- still inside the sane band, so neither radius
    // warning fires and this is the only thing that would tell you.
    const a = adviseContour({ ...OK, gridRes: 80 }, GEOM)
    expect(texts(a)).toMatch(/grid pixels/i)
    expect(texts(a)).toMatch(/1\.2|1\.3/)
  })

  it('does not repeat the coupling when a radius warning already said it', () => {
    // grid 600 pushes the radius to 0.17 spacings, which the speckle warning
    // states directly. Saying it twice is noise.
    const a = adviseContour({ ...OK, gridRes: 600 }, GEOM)
    expect(a.filter((x) => /grid pixels/i.test(x.text))).toHaveLength(0)
  })

  it('warns when log is off and the matrix looks like raw counts', () => {
    const a = adviseContour(
      { ...OK, logTransform: false, matrixScale: 'raw_counts', matrixMax: 8134 },
      GEOM,
    )
    expect(texts(a)).toMatch(/raw counts/i)
    expect(texts(a)).toMatch(/8,?134/)
  })

  it('warns when log is on and the matrix is already log-normalized', () => {
    const a = adviseContour({ ...OK, logTransform: true }, GEOM)
    expect(texts(a)).toMatch(/second log|already/i)
  })

  it('warns when log is on and the matrix is z-scored', () => {
    const a = adviseContour(
      { ...OK, logTransform: true, matrixScale: 'z_scored', matrixMax: 4.1 },
      GEOM,
    )
    expect(texts(a)).toMatch(/negative/i)
  })

  it('says nothing about log when the scale is unknown', () => {
    const a = adviseContour({ ...OK, matrixScale: null, matrixMax: null }, GEOM)
    expect(texts(a)).not.toMatch(/log/i)
  })

  it('mentions an unused section column', () => {
    const a = adviseContour({ ...OK, sectionCandidates: ['section'] }, GEOM)
    expect(texts(a)).toMatch(/section/i)
    expect(texts(a)).toMatch(/bleed|across/i)
  })

  it('stays quiet once a section column is chosen', () => {
    const a = adviseContour(
      { ...OK, sectionCol: 'section', sectionCandidates: ['section'] }, GEOM,
    )
    expect(texts(a)).not.toMatch(/bleed/i)
  })

  it('warns about multi-set contouring without PCA before the run fails', () => {
    const a = adviseContour({ ...OK, nSources: 3, hasPca: false }, GEOM)
    expect(texts(a)).toMatch(/PCA/)
  })

  it('does not mention PCA for a single-set contour', () => {
    const a = adviseContour({ ...OK, nSources: 1, hasPca: false }, GEOM)
    expect(texts(a)).not.toMatch(/PCA/)
  })

  it('notes a source with almost no genes', () => {
    const a = adviseContour({ ...OK, smallestSourceSize: 1 }, GEOM)
    expect(texts(a)).toMatch(/one gene/i)
  })

  it('explains that more levels only subdivide the cutoff', () => {
    const a = adviseContour({ ...OK, contourLevels: 8 }, GEOM)
    expect(texts(a)).toMatch(/cutoff/i)
  })

  it('puts warnings before infos', () => {
    const a = adviseContour(
      { ...OK, gridRes: 600, contourLevels: 8, sectionCandidates: ['section'] },
      GEOM,
    )
    const firstInfo = a.findIndex((x) => x.level === 'info')
    const lastWarn = a.map((x) => x.level).lastIndexOf('warn')
    expect(firstInfo === -1 || lastWarn < firstInfo).toBe(true)
  })
})
