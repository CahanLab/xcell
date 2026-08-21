import { describe, it, expect } from 'vitest'
import { adviseLayers } from './localizeAdvice'

const texts = (a: { text: string }[]) => a.map((x) => x.text).join(' | ')

describe('adviseLayers', () => {
  it('stays silent on the ordinary case: log-normalized both sides, z-scored', () => {
    expect(adviseLayers({
      queryScale: 'log_normalized', referenceScale: 'log_normalized', transform: 'zscore',
    })).toEqual([])
  })

  it('warns when no transform is applied and the two matrices are on different scales', () => {
    const a = adviseLayers({ queryScale: 'raw_counts', referenceScale: 'log_normalized', transform: 'none' })
    expect(a).toHaveLength(1)
    expect(a[0].level).toBe('warn')
    expect(texts(a)).toMatch(/different scale/i)
  })

  it('does not warn about a scale difference that the per-dataset transform removes', () => {
    expect(adviseLayers({
      queryScale: 'raw_counts', referenceScale: 'log_normalized', transform: 'rank',
    }).filter((x) => /different scale/i.test(x.text))).toEqual([])
  })

  it('warns that z-scoring linear counts lets high expressers dominate', () => {
    const a = adviseLayers({ queryScale: 'raw_counts', referenceScale: 'raw_counts', transform: 'zscore' })
    expect(a.some((x) => x.level === 'warn' && /dominat/i.test(x.text))).toBe(true)
  })

  it('says rank makes the log-versus-linear choice irrelevant', () => {
    const a = adviseLayers({ queryScale: 'raw_counts', referenceScale: 'log_normalized', transform: 'rank' })
    expect(texts(a)).toMatch(/monotone|rank/i)
  })

  it('notes that z-scoring an already z-scored matrix is redundant', () => {
    const a = adviseLayers({ queryScale: 'z_scored', referenceScale: 'log_normalized', transform: 'zscore' })
    expect(a.some((x) => x.level === 'info' && /already/i.test(x.text))).toBe(true)
  })

  it('says nothing when a scale could not be determined', () => {
    expect(adviseLayers({ queryScale: 'unknown', referenceScale: null, transform: 'none' })).toEqual([])
  })

  it('treats raw counts and library-normalized values as the same family', () => {
    expect(adviseLayers({
      queryScale: 'raw_counts', referenceScale: 'normalized_linear', transform: 'none',
    }).filter((x) => /different scale/i.test(x.text))).toEqual([])
  })
})
