import { describe, it, expect } from 'vitest'
import {
  adviseParameters, COLLAPSE_RISK,
  type LocalizeSettings, type PopulationGeometry,
} from './localizeAdvice'

// The real limb pair: 9,145 spots as reference, 2,683 dissociated cells as
// query, 12,449 genes shared. `median` has no rule of its own, so this is a
// settings object nothing should say anything about.
const OK: LocalizeSettings = {
  k: 15, transform: 'zscore', metric: 'correlation', aggregation: 'median',
  minConfidence: 0, nReferenceCells: 9145, nQueryCells: 2683,
  nSharedGenes: 12449, populations: [], epsilon: 0.5,
}

const pop = (name: string, risk: number | null): PopulationGeometry => ({
  name, risk, distance_ratio: risk == null ? null : 1 / (1 - risk),
  population_fraction: 0.2, n_population: 100,
})

const texts = (a: { text: string }[]) => a.map((x) => x.text).join(' | ')
const warns = (a: { level: string }[]) => a.filter((x) => x.level === 'warn')

describe('the weighted_mean collapse warning', () => {
  it('names the populations this reference would lose', () => {
    const a = adviseParameters({
      ...OK, aggregation: 'weighted_mean',
      populations: [pop('skin', 0.98), pop('muscle', 0.02)],
    })
    expect(texts(a)).toContain('skin')
    expect(texts(a)).not.toContain('muscle')
    expect(warns(a).length).toBeGreaterThan(0)
  })

  it('stays quiet when no population is hollow', () => {
    const a = adviseParameters({
      ...OK, aggregation: 'weighted_mean', populations: [pop('muscle', 0.02)],
    })
    expect(warns(a)).toHaveLength(0)
  })

  it('stays quiet for an aggregation that does not average', () => {
    const a = adviseParameters({
      ...OK, aggregation: 'best_match', populations: [pop('skin', 0.98)],
    })
    expect(texts(a)).not.toContain('skin')
  })

  it('ignores a population whose risk could not be computed', () => {
    const a = adviseParameters({
      ...OK, aggregation: 'weighted_mean', populations: [pop('tiny', null)],
    })
    expect(warns(a)).toHaveLength(0)
  })

  it('caps the list rather than printing every population', () => {
    const many = ['a', 'b', 'c', 'd', 'e'].map((n) => pop(n, 0.9))
    const t = texts(adviseParameters({
      ...OK, aggregation: 'weighted_mean', populations: many,
    }))
    expect(t).toContain('+2 more')
  })

  it('uses the documented threshold', () => {
    const just = adviseParameters({
      ...OK, aggregation: 'weighted_mean', populations: [pop('edge', COLLAPSE_RISK)],
    })
    const under = adviseParameters({
      ...OK, aggregation: 'weighted_mean',
      populations: [pop('edge', COLLAPSE_RISK - 0.01)],
    })
    expect(warns(just).length).toBeGreaterThan(0)
    expect(warns(under)).toHaveLength(0)
  })

  it('still gives the generic caveat when there is nothing specific to say', () => {
    const t = texts(adviseParameters({ ...OK, aggregation: 'weighted_mean' }))
    expect(t).toContain('gap the tissue does not have')
  })

  it('does not repeat the generic caveat alongside the named one', () => {
    const t = texts(adviseParameters({
      ...OK, aggregation: 'weighted_mean', populations: [pop('skin', 0.98)],
    }))
    expect(t).not.toContain('gap the tissue does not have')
  })
})

describe('injective', () => {
  it('says what one-to-one asserts about the query', () => {
    expect(texts(adviseParameters({ ...OK, aggregation: 'injective' })))
      .toContain('composition')
  })

  it('does not claim to be better than best match', () => {
    expect(texts(adviseParameters({ ...OK, aggregation: 'injective' })))
      .toContain('pile-up, not placement')
  })

  it('warns when there are more cells than spots', () => {
    const a = adviseParameters({
      ...OK, aggregation: 'injective', nReferenceCells: 900, nQueryCells: 2683,
    })
    expect(warns(a).length).toBeGreaterThan(0)
    expect(texts(a)).toContain('2,683')
  })

  it('does not warn about size when the reference is big enough', () => {
    expect(warns(adviseParameters({ ...OK, aggregation: 'injective' })))
      .toHaveLength(0)
  })
})

describe('the rules that were already there', () => {
  it('still catches a k that is too small', () => {
    expect(texts(adviseParameters({ ...OK, k: 3 }))).toContain('k=3')
  })

  it('still catches k over a tenth of the reference', () => {
    expect(texts(adviseParameters({ ...OK, k: 2000 }))).toContain('toward its centre')
  })

  it('still catches none + euclidean', () => {
    const t = texts(adviseParameters({
      ...OK, transform: 'none', metric: 'euclidean',
    }))
    expect(t).toContain('sequencing depth')
  })

  it('still catches too few shared genes', () => {
    expect(texts(adviseParameters({ ...OK, nSharedGenes: 12 }))).toContain('12 genes')
  })

  it('says nothing at all about sensible settings', () => {
    expect(adviseParameters(OK)).toHaveLength(0)
  })
})

describe('transport', () => {
  // The measured default: hull area 0.44 on the limb pair, converging.
  const OT: LocalizeSettings = { ...OK, aggregation: 'transport', epsilon: 0.05 }

  it('states the composition assumption the reference marginal makes', () => {
    expect(texts(adviseParameters(OT))).toContain('composition')
  })

  it('warns that a large epsilon collapses the map toward the centre', () => {
    expect(texts(adviseParameters({ ...OT, epsilon: 3.0 }))).toContain('centre')
  })

  it('warns at the epsilon where the epidermis inverted again', () => {
    expect(texts(adviseParameters({ ...OT, epsilon: 0.5 }))).toContain('centre')
  })

  it('warns that an epsilon below the band will not converge', () => {
    expect(texts(adviseParameters({ ...OT, epsilon: 0.01 }))).toContain('converges slowly')
  })

  it('says nothing about epsilon at the measured default', () => {
    const t = texts(adviseParameters(OT))
    expect(t).not.toContain('centre')
    expect(t).not.toContain('converges slowly')
  })

  it('stays silent about epsilon for every other aggregation', () => {
    const t = texts(adviseParameters({ ...OK, epsilon: 3.0 }))
    expect(t).not.toContain('ε')
  })
})
