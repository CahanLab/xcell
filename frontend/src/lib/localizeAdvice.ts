/**
 * What is wrong with the current Localize settings, given this reference.
 *
 * Only fires on the current selection — a static table of caveats gets skimmed;
 * a line that appears when you pick the thing does not. Kept pure and out of the
 * modal so each rule can be tested, including the cases where it must stay
 * silent: advice that fires on ordinary settings trains the user to ignore it.
 */

export interface Advice { level: 'warn' | 'info'; text: string }

/**
 * One population's geometry in the reference, from
 * `POST /api/localize/reference_geometry`.
 *
 * `risk` is how far the population's own mean position sits from the population,
 * in units of how far its members sit from each other — 0 is an ordinary place
 * inside it, 1 is far outside. See `localize_metrics.mean_collapse_risk`.
 */
export interface PopulationGeometry {
  name: string
  risk: number | null
  distance_ratio: number | null
  population_fraction: number
  n_population: number
}

export interface LocalizeSettings {
  k: number
  transform: string
  metric: string
  aggregation: string
  minConfidence: number
  /** The transport dial, in units of the cost matrix's own spread. */
  epsilon: number
  nReferenceCells: number
  nQueryCells: number
  nSharedGenes: number | null
  populations: PopulationGeometry[]
}

/**
 * Above this, a population's own mean is somewhere that population is not.
 *
 * 0.5 is exactly "twice as far out as the population's own spacing", which is
 * where the number stops describing an ordinary place inside it. Measured on
 * synthetic geometry, a scattered population reaches 0.20 at worst while a ring
 * scores 0.96, two patches 0.94 and a one-cell-wide perimeter 0.79 — so the
 * threshold sits about 1.6x clear of every case in both directions.
 */
export const COLLAPSE_RISK = 0.5

export function adviseParameters(o: LocalizeSettings): Advice[] {
  const out: Advice[] = []

  if (o.k < 5) {
    out.push({ level: 'warn', text:
      `k=${o.k} is too few to estimate confidence from — the spread of ${o.k} points is noisy, so the score will be unreliable in both directions.` })
  }
  if (o.nReferenceCells > 0 && o.k > 0.1 * o.nReferenceCells) {
    out.push({ level: 'warn', text:
      `k=${o.k} is over a tenth of the ${o.nReferenceCells.toLocaleString()} reference cells. Every prediction averages a large slice of the tissue, so all of them drift toward its centre.` })
  }

  if (o.aggregation === 'transport') {
    out.push({ level: 'info', text:
      'Transport places the population so that it occupies the tissue, which assumes the query’s composition matches the tissue’s. Dissociation biases which cell types survive, so a type under-represented in the query will still have its share of the tissue filled by something else.' })
    // The band either side of which the dial stops trading and starts simply
    // failing. Measured on a synthetic pair: spread falls monotonically from
    // 93% of the reference's own at ε=0.05 to 32% at ε=2.0.
    if (o.epsilon >= 2.0) {
      out.push({ level: 'warn', text:
        `ε=${o.epsilon} spreads each cell’s posterior over much of the reference, so the averaged prediction collapses toward the tissue centre — the failure this estimator exists to avoid.` })
    } else if (o.epsilon <= 0.05) {
      out.push({ level: 'warn', text:
        `ε=${o.epsilon} makes each cell pick almost a single spot, which throws away the averaging that keeps the gradient. Expect the axis correlations to fall toward best_match’s.` })
    }
  }

  if (o.transform === 'none' && o.metric === 'euclidean') {
    out.push({ level: 'warn', text:
      'none + euclidean compares raw magnitudes. Across two platforms, sequencing depth will dominate the distance and swamp the biology.' })
  } else if (o.transform === 'none') {
    out.push({ level: 'warn', text:
      'No per-dataset transform. Only safe when both datasets are already normalized the same way — otherwise per-gene capture differences drive the matches.' })
  }

  if (o.aggregation === 'densest' && o.k < 12) {
    out.push({ level: 'info', text:
      `densest picks the tightest cluster among the neighbours, and with k=${o.k} there is not much to cluster — it will behave much like the mean. Raise k to about 20 to get the benefit.` })
  }
  if (o.aggregation === 'best_match') {
    out.push({ level: 'info', text:
      'best_match takes the single nearest cell’s position, so k no longer affects the coordinate — only the confidence score, which is still measured over k neighbours.' })
  }

  if (o.aggregation === 'injective') {
    if (o.nQueryCells > o.nReferenceCells) {
      out.push({ level: 'warn', text:
        `injective needs at least one reference spot per query cell, and there are ${o.nQueryCells.toLocaleString()} cells against ${o.nReferenceCells.toLocaleString()} spots. Subsample the query, or use best match.` })
    }
    out.push({ level: 'info', text:
      'injective gives every cell a different spot, which asserts that the query’s composition matches the tissue’s. Dissociation biases which cell types survive, so an over-represented type gets pushed into locations it does not belong. It fixes pile-up, not placement — its axis correlations are as flat as best match’s, so compare them under Map quality rather than assuming it is an improvement.' })
  }

  // The reference's own geometry, which is enough to know this in advance: the
  // mean of a ring is its hole, and weighted_mean returns roughly that mean for
  // every query cell of the type.
  const hollow = o.populations.filter((p) => p.risk != null && p.risk >= COLLAPSE_RISK)
  if (o.aggregation === 'weighted_mean' && hollow.length > 0) {
    const named = hollow.slice(0, 3).map((p) => p.name).join(', ')
    const more = hollow.length > 3 ? ` (+${hollow.length - 3} more)` : ''
    const one = hollow.length === 1
    out.push({ level: 'warn', text:
      `In this reference ${named}${more} ${one ? 'forms a ring or hugs the tissue edge' : 'form rings or hug the tissue edge'}, so the mean of ${one ? 'its' : 'their'} positions is a place none of those cells occupy. weighted mean will predict every query cell of ${one ? 'that type' : 'those types'} into that hole — this is what inverted the epidermis on an E11.5 limb, placing the embryo’s outermost tissue in the middle of the bud. best match or densest keep predictions on real tissue, at the cost of the gradients; score both under Map quality before choosing.` })
  } else if (o.aggregation === 'weighted_mean') {
    out.push({ level: 'info', text:
      'weighted_mean can land in a gap the tissue does not have, most visibly for a cell type present in two separate regions. densest or best_match keep predictions on real tissue.' })
  }

  if (o.nSharedGenes != null && o.nSharedGenes < 30) {
    out.push({ level: 'warn', text:
      `Only ${o.nSharedGenes} genes drive the similarity. That is enough to run but not to separate fine structure; treat the map as coarse and check the accuracy figures.` })
  }
  if (o.minConfidence > 0) {
    out.push({ level: 'info', text:
      `Cells scoring under ${o.minConfidence} will get no coordinate at all rather than a guessed one. Confidence depends on the gene basis, so a threshold tuned for one basis does not carry to another.` })
  }
  return out
}
