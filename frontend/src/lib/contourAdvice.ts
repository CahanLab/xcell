/**
 * What is wrong with the current contour settings, given this dataset.
 *
 * Contour's parameters are not independent and the coupling is invisible on the
 * form: sigma is measured in *grid pixels*, so one pixel is `extent / gridRes`
 * in data units and the physical smoothing radius is `sigma × extent / gridRes`.
 * Raising the grid alone shrinks the actual smoothing by the same factor.
 *
 * Every rule is therefore expressed in **spot spacings** — a unit visible in the
 * plot — rather than as a disagreement with a suggested number the user never
 * chose. That also keeps the thresholds honest: `suggest_grid_res` has a floor
 * of 50, so on any dataset under ~2500 spots the suggestion itself already puts
 * two or three grid pixels between adjacent spots. Advice that fires on its own
 * defaults is noise, so the thresholds sit well clear of that point.
 */

export interface Advice { level: 'warn' | 'info'; text: string }

// A section column is named like one. Cardinality alone says nothing —
// cell_type, condition and sex are all low-cardinality categoricals, and
// suggesting them as sections is worse than saying nothing at all.
const SECTION_NAME = /(^|[_.\- ])(section|sample|slide|slice|array|library|capture|puck|fov|batch)([_.\- ]|$)/i

/**
 * Categorical columns that plausibly label tissue sections.
 *
 * Both signals are required: the name has to read like a section, and the level
 * count has to be small enough to be one (a per-cell barcode column called
 * `sample_barcode` is categorical and named right, but 4,000 levels is not a
 * set of sections).
 */
export function sectionColumnCandidates(
  cols: { name: string; dtype?: string; nCategories?: number }[],
): string[] {
  return cols
    .filter((c) => c.dtype === 'category'
      && (c.nCategories ?? 0) >= 2 && (c.nCategories ?? 0) <= 20
      && SECTION_NAME.test(c.name))
    .map((c) => c.name)
}

export interface ContourGeometry {
  nSpots: number
  medianSpacing: number | null
  extent: number | null
  suggestedGridRes: number
  suggestedSigma: number
}

export interface ContourSettings {
  gridRes: number
  smoothSigma: number
  contourLevels: number
  logTransform: boolean
  sectionCol: string
  sectionCandidates: string[]
  nSources: number
  smallestSourceSize: number | null
  hasPca: boolean
  matrixScale: string | null
  matrixMax: number | null
}

// More than five grid pixels between adjacent spots. The suggested grid sits at
// ~2.6 on a small dataset, so this only fires on a deliberate over-fine grid.
const MIN_PX_PER_SPACING = 0.2
// One pixel spanning three spots: a zone a few spots across cannot survive.
const MAX_PX_PER_SPACING = 3
// The Gaussian has to reach the neighbouring spot to average anything at all.
const MIN_RADIUS_SPACINGS = 1
// Wide enough to span a zone rather than define one.
const MAX_RADIUS_SPACINGS = 6

const round1 = (v: number) => Math.round(v * 10) / 10
const round2 = (v: number) => Math.round(v * 100) / 100

export function adviseContour(
  s: ContourSettings,
  g: ContourGeometry | null,
): Advice[] {
  const warns: Advice[] = []
  const infos: Advice[] = []

  // Without spacing and extent there is nothing to compare against, and a
  // comparison against a missing number is worse than silence.
  const spacing = g?.medianSpacing ?? null
  const extent = g?.extent ?? null
  if (g && spacing && extent && spacing > 0 && extent > 0 && s.gridRes > 0) {
    const px = extent / s.gridRes            // data units per grid pixel
    const pxPerSpacing = px / spacing
    const radiusSpacings = (s.smoothSigma * px) / spacing

    if (pxPerSpacing < MIN_PX_PER_SPACING) {
      warns.push({ level: 'warn', text:
        `A grid of ${s.gridRes} over ${g.nSpots.toLocaleString()} spots puts ` +
        `${Math.round(1 / pxPerSpacing)} pixels between neighbouring spots — far finer ` +
        `than the data samples, so the detail between spots is interpolated rather ` +
        `than measured. About ${g.suggestedGridRes} matches this tissue.` })
    } else if (pxPerSpacing > MAX_PX_PER_SPACING) {
      warns.push({ level: 'warn', text:
        `Each grid pixel spans ${round1(pxPerSpacing)} spot spacings, so a zone smaller ` +
        `than about ${Math.ceil(pxPerSpacing)} spots across cannot appear at all. ` +
        `About ${g.suggestedGridRes} matches this tissue.` })
    }

    if (radiusSpacings < MIN_RADIUS_SPACINGS) {
      warns.push({ level: 'warn', text:
        `Smoothing reaches ${round2(radiusSpacings)} spot spacings ` +
        `(σ ${s.smoothSigma} × ${round2(px)} per pixel), so it never reaches the next ` +
        `spot. The bands will follow per-spot noise instead of forming zones.` })
    } else if (radiusSpacings > MAX_RADIUS_SPACINGS) {
      warns.push({ level: 'warn', text:
        `Smoothing reaches ${round1(radiusSpacings)} spot spacings — wide enough to span ` +
        `a whole zone, so adjacent tissues will merge into one band.` })
    } else if (s.gridRes !== g.suggestedGridRes &&
               Math.abs(s.smoothSigma - g.suggestedSigma) < 0.01) {
      // Only when neither radius rule already said it more directly: σ is in
      // grid pixels, so editing the grid rescaled the smoothing silently.
      const suggestedRadius = (g.suggestedSigma * (extent / g.suggestedGridRes)) / spacing
      infos.push({ level: 'info', text:
        `σ is measured in grid pixels, so moving the grid to ${s.gridRes} changed the ` +
        `smoothing from ${round2(suggestedRadius)} to ${round2(radiusSpacings)} spot ` +
        `spacings without you touching σ. σ ${round1(2 * spacing / px)} would restore ` +
        `the original width.` })
    }
  }

  // Scale of .X. Only speak when layer_scale actually reached a verdict.
  if (s.matrixScale === 'raw_counts' && !s.logTransform) {
    const max = s.matrixMax != null ? ` (max ${s.matrixMax.toLocaleString()})` : ''
    warns.push({ level: 'warn', text:
      `.X looks like raw counts${max} and log transform is off. Genes are clipped at ` +
      `the 1st/99th percentile, so the tail is handled — but the bulk of a count ` +
      `distribution is multiplicative, and without the log a few bright spots set the ` +
      `range for everything else.` })
  }
  if (s.logTransform &&
      (s.matrixScale === 'log_normalized' || s.matrixScale === 'log_transformed')) {
    warns.push({ level: 'warn', text:
      `.X is already on a log scale, so this applies a second log. That flattens the ` +
      `field and pulls every band toward the middle.` })
  }
  if (s.logTransform && s.matrixScale === 'z_scored') {
    warns.push({ level: 'warn', text:
      `.X is centred and scaled, so it holds negative values — log1p of those is not a ` +
      `number the score can use.` })
  }

  if (!s.sectionCol && s.sectionCandidates.length > 0) {
    const named = s.sectionCandidates.slice(0, 3).join(', ')
    infos.push({ level: 'info', text:
      `No section column set, but ${named} ` +
      `${s.sectionCandidates.length === 1 ? 'looks like one' : 'look like candidates'}. ` +
      `The interpolation grid spans the gap between sections, so expression bleeds ` +
      `across it.` })
  }

  if (s.nSources >= 2 && !s.hasPca) {
    warns.push({ level: 'warn', text:
      `Fusing ${s.nSources} modules needs X_pca to resolve spots that are high in more ` +
      `than one. Run PCA first, or contour a single set.` })
  }

  if (s.smallestSourceSize != null && s.smallestSourceSize <= 2) {
    infos.push({ level: 'info', text:
      `A gene set of ${s.smallestSourceSize} makes the module score essentially ` +
      `${s.smallestSourceSize === 1 ? 'one gene' : 'two genes'}' spatial field, ` +
      `dropout included.` })
  }

  if (s.contourLevels >= 6) {
    infos.push({ level: 'info', text:
      `${s.contourLevels} levels only subdivide the choice of cutoff — the underlying ` +
      `field is unchanged. Thresholds are spaced evenly in value, not in spot count, so ` +
      `the upper bands stay small on a skewed field.` })
  }

  return [...warns, ...infos]
}

export interface ModuleHighs { name: string; highCount: number }

// Above this share of the tissue a module stops distinguishing anything.
const BROAD_FRACTION = 0.5
// Two modules this broad overlap over most spots, so the fused annotation is
// mostly the neighbour vote rather than the contours under review.
const OVERLAP_FRACTION = 0.4
// Past this the vote reaches well beyond the local neighbourhood.
const WIDE_PROFILE_K = 50

/**
 * Whether the chosen cutoffs can actually produce a tissue map.
 *
 * A module high nowhere contributes nothing; a module high everywhere wins
 * conflicts by size rather than by signal, which is the quiet way a fused
 * annotation turns into one label with decoration.
 */
export function adviseModules(
  modules: ModuleHighs[], nSpots: number, profileK: number,
): Advice[] {
  const out: Advice[] = []
  if (modules.length === 0 || nSpots <= 0) return out

  const frac = (m: ModuleHighs) => m.highCount / nSpots

  for (const m of modules) {
    if (m.highCount === 0) {
      out.push({ level: 'warn', text:
        `${m.name} has no spots above its cutoff, so it cannot contribute a tissue. ` +
        `Lower the cutoff, or drop the set.` })
    } else if (frac(m) > BROAD_FRACTION) {
      out.push({ level: 'warn', text:
        `${m.name} is high in ${Math.round(frac(m) * 100)}% of spots. A module high ` +
        `nearly everywhere wins overlaps by size rather than by signal.` })
    }
  }

  const broad = modules.filter((m) => frac(m) > OVERLAP_FRACTION).map((m) => m.name)
  if (broad.length >= 2) {
    out.push({ level: 'info', text:
      `${broad.join(' and ')} are each high in over ${OVERLAP_FRACTION * 100}% of spots, ` +
      `so most assignments will come from the neighbour vote rather than from the ` +
      `contours themselves.` })
  }

  if (profileK > WIDE_PROFILE_K) {
    out.push({ level: 'info', text:
      `Profile k of ${profileK} votes over a wide neighbourhood, which pulls ambiguous ` +
      `spots toward whatever is regionally dominant rather than locally adjacent.` })
  }

  return out
}
