/**
 * Which loaded dataset is the spatial reference, and which is being localized.
 *
 * Localize is the one feature that reads two datasets at once, and the roles
 * used to be implicit: the query was whatever slot happened to be active, and
 * the reference defaulted to the literal string 'secondary'. Both assumptions
 * fail on the natural workflow — load the spatial data first, look at it, then
 * try to map onto it — and the resulting error blamed the coordinates rather
 * than the arrangement.
 *
 * Deriving the roles from the data fixed that for two datasets, and broke once
 * there could be more than two: with two ST and two SC loaded, the reference
 * was pinned to whichever ST came first and the second pair was unreachable.
 * So the derivation is now only a *default*. The caller states both roles and
 * can override either; this module says which datasets are eligible for each.
 */

export interface RefSlot {
  slot: string
  filename: string
  n_cells: number
  n_genes: number
  has_spatial: boolean
  spatial_key: string | null
  /** Server's view, relative to whichever slot it was asked about. Deliberately
   *  unused here — depending on it is what tied the answer to the active slot. */
  is_query: boolean
}

export interface Roles {
  referenceSlot: string | null
  querySlot: string | null
  /** Why no assignment was possible, phrased as the thing that is actually
   *  true. Null when roles were assigned. */
  problem: string | null
  /** Datasets that could serve as the reference: the ones with coordinates. */
  referenceOptions: RefSlot[]
  /** Datasets that could serve as the query, given the chosen reference.
   *  Everything else — mapping tissue onto tissue is a real thing to want; only
   *  mapping a dataset onto itself is refused, and the backend refuses it too. */
  queryOptions: RefSlot[]
}

const NONE = (problem: string): Roles => ({
  referenceSlot: null, querySlot: null, problem,
  referenceOptions: [], queryOptions: [],
})

/**
 * @param refs         every loaded dataset, from /localize/suggest
 * @param preferRef    slot the user chose as the reference. Ignored when that
 *                     slot has no coordinates.
 * @param preferQuery  slot the user chose as the query. Ignored when it is not
 *                     loaded, or is the reference.
 */
export function assignRoles(
  refs: RefSlot[],
  preferRef?: string | null,
  preferQuery?: string | null,
): Roles {
  if (refs.length === 0) {
    return NONE('No dataset is loaded.')
  }

  const spatial = refs.filter((r) => r.has_spatial)

  if (spatial.length === 0) {
    return NONE(
      'No loaded dataset has spatial coordinates, so there is nothing to '
      + "borrow a tissue map from. Localize needs a spatial dataset in .obsm['spatial'] "
      + "or .obsm['X_spatial'].",
    )
  }

  if (refs.length < 2) {
    // The old message claimed nothing had coordinates, which was the opposite
    // of the truth — name what was found so the user knows it was seen.
    const only = spatial[0]
    return NONE(
      `${only.filename} has spatial coordinates`
      + `${only.spatial_key ? ` (${only.spatial_key})` : ''}, but Localize needs a `
      + 'second dataset — the dissociated cells to place on it. Load one into the '
      + 'other slot.',
    )
  }

  // Honour an explicit choice only when it can actually serve.
  const chosen = preferRef && spatial.some((r) => r.slot === preferRef)
    ? spatial.find((r) => r.slot === preferRef)!
    : spatial[0]

  const queryOptions = refs.filter((r) => r.slot !== chosen.slot)
  if (queryOptions.length === 0) {
    return NONE('Localize needs two different datasets.')
  }

  // Default to dissociated cells: placing them on a tissue map is what this is
  // for, and picking the other tissue instead — which loading ST, ST, SC, SC
  // used to do — is never what was meant.
  // `??` would not coalesce the empty string a cleared <select> yields, so the
  // explicit choice is resolved before the fallbacks rather than beside them.
  const explicit = preferQuery
    ? queryOptions.find((r) => r.slot === preferQuery)
    : undefined
  const query = explicit
    ?? queryOptions.find((r) => !r.has_spatial)
    ?? queryOptions[0]

  return {
    referenceSlot: chosen.slot,
    querySlot: query.slot,
    problem: null,
    referenceOptions: spatial,
    queryOptions,
  }
}
