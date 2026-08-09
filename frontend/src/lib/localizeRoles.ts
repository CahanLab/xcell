/**
 * Which loaded dataset is the spatial reference, and which is being localized.
 *
 * Localize is the one feature that spans two dataset slots, and the roles used
 * to be implicit: the query was whatever slot happened to be active, and the
 * reference defaulted to the literal string 'secondary'. Both assumptions fail
 * on the natural workflow — load the spatial data first, look at it, then try
 * to map onto it — and the resulting error blamed the coordinates rather than
 * the arrangement.
 *
 * Roles are derived from the data instead: the dataset holding coordinates is
 * the reference, the other is the query. The caller states the choice on screen
 * so it is a visible decision rather than a hidden one.
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
  /** Both datasets carry coordinates, so which is which is a real choice. */
  swappable: boolean
}

const NONE = (problem: string): Roles => ({
  referenceSlot: null, querySlot: null, problem, swappable: false,
})

/**
 * @param refs      every loaded dataset, from /localize/suggest
 * @param preferRef slot the user explicitly chose as the reference, if any.
 *                  Ignored when that slot has no coordinates.
 */
export function assignRoles(refs: RefSlot[], preferRef?: string | null): Roles {
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

  const query = refs.find((r) => r.slot !== chosen.slot)
  if (!query) {
    return NONE('Localize needs two different datasets.')
  }

  return {
    referenceSlot: chosen.slot,
    querySlot: query.slot,
    problem: null,
    swappable: spatial.length >= 2,
  }
}
