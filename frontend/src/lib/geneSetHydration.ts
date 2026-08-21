/** Reconcile a stored gene-set payload with the categories this build knows.
 *
 * The backend gene-set store is deliberately dumb — it round-trips whatever the
 * frontend last sent, without interpreting it. That is the right split, but it
 * means a payload written by a different build can arrive here: one saved
 * before a category existed lacks that key entirely, and the panel, which walks
 * a fixed CATEGORY_ORDER, then reads `.geneSets` off undefined and takes the
 * whole Gene panel down with it (there is no error boundary above it).
 *
 * So hydration overlays the payload on the defaults rather than replacing them:
 * every known category is present afterwards, whatever the payload held.
 */

const isRecord = (v: unknown): v is Record<string, unknown> =>
  typeof v === 'object' && v !== null && !Array.isArray(v)

export function mergeHydratedCategories<T extends { folders: unknown[]; geneSets: unknown[] }>(
  defaults: Record<string, T>,
  payload: unknown,
): Record<string, T> {
  if (!isRecord(payload)) return defaults

  const out = { ...defaults }
  for (const key of Object.keys(defaults)) {
    const stored = payload[key]
    if (!isRecord(stored)) continue
    // A category that lost its arrays still carries the user's sets in
    // whatever fields survived, so repair it rather than discarding it.
    out[key] = {
      ...defaults[key],
      ...stored,
      folders: Array.isArray(stored.folders) ? stored.folders : [],
      geneSets: Array.isArray(stored.geneSets) ? stored.geneSets : [],
    } as T
  }
  return out
}
