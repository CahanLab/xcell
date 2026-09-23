/**
 * The optional cap on how many associated genes a line run returns.
 *
 * The cap used to be mandatory (50 per direction, or per module), which
 * silently threw away most of a large result. It is now a filter: an empty
 * field means "every significant gene", and `null` is what travels to the
 * backend for that. Kept outside the modals so both the single-line and the
 * multi-line panel read the field the same way.
 */

/** Above this, a cap is not filtering anything a user can read anyway. */
export const GENE_CAP_MAX = 5000

export function parseGeneCap(raw: string): number | null {
  const text = raw.trim()
  if (text === '') return null
  const n = Math.floor(Number(text))
  if (!Number.isFinite(n) || n <= 0) return null
  return Math.min(n, GENE_CAP_MAX)
}
