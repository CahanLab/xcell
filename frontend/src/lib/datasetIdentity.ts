import type { DatasetSlot, Schema } from '../store'

/**
 * A string that changes whenever "the data currently in front of me" changes.
 *
 * Panels that hold a computed result in local state need to know when that
 * result stopped describing the loaded data. Watching the active slot alone is
 * not enough: a user can load a different file into the *same* slot, which
 * leaves the slot name unchanged while invalidating everything derived from it.
 *
 * Shape is part of the identity, not just the filename, because operations that
 * add or remove cells and genes invalidate different things — a per-cell
 * classification stops lining up when cells are dropped, and a classifier's
 * gene coverage has to be recomputed when genes are.
 *
 * Two genuinely different files with identical shape and name would collide,
 * which in practice means the same file reloaded — where reusing the result is
 * the right answer anyway.
 */
export function datasetIdentity(slot: DatasetSlot, schema: Schema | null): string {
  if (!schema) return `${slot}:<none>`
  return `${slot}:${schema.filename ?? '<unnamed>'}:${schema.n_cells}:${schema.n_genes}`
}
