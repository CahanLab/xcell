/**
 * Column-policy bookkeeping for File → Combine datasets.
 *
 * Each .obs / .var column across the inputs gets one of a few outcomes, and
 * the backend rejects a policy naming a column no dataset has — so the seed
 * has to be re-derived whenever the file list changes, not merged blindly.
 */

export type ObsPolicy = 'merge' | 'separate' | 'drop'
export type VarPolicy = 'first' | 'separate' | 'drop'
export type Policy = ObsPolicy | VarPolicy

export interface ColumnInfo {
  name: string
  datasets: string[]
  kind: string
  suggested: Policy
  reason: string
  identical?: boolean
  n_unique?: number | null
}

/**
 * Every offered column mapped to a policy: the user's choice where they made
 * one, the backend's suggestion otherwise. Choices for columns that are no
 * longer offered are dropped, since sending one is an error.
 */
export function seedPolicy(
  columns: ColumnInfo[],
  existing: Record<string, string> = {},
): Record<string, string> {
  const out: Record<string, string> = {}
  for (const col of columns) {
    out[col.name] = existing[col.name] ?? col.suggested
  }
  return out
}

/** "9 merged · 2 kept separate · 1 dropped" — what will happen, in one line. */
export function summarizePolicy(
  columns: ColumnInfo[],
  policy: Record<string, string>,
): string {
  if (columns.length === 0) return 'none'
  let kept = 0
  let separate = 0
  let dropped = 0
  for (const col of columns) {
    const value = policy[col.name] ?? col.suggested
    if (value === 'drop') dropped += 1
    else if (value === 'separate') separate += 1
    else kept += 1
  }
  const parts: string[] = []
  if (kept) parts.push(`${kept} merged`)
  if (separate) parts.push(`${separate} kept separate`)
  if (dropped) parts.push(`${dropped} dropped`)
  return parts.join(' · ')
}
