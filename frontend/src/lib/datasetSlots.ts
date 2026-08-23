/** Just enough of a dataset to tell whether a slot is holding one. */
type SlotContents = Record<string, { schema: unknown } | undefined>

/** The slots that actually have a dataset in them, in the order they were created.
 *
 * A slot can exist and be empty: 'secondary' is there from startup with a null
 * schema, long before anyone loads anything into it.
 */
export function loadedSlots(datasets: SlotContents): string[] {
  return Object.keys(datasets).filter((slot) => datasets[slot]?.schema != null)
}

/** Which datasets a colouring applied to the active one should be copied onto.
 *
 * Replaces `otherSlot()`, which answered "the one that isn't this one" — a
 * question with no answer once a third dataset is loaded, and one that would
 * have quietly kept mirroring onto whichever slot happened to be named second.
 */
export function mirrorTargets(activeSlot: string, datasets: SlotContents): string[] {
  return loadedSlots(datasets).filter((slot) => slot !== activeSlot)
}

/** The key for a newly loaded dataset.
 *
 * 'primary' has to keep its name — `appendDataset()` omits `?dataset=`
 * entirely for it — and 'secondary' is kept for the second out of habit and
 * compatibility. After that they are numbered. The first *free* name wins, so
 * unloading one and loading another reuses the gap instead of climbing forever.
 */
export function nextSlotKey(existing: string[]): string {
  const taken = new Set(existing)
  if (!taken.has('primary')) return 'primary'
  if (!taken.has('secondary')) return 'secondary'
  for (let n = 3; n < 1000; n++) {
    const key = `slot${n}`
    if (!taken.has(key)) return key
  }
  throw new Error('no free dataset slot')
}

// The extensions xcell loads. Anything else — a 10x matrix directory, say — is
// a name in its own right and is left alone.
const DATA_EXTENSIONS = ['.h5ad', '.h5', '.rds']

/** What to call a dataset on screen.
 *
 * The file it came from, without its directory or extension. Four tabs reading
 * Primary, Secondary, Slot3, Slot4 tell you nothing; four reading
 * `E11.5_hindlimbs_102424` and `GSM8155797_scMm_Shox2trac_HL_E115_rep2` do.
 * Only a known extension is allowed to end the name — real filenames here carry
 * version dots ('E11.5') that are part of what the dataset is called.
 */
export function datasetLabel(slot: string, filename?: string, displayName?: string | null): string {
  const typed = (displayName ?? '').trim()
  if (typed) return typed
  const base = (filename ?? '').split('/').pop() ?? ''
  if (base) {
    const lower = base.toLowerCase()
    const ext = DATA_EXTENSIONS.find((e) => lower.endsWith(e))
    const stripped = ext ? base.slice(0, -ext.length) : base
    if (stripped) return stripped
  }
  return slot.charAt(0).toUpperCase() + slot.slice(1)
}

/** Which dataset should be showing once `unloaded` is gone.
 *
 * Null means none is left. Closing the dataset you are looking at should land
 * you on its neighbour rather than on whichever slot happens to sort first —
 * the one on the left, or on the right when you closed the leftmost.
 */
export function slotAfterUnload(
  unloaded: string,
  order: string[],
  activeSlot: string,
): string | null {
  if (unloaded !== activeSlot) return activeSlot
  const i = order.indexOf(unloaded)
  const remaining = order.filter((slot) => slot !== unloaded)
  if (remaining.length === 0) return null
  return remaining[Math.max(0, i - 1)]
}

/** Move the boundary between two panes, leaving every other pane where it is.
 *
 * Widths are flex weights, so the arithmetic goes through pixels and back: the
 * pair either side of the divider trade width between them and their sum is
 * preserved, which is what keeps the rest of the row still.
 */
export function resizePanes(
  weights: number[],
  index: number,
  deltaPx: number,
  containerPx: number,
  minPx = 120,
): number[] {
  if (deltaPx === 0) return weights
  if (index < 0 || index + 1 >= weights.length) return weights
  const total = weights.reduce((a, b) => a + b, 0)
  if (total <= 0 || containerPx <= 0) return weights

  const pxPerWeight = containerPx / total
  const aPx = weights[index] * pxPerWeight
  const pairPx = aPx + weights[index + 1] * pxPerWeight
  // Too cramped to honour the minimum on both sides — refuse rather than
  // pick a side.
  if (pairPx < minPx * 2) return weights

  const newAPx = Math.min(Math.max(aPx + deltaPx, minPx), pairPx - minPx)
  const out = [...weights]
  out[index] = newAPx / pxPerWeight
  out[index + 1] = (pairPx - newAPx) / pxPerWeight
  return out
}


/** How to arrange n panes.
 *
 * Two or three stay in one row — that is what a wide screen is for, and it is
 * what side-by-side comparison has always looked like. Past that a row makes
 * every pane a sliver, so they square off instead, which is what spatial data
 * wants. An awkward count leaves a gap rather than an uneven row.
 */
export function paneGrid(n: number): { rows: number; cols: number } {
  if (n <= 0) return { rows: 0, cols: 0 }
  if (n <= 3) return { rows: 1, cols: n }
  const cols = Math.ceil(Math.sqrt(n))
  return { rows: Math.ceil(n / cols), cols }
}

/** Move one item of an array to another index, without mutating it.
 *
 * Out-of-range indices are a drop that landed outside the strip: the order is
 * returned unchanged rather than guessed at.
 */
export function moveItem<T>(items: T[], from: number, to: number): T[] {
  if (from < 0 || from >= items.length) return items
  if (to < 0 || to >= items.length) return items
  if (from === to) return items
  const out = [...items]
  const [moved] = out.splice(from, 1)
  out.splice(to, 0, moved)
  return out
}
