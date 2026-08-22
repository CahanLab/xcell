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
