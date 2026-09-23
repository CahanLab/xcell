/**
 * Keeping "Select cells by expression" honest about which run it is showing.
 *
 * The modal is mounted unconditionally in App and hidden by returning null
 * *after* its hooks, so closing it does not unmount it: whatever local state
 * the last run left behind is still there when it reopens on another gene
 * set. The outcome of an Apply is the state that matters — while it says
 * "success" the footer shows counts and offers only Open Diff Exp / Close, so
 * a stale one both reports the wrong numbers and hides the Apply button
 * entirely, making a second annotation impossible.
 *
 * So an outcome is recorded against the run that produced it — the source
 * object the modal was opened with, plus the inputs that defined the run —
 * and is only shown while both still match.
 */

export interface RunInputs {
  mode: string
  lo: number
  hi: number
  action: string
  labelContext: string
  annotationName: string
  highLabel: string
  lowLabel: string
}

/** A fingerprint of the inputs one Apply ran with. */
export function runSignature(i: RunInputs): string {
  return JSON.stringify([
    i.mode, i.lo, i.hi, i.action, i.labelContext,
    i.annotationName, i.highLabel, i.lowLabel,
  ])
}

/**
 * Whether a recorded outcome still describes what the modal is showing.
 *
 * The source is compared by identity, not by content: every "Select cells…"
 * click builds a fresh source object, so reopening the same gene set is a new
 * run and starts clean.
 */
export function isOutcomeCurrent(
  recorded: { source: unknown; signature: string } | null,
  source: unknown,
  signature: string,
): boolean {
  return recorded != null && recorded.source === source && recorded.signature === signature
}
