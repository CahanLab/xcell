// How the datasets are arranged, remembered between visits: which order the
// tabs are in, what the user renamed them to, and how wide and tall the panes
// were dragged. The datasets themselves are not stored — the backend already
// holds those, and the app asks it what is loaded on startup.

const VERSION = 1
const STORAGE_KEY = 'xcell_workspace'

export interface Workspace {
  /** Slot keys in tab order. */
  order: string[]
  /** Slot key → the name the user typed for it. */
  names: Record<string, string>
  /** Flex weights per grid column and per grid row. */
  cols: number[]
  rows: number[]
  /** Whether the datasets were tiled. Pane sizes restored into a single-pane
   *  view would be stored but invisible. */
  tiled: boolean
}

export function encodeWorkspace(w: Workspace): Record<string, unknown> {
  return { version: VERSION, order: w.order, names: w.names, cols: w.cols, rows: w.rows, tiled: w.tiled }
}

const isFiniteWeight = (v: unknown): v is number =>
  typeof v === 'number' && Number.isFinite(v) && v > 0

function weights(v: unknown): number[] {
  if (!Array.isArray(v)) return []
  // All or nothing: a track list with one bad entry would lay panes out at the
  // wrong widths rather than fail, so it is discarded whole.
  return v.every(isFiniteWeight) ? (v as number[]) : []
}

/** Read a stored workspace back, or null if there isn't a usable one.
 *
 * Anything stored is a value from some earlier version of this code, a
 * half-written entry, or another tab's idea of the truth. None of it is worth a
 * blank screen, so every branch here ends in a default rather than a throw.
 */
export function decodeWorkspace(raw: string | null): Workspace | null {
  if (!raw) return null
  let parsed: unknown
  try {
    parsed = JSON.parse(raw)
  } catch {
    return null
  }
  if (typeof parsed !== 'object' || parsed === null || Array.isArray(parsed)) return null
  const o = parsed as Record<string, unknown>
  if (o.version !== VERSION) return null

  const order = Array.isArray(o.order)
    ? o.order.filter((s): s is string => typeof s === 'string')
    : []
  const names: Record<string, string> = {}
  if (typeof o.names === 'object' && o.names !== null) {
    for (const [k, v] of Object.entries(o.names as Record<string, unknown>)) {
      if (typeof v === 'string' && v.trim()) names[k] = v
    }
  }
  return { order, names, cols: weights(o.cols), rows: weights(o.rows), tiled: o.tiled === true }
}

export function saveWorkspace(w: Workspace): void {
  try {
    localStorage.setItem(STORAGE_KEY, JSON.stringify(encodeWorkspace(w)))
  } catch {
    // Private browsing, a full quota, storage disabled — none of it is a reason
    // to interrupt what the user is doing.
  }
}

export function loadWorkspace(): Workspace | null {
  try {
    return decodeWorkspace(localStorage.getItem(STORAGE_KEY))
  } catch {
    return null
  }
}
