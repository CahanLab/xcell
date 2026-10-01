/**
 * Diffusion maps and diffusion pseudotime (Analyze → Cells).
 *
 * A diffusion map is scanpy's layout: column 0 is the stationary state, so the
 * first informative view is DC1 × DC2 and the axes are labelled from DC0.
 * The DPT request is built here from the modal's state, so its shape is pinned
 * by a test and Run cannot send a value its callback forgot to list.
 */

/** `X_diffmap`, or one derived from it (`X_diffmap_<graph>`, `X_diffmap_<subset>`). */
export function isDiffmapKey(name: string): boolean {
  return /^X_diffmap(_|$)/.test(name)
}

export function diffmapAxisLabel(i: number): string {
  return `DC${i}`
}

/** Axes for a map nobody has chosen axes on: DC1 × DC2. Null below three
 * columns — the backend clamps out-of-range dims and useEmbedding would then
 * refetch forever on the mismatch. */
export function defaultDiffmapDims(ncols: number): { x: number; y: number } | null {
  return ncols >= 3 ? { x: 1, y: 2 } : null
}

/** The backend's default pseudotime column for a map (`_dpt_output_name`). */
export function dptOutputName(diffmapKey: string): string {
  if (diffmapKey.startsWith('X_diffmap')) return 'dpt_pseudotime' + diffmapKey.slice('X_diffmap'.length)
  const tail = diffmapKey.startsWith('X_') ? diffmapKey.slice(2) : diffmapKey
  return `dpt_pseudotime_${tail}`
}

/** How the root cell is chosen. `potency` is the stemFinder preset; it reaches
 * the backend as the lowest or highest value of a potency column. */
export type RootMode = 'potency' | 'obs_min' | 'obs_max' | 'group' | 'cells' | 'dc_min' | 'dc_max'

export interface DptInputs {
  diffmapKey: string
  nDcs: string
  rootMode: RootMode
  potencyColumn: string
  potencyDirection: 'min' | 'max'
  /** The column for obs_min / obs_max / group. */
  column: string
  groupValue: string
  component: string
  /** The current selection, for `cells`. */
  selection: number[]
  keyAdded: string
}

export function dptParams(i: DptInputs): Record<string, unknown> {
  const p: Record<string, unknown> = { diffmap_key: i.diffmapKey }
  const n = parseInt(i.nDcs, 10)
  if (Number.isFinite(n)) p.n_dcs = n
  switch (i.rootMode) {
    case 'potency':
      p.root_mode = i.potencyDirection === 'max' ? 'obs_max' : 'obs_min'
      p.root_column = i.potencyColumn
      break
    case 'obs_min':
    case 'obs_max':
      p.root_mode = i.rootMode
      p.root_column = i.column
      break
    case 'group':
      p.root_mode = 'group'
      p.root_column = i.column
      p.root_value = i.groupValue
      break
    case 'cells':
      p.root_mode = 'cells'
      p.root_cells = i.selection
      break
    case 'dc_min':
    case 'dc_max':
      p.root_mode = i.rootMode
      p.root_component = parseInt(i.component, 10)
      break
  }
  const name = i.keyAdded.trim()
  if (name) p.key_added = name
  return p
}

/** Why Run is disabled, or null when it can run. */
export function dptBlocker(i: DptInputs): string | null {
  if (!i.diffmapKey) return 'Run Diffusion map first (Analyze → Cells → Diffusion map).'
  switch (i.rootMode) {
    case 'potency':
      return i.potencyColumn ? null : 'No stemFinder score yet — run Differentiation (stemFinder) first.'
    case 'obs_min':
    case 'obs_max':
      return i.column ? null : 'Pick the column to take the root from.'
    case 'group':
      if (!i.column) return 'Pick the column that holds the group.'
      return i.groupValue ? null : 'Pick the group to start from.'
    case 'cells':
      return i.selection.length ? null : 'Select the root cells on the plot first.'
    case 'dc_min':
    case 'dc_max':
      return Number.isFinite(parseInt(i.component, 10)) ? null : 'Pick a diffusion component.'
  }
  return null
}
