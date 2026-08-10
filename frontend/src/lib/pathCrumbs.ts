/**
 * Turning a directory path into something you can read at a glance.
 *
 * Research data lives deep. A real path from this codebase's own test runs is
 * eleven segments and 140 characters:
 *
 *   /Users/pcahan/CahanLab Dropbox/Patrick Cahan/PC/PC/projects/SJD/data/
 *   spatial/xcell/h5ad_empty_removed/best_per_stage
 *
 * Rendered in full it overflows the dialog and scrolls horizontally, anchored
 * left — so what you see is `/Users/pcahan/CahanLab Dro…` and the one segment
 * that matters, the folder you are actually in, is off-screen.
 *
 * Two decisions follow, and they are the whole module:
 *
 * **Elide the middle, never the ends.** The tail is where you are, the head is
 * how you get back out, and the middle is traversal nobody reads. Keeping both
 * ends and collapsing the rest makes the bar a fixed height that never scrolls.
 *
 * **Say `Home`, not `/Users/pcahan`.** The browser already knows the user's
 * shortcuts, so a path underneath one can be named in terms of it. That is not
 * a generic breadcrumb doing generic truncation — it is this app using what it
 * already knows about where you keep things.
 */

export interface Crumb {
  /** What to show. A shortcut's name where one matched, else the folder name. */
  label: string
  /** Where clicking goes. */
  path: string
}

export interface CrumbTrail {
  /** The crumbs to render, in order, already collapsed. */
  crumbs: Crumb[]
  /**
   * The middle segments that were collapsed away, in order, so the caller can
   * offer them behind a "…". Empty when nothing was hidden.
   */
  hidden: Crumb[]
}

/** Longest shortcut that is a prefix of `path`, or null. */
function matchingShortcut(
  path: string,
  shortcuts: { name: string; path: string }[],
): { name: string; path: string } | null {
  let best: { name: string; path: string } | null = null
  for (const sc of shortcuts) {
    if (!sc.path) continue
    // A prefix has to end at a boundary: /Users/pat must not match /Users/patricia.
    if (path === sc.path || path.startsWith(sc.path.replace(/\/$/, '') + '/')) {
      if (!best || sc.path.length > best.path.length) best = sc
    }
  }
  return best
}

/**
 * Build the trail for a directory.
 *
 * @param current absolute directory path, e.g. `/Users/pat/data/spatial`
 * @param shortcuts the browser's named locations, longest match wins
 * @param maxVisible how many crumbs to show before collapsing the middle
 */
export function buildCrumbs(
  current: string | null | undefined,
  shortcuts: { name: string; path: string }[] = [],
  maxVisible = 4,
): CrumbTrail {
  if (!current) return { crumbs: [], hidden: [] }

  const sc = matchingShortcut(current, shortcuts)
  const root: Crumb = sc
    ? { label: sc.name, path: sc.path }
    : { label: '/', path: '/' }

  const rest = sc ? current.slice(sc.path.replace(/\/$/, '').length) : current
  const parts = rest.split('/').filter(Boolean)

  const all: Crumb[] = [root]
  let acc = sc ? sc.path.replace(/\/$/, '') : ''
  for (const part of parts) {
    acc = `${acc}/${part}`
    all.push({ label: part, path: acc })
  }

  if (all.length <= maxVisible) return { crumbs: all, hidden: [] }

  // Keep the anchor and the deepest few. The anchor is how you get out; the
  // tail is where you are. Everything between is the part nobody reads.
  const tail = all.slice(all.length - (maxVisible - 1))
  return { crumbs: [all[0], ...tail], hidden: all.slice(1, all.length - (maxVisible - 1)) }
}
