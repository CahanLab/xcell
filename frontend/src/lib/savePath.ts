/**
 * Composing a "save as" path from a browsed directory and a typed filename.
 *
 * Picking where to *write* a file is a different job from picking one to read:
 * the directory comes from navigation, the name from the keyboard, and neither
 * exists yet. These helpers keep the two halves joinable in both directions so
 * a path field can stay the single source of truth.
 */

const PICKLE_SUFFIXES = ['.pkl', '.pickle']

function hasPickleSuffix(name: string): boolean {
  const lower = name.toLowerCase()
  return PICKLE_SUFFIXES.some((ext) => lower.endsWith(ext))
}

/**
 * Join a directory and filename into an absolute path, or '' if either is
 * missing. A name without a pickle extension gets `.pkl`, matching what the
 * backend does anyway — better to show it than to surprise someone with a
 * different filename than the one they typed.
 */
export function composeSavePath(dir: string, filename: string): string {
  const d = dir.trim()
  const name = filename.trim()
  if (!d || !name) return ''
  const withExt = hasPickleSuffix(name) ? name : `${name}.pkl`
  const base = d.endsWith('/') ? d.slice(0, -1) : d
  return `${base}/${withExt}`
}

/** Inverse of {@link composeSavePath}, for seeding the browser from a path. */
export function splitSavePath(path: string): { dir: string; name: string } {
  const p = path.trim()
  if (!p) return { dir: '', name: '' }
  const idx = p.lastIndexOf('/')
  if (idx < 0) return { dir: '', name: p }
  return {
    // A file directly under / keeps '/' as its directory rather than ''.
    dir: idx === 0 ? '/' : p.slice(0, idx),
    name: p.slice(idx + 1),
  }
}
