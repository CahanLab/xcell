/**
 * Composing a "save as" path from a browsed directory and a typed filename.
 *
 * Picking where to *write* a file is a different job from picking one to read:
 * the directory comes from navigation, the name from the keyboard, and neither
 * exists yet. These helpers keep the two halves joinable in both directions so
 * a path field can stay the single source of truth.
 */

/**
 * Join a directory and filename into an absolute path, or '' if either is
 * missing.
 *
 * `ext` is what a bare name gets appended, so the field shows the filename that
 * will actually be written rather than surprising someone afterwards. Pass a
 * list — `['.pkl', '.pickle']` — where several suffixes are equally valid: the
 * first is appended, any of them counts as already present.
 *
 * There is deliberately no default. A silent fallback is exactly how an h5ad
 * ends up named `.pkl`.
 */
export function composeSavePath(
  dir: string, filename: string, ext: string | string[],
): string {
  const d = dir.trim()
  const name = filename.trim()
  if (!d || !name) return ''
  const exts = (Array.isArray(ext) ? ext : [ext]).filter(Boolean)
  const lower = name.toLowerCase()
  const has = exts.length === 0 || exts.some((e) => lower.endsWith(e.toLowerCase()))
  const withExt = has ? name : `${name}${exts[0]}`
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
