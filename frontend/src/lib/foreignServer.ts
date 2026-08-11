/**
 * Recognise a reply that did not come from xcell's backend.
 *
 * The proxy in `vite.config.ts` handles the backend being *absent*
 * (ECONNREFUSED/ETIMEDOUT). It cannot handle the backend port being taken by
 * something else: a foreign dev server answers `/api/*` with its own SPA shell
 * at HTTP 200, so every `if (!response.ok)` guard passes and `response.json()`
 * is what finally fails — reporting `Unexpected token '<'`, which points at
 * the parser rather than at the port collision that caused it.
 *
 * 8000 and 5173 are ordinary defaults, so this collides with whatever else the
 * user happens to be running. Naming the intruder is the whole diagnostic.
 */

/** xcell's backend port, per the `XCELL_BACKEND` default in `vite.config.ts`. */
const BACKEND_PORT = 8000

/**
 * Returns an actionable message when a response body is not JSON, or `null`
 * when it is fine. A missing content type (204 and friends) is left alone.
 */
export function describeNonJsonResponse(
  contentType: string | null,
  body: string
): string | null {
  if (!contentType) return null
  if (contentType.includes('json')) return null

  const title = /<title[^>]*>([^<]*)<\/title>/i.exec(body)?.[1]?.trim()
  const who = title ? `A server calling itself "${title}"` : 'Another server'

  return (
    `${who} answered xcell's API with HTML instead of JSON — something other ` +
    `than xcell's backend is listening on port ${BACKEND_PORT}. Find it with ` +
    `\`lsof -nP -iTCP:${BACKEND_PORT} -sTCP:LISTEN\`, stop it, then start the ` +
    'backend with `pixi run backend`.'
  )
}

/**
 * Throw the explanation above if `response` did not come from xcell's backend.
 *
 * Checks the header before touching the body, so a JSON response is left
 * unread and the caller can still call `.json()` on it.
 */
export async function assertJsonResponse(response: Response): Promise<void> {
  const contentType = response.headers.get('content-type')
  if (!contentType || contentType.includes('json')) return
  const foreign = describeNonJsonResponse(contentType, await response.text())
  if (foreign) throw new Error(foreign)
}
