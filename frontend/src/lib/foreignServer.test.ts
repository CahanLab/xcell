import { describe, it, expect } from 'vitest'
import { assertJsonResponse, describeNonJsonResponse } from './foreignServer'

// The body that actually caused this, from a real session: another project's
// dev server (lab-manager) held :8000, and answered every /api/* path with its
// own SPA shell at HTTP 200. The frontend called .json() on it and reported
// `Unexpected token '<', "<!doctype "... is not valid JSON`, which says
// nothing about the port collision that caused it.
const FOREIGN_SPA = `<!doctype html>
<html lang="en">
  <head>
    <meta charset="UTF-8" />
    <title>Lab Manager</title>
  </head>
  <body><div id="root"></div></body>
</html>`

describe('describeNonJsonResponse', () => {
  it('passes a normal JSON response through', () => {
    expect(describeNonJsonResponse('application/json', '{"n_obs":100}')).toBeNull()
  })

  it('passes JSON with a charset through', () => {
    expect(
      describeNonJsonResponse('application/json; charset=utf-8', '{"n_obs":100}')
    ).toBeNull()
  })

  it('stays quiet when there is no content type, as on a 204', () => {
    expect(describeNonJsonResponse(null, '')).toBeNull()
  })

  it('explains an HTML response instead of letting JSON.parse fail', () => {
    const msg = describeNonJsonResponse('text/html; charset=utf-8', FOREIGN_SPA)
    expect(msg).not.toBeNull()
    expect(msg).toMatch(/HTML/i)
  })

  it('names the server that answered, which is what identifies the intruder', () => {
    const msg = describeNonJsonResponse('text/html; charset=utf-8', FOREIGN_SPA)
    expect(msg).toContain('Lab Manager')
  })

  it('points at the backend port, where the collision actually is', () => {
    const msg = describeNonJsonResponse('text/html; charset=utf-8', FOREIGN_SPA)
    expect(msg).toContain('8000')
  })

  it('still explains itself when the HTML has no title', () => {
    const msg = describeNonJsonResponse('text/html', '<!doctype html><html></html>')
    expect(msg).not.toBeNull()
    expect(msg).toMatch(/HTML/i)
  })
})

// Real Response objects, not mocks — the guard's whole job is reading headers
// and body off one, and a mock would not prove it does that correctly.
describe('assertJsonResponse', () => {
  it('lets a JSON response pass', async () => {
    const res = new Response('{"n_obs":100}', {
      headers: { 'content-type': 'application/json' },
    })
    await expect(assertJsonResponse(res)).resolves.toBeUndefined()
  })

  it('throws the foreign-server explanation on a 200 of HTML', async () => {
    const res = new Response(FOREIGN_SPA, {
      status: 200,
      headers: { 'content-type': 'text/html; charset=utf-8' },
    })
    await expect(assertJsonResponse(res)).rejects.toThrow(/Lab Manager/)
  })

  it('leaves the body readable for the caller', async () => {
    const res = new Response('{"n_obs":100}', {
      headers: { 'content-type': 'application/json' },
    })
    await assertJsonResponse(res)
    expect(await res.json()).toEqual({ n_obs: 100 })
  })
})
