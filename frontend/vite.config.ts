import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

const BACKEND = process.env.XCELL_BACKEND || 'http://127.0.0.1:8000'

// One warning per outage, not one per request — a page load fires many /api
// calls and would otherwise bury the message in repeats.
let warnedUnreachable = false

// https://vitejs.dev/config/
export default defineConfig({
  plugins: [react()],
  server: {
    port: 5173,
    // Proxy API requests to FastAPI backend during development.
    // Use 127.0.0.1, not localhost: modern Node may resolve "localhost" to
    // IPv6 ::1, but uvicorn binds IPv4 127.0.0.1 only -> ECONNREFUSED.
    // Override the target with XCELL_BACKEND to point at a non-default backend.
    proxy: {
      '/api': {
        target: BACKEND,
        changeOrigin: true,
        configure: (proxy) => {
          // Translate the two ways "the backend isn't there" shows up into one
          // actionable line. ETIMEDOUT is the confusing one: `uvicorn --reload`
          // binds and listens in the parent process *before* the worker has
          // imported the app, so connections during startup queue in the kernel
          // backlog unserviced rather than being refused. Node then reports a
          // connect timeout, which reads like a network fault rather than
          // "not ready yet". The first start in a fresh pixi environment takes
          // ~1 minute (cold bytecode caches for the scientific stack) versus a
          // few seconds warm, which is when people actually hit this.
          proxy.on('error', (err, _req, res) => {
            // Structural, not NodeJS.ErrnoException — this file is outside the
            // app's tsconfig and @types/node is not a dependency.
            const code = (err as Error & { code?: string }).code
            if (code === 'ETIMEDOUT' || code === 'ECONNREFUSED') {
              if (!warnedUnreachable) {
                warnedUnreachable = true
                const why = code === 'ETIMEDOUT'
                  ? 'it is still starting up (the port is open but not answering yet)'
                  : 'nothing is listening there'
                console.warn(
                  `\n  [xcell] Backend at ${BACKEND} is not reachable — ${why}.\n` +
                  '          Start it with `pixi run backend` (or `pixi run -e pyscn backend`)\n' +
                  "          and wait for `Application startup complete`, then reload the page.\n" +
                  '          A first start in a fresh environment can take ~1 minute.\n',
                )
              }
              // Answer the browser instead of leaving the request hanging until
              // the fetch itself times out.
              if (res && 'writeHead' in res && !res.headersSent) {
                res.writeHead(503, { 'Content-Type': 'application/json' })
                res.end(JSON.stringify({
                  detail: `Backend at ${BACKEND} is not reachable yet. If you just started it, `
                    + 'wait for "Application startup complete" and reload.',
                }))
              }
              return
            }
            console.error(`  [xcell] proxy error (${code ?? 'unknown'}):`, err.message)
          })
          // Reset once the backend answers, so a later outage warns again.
          proxy.on('proxyRes', () => { warnedUnreachable = false })
        },
      },
    },
  },
})
