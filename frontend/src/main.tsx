import React from 'react'
import ReactDOM from 'react-dom/client'
import App from './App'
import { useStore } from './store'

// Load config defaults BEFORE the first render. Several modals are always mounted
// (they render null until opened) and read their defaults via cfgDefault(...) in a
// useState initializer that runs once at mount — so the config must already be in
// the store by then, otherwise a user's overrides for those modals wouldn't apply.
// Failure or timeout falls back to the hardcoded defaults (App re-tries on mount).
async function preloadConfigDefaults(): Promise<void> {
  try {
    const controller = new AbortController()
    const timer = setTimeout(() => controller.abort(), 3000)
    const res = await fetch('/api/config/defaults', { signal: controller.signal })
    clearTimeout(timer)
    if (!res.ok) return
    const data = await res.json()
    if (data && typeof data.config === 'object' && data.config !== null) {
      useStore.getState().setUserConfig(data.config as Record<string, unknown>)
      useStore.getState().applyConfigDefaults()
    }
  } catch {
    /* no config file / backend not ready → hardcoded defaults apply */
  }
}

preloadConfigDefaults().finally(() => {
  ReactDOM.createRoot(document.getElementById('root')!).render(
    <React.StrictMode>
      <App />
    </React.StrictMode>,
  )
})
