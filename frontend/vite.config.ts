import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

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
        target: process.env.XCELL_BACKEND || 'http://127.0.0.1:8000',
        changeOrigin: true,
      },
    },
  },
})
