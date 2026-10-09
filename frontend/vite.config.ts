import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

const BACKEND = process.env.VITE_BACKEND_URL ?? 'http://localhost:8000'

// The FastAPI backend serves routes at the root; proxy each API prefix.
const apiPaths = [
  '/health',
  '/users',
  '/songs',
  '/artists',
  '/search',
  '/events',
  '/feedback',
  '/sessions',
]

export default defineConfig({
  plugins: [react()],
  server: {
    port: 5173,
    proxy: Object.fromEntries(apiPaths.map((p) => [p, { target: BACKEND, changeOrigin: true }])),
  },
  build: {
    outDir: 'dist',
    sourcemap: false,
  },
})
