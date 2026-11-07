import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

// The sim (or the mock) serves /ws and /api on port 8000. Override with SIM_URL,
// e.g. SIM_URL=http://192.168.1.20:8000 npm run dev
const target = process.env.SIM_URL || 'http://localhost:8000'

const proxy = {
  '/ws': { target, ws: true, changeOrigin: true },
  '/api': { target, changeOrigin: true },
}

export default defineConfig({
  // Relative asset URLs, so the build works wherever dist/ is served from.
  base: './',
  plugins: [react()],
  server: { port: 5173, proxy },
  preview: { port: 4173, proxy },
  build: {
    chunkSizeWarningLimit: 1600,
  },
})
