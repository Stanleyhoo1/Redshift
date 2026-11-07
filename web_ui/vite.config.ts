/// <reference types="vitest/config" />
import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'
import tailwindcss from '@tailwindcss/vite'

const BACKEND = process.env.REDSHIFT_BACKEND ?? 'localhost:8000'

const proxy = {
  '/api': { target: `http://${BACKEND}`, changeOrigin: true },
  '/ws': { target: `ws://${BACKEND}`, ws: true, changeOrigin: true },
}

export default defineConfig({
  plugins: [react(), tailwindcss()],
  server: { port: 5173, strictPort: true, proxy },
  preview: { port: 4173, proxy },
  build: {
    target: 'es2022',
    chunkSizeWarningLimit: 1600,
  },
  test: {
    environment: 'node',
    include: ['src/**/*.test.ts'],
  },
})
