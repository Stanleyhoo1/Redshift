// Camera pipeline throughput without the 3D view: the same steps as useCameraStream (fetch,
// headers, createImageBitmap, draw) in a bare page, for a few seconds. Shows what the stream
// can do when the main thread is free (software GL in CI starves it inside the app).
// Usage: node scripts/cam-rate.mjs [url] [w] [h]   (mock or sim behind the preview proxy)
import { chromium } from 'playwright-core'
import fs from 'node:fs'
const url = process.argv[2] || 'http://localhost:4173'
const w = Number(process.argv[3] || 320)
const h = Number(process.argv[4] || 240)
const exe = [process.env.CHROME, '/opt/pw-browsers/chromium-1243/chrome-linux64/chrome'].filter(Boolean).find((p) => fs.existsSync(p))
const browser = await chromium.launch({ executablePath: exe, args: ['--no-sandbox'] })
const page = await browser.newPage()
await page.goto(`${url}/api/health`)
const r = await page.evaluate(
  async ({ w, h }) => {
    const cv = document.createElement('canvas')
    cv.width = w
    cv.height = h
    const ctx = cv.getContext('2d')
    let frames = 0
    let dups = 0
    let last = null
    let dets = 0
    const t0 = performance.now()
    while (performance.now() - t0 < 5000) {
      const started = performance.now()
      const res = await fetch(`/api/camera/r1.jpg?view=fpv&w=${w}&h=${h}&_=${Date.now()}`, { cache: 'no-store' })
      const no = res.headers.get('X-Frame')
      const d = res.headers.get('X-Detections')
      const blob = await res.blob()
      if (no !== null && no === last) {
        dups++
        await new Promise((ok) => setTimeout(ok, 70))
        continue
      }
      last = no
      const bmp = await createImageBitmap(blob)
      ctx.drawImage(bmp, 0, 0)
      bmp.close()
      frames++
      if (d) dets = JSON.parse(d).length
      const gap = 33 - (performance.now() - started)
      if (gap > 0) await new Promise((ok) => setTimeout(ok, gap))
    }
    return { fps: frames / 5, dups, dets }
  },
  { w, h },
)
console.log(`${w}x${h}: ${r.fps.toFixed(1)} new frames/s, ${r.dups} repeats skipped, ${r.dets} detections in the last frame`)
await browser.close()
