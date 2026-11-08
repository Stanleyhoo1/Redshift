// Screenshots of the v7 landforms (mesas, canyon, pits, dunes) and the look-only ground
// detail against a real sim with the v7 terrain flags on. Slope tint off, Reality on, then an
// overview and a closer shot of the first feature of each kind. Exits 1 on a console error.
// Usage: node scripts/terrain-look-check.mjs [url] [outDir]
// Env: CHROME, WIDTH, HEIGHT, WAIT_S (seconds to let the fleet map before the shots),
//   REALITY=0 (leave Reality off), GRAPHICS=low
import { chromium } from 'playwright-core'
import fs from 'node:fs'
import path from 'node:path'

const url = process.argv[2] || 'http://localhost:5212'
const outDir = process.argv[3] || 'screenshots'
const width = Number(process.env.WIDTH || 1400)
const height = Number(process.env.HEIGHT || 820)
const waitS = Number(process.env.WAIT_S || 20)
const candidates = [process.env.CHROME, `${process.env.HOME}/.cache/ms-playwright/chromium-1254/chrome-linux64/chrome`, '/usr/bin/google-chrome', '/usr/bin/chromium'].filter(Boolean)
const executablePath = candidates.find((p) => fs.existsSync(p))
if (!executablePath) {
  console.error('No Chrome or Chromium found. Set CHROME=/path/to/chrome.')
  process.exit(2)
}
fs.mkdirSync(outDir, { recursive: true })
const browser = await chromium.launch({
  executablePath,
  args: ['--use-gl=swiftshader', '--use-angle=swiftshader', '--enable-unsafe-swiftshader', '--ignore-gpu-blocklist', '--no-sandbox'],
})
const problems = []
const page = await (await browser.newContext({ viewport: { width, height }, deviceScaleFactor: 1 })).newPage()
page.setDefaultTimeout(60000)
page.on('console', (msg) => {
  if (msg.type() !== 'error') return
  const t = msg.text()
  if (/GPU stall due to ReadPixels/.test(t)) return
  problems.push(`[console.error] ${t}`)
})
page.on('pageerror', (err) => problems.push(`[pageerror] ${err.message}`))
await page.goto(`${url}/?debug`, { waitUntil: 'domcontentloaded' })
await page.waitForFunction(() => window.__redshift && window.__redshift.state().features, null, { timeout: 60000 })
await page.waitForTimeout(waitS * 1000)
// Slope colours off and Reality on, through the layer toggles (aria-pressed is their state).
for (const [label, want] of [['Slope colours', false], ['Reality', process.env.REALITY !== '0']]) {
  const b = page.locator('button.toggle', { hasText: label }).first()
  if ((await b.getAttribute('aria-pressed')) !== String(want)) await b.click()
}
if (process.env.GRAPHICS === 'low') await page.locator('.gfx-seg button', { hasText: 'Low' }).click()
await page.waitForTimeout(2500)
const info = await page.evaluate(() => {
  const s = window.__redshift.state()
  const f = s.features || {}
  const first = (k) => (Array.isArray(f[k]) && f[k].length ? f[k][0] : null)
  return { mesa: first('mesas'), pit: first('pits'), dune: first('dunes'), canyon: f.canyon || null, counts: Object.fromEntries(Object.entries(f).map(([k, v]) => [k, Array.isArray(v) ? v.length : v ? 1 : 0])) }
})
console.log('features', JSON.stringify(info.counts))
const shot = async (name, x, y, az, el, dist) => {
  await page.evaluate(([a, b, c, d, e]) => window.__redshift.view(a, b, c, d, e), [x, y, az, el, dist])
  await page.waitForTimeout(1800)
  const file = path.join(outDir, `terrain-${name}.png`)
  await page.screenshot({ path: file })
  console.log('saved', file)
}
await shot('overview', 0, 0, 20, 62, 260)
if (info.mesa) await shot('mesa', info.mesa.x, info.mesa.y, 30, 32, info.mesa.radius * 4 + 20)
if (info.canyon?.path?.length) {
  const p = info.canyon.path[Math.floor(info.canyon.path.length / 2)]
  await shot('canyon', p[0], p[1], 40, 45, (info.canyon.width || 10) * 4 + 30)
}
if (info.pit) await shot('pit', info.pit.x, info.pit.y, 10, 50, info.pit.radius * 5 + 15)
if (info.dune) await shot('dune', info.dune.x, info.dune.y, 0, 50, info.dune.radius * 3 + 20)
const r = await page.evaluate(() => Object.values(window.__redshift.state().rovers)[0])
if (r) await shot('detail', r.x + 6, r.y + 3, 30, 30, 9)
await browser.close()
if (problems.length) {
  console.error(problems.join('\n'))
  process.exit(1)
}
