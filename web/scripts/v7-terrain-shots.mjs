// Screenshots of the v7 terrain features (flora, ice, geysers, dunes, landforms, pits, canyon,
// boulders) and the legend, through the real UI against a sim with the terrain flags on.
// Turns Reality on, then flies the debug camera to the first feature of each kind. Prints the
// legend box against the viewport and any console error or warning (exit 1 on one).
// Usage: node scripts/v7-terrain-shots.mjs [url] [outDir]
// Env: CHROME, WIDTH (1280), HEIGHT (720), PREFIX (file name prefix), GRAPHICS (low|high)
import { chromium } from 'playwright-core'
import fs from 'node:fs'
import path from 'node:path'

const url = process.argv[2] || 'http://localhost:5213'
const outDir = process.argv[3] || 'screenshots'
const width = Number(process.env.WIDTH || 1280)
const height = Number(process.env.HEIGHT || 720)
const prefix = process.env.PREFIX || 'v7t'
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
const context = await browser.newContext({ viewport: { width, height }, deviceScaleFactor: 1 })
const page = await context.newPage()
page.setDefaultTimeout(30000)
page.on('console', (msg) => {
  if (msg.type() !== 'error' && msg.type() !== 'warning') return
  const t = msg.text()
  if (/GPU stall due to ReadPixels|THREE\.Clock: This module has been deprecated/.test(t)) return
  problems.push(`[console.${msg.type()}] ${t}`)
})
page.on('pageerror', (err) => problems.push(`[pageerror] ${err.message}`))
await page.goto(`${url}/?debug`, { waitUntil: 'domcontentloaded' })
await page.waitForFunction(() => window.__redshift && window.__redshift.state().rovers.r2, null, { timeout: 90000 })
await page.waitForTimeout(4000)
if (process.env.GRAPHICS) await page.evaluate((g) => window.__redshiftGraphics?.(g), process.env.GRAPHICS)

const shot = async (name) => {
  const file = path.join(outDir, `${prefix}-${name}.png`)
  await page.screenshot({ path: file })
  console.log(`saved ${file}`)
}

await shot('overview-plain')
const legend = await page.locator('.hud-legend').boundingBox()
console.log(`legend box ${JSON.stringify(legend)} viewport ${width}x${height} fits ${legend ? legend.y >= 0 && legend.y + legend.height <= height : 'n/a'}`)
await page.locator('.hud-legend').screenshot({ path: path.join(outDir, `${prefix}-legend.png`) })
await page.keyboard.press('r')
await page.waitForTimeout(1500)
await shot('overview-reality')

const f = await page.evaluate(() => {
  const s = window.__redshift.state()
  const pick = (a) => (Array.isArray(a) && a.length ? a : [])
  return {
    flora: pick(s.features.flora),
    ice: pick(s.features.ice),
    geysers: pick(s.features.geysers),
    dunes: pick(s.features.dunes),
    mesas: pick(s.features.mesas),
    pits: pick(s.features.pits),
    boulders: pick(s.features.boulder_fields),
    canyon: s.features.canyon || null,
    fleet: { ice_zones: s.fleet.ice_zones, flora_zones: s.fleet.flora_zones, geyser_zones: s.fleet.geyser_zones },
    config: s.config,
  }
})
console.log(`features: flora ${f.flora.length}, ice ${f.ice.length}, geysers ${f.geysers.length}, dunes ${f.dunes.length}, mesas ${f.mesas.length}, pits ${f.pits.length}, boulder fields ${f.boulders.length}, canyon ${f.canyon ? 'yes' : 'no'}`)
console.log(`fleet zones: ${JSON.stringify(f.fleet)}`)

const close = async (name, x, y, dist, el = 32, az = 20) => {
  await page.evaluate(([x, y, az, el, d]) => window.__redshift.view(x, y, az, el, d), [x, y, az, el, dist])
  await page.waitForTimeout(1200)
  await shot(name)
}
const tree = f.flora.find((p) => p.kind === 'tree')
if (tree) await close('flora-grove', tree.x, tree.y, 14 + 4 * tree.height)
const bush = f.flora.find((p) => p.kind === 'bush')
if (bush) await close('flora-bush', bush.x, bush.y, 7)
if (f.ice[0]) await close('ice', f.ice[0].x, f.ice[0].y, 4 * f.ice[0].radius, 30)
if (f.ice[0]) await close('ice-low', f.ice[0].x, f.ice[0].y, 3 * f.ice[0].radius, 10)
if (f.boulders[0]) await close('boulders', f.boulders[0].x, f.boulders[0].y, 3 * f.boulders[0].radius, 28)
if (f.geysers[0]) await close('geyser', f.geysers[0].x, f.geysers[0].y, 18, 24)
if (f.dunes[0]) await close('dunes', f.dunes[0].x, f.dunes[0].y, 3 * f.dunes[0].radius, 30)
if (f.mesas[0]) await close('mesa', f.mesas[0].x, f.mesas[0].y, 4 * f.mesas[0].radius, 22)
if (f.pits[0]) await close('pit', f.pits[0].x, f.pits[0].y, 5 * f.pits[0].radius, 45)
if (f.canyon && Array.isArray(f.canyon.path) && f.canyon.path.length) {
  const m = f.canyon.path[Math.floor(f.canyon.path.length / 2)]
  await close('canyon', m[0], m[1], 60, 40)
}
const iz = Array.isArray(f.fleet.ice_zones) ? f.fleet.ice_zones[0] : null
if (iz) await close('ice-zone', iz[0], iz[1], 4 * iz[2], 40)

await browser.close()
if (problems.length) {
  console.error(problems.join('\n'))
  process.exit(1)
}
console.log('no console errors')
