// Screenshots of the v7 geysers (Geysers.jsx) against a real sim with GEYSERS on. Turns Reality on,
// flies the camera to each vent, waits for an eruption at the first one and shoots it erupting.
// Exits 1 on any console error or warning, or when the sim sends no geysers.
// Usage: node scripts/v7-geysers-check.mjs [url] [outDir]
// Env: CHROME, WIDTH, HEIGHT, WAIT_S (longest wait for an eruption, wall seconds, default 90),
// LOW=1 (switch Graphics to Low first)
import { chromium } from 'playwright-core'
import fs from 'node:fs'
import path from 'node:path'

const url = process.argv[2] || 'http://localhost:5211'
const outDir = process.argv[3] || 'screenshots'
const width = Number(process.env.WIDTH || 1400)
const height = Number(process.env.HEIGHT || 800)
const waitS = Number(process.env.WAIT_S || 90)
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
// A git worktree whose node_modules is a symlink makes vite refuse the font files (403). That is
// the checkout, not the scene, so those are only reported.
let fontBlocked = 0
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
page.on('response', (r) => {
  if (r.status() < 400) return
  if (/\.woff2?$/.test(r.url())) fontBlocked++
  console.log(`HTTP ${r.status()} ${r.url()}`)
})
await page.goto(`${url}/?debug`, { waitUntil: 'domcontentloaded' })
await page.waitForFunction(() => window.__redshift && window.__redshift.state().rovers.r2, null, { timeout: 60000 })
await page.waitForTimeout(2500)

const shot = async (name) => {
  const p = path.join(outDir, `geysers-${process.env.LOW === '1' ? 'low-' : ''}${name}.png`)
  await page.screenshot({ path: p })
  console.log(`saved ${p}`)
}
const vents = await page.evaluate(() => window.__redshift.state().features?.geysers || [])
const zones = await page.evaluate(() => window.__redshift.state().fleet?.geyser_zones || [])
console.log(`vents: ${JSON.stringify(vents)}`)
console.log(`zones: ${JSON.stringify(zones)}`)
let failed = false
if (!vents.length) {
  console.error('FAIL: the sim sent no geysers')
  failed = true
} else {
  if (process.env.LOW === '1') await page.getByRole('button', { name: 'Low', exact: true }).click()
  await page.keyboard.press('r')
  await page.waitForTimeout(800)
  for (const v of vents.slice(0, 3)) {
    await page.evaluate(([x, y]) => window.__redshift.view(x, y, 30, 25, 22), [v.x, v.y])
    await page.waitForTimeout(900)
    await shot(`${v.id}-${v.active ? 'active' : 'idle'}`)
  }
  // Park the camera on the idle vent due next, wait for it to erupt, then shoot the column as it
  // builds and once more from close in (at SIM_SPEED=6 an eruption lasts only 1.3 to 2.5 wall s).
  const geysers = () => page.evaluate(() => window.__redshift.state().features?.geysers || [])
  const due = (await geysers()).filter((g) => !g.active).sort((a, b) => a.next_s - b.next_s)[0]
  let hit = null
  if (due) {
    await page.evaluate(([x, y, h]) => window.__redshift.view(x, y, 20, 14, 10 + 1.6 * h), [due.x, due.y, due.height])
    const t0 = Date.now()
    while (!hit && Date.now() - t0 < waitS * 1000) {
      hit = (await geysers()).find((g) => g.id === due.id && g.active) || null
      if (!hit) await page.waitForTimeout(150)
    }
  }
  if (hit) {
    await page.waitForTimeout(1000)
    await shot(`${hit.id}-erupting`)
    await page.evaluate(([x, y]) => window.__redshift.view(x, y, 200, 35, 11), [hit.x, hit.y])
    await page.waitForTimeout(150)
    await shot(`${hit.id}-erupting-close`)
    console.log(`eruption at ${hit.id}, height ${hit.height} m`)
  } else console.log(`no eruption within ${waitS} s`)
  // Reality off: only the vents (and keep-out rings) on ground the fleet knows.
  const z = zones[0]
  await page.keyboard.press('r')
  if (z) await page.evaluate(([x, y, r]) => window.__redshift.view(x, y, 10, 50, 4 * r), z)
  else await page.evaluate(([x, y]) => window.__redshift.view(x, y, 10, 50, 30), [vents[0].x, vents[0].y])
  await page.waitForTimeout(800)
  await shot(z ? 'zone-reality-off' : 'reality-off')
}
for (let k = 0; k < fontBlocked; k++) {
  const i = problems.findIndex((p) => p.includes('Failed to load resource'))
  if (i >= 0) problems.splice(i, 1)
}
if (problems.length) {
  failed = true
  for (const p of problems) console.error(p)
}
await browser.close()
process.exit(failed ? 1 : 0)
