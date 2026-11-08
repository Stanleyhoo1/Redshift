// Takes one picture of each v7 biome for the README: the whole sector with Reality on, the
// dashboard panels hidden. Needs a sim with AUTO_LAUNCH=1 (biomes on, the default from the
// environment) and vite pointed at it.
//   node scripts/biome-shots.mjs [appUrl] [simUrl] [outDir]
// Seeds: BIOME_SEEDS="basalt:4,verdigris:31,..." to change them.
import fs from 'node:fs'
import path from 'node:path'
import { chromium } from 'playwright-core'

const app = process.argv[2] || 'http://localhost:5183'
const sim = process.argv[3] || 'http://localhost:8103'
const outDir = process.argv[4] || 'screenshots/biomes'
const seeds = (process.env.BIOME_SEEDS || 'basalt:4,verdigris:31,frost:1,ash:10,salt:6,obsidian:42')
  .split(',')
  .map((p) => p.split(':'))
  .map(([id, seed]) => ({ id, seed: Number(seed) }))
const width = Number(process.env.WIDTH || 1600)
const height = Number(process.env.HEIGHT || 900)
const candidates = [process.env.CHROME, `${process.env.HOME}/.cache/ms-playwright/chromium-1254/chrome-linux64/chrome`, '/usr/bin/google-chrome', '/usr/bin/chromium'].filter(Boolean)
const executablePath = candidates.find((p) => fs.existsSync(p))
fs.mkdirSync(outDir, { recursive: true })

const browser = await chromium.launch({ executablePath, args: ['--use-gl=angle', '--use-angle=swiftshader', '--ignore-gpu-blocklist'] })
const page = await browser.newPage({ viewport: { width, height }, deviceScaleFactor: 1 })
const problems = []
page.on('pageerror', (e) => problems.push(String(e)))
page.on('console', (m) => {
  if (m.type() === 'error' && !/favicon|Failed to load resource/.test(m.text())) problems.push(m.text())
})

const reset = async (seed) => {
  const r = await fetch(`${sim}/api/reset`, { method: 'POST', headers: { 'content-type': 'application/json' }, body: JSON.stringify({ seed }) })
  if (!r.ok) throw new Error(`reset ${seed}: ${r.status} ${await r.text()}`)
}

await reset(seeds[0].seed)
await page.goto(`${app}/?debug`, { waitUntil: 'load' })
await page.waitForFunction(() => window.__redshift?.state?.().world, null, { timeout: 60000 })
await page.waitForTimeout(3000)
await page.keyboard.press('r') // Reality on: the whole true sector
for (const k of ['f', 'p', 'l']) await page.keyboard.press(k) // frontiers, paths and LiDAR off
await page.addStyleTag({ content: 'body { visibility: hidden !important } canvas { visibility: visible !important }' })

for (const { id, seed } of seeds) {
  if (seed !== seeds[0].seed || id !== seeds[0].id) {
    await reset(seed)
    await page.waitForFunction((s) => window.__redshift.state().world?.seed === s, seed, { timeout: 60000 })
  }
  await page.waitForTimeout(6000)
  const biome = await page.evaluate(() => window.__redshift.state().world?.biome?.id ?? null)
  if (biome !== id) problems.push(`seed ${seed}: expected ${id}, got ${biome}`)
  await page.evaluate(() => window.__redshift.view(0, 0, 25, 40, 122))
  await page.waitForTimeout(3500)
  const file = path.join(outDir, `${id}.png`)
  // Only the 3D canvas (the sidebar sits beside it, not over it).
  const box = await page.locator('canvas').first().boundingBox()
  await page.screenshot({ path: file, clip: box })
  console.log(`saved ${file} (seed ${seed}, biome ${biome})`)
}

await browser.close()
if (problems.length) {
  console.log('problems:\n' + problems.join('\n'))
  process.exit(1)
}
