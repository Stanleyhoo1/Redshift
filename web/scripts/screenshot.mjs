// Takes screenshots of the running UI with a local Chrome (no browser download).
// Usage: node scripts/screenshot.mjs [url] [outDir] [delays in s, comma separated]
//   e.g. node scripts/screenshot.mjs http://localhost:4173 screenshots 5,30
// Env: CHROME=/path/to/chrome, WIDTH, HEIGHT, ACTIONS=1 to also exercise the controls.
import { chromium } from 'playwright-core'
import fs from 'node:fs'
import path from 'node:path'

const url = process.argv[2] || 'http://localhost:4173'
const outDir = process.argv[3] || 'screenshots'
const delays = (process.argv[4] || '5,30').split(',').map(Number)
const width = Number(process.env.WIDTH || 1600)
const height = Number(process.env.HEIGHT || 900)
const candidates = [process.env.CHROME, '/usr/bin/google-chrome', '/usr/bin/google-chrome-stable', '/usr/bin/chromium', '/usr/bin/chromium-browser'].filter(Boolean)
const executablePath = candidates.find((p) => fs.existsSync(p))
if (!executablePath) {
  console.error('No Chrome or Chromium found. Set CHROME=/path/to/chrome.')
  process.exit(2)
}
fs.mkdirSync(outDir, { recursive: true })

const browser = await chromium.launch({
  executablePath,
  args: ['--use-angle=swiftshader', '--enable-unsafe-swiftshader', '--ignore-gpu-blocklist', '--no-sandbox'],
})
const page = await browser.newPage({ viewport: { width, height }, deviceScaleFactor: 1 })
const problems = []
page.on('console', (msg) => {
  if (msg.type() === 'error' || msg.type() === 'warning') problems.push(`[console.${msg.type()}] ${msg.text()}`)
})
page.on('pageerror', (err) => problems.push(`[pageerror] ${err.message}`))

const t0 = Date.now()
await page.goto(url, { waitUntil: 'domcontentloaded' })
const shot = async (name) => {
  const file = path.join(outDir, name)
  await page.screenshot({ path: file })
  console.log(`saved ${file} at ${((Date.now() - t0) / 1000).toFixed(1)} s`)
}
for (const d of delays) {
  const wait = d * 1000 - (Date.now() - t0)
  if (wait > 0) await page.waitForTimeout(wait)
  await shot(`ui-${d}s.png`)
}

if (process.env.ACTIONS) {
  // Exercise the controls: Reality on, slope colours off, follow rover 1, then reset.
  await page.getByRole('button', { name: /^Reality/ }).click()
  await page.getByRole('button', { name: /^Slope colours/ }).click()
  await page.waitForTimeout(2500)
  await shot('ui-reality-height.png')
  await page.getByRole('button', { name: /^Reality/ }).click()
  await page.getByRole('button', { name: /^Slope colours/ }).click()
  await page.getByRole('group', { name: 'Follow' }).getByRole('button').nth(1).click()
  await page.waitForTimeout(4000)
  await shot('ui-follow.png')
  await page.getByRole('button', { name: 'Reset sim' }).click()
  await page.waitForTimeout(400)
  await shot('ui-reset-confirm.png')
  await page.getByRole('button', { name: 'Reset', exact: true }).click()
  await page.waitForTimeout(4000)
  await shot('ui-after-reset.png')
}

const fps = await page.evaluate(
  () =>
    new Promise((resolve) => {
      let frames = 0
      const start = performance.now()
      const step = () => {
        frames++
        if (performance.now() - start < 2000) requestAnimationFrame(step)
        else resolve((frames * 1000) / (performance.now() - start))
      }
      requestAnimationFrame(step)
    }),
)
console.log(`frame rate in this (software rendered) browser: ${fps.toFixed(1)} fps`)
console.log(problems.length ? `console problems:\n${problems.join('\n')}` : 'no console errors or warnings')
await browser.close()
