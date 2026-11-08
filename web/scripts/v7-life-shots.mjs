// Screenshots of the v7 lifeform kinds (grazers, skitterers, drifters, crawlers, burrowers)
// through the real UI against a sim with LIFE_VARIETY on. Turns Reality on, then flies the
// debug camera to the first creature of each kind (a burrower once it is up). Prints any
// console error or warning (exit 1 on one).
// Usage: node scripts/v7-life-shots.mjs [url] [outDir]
// Env: CHROME, WIDTH (1280), HEIGHT (720), PREFIX (file name prefix), GRAPHICS (low|high)
import { chromium } from 'playwright-core'
import fs from 'node:fs'
import path from 'node:path'

const url = process.argv[2] || 'http://localhost:5213'
const outDir = process.argv[3] || 'screenshots'
const width = Number(process.env.WIDTH || 1280)
const height = Number(process.env.HEIGHT || 720)
const prefix = process.env.PREFIX || 'v7life'
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
  if (/GPU stall due to ReadPixels|THREE\.Clock: This module has been deprecated|Failed to load resource.*403/.test(t)) return
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
await page.keyboard.press('r')
await page.waitForTimeout(1500)
await shot('overview-reality')
const box = await page.locator('.hud-legend').boundingBox()
if (box) await page.screenshot({ path: path.join(outDir, `${prefix}-legend.png`), clip: box })

const life = await page.evaluate(() => window.__redshift.state().lifeforms)
const kinds = {}
for (const l of life) (kinds[l.kind] ||= []).push(l)
console.log(`lifeforms: ${Object.entries(kinds).map(([k, v]) => `${k} ${v.length}`).join(', ')}`)
const view = async (name, id, dist, el, az = 25) => {
  const l = await page.evaluate((id) => window.__redshift.state().lifeforms.find((q) => q.id === id), id)
  if (!l) return
  await page.evaluate(([x, y, az, el, d]) => window.__redshift.view(x, y, az, el, d), [l.x, l.y, az, el, dist])
  await page.waitForTimeout(1500)
  await shot(name)
}
// close-ups: hide the panels so the creature is not behind them
const hideHud = (on) =>
  page.evaluate((on) => {
    for (const el of document.querySelectorAll('body *')) {
      if (el.querySelector('canvas') || el.tagName === 'CANVAS' || el.closest('canvas')) continue
      const pos = getComputedStyle(el).position
      if (pos === 'absolute' || pos === 'fixed' || el.matches('aside, .hud, [class*="panel"], [class*="sidebar"]')) el.style.visibility = on ? 'hidden' : ''
    }
  }, on)
await hideHud(true)
const looks = { grazer: [6, 22], skitterer: [5, 30], drifter: [10, 10], crawler: [7, 24], burrower: [4, 20] }
for (const [kind, [dist, el]] of Object.entries(looks)) {
  const list = kinds[kind]
  if (!list) continue
  let id = list[0].id
  if (kind === 'burrower') {
    // wait (up to 40 s) for one that is up
    for (let k = 0; k < 40; k++) {
      const up = await page.evaluate(() => window.__redshift.state().lifeforms.find((q) => q.kind === 'burrower' && q.up > 0.9))
      if (up) {
        id = up.id
        break
      }
      await page.waitForTimeout(1000)
    }
  }
  await view(kind, id, dist, el)
}
await browser.close()
if (problems.length) {
  console.error(problems.join('\n'))
  process.exit(1)
}
console.log('no console errors')
