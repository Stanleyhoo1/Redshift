// Drives the v2 features through the real UI (sensor panel, hazard buttons, event
// fly-to) and saves screenshots. Needs the mock (or sim) on 8000 and `npm run preview`.
// Usage: node scripts/v2-shots.mjs [url] [outDir]   Env: CHROME, WIDTH, HEIGHT
import { chromium } from 'playwright-core'
import fs from 'node:fs'
import path from 'node:path'

const url = process.argv[2] || 'http://localhost:4173'
const width = Number(process.env.WIDTH || 1600)
const height = Number(process.env.HEIGHT || 900)
const outDir = process.argv[3] || 'screenshots'
const tag = `${width}x${height}`
const candidates = [process.env.CHROME, '/opt/pw-browsers/chromium-1243/chrome-linux64/chrome', '/usr/bin/google-chrome', '/usr/bin/chromium'].filter(Boolean)
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
const shot = async (name) => {
  const file = path.join(outDir, `v2-${tag}-${name}.png`)
  await page.screenshot({ path: file })
  console.log(`saved ${file} at ${((Date.now() - t0) / 1000).toFixed(1)} s`)
}
const wait = (ms) => page.waitForTimeout(ms)
const origin = new URL(url).origin

// Fresh world for every run.
await fetch(`${origin}/api/reset`, { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ seed: 42 }) })
await page.goto(url, { waitUntil: 'domcontentloaded' })
await wait(7000)
await shot('01-overview')

// Fire a hazard with its button; returns the server's hazard (or the error text).
const hazardBtn = (label) => page.locator('.hz-btn', { hasText: label })
async function fire(label) {
  const resp = page.waitForResponse((r) => r.url().includes('/api/hazard'))
  await hazardBtn(label).click()
  const r = await resp
  const j = await r.json().catch(() => ({}))
  console.log(`${label}: HTTP ${r.status()} ${j.error || j.hazard?.id || ''} ${j.hazard?.rover || ''}`)
  return { status: r.status(), ...j }
}
const card = (roverId) => page.locator('.rover').nth(Number(String(roverId).replace(/\D/g, '')) - 1)

// 1. Sensor panel for Rover 1 (click its card).
await card('r1').click()
await wait(3500)
await shot('02-sensors')

// 2. Rockfall on a route, panel on the targeted rover so the radar shows the boulders.
let rock = await fire('Rockfall')
for (let k = 0; rock.status !== 200 && k < 6; k++) {
  await wait(4500)
  rock = await fire('Rockfall')
}
if (rock.hazard?.rover) {
  await page.keyboard.press('Escape')
  await card(rock.hazard.rover).click()
}
await wait(450)
await shot('03-rockfall-drop')
await wait(3500)
await shot('04-rockfall')

// 3. Event click: fly to the rockfall (the hazard row in the log).
await page.keyboard.press('Escape') // close the panel, keep following
await page.locator('.ev[data-level="hazard"] .ev-btn').first().click()
await wait(1300)
await shot('05-event-fly')
await wait(2000)
await shot('06-event-arrived')
await page.keyboard.press('0')

// 4. Soft sand: follow the targeted rover and wait for the slip spike.
let sand = await fire('Soft sand')
for (let k = 0; sand.status !== 200 && k < 6; k++) {
  await wait(4500)
  sand = await fire('Soft sand')
}
if (sand.hazard?.rover) await card(sand.hazard.rover).click()
let slipShot = false
for (let k = 0; k < 80; k++) {
  await wait(250)
  const txt = await page.locator('.sp-row-bar').first().locator('.sp-num').textContent().catch(() => '')
  if (parseInt(txt, 10) >= 30) {
    await shot('07-sand-slip')
    slipShot = true
    break
  }
}
if (!slipShot) await shot('07-sand-noslip')
await wait(3000)
// Fly to the sand event with Reality on: the true patch (dashed) and the zone the
// fleet marked from wheel slip (hatched).
await page.keyboard.press('Escape')
await page.locator('.ev[data-level="hazard"] .ev-btn', { hasText: 'Soft sand' }).first().click()
await page.getByRole('button', { name: /^Reality/ }).click()
await wait(3500)
await shot('08-sand-zone-reality')
await page.getByRole('button', { name: /^Reality/ }).click()
await page.keyboard.press('0')

// 5. Dust storm with the panel open: haze, banner, shrunken LiDAR ring.
const dust = await fire('Dust storm')
await wait(4000)
await shot('09-dust')
await page.keyboard.press('Escape')
await page.keyboard.press('Escape')
await wait(2500)
await shot('10-dust-overview')

// 6. Motor fault on Rover 2.
await page.locator('.hz-target .seg button', { hasText: '2' }).click()
const fault = await fire('Motor fault')
await wait(2500)
await shot('11-fault')
await card('r2').click()
await wait(2500)
await shot('12-fault-panel')
// A second fault on the same rover: the server says no, shown inline.
await wait(1500)
const again = await fire('Motor fault')
await wait(400)
await shot('13-error-inline')
const errText = await page.locator('.hz-status').textContent()
console.log(`inline status: ${errText}`)

const fps = await page.evaluate(
  () =>
    new Promise((resolve) => {
      let frames = 0
      const start = performance.now()
      const step = () => {
        frames++
        if (performance.now() - start < 3000) requestAnimationFrame(step)
        else resolve((frames * 1000) / (performance.now() - start))
      }
      requestAnimationFrame(step)
    }),
)
console.log(`hazards: rock ${rock.status}, sand ${sand.status}, dust ${dust.status}, fault ${fault.status}, fault again ${again.status}`)
console.log(`frame rate in this (software rendered) browser: ${fps.toFixed(1)} fps`)
console.log(problems.length ? `console problems:\n${problems.join('\n')}` : 'no console errors or warnings')
await browser.close()
