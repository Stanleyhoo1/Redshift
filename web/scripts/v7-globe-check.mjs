// Check of the v7 globe view (spec section 5) through the real UI, against the mock. Each
// check starts on the sector view; the runner puts it back there afterwards. Saves
// screenshots and exits 1 on a failed check or on any console error or warning.
// Needs the mock on 8102 (PORT=8102 MOCK_FIRST_QUESTION_S=99999 node mock/mock-server.mjs)
// and vite on 5182 proxying to it (SIM_URL=http://localhost:8102), see web/README.md.
// Usage: node scripts/v7-globe-check.mjs [url] [outDir]
// Env: CHROME, WIDTH, HEIGHT, ONLY (check names, comma separated)
import { chromium } from 'playwright-core'
import fs from 'node:fs'
import path from 'node:path'

const url = process.argv[2] || 'http://localhost:5182'
const outDir = process.argv[3] || 'screenshots'
const width = Number(process.env.WIDTH || 1600)
const height = Number(process.env.HEIGHT || 900)
const only = process.env.ONLY ? process.env.ONLY.split(',').map((s) => s.trim()) : null
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
let failed = false
const fail = (m) => {
  failed = true
  console.error(`FAIL: ${m}`)
}
const t0 = Date.now()
const log = (m) => console.log(`${((Date.now() - t0) / 1000).toFixed(1)} s  ${m}`)
const origin = new URL(url).origin
const api = async (p, body) => {
  const r = await fetch(`${origin}${p}`, { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(body || {}) })
  return { status: r.status, ...(await r.json().catch(() => ({}))) }
}
const dot = (a, b) => a[0] * b[0] + a[1] * b[1] + a[2] * b[2]

async function openPage(options = {}) {
  const context = await browser.newContext({ viewport: { width, height }, deviceScaleFactor: 1, ...options })
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
  await page.waitForFunction(() => window.__redshift && window.__redshift.state().rovers.r2, null, { timeout: 60000 })
  await page.waitForTimeout(2500)
  return { context, page }
}

const viewOf = (page) =>
  page.evaluate(() => {
    const s = window.__redshift.state()
    return { view: s.view, render: s.viewRender, phase: s.viewPhase }
  })
// Wait until `want` is both asked for and drawn, with the switch finished.
const settled = (page, want, timeout = 30000) =>
  page.waitForFunction(
    (w) => {
      const s = window.__redshift.state()
      return s.view === w && s.viewRender === w && s.viewPhase === 'idle'
    },
    want,
    { timeout },
  )
const toGlobe = async (page) => {
  await page.evaluate(() => window.__redshift.setView('globe'))
  await settled(page, 'globe')
}
const toPlane = async (page) => {
  await page.evaluate(() => window.__redshift.setView('plane'))
  await settled(page, 'plane')
}
async function shot(page, name) {
  const file = path.join(outDir, `v7-globe-${width}x${height}-${name}.png`)
  await page.screenshot({ path: file, timeout: 120000 })
  log(`saved ${file}`)
}
// Records, once per frame, every view/render/phase the page goes through, whether the fade
// overlay was ever on, and how close the planet camera came in each phase. It runs in
// requestAnimationFrame right after the 3D view's own frame: with software rendering, timers
// can starve between slow frames and miss a phase that lasts one frame, and screenshots
// take seconds, so mid-flight samples are unreliable.
const startPhaseLog = (page) =>
  page.evaluate(() => {
    if (window.__phaseLog) window.__phaseLog.on = false
    const L = { on: true, phases: [], fadeSeen: false, minDist: {} }
    window.__phaseLog = L
    const step = () => {
      if (!L.on) return
      const s = window.__redshift.state()
      const k = `${s.view}/${s.viewRender}/${s.viewPhase}`
      if (L.phases[L.phases.length - 1] !== k) L.phases.push(k)
      if (document.querySelector('.globe-fade[data-on]')) L.fadeSeen = true
      if (s.viewRender === 'globe') {
        const d = window.__redshift.globeState().dist
        L.minDist[s.viewPhase] = Math.min(L.minDist[s.viewPhase] ?? Infinity, d)
      }
      requestAnimationFrame(step)
    }
    requestAnimationFrame(step)
  })
const phaseLog = (page) =>
  page.evaluate(() => {
    const L = window.__phaseLog
    L.on = false
    return { phases: L.phases, fadeSeen: L.fadeSeen, minDist: L.minDist }
  })
// True when the entries of `seq` appear in `list` in that order (others may sit between).
const inOrder = (list, seq) => {
  let i = 0
  for (const x of list) if (x === seq[i]) i++
  return i === seq.length
}
// A point on the 3D view that no panel covers, near (fx, fy) as fractions of the canvas.
const freePoint = (page, fx, fy) =>
  page.evaluate(
    ([fx, fy]) => {
      const c = document.querySelector('main.view canvas')
      const r = c.getBoundingClientRect()
      for (let k = 0; k < 48; k++) {
        const x = r.left + r.width * Math.min(0.95, fx + (k % 8) * 0.02)
        const y = r.top + r.height * Math.min(0.95, fy - Math.floor(k / 8) * 0.04)
        if (document.elementFromPoint(x, y) === c) return { x, y }
      }
      return null
    },
    [fx, fy],
  )

const checks = []
const add = (name, fn) => checks.push({ name, fn })

// G goes up through the rise, the fade and the arrival; G again comes back down.
add('toggle-key', async (page) => {
  await startPhaseLog(page)
  await page.keyboard.press('g')
  const mid = await viewOf(page)
  const tags = await page.locator('.rover-tag').count()
  log(`right after G: ${JSON.stringify(mid)}, ${tags} rover tags`)
  if (mid.view !== 'globe') fail(`G did not ask for the planet (view ${mid.view})`)
  if (tags) fail('rover labels stay up while the camera rises')
  await settled(page, 'globe')
  const up = await phaseLog(page)
  log(`up: ${up.phases.join(' > ')}`)
  if (!inOrder(up.phases, ['globe/plane/out', 'globe/plane/fade', 'globe/globe/in', 'globe/globe/idle'])) fail('the way up skipped a phase')
  if (!up.fadeSeen) fail('no fade between the views')
  await page.waitForTimeout(1200)
  await shot(page, 'planet')
  await startPhaseLog(page)
  await page.keyboard.press('g')
  await settled(page, 'plane')
  const down = await phaseLog(page)
  log(`down: ${down.phases.join(' > ')}`)
  if (!inOrder(down.phases, ['plane/globe/out', 'plane/globe/fade', 'plane/plane/in', 'plane/plane/idle'])) fail('the way down skipped a phase')
  await page.waitForSelector('.rover-tag', { timeout: 15000 })
})

// The View control in the HUD does the same, and shows which view is up.
add('view-control', async (page) => {
  const group = page.locator('[aria-labelledby="viewmode-label"]')
  await group.locator('button', { hasText: 'Planet' }).click({ timeout: 10000 })
  await settled(page, 'globe')
  if ((await group.locator('button[aria-pressed="true"]').innerText()).trim() !== 'Planet') fail('the Planet button is not pressed on the planet')
  await group.locator('button', { hasText: 'Sector' }).click()
  await settled(page, 'plane')
  if ((await group.locator('button[aria-pressed="true"]').innerText()).trim() !== 'Sector') fail('the Sector button is not pressed on the sector')
})

// The debug hook: setView (instant on request) and the planet camera's state.
add('debug-hook', async (page) => {
  const v = await page.evaluate(() => window.__redshift.setView('globe', true))
  const s = await viewOf(page)
  if (v !== 'globe' || s.render !== 'globe' || s.phase !== 'idle') fail(`setView('globe', true) should switch at once, got ${JSON.stringify(s)}`)
  await page.waitForFunction(() => window.__redshift.globeState().dist > 1, null, { timeout: 15000 })
  const g = await page.evaluate(() => window.__redshift.globeState())
  log(`globeState ${JSON.stringify(g)}`)
  if (!(g.dist > 1.3 && g.dist < 9.5)) fail(`the planet camera is ${g.dist} radii out`)
  await page.evaluate(() => window.__redshift.setView('plane', true))
  if ((await viewOf(page)).render !== 'plane') fail("setView('plane', true) did not switch back")
})

// Hazard placement never survives the trip up.
add('placing-cancelled', async (page) => {
  await page.locator('.hz-btn[data-type="sand"]').click({ timeout: 10000 })
  await page.waitForSelector('.hz-btn[data-type="sand"][data-placing]', { timeout: 10000 })
  await page.keyboard.press('g')
  await settled(page, 'globe')
  if (await page.locator('.hz-btn[data-placing]').count()) fail('hazard placement survived the switch to the planet')
})

// The View row fits: the controls end above the map legend with slope colours off and on
// (run it once with WIDTH=1280 HEIGHT=720), and the Follow buttons stay the first buttons
// in the controls (v4-shots and visual-shots pick them by index).
add('hud-fit', async (page) => {
  const segs = (await page.locator('.hud-controls .seg button').allInnerTexts()).map((s) => s.trim())
  log(`controls buttons: ${segs.join(', ')}`)
  if (segs.slice(0, 4).join(',') !== 'Off,1,2,3') fail('the Follow buttons are no longer the first buttons in the controls')
  for (const slope of [false, true]) {
    if (slope) await page.keyboard.press('s')
    await page.waitForTimeout(800)
    const r = await page.evaluate(() => {
      const lg = document.querySelector('.hud-legend')?.getBoundingClientRect()
      return { controls: document.querySelector('.hud-controls').getBoundingClientRect().bottom, legend: lg && lg.height > 0 ? lg.top : null }
    })
    log(`slope colours ${slope ? 'on' : 'off'}: the controls end at ${r.controls.toFixed(1)} px, the legend starts at ${r.legend === null ? 'none' : r.legend.toFixed(1)} px`)
    if (r.legend !== null && r.controls > r.legend) fail(`the controls run ${(r.controls - r.legend).toFixed(1)} px into the map legend (slope colours ${slope ? 'on' : 'off'})`)
    await shot(page, slope ? 'hud-slope' : 'hud')
  }
  await page.keyboard.press('s') // slope colours off again
})

// Run every check in order.
await api('/api/reset', { seed: 42 })
await api('/api/_mock/rearm', { armed: false })
const { context, page } = await openPage()
for (const c of checks) {
  if (only && !only.includes(c.name)) continue
  log(`check ${c.name}`)
  try {
    await c.fn(page)
  } catch (err) {
    fail(`${c.name}: ${String(err.message).split('\n')[0]}`)
  }
  // Back to a plain sector view for the next check, whatever happened.
  await page.evaluate(() => window.__redshift?.setView?.('plane', true)).catch(() => {})
  await page.keyboard.press('Escape').catch(() => {})
  await page.keyboard.press('Escape').catch(() => {})
}
await context.close()

if (problems.length) {
  console.log('\nConsole problems:')
  for (const p of problems) console.log(`  ${p}`)
} else console.log('\nNo console errors or warnings.')
await browser.close()
console.log(failed ? 'Some checks failed.' : 'All checks passed.')
process.exit(failed || problems.length ? 1 : 0)
