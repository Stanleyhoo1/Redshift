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
import { IDLE_SWAY } from '../src/globe/globeMath.js'

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

// Graphics Low compiles the planet without GLOBE_HQ (fewer octaves, one crater size).
add('graphics-low', async (page) => {
  const gfx = page.locator('[aria-labelledby="gfx-label"]')
  await toGlobe(page)
  if (!(await page.evaluate(() => window.__redshift.globeState().hq))) fail('Graphics High is not using the detailed planet shader')
  await gfx.locator('button', { hasText: 'Low' }).click({ timeout: 10000 })
  try {
    await page.waitForFunction(() => window.__redshift.globeState().hq === false, null, { timeout: 15000 })
    await page.waitForTimeout(1500)
    await shot(page, 'planet-low')
  } catch {
    fail('Graphics Low did not switch the planet to the cheaper shader')
  } finally {
    await gfx.locator('button', { hasText: 'High' }).click()
    await toPlane(page)
  }
})

// On the planet: no sector labels, the planet legend instead of the map legend, and the
// live sector card next to the marker.
add('planet-dom', async (page) => {
  await toGlobe(page)
  const tags = await page.locator('.rover-tag').count()
  const marks = await page.locator('.place-mark').count()
  if (tags || marks) fail(`sector labels over the planet: ${tags} rover tags, ${marks} place marks`)
  const legend = (await page.locator('.globe-legend').innerText({ timeout: 10000 })).replace(/\s+/g, ' ')
  log(`legend: ${legend}`)
  for (const w of ['Redshift-b', 'Radius 12,742 km', '2.0 g', 'Landing site 4.4° N, 44.9° E', 'not to scale']) if (!legend.includes(w)) fail(`the planet legend lacks "${w}"`)
  if (await page.locator('.hud-legend:not(.globe-legend)').count()) fail('the map legend is still up')
  if (await page.locator('.dust-haze').count()) fail('the dust veil is over the planet')
  await page.waitForSelector('.globe-card[data-settled]', { timeout: 15000 })
  const card = (await page.locator('.globe-card').innerText()).replace(/\s+/g, ' ')
  log(`card: ${card}`)
  for (const w of ['Landing sector', '120 x 120 m, not to scale', 'Explored', 'Gravity', '2.0 g', 'Sim time', 'Fly down to the sector']) if (!card.includes(w)) fail(`the sector card lacks "${w}"`)
  const rows = await page.locator('.globe-card .gc-rover').count()
  if (rows < 3) fail(`the card lists ${rows} rovers`)
  await shot(page, 'planet-card')
})

// The marker: in view at home, a pointer on hover, and a click flies back down.
add('marker-click', async (page) => {
  await toGlobe(page)
  await page.waitForSelector('.globe-card[data-settled]', { timeout: 15000 })
  const m = await page.evaluate(() => window.__redshift.globeSiteScreen())
  log(`marker at ${JSON.stringify(m)}`)
  if (!m?.visible) return fail('the landing marker is not in view at home')
  const top = await page.evaluate(([x, y]) => document.elementFromPoint(x, y)?.tagName, [m.x, m.y])
  if (top !== 'CANVAS') fail(`something covers the marker (${top})`)
  await page.mouse.move(m.x, m.y)
  await page.waitForTimeout(500)
  const cursor = await page.evaluate(() => document.body.style.cursor)
  if (cursor !== 'pointer') fail(`hovering the marker shows cursor "${cursor}"`)
  await page.mouse.click(m.x, m.y)
  await settled(page, 'plane')
  await page.waitForSelector('.rover-tag', { timeout: 15000 })
  if (await page.locator('.globe-legend').count()) fail('the planet legend stayed after flying down')
  await shot(page, 'back-on-sector')
})

// The card's own button flies back down too.
add('card-button', async (page) => {
  await toGlobe(page)
  await page.locator('.globe-card[data-settled] .gc-down').click({ timeout: 15000 })
  await settled(page, 'plane')
})

// An open sensor panel hides on the planet (the planet legend shows instead) and comes back
// as it was, with the camera view the user picked.
add('sensors-kept', async (page) => {
  await page.keyboard.press('2')
  const panel = page.locator('section.hud-sensors')
  await panel.waitFor({ timeout: 15000 })
  await panel.locator('[aria-label="Camera view"] button', { hasText: 'Chase' }).click({ timeout: 10000 })
  await page.keyboard.press('g')
  await settled(page, 'globe')
  if (await panel.isVisible()) fail('the sensor panel is over the planet')
  if (!(await page.locator('.globe-legend').isVisible())) fail('no planet legend while a rover is selected')
  await page.keyboard.press('g')
  await settled(page, 'plane')
  await panel.waitFor({ timeout: 15000 })
  const pressed = (await panel.locator('[aria-label="Camera view"] button[aria-pressed="true"]').innerText()).trim()
  if (pressed !== 'Chase') fail(`the camera view came back as ${pressed}, not Chase`)
})

// Arriving, the planet camera starts low over the site and eases out to the home view.
add('arrival', async (page) => {
  await startPhaseLog(page)
  await page.keyboard.press('g')
  await settled(page, 'globe')
  const rec = await phaseLog(page)
  const home = await page.evaluate(() => window.__redshift.globeState())
  const closest = rec.minDist.in ?? Infinity
  log(`arrival: closest ${closest.toFixed(2)} radii while easing out, home ${home.dist.toFixed(2)}`)
  if (!(closest < home.dist - 1)) fail('the planet camera did not start close over the site')
  if (!(await page.evaluate(() => window.__redshift.globeSiteScreen()))?.visible) fail('the landing site is not in view at home')
})

// Leaving, it dives back towards the site before the fade.
add('dive', async (page) => {
  await toGlobe(page)
  const home = await page.evaluate(() => window.__redshift.globeState())
  await startPhaseLog(page)
  await page.keyboard.press('g')
  await settled(page, 'plane')
  const rec = await phaseLog(page)
  const low = Math.min(rec.minDist.out ?? Infinity, rec.minDist.fade ?? Infinity)
  log(`dive: ${home.dist.toFixed(2)} -> ${low.toFixed(2)} radii before the fade`)
  if (!(low < home.dist - 1)) fail('the planet camera did not dive towards the site')
  await shot(page, 'back-down')
})

// The sector camera comes back to the view it left (a user view here, so it flies back).
add('round-trip', async (page) => {
  await page.evaluate(() => window.__redshift.view(10, -6, 30, 50, 40))
  await page.waitForTimeout(1500)
  const before = await page.evaluate(() => window.__redshift.screenOf(0, 0))
  await toGlobe(page)
  await toPlane(page)
  await page
    .waitForFunction(
      (b) => {
        const p = window.__redshift.screenOf(0, 0)
        return Math.hypot(p.x - b.x, p.y - b.y) < 3
      },
      before,
      { timeout: 20000 },
    )
    .catch(() => {})
  const after = await page.evaluate(() => window.__redshift.screenOf(0, 0))
  const off = Math.hypot(after.x - before.x, after.y - before.y)
  log(`round trip: the lander is ${off.toFixed(1)} px from where it was`)
  if (off > 3) fail('the sector camera did not come back to the view it left')
})

// Drag turns the planet, scroll zooms, 0 glides back to the home view.
add('orbit-home', async (page) => {
  await toGlobe(page)
  const home = await page.evaluate(() => window.__redshift.globeState())
  const p = await freePoint(page, 0.55, 0.86)
  if (!p) return fail('no free spot on the planet view to drag')
  await page.mouse.move(p.x, p.y)
  await page.mouse.down()
  await page.mouse.move(p.x + 240, p.y - 30, { steps: 10 })
  await page.mouse.up()
  try {
    await page.waitForFunction((h) => {
      const d = window.__redshift.globeState().dir
      return d[0] * h[0] + d[1] * h[1] + d[2] * h[2] < 0.995
    }, home.dir, { timeout: 20000 })
  } catch {
    fail('dragging did not turn the planet')
  }
  const before = await page.evaluate(() => window.__redshift.globeState().dist)
  await page.mouse.move(p.x, p.y)
  await page.mouse.wheel(0, -600)
  try {
    await page.waitForFunction((d0) => window.__redshift.globeState().dist < d0 - 0.1, before, { timeout: 20000 })
  } catch {
    fail('scrolling did not zoom in')
  }
  await page.keyboard.press('0')
  try {
    await page.waitForFunction((h) => {
      const g = window.__redshift.globeState()
      const d = g.dir
      return !g.gliding && d[0] * h.dir[0] + d[1] * h.dir[1] + d[2] * h.dir[2] > 0.9995 && Math.abs(g.dist - h.dist) < 0.02 && Math.abs(g.spin) < 0.01
    }, home, { timeout: 20000 })
  } catch {
    fail('0 did not glide back to the home view')
  }
  const g = await page.evaluate(() => window.__redshift.globeState())
  log(`after 0: dot ${dot(g.dir, home.dir).toFixed(5)}, dist ${g.dist.toFixed(2)}`)
})

// Left alone, the planet turns slowly (the sway starts 8 s after the last input, from rest).
add('idle-spin', async (page) => {
  await toGlobe(page)
  const a = await page.evaluate(() => window.__redshift.globeState().spin)
  await page.waitForTimeout(20000)
  const b = await page.evaluate(() => window.__redshift.globeState().spin)
  log(`idle spin ${a.toFixed(4)} -> ${b.toFixed(4)} rad`)
  if (!(Math.abs(b - a) > 0.003)) fail('the planet did not turn when left alone')
  await shot(page, 'planet-turned')
})

// Left alone for over a minute it only sways towards the sun and back: the middle of the
// disc stays in daylight, and the site and its card stay in view the whole time.
add('idle-long', async (page) => {
  await toGlobe(page)
  await page.waitForSelector('.globe-card[data-settled]', { timeout: 15000 })
  const bad = new Set()
  let peak = 0
  for (let k = 1; k <= 18; k++) {
    await page.waitForTimeout(4000)
    const g = await page.evaluate(() => window.__redshift.globeState())
    const m = await page.evaluate(() => window.__redshift.globeSiteScreen())
    const away = await page.locator('.globe-card[data-away]').count()
    peak = Math.max(peak, g.spin)
    if (g.spin < -0.01 || g.spin > IDLE_SWAY + 0.01) bad.add(`it turned ${g.spin.toFixed(3)} rad, past the sway`)
    if (!(g.lit > 0.2)) bad.add(`the sun stood at ${Number(g.lit).toFixed(2)} over the middle of the disc`)
    if (!m?.visible) bad.add('the landing site went out of view')
    if (away) bad.add('the sector card hid')
  }
  log(`72 s alone: turned at most ${peak.toFixed(3)} rad`)
  for (const b of bad) fail(`left alone: ${b}`)
  if (!(peak > 0.02)) fail('the planet did not turn when left alone for a minute')
  await shot(page, 'planet-idle-long')
})

// A new world while on the planet (Reset sim stays in the controls): the camera glides to
// the new landing site and frames it where it framed the old one; then back to seed 42,
// which the later checks expect.
add('reset-on-planet', async (page) => {
  await toGlobe(page)
  await page.waitForSelector('.globe-card[data-settled]', { timeout: 15000 })
  const before = await page.evaluate(() => window.__redshift.globeSiteScreen())
  // Sites from siteFromSeed (globeMath.js): seed 7 lands about 40 degrees from seed 42.
  for (const [seed, site] of [
    [7, { lat: -19.9, lon: 10.5 }],
    [42, { lat: 4.4, lon: 44.9 }],
  ]) {
    await api('/api/reset', { seed })
    try {
      await page.waitForFunction(
        ([want, b]) => {
          const g = window.__redshift.globeState()
          const m = window.__redshift.globeSiteScreen()
          return g.site?.lat === want.lat && g.site?.lon === want.lon && !g.gliding && m?.visible && Math.hypot(m.x - b.x, m.y - b.y) < 40
        },
        [site, before],
        { timeout: 30000 },
      )
    } catch {
      const m = await page.evaluate(() => window.__redshift.globeSiteScreen())
      fail(`after a reset to seed ${seed} the site is at ${JSON.stringify(m)}, not framed where the old one was ${JSON.stringify(before)}`)
    }
    if (await page.locator('.globe-card[data-away]').count()) fail(`the sector card is hidden after a reset to seed ${seed}`)
  }
  await page.waitForFunction(() => window.__redshift.state().rovers.r2, null, { timeout: 30000 })
})

// Reduced motion: the switch is instant both ways, and the planet stays still.
add('reduced-motion', async () => {
  const { context, page } = await openPage({ reducedMotion: 'reduce' })
  try {
    await startPhaseLog(page)
    await page.keyboard.press('g')
    const s = await viewOf(page)
    if (s.render !== 'globe' || s.phase !== 'idle') fail(`with reduced motion G should switch at once, got ${JSON.stringify(s)}`)
    await page.waitForTimeout(10000)
    const g = await page.evaluate(() => window.__redshift.globeState())
    if (Math.abs(g.spin) > 1e-6) fail('the planet turns with reduced motion')
    await shot(page, 'planet-reduced-motion')
    await page.keyboard.press('g')
    const s2 = await viewOf(page)
    if (s2.render !== 'plane' || s2.phase !== 'idle') fail(`with reduced motion G should come back at once, got ${JSON.stringify(s2)}`)
    const log2 = await phaseLog(page)
    if (log2.phases.some((k) => k.endsWith('/out') || k.endsWith('/fade'))) fail(`reduced motion went through ${log2.phases.join(' > ')}`)
  } finally {
    await context.close()
  }
})

// A rover key on the planet: back down with that rover's sensors open.
add('rover-key', async (page) => {
  await toGlobe(page)
  await page.keyboard.press('2')
  await settled(page, 'plane')
  await page.waitForSelector('section.hud-sensors', { timeout: 15000 })
})

// A rover row in the planet's sector card does the same.
add('card-rover', async (page) => {
  await toGlobe(page)
  await page.locator('.globe-card[data-settled] .gc-rover').first().click({ timeout: 15000 })
  await settled(page, 'plane')
  await page.waitForSelector('section.hud-sensors', { timeout: 15000 })
})

// A hazard button on the planet flies down first, then places as usual.
add('hazard-button', async (page) => {
  await toGlobe(page)
  await page.locator('.hz-btn[data-type="rockfall"]').click({ timeout: 10000 })
  await settled(page, 'plane')
  if (!(await page.locator('.hz-btn[data-type="rockfall"][data-placing]').count())) fail('the rockfall button is not placing after the flight down')
  await page.keyboard.press('Escape')
})

// Clicking an event with a place on the planet flies down to it.
add('event-click', async (page) => {
  await toGlobe(page)
  // The mock refuses a spot near the lander, the map edge or another hazard ("Try again in a
  // few seconds"), which depends on where the rovers are by now, so ask a few times: for
  // Rover 2 first, then for any rover.
  let r
  for (let i = 0; i < 8; i++) {
    r = await api('/api/hazard', i % 2 ? { type: 'rockfall' } : { type: 'rockfall', rover: 'r2' })
    if (r.status === 200) break
    log(`rockfall refused (${r.error || r.status}), asking again`)
    await page.waitForTimeout(1500)
  }
  if (r.status !== 200) return fail(`could not start a rockfall (${r.status} ${r.error || ''})`)
  await page.click('#tab-events')
  const row = page.locator('li.ev.ev-link .ev-btn').first()
  await row.waitFor({ timeout: 20000 })
  await row.click()
  await settled(page, 'plane')
  await shot(page, 'event-from-planet')
})

// A rover asking to pick a place brings the sector back, with the candidates shown.
add('place-question', async (page) => {
  log(`reveal ${JSON.stringify(await api('/api/_mock/reveal', { water: true }))}`)
  for (const id of ['r1', 'r2', 'r3']) await api('/api/_mock/battery', { rover: id, battery: 0.95 })
  await toGlobe(page)
  log(`chat ${JSON.stringify(await api('/api/chat', { text: 'go to the water source' }))}`)
  await settled(page, 'plane', 45000)
  await page.waitForFunction(() => document.querySelectorAll('.place-mark').length >= 2, null, { timeout: 30000 })
  const q = (await page.evaluate(() => window.__redshift.state().requests)).find((x) => x.kind === 'choose_place')
  if (!q) return fail('no choose_place request after the order')
  await shot(page, 'place-from-planet')
  await api('/api/answer', { id: q.id, option: q.options[0].id })
  await page.waitForFunction((id) => !window.__redshift.state().requests.some((x) => x.id === id), q.id, { timeout: 20000 })
})

// After a trip to the planet and back, click-to-place still works end to end: a rockfall
// placed by clicking the map (as v4-shots does) reaches the sim and shows up. A fresh
// world first, so no earlier hazard is in the way.
add('place-after-trip', async (page) => {
  await api('/api/reset', { seed: 42 })
  await page.waitForFunction(() => {
    const s = window.__redshift.state()
    return !!s.rovers.r2 && s.hazards.length === 0
  }, null, { timeout: 30000 })
  await page.waitForTimeout(3000)
  await page.keyboard.press('g')
  await settled(page, 'globe')
  await page.keyboard.press('g')
  await settled(page, 'plane')
  await page.locator('.hz-btn[data-type="rockfall"]').click({ timeout: 10000 })
  await page.waitForSelector('.hz-btn[data-type="rockfall"][data-placing]', { timeout: 10000 })
  const s = await page.evaluate(() => window.__redshift.state())
  const inPoly = (x, y, poly) => {
    let inside = false
    for (let i = 0, j = poly.length - 1; i < poly.length; j = i++) {
      const [xi, yi] = poly[i]
      const [xj, yj] = poly[j]
      if (yi > y !== yj > y && x < ((xj - xi) * (y - yi)) / (yj - yi) + xi) inside = !inside
    }
    return inside
  }
  // The sim refuses water it has not identified too, so keep well clear of every lake.
  const wet = (x, y) => (s.features?.water || []).some((w) => inPoly(x, y, w.outline) || w.outline.some(([px, py]) => Math.hypot(px - x, py - y) < 5))
  let placed = null
  for (let k = 0; k < 12 && !placed; k++) {
    const x = s.lander.x + Math.cos((k / 12) * Math.PI * 2) * 9
    const y = s.lander.y + Math.sin((k / 12) * Math.PI * 2) * 9
    const rovers = await page.evaluate(() => window.__redshift.state().rovers)
    if (wet(x, y) || Object.values(rovers).some((r) => Math.hypot(r.x - x, r.y - y) < 6)) continue
    const p = await page.evaluate(([wx, wy]) => window.__redshift.screenOf(wx, wy), [x, y])
    if (!p.visible || !(await page.evaluate(([px, py]) => document.elementFromPoint(px, py)?.tagName === 'CANVAS', [p.x, p.y]))) continue
    await page.mouse.move(p.x, p.y, { steps: 3 })
    await page.waitForTimeout(1200)
    if (await page.locator('.ghost-tag[data-bad]').count()) continue
    const resp = page.waitForResponse((r) => r.url().includes('/api/hazard'), { timeout: 15000 }).catch(() => null)
    await page.mouse.click(p.x, p.y)
    const r = await resp
    log(`rockfall at (${x.toFixed(1)}, ${y.toFixed(1)}): HTTP ${r ? r.status() : 'no request'}`)
    if (r?.status() === 200) placed = (await r.json()).hazard
  }
  if (!placed) return fail('no rockfall could be placed by clicking the map after a trip to the planet')
  await page.waitForFunction((id) => window.__redshift.state().hazards.some((h) => h.id === id), placed.id, { timeout: 20000 })
  await page.mouse.move(40, height - 40)
  await shot(page, 'placed-after-trip')
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
