// Takes the README pictures of the version 7 terrain features (docs/images/terrain): one close view
// of each feature with Reality on, the frontiers, paths and LiDAR off and the dashboard panels hidden,
// like biome-shots.mjs. Needs a sim started with AUTO_LAUNCH=1 (every v7 feature and biomes on, the
// default from the environment) and a vite (dev or preview) pointed at it. It resets the sim to each
// shot's seed and speed, so do not point it at a sim you are using.
//   node scripts/feature-shots.mjs [appUrl] [simUrl] [outDir]
//   e.g. from web/: node scripts/feature-shots.mjs http://localhost:5173 http://localhost:8000 ../docs/images/terrain
// Writes <outDir>/<name>.jpg, 960 px wide, JPEG quality 82 (a little lower if a picture would pass
// 150 KB). Env: ONLY=mesa,geyser retakes just those; SEEDS="mesa:51,pits:62" tries other seeds (each
// shot aims at a feature by id or at a fixed point, so a new seed usually needs a new subject too);
// RAW=<dir> keeps the full-size PNGs; CHROME=/path/to/chrome.
//
// Camera: cam is [azimuth, elevation, distance] as in window.__redshift.view (az 0 = camera south of
// the subject looking north; the sun shines from about az 244), but the distance is from the camera to
// the subject point, `lift` metres above the ground at the subject. The script aims from the true
// heights (/api/truth), so the subject lands in the middle of the picture after `crop` ([top, bottom]
// fractions of the canvas) unless `place` ([x, y] fractions of the picture) says otherwise.
import fs from 'node:fs'
import path from 'node:path'
import { chromium } from 'playwright-core'

const app = process.argv[2] || 'http://localhost:5173'
const sim = process.argv[3] || 'http://localhost:8000'
const outDir = process.argv[4] || '../docs/images/terrain'
const rawDir = process.env.RAW || ''
const WIDTH = 1600 // viewport, as biome-shots.mjs (the canvas is the part left of the sidebar)
const HEIGHT = 900
const OUT_W = 960
const QUALITY = 0.82
const MAX_BYTES = 150 * 1024
const FOV = 42 // the scene camera's vertical field of view (App.jsx)

// Subjects: { kind: 'mesas', id } aims at that feature in state().features; { point: [x, y] } at a
// fixed spot. speed: sim seconds per real second (slow, so the rovers barely move and a geyser
// erupts for 8 to 15 real seconds). erupt: wait for that vent to start erupting before the shot.
const SHOTS = [
  // A flat-topped mesa with banded cliff walls, more mesas and buttes behind it.
  { name: 'mesa', seed: 51, subject: { kind: 'mesas', id: 'mesa3' }, lift: 3, cam: [200, 16, 38], crop: [0.27, 0] },
  // Looking north along the canyon from its southern ramp: the floor drops away between the walls.
  { name: 'canyon', seed: 69, subject: { point: [-39.6, -6] }, cam: [0, 32, 46], crop: [0.12, 0] },
  // A sinkhole: a sheer round drop into the dark.
  { name: 'pits', seed: 62, subject: { kind: 'pits', id: 'pit4' }, cam: [20, 40, 18], crop: [0, 0] },
  { name: 'boulders', seed: 35, subject: { kind: 'boulder_fields', id: 'bfield1' }, cam: [160, 24, 16], crop: [0, 0] },
  { name: 'dunes', seed: 75, subject: { kind: 'dunes', id: 'dune1' }, cam: [64, 42, 22], crop: [0, 0] },
  // Looking towards the sun, so the glossy ice shows its sheen.
  { name: 'ice', seed: 28, subject: { kind: 'ice', id: 'ice3' }, cam: [64, 34, 18], crop: [0, 0] },
  { name: 'geyser', seed: 40, speed: 0.5, subject: { kind: 'geysers', id: 'g1' }, lift: 4.5, cam: [154, 14, 24], crop: [0.24, 0], erupt: true },
  { name: 'flora', seed: 11, subject: { point: [-5, 31] }, lift: 2, cam: [200, 14, 18], crop: [0.1, 0] },
  { name: 'spires', seed: 28, subject: { point: [-8.4, -15.8] }, lift: 2, cam: [250, 10, 10], crop: [0, 0] },
  { name: 'water', seed: 0, subject: { kind: 'water', id: 'w1' }, cam: [154, 35, 26], crop: [0, 0] },
  // A small crater and cracks (the pebbles are the little brown stones).
  // TRY
  { name: 'canyon2', seed: 69, subject: { point: [-39.8, 4] }, cam: [350, 45, 38], crop: [0.1, 0] },
  { name: 'canyon3', seed: 69, subject: { point: [-38.8, -20] }, cam: [90, 40, 34], crop: [0.1, 0] },
  { name: 'dunes2', seed: 66, subject: { kind: 'dunes', id: 'dune1' }, cam: [64, 30, 30], crop: [0, 0] },
  { name: 'dunes3', seed: 66, subject: { kind: 'dunes', id: 'dune1' }, cam: [154, 28, 32], crop: [0, 0] },
  { name: 'spires2', seed: 28, subject: { point: [-8.4, -15.8] }, lift: 2.5, cam: [262, 14, 15], crop: [0.25, 0] },
  { name: 'water2', seed: 0, subject: { kind: 'water', id: 'w1' }, cam: [200, 24, 24], crop: [0, 0] },
  { name: 'bushes', seed: 11, subject: { point: [0.9, 33.0] }, lift: 0.3, cam: [100, 32, 5.5], crop: [0, 0] },
  { name: 'bushes2', seed: 11, subject: { point: [-0.7, 33.0] }, lift: 0.3, cam: [120, 25, 7], crop: [0, 0] },
  // END TRY
  { name: 'ground', seed: 9, subject: { point: [-2.2, -28.6] }, cam: [10, 22, 6.5], crop: [0, 0] },
]
const DEFAULT_SPEED = 1

const only = (process.env.ONLY || '').split(',').filter(Boolean)
const seedOverride = Object.fromEntries((process.env.SEEDS || '').split(',').filter(Boolean).map((p) => p.split(':')).map(([n, s]) => [n, Number(s)]))
const shots = SHOTS.filter((s) => !only.length || only.includes(s.name)).map((s) => ({ ...s, seed: seedOverride[s.name] ?? s.seed }))
if (!shots.length) {
  console.error(`No shot named ${only.join(', ')}. Shots: ${SHOTS.map((s) => s.name).join(', ')}`)
  process.exit(2)
}
const candidates = [process.env.CHROME, `${process.env.HOME}/.cache/ms-playwright/chromium-1254/chrome-linux64/chrome`, '/usr/bin/google-chrome', '/usr/bin/chromium'].filter(Boolean)
const executablePath = candidates.find((p) => fs.existsSync(p))
if (!executablePath) {
  console.error('No Chrome or Chromium found. Set CHROME=/path/to/chrome.')
  process.exit(2)
}
fs.mkdirSync(outDir, { recursive: true })
if (rawDir) fs.mkdirSync(rawDir, { recursive: true })

const browser = await chromium.launch({ executablePath, args: ['--use-gl=angle', '--use-angle=swiftshader', '--ignore-gpu-blocklist'] })
const page = await browser.newPage({ viewport: { width: WIDTH, height: HEIGHT }, deviceScaleFactor: 1 })
page.setDefaultTimeout(180000) // software WebGL on a busy machine can take a while per frame
const conv = await browser.newPage() // a blank page that crops, scales and encodes the JPEGs
const problems = []
page.on('pageerror', (e) => problems.push(String(e)))
page.on('console', (m) => {
  if (m.type() === 'error' && !/favicon|Failed to load resource/.test(m.text())) problems.push(m.text())
})

const rad = (d) => (d * Math.PI) / 180
const reset = async (seed, speed) => {
  const r = await fetch(`${sim}/api/reset`, { method: 'POST', headers: { 'content-type': 'application/json' }, body: JSON.stringify({ seed, sim_speed: speed }) })
  if (!r.ok) throw new Error(`reset ${seed}: ${r.status} ${await r.text()}`)
}
// The true ground, sampled like the dashboard does where the fleet has not mapped (nearest sample).
const loadTruth = async () => {
  const t = await (await fetch(`${sim}/api/truth`)).json()
  return (x, y) => {
    const i = Math.min(t.nx - 1, Math.max(0, Math.round((x - t.origin[0]) / t.res)))
    const j = Math.min(t.ny - 1, Math.max(0, Math.round((y - t.origin[1]) / t.res)))
    return t.heights[j * t.nx + i]
  }
}
const subjectAt = (features, s) => {
  if (s.point) return { x: s.point[0], y: s.point[1] }
  const list = features?.[s.kind]
  const f = Array.isArray(list) ? (s.id ? list.find((o) => o.id === s.id) : list[s.index ?? 0]) : null
  if (!f) throw new Error(`no ${s.kind} ${s.id ?? s.index ?? 0} in this world`)
  if (Array.isArray(f.outline)) {
    // A water body: the middle of its outline.
    const n = f.outline.length
    return { x: f.outline.reduce((a, p) => a + p[0], 0) / n, y: f.outline.reduce((a, p) => a + p[1], 0) / n }
  }
  return { x: f.x, y: f.y }
}

// Point the camera so the subject point P (ground + lift) shows at screen point `want`, the camera
// `dist` metres from it. The orbit target is where the line of sight through P meets the ground. The
// dashboard shifts the view a little to make room for its panels, so a second pass corrects for that.
const aim = async (ground, P, lift, [az, el, dist], want, box) => {
  const a = rad(az)
  const e = rad(el)
  const u = [Math.sin(a) * Math.cos(e), -Math.cos(a) * Math.cos(e), Math.sin(e)] // subject to camera
  const right = [Math.cos(a), Math.sin(a), 0]
  const up = [-Math.sin(a) * Math.sin(e), Math.cos(a) * Math.sin(e), Math.cos(e)]
  const m = dist / (box.height / 2 / Math.tan(rad(FOV / 2))) // metres per pixel at the subject
  const Q = [P[0], P[1], P[2]] // the point the middle of the view passes through
  let err = Infinity
  for (let pass = 0; pass < 4 && err > 3; pass++) {
    let s = 0
    while (s < 400 && Q[2] - s * u[2] > ground(Q[0] - s * u[0], Q[1] - s * u[1])) s += 0.05
    await page.evaluate((v) => window.__redshift.view(...v), [Q[0] - s * u[0], Q[1] - s * u[1], az, el, dist + s])
    await page.waitForTimeout(pass ? 900 : 1500)
    const sc = await page.evaluate(([x, y, l]) => window.__redshift.screenOf(x, y, l), [P[0], P[1], lift])
    const ex = sc.x - want.x
    const ey = sc.y - want.y
    err = Math.hypot(ex, ey)
    for (let k = 0; k < 3; k++) Q[k] += right[k] * ex * m - up[k] * ey * m
  }
  return err
}

const toJpeg = async (png, crop) => {
  const run = (q) => conv.evaluate(async ([b64, w, q, top, bottom]) => {
    const img = new Image()
    img.src = `data:image/png;base64,${b64}`
    await img.decode()
    const sy = Math.round(img.height * top)
    const sh = Math.round(img.height * (1 - top - bottom))
    const h = Math.round((sh * w) / img.width)
    const c = document.createElement('canvas')
    c.width = w
    c.height = h
    const g = c.getContext('2d')
    g.imageSmoothingQuality = 'high'
    g.drawImage(img, 0, sy, img.width, sh, 0, 0, w, h)
    return c.toDataURL('image/jpeg', q)
  }, [png.toString('base64'), OUT_W, q, crop[0], crop[1]])
  let q = QUALITY
  let buf = Buffer.from((await run(q)).split(',')[1], 'base64')
  while (buf.length > MAX_BYTES && q > 0.6) {
    q = Math.round((q - 0.04) * 100) / 100
    buf = Buffer.from((await run(q)).split(',')[1], 'base64')
  }
  return { buf, q }
}

let loaded = false
for (const shot of shots) {
  const speed = shot.speed ?? DEFAULT_SPEED
  await reset(shot.seed, speed)
  if (!loaded) {
    await page.goto(`${app}/?debug`, { waitUntil: 'load' })
    await page.waitForFunction(() => window.__redshift?.state?.().world, null, { timeout: 90000 })
    await page.waitForTimeout(3000)
    await page.keyboard.press('r') // Reality on: the whole true sector
    for (const k of ['f', 'p', 'l']) await page.keyboard.press(k) // frontiers, paths and LiDAR off
    await page.addStyleTag({ content: 'body { visibility: hidden !important } canvas { visibility: visible !important }' })
    loaded = true
  }
  await page.waitForFunction((s) => window.__redshift.state().world?.seed === s, shot.seed, { timeout: 90000 })
  await page.waitForTimeout(4000)
  const ground = await loadTruth()
  // Reality off and on again, now that the true heights are in: the ice sheets take their height
  // from the ground when they are drawn, and are drawn under it if that was before the truth loaded.
  for (const k of ['r', 'r']) {
    await page.keyboard.press(k)
    await page.waitForTimeout(600)
  }
  const info = await page.evaluate(() => {
    const s = window.__redshift.state()
    return { features: s.features, biome: s.world?.biome ?? null }
  })
  const { x, y } = subjectAt(info.features, shot.subject)
  const lift = shot.lift ?? 0
  const crop = shot.crop ?? [0, 0]
  const box = await page.locator('canvas').first().boundingBox()
  const place = shot.place ?? [0.5, 0.5]
  const want = { x: box.x + place[0] * box.width, y: box.y + box.height * (crop[0] + place[1] * (1 - crop[0] - crop[1])) }
  const err = await aim(ground, [x, y, ground(x, y) + lift], lift, shot.cam, want, box)
  let png
  let note = ''
  if (shot.erupt) {
    // Catch an eruption early (at least 6 sim seconds to go), give the steam 2.5 s to rise, and
    // check the vent was still erupting once the frame was taken; else try the next eruption.
    const vent = () => page.evaluate((id) => {
      const s = window.__redshift.state()
      const g = (s.features.geysers || []).find((v) => v.id === id)
      return g ? { active: g.active, left: g.next_s - s.t } : null
    }, shot.subject.id)
    for (let attempt = 0; attempt < 3 && !note; attempt++) {
      await page.waitForFunction((id) => {
        const s = window.__redshift.state()
        const g = (s.features.geysers || []).find((v) => v.id === id)
        return g && g.active && g.next_s - s.t > 6
      }, shot.subject.id, { timeout: 400000, polling: 200 })
      await page.waitForTimeout(2500)
      png = await page.screenshot({ clip: box })
      const v = await vent()
      if (v?.active) note = `, erupting (${v.left.toFixed(1)} sim s to go)`
    }
    if (!note) {
      note = ', NOT erupting'
      problems.push(`${shot.name}: the vent stopped before the shot three times`)
    }
  } else {
    await page.waitForTimeout(shot.wait ?? 2500)
    png = await page.screenshot({ clip: box })
  }
  if (rawDir) fs.writeFileSync(path.join(rawDir, `${shot.name}.png`), png)
  const { buf, q } = await toJpeg(png, crop)
  const file = path.join(outDir, `${shot.name}.jpg`)
  fs.writeFileSync(file, buf)
  const b = info.biome ? `${info.biome.id}, ${info.biome.secondary} regions` : 'no biome'
  console.log(`saved ${file} (seed ${shot.seed}, ${b}, subject ${x.toFixed(1)},${y.toFixed(1)}, aim off ${err.toFixed(0)} px, ${Math.round(buf.length / 1024)} KB at q ${q}${note})`)
}

await browser.close()
if (problems.length) {
  console.log('problems:\n' + problems.join('\n'))
  process.exit(1)
}
