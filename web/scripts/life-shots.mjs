// Takes the README pictures of the lifeform kinds (docs/images/life/): a grazer herd, a skitterer
// swarm, drifters, a crawler, a burrower that is up, and a rover observing a creature. Same look as
// biome-shots.mjs: Reality on, frontiers, paths and LiDAR off, only the 3D canvas, 1600x900 window.
// Needs a sim with AUTO_LAUNCH=1 (biomes and More lifeforms on, the default from the environment),
// vite (dev or preview) pointed at it, and ImageMagick's convert for the JPEGs.
//   node scripts/life-shots.mjs [appUrl] [simUrl] [outDir]
// Env: LIFE_SEEDS="grazer:14,crawler:31,..." changes seeds, LIFE_SET="drifter.el=12,observe.az=side2"
// changes other settings below, ONLY=crawler,burrower retakes some, RAW_DIR keeps the full size
// PNGs (default: a temp folder), CHROME picks the browser.
//
// How each picture is taken: reset the sim to the picture's seed, wait until the fleet tracks a
// creature of that kind (only tracked creatures are drawn solid, the others are faint Reality
// ghosts), then hold back the sim's messages to the page so the scene stays on that instant while
// the camera flies in and the slow software renderer settles (the sim keeps running; the held
// messages are passed on before the next reset). Where the creatures are differs from run to run,
// so the camera is placed from the creature, the rover and the true terrain: of 24 directions it
// takes the one with a clear line of sight (hills, spires, trees), the fewest target markers around
// the subject and the least turn from the preferred direction.
import fs from 'node:fs'
import os from 'node:os'
import path from 'node:path'
import { execFileSync } from 'node:child_process'
import { chromium } from 'playwright-core'

const app = process.argv[2] || 'http://localhost:5183'
const sim = process.argv[3] || 'http://localhost:8103'
const outDir = process.argv[4] || 'docs/images/life'
const rawDir = process.env.RAW_DIR || fs.mkdtempSync(path.join(os.tmpdir(), 'life-shots-'))
const width = 1600
const height = 900
const outWidth = 960
const quality = 82
const speed = 3 // sim seconds per real second while waiting for a subject
const TIMEOUT_S = 240 // wait this long for a subject, then reset and try once more

// One entry per picture.
//   kind: lifeform kind. tracked: a live fleet track on it. up: a burrower that is all the way up.
//   observed: a rover is observing it (that rover is then "the rover" below, else the nearest).
//   clear: no found target within this many metres of the rover to creature line.
//   minTracked: at least this many tracked in its group (creatures of the kind within groupR m).
//   center: 'one' (the creature), 'group' (middle of its tracked group) or 'mid' (halfway to the rover).
//   lift: height (m) of the subject's middle above the ground (drifters use their own height).
//   az: preferred camera direction, degrees (0: camera south of the subject), or named from the
//   rover: 'side' and 'side2' (off to one side of the rover to creature line), 'behind' (on the
//   rover's side, looking past it at the creature), 'front' (beyond the creature, looking back).
//   el, dist: camera elevation (degrees) and distance (m) for window.__redshift.view. The orbit
//   controls keep the camera at least 5 m away and about 8.6 degrees up, so closer or lower is clamped.
//   near, lookH: instead of dist, put the camera `near` metres (along the ground) from the subject
//   and aim past it so the view's centre line crosses `lookH` metres above it (for the floating
//   drifters, which a camera looking down at their ground spot would put at the top of the frame).
//   zoom: crop around the subject by this factor before scaling to 960 px wide (1.27 keeps the
//   canvas's own pixels). aspect: output height / width.
const SHOTS = [
  { id: 'grazer', seed: 14, kind: 'grazer', tracked: true, minTracked: 3, groupR: 6, center: 'group', lift: 0.8, az: 'side2', el: 12, dist: 9, zoom: 1.1, aspect: 0.7 },
  { id: 'skitterer', seed: 14, kind: 'skitterer', tracked: true, minTracked: 5, groupR: 4, center: 'group', lift: 0.2, az: 'side', el: 25, dist: 5.5, zoom: 1.1, aspect: 0.7 },
  { id: 'drifter', seed: 35, kind: 'drifter', tracked: true, groupR: 6, center: 'group', lift: 0, az: 'side', el: 10, near: 6, lookH: 1.6, zoom: 1.1, aspect: 0.7 },
  { id: 'crawler', seed: 31, kind: 'crawler', tracked: true, groupR: 4, center: 'one', lift: 0.5, az: 'side', el: 16, dist: 6, zoom: 1.1, aspect: 0.7 },
  { id: 'burrower', seed: 9, kind: 'burrower', tracked: true, up: true, groupR: 5, center: 'one', lift: 0.35, az: 'behind', el: 12, dist: 5, zoom: 1.27, aspect: 0.7 },
  { id: 'observe', seed: 9, kind: 'crawler', observed: true, clear: 3, groupR: 4, center: 'mid', lift: 0.5, az: 'side', el: 24, dist: 11, zoom: 1.1, aspect: 0.7 },
]

for (const p of (process.env.LIFE_SEEDS || '').split(',').filter(Boolean)) {
  const [id, seed] = p.split(':')
  const s = SHOTS.find((q) => q.id === id)
  if (!s) throw new Error(`LIFE_SEEDS: no picture called ${id}`)
  s.seed = Number(seed)
}
for (const p of (process.env.LIFE_SET || '').split(',').filter(Boolean)) {
  const [, id, key, value] = p.match(/^(\w+)\.(\w+)=(.*)$/) || []
  const s = SHOTS.find((q) => q.id === id)
  if (!s) throw new Error(`LIFE_SET: cannot use ${p}`)
  s[key] = value === 'true' ? true : value === 'false' ? false : Number.isFinite(Number(value)) && value !== '' ? Number(value) : value
}
const only = (process.env.ONLY || '').split(',').filter(Boolean)
const shots = only.length ? SHOTS.filter((s) => only.includes(s.id)) : SHOTS

const candidates = [process.env.CHROME, `${process.env.HOME}/.cache/ms-playwright/chromium-1254/chrome-linux64/chrome`, '/usr/bin/google-chrome', '/usr/bin/chromium'].filter(Boolean)
const executablePath = candidates.find((p) => fs.existsSync(p))
fs.mkdirSync(outDir, { recursive: true })
fs.mkdirSync(rawDir, { recursive: true })

const browser = await chromium.launch({ executablePath, args: ['--use-gl=angle', '--use-angle=swiftshader', '--ignore-gpu-blocklist'] })
const page = await browser.newPage({ viewport: { width, height }, deviceScaleFactor: 1 })
const problems = []
page.on('pageerror', (e) => problems.push(String(e)))
page.on('console', (m) => {
  if (m.type() === 'error' && !/favicon|Failed to load resource/.test(m.text())) problems.push(m.text())
})

// The page's sim socket goes through here, so its messages can be held back for a while.
let holding = false
let held = []
let toPage = null
await page.routeWebSocket(/\/ws$/, (ws) => {
  const server = ws.connectToServer()
  ws.onMessage((m) => server.send(m))
  server.onMessage((m) => (holding ? held.push(m) : ws.send(m)))
  toPage = ws
})
const hold = () => (holding = true)
const release = () => {
  holding = false
  const h = held
  held = []
  for (const m of h) toPage?.send(m)
}

const reset = async (seed) => {
  const r = await fetch(`${sim}/api/reset`, { method: 'POST', headers: { 'content-type': 'application/json' }, body: JSON.stringify({ seed, sim_speed: speed }) })
  if (!r.ok) throw new Error(`reset ${seed}: ${r.status} ${await r.text()}`)
}
// Resolves after n more frames (the software renderer can take seconds per frame).
const frames = (n) => page.evaluate((n) => new Promise((res) => { let k = 0; const f = () => (++k >= n ? res(k) : requestAnimationFrame(f)); requestAnimationFrame(f) }), n)

// The subject for a picture in the page's current state, or null.
const find = (s) =>
  page.evaluate((s) => {
    const st = window.__redshift.state()
    const live = st.tracks.filter((k) => k.status === 'tracking')
    // Within 2.5 m: the page draws a creature solid when a live track is within 3 m of it.
    const trackOf = (l) => live.find((k) => Math.abs(k.x - l.x) < 2.5 && Math.abs(k.y - l.y) < 2.5 && Math.hypot(k.x - l.x, k.y - l.y) < 2.5)
    const kindOf = (l) => l.kind || 'grazer'
    const watcher = (l) => { const k = trackOf(l); return k && Object.values(st.rovers).find((r) => r.state === 'observing' && r.track_id === k.id) }
    const groupOf = (l) => st.lifeforms.filter((q) => kindOf(q) === s.kind && Math.hypot(q.x - l.x, q.y - l.y) < s.groupR)
    const tracked = (g) => g.filter((q) => trackOf(q)).length
    let c = st.lifeforms.filter((l) => kindOf(l) === s.kind)
    if (s.up) c = c.filter((l) => (l.up ?? 1) >= 0.99)
    if (s.tracked) c = c.filter(trackOf)
    if (s.observed) c = c.filter(watcher)
    if (s.clear) {
      // No found target (a big marker) within `clear` metres of the rover to creature line.
      const segDist = (t, a, b) => {
        const vx = b.x - a.x, vy = b.y - a.y
        const k = Math.max(0, Math.min(1, ((t.x - a.x) * vx + (t.y - a.y) * vy) / (vx * vx + vy * vy || 1)))
        return Math.hypot(t.x - a.x - k * vx, t.y - a.y - k * vy)
      }
      c = c.filter((l) => { const r = watcher(l) || l; return !(st.targets || []).some((t) => segDist(t, r, l) < s.clear) })
    }
    if (s.minTracked) c = c.filter((l) => tracked(groupOf(l)) >= s.minTracked)
    if (!c.length) return null
    c.sort((a, b) => tracked(groupOf(b)) - tracked(groupOf(a)))
    const l = c[0]
    const rovers = Object.values(st.rovers).map((r) => ({ id: r.id, x: r.x, y: r.y, state: r.state, d: Math.hypot(r.x - l.x, r.y - l.y) })).sort((a, b) => a.d - b.d)
    const w = s.observed && watcher(l)
    const rover = w ? rovers.find((r) => r.id === w.id) : rovers[0]
    const group = groupOf(l).filter((q) => trackOf(q)).map((q) => ({ x: q.x, y: q.y, z: q.z }))
    const f = st.features || {}
    return {
      t: st.t,
      biome: st.world?.biome?.id ?? null,
      l: { id: l.id, x: l.x, y: l.y, z: l.z, up: l.up },
      rover,
      group,
      spires: (f.spires || []).filter((p) => p.standing !== false).map((p) => ({ x: p.x, y: p.y, r: p.radius, h: p.height })),
      trees: (f.flora || []).filter((p) => p.kind === 'tree').map((p) => ({ x: p.x, y: p.y, r: p.radius, h: p.height })),
      found: (st.targets || []).map((p) => ({ x: p.x, y: p.y })),
    }
  }, s)

// ---------------------------------------------------------------- camera placement
const RAD = Math.PI / 180
const TAN_V = Math.tan(21 * RAD) // half the vertical field of view (42 degrees, App.jsx)

// Height of the true terrain at (x, y), from /api/truth.
const heightOf = (truth) => (x, y) => {
  const i = Math.min(truth.nx - 1, Math.max(0, Math.round((x - truth.origin[0]) / truth.res)))
  const j = Math.min(truth.ny - 1, Math.max(0, Math.round((y - truth.origin[1]) / truth.res)))
  return truth.heights[j * truth.nx + i]
}

// view() arguments for a picture with the camera in direction az from the subject point (sx, sy).
function viewFor(s, sx, sy, az) {
  if (!s.near) return [sx, sy, az, s.el, s.dist]
  // Look past the subject: aim `push` metres beyond it (away from the camera).
  const push = s.lookH / Math.tan(s.el * RAD)
  return [sx - push * Math.sin(az * RAD), sy + push * Math.cos(az * RAD), az, s.el, (s.near + push) / Math.cos(s.el * RAD)]
}

// Camera position and a projector to the screen (x, y in -1..1 of the canvas, before the
// dashboard's own view shift) for view() arguments v.
function cameraOf(v, H, aspect) {
  const [tx, ty, az, el, dist] = v
  const d = Math.max(5, dist) // the orbit controls' minimum distance
  const e = Math.max(8.6, el) * RAD // and lowest elevation
  const tz = H(tx, ty)
  const C = [tx + d * Math.sin(az * RAD) * Math.cos(e), ty - d * Math.cos(az * RAD) * Math.cos(e), tz + d * Math.sin(e)]
  const f = [(tx - C[0]) / d, (ty - C[1]) / d, (tz - C[2]) / d]
  const rn = Math.hypot(f[0], f[1]) || 1
  const r = [f[1] / rn, -f[0] / rn, 0]
  const u = [r[1] * f[2] - r[2] * f[1], r[2] * f[0] - r[0] * f[2], r[0] * f[1] - r[1] * f[0]]
  const dot = (a, b) => a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
  const project = (P) => {
    const q = [P[0] - C[0], P[1] - C[1], P[2] - C[2]]
    const depth = dot(q, f)
    return { depth, x: dot(q, r) / (depth * TAN_V * aspect), y: dot(q, u) / (depth * TAN_V) }
  }
  return { C, project }
}

// Is the straight line from the camera C to P blocked by the terrain, a spire or a tree?
function blocked(C, P, H, f) {
  for (let k = 1; k < 38; k++) {
    const a = k / 40 // the last half metre or so is the subject itself
    const x = C[0] + (P[0] - C[0]) * a
    const y = C[1] + (P[1] - C[1]) * a
    const z = C[2] + (P[2] - C[2]) * a
    if (H(x, y) > z - 0.05) return true
    for (const o of [...f.spires, ...f.trees]) {
      if (Math.hypot(x - o.x, y - o.y) < o.r + 0.2 && z < H(o.x, o.y) + o.h) return true
    }
  }
  return false
}

// Picks the camera direction: clear sight to every key point (and the camera above the ground),
// then the fewest target markers and near spires around the subject, then the least turn.
function placeCamera(s, f, H, sx, sy, keys, aspect) {
  const { l, rover: r } = f
  const n = Math.hypot(l.x - r.x, l.y - r.y) || 1
  const ux = (l.x - r.x) / n
  const uy = (l.y - r.y) / n
  const azFrom = (cx, cy) => Math.atan2(cx, -cy) / RAD
  const named = { side: azFrom(-uy, ux), side2: azFrom(uy, -ux), behind: azFrom(-ux, -uy), front: azFrom(ux, uy) }
  const want = typeof s.az === 'number' ? s.az : named[s.az]
  const wx = 1 / s.zoom // the crop's half size on screen
  const wy = (s.aspect * aspect) / s.zoom
  let best = null
  for (let k = 0; k < 24; k++) {
    const off = ((k + 1) >> 1) * 15 * (k % 2 ? 1 : -1)
    const v = viewFor(s, sx, sy, want + off)
    const cam = cameraOf(v, H, aspect)
    let score = Math.abs(off) * 0.004
    if (H(cam.C[0], cam.C[1]) > cam.C[2] - 0.4) score += 1000
    for (const P of keys) if (blocked(cam.C, P, H, f)) score += 1000
    const mid = cam.project([sx, sy, H(sx, sy) + s.lift])
    // A target marker (a pole about 2.9 m high) or a spire in the cropped picture.
    const clutter = (x, y, top, w) => {
      let hit = 0
      for (const a of [0.2, 0.5, 0.9]) {
        const p = cam.project([x, y, H(x, y) + top * a])
        if (p.depth < 0.5) continue
        const dx = Math.abs(p.x - mid.x)
        const dy = Math.abs(p.y - mid.y)
        if (dx < wx && dy < wy) hit = Math.max(hit, (dx < wx / 2 && dy < wy / 2 ? 1.5 : 0.7) * (p.depth < mid.depth ? 1.5 : 1))
      }
      return hit * w
    }
    for (const t of f.found) score += clutter(t.x, t.y, 2.9, 1)
    for (const t of f.truthTargets) score += clutter(t.x, t.y, 2.9, 0.5)
    for (const p of f.spires) if (cam.project([p.x, p.y, H(p.x, p.y)]).depth < mid.depth) score += clutter(p.x, p.y, p.h, 1)
    // Every key point inside the crop.
    for (const P of keys) {
      const p = cam.project(P)
      if (p.depth < 0.5 || Math.abs(p.x - mid.x) > wx * 0.9 || Math.abs(p.y - mid.y) > wy * 0.9) score += 5
    }
    if (!best || score < best.score) best = { v, score, off }
  }
  return best
}

await page.goto(`${app}/?debug`, { waitUntil: 'load' })
await page.waitForFunction(() => window.__redshift?.state?.().world, null, { timeout: 90000 })
await frames(2)
await page.keyboard.press('r') // Reality on: the whole true sector
for (const k of ['f', 'p', 'l']) await page.keyboard.press(k) // frontiers, paths and LiDAR off
await page.addStyleTag({ content: 'body { visibility: hidden !important } canvas { visibility: visible !important }' })
const box = await page.locator('canvas').first().boundingBox()
const aspect = box.width / box.height

for (const s of shots) {
  let f = null
  for (let attempt = 0; attempt < 2 && !f; attempt++) {
    release()
    const gen = await page.evaluate(() => window.__redshift.state().gen)
    await reset(s.seed)
    await page.waitForFunction(([seed, g]) => window.__redshift.state().world?.seed === seed && window.__redshift.state().gen !== g, [s.seed, gen], { timeout: 120000, polling: 1000 })
    await page.evaluate(() => window.__redshift.view(0, 0, 0, 40, 12)) // a cheap close view while waiting
    const t0 = Date.now()
    while (!f && Date.now() - t0 < TIMEOUT_S * 1000) {
      f = await find(s)
      if (!f) await page.waitForTimeout(700)
    }
  }
  if (!f) {
    problems.push(`${s.id}: no ${s.kind} to photograph on seed ${s.seed}`)
    continue
  }
  hold()
  await page.waitForTimeout(300) // let a message already on its way land
  f = (await find(s)) || f
  const truth = await (await fetch(`${sim}/api/truth`)).json()
  f.truthTargets = truth.targets || []
  const H = heightOf(truth)

  // The subject point, and the points that must be in clear sight and in the picture.
  let [sx, sy] = [f.l.x, f.l.y]
  if (s.center === 'group' && f.group.length) {
    sx = f.group.reduce((a, q) => a + q.x, 0) / f.group.length
    sy = f.group.reduce((a, q) => a + q.y, 0) / f.group.length
  } else if (s.center === 'mid') {
    sx = (f.l.x + f.rover.x) / 2
    sy = (f.l.y + f.rover.y) / 2
  }
  const lz = (q) => (s.kind === 'drifter' && Number.isFinite(q.z) ? q.z : H(q.x, q.y) + s.lift)
  const keys = (s.center === 'group' && f.group.length ? f.group : [f.l]).map((q) => [q.x, q.y, lz(q)])
  if (s.center === 'mid') keys.push([f.rover.x, f.rover.y, H(f.rover.x, f.rover.y) + 0.8])
  // Drifters: centre the crop a little below their bells, so the tendrils and ground show.
  const lift = s.kind === 'drifter' ? (0.7 * keys.reduce((a, p) => a + p[2] - H(p[0], p[1]), 0)) / keys.length : s.lift
  const best = placeCamera({ ...s, lift }, f, H, sx, sy, keys, aspect)
  const view = best.v

  // Twice: the first call can land while the camera still eases to its start view.
  await page.evaluate((v) => window.__redshift.view(...v), view)
  await frames(2)
  await page.evaluate((v) => window.__redshift.view(...v), view)
  await frames(5) // the creatures ease into their held positions
  const raw = path.join(rawDir, `${s.id}.png`)
  await page.screenshot({ path: raw, clip: box })
  // Crop around the subject (zoom), then scale to the output width.
  const at = await page.evaluate(([x, y, h]) => window.__redshift.screenOf(x, y, h), [sx, sy, lift])
  const w = Math.round(box.width / s.zoom)
  const h = Math.min(Math.round(w * s.aspect), Math.round(box.height))
  const cx = Math.round(Math.min(Math.max(at.x - box.x - w / 2, 0), box.width - w))
  const cy = Math.round(Math.min(Math.max(at.y - box.y - h / 2, 0), box.height - h))
  const file = path.join(outDir, `${s.id}.jpg`)
  execFileSync('convert', [raw, '-crop', `${w}x${h}+${cx}+${cy}`, '+repage', '-resize', `${outWidth}x`, '-strip', '-interlace', 'Plane', '-quality', String(quality), file])
  console.log(`saved ${file} (seed ${s.seed}, biome ${f.biome}, t ${Math.round(f.t)} s, ${s.kind} ${f.l.id}, rover ${f.rover.id} ${f.rover.d.toFixed(1)} m ${f.rover.state}, az ${view[2].toFixed(0)} (turned ${best.off}), score ${best.score.toFixed(2)})`)
}
release()

await browser.close()
if (problems.length) {
  console.log('problems:\n' + problems.join('\n'))
  process.exit(1)
}
