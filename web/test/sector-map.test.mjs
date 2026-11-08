import test from 'node:test'
import assert from 'node:assert/strict'
import { MARKER_R, planetLook } from '../src/globe/globeMath.js'
import { MAP_RAMP, PATCH_HALF_DEG, PATCH_LIFT, SITE_LIFT, paintSectorMap, patchDir, patchGeometry, patchHalfAngles, ringMinScale } from '../src/globe/sectorMap.js'
import { SECTOR_FRAG, SECTOR_VERT, makeSectorMapMaterial } from '../src/globe/sectorMapMaterial.js'
import { PLANET_FRAG, makePlanetMaterial } from '../src/globe/planetMaterial.js'
import { applyCells, createTerrain, normalizeMap } from '../src/state/simState.js'
import { balanced, declaredUniforms } from './glsl.mjs'

const dot = (a, b) => a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
const deg = (a, b) => (Math.acos(Math.min(1, Math.max(-1, dot(a, b)))) * 180) / Math.PI
const vert = (g, k) => [g.positions[k * 3], g.positions[k * 3 + 1], g.positions[k * 3 + 2]]
const unit = (v) => {
  const l = Math.hypot(...v)
  return v.map((x) => x / l)
}

const MAP = normalizeMap({ nx: 21, ny: 21, res: 0.5, min_h: 0, max_h: 4 })
const look = planetLook({ name: 'Redshift-b', gravity_g: 2, seed: 42 })

test('the patch sits on the site, square, PATCH_HALF_DEG either side, north up', () => {
  const segs = 8
  const g = patchGeometry(look.frame, MAP, { segs })
  const n = segs + 1
  assert.equal(g.positions.length, n * n * 3)
  assert.equal(g.indices.length, segs * segs * 6)
  const mid = (segs / 2) * n + segs / 2
  const c = vert(g, mid)
  assert.ok(Math.abs(Math.hypot(...c) - (1 + PATCH_LIFT)) < 1e-6)
  assert.ok(deg(unit(c), look.frame.up) < 1e-4, 'the middle is on the site')
  // The middle of the north edge (v = 1) lies due north of the site, the half-size away.
  const north = unit(vert(g, segs * n + segs / 2))
  assert.ok(Math.abs(deg(north, look.frame.up) - PATCH_HALF_DEG) < 1e-3)
  assert.ok(dot(north, look.frame.north) > 0 && Math.abs(dot(north, look.frame.east)) < 1e-6)
  const east = unit(vert(g, mid + segs / 2))
  assert.ok(Math.abs(deg(east, look.frame.up) - PATCH_HALF_DEG) < 1e-3)
  assert.ok(dot(east, look.frame.east) > 0 && Math.abs(dot(east, look.frame.north)) < 1e-6)
  // Map north is the planet's north there: the north edge is nearer the pole.
  assert.ok(north[1] > unit(c)[1])
  // Every vertex is off the ground.
  for (let k = 0; k < n * n; k++) assert.ok(Math.hypot(...vert(g, k)) > 1.0015)
})

test('a non-square sector keeps its aspect; uvs land on texel centres', () => {
  const wide = normalizeMap({ nx: 41, ny: 21, res: 1 })
  const h = patchHalfAngles(wide)
  assert.ok(Math.abs((h.east * 180) / Math.PI - PATCH_HALF_DEG) < 1e-9)
  assert.ok(Math.abs(h.north / h.east - 0.5) < 1e-9)
  const g = patchGeometry(look.frame, wide, { segs: 4 })
  assert.ok(Math.abs(g.uvs[0] - 0.5 / 41) < 1e-6 && Math.abs(g.uvs[1] - 0.5 / 21) < 1e-6)
  const last = (g.uvs.length >> 1) - 1
  assert.ok(Math.abs(g.uvs[last * 2] - 40.5 / 41) < 1e-6 && Math.abs(g.uvs[last * 2 + 1] - 20.5 / 21) < 1e-6)
  assert.deepEqual([g.local[0], g.local[1], g.local[last * 2], g.local[last * 2 + 1]], [0, 0, 1, 1])
  assert.ok(deg(patchDir(look.frame, 0, 0), look.frame.up) < 1e-6)
})

test('the ring frames the patch corners', () => {
  const s = ringMinScale(MAP)
  const h = patchHalfAngles(MAP)
  const corner = Math.atan(Math.hypot(Math.tan(h.east), Math.tan(h.north)))
  const ringAngle = Math.atan((MARKER_R * s) / SITE_LIFT)
  assert.ok(ringAngle > corner && ringAngle < corner * 1.2)
})

test('a fresh map is fully transparent; known cells are opaque and deterministic', () => {
  const T = createTerrain(MAP)
  const empty = paintSectorMap(T, {})
  assert.equal(empty.length, 21 * 21 * 4)
  assert.ok(empty.every((v) => v === 0))
  applyCells(T, [[3, 4, 1.0, 5], [4, 4, 1.2, 5], [10, 10, 4, 0], [0, 0, 0, 0]], 0)
  const a = paintSectorMap(T, {})
  const alpha = (px, i, j) => px[(j * 21 + i) * 4 + 3]
  assert.equal(alpha(a, 3, 4), 255)
  assert.equal(alpha(a, 10, 10), 255)
  assert.equal(alpha(a, 5, 5), 0)
  let opaque = 0
  for (let k = 3; k < a.length; k += 4) if (a[k]) opaque++
  assert.equal(opaque, 4)
  // Same map, same bytes, also when painted into the previous buffer.
  const copy = Uint8Array.from(a)
  assert.equal(paintSectorMap(T, {}, a), a)
  assert.deepEqual(a, copy)
  // Height tints: the highest flat cell is the ramp's top colour, the lowest its bottom.
  const rgbAt = (px, i, j) => Array.from(px.slice((j * 21 + i) * 4, (j * 21 + i) * 4 + 3))
  assert.deepEqual(rgbAt(a, 10, 10), MAP_RAMP[MAP_RAMP.length - 1])
  assert.deepEqual(rgbAt(a, 0, 0), MAP_RAMP[0])
})

test('fleet zones show: water opaque blue even over unmapped ground', () => {
  const T = createTerrain(MAP)
  const px = paintSectorMap(T, { water_zones: [[MAP.ox + 5, MAP.oy + 5, 1]], flora_zones: [[0, 0, 0]] })
  const k = (10 * 21 + 10) * 4
  assert.equal(px[k + 3], 255)
  assert.ok(px[k + 2] > px[k], 'blue')
  assert.equal(px[3], 0)
})

test('a reset to a different grid gets a buffer of the new size', () => {
  const big = paintSectorMap(createTerrain(MAP), {})
  const small = createTerrain(normalizeMap({ nx: 11, ny: 7 }))
  applyCells(small, [[1, 1, 0, 0]], 0)
  const px = paintSectorMap(small, {}, big)
  assert.notEqual(px, big)
  assert.equal(px.length, 11 * 7 * 4)
  assert.equal(px[(1 * 11 + 1) * 4 + 3], 255)
  assert.equal(paintSectorMap(null, {}).length, 0)
})

test('the map shader is well formed and its uniforms provided', () => {
  assert.ok(balanced(SECTOR_VERT) && balanced(SECTOR_FRAG))
  const m = makeSectorMapMaterial()
  for (const name of [...declaredUniforms(SECTOR_VERT), ...declaredUniforms(SECTOR_FRAG)]) assert.ok(name in m.uniforms, `${name} missing`)
  assert.equal(m.depthWrite, false)
  assert.equal(m.transparent, true)
  assert.equal(m.polygonOffset, true)
})

test('the planet starts unsurveyed and only works out the true surface when revealed', () => {
  assert.equal(makePlanetMaterial(true).uniforms.uReveal.value, 0)
  assert.match(PLANET_FRAG, /if \(uReveal > 0\.001\) col = mix\(blank, revealed\(n0, L, V\), uReveal\);/)
})
