import test from 'node:test'
import assert from 'node:assert/strict'
import { SUN_DIR } from '../src/scene/sceneConstants.js'
import {
  GLOBE_FOV,
  IDLE_SWAY,
  MAX_DIST,
  formatKm,
  formatLatLon,
  gravityG,
  hashSeed,
  homeDir,
  homeDistance,
  idleSway,
  latLonToDir,
  lerpPose,
  planetLook,
  planetName,
  planetRadiusKm,
  seedLook,
  siteFrame,
  siteFromSeed,
  slerp,
  smoother,
  sunDirFor,
  turnY,
  wrapAngle,
} from '../src/globe/globeMath.js'

const dot = (a, b) => a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
const len = (a) => Math.hypot(a[0], a[1], a[2])
const near = (a, b, eps = 1e-9) => a.forEach((v, i) => assert.ok(Math.abs(v - b[i]) < eps, `${a} vs ${b}`))

test('radius from gravity at Earth density, with a readable label', () => {
  assert.equal(planetRadiusKm(2), 12742)
  assert.equal(planetRadiusKm(1), 6371)
  assert.equal(planetRadiusKm(NaN), 6371)
  assert.equal(formatKm(12742), '12,742 km')
  assert.equal(formatKm(950), '950 km')
  assert.equal(formatKm(1234567), '1,234,567 km')
})

test('gravity and name from the world, with fallbacks', () => {
  assert.equal(gravityG({ gravity_g: 2 }), 2)
  assert.ok(Math.abs(gravityG({ gravity: 19.62 }) - 2) < 1e-9)
  assert.equal(gravityG(null), 2)
  assert.equal(planetName({ name: 'Redshift-b' }), 'Redshift-b')
  assert.equal(planetName({ name: '  ' }), 'Unnamed planet')
  assert.equal(planetName(undefined), 'Unnamed planet')
})

test('the site comes from the seed, near the equator', () => {
  assert.equal(hashSeed(42), 2279835011)
  assert.deepEqual(siteFromSeed(42), { lat: 4.4, lon: 44.9 })
  assert.deepEqual(siteFromSeed('42'), siteFromSeed(42))
  assert.deepEqual(siteFromSeed('abc'), { lat: -0.8, lon: -26.4 })
  assert.notDeepEqual(siteFromSeed(43), siteFromSeed(42))
  for (let s = 0; s < 200; s++) {
    const { lat, lon } = siteFromSeed(s)
    assert.ok(Math.abs(lat) <= 32 && lon >= -180 && lon < 180, `${s}: ${lat}, ${lon}`)
  }
})

test('latitude and longitude read like a map', () => {
  assert.equal(formatLatLon({ lat: 12.3, lon: -103.7 }), '12.3° N, 103.7° W')
  assert.equal(formatLatLon({ lat: -5, lon: 20 }), '5.0° S, 20.0° E')
  assert.equal(formatLatLon(siteFromSeed(42)), '4.4° N, 44.9° E')
})

test('directions: y is north, longitude 0 faces +z, east is +x there', () => {
  near(latLonToDir(0, 0), [0, 0, 1])
  near(latLonToDir(90, 30), [0, 1, 0], 1e-12)
  near(latLonToDir(0, 90), [1, 0, 0], 1e-12)
  const f = siteFrame(4.4, 44.9)
  for (const v of [f.up, f.east, f.north]) assert.ok(Math.abs(len(v) - 1) < 1e-12)
  assert.ok(Math.abs(dot(f.up, f.east)) < 1e-12)
  assert.ok(Math.abs(dot(f.up, f.north)) < 1e-12)
  assert.ok(Math.abs(dot(f.east, f.north)) < 1e-12)
  assert.ok(f.north[1] > 0, 'north points towards the north pole')
})

test('the site is in daylight under the sector sun, lit from the west-north-west', () => {
  const f = siteFrame(4.4, 44.9)
  const sun = sunDirFor(f)
  const k = Math.hypot(...SUN_DIR)
  assert.ok(Math.abs(len(sun) - 1) < 1e-12)
  assert.ok(Math.abs(dot(sun, f.up) - SUN_DIR[1] / k) < 1e-9)
  assert.ok(dot(sun, f.east) < 0)
  assert.ok(dot(sun, f.north) > 0)
})

test('home view: a little east and south of overhead, the planet filling the free view', () => {
  const f = siteFrame(4.4, 44.9)
  const h = homeDir(f)
  assert.ok(dot(h, f.up) > 0.9)
  assert.ok(dot(h, f.east) > 0)
  assert.ok(dot(h, f.north) < 0)
  assert.ok(Math.abs(homeDistance(GLOBE_FOV) - 4.5171) < 1e-3)
  assert.ok(Math.abs(homeDistance(GLOBE_FOV, 0.88) - 5.1046) < 1e-3)
  assert.ok(homeDistance(GLOBE_FOV, 0.5) > homeDistance(GLOBE_FOV, 1))
  assert.equal(homeDistance(GLOBE_FOV, 0.05), MAX_DIST)
})

test('poses ease: slerped direction, log distance, the short way round', () => {
  const a = { dir: [0, 0, 1], dist: 1.3, spin: 3 }
  const b = { dir: [1, 0, 0], dist: 5.2, spin: -3 }
  const p0 = lerpPose(a, b, 0)
  near(p0.dir, a.dir, 1e-12)
  assert.ok(Math.abs(p0.dist - 1.3) < 1e-12)
  const p1 = lerpPose(a, b, 1)
  near(p1.dir, b.dir, 1e-12)
  assert.ok(Math.abs(p1.dist - 5.2) < 1e-9)
  const mid = lerpPose(a, b, 0.5)
  assert.ok(Math.abs(mid.dist - Math.sqrt(1.3 * 5.2)) < 1e-9)
  assert.ok(Math.abs(mid.spin - Math.PI) < 1e-9, `spin ${mid.spin}`)
  assert.ok(Math.abs(len(slerp([0, 0, 1], [0.6, 0.8, 0], 0.37)) - 1) < 1e-12)
  assert.ok(Math.abs(len(slerp([0, 0, 1], [0, 0, -1], 0.5)) - 1) < 1e-12)
  assert.equal(smoother(0), 0)
  assert.equal(smoother(1), 1)
  assert.equal(smoother(0.5), 0.5)
  assert.equal(smoother(2), 1)
  assert.ok(Math.abs(wrapAngle(-6) - (2 * Math.PI - 6)) < 1e-12)
  assert.ok(Math.abs(wrapAngle(7)) < Math.PI)
})

test('the idle sway turns towards the sun and back, so the site and the middle of the disc stay lit', () => {
  assert.equal(idleSway(0), 0)
  assert.ok(Math.abs(idleSway(Math.PI) - IDLE_SWAY) < 1e-12)
  assert.ok(Math.abs(idleSway(2 * Math.PI)) < 1e-12)
  near(turnY([0, 0, 1], Math.PI / 2), [1, 0, 0], 1e-12) // the same turn as three.js rotation.y
  for (let s = 0; s < 200; s++) {
    const { lat, lon } = siteFromSeed(s)
    const f = siteFrame(lat, lon)
    const sun = sunDirFor(f)
    const h = homeDir(f)
    for (let k = 0; k <= 8; k++) {
      // The camera at home, in the frame of the planet turned by the sway.
      const c = turnY(h, -idleSway((k / 8) * Math.PI))
      assert.ok(dot(c, sun) > 0.25, `seed ${s}: the sun is at ${dot(c, sun)} over the middle of the disc`)
      assert.ok(dot(c, f.up) > 0.9, `seed ${s}: the site faces the camera at ${dot(c, f.up)}`)
    }
  }
})

test('surface knobs per seed, deterministic and in range', () => {
  assert.deepEqual(seedLook(42), seedLook(42))
  assert.notDeepEqual(seedLook(42).offset, seedLook(43).offset)
  for (let s = 0; s < 50; s++) {
    const l = seedLook(s)
    assert.ok(l.maria >= 0.06 && l.maria <= 0.18)
    assert.ok(l.craters >= 0.45 && l.craters <= 0.8)
    assert.ok(l.capN >= 0.86 && l.capN <= 0.92)
    assert.ok(l.capS >= 0.83 && l.capS <= 0.91)
    assert.ok(l.rust >= 0.25 && l.rust <= 0.75)
    assert.ok(l.offset.every((v) => Math.abs(v) <= 48.5))
  }
})

test('planetLook gathers it all', () => {
  const p = planetLook({ name: 'Redshift-b', gravity_g: 2, seed: 42 })
  assert.equal(p.name, 'Redshift-b')
  assert.equal(p.radiusKm, 12742)
  assert.deepEqual(p.site, { lat: 4.4, lon: 44.9 })
  assert.ok(dot(p.sun, p.frame.up) > 0.5)
  assert.deepEqual(p.surface, seedLook(42))
  assert.equal(planetLook(null).name, 'Unnamed planet')
})
