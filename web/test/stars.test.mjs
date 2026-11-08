import test from 'node:test'
import assert from 'node:assert/strict'
import { BAND, STAR_RADIUS, starData } from '../src/globe/stars.js'

test('the same sky every time', () => {
  const a = starData(400)
  const b = starData(400)
  assert.deepEqual(a.position, b.position)
  assert.deepEqual(a.mag, b.mag)
  assert.deepEqual(a.kind, b.kind)
})

test('stars sit on the sky sphere, mostly dim, with a band across', () => {
  const n = 2000
  const s = starData(n)
  let inBand = 0
  for (let i = 0; i < n; i++) {
    const x = s.position[i * 3]
    const y = s.position[i * 3 + 1]
    const z = s.position[i * 3 + 2]
    const r = Math.hypot(x, y, z)
    assert.ok(Math.abs(r - STAR_RADIUS) < 1e-3, `radius ${r}`)
    if (Math.abs((x * BAND[0] + y * BAND[1] + z * BAND[2]) / r) < 0.2) inBand++
  }
  assert.ok(inBand / n > 0.4, `band share ${inBand / n}`)
  assert.ok([...s.mag].filter((m) => m < 0.1).length / n > 0.5)
  assert.ok([...s.kind].every((k) => k <= 2))
  assert.ok([...s.phase].every((p) => p >= 0 && p < 2 * Math.PI))
})
