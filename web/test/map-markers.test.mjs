import test from 'node:test'
import assert from 'node:assert/strict'
import { GLYPH_COLOR, MAP_MARKER_FRAG, MAP_MARKER_VERT, MARKER_FAR, MARKER_NEAR, floraGlyphColor, hexRgb, makeMapMarkerGeometry, makeMapMarkerMaterial, markerFade, ventPulse } from '../src/scene/mapMarkers.js'
import { FLORA, TERRAIN } from '../src/lib/terrainTheme.js'
import { balanced, declaredUniforms } from './glsl.mjs'

const close = (a, b, eps = 1e-9) => a.length === b.length && a.every((v, k) => Math.abs(v - b[k]) <= eps)

test('markers fade out close to the camera and are full from the overview', () => {
  assert.equal(markerFade(0), 0)
  assert.equal(markerFade(MARKER_NEAR - 1), 0)
  assert.equal(markerFade(MARKER_NEAR), 0)
  assert.equal(markerFade(MARKER_FAR), 1)
  assert.equal(markerFade(140), 1)
  assert.equal(markerFade(Number.NaN), 0)
  const mid = markerFade((MARKER_NEAR + MARKER_FAR) / 2)
  assert.ok(Math.abs(mid - 0.5) < 1e-9)
  let prev = -1
  for (let d = 0; d <= 80; d += 0.5) {
    const f = markerFade(d)
    assert.ok(f >= prev && f >= 0 && f <= 1)
    prev = f
  }
  // Gone at a close-up, full at a sector overview.
  assert.ok(MARKER_NEAR >= 15 && MARKER_FAR <= 60)
})

test('tree glyphs take the cap colour by hue, the others the legend colours', () => {
  assert.ok(close(floraGlyphColor(0), hexRgb(FLORA.violet)))
  assert.ok(close(floraGlyphColor(1), hexRgb(FLORA.teal)))
  assert.ok(close(floraGlyphColor(-3), hexRgb(FLORA.violet)))
  assert.ok(close(floraGlyphColor(7), hexRgb(FLORA.teal)))
  assert.ok(close(floraGlyphColor(undefined), hexRgb(FLORA.violet)))
  const half = floraGlyphColor(0.5)
  const v = hexRgb(FLORA.violet)
  const t = hexRgb(FLORA.teal)
  assert.ok(close(half, v.map((a, k) => (a + t[k]) / 2)))
  assert.ok(close(hexRgb('#ff8000'), [1, 128 / 255, 0]))
  assert.ok(close(GLYPH_COLOR.bush, hexRgb(FLORA.bulb)))
  assert.ok(close(GLYPH_COLOR.geyser, hexRgb(TERRAIN.geyser)))
  assert.ok(close(GLYPH_COLOR.boulders, hexRgb(TERRAIN.boulder)))
})

test('vents pulse only while they erupt', () => {
  assert.equal(ventPulse(false, 3), 0)
  assert.equal(ventPulse(true, 3, true), 1)
  for (let t = 0; t < 10; t += 0.1) {
    const p = ventPulse(true, t)
    assert.ok(p >= 0.35 - 1e-9 && p <= 1 + 1e-9)
  }
})

test('marker shaders are well formed and every uniform is provided', () => {
  assert.ok(balanced(MAP_MARKER_VERT) && balanced(MAP_MARKER_FRAG))
  const m = makeMapMarkerMaterial()
  for (const name of [...declaredUniforms(MAP_MARKER_VERT), ...declaredUniforms(MAP_MARKER_FRAG)]) assert.ok(name in m.uniforms, `${name} missing`)
  assert.equal(m.uniforms.uNear.value, MARKER_NEAR)
  assert.equal(m.uniforms.uFar.value, MARKER_FAR)
  assert.equal(m.depthWrite, false)
  assert.equal(m.depthTest, true)
  assert.equal(m.toneMapped, false)
  const g = makeMapMarkerGeometry(8)
  for (const a of ['position', 'aCenter', 'aColor', 'aData']) assert.ok(g.getAttribute(a), `${a} missing`)
  assert.equal(g.getAttribute('aData').count, 8)
  assert.equal(g.instanceCount, 0)
})
