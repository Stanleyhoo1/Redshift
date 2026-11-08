import test from 'node:test'
import assert from 'node:assert/strict'
import { AIR_FRAG, GLOBE_VERT, LOOK, PLANET_FRAG, applyLook, makeAirMaterial, makePlanetMaterial } from '../src/globe/planetMaterial.js'
import { planetLook } from '../src/globe/globeMath.js'
import { balanced, declaredUniforms } from './glsl.mjs'

test('the shaders are well formed and end in the renderer chunks', () => {
  for (const src of [GLOBE_VERT, PLANET_FRAG, AIR_FRAG]) assert.ok(balanced(src))
  for (const src of [PLANET_FRAG, AIR_FRAG]) {
    assert.match(src, /#include <tonemapping_fragment>/)
    assert.match(src, /#include <colorspace_fragment>/)
  }
  assert.match(PLANET_FRAG, /#ifdef GLOBE_HQ/)
  // Words GLSL ES 3.0 reserves, which a refactor could slip in as variable names.
  for (const word of ['half', 'fixed', 'sample', 'filter', 'input', 'output', 'common', 'active']) {
    assert.doesNotMatch(PLANET_FRAG + AIR_FRAG, new RegExp(`\\b(float|vec[234]|int)\\s+${word}\\b`))
  }
})

test('every declared uniform is provided', () => {
  const cases = [
    [makePlanetMaterial(true), [GLOBE_VERT, PLANET_FRAG]],
    [makeAirMaterial(), [GLOBE_VERT, AIR_FRAG]],
  ]
  for (const [m, srcs] of cases) for (const name of srcs.flatMap(declaredUniforms)) assert.ok(name in m.uniforms, `${name} missing`)
})

test('Graphics Low drops GLOBE_HQ; the air is an additive back-faced shell', () => {
  assert.ok('GLOBE_HQ' in makePlanetMaterial(true).defines)
  assert.ok(!('GLOBE_HQ' in makePlanetMaterial(false).defines))
  const air = makeAirMaterial()
  assert.equal(air.depthWrite, false)
  assert.equal(air.transparent, true)
})

test('applyLook puts the world on both materials', () => {
  const look = planetLook({ name: 'Redshift-b', gravity_g: 2, seed: 42 })
  const p = makePlanetMaterial(false)
  const a = makeAirMaterial()
  applyLook(p, a, look)
  assert.ok(Math.abs(p.uniforms.uSun.value.length() - 1) < 1e-9)
  assert.deepEqual(p.uniforms.uSun.value.toArray(), a.uniforms.uSun.value.toArray())
  assert.equal(p.uniforms.uMaria.value, look.surface.maria)
  assert.deepEqual(p.uniforms.uOffset.value.toArray(), look.surface.offset)
  assert.equal(p.uniforms.uSunI.value, LOOK.sunI)
})
