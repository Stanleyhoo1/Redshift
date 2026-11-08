import test from 'node:test'
import assert from 'node:assert/strict'
import { STAR_FRAG, STAR_VERT, makeStarMaterial } from '../src/globe/starMaterial.js'
import { GLOBE } from '../src/globe/globePalette.js'
import { balanced, declaredUniforms } from './glsl.mjs'

test('star shaders are well formed and every uniform is provided', () => {
  assert.ok(balanced(STAR_VERT) && balanced(STAR_FRAG))
  assert.match(STAR_FRAG, /#include <colorspace_fragment>/)
  const m = makeStarMaterial()
  for (const name of [...declaredUniforms(STAR_VERT), ...declaredUniforms(STAR_FRAG)]) assert.ok(name in m.uniforms, `${name} missing`)
  assert.equal(m.toneMapped, false)
  assert.equal(m.depthWrite, false)
})

test('palette: hex colours, theme accent for the marker', () => {
  for (const [k, v] of Object.entries(GLOBE)) assert.match(v, /^#[0-9a-f]{6}$/i, k)
  assert.equal(GLOBE.signal, '#4fe0ef')
})
