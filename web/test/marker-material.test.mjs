import test from 'node:test'
import assert from 'node:assert/strict'
import { MARKER_FRAG, MARKER_VERT, makeMarkerMaterial } from '../src/globe/markerMaterial.js'
import { balanced, declaredUniforms } from './glsl.mjs'

test('marker shaders are well formed, every uniform provided, HUD-like', () => {
  assert.ok(balanced(MARKER_VERT) && balanced(MARKER_FRAG))
  const m = makeMarkerMaterial()
  for (const name of [...declaredUniforms(MARKER_VERT), ...declaredUniforms(MARKER_FRAG)]) assert.ok(name in m.uniforms, `${name} missing`)
  assert.equal(m.toneMapped, false)
  assert.equal(m.depthWrite, false)
  assert.equal(m.uniforms.uColor.value.getHexString(), '4fe0ef')
})
