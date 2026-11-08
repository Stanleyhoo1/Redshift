import test from 'node:test'
import assert from 'node:assert/strict'
import { BIOMES, BIOME_IDS, biomeMapOf, biomeOf, biomeWeightAt, groundSwatch, mixHex, mixPalette, worldPalette } from '../src/lib/biomes.js'
import { ATMOSPHERE, REGOLITH } from '../src/lib/theme.js'
import { MAP_RAMP, paintSectorMap, sectorLook } from '../src/globe/sectorMap.js'

const EM_DASH = String.fromCharCode(0x2014)

test('every biome of the sim has a full palette', () => {
  assert.deepEqual(BIOME_IDS.slice().sort(), ['ash', 'basalt', 'frost', 'obsidian', 'rust', 'salt', 'verdigris'])
  const keys = Object.keys(BIOMES.rust.ground)
  const airKeys = Object.keys(BIOMES.rust.air)
  for (const id of BIOME_IDS) {
    const b = BIOMES[id]
    assert.deepEqual(Object.keys(b.ground), keys, id)
    assert.deepEqual(Object.keys(b.air), airKeys, id)
    assert.equal(b.map.length, 3, id)
    for (const hex of [...Object.values(b.ground), ...Object.values(b.air), ...b.map]) assert.match(hex, /^#[0-9a-f]{6}$/, id)
    assert.ok(!b.name.includes(EM_DASH))
  }
})

test('rust is exactly today\'s look', () => {
  assert.equal(BIOMES.rust.ground.dust, REGOLITH.dust)
  assert.equal(BIOMES.rust.ground.strataB, REGOLITH.strataB)
  assert.equal(BIOMES.rust.air.haze, ATMOSPHERE.haze)
  const pal = worldPalette({ name: 'Redshift-b', seed: 42 })
  assert.equal(pal.biome, null)
  assert.equal(pal.ground, BIOMES.rust.ground)
  assert.equal(pal.ground2, BIOMES.rust.ground)
  assert.equal(groundSwatch({}), 'linear-gradient(135deg, #b19885 0%, #8f7766 45%, #4b4440 100%)')
})

test('biomeOf checks the init world', () => {
  assert.equal(biomeOf(null), null)
  assert.equal(biomeOf({ biome: { id: 'lava' } }), null)
  assert.deepEqual(biomeOf({ biome: { id: 'frost', name: 'Frost plains', secondary: 'rust' } }), { id: 'frost', name: 'Frost plains', secondary: 'rust' })
  assert.deepEqual(biomeOf({ biome: { id: 'salt', secondary: 'nope' } }), { id: 'salt', name: 'Salt flats', secondary: null })
  const pal = worldPalette({ biome: { id: 'basalt', name: 'Basalt fields', secondary: 'verdigris' } })
  assert.equal(pal.ground, BIOMES.basalt.ground)
  assert.equal(pal.ground2, BIOMES.verdigris.ground)
  assert.equal(pal.air, BIOMES.basalt.air)
})

test('mixHex and mixPalette blend in sRGB', () => {
  assert.equal(mixHex('#000000', '#ffffff', 0), '#000000')
  assert.equal(mixHex('#000000', '#ffffff', 1), '#ffffff')
  assert.equal(mixHex('#000000', '#ffffff', 0.5), '#808080')
  assert.equal(mixHex('#102030', '#302010', 2), '#302010') // clamped
  assert.deepEqual(mixPalette(['#000000', '#ff0000'], ['#ffffff', '#00ff00'], 0.5), ['#808080', '#808000'])
  assert.deepEqual(mixPalette({ a: '#000000' }, { a: '#202020' }, 0.5), { a: '#101010' })
})

const MAP = { nx: 3, ny: 2, origin: [-4, -2], res: 4, w: [0, 1, 0, 0.5, 0.5, 1] }

test('biome_map: checked, then sampled bilinearly', () => {
  assert.equal(biomeMapOf(null), null)
  assert.equal(biomeMapOf({ biome_map: { ...MAP, w: [0, 1] } }), null)
  assert.equal(biomeMapOf({ biome_map: { ...MAP, res: 0 } }), null)
  const m = biomeMapOf({ biome_map: MAP })
  assert.equal(m.nx, 3)
  assert.equal(m.ox, -4)
  assert.equal(biomeWeightAt(m, -4, -2), 0)
  assert.equal(biomeWeightAt(m, 0, -2), 1)
  assert.equal(biomeWeightAt(m, -2, -2), 0.5)
  assert.equal(biomeWeightAt(m, -4, 0), 0.25)
  assert.equal(biomeWeightAt(m, 4, 2), 1)
  assert.equal(biomeWeightAt(m, 5, 2), 0) // outside
  assert.equal(biomeWeightAt(null, 0, 0), 0)
})

test('the sector map paints the biome ramps, and today\'s ramp without one', () => {
  const T = { nx: 2, ny: 1, res: 4, ox: -4, oy: -2, h: new Float32Array([0, 0]), slope: new Float32Array([0, 0]), known: new Uint8Array([1, 1]), minH: 0, maxH: 1, knownCount: 2 }
  const plain = paintSectorMap(T, {})
  assert.deepEqual([...plain.slice(0, 3)], MAP_RAMP[0])
  assert.equal(sectorLook({ seed: 1 }, {}), null)
  const world = { biome: { id: 'frost', name: 'Frost plains', secondary: 'obsidian' } }
  const look = sectorLook(world, { biome_map: { nx: 2, ny: 2, origin: [-4, -2], res: 4, w: [0, 1, 0, 1] } })
  const px = paintSectorMap(T, {}, null, look)
  const rgb = (hex) => [1, 3, 5].map((k) => parseInt(hex.slice(k, k + 2), 16))
  assert.deepEqual([...px.slice(0, 3)], rgb(BIOMES.frost.map[0])) // primary at w 0
  assert.deepEqual([...px.slice(4, 7)], rgb(BIOMES.obsidian.map[0])) // secondary at w 1
})
