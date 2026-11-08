import { mulberry } from '../scene/proceduralTextures.js'

// The planet view's sky as plain arrays: the same stars on every visit and every world
// (never Math.random, so screenshots are repeatable). About 40 % lie near a band, like a
// galaxy seen edge on; brightness falls off steeply, so most stars are faint.

export const STAR_RADIUS = 300
const k = Math.hypot(0.32, 0.86, -0.4)
export const BAND = [0.32 / k, 0.86 / k, -0.4 / k] // the band's pole, tilted against the planet's axis

export function starData(count) {
  const rand = mulberry(0x5eed1)
  const position = new Float32Array(count * 3)
  const mag = new Float32Array(count)
  const kind = new Uint8Array(count) // 0 warm, 1 white, 2 cool
  const phase = new Float32Array(count)
  for (let i = 0; i < count; i++) {
    const z = rand() * 2 - 1
    const a = rand() * Math.PI * 2
    const s = Math.sqrt(1 - z * z)
    let d = [s * Math.cos(a), z, s * Math.sin(a)]
    if (rand() < 0.4) {
      // Pulled onto the band: keep only a small, roughly normal height above its plane.
      const h = d[0] * BAND[0] + d[1] * BAND[1] + d[2] * BAND[2]
      const keep = 0.12 * (rand() + rand() + rand() - 1.5)
      const q = [d[0] - BAND[0] * (h - keep), d[1] - BAND[1] * (h - keep), d[2] - BAND[2] * (h - keep)]
      const l = Math.hypot(q[0], q[1], q[2])
      if (l > 1e-6) d = [q[0] / l, q[1] / l, q[2] / l]
    }
    position[i * 3] = d[0] * STAR_RADIUS
    position[i * 3 + 1] = d[1] * STAR_RADIUS
    position[i * 3 + 2] = d[2] * STAR_RADIUS
    mag[i] = Math.pow(rand(), 5)
    const t = rand()
    kind[i] = t < 0.2 ? 0 : t < 0.82 ? 1 : 2
    phase[i] = rand() * Math.PI * 2
  }
  return { position, mag, kind, phase }
}
