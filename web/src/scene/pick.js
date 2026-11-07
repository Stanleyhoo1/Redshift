import * as THREE from 'three'

// Bilinear height over the known corners (like sampleGround, without allocating).
function knownHeight(T, x, y, fallback) {
  const fx = (x - T.ox) / T.res
  const fy = (y - T.oy) / T.res
  if (fx < 0 || fy < 0 || fx > T.nx - 1 || fy > T.ny - 1) return fallback
  const i0 = Math.min(T.nx - 2, Math.floor(fx))
  const j0 = Math.min(T.ny - 2, Math.floor(fy))
  const tx = fx - i0
  const ty = fy - j0
  const k = j0 * T.nx + i0
  const K = T.known
  const H = T.h
  let sum = 0
  let w = 0
  let q = (1 - tx) * (1 - ty)
  if (K[k]) (sum += H[k] * q), (w += q)
  q = tx * (1 - ty)
  if (K[k + 1]) (sum += H[k + 1] * q), (w += q)
  q = (1 - tx) * ty
  if (K[k + T.nx]) (sum += H[k + T.nx] * q), (w += q)
  q = tx * ty
  if (K[k + T.nx + 1]) (sum += H[k + T.nx + 1] * q), (w += q)
  return w < 1e-6 ? fallback : sum / w
}

const _hit = { x: 0, y: 0, h: 0, onMap: true }

// Where a ray (three.js coordinates) meets the terrain as drawn: the known heightfield,
// or the dark floor (fogY) where the fleet knows nothing. Marches the heightfield in
// half-sample steps inside the map's bounding box, then refines by bisection. Outside
// the map it lands on the floor plane (onMap false). Returns a shared object or null.
// Cost: a few hundred height lookups per call, so it is fine on every pointer move.
export function pickGround(ray, T, fogY) {
  if (!T) return null
  const o = ray.origin
  const d = ray.direction
  const heightAt = (x, z) => knownHeight(T, x, -z, fogY)
  // Slab test against the map box.
  const box = [
    [T.ox, T.ox + T.width],
    [fogY - 0.1, T.maxH + 1],
    [-(T.oy + T.depth), -T.oy],
  ]
  let t0 = 0
  let t1 = 1e5
  const oa = [o.x, o.y, o.z]
  const da = [d.x, d.y, d.z]
  for (let a = 0; a < 3; a++) {
    if (Math.abs(da[a]) < 1e-9) {
      if (oa[a] < box[a][0] || oa[a] > box[a][1]) {
        t1 = -1
        break
      }
      continue
    }
    let ta = (box[a][0] - oa[a]) / da[a]
    let tb = (box[a][1] - oa[a]) / da[a]
    if (ta > tb) [ta, tb] = [tb, ta]
    t0 = Math.max(t0, ta)
    t1 = Math.min(t1, tb)
  }
  if (t1 >= t0) {
    const step = T.res * 0.5
    let prevT = t0
    let prevAbove = o.y + d.y * t0 - heightAt(o.x + d.x * t0, o.z + d.z * t0)
    if (prevAbove <= 0) return finish(o, d, t0, true)
    for (let t = t0 + step; t <= t1 + 1e-6; t += step) {
      const above = o.y + d.y * t - heightAt(o.x + d.x * t, o.z + d.z * t)
      if (above <= 0) {
        let lo = prevT
        let hi = t
        for (let it = 0; it < 8; it++) {
          const mid = (lo + hi) / 2
          if (o.y + d.y * mid - heightAt(o.x + d.x * mid, o.z + d.z * mid) > 0) lo = mid
          else hi = mid
        }
        return finish(o, d, hi, true)
      }
      prevT = t
      prevAbove = above
    }
  }
  // Missed the map: the floor plane around it.
  if (d.y >= -1e-6) return null
  const t = (fogY - o.y) / d.y
  if (t <= 0) return null
  return finish(o, d, t, false)
}

function finish(o, d, t, onMap) {
  _hit.x = o.x + d.x * t
  _hit.h = o.y + d.y * t
  _hit.y = -(o.z + d.z * t)
  _hit.onMap = onMap
  return _hit
}

export const _ndc = new THREE.Vector2()
