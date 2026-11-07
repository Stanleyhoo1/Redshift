// Sector (world metres) ↔ globe (unit sphere) mapping.
//
// The mission sector is drawn as a 90°×90° lat/lon patch centred on the camera-facing point
// (+Z). World x (east) → longitude, world y (north) → latitude:
//   lon = (x / L) · 90°,  lat = (y / L) · 90°,  x, y ∈ [−L/2, L/2]  ⇒  patch spans ±45°
//   radius = 1 + elevNorm · ELEV_EXAGGERATION
//   p = r · (cos lat · sin lon,  sin lat,  cos lat · cos lon)
// The same formula is implemented in GLSL in globe/sectorShader.ts — keep them identical.

export type Vec3 = [number, number, number]

export const PATCH_SPAN_DEG = 90
export const PATCH_SPAN_RAD = (PATCH_SPAN_DEG * Math.PI) / 180
export const ELEV_EXAGGERATION = 0.04

export function sectorToLatLon(x: number, y: number, L: number): { lat: number; lon: number } {
  return { lon: (x / L) * PATCH_SPAN_DEG, lat: (y / L) * PATCH_SPAN_DEG }
}

export function sectorToSphere(x: number, y: number, elevNorm: number, L: number): Vec3 {
  const lon = (x / L) * PATCH_SPAN_RAD
  const lat = (y / L) * PATCH_SPAN_RAD
  const r = 1 + elevNorm * ELEV_EXAGGERATION
  const cl = Math.cos(lat)
  return [r * cl * Math.sin(lon), r * Math.sin(lat), r * cl * Math.cos(lon)]
}

/** Same as sectorToSphere but with an explicit radius (for markers hovering above the surface). */
export function sectorToSphereR(x: number, y: number, radius: number, L: number): Vec3 {
  const lon = (x / L) * PATCH_SPAN_RAD
  const lat = (y / L) * PATCH_SPAN_RAD
  const cl = Math.cos(lat)
  return [radius * cl * Math.sin(lon), radius * Math.sin(lat), radius * cl * Math.cos(lon)]
}

export function sphereToSector(p: readonly number[], L: number): { x: number; y: number; elevNorm: number } {
  const r = Math.hypot(p[0], p[1], p[2])
  const lat = Math.asin(p[1] / r)
  const lon = Math.atan2(p[0], p[2])
  return {
    x: (lon / PATCH_SPAN_RAD) * L,
    y: (lat / PATCH_SPAN_RAD) * L,
    elevNorm: (r - 1) / ELEV_EXAGGERATION,
  }
}

/** Texture coordinates of a world point. v = 0 is the southmost row (PNG row 0, protocol §5). */
export function sectorToUv(x: number, y: number, L: number): [number, number] {
  return [x / L + 0.5, y / L + 0.5]
}

/** East/north/up unit vectors of the surface at world (x, y). */
export function localFrame(x: number, y: number, L: number): { east: Vec3; north: Vec3; up: Vec3 } {
  const lon = (x / L) * PATCH_SPAN_RAD
  const lat = (y / L) * PATCH_SPAN_RAD
  const sl = Math.sin(lon)
  const clo = Math.cos(lon)
  const sa = Math.sin(lat)
  const ca = Math.cos(lat)
  return {
    east: [clo, 0, -sl],
    north: [-sa * sl, ca, -sa * clo],
    up: [ca * sl, sa, ca * clo],
  }
}

/** World heading (yaw, rad, CCW from east) → tangent direction on the sphere. */
export function headingVector(x: number, y: number, yaw: number, L: number): Vec3 {
  const { east, north } = localFrame(x, y, L)
  const c = Math.cos(yaw)
  const s = Math.sin(yaw)
  return [c * east[0] + s * north[0], c * east[1] + s * north[1], c * east[2] + s * north[2]]
}

/** Metres on the ground → world units on the globe (arc length at radius 1). */
export function metresToUnits(m: number, L: number): number {
  return (m / L) * PATCH_SPAN_RAD
}

export function clamp(v: number, lo: number, hi: number): number {
  return v < lo ? lo : v > hi ? hi : v
}

export function elevToNorm(z: number, elevMin: number, elevMax: number): number {
  const span = elevMax - elevMin
  if (!(span > 1e-6)) return 0
  return clamp((z - elevMin) / span, 0, 1)
}
