import { describe, expect, it } from 'vitest'
import {
  ELEV_EXAGGERATION,
  PATCH_SPAN_DEG,
  localFrame,
  sectorToLatLon,
  sectorToSphere,
  sectorToUv,
  sphereToSector,
} from './geo'

const L = 100
const len = (v: number[]) => Math.hypot(v[0], v[1], v[2])
const close = (a: readonly number[], b: readonly number[]) => a.forEach((v, i) => expect(v).toBeCloseTo(b[i], 9))

describe('sectorToSphere', () => {
  it('maps the sector centre to the camera-facing point (+Z) at radius 1', () => {
    const p = sectorToSphere(0, 0, 0, L)
    close(p, [0, 0, 1])
  })

  it('applies the elevation exaggeration to the radius', () => {
    expect(len(sectorToSphere(0, 0, 1, L))).toBeCloseTo(1 + ELEV_EXAGGERATION, 12)
    expect(len(sectorToSphere(12, -30, 0.5, L))).toBeCloseTo(1 + 0.5 * ELEV_EXAGGERATION, 12)
    expect(ELEV_EXAGGERATION).toBe(0.04)
  })

  it('spans ±45° of longitude and latitude at the sector edges', () => {
    expect(PATCH_SPAN_DEG).toBe(90)
    const east = sectorToSphere(L / 2, 0, 0, L)
    close(east, [Math.SQRT1_2, 0, Math.SQRT1_2])
    const north = sectorToSphere(0, L / 2, 0, L)
    close(north, [0, Math.SQRT1_2, Math.SQRT1_2])
    const west = sectorToSphere(-L / 2, 0, 0, L)
    close(west, [-Math.SQRT1_2, 0, Math.SQRT1_2])
    const south = sectorToSphere(0, -L / 2, 0, L)
    close(south, [0, -Math.SQRT1_2, Math.SQRT1_2])
  })

  it('places the corners at lat/lon ±45° with x east (+X) and y north (+Y)', () => {
    const ne = sectorToSphere(L / 2, L / 2, 0, L)
    const c = Math.cos(Math.PI / 4)
    close(ne, [c * c, c, c * c])
    const sw = sectorToSphere(-L / 2, -L / 2, 0, L)
    close(sw, [-c * c, -c, c * c])
    for (const [x, y] of [[L / 2, L / 2], [-L / 2, L / 2], [L / 2, -L / 2], [-L / 2, -L / 2]]) {
      const { lat, lon } = sectorToLatLon(x, y, L)
      expect(Math.abs(lat)).toBeCloseTo(45, 12)
      expect(Math.abs(lon)).toBeCloseTo(45, 12)
    }
  })

  it('is monotonic: +x moves east (X grows), +y moves north (Y grows)', () => {
    let prev = -Infinity
    for (let x = -L / 2; x <= L / 2; x += 5) {
      const X = sectorToSphere(x, 10, 0.3, L)[0]
      expect(X).toBeGreaterThan(prev)
      prev = X
    }
    prev = -Infinity
    for (let y = -L / 2; y <= L / 2; y += 5) {
      const Y = sectorToSphere(-20, y, 0.3, L)[1]
      expect(Y).toBeGreaterThan(prev)
      prev = Y
    }
  })

  it('every point in the sector faces the camera (Z > 0)', () => {
    for (let x = -L / 2; x <= L / 2; x += 10) for (let y = -L / 2; y <= L / 2; y += 10) {
      expect(sectorToSphere(x, y, 0, L)[2]).toBeGreaterThan(0.49)
    }
  })

  it('scales with sector size (same relative position → same point)', () => {
    close(sectorToSphere(25, -10, 0.2, 100), sectorToSphere(50, -20, 0.2, 200))
  })

  it('round-trips through sphereToSector', () => {
    for (const [x, y, e] of [[0, 0, 0], [12.5, -40, 0.7], [-49, 49, 1], [33, 7, 0.25]]) {
      const back = sphereToSector(sectorToSphere(x, y, e, L), L)
      expect(back.x).toBeCloseTo(x, 9)
      expect(back.y).toBeCloseTo(y, 9)
      expect(back.elevNorm).toBeCloseTo(e, 9)
    }
  })
})

describe('sectorToUv', () => {
  it('maps south-west corner to (0,0) and north-east to (1,1) (PNG row 0 = south)', () => {
    expect(sectorToUv(-L / 2, -L / 2, L)).toEqual([0, 0])
    expect(sectorToUv(L / 2, L / 2, L)).toEqual([1, 1])
    expect(sectorToUv(0, 0, L)).toEqual([0.5, 0.5])
  })
})

describe('localFrame', () => {
  it('returns an orthonormal east/north/up basis with up = radial', () => {
    for (const [x, y] of [[0, 0], [20, -35], [-45, 40]]) {
      const { east, north, up } = localFrame(x, y, L)
      expect(len(east)).toBeCloseTo(1, 12)
      expect(len(north)).toBeCloseTo(1, 12)
      expect(len(up)).toBeCloseTo(1, 12)
      const dot = (a: number[], b: number[]) => a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
      expect(dot(east, north)).toBeCloseTo(0, 12)
      expect(dot(east, up)).toBeCloseTo(0, 12)
      expect(dot(north, up)).toBeCloseTo(0, 12)
      close(up, sectorToSphere(x, y, 0, L))
    }
  })

  it('points east toward +x and north toward +y on the surface', () => {
    const { east, north } = localFrame(0, 0, L)
    close(east, [1, 0, 0])
    close(north, [0, 1, 0])
  })
})
