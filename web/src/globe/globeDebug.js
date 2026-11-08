import * as THREE from 'three'
import { turnY } from './globeMath.js'
import { viewMode } from './viewMode.js'

// Test aid (?debug only; Scene.jsx merges it into window.__redshift). The planet view's
// objects register in globeRefs when they mount.
export const globeRefs = { camera: null, gl: null, spin: null, site: null, rig: null, info: null, planet: null, sectorMap: null }

const _p = new THREE.Vector3()
const _n = new THREE.Vector3()

export function globeDebug() {
  return {
    // setView('globe' | 'plane'): animated, or at once with `instant` (or reduced motion).
    setView(v, instant) {
      viewMode.request(v, instant ? { instant: true } : undefined)
      return viewMode.get().view
    },
    // Where the landing marker is on screen, and whether it faces the camera.
    globeSiteScreen() {
      const { camera, gl, site } = globeRefs
      if (!camera || !gl || !site) return null
      site.getWorldPosition(_p)
      _n.copy(camera.position).sub(_p).normalize()
      const facing = _p.clone().normalize().dot(_n)
      _p.project(camera)
      const r = gl.domElement.getBoundingClientRect()
      return { x: r.left + ((_p.x + 1) / 2) * r.width, y: r.top + ((1 - _p.y) / 2) * r.height, facing, visible: _p.z < 1 && facing > 0.2 }
    },
    // The planet camera and what the planet shows.
    globeState() {
      const { camera, spin, rig, info, planet, sectorMap } = globeRefs
      const v = viewMode.get()
      const d = camera ? camera.position.length() : 0
      const dir = camera && d ? camera.position.toArray().map((x) => x / d) : null
      const turn = spin ? spin.rotation.y : 0
      // The cosine of the sun's angle from the point under the camera, on the turned planet:
      // above 0 the middle of the disc is in daylight (about 0.29 at home).
      const under = dir ? turnY(dir, -turn) : null
      return {
        view: v.view,
        render: v.render,
        phase: v.phase,
        name: info?.name ?? null,
        radiusKm: info?.radiusKm ?? null,
        site: info?.site ?? null,
        dist: d,
        dir,
        spin: turn,
        lit: under && info ? under[0] * info.sun[0] + under[1] * info.sun[1] + under[2] * info.sun[2] : null,
        gliding: !!rig?.glide,
        hq: !!planet && 'GLOBE_HQ' in planet.defines,
        reveal: planet ? planet.uniforms.uReveal.value : null, // 0 unsurveyed, 1 the true planet
        mapPixels: sectorMap ? sectorMap.opaque : null, // mapped cells on the planet's sector map
      }
    },
  }
}
