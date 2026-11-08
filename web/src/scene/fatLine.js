import * as THREE from 'three'
import { LineSegments2 } from 'three/addons/lines/LineSegments2.js'
import { LineSegmentsGeometry } from 'three/addons/lines/LineSegmentsGeometry.js'
import { LineMaterial } from 'three/addons/lines/LineMaterial.js'

// A screen-space-width polyline with preallocated buffers that are updated in place.
// (LineGeometry.setPositions allocates new GPU buffers on every call and leaks them,
// which matters when trails and paths change several times per second.)
export class FatPolyline {
  constructor(maxPoints, { color = '#ffffff', width = 2, opacity = 1, dashed = false, dashSize = 0.6, gapSize = 0.4, vertexColors = false } = {}) {
    this.max = Math.max(2, maxPoints)
    const segs = this.max - 1
    const geo = new LineSegmentsGeometry()

    this.pos = new Float32Array(segs * 6)
    this.posBuf = new THREE.InstancedInterleavedBuffer(this.pos, 6, 1).setUsage(THREE.DynamicDrawUsage)
    geo.setAttribute('instanceStart', new THREE.InterleavedBufferAttribute(this.posBuf, 3, 0))
    geo.setAttribute('instanceEnd', new THREE.InterleavedBufferAttribute(this.posBuf, 3, 3))

    if (vertexColors) {
      this.col = new Float32Array(segs * 6)
      this.colBuf = new THREE.InstancedInterleavedBuffer(this.col, 6, 1).setUsage(THREE.DynamicDrawUsage)
      geo.setAttribute('instanceColorStart', new THREE.InterleavedBufferAttribute(this.colBuf, 3, 0))
      geo.setAttribute('instanceColorEnd', new THREE.InterleavedBufferAttribute(this.colBuf, 3, 3))
    }
    if (dashed) {
      this.dist = new Float32Array(segs * 2)
      this.distBuf = new THREE.InstancedInterleavedBuffer(this.dist, 2, 1).setUsage(THREE.DynamicDrawUsage)
      geo.setAttribute('instanceDistanceStart', new THREE.InterleavedBufferAttribute(this.distBuf, 1, 0))
      geo.setAttribute('instanceDistanceEnd', new THREE.InterleavedBufferAttribute(this.distBuf, 1, 1))
    }
    geo.instanceCount = 0
    this.geometry = geo

    this.material = new LineMaterial({
      color,
      linewidth: width,
      transparent: opacity < 1,
      opacity,
      dashed,
      dashSize,
      gapSize,
      vertexColors,
      worldUnits: false,
    })
    // HUD layer: exact theme colours, no tone mapping or haze.
    this.material.toneMapped = false
    this.material.fog = false
    this.object = new LineSegments2(geo, this.material)
    this.object.frustumCulled = false
    this.object.visible = false
    this._c = [0, 0, 0]
  }

  // xyz: flat array of three.js coordinates, n points. colorAt(q, n, out) fills out[0..2].
  set(xyz, n, colorAt) {
    n = Math.min(n, this.max)
    let acc = 0
    const c0 = this._c
    for (let q = 0; q < n - 1; q++) {
      const a = q * 3
      const o = q * 6
      for (let k = 0; k < 6; k++) this.pos[o + k] = xyz[a + k]
      if (this.dist) {
        this.dist[q * 2] = acc
        acc += Math.hypot(xyz[a + 3] - xyz[a], xyz[a + 4] - xyz[a + 1], xyz[a + 5] - xyz[a + 2])
        this.dist[q * 2 + 1] = acc
      }
      if (this.col && colorAt) {
        colorAt(q, n, c0)
        this.col[o] = c0[0]
        this.col[o + 1] = c0[1]
        this.col[o + 2] = c0[2]
        colorAt(q + 1, n, c0)
        this.col[o + 3] = c0[0]
        this.col[o + 4] = c0[1]
        this.col[o + 5] = c0[2]
      }
    }
    this.geometry.instanceCount = Math.max(0, n - 1)
    this.posBuf.needsUpdate = true
    if (this.distBuf) this.distBuf.needsUpdate = true
    if (this.colBuf) this.colBuf.needsUpdate = true
    this.object.visible = n >= 2
  }

  setResolution(w, h) {
    this.material.resolution.set(w, h)
  }

  dispose() {
    this.geometry.dispose()
    this.material.dispose()
  }
}
