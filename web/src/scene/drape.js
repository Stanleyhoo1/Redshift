import { useEffect, useRef } from 'react'
import { useFrame } from '@react-three/fiber'
import * as THREE from 'three'
import { simStore } from '../state/store.js'
import { groundAt } from './ground.js'

// A disc lying on the terrain: a polar grid whose heights follow the ground.
// uvMode 'world': UVs in world metres / 0.9 (a hatch tiles evenly at any size);
// uvMode 'disc': UVs span the disc (0..1), for a decal drawn over the whole disc.
// shape(a) optionally scales the radius by angle (an irregular patch).
export class DrapedDisc {
  constructor(rings = 7, segs = 56, { uvMode = 'world', shape = null, normals = false } = {}) {
    this.rings = rings
    this.segs = segs
    this.uvMode = uvMode
    this.shape = shape
    this.normals = normals
    const n = 1 + rings * segs
    this.pos = new Float32Array(n * 3)
    this.uv = new Float32Array(n * 2)
    const index = []
    for (let s = 0; s < segs; s++) index.push(0, 1 + s, 1 + ((s + 1) % segs))
    for (let r = 1; r < rings; r++) {
      for (let s = 0; s < segs; s++) {
        const a = 1 + (r - 1) * segs + s
        const b = 1 + (r - 1) * segs + ((s + 1) % segs)
        const c = 1 + r * segs + s
        const d = 1 + r * segs + ((s + 1) % segs)
        index.push(a, c, b, b, c, d)
      }
    }
    this.geometry = new THREE.BufferGeometry()
    this.geometry.setAttribute('position', new THREE.BufferAttribute(this.pos, 3).setUsage(THREE.DynamicDrawUsage))
    this.geometry.setAttribute('uv', new THREE.BufferAttribute(this.uv, 2))
    if (normals) {
      this.nor = new Float32Array(n * 3)
      for (let k = 0; k < n; k++) this.nor[k * 3 + 1] = 1
      this.geometry.setAttribute('normal', new THREE.BufferAttribute(this.nor, 3).setUsage(THREE.DynamicDrawUsage))
    }
    this.geometry.setIndex(index)
  }

  // Also fills ring[] (outer edge, closed) for an outline.
  update(cx, cy, radius, lift, ring) {
    const { rings, segs, pos, uv } = this
    const put = (k, x, y) => {
      pos[k * 3] = x
      pos[k * 3 + 1] = groundAt(x, y) + lift
      pos[k * 3 + 2] = -y
      if (this.uvMode === 'disc') {
        uv[k * 2] = 0.5 + (x - cx) / (2 * radius)
        uv[k * 2 + 1] = 0.5 + (y - cy) / (2 * radius)
      } else {
        uv[k * 2] = x / 0.9
        uv[k * 2 + 1] = y / 0.9
      }
    }
    put(0, cx, cy)
    for (let r = 1; r <= rings; r++) {
      for (let s = 0; s < segs; s++) {
        const a = (s / segs) * Math.PI * 2
        const rr = ((radius * r) / rings) * (this.shape ? this.shape(a) : 1)
        put(1 + (r - 1) * segs + s, cx + Math.cos(a) * rr, cy + Math.sin(a) * rr)
      }
    }
    if (ring) {
      const o = 1 + (rings - 1) * segs
      for (let s = 0; s <= segs; s++) {
        const k = o + (s % segs)
        ring[s * 3] = pos[k * 3]
        ring[s * 3 + 1] = pos[k * 3 + 1] + 0.03
        ring[s * 3 + 2] = pos[k * 3 + 2]
      }
    }
    this.geometry.attributes.position.needsUpdate = true
    this.geometry.attributes.uv.needsUpdate = true
    if (this.normals) {
      this.geometry.computeVertexNormals()
    }
    this.geometry.computeBoundingSphere()
  }

  dispose() {
    this.geometry.dispose()
  }
}

// Re-drape at most twice a second, and only when the known terrain changed.
export function useDrape(update, deps) {
  const st = useRef({ version: -1, at: 0 })
  useEffect(() => {
    st.current.version = -1
  }, deps) // eslint-disable-line react-hooks/exhaustive-deps
  useFrame(() => {
    const T = simStore.get().terrain
    const v = T ? T.version : 0
    const now = performance.now()
    if (v === st.current.version || now - st.current.at < 500) return
    st.current.version = v
    st.current.at = now
    update()
  })
}
