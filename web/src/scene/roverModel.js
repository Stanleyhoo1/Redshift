import * as THREE from 'three'
import { RoundedBoxGeometry } from 'three/addons/geometries/RoundedBoxGeometry.js'
import { boxUV, grime, merge, place, strut } from './hardware.js'

// A procedural model of the sim's rover (sim/redshift/world.py: chassis 0.8 x 0.5 x 0.2 m,
// four 0.18 m wheels on a wide track and a short wheelbase, skid steer, the front pair on
// an oscillating axle, a mast with the camera head and the LiDAR at 0.75 m). True metres,
// x forward, y up, z to the right; the scene scales it up a little for legibility.
// Static parts are merged per material; wheels and the LiDAR head are separate so they
// can turn. Built once and shared by every rover.

export const WHEEL_R = 0.18
export const WHEEL_W = 0.115
export const WHEELS = [
  [0.22, WHEEL_R, 0.4],
  [-0.22, WHEEL_R, 0.4],
  [0.22, WHEEL_R, -0.4],
  [-0.22, WHEEL_R, -0.4],
]
export const MAST_TOP = 1.1 // y of the beacon light

const V = (x, y, z) => new THREE.Vector3(x, y, z)

// Tyre with grousers: an extruded toothed ring round the z axis.
function tyreGeometry() {
  const teeth = 20
  const shape = new THREE.Shape()
  const ro = WHEEL_R
  const rg = WHEEL_R - 0.014
  for (let k = 0; k < teeth; k++) {
    const a0 = (k / teeth) * Math.PI * 2
    const a1 = a0 + (0.42 / teeth) * Math.PI * 2
    const a2 = a0 + (0.5 / teeth) * Math.PI * 2
    const a3 = a0 + (0.92 / teeth) * Math.PI * 2
    const pts = [
      [ro, a0],
      [ro, a1],
      [rg, a2],
      [rg, a3],
    ]
    for (const [r, a] of pts) {
      if (k === 0 && r === ro && a === a0) shape.moveTo(Math.cos(a) * r, Math.sin(a) * r)
      else shape.lineTo(Math.cos(a) * r, Math.sin(a) * r)
    }
  }
  shape.closePath()
  const hole = new THREE.Path()
  hole.absarc(0, 0, 0.118, 0, Math.PI * 2, true)
  shape.holes.push(hole)
  const g = new THREE.ExtrudeGeometry(shape, { depth: WHEEL_W, bevelEnabled: true, bevelThickness: 0.006, bevelSize: 0.005, bevelSegments: 1, curveSegments: 6 })
  g.translate(0, 0, -WHEEL_W / 2)
  g.computeVertexNormals()
  return g
}

// Hub: a dished wheel with spokes and a cap (axis z).
function hubGeometry() {
  const parts = []
  parts.push(new THREE.CylinderGeometry(0.12, 0.12, WHEEL_W * 0.9, 20).rotateX(Math.PI / 2))
  for (let k = 0; k < 5; k++) {
    const a = (k / 5) * Math.PI * 2
    const g = new THREE.BoxGeometry(0.022, 0.1, 0.02)
    place(g, Math.cos(a) * 0.06, Math.sin(a) * 0.06, WHEEL_W * 0.47, 0, 0, a - Math.PI / 2)
    parts.push(g)
  }
  parts.push(new THREE.CylinderGeometry(0.035, 0.035, WHEEL_W * 1.08, 12).rotateX(Math.PI / 2))
  return merge(parts)
}

let parts = null

export function roverParts() {
  if (parts) return parts
  const paint = []
  const gold = []
  const solar = []
  const dark = []
  const metal = []
  const glass = []
  const lamp = []
  const accent = []

  // Chassis: rounded box with panel texture, dirtier towards the bottom.
  paint.push(place(boxUV(new RoundedBoxGeometry(0.8, 0.2, 0.5, 2, 0.028)), 0, 0.2, 0))
  // Camera head and a small high-gain dish.
  paint.push(place(boxUV(new RoundedBoxGeometry(0.09, 0.075, 0.2, 1, 0.012)), 0.205, 0.95, 0))
  const dish = new THREE.LatheGeometry(
    Array.from({ length: 7 }, (_, k) => new THREE.Vector2((k / 6) * 0.085, ((k / 6) * (k / 6)) * 0.03)),
    18,
  )
  paint.push(place(boxUV(dish), -0.3, 0.47, 0.17, -0.5, 0, 0.35))
  paint.push(place(boxUV(new THREE.BoxGeometry(0.12, 0.05, 0.1)), 0.05, 0.415, -0.18))
  for (const g of paint) grime(g, 0.1, 0.32)

  // Warm electronics box under the deck: gold thermal blanket.
  gold.push(place(new THREE.BoxGeometry(0.52, 0.07, 0.42), -0.06, 0.335, 0))
  // Solar deck.
  solar.push(place(boxUV(new THREE.BoxGeometry(0.64, 0.016, 0.52)), -0.07, 0.378, 0))

  // Dark anodised parts: mast, joints, bumper, axle, wheel mounts, antenna, deck frame.
  dark.push(strut(V(0.18, 0.37, 0), V(0.18, 0.9, 0), 0.022, 0.018))
  dark.push(place(new THREE.BoxGeometry(0.05, 0.04, 0.05), 0.18, 0.905, 0))
  dark.push(place(new THREE.BoxGeometry(0.05, 0.06, 0.44), 0.415, 0.165, 0))
  dark.push(strut(V(0.22, WHEEL_R, -0.33), V(0.22, WHEEL_R, 0.33), 0.024))
  for (const s of [1, -1]) {
    dark.push(place(new THREE.BoxGeometry(0.07, 0.05, 0.11), -0.22, WHEEL_R, s * 0.29))
    dark.push(place(new THREE.BoxGeometry(0.66, 0.012, 0.012), -0.07, 0.386, s * 0.26))
  }
  dark.push(strut(V(-0.34, 0.38, -0.19), V(-0.34, 0.86, -0.19), 0.005, 0.004, 5))
  dark.push(strut(V(-0.3, 0.38, 0.17), V(-0.3, 0.45, 0.17), 0.012, 0.012, 6))
  dark.push(place(new THREE.BoxGeometry(0.03, 0.06, 0.05), 0.15, 0.95, 0))

  // Machined metal: motor housings at each wheel, antenna tip, LiDAR base, lens rims.
  for (const [x, , z] of WHEELS) metal.push(place(new THREE.CylinderGeometry(0.048, 0.048, 0.06, 14).rotateX(Math.PI / 2), x, WHEEL_R, Math.sign(z) * 0.325))
  metal.push(place(new THREE.SphereGeometry(0.011, 8, 6), -0.34, 0.865, -0.19))
  metal.push(place(new THREE.CylinderGeometry(0.044, 0.046, 0.018, 20), 0.205, 0.997, 0))
  metal.push(place(new THREE.CylinderGeometry(0.044, 0.044, 0.012, 20), 0.205, 1.056, 0))
  for (const z of [0.052, -0.052]) metal.push(place(new THREE.CylinderGeometry(0.027, 0.027, 0.012, 16).rotateZ(Math.PI / 2), 0.252, 0.955, z))

  // Optics: stereo cameras, LiDAR window, hazard cameras.
  for (const z of [0.052, -0.052]) glass.push(place(new THREE.CylinderGeometry(0.02, 0.02, 0.016, 16).rotateZ(Math.PI / 2), 0.257, 0.955, z))
  glass.push(place(new THREE.CylinderGeometry(0.04, 0.04, 0.042, 24), 0.205, 1.027, 0))
  for (const z of [0.08, -0.08]) glass.push(place(new THREE.CylinderGeometry(0.016, 0.016, 0.01, 12).rotateZ(Math.PI / 2), 0.442, 0.165, z))

  // Headlights.
  for (const z of [0.15, -0.15]) lamp.push(place(new THREE.BoxGeometry(0.008, 0.026, 0.07), 0.4015, 0.245, z))

  // Identity colour: side stripes, a band round the mast and the nose.
  for (const s of [1, -1]) accent.push(place(new THREE.BoxGeometry(0.6, 0.032, 0.004), -0.02, 0.24, s * 0.2515))
  accent.push(place(new THREE.BoxGeometry(0.004, 0.022, 0.3), 0.4015, 0.282, 0))
  accent.push(place(new THREE.CylinderGeometry(0.026, 0.026, 0.05, 12), 0.18, 0.78, 0))

  // The LiDAR's spinning emitter mark (turns with the head).
  const laser = new THREE.BoxGeometry(0.004, 0.03, 0.014)
  laser.translate(0.0405, 0, 0)

  parts = {
    paint: merge(paint),
    gold: merge(gold),
    solar: merge(solar),
    dark: merge(dark),
    metal: merge(metal),
    glass: merge(glass),
    lamp: merge(lamp),
    accent: merge(accent),
    laser,
    tyre: tyreGeometry(),
    hub: hubGeometry(),
    lidarAt: [0.205, 1.027, 0],
  }
  return parts
}
