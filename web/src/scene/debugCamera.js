import * as THREE from 'three'
import { groundAt } from './ground.js'

// Test aid (?debug only, see Scene.jsx): place the camera for screenshots.
//   view(x, y, az, el, dist): look at world (x, y) from azimuth az and elevation el
//     (degrees; az 0 = camera south of the point looking north), dist metres away. Hands
//     the camera over to "user" mode, like an orbit drag would.
//   orbit(az, el, dist): same angles around the current orbit target (keeps follow mode).
const _t = new THREE.Vector3()

function offset(az, el, dist, out) {
  const a = THREE.MathUtils.degToRad(az)
  const e = THREE.MathUtils.degToRad(el)
  return out.set(Math.sin(a) * Math.cos(e), Math.sin(e), Math.cos(a) * Math.cos(e)).multiplyScalar(dist)
}

export function cameraDebug(camera, getControls) {
  return {
    view(x, y, az = 0, el = 45, dist = 30) {
      const c = getControls()
      if (!c) return false
      c.dispatchEvent({ type: 'start' })
      c.target.set(x, groundAt(x, y), -y)
      camera.position.copy(c.target).add(offset(az, el, dist, _t))
      camera.lookAt(c.target)
      c.update?.()
      c.dispatchEvent({ type: 'end' })
      return true
    },
    orbit(az = 0, el = 45, dist = 20) {
      const c = getControls()
      if (!c) return false
      // Like a user drag: ends the follow zoom, keeps follow mode.
      c.dispatchEvent({ type: 'start' })
      camera.position.copy(c.target).add(offset(az, el, dist, _t))
      camera.lookAt(c.target)
      c.update?.()
      c.dispatchEvent({ type: 'end' })
      return true
    },
  }
}
