// The sector camera's side of the switch to the planet (see CameraRig.jsx). Plain arrays
// in three.js axes (x east, y up, z south), so Node can test it.

export const RISE_RATE = 4.2 // per second: how fast the camera eases up to the top-down pose
export const RETURN_OUT = 1.8 // back from the planet it starts this much further out, then glides in

// Straight above the middle of the map, high enough that the whole sector fills `fill` of
// the view height: the moment before the planet takes over. A hair south of vertical, so
// the orbit controls keep a heading and north stays up on screen.
export function topDownPose(map, fovDeg, fill = 0.62) {
  const cx = map.ox + map.width / 2
  const cy = map.oy + map.depth / 2
  const h = (map.minH + map.maxH) / 2
  const reach = Math.max(map.width, map.depth, 1) / 2
  const d = reach / (fill * Math.tan((fovDeg * Math.PI) / 360))
  return { target: [cx, h, -cy], position: [cx, h + d, -cy + d * 0.002] }
}

// Orbit controls keep a damped remainder of the last drag. With damping off for one update
// it is applied at once and cleared, so it cannot resume after a scripted move.
export function settleControls(c) {
  if (!c) return
  const damping = c.enableDamping
  c.enableDamping = false
  c.update()
  c.enableDamping = damping
}
