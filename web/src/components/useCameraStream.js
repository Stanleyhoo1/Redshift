import { useEffect, useRef, useState } from 'react'
import { camDown, fetchCameraFrame } from '../net/connection.js'

const RETRY_NET = 4000 // no connection: try again soon
export const RETRY_DOWN = 30000 // 503 / 404: rendering is off on that machine, rarely comes back

export function cameraProblem(err) {
  if (err.status === 404) return { title: 'Camera unavailable', sub: 'This rover has no camera feed' }
  if (err.status === 0) return { title: 'Camera unavailable', sub: 'No connection to the sim' }
  if (err.status === 503) return { title: 'Camera unavailable', sub: err.message && !/^HTTP/.test(err.message) ? err.message : 'Rendering is off on this machine' }
  return { title: 'Camera unavailable', sub: err.message || 'The sim did not send a picture' }
}

// Polls GET /api/camera/{id}.jpg while enabled. A new request starts only after the
// previous image has arrived and decoded (never stacked), at most one per `period` ms.
// Frames are decoded before they are swapped in, so the picture never flashes blank.
// Disabled: polling stops and the last frame stays. Returns
// { frame, at, problem, waiting, natural: { w, h } | null, retry }.
export function useCameraStream({ roverId, view, w, h, period, enabled = true }) {
  const [frame, setFrame] = useState(null)
  const [at, setAt] = useState(0) // performance.now() when the frame arrived
  const [natural, setNatural] = useState(null)
  const [problem, setProblem] = useState(() => (Date.now() < camDown.until ? camDown.problem : null))
  const [waiting, setWaiting] = useState(true)
  const [attempt, setAttempt] = useState(0)
  const current = useRef(null)

  useEffect(() => {
    if (!enabled || !roverId) return
    let stopped = false
    let timer = null
    let ctrl = null
    const loop = async () => {
      if (stopped) return
      const down = camDown.until - Date.now()
      if (down > 0) {
        setProblem(camDown.problem)
        setWaiting(false)
        timer = setTimeout(loop, down)
        return
      }
      const started = performance.now()
      ctrl = new AbortController()
      try {
        const blob = await fetchCameraFrame(roverId, view, w, h, ctrl.signal)
        const url = URL.createObjectURL(blob)
        const img = new Image()
        try {
          img.src = url
          await img.decode()
        } catch {
          URL.revokeObjectURL(url)
          throw Object.assign(new Error('The picture could not be decoded'), { status: -1 })
        }
        if (stopped) {
          URL.revokeObjectURL(url)
          return
        }
        const old = current.current
        current.current = url
        setFrame(url)
        setAt(performance.now())
        setNatural((n) => (n && n.w === img.naturalWidth && n.h === img.naturalHeight ? n : { w: img.naturalWidth, h: img.naturalHeight }))
        setProblem(null)
        setWaiting(false)
        if (old) setTimeout(() => URL.revokeObjectURL(old), 500)
        timer = setTimeout(loop, Math.max(0, period - (performance.now() - started)))
      } catch (err) {
        if (stopped || err?.name === 'AbortError') return
        const p = cameraProblem(err)
        setProblem(p)
        setWaiting(false)
        const down = err.status === 503 || err.status === 404
        if (down) {
          camDown.until = Date.now() + RETRY_DOWN
          camDown.problem = p
        }
        timer = setTimeout(loop, down ? RETRY_DOWN : RETRY_NET)
      }
    }
    loop()
    return () => {
      stopped = true
      clearTimeout(timer)
      ctrl?.abort()
    }
  }, [roverId, view, w, h, period, enabled, attempt])

  // Drop the last frame when the view goes away.
  useEffect(
    () => () => {
      if (current.current) URL.revokeObjectURL(current.current)
      current.current = null
    },
    [],
  )

  const retry = () => {
    camDown.until = 0
    setProblem(null)
    setWaiting(true)
    setAttempt((n) => n + 1)
  }
  return { frame, at, natural, problem, waiting, retry }
}
