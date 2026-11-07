import { useEffect, useState } from 'react'
import { api } from '../../lib/api'
import { localPhysicsPreview } from '../../lib/defaults'
import type { PhysicsPreview } from '../../lib/types'

/** Debounced GET /api/physics_preview; falls back to the same model computed locally. */
export function usePhysicsPreview(g: number, delayMs = 160): { preview: PhysicsPreview; live: boolean } {
  const [state, setState] = useState<{ preview: PhysicsPreview; live: boolean }>(() => ({
    preview: localPhysicsPreview(g),
    live: false,
  }))

  useEffect(() => {
    const ctrl = new AbortController()
    const timer = setTimeout(() => {
      api
        .physicsPreview(g, ctrl.signal)
        .then((p) => setState({ preview: p, live: true }))
        .catch((e) => {
          if ((e as Error).name !== 'AbortError') setState({ preview: localPhysicsPreview(g), live: false })
        })
    }, delayMs)
    return () => {
      clearTimeout(timer)
      ctrl.abort()
    }
  }, [g, delayMs])

  return state
}
