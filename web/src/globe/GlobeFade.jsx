import { useEffect, useRef } from 'react'
import { fading, viewMode } from './viewMode.js'
import '../styles/globe.css'

// The short fade to black between the two views. Over the 3D view, under the HUD. Set
// straight from the view store (which changes inside the render loop), not through a React
// render, so the overlay goes up in the very frame the fade starts, however busy the page is.
export function GlobeFade() {
  const ref = useRef(null)
  useEffect(() => {
    const apply = () => ref.current?.toggleAttribute('data-on', fading(viewMode.get()))
    apply()
    return viewMode.subscribe(apply)
  }, [])
  return <div ref={ref} className="globe-fade" aria-hidden="true" />
}
