import { Html as DreiHtml } from '@react-three/drei'
import { planeLabelsShown, useViewMode } from './viewMode.js'

// drei <Html> labels live in the page, not in the 3D scene, so not drawing the sector does
// not hide them. Every sector label goes through this drop-in: it renders nothing while
// the planet is up or the camera is rising towards it. (React 19 passes `ref` in props.)
export function Html(props) {
  const shown = useViewMode(planeLabelsShown)
  return shown ? <DreiHtml {...props} /> : null
}
