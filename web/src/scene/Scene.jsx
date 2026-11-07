import { Stars } from '@react-three/drei'
import { Terrain } from './Terrain.jsx'
import { Ground } from './Ground.jsx'
import { Lander } from './Lander.jsx'
import { Rovers } from './Rovers.jsx'
import { Targets } from './Targets.jsx'
import { Frontiers } from './Frontiers.jsx'
import { Reality } from './Reality.jsx'
import { CameraRig } from './CameraRig.jsx'
import { SUN_DIR, reducedMotion } from './sceneConstants.js'
import { VOID } from '../lib/theme.js'

export function Scene({ view, follow, overview }) {
  return (
    <>
      <color attach="background" args={[VOID]} />
      <ambientLight intensity={0.55} />
      <hemisphereLight args={['#c9dcff', '#3a2620', 0.35]} />
      <directionalLight position={SUN_DIR.map((v) => v * 100)} intensity={1.35} />
      <Stars radius={300} depth={90} count={2600} factor={4} saturation={0} fade speed={reducedMotion ? 0 : 0.35} />
      <Ground />
      <Terrain slopeColors={view.slope} />
      <Lander />
      {view.frontiers && <Frontiers />}
      <Targets />
      <Rovers showPaths={view.paths} />
      {view.reality && <Reality />}
      <CameraRig follow={follow} overview={overview} />
    </>
  )
}
