import { Stars } from '@react-three/drei'
import { Terrain } from './Terrain.jsx'
import { Ground } from './Ground.jsx'
import { Lander } from './Lander.jsx'
import { Rovers } from './Rovers.jsx'
import { Lidar } from './Lidar.jsx'
import { Targets } from './Targets.jsx'
import { Frontiers } from './Frontiers.jsx'
import { Reality } from './Reality.jsx'
import { HazardVisuals } from './HazardVisuals.jsx'
import { DustStorm } from './DustStorm.jsx'
import { FocusMarker } from './FocusMarker.jsx'
import { CameraRig } from './CameraRig.jsx'
import { SUN_DIR, reducedMotion } from './sceneConstants.js'
import { VOID } from '../lib/theme.js'

export function Scene({ view, follow, overview, selected, onSelect, focus }) {
  return (
    <>
      <color attach="background" args={[VOID]} />
      <ambientLight intensity={0.55} />
      <hemisphereLight args={['#c9dcff', '#3a2620', 0.35]} />
      <directionalLight position={SUN_DIR.map((v) => v * 100)} intensity={1.35} />
      <Stars radius={300} depth={90} count={2600} factor={4} saturation={0} fade speed={reducedMotion ? 0 : 0.35} />
      <DustStorm />
      <Ground />
      <Terrain slopeColors={view.slope} />
      <HazardVisuals reality={view.reality} />
      <Lander />
      {view.frontiers && <Frontiers />}
      <Targets />
      <Rovers showPaths={view.paths} selected={selected} onSelect={onSelect} />
      {view.lidar && <Lidar />}
      {view.reality && <Reality />}
      <FocusMarker focus={focus} />
      <CameraRig follow={follow} overview={overview} focus={focus} />
    </>
  )
}
