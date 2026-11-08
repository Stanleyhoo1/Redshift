import { useSim } from '../state/store.js'
import { FLORA, ICE, TERRAIN } from '../lib/terrainTheme.js'
import '../styles/terrain-legend.css'

// v7 terrain legend entries. Each one shows only when its field is present and non-empty
// (features or fleet), so an older sim, the mock or a world with the feature off adds nothing.

const some = (v) => Array.isArray(v) && v.length > 0

export function TerrainLegend() {
  const f = useSim((s) => s.features)
  const fleet = useSim((s) => s.fleet)
  const flora = some(f?.flora) ? f.flora : []
  const trees = flora.some((p) => p.kind === 'tree')
  const bushes = flora.some((p) => p.kind !== 'tree')
  const keepOut = some(fleet?.flora_zones) || some(fleet?.geyser_zones)
  const mesas = some(f?.mesas) ? f.mesas : []
  const hasMesa = mesas.some((m) => m.kind !== 'butte')
  const hasButte = mesas.some((m) => m.kind === 'butte')
  const items = []
  if (trees) items.push(['tree', 'Fungal tree', 'Fungal trees: glowing caps the rovers keep clear of', { '--c': FLORA.violet, '--c2': FLORA.teal }])
  if (bushes) items.push(['bush', 'Bush', 'Glowing bushes: small plants the rovers keep clear of', { '--c': FLORA.bulb }])
  if (keepOut) items.push(['keep', 'Keep-out', 'Keep-out ring the fleet holds around plants and geysers', { '--c': FLORA.ring }])
  if (some(f?.ice)) items.push(['ice', 'Ice sheet', 'Ice sheets: slick level ground', { '--c': ICE.sheet }])
  if (some(fleet?.ice_zones)) items.push(['icez', 'Ice zone', 'Ice the fleet found through wheel slip', { '--c': ICE.zone }])
  if (some(f?.geysers)) items.push(['geyser', 'Geyser', 'Geyser vents: erupt on a cycle, the rovers keep clear', { '--c': TERRAIN.geyser, '--c2': TERRAIN.steam }])
  if (some(f?.dunes)) items.push(['dune', 'Dunes', 'Dune fields: soft sand ripples', { '--c': TERRAIN.dune, '--c2': TERRAIN.duneShade }])
  if (hasMesa || hasButte) {
    const label = hasMesa && hasButte ? 'Mesa, butte' : hasButte ? 'Butte' : 'Mesa'
    items.push(['mesa', label, 'Flat-topped rock landforms with cliff sides', { '--c': TERRAIN.mesaTop, '--c2': TERRAIN.mesa }])
  }
  if (f?.canyon && Array.isArray(f.canyon.path)) items.push(['canyon', 'Canyon', 'A canyon with cliff walls and a few ramps down', { '--c': TERRAIN.canyon, '--c2': TERRAIN.mesa }])
  if (some(f?.pits)) items.push(['pit', 'Pit', 'Sinkholes and skylights: deep holes in the ground', { '--c': TERRAIN.pit, '--c2': TERRAIN.pitRim }])
  if (some(f?.boulder_fields)) items.push(['boulders', 'Boulders', 'Boulder fields: dense rock the rovers steer through', { '--c': TERRAIN.boulder }])
  if (!items.length) return null
  return (
    <div className="legend-terrain">
      <div className="legend-sub">Terrain</div>
      <ul className="legend-list legend-row">
        {items.map(([k, label, title, style]) => (
          <li key={k} title={title}>
            <i className={`sw sw-t-${k}`} style={style} />
            {label}
          </li>
        ))}
      </ul>
    </div>
  )
}
