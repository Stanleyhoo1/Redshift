import { useRobotIds } from '../lib/hooks'
import { fmt, robotName } from '../lib/format'
import { useStore } from '../lib/store'
import { robotColor, stateColor } from '../lib/theme'
import { FleetGlyph } from '../screens/prelaunch/ConfigFields'
import { StatePill } from '../ui/Panel'

/** One chip per robot: state, speed vs gravity limit, perception mode. Click to select. */
export function FleetStatus() {
  const ids = useRobotIds()
  return (
    <div className="flex min-w-0 items-center gap-1.5" role="list" aria-label="Fleet">
      {ids.map((id) => (
        <FleetChip key={id} id={id} />
      ))}
    </div>
  )
}

function FleetChip({ id }: { id: string }) {
  const r = useStore((s) => s.telemetry?.robots[id])
  const selected = useStore((s) => s.selectedRobot === id)
  if (!r) return null
  const isScout = r.type === 'scout'
  const sc = stateColor(r.state)
  return (
    <button
      role="listitem"
      onClick={() => useStore.getState().selectRobot(id)}
      aria-pressed={selected}
      className={`flex h-10 min-w-0 items-center gap-2 rounded-[3px] border px-2.5 text-left transition-colors ${selected ? 'border-rule-2 bg-bay-2' : 'border-transparent hover:border-rule'}`}
    >
      <FleetGlyph kind={isScout ? 'scout' : 'rover'} on size={14} color={robotColor(id)} />
      <span className="min-w-0">
        <span className="flex items-center gap-1.5">
          <span className="text-[13px] font-[680] text-bone">{robotName(id)}</span>
          <StatePill state={r.state} color={sc} />
        </span>
        <span className="num block text-[11px] leading-tight text-dim">
          {isScout
            ? `${fmt(r.speed, 1)} m/s, ${fmt(r.altitude_agl, 0)} m up`
            : `${fmt(r.speed, 2)} / ${fmt(r.v_limit, 2)} m/s${r.mode === 'LIDAR_ONLY' ? ', LiDAR only' : ''}`}
        </span>
      </span>
    </button>
  )
}
