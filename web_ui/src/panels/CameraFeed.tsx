import { useRobotIds, useSelected, useTick } from '../lib/hooks'
import { fmt, robotName } from '../lib/format'
import { useStore } from '../lib/store'
import { C, stateColor } from '../lib/theme'
import { FleetGlyph } from '../screens/prelaunch/ConfigFields'
import { Badge, Panel, StatePill } from '../ui/Panel'

export function CameraFeed() {
  const ids = useRobotIds()
  const [sel, robot] = useSelected()
  const frame = useStore((s) => (sel ? s.cameras[sel] : undefined))
  useTick(500)

  const age = frame ? performance.now() - frame.receivedAt : Infinity
  const live = age < 1500
  const lidarOnly = robot?.mode === 'LIDAR_ONLY'
  const isScout = robot?.type === 'scout'

  return (
    <Panel
      title="Perception camera"
      sensors={isScout ? ['CAM'] : ['CAM', 'DEPTH']}
      chipsFrom={380}
      right={
        <div role="tablist" aria-label="Camera source" className="flex gap-0.5">
          {ids.map((id) => {
            const on = id === sel
            return (
              <button
                key={id}
                role="tab"
                aria-selected={on}
                onClick={() => useStore.getState().selectRobot(id)}
                className={`flex h-6 items-center gap-1 rounded-[2px] px-2 text-[12px] font-[600] transition-colors ${on ? 'bg-bay-2 text-bone shadow-[inset_0_0_0_1px_var(--color-rule-2)]' : 'text-dim hover:text-haze'}`}
              >
                <FleetGlyph kind={id.startsWith('scout') ? 'scout' : 'rover'} on={on} size={10} />
                {robotName(id)}
              </button>
            )
          })}
        </div>
      }
    >
      <div className="relative aspect-[4/3] w-full overflow-hidden bg-[#0f0c1e]">
        {frame ? (
          <img
            src={`data:image/jpeg;base64,${frame.jpeg_b64}`}
            alt={`${sel ? robotName(sel) : ''} annotated camera frame`}
            className={`absolute inset-0 h-full w-full object-cover ${live ? '' : 'opacity-50 grayscale'}`}
            draggable={false}
          />
        ) : (
          <div className="absolute inset-0 grid place-items-center px-6 text-center text-[13px] text-dim">
            {sel ? `Waiting for frames from ${robotName(sel)}` : 'No robot selected'}
          </div>
        )}
        {lidarOnly && <div className="pointer-events-none absolute inset-0 bg-[repeating-linear-gradient(0deg,rgba(22,19,43,0.0)_0_3px,rgba(22,19,43,0.35)_3px_4px)]" />}

        <div className="absolute top-2 left-2 flex items-center gap-1.5">
          {live ? (
            <Badge color={C.ha} solid>
              <span className="h-1.5 w-1.5 rounded-full bg-void" /> LIVE
            </Badge>
          ) : (
            <Badge color={C.dim}>{frame ? 'NO SIGNAL' : 'STANDBY'}</Badge>
          )}
          {live && <Badge color={C.bone}>{fmt(frame?.fps, 1)} fps</Badge>}
        </div>
        <div className="absolute top-2 right-2">
          {robot && (lidarOnly ? <Badge color={C.nad} solid>CAM DEGRADED — LIDAR ONLY</Badge> : <Badge color={C.o3}>FUSED</Badge>)}
        </div>
        {robot && (
          <div className="absolute right-2 bottom-2 left-2 flex items-end justify-between gap-2">
            <StatePill state={robot.state} color={stateColor(robot.state)} />
            <span className="num rounded-[2px] bg-void/70 px-1.5 py-0.5 text-[11.5px] text-haze">
              uncertainty <span className="text-bone">{fmt(robot.uncertainty.score, 2)}</span>
              {isScout && robot.altitude_agl !== null ? <>, {fmt(robot.altitude_agl, 1)} m AGL</> : null}
            </span>
          </div>
        )}
      </div>
    </Panel>
  )
}
