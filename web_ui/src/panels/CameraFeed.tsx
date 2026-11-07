import { useRobotIds, useSelected, useTick } from '../lib/hooks'
import { fmt, robotName, robotShort } from '../lib/format'
import { healthTone } from '../lib/sensors'
import { useStore, type CameraFrame } from '../lib/store'
import { C, stateColor, TONE_COLOR } from '../lib/theme'
import type { CameraKind } from '../lib/types'
import { FleetGlyph } from '../screens/prelaunch/ConfigFields'
import { Badge, Panel, StatePill } from '../ui/Panel'

const STALE_MS = 1500
const TILE_STALE_MS = 2500 // non-selected robots stream at ≤ 4 Hz

const isLive = (f: CameraFrame | undefined, ms: number) => !!f && performance.now() - f.receivedAt < ms

/** Main feed of the selected robot (RGB or depth) plus a live thumbnail of every other robot. */
export function CameraFeed() {
  const ids = useRobotIds()
  const [sel] = useSelected()
  const others = ids.filter((id) => id !== sel)
  useTick(500)

  return (
    <Panel title="Perception cameras" sensors={['CAM', 'DEPTH']} chipsFrom={380} right={<KindToggle />}>
      <MainView />
      {others.length > 0 && (
        <div className="grid grid-cols-2 gap-1.5 border-t border-rule/70 p-1.5" role="list" aria-label="Other robots' cameras">
          {others.map((id) => (
            <Tile key={id} id={id} />
          ))}
        </div>
      )}
    </Panel>
  )
}

function KindToggle() {
  const kind = useStore((s) => s.cameraKind)
  const opts: [CameraKind, string][] = [
    ['rgb', 'RGB'],
    ['depth', 'Depth'],
  ]
  return (
    <div role="radiogroup" aria-label="Camera channel" className="flex rounded-[3px] border border-rule bg-void/60 p-0.5">
      {opts.map(([k, label]) => {
        const on = k === kind
        return (
          <button
            key={k}
            type="button"
            role="radio"
            aria-checked={on}
            onClick={() => useStore.getState().setCameraKind(k)}
            className={`rounded-[2px] px-2 py-[1px] text-[12px] font-[650] transition-colors ${on ? 'bg-bay-2 text-bone shadow-[inset_0_0_0_1px_var(--color-rule-2)]' : 'text-dim hover:text-haze'}`}
          >
            {label}
          </button>
        )
      })}
    </div>
  )
}

function MainView() {
  const [sel, robot] = useSelected()
  const kind = useStore((s) => s.cameraKind)
  const rgb = useStore((s) => (sel ? s.cameras[sel]?.rgb : undefined))
  const depth = useStore((s) => (sel ? s.cameras[sel]?.depth : undefined))
  const frame = kind === 'depth' ? depth : rgb
  const live = isLive(frame, STALE_MS)
  const lidarOnly = robot?.mode === 'LIDAR_ONLY'
  const isScout = robot?.type === 'scout'
  const rescue = robot?.rescue
  const rescuer =
    rescue?.role === 'rescuer' && rescue.peer && rescue.phase !== 'DONE' && rescue.phase !== 'ABORTED'
      ? { ...rescue, peer: rescue.peer }
      : null
  const locked = !!rescuer?.locked
  const sos = robot?.state === 'AWAIT_RESCUE'

  return (
    <div className="relative aspect-[4/3] w-full overflow-hidden bg-[#0f0c1e] [@media(max-height:960px)]:aspect-[16/10]">
      {frame ? (
        <img
          src={`data:image/jpeg;base64,${frame.jpeg_b64}`}
          alt={`${sel ? robotName(sel, robot?.name) : ''} ${kind === 'depth' ? 'depth map' : 'annotated camera frame'}`}
          className={`absolute inset-0 h-full w-full object-cover [@media(max-height:960px)]:object-top ${live ? '' : 'opacity-50 grayscale'}`}
          draggable={false}
        />
      ) : (
        <div className="absolute inset-0 grid place-items-center px-6 text-center text-[13px] text-dim">
          {!sel ? 'No robot selected' : kind === 'depth' ? `No depth stream from ${robotName(sel, robot?.name)} yet. Switch to RGB to see the camera.` : `Waiting for frames from ${robotName(sel, robot?.name)}`}
        </div>
      )}
      {lidarOnly && kind === 'rgb' && <div className="pointer-events-none absolute inset-0 bg-[repeating-linear-gradient(0deg,rgba(22,19,43,0.0)_0_3px,rgba(22,19,43,0.35)_3px_4px)]" />}
      {locked && <LockBrackets />}

      <div className="absolute top-2 left-2 flex items-center gap-1.5">
        {live ? (
          <Badge color={C.ha} solid>
            <span className="h-1.5 w-1.5 rounded-full bg-void" /> LIVE
          </Badge>
        ) : (
          <Badge color={C.dim}>{frame ? 'NO SIGNAL' : 'STANDBY'}</Badge>
        )}
        {live && <Badge color={C.bone}>{fmt(frame?.fps, 1)} fps</Badge>}
        {kind === 'depth' && <Badge color="#b3a4ff">DEPTH</Badge>}
      </div>
      {/* one badge, most important first, so the frame's own labels stay readable */}
      <div className="absolute top-2 right-2">
        {sos ? (
          <Badge color={C.ha} solid pulse>
            SOS SENT
          </Badge>
        ) : rescuer ? (
          <Badge color={C.rescue} solid={locked}>
            {locked ? `LOCKED ON ${robotShort(rescuer.peer)}` : `RESCUING ${robotShort(rescuer.peer)}`}
            {rescuer.range_m != null ? `, ${fmt(rescuer.range_m, 1)} m` : ''}
          </Badge>
        ) : robot ? (
          lidarOnly ? (
            <Badge color={C.nad} solid>
              CAM DEGRADED — LIDAR ONLY
            </Badge>
          ) : (
            <Badge color={C.o3}>FUSED</Badge>
          )
        ) : null}
      </div>
      {kind === 'depth' && frame && <DepthScale />}
      {robot && sel && (
        <div className="absolute right-2 bottom-2 left-2 flex items-end justify-between gap-2">
          <span className="flex min-w-0 items-center gap-1.5">
            <span className="font-cond rounded-[2px] bg-void/75 px-1.5 py-[3px] text-[12.5px] leading-none font-[800] tracking-[0.03em] text-bone">{robotShort(sel)}</span>
            <StatePill state={robot.state} color={stateColor(robot.state)} />
          </span>
          <span className="num shrink-0 rounded-[2px] bg-void/70 px-1.5 py-0.5 text-[11.5px] text-haze">
            uncertainty <span className="text-bone">{fmt(robot.uncertainty.score, 2)}</span>
            {isScout && robot.altitude_agl !== null ? <>, {fmt(robot.altitude_agl, 1)} m AGL</> : null}
          </span>
        </div>
      )}
    </div>
  )
}

/** Viewfinder corners while the recovery rover's perception holds a lock on its teammate. */
function LockBrackets() {
  const corner = 'absolute h-7 w-7 border-rescue'
  return (
    <div className="pointer-events-none absolute inset-3" aria-hidden>
      <span className={`${corner} top-0 left-0 border-t-2 border-l-2`} />
      <span className={`${corner} top-0 right-0 border-t-2 border-r-2`} />
      <span className={`${corner} bottom-0 left-0 border-b-2 border-l-2`} />
      <span className={`${corner} right-0 bottom-0 border-r-2 border-b-2`} />
    </div>
  )
}

/** Turbo colour scale: the depth frames are colour-mapped near = red, far = blue, black = no return. */
function DepthScale() {
  return (
    <div className="absolute right-2 bottom-9 flex items-center gap-1.5 rounded-[2px] bg-void/75 px-1.5 py-1 text-[10.5px] text-haze">
      near
      <span className="h-1.5 w-20 rounded-[1px]" style={{ background: 'linear-gradient(90deg,#7a0403,#f1621b,#e4cf34,#4ef06a,#26bce1,#3e378f)' }} />
      far
    </div>
  )
}

function Tile({ id }: { id: string }) {
  const r = useStore((s) => s.telemetry?.robots[id])
  const frame = useStore((s) => s.cameras[id]?.rgb)
  if (!r) return null
  const live = isLive(frame, TILE_STALE_MS)
  const h = r.health
  const tone = healthTone(h?.pct, h?.critical)
  const hc = h ? TONE_COLOR[tone] : C.dim
  const crit = tone === 'crit'
  const name = robotName(id, r.name)
  return (
    <button
      type="button"
      role="listitem"
      onClick={() => useStore.getState().selectRobot(id)}
      title={`Show ${name}'s camera`}
      className={`group relative aspect-[16/9] overflow-hidden rounded-[2px] border bg-[#0f0c1e] text-left transition-colors [@media(max-height:960px)]:aspect-[2/1] ${crit ? 'animate-crit-glow border-ha' : 'border-rule hover:border-rule-2'}`}
    >
      {frame ? (
        <img src={`data:image/jpeg;base64,${frame.jpeg_b64}`} alt={`${name} camera`} className={`absolute inset-0 h-full w-full object-cover ${live ? '' : 'opacity-45 grayscale'}`} draggable={false} />
      ) : (
        <span className="absolute inset-0 grid place-items-center text-[11px] text-dim">No frames yet</span>
      )}
      <span className="absolute inset-x-0 bottom-0 h-9 bg-[linear-gradient(to_top,rgba(15,12,30,0.92),rgba(15,12,30,0))]" />
      <span className="absolute inset-x-1 top-1 flex items-center justify-between gap-1">
        <span className="flex items-center gap-1 rounded-[2px] bg-void/55 px-1 py-[1px]">
          <span className={`h-1.5 w-1.5 rounded-full ${live ? 'bg-ha' : 'bg-dim'}`} aria-hidden />
          <span className="text-[10px] font-[700] text-bone/80">{live ? 'LIVE' : 'STALE'}</span>
        </span>
        <span className="font-cond rounded-[2px] px-1 py-[2px] text-[10px] leading-none font-[750] whitespace-nowrap" style={{ color: stateColor(r.state), backgroundColor: 'rgba(22,19,43,0.8)' }}>
          {r.state.replace(/_/g, ' ')}
        </span>
      </span>
      <span className="absolute right-1 bottom-1 left-1 flex items-center justify-between gap-1">
        <span className="flex min-w-0 items-center gap-1">
          <FleetGlyph kind={r.type === 'scout' ? 'scout' : 'rover'} on size={9} />
          <span className="font-cond truncate text-[12px] leading-none font-[800] tracking-[0.03em] text-bone">{robotShort(id)}</span>
        </span>
        <span className="flex shrink-0 items-center gap-1">
          <span className="flex items-center gap-0.5" title={h ? `Health ${fmt(h.pct, 0)} %` : 'No health telemetry'}>
            <span className={`h-2 w-2 rounded-full ${crit ? 'animate-pulse' : ''}`} style={{ backgroundColor: hc, boxShadow: crit ? `0 0 6px ${hc}` : undefined }} />
            {h && Number.isFinite(h.pct) && <span className="num text-[10.5px] leading-none font-[650]" style={{ color: hc }}>{h.pct.toFixed(0)}%</span>}
          </span>
        </span>
      </span>
    </button>
  )
}
