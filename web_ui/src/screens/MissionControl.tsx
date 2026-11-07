import { lazy, Suspense, useEffect, useRef, useState } from 'react'
import { api } from '../lib/api'
import { fmtElapsed, robotName } from '../lib/format'
import { useStore } from '../lib/store'
import { C } from '../lib/theme'
import { CameraFeed } from '../panels/CameraFeed'
import { DecisionTrace } from '../panels/DecisionTrace'
import { EdgeHud } from '../panels/EdgeHud'
import { FleetStatus } from '../panels/FleetStatus'
import { ImuAttitude } from '../panels/ImuAttitude'
import { InjectBar } from '../panels/InjectBar'
import { LidarRadar } from '../panels/LidarRadar'
import { MissionProgress } from '../panels/MissionProgress'

const Globe = lazy(() => import('../globe/Globe'))

export default function MissionControl() {
  return (
    <div className="grid h-full grid-rows-[56px_minmax(0,1fr)]">
      <TopBar />
      <div className="grid min-h-0 grid-cols-[clamp(340px,22.5vw,440px)_minmax(0,1fr)_clamp(370px,24.5vw,480px)] gap-2.5 px-2.5 pb-2.5">
        <aside className="scroll-thin flex min-h-0 flex-col gap-2.5 overflow-y-auto" aria-label="Sensors">
          <CameraFeed />
          <div className="grid min-h-[220px] flex-1 grid-cols-[minmax(0,1.15fr)_minmax(0,1fr)] gap-2.5">
            <LidarRadar />
            <ImuAttitude />
          </div>
          <EdgeHud />
        </aside>
        <GlobeStage />
        <aside className="min-h-0" aria-label="Decisions">
          <DecisionTrace />
        </aside>
      </div>
    </div>
  )
}

function GlobeStage() {
  const seismic = useStore((s) => !!s.telemetry?.mission.env.seismic)
  const follow = useStore((s) => s.followRobot)
  const sel = useStore((s) => s.selectedRobot)
  const status = useStore((s) => s.telemetry?.mission.status ?? s.status?.status)
  const detail = useStore((s) => s.status?.detail)
  return (
    <section className="relative min-h-0 overflow-hidden rounded-[3px] border border-rule bg-[radial-gradient(ellipse_at_50%_45%,#231c42_0%,#16132b_62%)]" aria-label="Globe">
      <div className={`absolute inset-0 ${seismic ? 'animate-[shake_0.18s_linear_infinite]' : ''}`}>
        <Suspense fallback={<div className="grid h-full place-items-center text-dim">Rendering the planet…</div>}>
          <Globe />
        </Suspense>
      </div>
      <div className="pointer-events-none absolute inset-0 flex flex-col justify-between p-3">
        <div className="flex items-start justify-between gap-3">
          <MissionProgress />
          <div className="pointer-events-auto flex flex-col items-end gap-2">
            <button
              type="button"
              onClick={() => useStore.getState().setFollow(!follow)}
              aria-pressed={follow}
              className={`rounded-[3px] border px-3 py-1.5 text-[12.5px] font-[650] backdrop-blur-md transition-colors ${follow ? 'border-hb bg-hb/20 text-bone' : 'border-rule-2 bg-void/70 text-haze hover:text-bone'}`}
            >
              {follow ? `Following ${sel ? robotName(sel) : ''}` : 'Follow selected robot'}
            </button>
            <MapLegend />
          </div>
        </div>
        {status === 'degraded' || status === 'failed' ? (
          <div className="pointer-events-auto mx-auto rounded-[3px] border px-4 py-2 text-[13px] backdrop-blur-md" style={{ borderColor: status === 'failed' ? C.ha : C.nad, color: status === 'failed' ? C.ha : C.nad, backgroundColor: '#16132bcc' }}>
            {status === 'failed' ? 'Mission failed' : 'Running degraded'}
            {detail ? `: ${detail}` : ''}
          </div>
        ) : null}
        <div className="mx-auto w-full max-w-[980px]">
          <InjectBar />
        </div>
      </div>
    </section>
  )
}

const LEGEND: [string, string, string?][] = [
  ['Unmapped', '#3a3850'],
  ['Mapped terrain', 'linear-gradient(90deg,#2b1f21,#8f3a1f,#db9a4d)'],
  ['Caution slope', C.nad],
  ['Hazard', C.ha],
  ['No-go zone', '#a8102c', 'hatch'],
  ['Science target', C.sci],
  ['Rock', '#5c3438'],
  ['Just revealed', '#40d9ff'],
]

function MapLegend() {
  return (
    <ul className="grid gap-1 rounded-[3px] border border-rule bg-void/70 px-3 py-2 text-[11.5px] text-haze backdrop-blur-md">
      {LEGEND.map(([label, bg, hatch]) => (
        <li key={label} className="flex items-center gap-2">
          <span className={`h-2.5 w-4 rounded-[1px] ${hatch ? 'hatch text-[#ff4a5c]' : ''}`} style={{ background: hatch ? undefined : bg, backgroundColor: hatch ? bg : undefined }} />
          {label}
        </li>
      ))}
      <li className="mt-1 max-w-[150px] border-t border-rule pt-1.5 text-[10.5px] leading-snug text-dim">Drag to orbit, scroll to zoom, double-click to reset</li>
    </ul>
  )
}

function TopBar() {
  const mission = useStore((s) => s.telemetry?.mission)
  const wsStatus = useStore((s) => s.status?.status)
  const status = mission?.status ?? wsStatus
  const conn = useStore((s) => s.conn)
  const cfg = useStore((s) => s.telemetry?.mission.config ?? s.lastConfig)
  const sectors = useStore((s) => s.options.sectors)
  const sector = sectors.find((x) => x.id === cfg?.sector)
  const g = cfg?.gravity_g ?? 2.2
  const t = Math.max(0, Math.min(1, (g - 1) / 2))
  const backend = mission?.backend ?? cfg?.sim_backend
  const statusColor = status === 'running' ? C.o3 : status === 'degraded' ? C.nad : status === 'failed' ? C.ha : C.haze

  return (
    <header className="flex min-w-0 items-center gap-5 px-4">
      <span className="text-[26px] leading-none text-bone" style={{ fontStretch: `${100 - 36 * t}%`, fontWeight: Math.round(620 + 260 * t) }}>
        REDSHIFT
      </span>
      <div className="flex items-center gap-2 border-l border-rule pl-5">
        <span className="h-2 w-2 rounded-full" style={{ backgroundColor: statusColor, boxShadow: `0 0 8px ${statusColor}` }} />
        <span className="text-[14px] font-[650] capitalize" style={{ color: statusColor }}>
          {status ?? 'connecting'}
        </span>
        <span className="num ml-2 text-[14px] text-bone">T+{fmtElapsed(mission?.elapsed_s ?? 0)}</span>
      </div>
      <div className="num hidden items-baseline gap-4 text-[12.5px] text-haze xl:flex">
        <span>
          <span className="text-bone">{g.toFixed(1)} g</span> gravity
        </span>
        <span>
          <span className="text-bone">{sector ? `${sector.size_m} m` : '—'}</span> sector
        </span>
        <span>{backend === 'gazebo' ? 'Gazebo Harmonic' : backend === 'mock' ? 'Mock physics' : (backend ?? '')}</span>
      </div>
      <div className="ml-auto flex min-w-0 items-center gap-3">
        <FleetStatus />
        {conn !== 'open' && <span className="text-[12px] text-nad">Link {conn === 'connecting' ? 'reconnecting…' : 'lost'}</span>}
        <AbortButton />
      </div>
    </header>
  )
}

function AbortButton() {
  const [armed, setArmed] = useState(false)
  const [busy, setBusy] = useState(false)
  const timer = useRef<ReturnType<typeof setTimeout> | undefined>(undefined)
  useEffect(() => () => clearTimeout(timer.current), [])
  const click = async () => {
    if (!armed) {
      setArmed(true)
      timer.current = setTimeout(() => setArmed(false), 3000)
      return
    }
    clearTimeout(timer.current)
    setBusy(true)
    try {
      await api.abort()
    } catch {
      /* backend gone: still return to configuration */
    }
    setBusy(false)
    setArmed(false)
    useStore.getState().setScreen('prelaunch')
  }
  return (
    <button
      type="button"
      onClick={click}
      disabled={busy}
      className={`h-9 shrink-0 rounded-[3px] border px-3.5 text-[13px] font-[700] transition-colors ${armed ? 'border-ha bg-ha text-void' : 'border-ha/60 text-ha hover:bg-ha/10'}`}
    >
      {busy ? 'Aborting…' : armed ? 'Click again to abort' : 'Abort mission'}
    </button>
  )
}
