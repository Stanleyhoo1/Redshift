import { useEffect, useState, type ReactNode } from 'react'
import { api } from '../lib/api'
import { useRobotIds, useTick } from '../lib/hooks'
import { robotName } from '../lib/format'
import { isLiveStatus, useStore } from '../lib/store'
import { C } from '../lib/theme'
import type { OkResponse } from '../lib/types'

type Kind = 'rockslide' | 'anomaly' | 'tip'

interface Result {
  ok: boolean
  text: string
}

const COOLDOWN_MS: Record<Kind, number> = { rockslide: 10000, anomaly: 30000, tip: 25000 }

export function InjectBar() {
  const ids = useRobotIds()
  const rovers = ids.filter((i) => i.startsWith('rover'))
  const selected = useStore((s) => s.selectedRobot)
  const status = useStore((s) => s.telemetry?.mission.status ?? s.status?.status)
  const dust = useStore((s) => s.telemetry?.mission.env.dust_level ?? 0)
  const [target, setTarget] = useState<string | null>(null)
  const live = isLiveStatus(status)

  // follow the operator's selection when it is a rover
  useEffect(() => {
    if (selected?.startsWith('rover')) setTarget(selected)
  }, [selected])
  const rover = target && rovers.includes(target) ? target : rovers[0]

  return (
    <div className="@container pointer-events-auto rounded-[3px] border border-rule-2 bg-void/88 p-2.5 shadow-[0_18px_50px_-20px_rgba(0,0,0,0.8)] backdrop-blur-md">
      <div className="grid grid-cols-3 items-start gap-2 @min-[860px]:grid-cols-[150px_repeat(3,minmax(0,1fr))]">
        {/* wide: a title block left of the buttons; narrow: one header line above them */}
        <div className="col-span-3 flex items-baseline justify-between gap-3 px-0.5 @min-[860px]:col-span-1 @min-[860px]:min-h-[60px] @min-[860px]:flex-col @min-[860px]:items-start @min-[860px]:justify-between @min-[860px]:gap-1.5 @min-[860px]:px-0 @min-[860px]:pr-1">
          <h2 className="font-cond text-[17px] leading-[1.02] font-[800] text-bone @min-[860px]:text-[19px]">Inject unscripted change</h2>
          <div className="flex flex-wrap items-center gap-1 text-[11.5px] text-haze">
            {rovers.length > 1 ? 'Target' : null}
            {rovers.length > 1 &&
              rovers.map((r) => (
                <button
                  key={r}
                  onClick={() => setTarget(r)}
                  className={`rounded-[2px] px-1.5 py-0.5 font-[650] ${r === rover ? 'bg-bay-2 text-bone shadow-[inset_0_0_0_1px_var(--color-rule-2)]' : 'text-dim hover:text-haze'}`}
                >
                  {robotName(r)}
                </button>
              ))}
            {rovers.length === 1 && <span>Target: {robotName(rovers[0])}</span>}
            {!rovers.length && <span className="text-dim">No rover online</span>}
          </div>
        </div>
        <InjectButton
          kind="rockslide"
          title="Spawn Meteor / Rockslide Ahead"
          sub={rover ? `Boulders land across ${robotName(rover)}'s path` : 'Needs a rover'}
          color={C.nad}
          icon={<MeteorIcon />}
          disabled={!live || !rover}
          run={() => api.injectRockslide(rover)}
        />
        <InjectButton
          kind="anomaly"
          title="Trigger High-G Seismic / Dust Anomaly"
          sub={dust > 0 ? `Dust storm active, level ${dust.toFixed(2)}` : 'Dust haze for 25 s with a 4 s tremor'}
          color={C.ochre}
          icon={<DustIcon />}
          disabled={!live}
          run={() => api.injectAnomaly({})}
        />
        <InjectButton
          kind="tip"
          title="Trigger Tip/Slip Failure"
          sub={rover ? `Physical jolt to ${robotName(rover)}'s chassis` : 'Needs a rover'}
          color={C.ha}
          icon={<TipIcon />}
          disabled={!live || !rover}
          run={() => api.injectTip(rover)}
        />
      </div>
    </div>
  )
}

function InjectButton({
  kind,
  title,
  sub,
  color,
  icon,
  disabled,
  run,
}: {
  kind: Kind
  title: string
  sub: string
  color: string
  icon: ReactNode
  disabled: boolean
  run: () => Promise<OkResponse>
}) {
  const [busy, setBusy] = useState(false)
  const [until, setUntil] = useState(0)
  const [result, setResult] = useState<Result | null>(null)
  useTick(200)
  const now = performance.now()
  const remaining = Math.max(0, until - now)
  const cooling = remaining > 0

  const fire = async () => {
    setBusy(true)
    setResult(null)
    try {
      const r = await run()
      setResult({ ok: r.ok, text: r.detail ?? (r.ok ? 'Injected' : 'Rejected by backend') })
      if (r.ok) setUntil(performance.now() + COOLDOWN_MS[kind])
    } catch (e) {
      setResult({ ok: false, text: (e as Error).message })
    } finally {
      setBusy(false)
    }
  }

  return (
    <div className="flex min-w-0 flex-col gap-1">
      <button
        type="button"
        onClick={fire}
        disabled={disabled || busy || cooling}
        className="group relative flex min-h-[60px] items-center gap-2.5 overflow-hidden rounded-[3px] border px-3 py-1.5 text-left transition-colors disabled:opacity-55"
        style={{ borderColor: `${color}88`, backgroundColor: `${color}14` }}
      >
        <span className="hidden shrink-0 @min-[700px]:block" style={{ color }}>
          {icon}
        </span>
        <span className="min-w-0">
          <span className="font-semi-cond block text-[14.5px] leading-[1.1] font-[780] text-bone">{busy ? 'Injecting…' : title}</span>
          <span className="block truncate text-[11.5px] text-haze">{cooling ? `Ready again in ${Math.ceil(remaining / 1000)} s` : sub}</span>
        </span>
        <span className="absolute inset-0 opacity-0 transition-opacity group-enabled:group-hover:opacity-100" style={{ boxShadow: `inset 0 0 0 1px ${color}` }} />
        {cooling && (
          <span className="absolute bottom-0 left-0 h-[3px]" style={{ width: `${(remaining / COOLDOWN_MS[kind]) * 100}%`, backgroundColor: color }} />
        )}
      </button>
      <p className={`h-4 truncate px-1 text-[11.5px] ${result ? (result.ok ? 'text-o3' : 'text-ha') : 'text-transparent'}`} role="status">
        {result ? `${result.ok ? 'Sent:' : 'Failed:'} ${result.text}` : '.'}
      </p>
    </div>
  )
}

const MeteorIcon = () => (
  <svg width="30" height="30" viewBox="0 0 30 30" aria-hidden fill="none" stroke="currentColor" strokeWidth="1.8">
    <path d="M3 3 L17 17" strokeLinecap="round" />
    <path d="M8 2 L19 13 M2 8 L13 19" strokeLinecap="round" opacity="0.5" />
    <circle cx="20" cy="20" r="5" fill="currentColor" />
    <path d="M12 28 H28" />
  </svg>
)
const DustIcon = () => (
  <svg width="30" height="30" viewBox="0 0 30 30" aria-hidden fill="none" stroke="currentColor" strokeWidth="1.8" strokeLinecap="round">
    <path d="M3 9 H19 a3.5 3.5 0 1 0 -3.5 -3.5" />
    <path d="M3 15 H25 a3.5 3.5 0 1 1 -3.5 3.5" />
    <path d="M3 21 H13" />
    <path d="M2 27 L6 24 L9 27 L13 23 L16 27 L20 24 L23 27 L28 24" strokeWidth="1.5" />
  </svg>
)
const TipIcon = () => (
  <svg width="30" height="30" viewBox="0 0 30 30" aria-hidden fill="none" stroke="currentColor" strokeWidth="1.8">
    <g transform="rotate(-28 15 18)">
      <rect x="6" y="11" width="18" height="7" rx="1.5" fill="currentColor" fillOpacity="0.25" />
      <circle cx="9.5" cy="20.5" r="2.5" />
      <circle cx="20.5" cy="20.5" r="2.5" />
    </g>
    <path d="M2 27 H28" />
    <path d="M24 3 l2 4 M28 6 l-4 1" strokeLinecap="round" />
  </svg>
)
