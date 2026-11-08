import { useSelected } from '../lib/hooks'
import { fmt } from '../lib/format'
import { imuHistory } from '../lib/history'
import { C } from '../lib/theme'
import { Panel } from '../ui/Panel'

const R = 54 // horizon radius (svg units)
const PITCH_PX_PER_DEG = 1.5

export function ImuAttitude() {
  const [sel, robot] = useSelected()
  const roll = robot?.imu.roll_deg ?? 0
  const pitch = robot?.imu.pitch_deg ?? 0
  const alarm = robot?.tip_alarm_deg ?? 30
  const worst = Math.max(Math.abs(roll), Math.abs(pitch))
  const tone = worst >= alarm ? C.ha : worst >= 0.7 * alarm ? C.nad : C.bone

  return (
    <Panel title="IMU" sensors={['IMU']} chipsFrom={185} right={<span className="num text-[11.5px] text-ha">alarm ±{fmt(alarm, 1)}°</span>} bodyClass="flex flex-col justify-between gap-1.5 px-2.5 pt-1.5 pb-2">
      <div className="flex justify-center">
        <Horizon roll={roll} pitch={pitch} alarm={alarm} />
      </div>
      <dl className="grid grid-cols-2 gap-2 text-[12px]">
        <Readout label="Roll" value={roll} color={Math.abs(roll) >= 0.7 * alarm ? tone : C.bone} />
        <Readout label="Pitch" value={pitch} color={Math.abs(pitch) >= 0.7 * alarm ? tone : C.bone} />
      </dl>
      <Sparkline robot={sel} alarm={alarm} />
    </Panel>
  )
}

function Readout({ label, value, color }: { label: string; value: number; color: string }) {
  return (
    <div className="flex min-w-0 flex-col gap-0.5 border-b border-rule/70 pb-1 @min-[180px]:flex-row @min-[180px]:items-baseline @min-[180px]:justify-between @min-[180px]:gap-1.5">
      <dt className="text-[11px] text-haze @min-[180px]:text-[12px]">{label}</dt>
      <dd className="font-semi-cond num text-[16px] leading-none font-[750] @min-[180px]:text-[18px]" style={{ color }}>
        {value >= 0 ? '+' : '−'}
        {fmt(Math.abs(value), 1)}°
      </dd>
    </div>
  )
}

function arcPath(r: number, fromDeg: number, toDeg: number): string {
  // 0° = straight up, positive = clockwise (roll right)
  const p = (d: number) => {
    const a = ((d - 90) * Math.PI) / 180
    return `${(r * Math.cos(a)).toFixed(2)} ${(r * Math.sin(a)).toFixed(2)}`
  }
  const large = Math.abs(toDeg - fromDeg) > 180 ? 1 : 0
  return `M ${p(fromDeg)} A ${r} ${r} 0 ${large} 1 ${p(toDeg)}`
}

function Horizon({ roll, pitch, alarm }: { roll: number; pitch: number; alarm: number }) {
  const off = Math.max(-40, Math.min(40, pitch)) * PITCH_PX_PER_DEG
  const lim = Math.min(80, alarm + 25)
  return (
    <svg viewBox="-70 -70 140 140" className="h-[96px] w-[96px] shrink-0 @min-[180px]:h-[148px] @min-[180px]:w-[148px]" role="img" aria-label={`Roll ${roll.toFixed(1)}°, pitch ${pitch.toFixed(1)}°`}>
      <defs>
        <clipPath id="hz-clip">
          <circle r={R} />
        </clipPath>
      </defs>
      <g clipPath="url(#hz-clip)">
        <g transform={`rotate(${-roll}) translate(0 ${off})`}>
          <rect x={-120} y={-160} width={240} height={160} fill="#2b2a5a" />
          <rect x={-120} y={0} width={240} height={160} fill="#5a2a1e" />
          <line x1={-120} x2={120} y1={0} y2={0} stroke={C.bone} strokeWidth={1.2} />
          {[-20, -10, 10, 20].map((d) => (
            <g key={d}>
              <line x1={-12} x2={12} y1={-d * PITCH_PX_PER_DEG} y2={-d * PITCH_PX_PER_DEG} stroke={C.bone} strokeOpacity={0.55} strokeWidth={0.8} />
              <text x={15} y={-d * PITCH_PX_PER_DEG + 3} fontSize={7} fill={C.bone} fillOpacity={0.6}>
                {Math.abs(d)}
              </text>
            </g>
          ))}
        </g>
      </g>
      <circle r={R} fill="none" stroke={C.rule2} />
      {/* roll scale with caution / alarm bands */}
      <path d={arcPath(R + 6, 0.7 * alarm, alarm)} stroke={C.nad} strokeWidth={3} fill="none" />
      <path d={arcPath(R + 6, -alarm, -0.7 * alarm)} stroke={C.nad} strokeWidth={3} fill="none" />
      <path d={arcPath(R + 6, alarm, lim)} stroke={C.ha} strokeWidth={3} fill="none" />
      <path d={arcPath(R + 6, -lim, -alarm)} stroke={C.ha} strokeWidth={3} fill="none" />
      {[-45, -30, -15, 0, 15, 30, 45].map((d) => (
        <line key={d} x1={0} x2={0} y1={-R} y2={-R + (d === 0 ? 7 : 4)} stroke={C.haze} strokeWidth={1} transform={`rotate(${d})`} />
      ))}
      <path d="M 0 -50 L -4 -43 L 4 -43 Z" fill={C.bone} transform={`rotate(${roll})`} />
      {/* fixed vehicle symbol */}
      <path d="M -26 0 H -9 L -4 6 L 0 2 L 4 6 L 9 0 H 26" fill="none" stroke={C.nad} strokeWidth={2.2} strokeLinejoin="round" />
      <circle r={1.6} fill={C.nad} />
    </svg>
  )
}

function Sparkline({ robot, alarm }: { robot: string | null; alarm: number }) {
  const hist = robot ? imuHistory(robot) : []
  const W = 200
  const H = 40
  const span = 20
  const tEnd = hist.length ? hist[hist.length - 1].t : 0
  const top = Math.max(alarm * 1.25, 10)
  const toXY = (t: number, v: number) => `${(W - ((tEnd - t) / span) * W).toFixed(1)},${(H - (Math.min(top, Math.abs(v)) / top) * H).toFixed(1)}`
  const roll = hist.map((s) => toXY(s.t, s.roll)).join(' ')
  const pitch = hist.map((s) => toXY(s.t, s.pitch)).join(' ')
  const alarmY = H - (alarm / top) * H
  return (
    <div>
      <svg viewBox={`0 0 ${W} ${H}`} preserveAspectRatio="none" className="h-8 w-full @min-[180px]:h-10" aria-hidden>
        <rect x={0} y={0} width={W} height={alarmY} fill={C.ha} fillOpacity={0.08} />
        <line x1={0} x2={W} y1={alarmY} y2={alarmY} stroke={C.ha} strokeDasharray="3 3" strokeWidth={1} vectorEffect="non-scaling-stroke" />
        <polyline points={pitch} fill="none" stroke={C.hb} strokeWidth={1.2} vectorEffect="non-scaling-stroke" />
        <polyline points={roll} fill="none" stroke={C.bone} strokeWidth={1.4} vectorEffect="non-scaling-stroke" />
      </svg>
      <div className="hidden justify-between text-[10.5px] text-dim @min-[180px]:flex">
        <span>
          <span className="text-bone">roll</span> <span className="text-hb">pitch</span>, last 20 s
        </span>
        <span className="text-ha/80">alarm</span>
      </div>
    </div>
  )
}
