import { FACTOR_LABEL, fmt } from '../../lib/format'
import type { PhysicsPreview } from '../../lib/types'

interface Props {
  g: number
  min: number
  max: number
  step: number
  onChange: (g: number) => void
  preview: PhysicsPreview
  live: boolean
}

/**
 * Wordmark crushed by gravity: as g rises it gets squatter, heavier and wider, while its size is
 * solved (container query units) so it always spans the column. Archivo caps ≈ 0.70 em wide at 100 %.
 */
export function Wordmark({ g, min, max }: { g: number; min: number; max: number }) {
  const t = Math.max(0, Math.min(1, (g - min) / (max - min || 1)))
  const stretch = 70 + 42 * t
  return (
    <div style={{ containerType: 'inline-size' }}>
      <h1
        aria-label="Redshift"
        className="select-none whitespace-nowrap leading-[0.78] text-bone"
        style={{
          fontSize: `min(184px, calc(100cqw / ${((5.72 * stretch) / 100).toFixed(3)}))`,
          fontStretch: `${stretch}%`,
          fontWeight: Math.round(480 + 420 * t),
          letterSpacing: `${(-0.01 + 0.015 * t).toFixed(3)}em`,
          transform: `scaleY(${(1.08 - 0.36 * t).toFixed(3)})`,
          transformOrigin: '0 100%',
          transition: 'font-size 240ms ease-out, font-stretch 240ms ease-out, font-weight 240ms ease-out, transform 240ms ease-out',
        }}
      >
        REDSHIFT
      </h1>
    </div>
  )
}

export function GravityControl({ g, min, max, step, onChange, preview, live }: Props) {
  const ticks: number[] = []
  for (let v = min; v <= max + 1e-6; v += step) ticks.push(Math.round(v * 10) / 10)
  const pct = ((g - min) / (max - min)) * 100
  return (
    <section aria-labelledby="gravity-label" className="flex flex-col gap-5">
      <div className="flex items-end justify-between gap-6">
        <div>
          <h2 id="gravity-label" className="text-[15px] font-medium text-haze">
            Surface gravity
          </h2>
          <p className="text-[13px] text-dim">{fmt(g * 9.81, 1)} m/s² at the surface</p>
        </div>
        <output htmlFor="gravity" className="font-cond num text-[64px] leading-none font-[850] text-bone">
          {g.toFixed(1)}
          <span className="ml-1 text-[34px] font-[600] text-haze">g</span>
        </output>
      </div>

      <div className="relative pt-1">
        <input
          id="gravity"
          type="range"
          className="g-slider w-full"
          min={min}
          max={max}
          step={step}
          value={g}
          onChange={(e) => onChange(Math.round(parseFloat(e.target.value) * 10) / 10)}
          style={{ ['--fill' as string]: `${pct}%` }}
          aria-valuetext={`${g.toFixed(1)} g`}
        />
        <div className="relative mt-2 h-5 text-[11px] text-dim num" aria-hidden>
          {ticks.map((v) => {
            const left = ((v - min) / (max - min)) * 100
            const major = Math.abs(v * 2 - Math.round(v * 2)) < 1e-6
            const edge = left < 1 ? 'left' : left > 99 ? 'right' : 'mid'
            const shift = edge === 'left' ? '' : edge === 'right' ? '-translate-x-full' : '-translate-x-1/2'
            const align = edge === 'left' ? 'text-left' : edge === 'right' ? 'text-right' : 'text-center'
            return (
              <span key={v} className={`absolute top-0 ${shift} ${align}`} style={{ left: `${left}%` }}>
                <span className={`block w-px ${major ? 'h-2 bg-haze/70' : 'h-1 bg-rule-2'} ${edge === 'left' ? '' : edge === 'right' ? 'ml-auto' : 'mx-auto'}`} />
                {major && <span className="mt-0.5 block whitespace-nowrap">{v === 1 ? '1.0 Earth' : v.toFixed(1)}</span>}
              </span>
            )
          })}
        </div>
      </div>

      <div className="grid grid-cols-[minmax(0,1fr)_auto] items-center gap-6">
        <SlopeDiagram p={preview} />
        <dl className="grid min-w-[172px] gap-3 text-[13px]">
          <Stat label="Max climbable slope" value={`${fmt(preview.max_slope_deg, 1)}°`} strong />
          <Stat label="Limited by" value={FACTOR_LABEL[preview.factor] ?? preview.factor} />
          <Stat label="Tip alarm" value={`${fmt(preview.tip_alarm_deg, 1)}°`} tone="text-ha" />
          <Stat label="Speed limit" value={`${fmt(preview.v_max, 2)} m/s`} />
          <Stat label="Flat-ground energy" value={`${fmt(preview.energy_j_per_m, 0)} J/m`} />
        </dl>
      </div>
      {!live && <p className="text-[12px] text-dim">Physics preview computed in the browser (backend not reachable).</p>}
    </section>
  )
}

function Stat({ label, value, strong, tone }: { label: string; value: string; strong?: boolean; tone?: string }) {
  return (
    <div className="flex items-baseline justify-between gap-4 border-b border-rule/70 pb-1.5">
      <dt className="text-haze">{label}</dt>
      <dd className={`num font-semi-cond ${strong ? 'text-[20px] font-[750] text-bone' : 'text-[15px] font-[600]'} ${tone ?? 'text-bone'}`}>
        {value}
      </dd>
    </div>
  )
}

/** Side view: the climbable wedge plus the three physical limits; the lowest one wins. */
function SlopeDiagram({ p }: { p: PhysicsPreview }) {
  const ox = 34
  const oy = 214
  const R = 250
  const pt = (deg: number, r = R) => {
    const a = (deg * Math.PI) / 180
    return [ox + r * Math.cos(a), oy - r * Math.sin(a)] as const
  }
  const limits = [
    { key: 'drive_torque', label: 'Torque', deg: p.drive_deg },
    { key: 'slip', label: 'Traction', deg: p.slip_deg },
    { key: 'tip', label: 'Tip-over', deg: p.tip_deg },
  ]
  const [mx, my] = pt(p.max_slope_deg)
  const [ax, ay] = pt(p.tip_alarm_deg, R + 6)
  // rover sitting on the max-slope line
  const roverAt = 150
  const [rx, ry] = pt(p.max_slope_deg, roverAt)
  const arc = (deg: number, r: number) => {
    const [x, y] = pt(deg, r)
    return `M ${ox + r} ${oy} A ${r} ${r} 0 0 0 ${x} ${y}`
  }
  return (
    <svg viewBox="8 0 384 236" className="w-full max-w-[400px]" role="img" aria-label={`Maximum climbable slope ${p.max_slope_deg.toFixed(1)} degrees`}>
      <defs>
        <linearGradient id="wedge" x1="0" y1="1" x2="1" y2="0">
          <stop offset="0" stopColor="#b5482a" stopOpacity="0.85" />
          <stop offset="1" stopColor="#d39a4a" stopOpacity="0.55" />
        </linearGradient>
      </defs>
      <line x1={ox - 20} y1={oy} x2={ox + R + 10} y2={oy} stroke="#4a4282" strokeWidth="1" />
      <path d={`M ${ox} ${oy} L ${ox + R} ${oy} A ${R} ${R} 0 0 0 ${mx} ${my} Z`} fill="url(#wedge)" style={{ transition: 'd 220ms' }} />
      {limits.map((l) => {
        const [x, y] = pt(l.deg)
        const [lx, ly] = pt(l.deg, R + 10)
        const active = l.key === p.factor
        return (
          <g key={l.key}>
            <line x1={ox} y1={oy} x2={x} y2={y} stroke={active ? '#ece7f5' : '#6e6797'} strokeWidth={active ? 1.5 : 1} />
            <text x={lx} y={ly + 4} fontSize="13" fill={active ? '#ece7f5' : '#a39cc8'} fontWeight={active ? 700 : 400}>
              {l.label} {l.deg.toFixed(1)}°
            </text>
          </g>
        )
      })}
      <line x1={ox} y1={oy} x2={ax} y2={ay} stroke="#ff4a5c" strokeWidth="1.25" strokeDasharray="4 4" />
      <text x={ax + 6} y={ay + 4} fontSize="13" fill="#ff4a5c">
        Tip alarm {p.tip_alarm_deg.toFixed(1)}°
      </text>
      <path d={arc(p.max_slope_deg, 58)} fill="none" stroke="#ece7f5" strokeWidth="1" />
      <text {...labelAt(pt(p.max_slope_deg / 2, 70))} fontSize="15" fontWeight="700" fill="#ece7f5">
        {p.max_slope_deg.toFixed(1)}°
      </text>
      <g transform={`translate(${rx} ${ry}) rotate(${-p.max_slope_deg})`}>
        <rect x={-16} y={-17} width={32} height={11} rx={2} fill="#ece7f5" />
        <rect x={-4} y={-24} width={10} height={7} fill="#a39cc8" />
        <circle cx={-10} cy={-4} r={4} fill="#16132b" stroke="#ece7f5" strokeWidth="1.5" />
        <circle cx={10} cy={-4} r={4} fill="#16132b" stroke="#ece7f5" strokeWidth="1.5" />
      </g>
    </svg>
  )
}

const labelAt = ([x, y]: readonly [number, number]) => ({ x, y: y + 4 })
