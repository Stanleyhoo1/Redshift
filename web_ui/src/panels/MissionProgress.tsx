import { fmtElapsed } from '../lib/format'
import { useStore } from '../lib/store'
import { envLine } from '../lib/storm'
import { C } from '../lib/theme'

/** Overlay on the globe stage: coverage first, then what the fleet has found. */
export function MissionProgress() {
  const m = useStore((s) => s.telemetry?.mission)
  const stats = useStore((s) => s.map?.stats)
  const pct = m?.percent_mapped ?? stats?.percent ?? 0
  const perm = stats?.permanent_hazards.length ?? 0
  const env = envLine(m?.env)
  const envColor = env?.storm ? C.storm : C.ochre

  return (
    <div className="pointer-events-auto w-[300px] rounded-[3px] border border-rule bg-void/75 px-4 py-3 backdrop-blur-md">
      <div className="flex items-baseline justify-between">
        <span className="text-[13px] text-haze">Sector mapped</span>
        <span className="num text-[12px] text-dim">T+{fmtElapsed(m?.elapsed_s ?? 0)}</span>
      </div>
      <div className="font-cond num mt-0.5 text-[56px] leading-[0.95] font-[850] text-bone">
        {pct.toFixed(1)}
        <span className="text-[28px] font-[650] text-haze">%</span>
      </div>
      <div className="mt-2 h-1.5 overflow-hidden rounded-full bg-rule">
        <div className="h-1.5 rounded-full transition-[width] duration-700" style={{ width: `${Math.min(100, pct)}%`, background: `linear-gradient(90deg, ${C.rust}, ${C.ochre})` }} />
      </div>
      <dl className="mt-3 grid grid-cols-3 gap-2 text-[11.5px]">
        <Count label="Science targets" value={m?.science_found ?? stats?.science.length ?? 0} color={C.sci} />
        <Count label="Hazard cells" value={m?.hazard_cells ?? stats?.hazard_cells ?? 0} color={C.ha} />
        <Count label="No-go zones" value={perm} color="#d23a55" />
      </dl>
      {env && (
        <p className="mt-3 flex items-center gap-2 rounded-[2px] px-2 py-1.5 text-[12px]" style={{ color: envColor, backgroundColor: `${envColor}26` }}>
          <span className="h-2 w-2 shrink-0 animate-pulse rounded-full" style={{ backgroundColor: envColor }} />
          {env.text}
        </p>
      )}
    </div>
  )
}

function Count({ label, value, color }: { label: string; value: number; color: string }) {
  return (
    <div className="flex min-w-0 flex-col-reverse">
      <dt className="mt-0.5 text-haze">{label}</dt>
      <dd className="font-semi-cond num text-[22px] leading-none font-[780]" style={{ color }}>
        {value.toLocaleString()}
      </dd>
    </div>
  )
}
