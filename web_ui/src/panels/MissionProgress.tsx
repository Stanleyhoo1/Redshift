import { fmtElapsed } from '../lib/format'
import { useStore } from '../lib/store'
import { C } from '../lib/theme'

/** Overlay on the globe stage: coverage first, then what the fleet has found. */
export function MissionProgress() {
  const m = useStore((s) => s.telemetry?.mission)
  const stats = useStore((s) => s.map?.stats)
  const pct = m?.percent_mapped ?? stats?.percent ?? 0
  const perm = stats?.permanent_hazards.length ?? 0
  const dust = m?.env.dust_level ?? 0
  const seismic = !!m?.env.seismic

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
      {(dust > 0 || seismic) && (
        <p className="mt-3 flex items-center gap-2 rounded-[2px] bg-ochre/15 px-2 py-1.5 text-[12px] text-ochre">
          <span className="h-2 w-2 animate-pulse rounded-full bg-ochre" />
          {dust > 0 ? `Dust storm, opacity ${dust.toFixed(2)}` : ''}
          {dust > 0 && seismic ? ', ' : ''}
          {seismic ? 'seismic tremor' : ''}
        </p>
      )}
    </div>
  )
}

function Count({ label, value, color }: { label: string; value: number; color: string }) {
  return (
    <div className="min-w-0">
      <dd className="font-semi-cond num text-[22px] leading-none font-[780]" style={{ color }}>
        {value.toLocaleString()}
      </dd>
      <dt className="mt-0.5 text-haze">{label}</dt>
    </div>
  )
}
