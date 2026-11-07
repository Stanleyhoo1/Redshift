import { useEffect, useMemo, useState } from 'react'
import { api } from '../lib/api'
import { DEFAULT_CONFIG } from '../lib/defaults'
import { useStore } from '../lib/store'
import type { MissionConfig } from '../lib/types'
import { BackendPicker, EdgePicker, Field, SectorPicker, Stepper } from './prelaunch/ConfigFields'
import { GravityControl, Wordmark } from './prelaunch/GravityControl'
import { usePhysicsPreview } from './prelaunch/usePhysicsPreview'

export default function PreLaunch() {
  const options = useStore((s) => s.options)
  const optionsLive = useStore((s) => s.optionsLive)
  const conn = useStore((s) => s.conn)
  const [cfg, setCfg] = useState<MissionConfig>(() => useStore.getState().lastConfig ?? options.defaults ?? DEFAULT_CONFIG)
  const [touched, setTouched] = useState(false)
  const [deploying, setDeploying] = useState(false)
  const [error, setError] = useState<string | null>(null)

  // adopt server defaults once they arrive, unless the operator already changed something
  useEffect(() => {
    if (optionsLive && !touched && !useStore.getState().lastConfig) setCfg(options.defaults)
  }, [optionsLive, options.defaults, touched])

  const update = (patch: Partial<MissionConfig>) => {
    setTouched(true)
    setCfg((c) => ({ ...c, ...patch }))
  }

  const { preview, live } = usePhysicsPreview(cfg.gravity_g)
  const sector = options.sectors.find((s) => s.id === cfg.sector)
  const profile = options.edge_profiles.find((p) => p.id === cfg.edge_profile)
  const backend = options.backends.find((b) => b.id === cfg.sim_backend)

  const summary = useMemo(() => {
    const r = cfg.fleet.rovers
    const d = cfg.fleet.drones
    const fleet = `${r} rover${r === 1 ? '' : 's'}${d ? ` and ${d} scout${d === 1 ? '' : 's'}` : ''}`
    return `${sector?.label ?? cfg.sector} sector at ${cfg.gravity_g.toFixed(1)} g with ${fleet} on ${profile?.label ?? cfg.edge_profile}, simulated in ${backend?.label ?? cfg.sim_backend}.`
  }, [cfg, sector, profile, backend])

  const deploy = async () => {
    setDeploying(true)
    setError(null)
    try {
      const res = await api.deploy(cfg)
      if (!res.ok) throw new Error(res.detail ?? 'Backend refused the deployment')
      useStore.getState().beginDeploy(cfg)
    } catch (e) {
      setError((e as Error).message)
    } finally {
      setDeploying(false)
    }
  }

  return (
    <main className="relative grid h-full grid-cols-[minmax(0,44fr)_minmax(0,56fr)] overflow-hidden max-lg:grid-cols-1 max-lg:overflow-y-auto">
      <PlanetLimb g={cfg.gravity_g} />

      <section className="relative z-10 flex min-h-0 flex-col justify-between gap-8 px-[clamp(24px,4vw,72px)] pt-[clamp(28px,5vh,64px)] pb-[15vh]">
        <header>
          <p className="mb-6 text-[13px] text-haze">Agentverse 2026, Robotics track</p>
          <Wordmark g={cfg.gravity_g} min={options.gravity.min} max={options.gravity.max} />
          <p className="font-semi-cond mt-5 max-w-[34ch] text-[22px] leading-snug font-[500] text-bone">
            An autonomous rover and aerial scout map a Super-Earth sector, thinking entirely on their own edge hardware.
          </p>
        </header>
        <div className="max-w-[640px]">
          <GravityControl
            g={cfg.gravity_g}
            min={options.gravity.min}
            max={options.gravity.max}
            step={options.gravity.step}
            onChange={(gravity_g) => update({ gravity_g })}
            preview={preview}
            live={live}
          />
        </div>
      </section>

      <section className="relative z-10 flex min-h-0 flex-col gap-[clamp(14px,2.4vh,24px)] overflow-y-auto border-l border-rule bg-void/70 px-[clamp(24px,3.2vw,56px)] pt-[clamp(28px,5vh,64px)] pb-8 backdrop-blur-[2px] scroll-thin max-lg:border-l-0 max-lg:border-t">
        <div className="flex items-baseline justify-between gap-4">
          <h2 className="font-cond text-[30px] leading-none font-[800] text-bone">Configure the mission</h2>
          <ConnectionNote live={optionsLive} conn={conn} />
        </div>

        <Field title="Sector size" hint="Square region the fleet must map">
          <SectorPicker sectors={options.sectors} value={cfg.sector} onChange={(sector) => update({ sector })} />
        </Field>

        <Field title="Fleet">
          <div className="grid grid-cols-2 gap-2.5">
            <Stepper label="Ground rovers" glyph="rover" unit={['rover', 'rovers']} range={options.fleet.rovers} value={cfg.fleet.rovers} onChange={(rovers) => update({ fleet: { ...cfg.fleet, rovers } })} />
            <Stepper label="Aerial scouts" glyph="scout" unit={['scout', 'scouts']} range={options.fleet.drones} value={cfg.fleet.drones} onChange={(drones) => update({ fleet: { ...cfg.fleet, drones } })} />
          </div>
        </Field>

        <Field title="Edge compute" hint="Simulated onboard computer: sets latency, power and planner budget">
          <EdgePicker profiles={options.edge_profiles} value={cfg.edge_profile} onChange={(edge_profile) => update({ edge_profile })} />
        </Field>

        <Field title="Simulation">
          <BackendPicker backends={options.backends} value={cfg.sim_backend} onChange={(sim_backend) => update({ sim_backend })} />
          <div className="mt-3 flex flex-wrap items-center gap-x-6 gap-y-2">
            <label className="flex items-center gap-2 text-[13px] text-haze">
              Terrain seed
              <input
                type="number"
                value={cfg.seed}
                onChange={(e) => update({ seed: Math.max(0, Math.floor(Number(e.target.value) || 0)) })}
                className="num w-24 rounded-[3px] border border-rule bg-bay px-2 py-1.5 text-bone outline-none focus:border-hb"
              />
            </label>
            <button type="button" className="text-[13px] text-hb hover:text-bone" onClick={() => update({ seed: Math.floor(Math.random() * 10000) })}>
              New terrain
            </button>
            <label className={`flex items-center gap-2 text-[13px] ${cfg.sim_backend === 'gazebo' ? 'text-haze' : 'text-dim'}`}>
              <input
                type="checkbox"
                className="h-4 w-4 accent-[#4cc3ff]"
                checked={cfg.gazebo_gui}
                disabled={cfg.sim_backend !== 'gazebo'}
                onChange={(e) => update({ gazebo_gui: e.target.checked })}
              />
              Open the Gazebo window too
            </label>
          </div>
        </Field>

        {/* the deploy action stays pinned in view when the form is taller than the screen */}
        <div className="sticky bottom-0 -mx-[clamp(24px,3.2vw,56px)] mt-auto -mb-8 bg-[rgba(22,19,43,0.94)] px-[clamp(24px,3.2vw,56px)] pt-3 pb-8 before:pointer-events-none before:absolute before:inset-x-0 before:-top-8 before:h-8 before:bg-[linear-gradient(to_top,rgba(22,19,43,0.94),rgba(22,19,43,0))] before:content-['']">
          <p className="mb-3 max-w-[70ch] text-[13px] text-haze">{summary}</p>
          <button
            type="button"
            onClick={deploy}
            disabled={deploying}
            className="font-cond group relative flex h-[68px] w-full items-center justify-between overflow-hidden rounded-[3px] bg-bone px-7 text-[30px] font-[850] text-void transition-[background-color] hover:bg-white disabled:opacity-70"
          >
            <span>{deploying ? 'Deploying fleet…' : 'Deploy fleet'}</span>
            <DeployGlyph />
          </button>
          {error && (
            <p role="alert" className="mt-3 text-[13px] text-ha">
              Deployment failed: {error}. Check that the backend is running on port 8000 (./start_all.sh), then deploy again.
            </p>
          )}
        </div>
      </section>
    </main>
  )
}

function ConnectionNote({ live, conn }: { live: boolean; conn: string }) {
  if (live && conn === 'open') return <span className="text-[12px] text-o3">Backend connected</span>
  if (live) return <span className="text-[12px] text-haze">Backend reachable, live link reconnecting</span>
  return <span className="text-[12px] text-nad">Backend offline: showing built-in options</span>
}

function DeployGlyph() {
  return (
    <svg width="44" height="24" viewBox="0 0 44 24" aria-hidden className="transition-transform group-hover:translate-x-1">
      <path d="M2 12 H36 M28 4 L37 12 L28 20" fill="none" stroke="currentColor" strokeWidth="3" />
    </svg>
  )
}

/** The planet's horizon behind the hero; it flattens and glows hotter as gravity rises. */
function PlanetLimb({ g }: { g: number }) {
  const t = Math.max(0, Math.min(1, (g - 1) / 2))
  return (
    <div aria-hidden className="pointer-events-none absolute inset-0 z-0 overflow-hidden">
      <div
        className="absolute left-[-30%] h-[200vh] w-[105%] rounded-[50%]"
        style={{
          top: `${88 - 3 * t}vh`,
          background: 'radial-gradient(ellipse at 50% 0%, #6a2a1c 0%, #3a1a22 22%, #1e1530 48%, #16132b 70%)',
          boxShadow: `0 -2px 0 0 rgba(255,74,92,${0.35 + 0.3 * t}), 0 -30px 90px -10px rgba(181,72,42,${0.25 + 0.25 * t})`,
          transition: 'top 300ms ease-out, box-shadow 300ms ease-out',
        }}
      />
    </div>
  )
}
