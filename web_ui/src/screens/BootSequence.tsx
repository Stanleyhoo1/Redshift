import { useEffect, useMemo, useRef, useState } from 'react'
import { api } from '../lib/api'
import { checklistLabel } from '../lib/format'
import { isLiveStatus, useStore } from '../lib/store'

export default function BootSequence() {
  const status = useStore((s) => s.status)
  const bootLog = useStore((s) => s.bootLog)
  const lastConfig = useStore((s) => s.lastConfig)
  const hasRobots = useStore((s) => !!s.telemetry && Object.keys(s.telemetry.robots).length > 0)
  const options = useStore((s) => s.options)
  const [busy, setBusy] = useState(false)
  const [actionError, setActionError] = useState<string | null>(null)
  const logRef = useRef<HTMLDivElement>(null)
  const startRef = useRef(performance.now())

  const st = status?.status
  const failed = st === 'failed'
  const notRunning = st === 'stopped' || st === 'idle'
  const ready = hasRobots && isLiveStatus(st)

  // auto-transition once the first robot telemetry arrives
  useEffect(() => {
    if (!ready) return
    const t = setTimeout(() => useStore.getState().setScreen('mission'), 900)
    return () => clearTimeout(t)
  }, [ready])

  useEffect(() => {
    const el = logRef.current
    if (el) el.scrollTop = el.scrollHeight
  }, [bootLog.length])

  const checklist = useMemo(() => Object.entries(status?.checklist ?? {}), [status?.checklist])
  const done = checklist.filter(([, v]) => v).length
  const t0 = bootLog[0]?.t

  const retry = async () => {
    if (!lastConfig) return
    setBusy(true)
    setActionError(null)
    try {
      await api.abort().catch(() => {})
      const res = await api.deploy(lastConfig)
      if (!res.ok) throw new Error(res.detail ?? 'Backend refused the deployment')
      useStore.getState().beginDeploy(lastConfig)
      startRef.current = performance.now()
    } catch (e) {
      setActionError((e as Error).message)
    } finally {
      setBusy(false)
    }
  }
  const back = async () => {
    setBusy(true)
    await api.abort().catch(() => {})
    setBusy(false)
    useStore.getState().setScreen('prelaunch')
  }

  const sector = options.sectors.find((s) => s.id === lastConfig?.sector)
  const headline = failed ? 'Launch failed' : ready ? 'Fleet online' : notRunning ? 'Mission is not running' : 'Launching the fleet'
  const tone = failed ? 'text-ha' : ready ? 'text-o3' : 'text-bone'

  return (
    <main className="grid h-full grid-cols-[minmax(0,1fr)_420px] gap-0 max-lg:grid-cols-1">
      <section className="flex min-h-0 flex-col px-[clamp(24px,4vw,64px)] py-[clamp(24px,5vh,56px)]">
        <header className="mb-6 flex items-end justify-between gap-6">
          <div>
            <p className="text-[13px] text-haze">
              {sector ? `${sector.label} sector, ${sector.size_m} m` : 'Mission'}
              {lastConfig ? `, ${lastConfig.gravity_g.toFixed(1)} g, seed ${lastConfig.seed}` : ''}
            </p>
            <h1 className={`font-cond mt-1 text-[64px] leading-none font-[850] ${tone}`}>{headline}</h1>
          </div>
          <p className="num pb-1 text-right text-[13px] text-haze">
            {status?.backend ? `Backend: ${status.backend === 'gazebo' ? 'Gazebo Harmonic' : status.backend === 'mock' ? 'Mock physics' : status.backend}` : 'Waiting for backend status'}
          </p>
        </header>

        <div
          ref={logRef}
          role="log"
          aria-live="polite"
          className="scroll-thin min-h-0 flex-1 overflow-y-auto rounded-[3px] border border-rule bg-[#120f24] px-5 py-4 font-mono text-[12.5px] leading-[1.7] [font-stretch:87%]"
        >
          {bootLog.length === 0 && <p className="text-dim">Waiting for launch output…</p>}
          {bootLog.map((l, i) => (
            <p key={i} className={`whitespace-pre-wrap break-words ${lineTone(l.line)}`}>
              <span className="mr-3 text-dim select-none">{t0 !== undefined ? `+${(l.t - t0).toFixed(1).padStart(5, ' ')}s` : ''}</span>
              {l.line}
            </p>
          ))}
          {!failed && !ready && <span className="inline-block h-[1em] w-[0.6em] translate-y-[2px] animate-blink bg-hb" />}
        </div>

        {(failed || notRunning || actionError) && (
          <div role="alert" className="mt-5 flex flex-wrap items-center gap-4 rounded-[3px] border border-ha/60 bg-ha/10 px-5 py-4">
            <p className="min-w-[30ch] flex-1 text-[14px] text-bone">
              {actionError ?? status?.detail ?? (notRunning ? 'The backend reports no active mission.' : 'A subsystem did not come up.')}
            </p>
            <button type="button" disabled={busy || !lastConfig} onClick={retry} className="font-semi-cond rounded-[3px] bg-bone px-5 py-2.5 text-[16px] font-[750] text-void disabled:opacity-50">
              Retry deploy
            </button>
            <button type="button" disabled={busy} onClick={back} className="font-semi-cond rounded-[3px] border border-rule-2 px-5 py-2.5 text-[16px] font-[650] text-bone hover:border-bone">
              Abort and reconfigure
            </button>
          </div>
        )}
      </section>

      <aside className="flex min-h-0 flex-col border-l border-rule bg-bay/60 px-8 py-[clamp(24px,5vh,56px)]">
        <h2 className="text-[15px] font-medium text-haze">Subsystems</h2>
        <div className="mt-2 flex items-baseline gap-2">
          <span className="font-cond num text-[44px] leading-none font-[850]">{done}</span>
          <span className="text-[15px] text-haze">of {checklist.length || '…'} ready</span>
        </div>
        <div className="mt-4 h-1 rounded-full bg-rule">
          <div className="h-1 rounded-full bg-o3 transition-[width] duration-500" style={{ width: `${checklist.length ? (done / checklist.length) * 100 : 0}%` }} />
        </div>
        <ul className="mt-6 flex flex-col">
          {checklist.map(([k, ok]) => (
            <li key={k} className="flex items-center justify-between gap-3 border-b border-rule/70 py-3">
              <span className={ok ? 'text-bone' : 'text-haze'}>{checklistLabel(k)}</span>
              <CheckMark ok={ok} failed={failed && !ok} />
            </li>
          ))}
        </ul>
        {!failed && !ready && (
          <p className="mt-auto text-[13px] text-dim">Mission control opens automatically when the first robot reports in.</p>
        )}
        {!failed && !ready && (
          <button type="button" onClick={back} disabled={busy} className="mt-4 self-start text-[13px] text-haze underline-offset-4 hover:text-bone hover:underline">
            Abort launch
          </button>
        )}
      </aside>
    </main>
  )
}

function lineTone(line: string): string {
  const l = line.toLowerCase()
  if (/\b(error|failed|fatal|traceback)\b/.test(l)) return 'text-ha'
  if (/\b(warn|warning|fallback|timeout)\b/.test(l)) return 'text-nad'
  if (/\b(ready|ok|online|started|up)\b/.test(l)) return 'text-bone'
  return 'text-haze'
}

function CheckMark({ ok, failed }: { ok: boolean; failed: boolean }) {
  if (ok)
    return (
      <svg width="18" height="18" viewBox="0 0 18 18" aria-label="ready" className="text-o3">
        <path d="M3 9.5 L7.5 14 L15 4.5" fill="none" stroke="currentColor" strokeWidth="2.2" />
      </svg>
    )
  if (failed)
    return (
      <svg width="18" height="18" viewBox="0 0 18 18" aria-label="failed" className="text-ha">
        <path d="M4 4 L14 14 M14 4 L4 14" stroke="currentColor" strokeWidth="2.2" />
      </svg>
    )
  return <span aria-label="pending" className="block h-3.5 w-3.5 animate-spin rounded-full border-2 border-rule-2 border-t-hb" />
}
