import { useState, type FormEvent, type ReactNode } from 'react'
import { useShallow } from 'zustand/react/shallow'
import { api } from '../lib/api'
import {
  actionText,
  advisorOf,
  brainChips,
  brainCounters,
  commandOutcome,
  commandPhase,
  COMMAND_TIMEOUT_MS,
  effectOf,
  effectRobot,
  eventLabel,
  fmtMs,
  fmtTarget,
  fmtTokens,
  sourceTag,
  verdictOf,
  type Tone,
} from '../lib/brain'
import { fmtClock, robotName, robotShort } from '../lib/format'
import { useTick } from '../lib/hooks'
import { isLiveStatus, sortRobots, useStore, type SentCommand } from '../lib/store'
import { BRAIN_CHIP_COLOR, C } from '../lib/theme'
import type { BrainDecision } from '../lib/types'
import { Panel, SensorChip } from '../ui/Panel'

/** fleet_brain_node routes a command without a robot to the scout rover */
const DEFAULT_COMMAND_ROBOT = 'rover_0'
const TONE: Record<Tone, string> = { ok: C.o3, info: C.hb, warn: C.nad, crit: C.ha }
/**
 * Demo shortcuts: a safe slow-down, a stop, lifting it again, and a request the local check must reject. The button
 * shows a short label; the full text (shown in its tooltip and, once sent, in the command list) is what is sent.
 */
const PRESETS: { label: string; text: string }[] = [
  { label: 'slow 0.2 m/s', text: 'slow down to 0.2 m/s' },
  { label: 'stop', text: 'stop' },
  { label: 'resume', text: 'resume at full speed' },
  { label: 'full speed → rockslide', text: 'drive at full speed into the rockslide' },
]
/** A drone takes altitude commands only (anything else is answered with a clarification request). */
const DRONE_PRESETS: { label: string; text: string }[] = [
  { label: 'drone 8 m', text: 'drone to 8 m AGL' },
  { label: 'drone higher', text: 'drone higher' },
  { label: 'drone lower', text: 'drone lower' },
]

/**
 * The advisory Gemma brain, told honestly: who has authority (always the on-board safety layer), who advises (the real
 * cloud model, or the local rules), the latest decision with its MEASURED cloud round-trip and token counts, the
 * module's labelled on-device projection, live counters, active caps, and the operator command box.
 *
 * Laid out as a wide strip (decision │ measurements │ operator command) so it sits under the globe without taking
 * height from the decision trace; it goes to two columns below ~900 px and stacks below ~600 px.
 */
export function BrainHud() {
  const latest = useStore((s) => s.brainDecisions[0] ?? null)
  const [open, setOpen] = useState(readOpen)
  const toggle = () => {
    setOpen(!open)
    writeOpen(!open)
  }
  return (
    <Panel
      title="Agent brain"
      right={
        <>
          <AuthorityLine />
          <button
            type="button"
            onClick={toggle}
            aria-expanded={open}
            title={open ? 'Collapse the agent brain panel' : 'Expand the agent brain panel'}
            className="ml-1 grid h-5 w-5 place-items-center rounded-[2px] text-dim hover:bg-bay-2 hover:text-bone"
          >
            <svg width="10" height="10" viewBox="0 0 10 10" aria-hidden className={open ? '' : 'rotate-180'}>
              <path d="M1 3.5 L5 7 L9 3.5" fill="none" stroke="currentColor" strokeWidth="1.5" />
            </svg>
          </button>
        </>
      }
      bodyClass={open ? 'min-h-0' : 'hidden'}
    >
      {/* fixed height from 600 px: each column scrolls on its own; below 900 px decision and measurements share one */}
      <div className="grid grid-cols-1 @min-[600px]:h-[206px] @min-[600px]:grid-cols-[minmax(0,1.3fr)_minmax(0,1fr)] @min-[600px]:grid-rows-[minmax(0,1fr)] @min-[900px]:grid-cols-[minmax(0,2.05fr)_minmax(0,1.1fr)]">
        <div className="scroll-thin grid min-h-0 grid-cols-1 content-start overflow-y-auto @min-[900px]:content-stretch @min-[900px]:grid-cols-[minmax(0,1.1fr)_minmax(0,0.95fr)] @min-[900px]:grid-rows-[minmax(0,1fr)] @min-[900px]:overflow-visible">
          <div className="scroll-thin flex flex-col gap-1.5 px-3 py-2 @min-[900px]:min-h-0 @min-[900px]:overflow-y-auto">
            {latest ? <LatestDecision d={latest} /> : <EmptyDecision />}
            <ActiveCaps />
            <Counters />
          </div>
          <div className="scroll-thin flex flex-col gap-1.5 border-t border-rule/70 px-3 py-2 @min-[900px]:min-h-0 @min-[900px]:overflow-y-auto @min-[900px]:border-t-0 @min-[900px]:border-l">
            <Measurements d={latest} />
            <LastError />
          </div>
        </div>
        <div className="scroll-thin min-h-0 overflow-y-auto border-t border-rule/70 @min-[600px]:border-t-0 @min-[600px]:border-l">
          <CommandBox />
        </div>
      </div>
    </Panel>
  )
}

// collapsed / expanded is a per-viewer convenience: storage may be unavailable (private window), so never rely on it
const OPEN_KEY = 'redshift.brainHud.open'
function readOpen(): boolean {
  try {
    return localStorage.getItem(OPEN_KEY) !== '0'
  } catch {
    return true
  }
}
function writeOpen(open: boolean) {
  try {
    localStorage.setItem(OPEN_KEY, open ? '1' : '0')
  } catch {
    /* ignore */
  }
}

/** "Authority on-board safety layer (local) │ Advisor gemma-… · Google AI cloud" in the panel header. */
function AuthorityLine() {
  const status = useStore((s) => s.telemetry?.mission.brain ?? null)
  const option = useStore((s) => s.options.brain)
  const cfgOn = useStore((s) => (s.telemetry?.mission.config ?? s.lastConfig)?.ai_brain)
  const adv = advisorOf(status, option, cfgOn)
  // a stale status is the node's last report, not now: no back-off figure from it
  const backoff = adv.mode === 'stale' ? 0 : (status?.backoff_s ?? 0)
  return (
    <span className="flex items-baseline gap-2.5 text-[12px] whitespace-nowrap">
      <span>
        <span className="text-dim">Authority </span>
        <span className="text-bone">on-board safety layer (local)</span>
      </span>
      <span className="text-rule-2" aria-hidden>
        │
      </span>
      <span title={adv.text}>
        <span className="text-dim">Advisor </span>
        {adv.mode === 'cloud' ? (
          <>
            <span className="font-mono text-[11.5px] text-bone [font-stretch:82%]">{adv.model}</span>
            <span className="text-haze"> · Google AI cloud</span>
          </>
        ) : (
          <span style={{ color: adv.mode === 'no-key' || adv.mode === 'stale' ? C.nad : C.haze }}>{adv.text}</span>
        )}
      </span>
      {backoff > 0 && <span className="num text-nad">back-off {Math.ceil(backoff)} s</span>}
    </span>
  )
}

function EmptyDecision() {
  return (
    <p className="text-[12px] leading-snug text-dim">
      No brain decisions yet. The advisor is asked when a rover meets a novel hazard, loses perception confidence, is damaged or hears an
      SOS, and on every operator command; the local safety layer has already acted by then.
    </p>
  )
}

function LatestDecision({ d }: { d: BrainDecision }) {
  const target = fmtTarget(d)
  const effect = useStore((s) => {
    const id = effectRobot(d)
    return effectOf(id ? s.telemetry?.robots[id] : undefined, d)
  })
  const v = verdictOf(d, effect)
  const actionColor = d.action === 'REJECT_UNSAFE_COMMAND' || d.action === 'REQUEST_CLARIFICATION' ? C.nad : TONE[v.tone]
  return (
    <div className="flex flex-col gap-1">
      <div className="flex min-w-0 items-center gap-2">
        <SensorChip tag={sourceTag(d.source)} size="md" />
        <span className="font-cond shrink-0 text-[13px] font-[780] tracking-[0.03em] text-bone" title={d.robot ? robotName(d.robot) : 'fleet'}>
          {d.robot ? robotShort(d.robot) : 'FLEET'}
        </span>
        <span className="min-w-0 truncate text-[12.5px] text-haze">{eventLabel(d.event)}</span>
        <time className="num ml-auto shrink-0 font-mono text-[10.5px] text-dim [font-stretch:80%]">{fmtClock(d.t)}</time>
      </div>
      <div className="flex min-w-0 flex-wrap items-baseline gap-x-2 gap-y-1">
        <span className="font-cond text-[15px] leading-tight font-[800] tracking-[0.02em]" style={{ color: actionColor }}>
          → {actionText(d.action)}
        </span>
        {target && <span className="num text-[12.5px] font-[650] text-bone">{target}</span>}
        {d.action === 'DEPLOY_RECOVERY_ROVER' && d.target_robot && <span className="text-[12px] text-haze">{robotShort(d.target_robot)}</span>}
        {d.immediate && (effect === 'applied' ? <Tag color={C.ha}>immediate local stop</Tag> : <Tag color={C.dim}>local stop sent, not confirmed</Tag>)}
        {d.stale && <Tag color={C.nad}>late answer</Tag>}
      </div>
      {d.command_text && (
        <p className="truncate font-mono text-[11px] text-haze [font-stretch:82%]" title={d.command_text}>
          operator: “{d.command_text}”
        </p>
      )}
      <p className="line-clamp-3 text-[12px] leading-snug text-haze" title={d.reasoning_trace}>
        {d.reasoning_trace}
      </p>
      {d.error && (
        <p className="truncate text-[11.5px] text-nad" title={d.error}>
          Advisor answer not used: {d.error}
        </p>
      )}
      {d.adjusted.length > 0 && (
        <p className="truncate text-[11.5px] text-dim" title={d.adjusted.join('\n')}>
          Local layer adjusted: {d.adjusted.join('; ')}
        </p>
      )}
    </div>
  )
}

/** The latest decision's real numbers: measured cloud round-trip, token counts, the deciding model, the projection. */
function Measurements({ d }: { d: BrainDecision | null }) {
  return (
    <div className="flex flex-col gap-1.5 text-[11.5px]">
      <div>
        <div className="flex items-baseline justify-between gap-2">
          <span className="text-haze">Cloud round-trip</span>
          <span className="font-semi-cond num text-[16px] leading-none font-[750] text-bone">
            {d?.latency_ms == null ? <span className="text-[12px] font-[400] text-dim">{d ? 'no cloud call' : '—'}</span> : fmtMs(d.latency_ms)}
          </span>
        </div>
        <div className="text-[10.5px] text-dim">measured on this host → Google AI cloud → back</div>
      </div>
      <dl className="grid grid-cols-[auto_minmax(0,1fr)] gap-x-3 gap-y-0.5">
        <dt className="text-dim">Tokens</dt>
        <dd className="num text-right text-bone">{d ? fmtTokens(d.tokens_in, d.tokens_out) : '—'}</dd>
        <dt className="text-dim">Decided by</dt>
        <dd className="truncate text-right font-mono text-[11px] text-bone [font-stretch:82%]" title={d?.model}>
          {d?.model ?? '—'}
        </dd>
      </dl>
      {d?.projected_note && (
        <div className="rounded-[2px] border border-dashed border-rule-2 px-2 py-1">
          <div className="text-[10.5px] text-dim">On device: projected (estimate), not measured</div>
          <p className="text-[11px] leading-snug text-haze">{d.projected_note}</p>
        </div>
      )}
    </div>
  )
}

function Counters() {
  const stats = useStore((s) => s.telemetry?.mission.brain?.stats)
  const stale = useStore((s) => s.telemetry?.mission.brain?.stale === true)
  const counters = brainCounters(stats, stale)
  return (
    <div className="mt-auto border-t border-rule/50 pt-1.5" title={stale ? 'Last values reported before the brain node stopped reporting' : undefined}>
      <div className="grid grid-cols-4 gap-2">
        {counters.map((c) => (
          <div key={c.label} className="min-w-0">
            <div
              className="font-semi-cond num text-[15px] leading-tight font-[750]"
              style={{ color: c.muted ? C.dim : c.label === 'rate-limited' && (c.value ?? 0) > 0 ? C.nad : C.bone }}
            >
              {c.value ?? '—'}
            </div>
            <div className="text-[10.5px] leading-tight text-dim">{c.label}</div>
          </div>
        ))}
      </div>
    </div>
  )
}

/** /brain/status last_error: why the latest cloud call could not be used (never contains the key). */
function LastError() {
  const lastError = useStore((s) => s.telemetry?.mission.brain?.last_error ?? null)
  if (!lastError) return null
  return (
    <p className="mt-auto line-clamp-2 text-[11px] leading-snug text-nad" title={lastError}>
      Last error: {lastError}
    </p>
  )
}

/**
 * Caps and altitudes the robots report (agent_state brain_cap / brain_agl): honoured ones in colour, ones the robot
 * does not report as applied muted and marked "pending".
 */
function ActiveCaps() {
  const rows = useStore(
    useShallow((s) => {
      const robots = s.telemetry?.robots ?? {}
      const out: string[] = []
      for (const id of sortRobots(Object.keys(robots))) {
        for (const chip of brainChips(robots[id])) out.push([id, chip.tone, chip.text].join('\u0000'))
      }
      return out
    }),
  )
  if (!rows.length) return null
  return (
    <ul className="flex flex-wrap gap-x-4 gap-y-0.5 text-[11.5px]" aria-label="Brain caps reported by the robots">
      {rows.map((row) => {
        const [id, tone, text] = row.split('\u0000') as [string, keyof typeof BRAIN_CHIP_COLOR, string]
        return (
          <li key={row} className="flex items-center gap-1.5 whitespace-nowrap">
            <span className="font-cond text-[12.5px] font-[780] tracking-[0.03em] text-bone">{robotShort(id)}</span>
            <span className="num" style={{ color: BRAIN_CHIP_COLOR[tone] }}>
              {text}
            </span>
          </li>
        )
      })}
    </ul>
  )
}

function CommandBox() {
  const robot = useStore((s) => s.selectedRobot)
  const target = robot ?? DEFAULT_COMMAND_ROBOT
  const drone = useStore((s) => (s.telemetry?.robots[target]?.type ?? (target.startsWith('scout') ? 'scout' : 'rover')) === 'scout')
  const status = useStore((s) => s.telemetry?.mission.status ?? s.status?.status)
  const commands = useStore((s) => s.commands)
  const live = isLiveStatus(status)
  const presets = drone ? DRONE_PRESETS : PRESETS
  const [text, setText] = useState('')
  const [busy, setBusy] = useState(false)
  const [error, setError] = useState<string | null>(null)

  const send = async (raw: string) => {
    // same normalisation as the backend (runs of whitespace → one space, 1–200 chars) so the list shows what was sent
    const t = raw.split(/\s+/).filter(Boolean).join(' ').slice(0, 200)
    if (!t || busy) return
    setBusy(true)
    setError(null)
    try {
      const res = await api.command(t, robot)
      if (!res?.ok || !res.id) throw new Error(res?.detail ?? 'Backend refused the command')
      useStore.getState().noteCommand({ id: res.id, text: t, robot, sentAt: performance.now() })
      setText('')
    } catch (e) {
      setError((e as Error).message)
    } finally {
      setBusy(false)
    }
  }
  const submit = (e: FormEvent) => {
    e.preventDefault()
    void send(text)
  }

  return (
    <div className="flex flex-col gap-1.5 px-3 py-2">
      <div className="flex items-baseline justify-between gap-2 text-[12px]">
        <span className="text-haze">Operator command</span>
        <span className="truncate text-dim" title={`To ${robotName(target)}. Checked by the local safety rules first; the advisor can only make it safer`}>
          to <span className="font-cond text-[12.5px] font-[750] tracking-[0.03em] text-bone">{robotShort(target)}</span>, checked locally
        </span>
      </div>
      {drone && <p className="text-[11px] leading-snug text-haze">Drones take altitude commands only, e.g. “drone to 8 m AGL”; speed and stop commands go to a rover.</p>}
      <form onSubmit={submit} className="flex gap-1.5">
        <input
          value={text}
          onChange={(e) => setText(e.target.value)}
          maxLength={200}
          disabled={!live}
          placeholder={live ? (drone ? 'e.g. drone to 8 m AGL' : 'e.g. slow down to 0.2 m/s') : 'Available while a mission runs'}
          aria-label={`Operator command to ${robotName(target)}`}
          className="min-w-0 flex-1 rounded-[3px] border border-rule bg-void/60 px-2 py-1.5 text-[12.5px] text-bone outline-none placeholder:text-dim focus:border-hb disabled:opacity-50"
        />
        <button
          type="submit"
          disabled={!live || busy || !text.trim()}
          className="shrink-0 rounded-[3px] border border-hb/70 px-3 text-[12.5px] font-[700] text-hb transition-colors hover:bg-hb/10 disabled:border-rule disabled:text-dim"
        >
          {busy ? 'Sending…' : 'Send'}
        </button>
      </form>
      <div className="scroll-thin flex gap-1 overflow-x-auto pb-0.5">
        {presets.map((p) => (
          <button
            key={p.text}
            type="button"
            disabled={!live || busy}
            onClick={() => void send(p.text)}
            title={`Send “${p.text}”`}
            className="shrink-0 rounded-[2px] border border-rule px-1 py-0.5 font-mono text-[10.5px] whitespace-nowrap text-haze [font-stretch:82%] hover:border-rule-2 hover:text-bone disabled:opacity-40"
          >
            {p.label}
          </button>
        ))}
      </div>
      {error && (
        <p role="alert" className="text-[11.5px] text-ha">
          Not sent: {error}
        </p>
      )}
      {commands.length > 0 && (
        <ol className="flex flex-col gap-1" aria-live="polite" aria-label="Recent operator commands">
          {commands.slice(0, 2).map((c) => (
            <CommandRow key={c.id} c={c} />
          ))}
        </ol>
      )}
    </div>
  )
}

/** One sent command and what the brain made of it: the instant local stop, then the advisor's (or fallback's) answer. */
function CommandRow({ c }: { c: SentCommand }) {
  useTick(1000)
  const outcome = useStore(useShallow((s) => commandOutcome(s.brainDecisions, c.id)))
  const now = performance.now()
  const phase = commandPhase(outcome, c.sentAt, now)
  const shown = outcome.final ?? outcome.immediate
  const sentTo = c.robot ?? DEFAULT_COMMAND_ROBOT
  // what the target robot's own telemetry confirms: nothing is reported as stopped / capped / in effect without it
  const target = shown ? (effectRobot(shown, sentTo) ?? sentTo) : sentTo
  const effect = useStore((s) => (shown ? effectOf(s.telemetry?.robots[target], shown) : 'none'))
  const stopTarget = outcome.immediate ? (effectRobot(outcome.immediate, sentTo) ?? sentTo) : sentTo
  const stopNow = useStore((s) => (outcome.immediate ? effectOf(s.telemetry?.robots[stopTarget], outcome.immediate) : 'none'))
  // the stop is past tense once the rover confirmed it: keep saying so after its cap expires or Gemma replaces it
  const [stopSeen, setStopSeen] = useState(false)
  if (stopNow === 'applied' && !stopSeen) setStopSeen(true)
  const stopConfirmed = stopSeen || stopNow === 'applied'
  const v = shown ? verdictOf(shown, shown.immediate && stopConfirmed ? 'applied' : effect) : null
  const explain = outcome.final && (outcome.final.action === 'REJECT_UNSAFE_COMMAND' || outcome.final.action === 'REQUEST_CLARIFICATION')
  return (
    <li className="rounded-[2px] bg-void/45 px-2 py-1" title={`${c.id}${c.robot ? ` → ${c.robot}` : ''}`}>
      <div className="flex min-w-0 items-center gap-1.5">
        <span className="min-w-0 truncate font-mono text-[11px] text-bone [font-stretch:82%]">“{c.text}”</span>
        <span className="ml-auto flex shrink-0 items-center gap-1.5 text-[11px]">
          {shown?.latency_ms != null && <span className="num text-dim">{fmtMs(shown.latency_ms)}</span>}
          {shown && <SensorChip tag={sourceTag(shown.source)} />}
        </span>
      </div>
      <div className="mt-1 flex min-w-0 items-center gap-1.5 text-[11px]">
        {outcome.immediate &&
          outcome.final &&
          (stopConfirmed ? <Tag color={C.ha}>stopped at once</Tag> : <Tag color={C.dim}>local stop not confirmed</Tag>)}
        {v && <Tag color={TONE[v.tone]}>{v.label}</Tag>}
        {outcome.final && effect === 'applied' && <span className="shrink-0 text-o3">in effect on {robotShort(target)}</span>}
        {outcome.final && effect === 'pending' && (
          <span className="shrink-0 text-dim" title={`${robotName(target)} reports it but is not applying it right now`}>
            pending on {robotShort(target)}
          </span>
        )}
        {phase === 'pending' && (
          <span className="flex shrink-0 items-center gap-1 text-haze">
            <span className="animate-blink h-1.5 w-1.5 rounded-full bg-hb" />
            {outcome.immediate ? 'advisor answering' : 'waiting for the brain'} · {Math.floor((now - c.sentAt) / 1000)} s
          </span>
        )}
        {phase === 'unanswered' && <span className="shrink-0 text-nad">no decision within {COMMAND_TIMEOUT_MS / 1000} s</span>}
        {explain && (
          <span className="min-w-0 truncate text-haze" title={outcome.final!.reasoning_trace}>
            {outcome.final!.reasoning_trace}
          </span>
        )}
      </div>
    </li>
  )
}

function Tag({ color, children }: { color: string; children: ReactNode }) {
  return (
    <span
      className="font-cond inline-flex items-center rounded-[2px] px-1.5 py-[3px] text-[11.5px] leading-none font-[750] tracking-[0.02em] whitespace-nowrap"
      style={{ color, boxShadow: `inset 0 0 0 1px ${color}66`, backgroundColor: `${color}14` }}
    >
      {children}
    </span>
  )
}
