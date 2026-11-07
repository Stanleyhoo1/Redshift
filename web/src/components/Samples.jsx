import { memo, useMemo } from 'react'
import { appStore, useApp, useSim } from '../state/store.js'
import { instrumentStyle, kindStyle } from '../lib/theme.js'
import { InstrumentIcon, ValueBar } from './SampleBits.jsx'

// v4b: every sample the fleet has taken, best first: kind, instrument, the on-board
// analysis, its science value, and where the tube is (in a rover or cached at the lander).
// Rows fly the camera to the target and keep its value tag up. Samples still being taken
// sit at the top with their progress.

const SampleRow = memo(function SampleRow({ t, carrier, picked, onPick }) {
  const ks = kindStyle(t.kind)
  const ins = instrumentStyle(t.instrument)
  const v = Number.isFinite(t.value) ? t.value : null
  return (
    <li className="smp" data-picked={picked || undefined} data-cached={t.cached || undefined}>
      <button type="button" className="smp-btn" onClick={() => onPick(t)} title={`Show the ${ks.long} ${t.id} on the map`}>
        <span className="smp-top">
          <i className="sw-diamond" style={{ '--c': ks.color }} />
          <span className="smp-kind">
            {ks.label} <span className="smp-id">{t.id}</span>
          </span>
          {t.instrument && (
            <span className="smp-ins">
              <InstrumentIcon kind={t.instrument} />
              {ins.label}
            </span>
          )}
          <span className="smp-v" data-none={v === null || undefined}>
            {v === null ? '--' : v}
            <small>/10</small>
          </span>
        </span>
        <span className="smp-a">{t.analysis || 'Analysing on board...'}</span>
        <span className="smp-foot">
          <ValueBar value={v} />
          {t.cached ? (
            <span className="smp-where" data-cached>
              <svg viewBox="0 0 12 12" aria-hidden="true">
                <path d="M6 1.2 10.6 3.8v4.4L6 10.8 1.4 8.2V3.8z" />
              </svg>
              cached at lander
            </span>
          ) : carrier ? (
            <span className="smp-where">
              <span className="dot" style={{ background: carrier.color }} />
              carried by {carrier.name}
            </span>
          ) : (
            <span className="smp-where">in transit</span>
          )}
        </span>
      </button>
    </li>
  )
})

function InProgress({ rover, meta, target }) {
  const ins = instrumentStyle(rover.sampling.instrument)
  const ks = kindStyle(target?.kind)
  const pct = Math.round(rover.sampling.progress * 100)
  return (
    <li className="smp smp-live" style={{ '--c': ks.color }}>
      <span className="smp-top">
        <InstrumentIcon kind={rover.sampling.instrument} />
        <span className="smp-kind">
          {ins.verb} <span className="smp-id">{target ? `${ks.long} ${target.id}` : rover.sampling.target}</span>
        </span>
        <span className="smp-pct">{pct}%</span>
      </span>
      <span className="smp-foot">
        <span className="sampling-bar smp-prog" aria-hidden="true">
          <i style={{ width: `${pct}%` }} />
        </span>
        <span className="smp-where">
          <span className="dot" style={{ background: meta?.color }} />
          {meta?.name ?? rover.id}
        </span>
      </span>
    </li>
  )
}

export function SamplesPanel({ hidden, onFocusEvent }) {
  const targets = useSim((s) => s.targets)
  const rovers = useSim((s) => s.rovers)
  const meta = useSim((s) => s.roverMeta)
  const fleet = useSim((s) => s.fleet)
  const picked = useApp((a) => a.pickedTarget)
  const byId = useMemo(() => Object.fromEntries(meta.map((m) => [m.id, m])), [meta])
  const samples = useMemo(
    () =>
      targets
        .filter((t) => t.status === 'sampled' || t.cached || t.carried_by)
        .sort((a, b) => (Number.isFinite(b.value) ? b.value : -1) - (Number.isFinite(a.value) ? a.value : -1) || String(a.id).localeCompare(String(b.id))),
    [targets],
  )
  const live = Object.values(rovers).filter((r) => r.sampling)
  const onPick = (t) => {
    appStore.patch({ pickedTarget: t.id })
    onFocusEvent?.({ at: [t.x, t.y], key: `sample:${t.id}`, level: 'discovery' })
  }
  const score = Number.isFinite(fleet.science_score) ? fleet.science_score : null
  return (
    <div className="samples" id="panel-samples" role="tabpanel" aria-labelledby="tab-samples" hidden={hidden}>
      <div className="smp-head">
        <span>
          Science score <b>{score ?? '--'}</b>
        </span>
        <span>
          <b>{Number.isFinite(fleet.samples_cached) ? fleet.samples_cached : samples.filter((t) => t.cached).length}</b> cached
        </span>
        <span>
          <b>{Number.isFinite(fleet.samples_carried) ? fleet.samples_carried : samples.filter((t) => !t.cached).length}</b> carried
        </span>
      </div>
      <ol className="smp-list">
        {live.map((r) => (
          <InProgress key={`live:${r.id}`} rover={r} meta={byId[r.id]} target={targets.find((t) => t.id === r.sampling.target)} />
        ))}
        {samples.map((t) => (
          <SampleRow key={t.id} t={t} carrier={t.carried_by ? byId[t.carried_by] : null} picked={picked === t.id} onPick={onPick} />
        ))}
        {!samples.length && !live.length && (
          <li className="empty-note">No samples yet. Rovers sample what they find: a drill for minerals, a scoop for ice, a sniffer for vents and a swab for organic traces. Full tubes go to the lander's cache.</li>
        )}
      </ol>
    </div>
  )
}
