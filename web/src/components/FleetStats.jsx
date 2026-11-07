import { useSim } from '../state/store.js'
import { degrees, fixed, mmss } from '../lib/format.js'
import { kindStyle } from '../lib/theme.js'

const GAUGE_MAX = 45 // degrees shown on the slope limit gauge

function SlopeLimit() {
  const limit = useSim((s) => s.limit)
  const { current, initial, prev, changedAt } = limit
  const changed = changedAt !== null && prev !== null
  const delta = changed ? current - prev : 0
  const pct = (v) => `${Math.max(0, Math.min(100, (v / GAUGE_MAX) * 100))}%`
  return (
    <div className="stat stat-limit">
      <div className="stat-label">Slope limit</div>
      <div className="stat-value limit-value" key={changedAt ?? 'base'} data-flash={changed || undefined}>
        {degrees(current)}
        <small> deg</small>
        {changed && (
          <span className="limit-delta" key={changedAt}>
            {delta > 0 ? '+' : '-'}
            {degrees(Math.abs(delta))}
          </span>
        )}
      </div>
      <div className="stat-sub">(started at {degrees(initial)})</div>
      {Number.isFinite(current) && (
        <div className="gauge" aria-hidden="true">
          <i className="gauge-fill" style={{ width: pct(current) }} />
          {Number.isFinite(initial) && <i className="gauge-mark" style={{ left: pct(initial) }} />}
        </div>
      )}
    </div>
  )
}

function TargetPips({ targets, total }) {
  const pips = targets.map((t) => (
    <i key={t.id} className="pip" data-status={t.status} style={{ '--kind': kindStyle(t.kind).color }} title={`${kindStyle(t.kind).label}, ${t.status}`} />
  ))
  const rest = Math.max(0, Math.min(24, (total ?? 0) - targets.length))
  for (let k = 0; k < rest; k++) pips.push(<i key={`rest${k}`} className="pip" data-status="unknown" />)
  return <div className="pips" data-many={targets.length + rest > 9 || undefined}>{pips}</div>
}

export const LifeIcon = () => (
  <svg className="life-icon" viewBox="0 0 16 16" aria-hidden="true">
    <ellipse cx="6.6" cy="8.6" rx="3.6" ry="2.2" />
    <circle cx="11.6" cy="7.6" r="1.6" />
    <path d="M4.4 10.6 3.4 13.2M6.6 10.8v2.6M8.8 10.6l1 2.6M12.6 6.2l1-2M11.2 6.1l-.3-2.1" />
  </svg>
)

// v4b: the fleet's science score and the samples in the lander's cache (one tube each).
function Science({ fleet, targets }) {
  const cached = targets.filter((t) => t.cached).sort((a, b) => (b.value ?? 0) - (a.value ?? 0))
  const nCached = Number.isFinite(fleet.samples_cached) ? fleet.samples_cached : cached.length
  const carried = Number.isFinite(fleet.samples_carried) ? fleet.samples_carried : 0
  return (
    <div className="stat stat-science">
      <div className="stat-label">Science score</div>
      <div className="stat-value">
        {fixed(fleet.science_score, 0)}
        <small> pts</small>
      </div>
      <div className="cache-row" aria-hidden="true">
        {cached.slice(0, 14).map((t) => (
          <i key={t.id} className="tube" data-full style={{ '--c': kindStyle(t.kind).color }} title={`${kindStyle(t.kind).label} ${t.id}${Number.isFinite(t.value) ? `, ${t.value}/10` : ''}`} />
        ))}
      </div>
      <div className="stat-sub">
        {nCached ? `${nCached} cached` : 'Cache empty'}
        {carried ? `, ${carried} carried` : ''}
      </div>
    </div>
  )
}

export function FleetStats() {
  const fleet = useSim((s) => s.fleet)
  const t = useSim((s) => s.t)
  const targets = useSim((s) => s.targets)
  const explored = fleet.explored_pct
  const found = Number.isFinite(fleet.targets_found) ? fleet.targets_found : targets.length
  const total = Number.isFinite(fleet.targets_total) ? fleet.targets_total : null
  const sampled = targets.filter((x) => x.status === 'sampled').length
  const science = Number.isFinite(fleet.science_score)
  const life = Number.isFinite(fleet.lifeforms_seen) ? fleet.lifeforms_seen : null
  return (
    <section className="stats" aria-label="Fleet">
      <div className="stat">
        <div className="stat-label">Explored</div>
        <div className="stat-value">
          {fixed(explored, 1)}
          <small>%</small>
        </div>
        <div className="meter" aria-hidden="true">
          <i style={{ width: `${Math.max(0, Math.min(100, explored || 0))}%` }} />
        </div>
      </div>
      {science ? (
        <Science fleet={fleet} targets={targets} />
      ) : (
        <div className="stat">
          <div className="stat-label">Sim time</div>
          <div className="stat-value">{mmss(t)}</div>
          <div className="stat-sub">{Number.isFinite(fleet.sim_speed) ? `${fixed(fleet.sim_speed, 1)}x speed` : 'Speed --'}</div>
        </div>
      )}
      <div className="stat">
        <div className="stat-label">Targets found</div>
        <div className="stat-value">
          {found}
          <small> of {total ?? '--'}</small>
        </div>
        <TargetPips targets={targets} total={total} />
        <div className="stat-sub">
          {sampled} sampled
          {life !== null && (
            <span className="life-chip" data-some={life > 0 || undefined} title="Distinct lifeforms the fleet has confirmed">
              <LifeIcon />
              {life} {life === 1 ? 'lifeform' : 'lifeforms'}
            </span>
          )}
        </div>
      </div>
      <SlopeLimit />
    </section>
  )
}
