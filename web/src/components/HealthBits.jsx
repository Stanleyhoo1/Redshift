import { memo } from 'react'
import { PARTS, SPARES, healthColor, partLabel, partLevel } from '../lib/theme.js'

// Version 9 rover health: a compact hp bar coloured by status and six small part
// indicators (drive left and right, LiDAR, camera, power, comms) with tooltips. Shared by
// the rover cards, the sensor panel, the event log and the lander tag.

const STATUS_WORD = { ok: 'OK', damaged: 'Damaged', immobile: 'Immobile', wrecked: 'Wrecked' }
export const statusWord = (s) => STATUS_WORD[s] || (s ? s[0].toUpperCase() + s.slice(1) : 'Unknown')

// Small line icons on a 16 x 16 grid (stroke currentColor; `.pi-fill` parts filled).
const ICON = {
  // Top view of the chassis with the left (or right) wheels filled.
  drive_l: (
    <>
      <rect x="5" y="3" width="6" height="10" rx="1.2" />
      <rect className="pi-fill" x="1.6" y="2.6" width="2.4" height="4" rx="0.8" />
      <rect className="pi-fill" x="1.6" y="9.4" width="2.4" height="4" rx="0.8" />
      <rect x="12" y="2.6" width="2.4" height="4" rx="0.8" />
      <rect x="12" y="9.4" width="2.4" height="4" rx="0.8" />
    </>
  ),
  drive_r: (
    <>
      <rect x="5" y="3" width="6" height="10" rx="1.2" />
      <rect x="1.6" y="2.6" width="2.4" height="4" rx="0.8" />
      <rect x="1.6" y="9.4" width="2.4" height="4" rx="0.8" />
      <rect className="pi-fill" x="12" y="2.6" width="2.4" height="4" rx="0.8" />
      <rect className="pi-fill" x="12" y="9.4" width="2.4" height="4" rx="0.8" />
    </>
  ),
  lidar: (
    <>
      <rect x="5.5" y="9" width="5" height="4.5" rx="1" />
      <path d="M8 9V7.4" />
      <path d="M4.6 5.6a4.8 4.8 0 0 1 6.8 0M2.6 3.6a7.6 7.6 0 0 1 10.8 0" />
    </>
  ),
  camera: (
    <>
      <path d="M2.2 5.6h2.6l1.2-1.8h4l1.2 1.8h2.6v7H2.2z" />
      <circle cx="8" cy="9" r="2.2" />
    </>
  ),
  power: (
    <>
      <rect x="2" y="4.6" width="11" height="7" rx="1.2" />
      <path d="M13 7v2.4" />
      <path className="pi-bolt" d="M8.4 5.6 6 8.4h2.2L7.4 10.6 10 7.6H7.8z" />
    </>
  ),
  comms: (
    <>
      <path d="M8 8.6v5.2M5.6 13.8h4.8" />
      <circle cx="8" cy="7.4" r="1.2" />
      <path d="M5.2 4.6a4 4 0 0 0 0 5.6M10.8 4.6a4 4 0 0 1 0 5.6M3.2 2.8a6.6 6.6 0 0 0 0 9.2M12.8 2.8a6.6 6.6 0 0 1 0 9.2" />
    </>
  ),
  // Spare kinds at the lander.
  drive: (
    <>
      <circle cx="8" cy="8" r="5.6" />
      <circle className="pi-fill" cx="8" cy="8" r="1.8" />
      <path d="M8 2.4v3.8M8 9.8v3.8M2.4 8h3.8M9.8 8h3.8" />
    </>
  ),
}
ICON.spare_lidar = ICON.lidar
ICON.spare_camera = ICON.camera
ICON.spare_power = ICON.power

export function PartIcon({ part, className = 'pi' }) {
  return (
    <svg className={className} viewBox="0 0 16 16" aria-hidden="true">
      {ICON[part] || <circle cx="8" cy="8" r="4.5" />}
    </svg>
  )
}

// The hp bar: filled to hp, coloured by status.
export function HealthBar({ health, className = '' }) {
  const hp = health?.hp
  const color = healthColor(health?.status, hp)
  return (
    <span className={`hb ${className}`} data-status={health?.status || undefined} style={{ '--hc': color }} aria-hidden="true">
      <i style={{ width: `${Number.isFinite(hp) ? Math.max(2, hp) : 0}%` }} />
    </span>
  )
}

// One indicator per part, coloured by level (title gives the part and its percentage).
export const PartDots = memo(function PartDots({ parts, className = '' }) {
  return (
    <span className={`pdots ${className}`} role="list" aria-label="Parts">
      {PARTS.map((p) => {
        const v = parts?.[p.key]
        const level = partLevel(v)
        return (
          <span key={p.key} role="listitem" className="pdot" data-level={level} title={`${p.label}: ${Number.isFinite(v) ? `${v}%` : 'no reading'}`} aria-label={`${p.label} ${Number.isFinite(v) ? `${v}%` : 'unknown'}`}>
            <PartIcon part={p.key} />
          </span>
        )
      })}
    </span>
  )
})

// Rover card row: bar, hp, part indicators. Hovering shows the rover's own diagnosis.
export function HealthRow({ health }) {
  if (!health) return null
  const hp = health.hp
  const title = `Health ${Number.isFinite(hp) ? `${hp}%` : '--'} (${statusWord(health.status)})${health.diagnosis ? `\n${health.diagnosis}` : ''}`
  return (
    <div className="rover-health" title={title} data-status={health.status || undefined}>
      <HealthBar health={health} />
      <span className="rover-hp" style={{ '--hc': healthColor(health.status, hp) }}>
        {Number.isFinite(hp) ? `${hp}%` : '--'}
      </span>
      <PartDots parts={health.parts} />
    </div>
  )
}

// The lander's spare parts: small icons with counts.
export function SpareIcons({ spares }) {
  if (!spares) return null
  const known = SPARES.filter((k) => Number.isFinite(spares[k.key]))
  const extra = Object.keys(spares).filter((k) => !SPARES.some((s) => s.key === k))
  return (
    <span className="spares" title={`Spare parts at the lander: ${[...known.map((k) => `${k.label.toLowerCase()} ${spares[k.key]}`), ...extra.map((k) => `${k} ${spares[k]}`)].join(', ')}`}>
      {known.map((k) => (
        <span key={k.key} className="spare" data-none={spares[k.key] === 0 || undefined}>
          <PartIcon part={k.key === 'drive' ? 'drive' : `spare_${k.key}`} />
          {spares[k.key]}
        </span>
      ))}
    </span>
  )
}

export { partLabel }
