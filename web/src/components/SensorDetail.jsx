import { useCallback, useEffect, useLayoutEffect, useMemo, useRef, useState } from 'react'
import { appStore, useSim } from '../state/store.js'
import { telemetry } from '../state/telemetry.js'
import { fixed } from '../lib/format.js'
import { HAZARD, SIGNAL } from '../lib/theme.js'
import { Battery, StatePill, fmtMs } from './RoverBits.jsx'
import { LidarRadar } from './SensorPanel.jsx'
import { TimeChart } from './TimeChart.jsx'
import { CloseButton, Modal, RoverSwitch, roverFromKey } from './Modal.jsx'

// Large sensor view: a big LiDAR scope (scroll to change the range it shows), the
// rover's running totals (protocol version 5 `stats`), and charts of the last minutes
// from the client-side telemetry buffers (every rover is recorded from every tick, so
// switching rover shows its history straight away).

const AMBER = '#f3b24a'
const OK = '#62d6a8'
const LOAD = '#b39dff'
const RANGES = [2, 3, 4, 6, 8, 10, 12, 16]
const WINDOWS = [
  { s: 60, label: '1 min' },
  { s: 120, label: '2 min' },
  { s: 300, label: '5 min' },
]
const num = (v) => typeof v === 'number' && Number.isFinite(v)

function useBoxSize(ref) {
  const [size, setSize] = useState({ w: 0, h: 0 })
  useLayoutEffect(() => {
    const el = ref.current
    if (!el) return
    const ro = new ResizeObserver(() => {
      const w = el.clientWidth
      const h = el.clientHeight
      setSize((s) => (s.w === w && s.h === h ? s : { w, h }))
    })
    ro.observe(el)
    return () => ro.disconnect()
  }, [ref])
  return size
}

const int = (v) => Math.round(v).toLocaleString('en-US')

function RoverStats({ stats, mapArea }) {
  if (!stats) {
    return (
      <div className="sd-stats sd-stats-none">
        <b>No running totals</b>
        <span>This sim does not report rover stats (protocol version 5).</span>
      </div>
    )
  }
  const pct = num(stats.mapped_m2) && num(mapArea) && mapArea > 0 ? (stats.mapped_m2 / mapArea) * 100 : null
  const items = [
    { k: 'Distance driven', v: num(stats.distance_m) ? int(stats.distance_m) : '--', u: 'm' },
    { k: 'Area mapped', v: num(stats.mapped_m2) ? int(stats.mapped_m2) : '--', u: 'm²', sub: pct === null ? null : `${fixed(pct, pct < 10 ? 1 : 0)}% of the map` },
    { k: 'Targets found', v: num(stats.targets_found) ? int(stats.targets_found) : '--' },
    { k: 'Samples taken', v: num(stats.samples_taken) ? int(stats.samples_taken) : '--' },
    { k: 'Hazards met', v: num(stats.hazards_met) ? int(stats.hazards_met) : '--', warn: num(stats.hazards_met) && stats.hazards_met > 0 },
    { k: 'Lifeforms seen', v: num(stats.lifeforms_seen) ? int(stats.lifeforms_seen) : '--', life: num(stats.lifeforms_seen) && stats.lifeforms_seen > 0 },
    { k: 'Replans', v: num(stats.replans) ? int(stats.replans) : '--' },
  ]
  return (
    <dl className="sd-stats">
      {items.map((it) => (
        <div key={it.k} data-warn={it.warn || undefined} data-life={it.life || undefined}>
          <dt>{it.k}</dt>
          <dd>
            {it.v}
            {it.u && <small> {it.u}</small>}
          </dd>
          {it.sub && <span className="sd-stat-sub">{it.sub}</span>}
        </div>
      ))}
    </dl>
  )
}

function BigRadar({ roverId, rover, range, setRange, focus }) {
  const box = useRef(null)
  const size = useBoxSize(box)
  const d = Math.max(120, Math.floor(Math.min(size.w, size.h)))
  useEffect(() => {
    const el = box.current
    if (!el) return
    const onWheel = (e) => {
      e.preventDefault()
      setRange((r) => {
        const i = RANGES.indexOf(r)
        return RANGES[Math.max(0, Math.min(RANGES.length - 1, (i < 0 ? 4 : i) + (e.deltaY > 0 ? 1 : -1)))]
      })
    }
    el.addEventListener('wheel', onWheel, { passive: false })
    return () => el.removeEventListener('wheel', onWheel)
  }, [setRange])
  const s = rover?.sensors
  const lr = num(s?.lidar_range) ? s.lidar_range : null
  const hits = num(s?.lidar_hits) ? s.lidar_hits : rover?.scan ? Math.floor(rover.scan.length / 3) : null
  const idx = RANGES.indexOf(range)
  return (
    <div className="sd-radar" data-focus={focus || undefined}>
      <div className="sd-radar-head">
        <span className="sd-h">LiDAR</span>
        <span className="sd-radar-read">
          <b>{lr === null ? '--' : `${fixed(lr, 1)} m`}</b> range
          {hits !== null && (
            <>
              , <b>{hits}</b> hits
            </>
          )}
        </span>
        <span className="seg seg-xs sd-range" role="group" aria-label="Range shown">
          <button type="button" onClick={() => setRange(RANGES[Math.max(0, idx - 1)])} disabled={idx <= 0} aria-label="Show less range" title="Closer (scroll up)">
            −
          </button>
          <span className="sd-range-v">{range} m</span>
          <button type="button" onClick={() => setRange(RANGES[Math.min(RANGES.length - 1, idx + 1)])} disabled={idx >= RANGES.length - 1} aria-label="Show more range" title="Further (scroll down)">
            +
          </button>
        </span>
      </div>
      <div className="sd-radar-box" ref={box} title="Scroll to change the range shown">
        {size.w > 0 && <LidarRadar key={roverId} roverId={roverId} size={d} viewRange={range} trails={6} labels />}
        {!rover?.scan && <span className="sp-radar-none">No LiDAR data</span>}
      </div>
      <ul className="sd-radar-key" aria-hidden="true">
        <li>
          <i style={{ background: HAZARD }} />
          High
        </li>
        <li>
          <i style={{ background: AMBER }} />
          Above
        </li>
        <li>
          <i style={{ background: SIGNAL }} />
          Level
        </li>
        <li>
          <i style={{ background: '#6f8cff' }} />
          Below
        </li>
        <li className="sd-radar-hint">Faded: the last few sweeps. Scroll to zoom.</li>
      </ul>
    </div>
  )
}

export function SensorDetail({ roverId: initialRover, focus = null }) {
  const meta = useSim((s) => s.roverMeta)
  const [roverId, setRoverId] = useState(initialRover)
  const [win, setWin] = useState(120)
  const [range, setRange] = useState(8)
  const [lit, setLit] = useState(focus)
  const rover = useSim((s) => s.rovers[roverId] || null)
  const hazards = useSim((s) => s.hazards)
  const budget = useSim((s) => s.fleet.budget_ms)
  const gemmaBudget = useSim((s) => s.planner?.budget_ms ?? null)
  const hasPlanner = useSim((s) => !!s.planner)
  const limit = useSim((s) => s.limit.current)
  const mapArea = useSim((s) => (s.map ? s.map.width * s.map.depth : null))
  const m = meta.find((x) => x.id === roverId) || null
  const close = useCallback(() => appStore.patch({ modal: null }), [])

  useEffect(() => {
    if (m) return
    if (meta.length) setRoverId(meta[0].id)
    else close()
  }, [m, meta, close])
  // The section the user expanded glows for a moment.
  useEffect(() => {
    if (!focus) return
    const id = setTimeout(() => setLit(null), 2200)
    return () => clearTimeout(id)
  }, [focus])

  const color = m?.color || SIGNAL
  const getRover = useCallback(() => telemetry.rovers.get(roverId) || null, [roverId])
  const getFleet = useCallback(() => telemetry.fleet, [])
  const b = num(budget) && budget > 0 ? budget : null
  const gb = num(gemmaBudget) && gemmaBudget > 0 ? gemmaBudget / 1000 : null
  const lim = num(limit) ? limit : null

  const specs = useMemo(
    () => [
      { id: 'speed', group: 'motion', title: 'Speed', unit: 'm/s', digits: 2, min: 0, minSpan: 0.6, lines: [{ s: 'speed', color, fill: true, label: 'Ground' }, { s: 'wheel', color: '#aab7c7', dash: [4, 3], width: 1.4, label: 'Wheels' }] },
      { id: 'slip', group: 'motion', title: 'Wheel slip', unit: '%', k: 100, digits: 0, min: 0, minSpan: 50, lines: [{ s: 'slip', color: AMBER, fill: true }], refs: [{ v: 20, color: 'rgba(243, 178, 74, 0.8)', label: '20% slipping' }, { v: 45, color: 'rgba(255, 59, 92, 0.85)', label: '45% stuck' }] },
      { id: 'load', group: 'motion', title: 'Motor load', unit: '%', k: 100, digits: 0, min: 0, max: 100, lines: [{ s: 'load', color: LOAD, fill: true }], refs: [{ v: 70, color: 'rgba(243, 178, 74, 0.8)', label: '70%' }, { v: 90, color: 'rgba(255, 59, 92, 0.85)', label: '90%' }] },
      {
        id: 'att',
        group: 'motion',
        title: 'Pitch and roll',
        unit: 'deg',
        digits: 1,
        symmetric: true,
        minSpan: lim ? lim * 2.3 : 20,
        lines: [{ s: 'pitch', color: SIGNAL, label: 'Pitch' }, { s: 'roll', color: AMBER, label: 'Roll' }],
        refs: lim ? [{ v: lim, color: 'rgba(255, 59, 92, 0.75)', label: `slope limit ${Math.round(lim)} deg` }, { v: -lim, color: 'rgba(255, 59, 92, 0.75)' }] : [],
      },
      {
        id: 'brain',
        group: 'compute',
        title: 'Brain time per cycle',
        unit: 'ms',
        digits: 1,
        min: 0,
        minSpan: b ? b * 1.2 : 10,
        lines: [{ s: 'brain', color: 'rgba(170, 183, 199, 0.55)', width: 1, label: 'Last' }, { s: 'brainAvg', color: OK, width: 2.2, label: 'Avg' }],
        refs: b ? [{ v: b, color: 'rgba(232, 238, 245, 0.8)', label: `budget ${fmtMs(b)} ms` }] : [],
        empty: 'No compute data from this sim',
      },
      {
        id: 'gemma',
        group: 'compute',
        title: 'Gemma latency (fleet)',
        unit: 's',
        k: 0.001,
        digits: 2,
        min: 0,
        minSpan: gb ? gb * 1.15 : 1,
        fleet: true,
        lines: [{ s: 'gemma', color: 'rgba(185, 197, 255, 0.55)', width: 1, label: 'Last' }, { s: 'gemmaAvg', color: '#b9c5ff', width: 2.2, label: 'Avg' }],
        refs: gb ? [{ v: gb, color: 'rgba(232, 238, 245, 0.8)', label: `budget ${fixed(gb, 1)} s per call` }] : [],
        empty: hasPlanner ? 'Gemma is not answering yet (warming up or off)' : 'This sim has no Gemma planner',
      },
      { id: 'battery', group: 'motion', title: 'Battery', unit: '%', k: 100, digits: 0, min: 0, max: 100, lines: [{ s: 'battery', color: OK, fill: true }], refs: [{ v: 20, color: 'rgba(243, 178, 74, 0.8)', label: 'low 20%' }] },
      { id: 'lidar', group: 'lidar', title: 'LiDAR range', unit: 'm', digits: 1, min: 0, minSpan: 9, lines: [{ s: 'lidar', color: SIGNAL, fill: true }], empty: 'No LiDAR readings from this sim' },
    ],
    [color, b, gb, lim, hasPlanner],
  )

  const onKey = (e) => {
    const id = roverFromKey(e, meta)
    if (id) setRoverId(id)
  }

  return (
    <Modal label={`${m?.name || roverId} sensors`} onClose={close} className="sd" onKey={onKey}>
      <header className="ix-head" style={{ '--rover': color }}>
        <h2 className="ix-title">
          <svg viewBox="0 0 16 16" aria-hidden="true">
            <path d="M1.5 9.5h3l2-5 3 8 2-4.5h3" />
          </svg>
          Sensors
        </h2>
        <RoverSwitch value={roverId} onChange={setRoverId} />
        <StatePill rover={rover} hazards={hazards} />
        <Battery value={rover?.battery} charging={rover?.state === 'charging'} />
        <span className="seg seg-sm sd-win" role="group" aria-label="Time shown">
          {WINDOWS.map((w) => (
            <button key={w.s} type="button" aria-pressed={win === w.s} onClick={() => setWin(w.s)}>
              {w.label}
            </button>
          ))}
        </span>
        <CloseButton onClick={close} />
      </header>
      <div className="sd-body">
        <div className="sd-left">
          <BigRadar roverId={roverId} rover={rover} range={range} setRange={setRange} focus={lit === 'lidar'} />
          <RoverStats stats={rover?.stats ?? null} mapArea={mapArea} />
        </div>
        <div className="sd-charts">
          {specs.map((sp) => (
            <TimeChart key={`${sp.id}:${sp.fleet ? 'fleet' : roverId}`} spec={sp} getBuf={sp.fleet ? getFleet : getRover} win={win} focus={lit && lit === sp.group} />
          ))}
        </div>
      </div>
      <footer className="ix-foot">
        <span>
          Last {win / 60} min of wall time, recorded in this browser from every tick for every rover (up to 6 min). <kbd>1</kbd>-<kbd>{Math.min(9, meta.length) || 1}</kbd> rover <kbd>Esc</kbd> close
        </span>
      </footer>
    </Modal>
  )
}
