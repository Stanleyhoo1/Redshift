import { useEffect, useRef, useState } from 'react'
import { appStore, useSim } from '../state/store.js'
import { getConfig, launch, resetSim } from '../net/connection.js'
import { CloseButton, Modal } from './Modal.jsx'
import { GROUP_TITLE, SEED_FIELD, diffSettings, fieldsByGroup, formatValue, gravityLook, metersPerSecond2, parseSeed, physicsAt, presetName, randomSeed, snap, summary } from '../lib/setup.js'
import '../styles/setup.css'

// The setup screen (v7, PROTOCOL.md "Version 7 additions"). Forced while the sim waits in its setup phase:
// no close button, Esc and the backdrop do nothing, and App closes it when the launched world's init
// arrives. Opened from "New mission" it can be closed, and launching replaces the running mission.
// An older sim (no GET /api/config) only gets the seed, through POST /api/reset.

const WAIT_MS = 20000 // after the sim said yes, how long to wait for the new world's init
const SIDE_GROUPS = ['fleet', 'sector', 'simulation']
const close = () => appStore.patch({ modal: null })
const noop = () => {}

export function SetupScreen({ forced = false }) {
  const gen = useSim((s) => s.gen)
  const liveSeed = useSim((s) => s.config?.seed ?? s.world?.seed)
  const [attempt, setAttempt] = useState(0)
  const [server, setServer] = useState(null) // GET /api/config, or the seed-only stand-in for an older sim
  const [loadError, setLoadError] = useState('')
  const [values, setValues] = useState(null)
  const [seedText, setSeedText] = useState('')
  const [busy, setBusy] = useState(false)
  const [error, setError] = useState('')
  const startGen = useRef(gen)
  const seedRef = useRef(liveSeed)
  seedRef.current = liveSeed

  useEffect(() => {
    let live = true
    setLoadError('')
    getConfig()
      .then((cfg) => {
        if (!live) return
        const seed = Number.isFinite(seedRef.current) ? seedRef.current : 42
        const got = cfg || { legacy: true, fields: [SEED_FIELD], current: { seed }, presets: {}, physics: [] }
        setServer(got)
        setValues({ ...got.current })
        setSeedText(String(got.current.seed ?? ''))
      })
      .catch((err) => {
        if (live) setLoadError(err?.message || 'Could not read the settings')
      })
    return () => {
      live = false
    }
  }, [attempt])

  // The launched world's init arrived: the mission is on (App closes the forced screen on the phase change).
  useEffect(() => {
    if (!busy || gen === startGen.current) return
    setBusy(false)
    if (!forced) close()
  }, [gen, busy, forced])
  useEffect(() => {
    if (!busy) return
    const id = setTimeout(() => {
      setBusy(false)
      setError('No new world arrived in 20 s. Check that the sim is still running.')
    }, WAIT_MS)
    return () => clearTimeout(id)
  }, [busy])

  const set = (key, v) => setValues((cur) => ({ ...cur, [key]: v }))
  const setAll = (fields, on) => setValues((cur) => ({ ...cur, ...Object.fromEntries(fields.map((f) => [f.key, on])) }))
  const seedField = server?.fields?.find((f) => f.key === 'seed') || SEED_FIELD
  const seedOk = parseSeed(seedText, seedField) !== null
  const typeSeed = (text) => {
    setSeedText(text)
    const n = parseSeed(text, seedField)
    if (n !== null) set('seed', n)
  }

  const go = async () => {
    if (busy || !server || !values || !seedOk) return
    setError('')
    startGen.current = gen
    setBusy(true)
    try {
      if (server.legacy) await resetSim(values.seed)
      else await launch(diffSettings(values, server.current))
    } catch (err) {
      setBusy(false)
      setError(err?.message || 'Launch refused')
    }
  }

  const title = forced ? 'Plan the landing' : 'New mission'
  const lede = forced
    ? 'The fleet waits on the lander pad. Nothing moves until you launch.'
    : 'Launching ends the current mission and lands the fleet on a new world with these settings.'
  const ready = !!server && !!values && !loadError
  const seedInput = <SeedField field={seedField} text={seedText} onText={typeSeed} onNew={() => typeSeed(String(randomSeed()))} onEnter={go} />

  let body
  if (loadError) {
    body = (
      <div className="su-body su-state">
        <p className="su-error" role="alert">
          {loadError}
        </p>
        <p className="su-help">Check that the sim is running, then try again.</p>
        <button type="button" className="btn" onClick={() => setAttempt((n) => n + 1)}>
          Try again
        </button>
      </div>
    )
  } else if (!ready) {
    body = (
      <div className="su-body su-state">
        <p className="su-help" role="status">
          Reading the sim's settings...
        </p>
      </div>
    )
  } else if (server.legacy) {
    body = (
      <div className="su-body su-legacy">
        {seedInput}
        <p className="su-note">This sim only lets you choose the seed. Gravity, fleet, sector and features come from its environment variables.</p>
      </div>
    )
  } else {
    const groups = fieldsByGroup(server.fields)
    const grav = server.fields.find((f) => f.key === 'gravity_g')
    const planetRest = (groups.planet || []).filter((f) => f.key !== 'gravity_g' && f.key !== 'seed')
    const field = (f) => <Field key={f.key} field={f} value={values[f.key]} onChange={(v) => set(f.key, v)} />
    body = (
      <div className="su-body">
        <div className="su-planet">
          {grav && <GravityHero field={grav} value={values.gravity_g} onChange={(v) => set('gravity_g', v)} presets={server.presets?.gravity_g} physics={server.physics} />}
          {seedInput}
          {planetRest.map(field)}
        </div>
        <div className="su-rest">
          <div className="su-trio">
            {SIDE_GROUPS.filter((g) => groups[g]).map((g) => (
              <section key={g} className="su-sec" aria-labelledby={`su-g-${g}`}>
                <h3 className="su-sec-title" id={`su-g-${g}`}>
                  {GROUP_TITLE[g]}
                </h3>
                {groups[g].map(field)}
              </section>
            ))}
          </div>
          {groups.features && <FeatureGrid fields={groups.features} values={values} set={set} setAll={setAll} />}
          {groups.wear && (
            <section className="su-sec" aria-labelledby="su-g-wear">
              <h3 className="su-sec-title" id="su-g-wear">
                {GROUP_TITLE.wear}
              </h3>
              {groups.wear.map(field)}
            </section>
          )}
        </div>
      </div>
    )
  }

  return (
    <Modal label={title} onClose={forced ? noop : close} className={`su${forced ? ' su-forced' : ''}`}>
      <header className="su-head">
        <div className="su-titles">
          <h2 className="su-title">{title}</h2>
          <p className="su-lede">{lede}</p>
        </div>
        {!forced && <CloseButton onClick={close} />}
      </header>
      {body}
      <footer className="su-foot">
        <div className="su-foot-text">
          {error && (
            <p className="su-error" role="alert">
              {error}
            </p>
          )}
          <p className="su-summary">{ready ? summary(values, server.fields) : ''}</p>
        </div>
        <button type="button" className="su-launch" disabled={!ready || busy || !seedOk} onClick={go}>
          {busy ? 'Launching...' : 'Launch'}
        </button>
      </footer>
    </Modal>
  )
}

function Field({ field, value, onChange }) {
  if (field.key === 'num_rovers') return <Stepper field={field} value={value} onChange={onChange} />
  if (field.type === 'bool') return <ToggleField field={field} value={value} onChange={onChange} />
  if (field.type === 'choice') return <ChoiceField field={field} value={value} onChange={onChange} />
  if (field.type === 'int' || field.type === 'float') return <RangeField field={field} value={value} onChange={onChange} />
  return null
}

const fillOf = (field, v) => `${(((Number(v) - field.min) / (field.max - field.min)) * 100).toFixed(1)}%`

function RangeField({ field, value, onChange }) {
  const id = `su-${field.key}`
  return (
    <div className="su-field">
      <div className="su-row">
        <label className="su-label" htmlFor={id}>
          {field.label}
        </label>
        <output className="su-value" htmlFor={id}>
          {formatValue(field, value)}
        </output>
      </div>
      <input
        id={id}
        className="su-range"
        type="range"
        min={field.min}
        max={field.max}
        step={field.step}
        value={value}
        style={{ '--fill': fillOf(field, value) }}
        aria-valuetext={formatValue(field, value)}
        aria-describedby={field.help ? `${id}-help` : undefined}
        onChange={(e) => onChange(snap(field, e.target.value))}
      />
      {field.help && (
        <p className="su-help" id={`${id}-help`}>
          {field.help}
        </p>
      )}
    </div>
  )
}

function Stepper({ field, value, onChange }) {
  const id = `su-${field.key}`
  const n = Number.isFinite(value) ? value : field.min
  return (
    <div className="su-field">
      <div className="su-row">
        <span className="su-label" id={id}>
          {field.label}
        </span>
        <div className="su-stepper" role="group" aria-labelledby={id}>
          <button type="button" className="su-step" aria-label="Fewer rovers" disabled={n <= field.min} onClick={() => onChange(snap(field, n - 1))}>
            <svg viewBox="0 0 12 12" aria-hidden="true">
              <path d="M2.5 6h7" />
            </svg>
          </button>
          <output className="su-count" aria-live="polite">
            {n}
          </output>
          <button type="button" className="su-step" aria-label="More rovers" disabled={n >= field.max} onClick={() => onChange(snap(field, n + 1))}>
            <svg viewBox="0 0 12 12" aria-hidden="true">
              <path d="M2.5 6h7M6 2.5v7" />
            </svg>
          </button>
        </div>
      </div>
      <div className="su-pips" aria-hidden="true">
        {Array.from({ length: field.max }, (_, i) => (
          <span key={i} className={i < n ? 'on' : undefined} />
        ))}
      </div>
      {field.help && <p className="su-help">{field.help}</p>}
    </div>
  )
}

function ToggleField({ field, value, onChange }) {
  return (
    <div className="su-field">
      <button type="button" className="toggle su-toggle" aria-pressed={!!value} onClick={() => onChange(!value)}>
        <span className="toggle-label">{field.label}</span>
        <span className="switch" aria-hidden="true" />
      </button>
      {field.help && <p className="su-help">{field.help}</p>}
    </div>
  )
}

function ChoiceField({ field, value, onChange }) {
  const id = `su-${field.key}`
  const picked = (field.choices || []).find((c) => c.value === value)
  return (
    <div className="su-field">
      <span className="su-hidden" id={id}>
        {field.label}
      </span>
      <div className="seg su-seg" role="group" aria-labelledby={id}>
        {(field.choices || []).map((c) => (
          <button key={c.value} type="button" aria-pressed={c.value === value} onClick={() => onChange(c.value)}>
            {c.label || c.value}
          </button>
        ))}
      </div>
      <p className="su-help">{[field.help, picked?.help].filter(Boolean).join(' ')}</p>
    </div>
  )
}

function GravityHero({ field, value, onChange, presets, physics }) {
  const list = Array.isArray(presets) ? presets : []
  const look = gravityLook(value, field.min, field.max)
  const name = presetName(list, value)
  const row = physicsAt(physics, value)
  return (
    <section className="su-gravity" aria-labelledby="su-g-planet">
      <h3 className="su-sec-title" id="su-g-planet">
        {GROUP_TITLE.planet}
      </h3>
      <div className="su-g-read">
        <span className="su-g-num" style={{ fontStretch: `${look.stretch}%`, fontWeight: look.weight, color: look.color }}>
          {value.toFixed(1)}
        </span>
        <span className="su-g-unit">g</span>
        <span className="su-g-side">
          <span className="su-g-name">{name || 'Custom'}</span>
          <span className="su-g-ms">
            {metersPerSecond2(value)} m/s<sup>2</sup>
          </span>
        </span>
      </div>
      <label className="su-hidden" htmlFor="su-gravity_g">
        {field.label}
      </label>
      <input
        id="su-gravity_g"
        className="su-range su-range-g"
        type="range"
        min={field.min}
        max={field.max}
        step={field.step}
        value={value}
        style={{ '--fill': fillOf(field, value) }}
        aria-valuetext={`${value.toFixed(1)} g${name ? `, ${name}` : ''}`}
        onChange={(e) => onChange(snap(field, e.target.value))}
      />
      {list.length > 0 && (
        <div className="seg su-presets" role="group" aria-label="Gravity presets">
          {list.map((p) => (
            <button key={p.label} type="button" aria-pressed={Math.abs(p.value - value) < 1e-6} onClick={() => onChange(snap(field, p.value))}>
              {p.label} <span className="su-preset-g">{p.value.toFixed(1)}</span>
            </button>
          ))}
        </div>
      )}
      {field.help && <p className="su-help">{field.help}</p>}
      {row && <SlopeFigure row={row} />}
    </section>
  )
}

// The steepest climb (cyan) and the towing limit (amber dashes) at this gravity, drawn to scale.
const FIG_O = [34, 104]
const FIG_L = 196
const at = (deg, len) => [FIG_O[0] + len * Math.cos((deg * Math.PI) / 180), FIG_O[1] - len * Math.sin((deg * Math.PI) / 180)]
const xy = (p) => `${p[0].toFixed(1)} ${p[1].toFixed(1)}`

function SlopeFigure({ row }) {
  const climb = Number(row.climb_deg)
  const tow = Number(row.tow_deg)
  const [ox, oy] = FIG_O
  const top = at(climb, FIG_L)
  const towTop = at(tow, FIG_L)
  return (
    <figure className="su-fig">
      <svg viewBox="0 0 260 120" role="img" aria-label={`Steepest climb ${climb.toFixed(1)} degrees. Towing ${tow.toFixed(1)} degrees.`}>
        <path className="su-fig-wedge" d={`M${ox} ${oy}L${ox + FIG_L} ${oy}A${FIG_L} ${FIG_L} 0 0 0 ${xy(top)}Z`} />
        <path className="su-fig-ground" d={`M${ox - 22} ${oy}H${ox + FIG_L + 18}`} />
        <path className="su-fig-arc" d={`M${ox + 46} ${oy}A46 46 0 0 0 ${xy(at(climb, 46))}`} />
        <path className="su-fig-tow" d={`M${ox} ${oy}L${xy(towTop)}`} />
        <path className="su-fig-climb" d={`M${ox} ${oy}L${xy(top)}`} />
        <g className="su-fig-rover" transform={`translate(${xy(at(climb, FIG_L * 0.55))}) rotate(${(-climb).toFixed(1)})`}>
          <rect x="-10" y="-10" width="20" height="6.5" rx="1.5" />
          <circle cx="-6.5" cy="-2.6" r="2.6" />
          <circle cx="6.5" cy="-2.6" r="2.6" />
        </g>
        <text x={(top[0] + 6).toFixed(1)} y={(top[1] + 4).toFixed(1)}>
          {climb.toFixed(1)}°
        </text>
        <text className="su-fig-tow-label" x={(towTop[0] + 6).toFixed(1)} y={(towTop[1] + 4).toFixed(1)}>
          {tow.toFixed(1)}°
        </text>
      </svg>
      <figcaption className="su-fig-cap">
        <span className="su-key su-key-climb">
          Steepest climb {climb.toFixed(1)}°{row.grip_limited ? ' (grip limits it here, not the motors)' : ''}
        </span>
        <span className="su-key su-key-tow">Towing a broken rover {tow.toFixed(1)}°</span>
        <span className="su-fig-note">Predicted from motor torque and rover mass. The fleet still learns its real limit on the slopes it drives.</span>
      </figcaption>
    </figure>
  )
}

function SeedField({ field, text, onText, onNew, onEnter }) {
  const bad = parseSeed(text, field) === null
  return (
    <div className="su-field su-seed">
      <label className="su-label" htmlFor="su-seed">
        {field.label}
      </label>
      <div className="su-seed-row">
        <input
          id="su-seed"
          className="su-input"
          inputMode="numeric"
          autoComplete="off"
          spellCheck={false}
          value={text}
          aria-invalid={bad || undefined}
          aria-describedby="su-seed-help"
          onChange={(e) => onText(e.target.value)}
          onKeyDown={(e) => {
            if (e.key === 'Enter') onEnter()
          }}
        />
        <button type="button" className="btn" onClick={onNew}>
          New seed
        </button>
      </div>
      <p className={`su-help${bad ? ' su-help-bad' : ''}`} id="su-seed-help">
        {bad ? `Use a whole number from ${field.min} to ${field.max}.` : field.help}
      </p>
    </div>
  )
}

function FeatureGrid({ fields, values, set, setAll }) {
  const bools = fields.filter((f) => f.type === 'bool')
  const others = fields.filter((f) => f.type !== 'bool')
  const on = bools.filter((f) => values[f.key]).length
  return (
    <section className="su-sec su-features" aria-labelledby="su-g-features">
      <div className="su-sec-head">
        <h3 className="su-sec-title" id="su-g-features">
          {GROUP_TITLE.features}
        </h3>
        <span className="su-count-text">
          {on} of {bools.length} on
        </span>
        <button type="button" className="su-link" disabled={on === bools.length} onClick={() => setAll(bools, true)}>
          All on
        </button>
        <button type="button" className="su-link" disabled={on === 0} onClick={() => setAll(bools, false)}>
          All off
        </button>
      </div>
      <div className="su-tiles">
        {bools.map((f) => (
          <button
            key={f.key}
            type="button"
            className="su-tile"
            aria-pressed={!!values[f.key]}
            aria-labelledby={`su-t-${f.key}`}
            aria-describedby={`su-th-${f.key}`}
            onClick={() => set(f.key, !values[f.key])}
          >
            <span className="su-tile-mark" aria-hidden="true" />
            <span className="su-tile-name" id={`su-t-${f.key}`}>
              {f.label}
            </span>
            <span className="su-tile-help" id={`su-th-${f.key}`}>
              {f.help}
            </span>
          </button>
        ))}
      </div>
      {others.map((f) => (
        <Field key={f.key} field={f} value={values[f.key]} onChange={(v) => set(f.key, v)} />
      ))}
    </section>
  )
}
