import { useEffect, useMemo, useRef, useState } from 'react'
import { appStore, nowSeconds, useSim } from '../state/store.js'
import { camDown, fetchImage, freshUrl } from '../net/connection.js'
import { cleanMsg } from '../lib/format.js'
import { autoDefault, clearPlaceHover, hasPlaces, hoverKey, isWaiting, optionLabel, placeChipProps, placeNumbers, sendAnswer, setPlaceHover, useAnswer, usePlaceHover } from '../lib/answers.js'
import { useNow } from '../lib/useNow.js'
import { useHudRect } from '../lib/useHudRect.js'
import { CAM_RETRY_DOWN, LidarRadar, cameraProblem } from './SensorPanel.jsx'
import { SourceTag } from './Planner.jsx'

// Clarification requests (PROTOCOL.md version 3, `requests`): a rover that cannot
// interpret what it senses asks the operator, and takes the safe default when nobody
// answers in time. One prominent card under the hazard bar; more open questions queue
// beneath it as slim strips.
// Version 6: a request may wait for the operator (`expires_in` null: no countdown, no
// default; the rover holds position), and a `choose_place` request names candidate
// places (options with x, y): numbered chips here, numbered markers on the map, and a
// small top-down map in place of the camera picture.

const IMG_PERIOD = 500 // ms between camera frames while a question is open
const GHOST_ANSWERED = 2.6 // s the confirmation stays after an answer
const GHOST_EXPIRED = 4.2 // s the "took the safe default" note stays
const GHOST_SENDING = 4.5 // s while our answer is still in flight when the question closes
const STALE_AFTER = 3 // s past the deadline before a request the sim still lists is hidden

const toSeconds = () => performance.now() / 1000

function useCompact() {
  const query = '(max-height: 820px), (max-width: 1240px)'
  const [match, setMatch] = useState(() => window.matchMedia(query).matches)
  useEffect(() => {
    const mq = window.matchMedia(query)
    const update = () => setMatch(mq.matches)
    mq.addEventListener('change', update)
    return () => mq.removeEventListener('change', update)
  }, [])
  return match
}

const ShieldIcon = () => (
  <svg viewBox="0 0 12 12" aria-hidden="true">
    <path d="M6 1 10.2 2.6v3c0 2.6-1.8 4.4-4.2 5.4C3.6 10 1.8 8.2 1.8 5.6v-3z" />
  </svg>
)
const CheckIcon = () => (
  <svg viewBox="0 0 12 12" aria-hidden="true">
    <path d="M2.2 6.4 4.8 9 9.8 3.2" />
  </svg>
)

// What the rover sees: its camera (polled about twice a second while the question is
// open), or its latest LiDAR sweep when the sim cannot render.
function RequestImage({ q, size }) {
  const [frame, setFrame] = useState(null)
  const [problem, setProblem] = useState(() => (!q.image ? { title: 'No camera image' } : Date.now() < camDown.until ? camDown.problem : null))
  const hasScan = useSim((s) => !!s.rovers[q.rover]?.scan)
  const current = useRef(null)

  useEffect(() => {
    if (!q.image) return
    let stopped = false
    let timer = null
    let ctrl = null
    const loop = async () => {
      if (stopped) return
      const down = camDown.until - Date.now()
      if (down > 0) {
        setProblem(camDown.problem)
        timer = setTimeout(loop, down)
        return
      }
      const started = performance.now()
      ctrl = new AbortController()
      try {
        const blob = await fetchImage(freshUrl(q.image), ctrl.signal)
        const url = URL.createObjectURL(blob)
        try {
          const img = new Image()
          img.src = url
          await img.decode()
        } catch {
          URL.revokeObjectURL(url)
          throw Object.assign(new Error('The picture could not be decoded'), { status: -1 })
        }
        if (stopped) {
          URL.revokeObjectURL(url)
          return
        }
        const old = current.current
        current.current = url
        setFrame(url)
        setProblem(null)
        if (old) setTimeout(() => URL.revokeObjectURL(old), 500)
        timer = setTimeout(loop, Math.max(0, IMG_PERIOD - (performance.now() - started)))
      } catch (err) {
        if (stopped || err?.name === 'AbortError') return
        const p = cameraProblem(err)
        setProblem(p)
        const down = err.status === 503 || err.status === 404
        if (down) {
          camDown.until = Date.now() + CAM_RETRY_DOWN
          camDown.problem = p
        }
        timer = setTimeout(loop, down ? CAM_RETRY_DOWN : 3000)
      }
    }
    loop()
    return () => {
      stopped = true
      clearTimeout(timer)
      ctrl?.abort()
    }
  }, [q.image])

  useEffect(
    () => () => {
      if (current.current) URL.revokeObjectURL(current.current)
      current.current = null
    },
    [],
  )

  const live = frame && !problem
  return (
    <figure className="rq-img" data-state={live ? 'live' : 'off'}>
      {live ? (
        <img src={frame} alt="What the rover's camera sees" />
      ) : hasScan ? (
        <span className="rq-img-lidar">
          <LidarRadar roverId={q.rover} size={size} showRange={false} />
        </span>
      ) : (
        <span className="rq-img-none">
          <svg viewBox="0 0 24 24" aria-hidden="true">
            <path d="M3 7.5h3l1.6-2.3h6.8L16 7.5h3A1.5 1.5 0 0 1 20.5 9v8.5A1.5 1.5 0 0 1 19 19H5a1.5 1.5 0 0 1-1.5-1.5V9A1.5 1.5 0 0 1 5 7.5" />
            <circle cx="12" cy="13" r="3.4" />
            <path d="M3.5 3.5l17 17" />
          </svg>
          No picture
        </span>
      )}
      <figcaption className="rq-img-cap">
        {live ? (
          <>
            <i className="rq-live" aria-hidden="true" />
            Camera
          </>
        ) : hasScan ? (
          'LiDAR, camera unavailable'
        ) : (
          problem?.sub || 'Camera unavailable'
        )}
      </figcaption>
    </figure>
  )
}

// v6: a top-down sketch of the asking rover and the numbered candidates (north up), in
// place of the camera picture for a `choose_place` request. Hover and click like the chips.
const PM_W = 160
const PM_H = 120
function PlaceMap({ q, color, answer }) {
  const at = useSim((s) => {
    const r = s.rovers[q.rover]
    return r ? `${Math.round(r.x)},${Math.round(r.y)}` : ''
  })
  const lander = useSim((s) => s.lander)
  const hover = usePlaceHover()
  const nums = placeNumbers(q)
  const places = q.options.filter((o) => nums[o.id])
  const rover = at ? at.split(',').map(Number) : null
  const pts = places.map((o) => [o.x, o.y])
  if (rover) pts.push(rover)
  let x0 = Infinity
  let x1 = -Infinity
  let y0 = Infinity
  let y1 = -Infinity
  for (const [x, y] of pts) {
    x0 = Math.min(x0, x)
    x1 = Math.max(x1, x)
    y0 = Math.min(y0, y)
    y1 = Math.max(y1, y)
  }
  const span = Math.max(x1 - x0, ((y1 - y0) * PM_W) / PM_H, 16)
  const k = (PM_W - 36) / span
  const cx = (x0 + x1) / 2
  const cy = (y0 + y1) / 2
  const sx = (x) => PM_W / 2 + (x - cx) * k
  const sy = (y) => PM_H / 2 - (y - cy) * k
  const pending = answer?.state === 'pending' || answer?.state === 'sent'
  const lx = lander ? sx(lander.x) : -99
  const ly = lander ? sy(lander.y) : -99
  return (
    <figure className="rq-img rq-map" style={{ '--c': color }}>
      <svg viewBox={`0 0 ${PM_W} ${PM_H}`} role="group" aria-label="Where the candidates are, north up">
        {lx > 4 && lx < PM_W - 4 && ly > 4 && ly < PM_H - 4 && <rect className="rq-map-lander" x={lx - 3} y={ly - 3} width="6" height="6" rx="1.2" />}
        {rover &&
          places.map((o) => <line key={o.id} className="rq-map-line" x1={sx(rover[0])} y1={sy(rover[1])} x2={sx(o.x)} y2={sy(o.y)} data-hl={hover === hoverKey(q.id, o.id) || undefined} />)}
        {places.map((o) => {
          const key = hoverKey(q.id, o.id)
          return (
            <g
              key={o.id}
              className="rq-map-pin"
              data-hl={hover === key || undefined}
              data-chosen={(answer?.option === o.id && pending) || undefined}
              transform={`translate(${sx(o.x)} ${sy(o.y)})`}
              onMouseEnter={() => setPlaceHover(key)}
              onMouseLeave={() => clearPlaceHover(key)}
              onClick={() => !pending && sendAnswer(q.id, o.id)}
            >
              <title>{o.label}</title>
              <circle r="9" />
              <text dy="3.6">{nums[o.id]}</text>
            </g>
          )
        })}
        {rover && <circle className="rq-map-rover" cx={sx(rover[0])} cy={sy(rover[1])} r="4.2" />}
      </svg>
      <figcaption className="rq-img-cap">Map, north up</figcaption>
    </figure>
  )
}

// Smooth countdown between ticks: one rAF loop writes the bar's scale directly.
function CountdownBar({ q, frozen }) {
  const fill = useRef(null)
  useEffect(() => {
    const el = fill.current
    if (!el || frozen) return
    if (q.deadline === null || !q.total) {
      el.style.transform = 'scaleX(1)'
      return
    }
    let raf = 0
    const step = () => {
      const left = q.deadline - toSeconds()
      el.style.transform = `scaleX(${Math.max(0, Math.min(1, left / q.total))})`
      el.toggleAttribute('data-low', left <= 5)
      if (left > 0) raf = requestAnimationFrame(step)
    }
    step()
    return () => cancelAnimationFrame(raf)
  }, [q.deadline, q.total, frozen])
  return (
    <div className="rq-bar" data-wait={(q.deadline === null && !frozen) || undefined} aria-hidden="true">
      <i ref={fill} />
    </div>
  )
}

function Outcome({ kind, q, name, answer, ghost }) {
  const chosen = optionLabel(q, answer?.option ?? ghost?.option)
  const def = optionLabel(q, q.default)
  if (kind === 'sending') {
    return (
      <p className="rq-out" data-kind="sending" role="status">
        <CheckIcon />
        <span>
          Sending: <b>{chosen}</b>...
        </span>
      </p>
    )
  }
  if (kind === 'answered') {
    return (
      <p className="rq-out" data-kind="answered" role="status">
        <CheckIcon />
        <span>
          Sent: <b>{chosen}</b>. {name} is acting on it.
        </span>
      </p>
    )
  }
  if (kind === 'expired') {
    return (
      <p className="rq-out" data-kind="expired" role="status">
        <ShieldIcon />
        <span>
          No answer in time. {name} took the safe default: <b>{def}</b>.
        </span>
      </p>
    )
  }
  if (kind === 'late') {
    return (
      <p className="rq-out" data-kind="late" role="status">
        <ShieldIcon />
        <span>{ghost?.error || answer?.error || 'Too late: this question already closed.'}</span>
      </p>
    )
  }
  return (
    <p className="rq-out" data-kind="closed" role="status">
      <CheckIcon />
      <span>Question closed: answered in Comms or on another screen, or no longer needed.</span>
    </p>
  )
}

function RequestCard({ q, ghost, meta, byId, source, now, imgSize, onSelect }) {
  const answer = useAnswer(q.id)
  const hover = usePlaceHover()
  const name = meta?.name || q.rover || 'A rover'
  const left = q.deadline === null ? null : Math.max(0, q.deadline - now)
  const pending = answer?.state === 'pending'
  // Our own answer decides first; otherwise what the ghost recorded when the question closed.
  const fromAnswer = answer?.state === 'sent' ? 'answered' : answer?.state === 'error' && answer.status === 409 ? 'late' : null
  let kind = fromAnswer || ghost?.kind || null
  if (kind === 'sending' && answer?.state === 'error') kind = 'closed'
  const auto = autoDefault(q)
  const def = optionLabel(q, auto)
  const waiting = isWaiting(q)
  const places = hasPlaces(q)
  const nums = places ? placeNumbers(q) : null
  const color = meta?.color || '#8d99aa'
  let hint
  if (waiting && places) hint = `${name} holds position until you choose. Click a numbered marker on the map, a choice here, or reply in Comms ("the second one").`
  else if (waiting) hint = `${name} holds position until you answer.`
  else if (auto) hint = `Moving cautiously until you answer. With no answer it takes the safe default: ${def.toLowerCase()}.`
  else hint = 'Moving cautiously until you answer.'
  return (
    <section className="hud hud-request" style={{ '--rover': color }} data-kind={kind || 'open'} data-wait={(waiting && !kind) || undefined} aria-label={`${name} asks the operator`}>
      {places ? <PlaceMap q={q} color={color} answer={answer} /> : <RequestImage q={q} size={imgSize} />}
      <div className="rq-main">
        <header className="rq-head">
          <span className="rover-dot" aria-hidden="true" />
          <button type="button" className="rq-name" onClick={() => q.rover && onSelect?.(q.rover)} title={`Show ${name}'s sensors`}>
            {name}
          </button>
          <span className="rq-asks">{kind ? 'asked' : 'asks you'}</span>
          {source && <SourceTag source={source} />}
          {!kind && left !== null && (
            <span className="rq-left" data-low={left <= 5 || undefined} aria-label={`${Math.ceil(left)} seconds left`}>
              {Math.ceil(left)}
              <small> s</small>
            </span>
          )}
          {!kind && waiting && (
            <span className="rq-wait" title="No time limit: the rover waits for your answer">
              <i aria-hidden="true" />
              Waiting for you
            </span>
          )}
        </header>
        <p className="rq-q">{cleanMsg(q.question, q.rover, byId)}</p>
        {kind ? (
          <Outcome kind={kind} q={q} name={name} answer={answer} ghost={ghost} />
        ) : (
          <>
            <div className="rq-opts" role="group" aria-label="Answer">
              {q.options.map((o) => {
                const isDefault = o.id === auto
                const n = nums?.[o.id]
                return (
                  <button
                    key={o.id}
                    type="button"
                    className="rq-opt"
                    data-default={isDefault || undefined}
                    data-place={n ? true : undefined}
                    aria-busy={(pending && answer.option === o.id) || undefined}
                    disabled={pending}
                    onClick={() => sendAnswer(q.id, o.id)}
                    title={n ? `${o.label} (marker ${n} on the map)` : undefined}
                    {...placeChipProps(q, o, hover)}
                  >
                    {n ? (
                      <span className="rq-opt-row">
                        <b className="place-num" aria-hidden="true">
                          {n}
                        </b>
                        <span className="rq-opt-label">{o.label}</span>
                      </span>
                    ) : (
                      <span className="rq-opt-label">{o.label}</span>
                    )}
                    {isDefault && (
                      <span className="rq-safe">
                        <ShieldIcon />
                        safe default
                      </span>
                    )}
                  </button>
                )
              })}
            </div>
            {answer?.state === 'error' ? (
              <p className="rq-err" role="alert">
                {answer.error}
              </p>
            ) : (
              <p className="rq-hint">{hint}</p>
            )}
          </>
        )}
      </div>
      <CountdownBar q={q} frozen={!!kind} />
    </section>
  )
}

function RequestStrip({ q, meta, byId, now, onOpen }) {
  const name = meta?.name || q.rover || 'A rover'
  const left = q.deadline === null ? null : Math.max(0, q.deadline - now)
  return (
    <button type="button" className="hud rq-strip" style={{ '--rover': meta?.color || '#8d99aa' }} onClick={onOpen} title="Show this question">
      <span className="rover-dot" aria-hidden="true" />
      <b>{name}</b>
      <span className="rq-strip-q">{cleanMsg(q.question, q.rover, byId)}</span>
      {left !== null ? <span className="rq-strip-left">{Math.ceil(left)} s</span> : <span className="rq-strip-wait">waiting</span>}
      <i className="rq-strip-bar" data-wait={left === null || undefined} style={{ transform: `scaleX(${q.total && left !== null ? Math.max(0, Math.min(1, left / q.total)) : 1})` }} aria-hidden="true" />
    </button>
  )
}

export function RequestCards({ onSelect }) {
  const requests = useSim((s) => s.requests)
  const meta = useSim((s) => s.roverMeta)
  const chat = useSim((s) => s.chat)
  const gen = useSim((s) => s.gen)
  const [pick, setPick] = useState(null)
  const [ghost, setGhost] = useState(null) // { q, kind, option, error, until }
  const [seen, setSeen] = useState(requests) // the list as of the last render
  const [seenGen, setSeenGen] = useState(gen)
  const shown = useRef(null) // id of the request in the committed main card
  const now = useNow(250, requests.length > 0 || !!ghost)
  const compact = useCompact()
  const stackRef = useHudRect('requests') // v6: the camera keeps place candidates out from under the card
  const byId = useMemo(() => Object.fromEntries(meta.map((m) => [m.id, m])), [meta])

  // When the question in the main card leaves the open list, keep it on screen for a
  // moment with what happened: answered (by us), expired (safe default), or closed.
  // Worked out while rendering, so the next card never flashes up in between.
  if (seenGen !== gen) {
    // A reset or reconnect starts afresh.
    setSeenGen(gen)
    setSeen(requests)
    setGhost(null)
    setPick(null)
  } else if (seen !== requests) {
    const id = shown.current
    const was = id ? seen.find((q) => q.id === id) : null
    if (was && !requests.some((q) => q.id === id)) {
      const a = appStore.get().answers[id]
      const t = nowSeconds()
      // 'sending': our answer is still on its way back (the sim often closes the
      // question a tick before the POST returns); the card follows the answer's state.
      const kind =
        a?.state === 'sent'
          ? 'answered'
          : a?.state === 'pending'
            ? 'sending'
            : a?.state === 'error' && a.status === 409
              ? 'late'
              : was.deadline !== null && t >= was.deadline - 1.5
                ? 'expired'
                : 'closed'
      setGhost({ q: was, kind, option: a?.option ?? null, error: a?.error ?? null, until: t + (kind === 'answered' ? GHOST_ANSWERED : kind === 'sending' ? GHOST_SENDING : GHOST_EXPIRED) })
    }
    setSeen(requests)
  }

  useEffect(() => {
    if (!ghost) return
    const id = setTimeout(() => setGhost(null), Math.max(0, (ghost.until - nowSeconds()) * 1000))
    return () => clearTimeout(id)
  }, [ghost])

  // The sim's order (oldest first). The main card stays put until it closes or you pick
  // another: questions asked together have near-equal deadlines, and a card that swapped
  // under the pointer could take a click meant for another rover.
  const open = requests.filter((q) => q.deadline === null || q.deadline > now - STALE_AFTER)
  const main = ghost ? ghost.q : open.find((q) => q.id === pick) || open.find((q) => q.id === shown.current) || open[0] || null
  useEffect(() => {
    shown.current = main?.id ?? null
  })
  if (!main) return null
  const rest = open.filter((q) => q.id !== main.id)
  const sourceOf = (id) => {
    for (let k = chat.length - 1; k >= 0; k--) if (chat[k].request_id === id && chat[k].kind === 'question') return chat[k].source
    return null
  }
  const openStrip = (id) => {
    setGhost(null)
    setPick(id)
  }
  return (
    <div className="rq-stack" ref={stackRef}>
      <RequestCard
        key={main.id}
        q={main}
        ghost={ghost && ghost.q.id === main.id ? ghost : null}
        meta={byId[main.rover]}
        byId={byId}
        source={sourceOf(main.id)}
        now={now}
        imgSize={compact ? 114 : 138}
        onSelect={onSelect}
      />
      {rest.slice(0, 2).map((q) => (
        <RequestStrip key={q.id} q={q} meta={byId[q.rover]} byId={byId} now={now} onOpen={() => openStrip(q.id)} />
      ))}
      {rest.length > 2 && <div className="rq-more">{rest.length - 2} more waiting</div>}
    </div>
  )
}
