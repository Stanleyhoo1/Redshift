import { memo, useEffect, useLayoutEffect, useMemo, useRef, useState } from 'react'
import { useSim } from '../state/store.js'
import { postChat } from '../net/connection.js'
import { cleanMsg, mmss } from '../lib/format.js'
import { optionLabel, sendAnswer, useAnswers } from '../lib/answers.js'
import { useNow } from '../lib/useNow.js'
import { SourceTag } from './Planner.jsx'
import '../styles/interaction.css'

// Operator chat (PROTOCOL.md version 3): the thread of `chat` messages and a composer
// that POSTs /api/chat. The server echoes the operator's own message in `chat`.
// Version 5: questions get `kind: "reply"` answers carrying `reply_to` (the operator
// message they answer); they are drawn linked to that question. Until something answers
// an operator message it shows a small "thinking" line (quietly dropped after 15 s).

const KIND_LABEL = { refuse: 'Refused', question: 'Question', reply: 'Reply' }
const THINK_S = 15

// Suggested questions for the current addressee (one rover, or the whole fleet).
export const QUESTIONS = ['What have you explored?', 'What have you found?', "How's your battery?", 'Why are you going there?', 'Seen any life?', 'Any problems?']

function suggestions(meta) {
  const first = meta[0]
  const second = meta[1] || meta[0]
  const out = [{ text: 'All rovers return' }]
  if (second) out.push({ text: `${second.name} explore north` })
  out.push({ text: 'Hold all' }, { text: 'Resume all' })
  if (first) out.push({ text: `${first.name} go to 10, -5` })
  return out
}

// How a closed question ended, from what followed it in the thread.
function resolution(m, chat, answers) {
  const a = answers[m.request_id]
  const mine = a?.state === 'sent'
  let reply = null
  for (const x of chat) {
    if (x.key <= m.key || x.request_id !== m.request_id) continue
    if (x.from === 'operator') return { kind: 'answered', text: `${mine ? 'You answered' : 'Answered'}: ${x.text}` }
    if (!reply) reply = x
  }
  if (mine) return { kind: 'answered', text: `You answered: ${a.option.replace(/_/g, ' ')}` }
  if (reply) return { kind: 'expired', text: 'No answer in time: it took the safe default' }
  return { kind: 'closed', text: 'Closed' }
}

function InlineQuestion({ m, req, answer, res, now }) {
  if (!req) {
    return (
      <p className="msg-res" data-kind={res.kind}>
        {res.text}
      </p>
    )
  }
  if (answer?.state === 'sent') {
    return (
      <p className="msg-res" data-kind="answered">
        Sent: {optionLabel(req, answer.option)}
      </p>
    )
  }
  const pending = answer?.state === 'pending'
  const left = req.deadline === null ? null : Math.max(0, Math.ceil(req.deadline - now))
  return (
    <div className="msg-q">
      <div className="msg-opts" role="group" aria-label={`Answer ${m.from}`}>
        {req.options.map((o) => (
          <button
            key={o.id}
            type="button"
            className="msg-opt"
            data-default={o.id === req.default || undefined}
            disabled={pending}
            aria-busy={(pending && answer.option === o.id) || undefined}
            onClick={() => sendAnswer(req.id, o.id)}
            title={o.id === req.default ? 'The safe default: taken if nobody answers' : undefined}
          >
            {o.label}
            {o.id === req.default && <small>safe default</small>}
          </button>
        ))}
      </div>
      <p className="msg-q-foot" data-low={(left !== null && left <= 5) || undefined}>
        {answer?.state === 'error' ? <span className="msg-q-err">{answer.error}</span> : left !== null ? `${left} s left, then the safe default` : 'Waiting for your answer'}
      </p>
    </div>
  )
}

// "Rover 2 is thinking": shown under an operator message until something answers it.
function Thinking({ to, byId }) {
  const rover = to && to !== 'all' ? byId[to] : null
  return (
    <p className="ix-think" style={rover ? { '--c': rover.color } : undefined} aria-label={`${rover ? rover.name : 'The fleet'} is thinking`}>
      <span className="ix-think-dots" aria-hidden="true">
        <i />
        <i />
        <i />
      </span>
      {rover ? rover.name : 'Fleet'} is thinking
    </p>
  )
}

// The question a reply answers, quoted above the reply. Click to jump to it.
function ReplyQuote({ q, onJump, onHover }) {
  return (
    <button
      type="button"
      className="ix-quote"
      onClick={() => onJump(q.id)}
      onMouseEnter={() => onHover(q.id)}
      onMouseLeave={() => onHover(null)}
      onFocus={() => onHover(q.id)}
      onBlur={() => onHover(null)}
      title="Show the question"
    >
      <svg viewBox="0 0 16 16" aria-hidden="true">
        <path d="M3 2.5v4.5a3 3 0 0 0 3 3h7M10 7l3 3-3 3" />
      </svg>
      <span className="ix-quote-who">{q.from === 'operator' ? 'You' : q.from}:</span>
      <span className="ix-quote-text">{q.text}</span>
    </button>
  )
}

const ChatMsg = memo(function ChatMsg({ m, byId, live, req, answer, res, now, onPick, quote, cont, thinking, hl, onJump, onHover }) {
  const mine = m.from === 'operator'
  const rover = byId[m.from]
  const target = m.to && m.to !== 'all' && m.to !== 'operator' ? byId[m.to]?.name || m.to : 'all rovers'
  const who = mine ? 'You' : rover ? rover.name : m.from === 'fleet' ? 'Fleet' : m.from
  const text = mine ? m.text : cleanMsg(m.text, m.from, byId)
  const linked = !mine && m.reply_to
  return (
    <li
      className="msg"
      data-side={mine ? 'right' : 'left'}
      data-kind={m.kind}
      data-fleet={(!mine && !rover) || undefined}
      data-live={live || undefined}
      data-mid={m.id || undefined}
      data-linked={linked || undefined}
      data-cont={cont || undefined}
      data-hl={hl || undefined}
      style={rover ? { '--c': rover.color } : undefined}
      onMouseEnter={linked ? () => onHover(m.reply_to) : mine && m.id ? () => onHover(m.id) : undefined}
      onMouseLeave={linked || (mine && m.id) ? () => onHover(null) : undefined}
    >
      {quote && <ReplyQuote q={quote} onJump={onJump} onHover={onHover} />}
      <div className="msg-head">
        {!mine && <span className="ev-dot" style={rover ? { background: rover.color } : undefined} data-fleet={!rover || undefined} aria-hidden="true" />}
        {rover ? (
          <button type="button" className="msg-who msg-who-btn" onClick={() => onPick(rover.id)} title={`Reply to ${rover.name}`}>
            {who}
          </button>
        ) : (
          <span className="msg-who">{who}</span>
        )}
        {mine && <span className="msg-to">{m.kind === 'answer' ? `answered ${target}` : `to ${target}`}</span>}
        {KIND_LABEL[m.kind] && !mine && <span className="msg-kind">{KIND_LABEL[m.kind]}</span>}
        {!mine && m.source && <SourceTag source={m.source} />}
        <span className="msg-time">{mmss(m.t)}</span>
      </div>
      <div className="msg-bubble">{text}</div>
      {m.kind === 'question' && m.request_id && !mine && <InlineQuestion m={m} req={req} answer={answer} res={res} now={now} />}
      {thinking && <Thinking to={m.to} byId={byId} />}
    </li>
  )
})

function Composer({ to, setTo, inputRef }) {
  const meta = useSim((s) => s.roverMeta)
  const ready = useSim((s) => s.ready)
  const v3 = useSim((s) => s.v3)
  const [text, setText] = useState('')
  const [pending, setPending] = useState(null) // the text being sent
  const [error, setError] = useState(null) // { text, key }
  const history = useRef([])
  const hIdx = useRef(-1)
  const chips = useMemo(() => suggestions(meta), [meta])
  const target = meta.find((m) => m.id === to) || null

  useEffect(() => {
    if (to !== 'all' && !target) setTo('all')
  }, [to, target, setTo])
  useEffect(() => {
    if (!error) return
    const id = setTimeout(() => setError(null), 9000)
    return () => clearTimeout(id)
  }, [error])

  // Chips name their own targets ("Hold all"), so they go without a rover.
  const send = async (msg, rover) => {
    const t = msg.trim()
    if (!t || pending) return false
    setPending(t)
    setError(null)
    try {
      await postChat(t, rover)
      history.current = [t, ...history.current.filter((h) => h !== t)].slice(0, 20)
      hIdx.current = -1
      return true
    } catch (err) {
      setError({ text: err?.message || 'Message refused', key: Date.now() })
      return false
    } finally {
      setPending(null)
    }
  }

  const submit = async (e) => {
    e.preventDefault()
    if (await send(text, to === 'all' ? null : to)) setText('')
    inputRef.current?.focus()
  }

  // Up and Down walk through what you sent before.
  const onKeyDown = (e) => {
    const h = history.current
    if (e.key === 'ArrowUp' && h.length && (text === '' || hIdx.current >= 0)) {
      e.preventDefault()
      hIdx.current = Math.min(h.length - 1, hIdx.current + 1)
      setText(h[hIdx.current])
    } else if (e.key === 'ArrowDown' && hIdx.current >= 0) {
      e.preventDefault()
      hIdx.current -= 1
      setText(hIdx.current >= 0 ? h[hIdx.current] : '')
    }
  }

  const askWho = target ? target.name : 'the fleet'
  return (
    <div className="composer">
      {v3 && (
        <div className="ix-chiprow" data-kind="ask">
          <span className="ix-chiprow-label" title={`Questions for ${askWho}: the answers come from what it has sensed`}>
            Ask {target ? target.name : 'all'}
          </span>
          <div className="cm-chips" role="group" aria-label={`Ask ${askWho}`}>
            {QUESTIONS.map((q) => (
              <button key={q} type="button" className="cm-chip ix-ask-chip" disabled={!ready || !!pending} onClick={() => send(q, to === 'all' ? null : to)} title={`Ask ${askWho}: "${q}"`} style={target ? { '--c': target.color } : undefined}>
                {q}
              </button>
            ))}
          </div>
        </div>
      )}
      <div className="ix-chiprow" data-kind="do">
        {v3 && <span className="ix-chiprow-label">Order</span>}
        <div className="cm-chips" role="group" aria-label="Suggested orders">
          {chips.map((c) => (
            <button key={c.text} type="button" className="cm-chip" disabled={!ready || !!pending} onClick={() => send(c.text, null)} title={`Send "${c.text}"`}>
              {c.text}
            </button>
          ))}
        </div>
      </div>
      <form className="cm-row" onSubmit={submit}>
        <label className="cm-to" title="Who the message is for">
          <span className="cm-to-dot" style={target ? { background: target.color } : undefined} data-all={!target || undefined} aria-hidden="true" />
          <select value={to} onChange={(e) => setTo(e.target.value)} aria-label="Send to">
            <option value="all">All</option>
            {meta.map((m) => (
              <option key={m.id} value={m.id}>
                {m.name}
              </option>
            ))}
          </select>
          <svg className="cm-to-chev" viewBox="0 0 10 10" aria-hidden="true">
            <path d="M2.5 4 5 6.5 7.5 4" />
          </svg>
        </label>
        <input
          ref={inputRef}
          className="cm-input"
          type="text"
          value={text}
          onChange={(e) => {
            setText(e.target.value)
            hIdx.current = -1
          }}
          onKeyDown={onKeyDown}
          placeholder={target ? `Ask ${target.name} anything` : 'Ask the fleet anything'}
          aria-label="Message"
          autoComplete="off"
          spellCheck={false}
          disabled={!ready}
        />
        <button type="submit" className="cm-send" disabled={!ready || !text.trim() || !!pending} aria-busy={!!pending || undefined} aria-label="Send" title="Send (Enter)">
          <svg viewBox="0 0 16 16" aria-hidden="true">
            <path d="M2.5 8h10M8.5 3.5 13 8l-4.5 4.5" />
          </svg>
        </button>
      </form>
      {error && (
        <p className="cm-err" key={error.key} role="alert">
          {error.text}
        </p>
      )}
    </div>
  )
}

// Does message x answer operator message m? With reply_to (version 5) that is exact.
// A sim that never sends reply_to (older protocol) answers in order: the next message
// from the addressed rover (or any rover or the fleet, for a message to all).
function answersMsg(x, m, exact) {
  if (x.from === 'operator' || x.request_id) return false
  if (x.reply_to) return x.reply_to === m.id
  const fromOk = !m.to || m.to === 'all' || x.from === m.to || x.from === 'fleet'
  if (!fromOk) return false
  return exact ? x.kind === 'question' : true
}

// Live operator messages nobody has answered yet (at most THINK_S seconds old), by id.
// Arrival times are taken from when this panel first saw each message.
function usePendingThinking(chat, initSeq, gen) {
  const seenAt = useRef(new Map())
  const [, bump] = useState(0)
  useEffect(() => {
    seenAt.current = new Map()
  }, [gen])
  const nowMs = performance.now()
  const exact = chat.some((m) => m.reply_to)
  const pending = new Set()
  let soonest = Infinity
  for (let i = Math.max(0, chat.length - 40); i < chat.length; i++) {
    const m = chat[i]
    if (!seenAt.current.has(m.key)) seenAt.current.set(m.key, nowMs)
    if (m.from !== 'operator' || m.kind === 'answer' || !m.id || m.key <= initSeq) continue
    const left = THINK_S * 1000 - (nowMs - seenAt.current.get(m.key))
    if (left <= 0) continue
    let done = false
    for (let j = i + 1; j < chat.length && !done; j++) done = answersMsg(chat[j], m, exact)
    if (done) continue
    pending.add(m.id)
    soonest = Math.min(soonest, left)
  }
  // Re-render when the oldest one times out, so it disappears quietly.
  useEffect(() => {
    if (soonest === Infinity) return
    const id = setTimeout(() => bump((n) => n + 1), soonest + 50)
    return () => clearTimeout(id)
  }, [soonest])
  return pending
}

export function CommsPanel({ hidden, inputRef, selected = null }) {
  const chat = useSim((s) => s.chat)
  const meta = useSim((s) => s.roverMeta)
  const requests = useSim((s) => s.requests)
  const initSeq = useSim((s) => s.chatInitSeq)
  const v3 = useSim((s) => s.v3)
  const gen = useSim((s) => s.gen)
  const answers = useAnswers()
  const [to, setTo] = useState(selected || 'all')
  const [hl, setHl] = useState(null) // operator message id whose question and replies are lit
  const list = useRef(null)
  const atBottom = useRef(true)
  const now = useNow(500, !hidden && requests.length > 0)
  const byId = useMemo(() => Object.fromEntries(meta.map((m) => [m.id, m])), [meta])
  const reqById = useMemo(() => Object.fromEntries(requests.map((q) => [q.id, q])), [requests])
  const byMsgId = useMemo(() => {
    const out = {}
    for (const m of chat) if (m.id) out[m.id] = m
    return out
  }, [chat])
  const thinking = usePendingThinking(chat, initSeq, gen)

  // The addressee follows the selected rover (All when none is selected); the menu
  // still picks anyone.
  useEffect(() => {
    setTo(selected || 'all')
  }, [selected])

  const jump = (id) => {
    const el = list.current?.querySelector(`[data-mid="${CSS.escape(id)}"]`)
    if (!el) return
    el.scrollIntoView({ block: 'center', behavior: 'smooth' })
    el.classList.remove('ix-flash')
    void el.offsetWidth
    el.classList.add('ix-flash')
  }

  useLayoutEffect(() => {
    atBottom.current = true
  }, [gen])
  // Oldest first, newest at the bottom by the input. Stay pinned to the bottom unless
  // someone scrolled up to read; jump there whenever the tab opens.
  useLayoutEffect(() => {
    if (hidden) return
    const el = list.current
    if (el && atBottom.current) el.scrollTop = el.scrollHeight
  }, [chat, hidden, gen, answers, thinking.size])
  useLayoutEffect(() => {
    if (!hidden) atBottom.current = true
  }, [hidden])

  const pick = (id) => {
    setTo(id)
    inputRef.current?.focus()
  }

  return (
    <div className="comms" id="panel-comms" role="tabpanel" aria-labelledby="tab-comms" hidden={hidden}>
      <ol
        className="chat"
        ref={list}
        aria-live={hidden ? 'off' : 'polite'}
        onScroll={(e) => {
          const el = e.currentTarget
          atBottom.current = el.scrollHeight - el.scrollTop - el.clientHeight < 40
        }}
      >
        {chat.map((m, i) => {
          const isQ = m.kind === 'question' && m.request_id
          const req = isQ ? reqById[m.request_id] || null : null
          // A reply quotes its question, except when it directly continues a run of
          // replies to the same question (then a thin line ties it to the first).
          const q = m.reply_to && m.from !== 'operator' ? byMsgId[m.reply_to] || null : null
          const prev = chat[i - 1]
          const cont = !!q && !!prev && prev.reply_to === m.reply_to && prev.from !== 'operator'
          const quote = q && !cont && (m.kind === 'reply' || prev?.id !== q.id) ? q : null
          return (
            <ChatMsg
              key={m.key}
              m={m}
              byId={byId}
              live={m.key > initSeq}
              req={req}
              answer={isQ ? answers[m.request_id] || null : null}
              res={isQ && !req ? resolution(m, chat, answers) : null}
              now={req ? now : 0}
              onPick={pick}
              quote={quote}
              cont={cont}
              thinking={m.from === 'operator' && !!m.id && thinking.has(m.id)}
              hl={!!hl && (m.id === hl || m.reply_to === hl)}
              onJump={jump}
              onHover={setHl}
            />
          )
        })}
        {!chat.length && (
          <li className="chat-empty">
            <b>Ask the rovers anything, or give them orders, in plain words.</b>
            <span>
              {v3
                ? 'Type your own question below ("What did you find near the lake?"), or tap a suggestion. Rovers answer from what they have sensed, confirm orders, refuse with their reason, or ask you when unsure.'
                : 'This sim has not sent any chat yet. Older sims (before protocol version 3) have no operator chat.'}
            </span>
          </li>
        )}
      </ol>
      <Composer to={to} setTo={setTo} inputRef={inputRef} />
    </div>
  )
}
