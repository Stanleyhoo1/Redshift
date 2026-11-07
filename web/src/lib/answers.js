import { appStore, useApp } from '../state/store.js'
import { postAnswer } from '../net/connection.js'

// Answers this client sends to clarification requests (POST /api/answer). The card and
// the question in the comms thread share this, so answering in one updates the other.

function setAnswer(id, value) {
  appStore.patch({ answers: { ...appStore.get().answers, [id]: value } })
}

export async function sendAnswer(id, option) {
  const cur = appStore.get().answers[id]
  if (cur && (cur.state === 'pending' || cur.state === 'sent')) return
  setAnswer(id, { state: 'pending', option })
  try {
    await postAnswer(id, option)
    setAnswer(id, { state: 'sent', option, at: performance.now() / 1000 })
  } catch (err) {
    setAnswer(id, { state: 'error', option, error: err?.message || 'Answer refused', status: err?.status ?? 0 })
  }
}

export const useAnswer = (id) => useApp((a) => (id ? a.answers[id] || null : null))
export const useAnswers = () => useApp((a) => a.answers)

export const optionLabel = (q, id) => q?.options?.find((o) => o.id === id)?.label || id || ''
