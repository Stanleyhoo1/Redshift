// Typed REST client for docs/protocol.md §6. All paths are same-origin (Vite proxies /api in dev).
import type {
  DeployResponse,
  MissionConfig,
  MissionStatusResponse,
  OkResponse,
  Options,
  PhysicsPreview,
} from './types'

export class ApiError extends Error {
  status: number
  constructor(message: string, status: number) {
    super(message)
    this.status = status
  }
}

async function request<T>(path: string, init?: RequestInit & { timeoutMs?: number }): Promise<T> {
  const ctrl = new AbortController()
  const timer = setTimeout(() => ctrl.abort(), init?.timeoutMs ?? 8000)
  const outer = init?.signal
  if (outer) outer.addEventListener('abort', () => ctrl.abort(), { once: true })
  try {
    const res = await fetch(path, {
      ...init,
      signal: ctrl.signal,
      headers: { 'Content-Type': 'application/json', ...(init?.headers ?? {}) },
    })
    const text = await res.text()
    let body: unknown = null
    try {
      body = text ? JSON.parse(text) : null
    } catch {
      body = null
    }
    if (!res.ok) {
      const raw = body && typeof body === 'object' && 'detail' in body ? (body as { detail: unknown }).detail : text
      throw new ApiError(formatDetail(raw) || `HTTP ${res.status}`, res.status)
    }
    return body as T
  } catch (e) {
    if (e instanceof ApiError) throw e
    if ((e as Error).name === 'AbortError' && !outer?.aborted) {
      throw new ApiError(`Backend did not answer ${path} in time`, 0)
    }
    if ((e as Error).name === 'AbortError') throw e
    throw new ApiError(`Backend unreachable (${path})`, 0)
  } finally {
    clearTimeout(timer)
  }
}

/** FastAPI/pydantic validation errors arrive as a list of {loc, msg}; show them as text, not [object Object]. */
export function formatDetail(detail: unknown): string {
  if (detail == null) return ''
  if (typeof detail === 'string') return detail
  if (Array.isArray(detail)) {
    return detail
      .map((d) => {
        if (d && typeof d === 'object' && 'msg' in d) {
          const loc = Array.isArray((d as { loc?: unknown }).loc) ? (d as { loc: unknown[] }).loc.filter((x) => x !== 'body') : []
          return `${loc.length ? loc.join('.') + ': ' : ''}${String((d as { msg: unknown }).msg)}`
        }
        return typeof d === 'string' ? d : JSON.stringify(d)
      })
      .join('; ')
  }
  return typeof detail === 'object' ? JSON.stringify(detail) : String(detail)
}

const post = <T>(path: string, body: unknown = {}) =>
  request<T>(path, { method: 'POST', body: JSON.stringify(body), timeoutMs: 15000 })

export const api = {
  options: (signal?: AbortSignal) => request<Options>('/api/options', { signal }),
  physicsPreview: (g: number, signal?: AbortSignal) =>
    request<PhysicsPreview>(`/api/physics_preview?g=${g.toFixed(2)}`, { signal, timeoutMs: 4000 }),
  deploy: (config: MissionConfig) => post<DeployResponse>('/api/mission/deploy', config),
  abort: () => post<OkResponse>('/api/mission/abort'),
  status: (signal?: AbortSignal) => request<MissionStatusResponse>('/api/mission/status', { signal }),
  injectRockslide: (robot?: string) => post<OkResponse>('/api/inject/rockslide', robot ? { robot } : {}),
  injectAnomaly: (body: { dust_level?: number; duration_s?: number } = {}) =>
    post<OkResponse>('/api/inject/anomaly', body),
  injectTip: (robot?: string) => post<OkResponse>('/api/inject/tip', robot ? { robot } : {}),
  /** §8.9: debris strike → critical damage on `robot` (default: the first rover) */
  injectDamage: (robot?: string) => post<OkResponse>('/api/inject/damage', robot ? { robot } : {}),
  setSpeed: (speed: number) => post<OkResponse & { speed?: number }>('/api/sim/speed', { speed }),
}
