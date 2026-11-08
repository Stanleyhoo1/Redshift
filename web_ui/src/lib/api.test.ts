import { afterEach, describe, expect, it, vi } from 'vitest'
import { api, formatDetail } from './api'

describe('formatDetail', () => {
  it('passes strings through', () => expect(formatDetail('No running mission')).toBe('No running mission'))
  it('flattens pydantic error lists', () =>
    expect(
      formatDetail([
        { loc: ['body', 'gravity_g'], msg: 'Input should be less than or equal to 3' },
        { loc: ['body'], msg: 'Field required' },
      ]),
    ).toBe('gravity_g: Input should be less than or equal to 3; Field required'))
  it('never yields [object Object]', () => expect(formatDetail({ a: 1 })).toBe('{"a":1}'))
  it('handles empty', () => expect(formatDetail(null)).toBe(''))
})

describe('inject endpoints', () => {
  const calls: { url: string; init: RequestInit }[] = []
  const stubFetch = (body: unknown, status = 200) => {
    calls.length = 0
    vi.stubGlobal('fetch', async (url: string, init: RequestInit) => {
      calls.push({ url, init })
      return new Response(JSON.stringify(body), { status })
    })
  }
  afterEach(() => vi.unstubAllGlobals())

  it('posts a pillar collapse for the given rover', async () => {
    stubFetch({ ok: true, detail: 'Pillar collapse armed across rover_1 planned path' })
    const r = await api.injectPillar('rover_1')
    expect(r.ok).toBe(true)
    expect(calls).toHaveLength(1)
    expect(calls[0].url).toBe('/api/inject/pillar')
    expect(calls[0].init.method).toBe('POST')
    expect(JSON.parse(String(calls[0].init.body))).toEqual({ robot: 'rover_1' })
  })
  it('lets the backend pick the rover when none is given', async () => {
    stubFetch({ ok: true })
    await api.injectPillar()
    expect(JSON.parse(String(calls[0].init.body))).toEqual({})
  })
  it('posts a dust storm with an empty body (the backend owns the storm parameters)', async () => {
    stubFetch({ ok: true, detail: 'Dust storm' })
    await api.injectStorm()
    expect(calls[0].url).toBe('/api/inject/storm')
    expect(calls[0].init.method).toBe('POST')
    expect(JSON.parse(String(calls[0].init.body))).toEqual({})
  })
  it('surfaces a rejected request as an error with the backend detail', async () => {
    stubFetch({ ok: false, detail: 'No running mission' }, 409)
    await expect(api.injectStorm()).rejects.toThrow('No running mission')
  })
})
