import { describe, expect, it } from 'vitest'
import { formatDetail } from './api'

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
