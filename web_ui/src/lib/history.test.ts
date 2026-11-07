import { describe, expect, it } from 'vitest'
import { distToPolyline, isDrasticReplan, polylineLength, remainingPath } from './history'
import type { XY } from './types'

const line = (x0: number, y0: number, x1: number, y1: number, n = 20): XY[] =>
  Array.from({ length: n + 1 }, (_, k) => [x0 + ((x1 - x0) * k) / n, y0 + ((y1 - y0) * k) / n] as XY)

describe('path geometry', () => {
  it('measures polyline length and point distance', () => {
    expect(polylineLength(line(0, 0, 10, 0))).toBeCloseTo(10, 9)
    expect(distToPolyline([5, 3], line(0, 0, 10, 0))).toBeCloseTo(3, 9)
  })
  it('drops the part of the path behind the robot', () => {
    const rest = remainingPath(line(0, 0, 10, 0, 10), [6.2, 0.4])
    expect(rest[0]).toEqual([6, 0])
  })
})

describe('isDrasticReplan', () => {
  it('ignores a near-identical re-publish of the same plan', () => {
    const a = line(0, 0, 20, 0)
    const b = a.map(([x, y]) => [x + 0.2, y + 0.3] as XY)
    expect(isDrasticReplan(a, b, [0, 0])).toBe(false)
  })
  it('flags a detour around a blockage', () => {
    const a = line(0, 0, 20, 0)
    const b: XY[] = [...line(0, 0, 8, 6, 8), ...line(8, 6, 20, 0, 8)]
    expect(isDrasticReplan(a, b, [0, 0])).toBe(true)
  })
  it('ignores divergence when the old plan was almost finished', () => {
    const a = line(0, 0, 20, 0)
    const b = line(19, 0, 19, 30)
    expect(isDrasticReplan(a, b, [18.5, 0])).toBe(false)
  })
})
