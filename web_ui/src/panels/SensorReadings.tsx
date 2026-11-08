import type { ReactNode } from 'react'
import { useSelected } from '../lib/hooks'
import { faultLabel, fmt, fmtHours, fmtSigned, robotName } from '../lib/format'
import { SENSOR_WINDOW_S, sensorHistory } from '../lib/history'
import { autoRange, batteryTone, healthTone, sparkSegments, sparkY, timeToEmptyH, type Range, type SensorKey, type SensorSample, type Tone } from '../lib/sensors'
import { useStore } from '../lib/store'
import { powerSave, WIND_ENTER_MPS, windReading } from '../lib/storm'
import { C, sensorColor, TONE_COLOR } from '../lib/theme'
import type { RobotTelemetry } from '../lib/types'
import { Badge, Panel, SensorChip } from '../ui/Panel'

const SPAN = SENSOR_WINDOW_S

interface Series {
  key: SensorKey
  color: string
  dashed?: boolean
}

interface Channel {
  group: string
  label: ReactNode
  series: Series[]
  value: (s: SensorSample) => ReactNode
  tone?: (s: SensorSample) => Tone
  minSpan: number
  include?: number[]
  clamp?: Range
  refs?: number[]
  /** two-line label */
  tall?: boolean
}

const toneOf = (t: Tone | undefined) => (t === 'crit' ? C.ha : t === 'warn' ? C.nad : C.bone)

const Unit = ({ children }: { children: ReactNode }) => <span className="ml-0.5 text-[11px] font-[500] text-haze">{children}</span>

function channels(r: RobotTelemetry, gravity: number): Channel[] {
  const alarm = r.tip_alarm_deg || 30
  const scout = r.type === 'scout'
  const imu: Channel[] = [
    {
      group: 'IMU',
      label: 'Acceleration',
      series: [{ key: 'accel_g', color: C.bone }],
      value: (s) => (
        <>
          {fmt(s.accel_g, 2)}
          <Unit>g</Unit>
        </>
      ),
      // shocks: more than 1.2 g away from the mission gravity (§8.7 health model)
      tone: (s) => (s.accel_g == null ? 'ok' : Math.abs(s.accel_g - gravity) > 1.2 ? 'crit' : Math.abs(s.accel_g - gravity) > 0.6 ? 'warn' : 'ok'),
      minSpan: 0.8,
      include: [gravity],
      refs: [gravity],
    },
    {
      group: 'IMU',
      label: 'Angular rate',
      series: [{ key: 'gyro_dps', color: C.bone }],
      value: (s) => (
        <>
          {fmt(s.gyro_dps, 1)}
          <Unit>°/s</Unit>
        </>
      ),
      tone: (s) => ((s.gyro_dps ?? 0) > 120 ? 'crit' : (s.gyro_dps ?? 0) > 45 ? 'warn' : 'ok'),
      minSpan: 12,
      include: [0],
    },
    {
      group: 'IMU',
      label: (
        <>
          <span className="text-bone">Roll</span>, <span className="text-hb">pitch</span>
        </>
      ),
      series: [
        { key: 'pitch', color: C.hb },
        { key: 'roll', color: C.bone },
      ],
      value: (s) => (
        <>
          <span className="text-bone">{signedDeg(s.roll)}</span>
          <span className="mx-0.5 text-dim">/</span>
          <span className="text-hb">{signedDeg(s.pitch)}</span>
        </>
      ),
      tone: (s) => {
        const w = Math.max(Math.abs(s.roll ?? 0), Math.abs(s.pitch ?? 0))
        return w >= alarm ? 'crit' : w >= 0.7 * alarm ? 'warn' : 'ok'
      },
      minSpan: 8,
      include: [0],
    },
  ]
  const range: Channel[] = scout
    ? [
        {
          group: 'ALT',
          label: 'Altitude AGL',
          series: [{ key: 'altitude_agl', color: '#c9b8ff' }],
          value: (s) => (
            <>
              {fmt(s.altitude_agl, 1)}
              <Unit>m</Unit>
            </>
          ),
          minSpan: 4,
          include: [0],
        },
        // §10.3: the drone's own estimate from drift (commanded vs odometry velocity); shown once it reports one
        ...(windReading(r)
          ? [
              {
                group: 'WIND',
                label: 'Wind, estimated',
                series: [{ key: 'wind_mps', color: sensorColor('WIND') }],
                value: (s: SensorSample) => (
                  <>
                    {fmt(s.wind_mps, 1)}
                    <Unit>m/s</Unit>
                  </>
                ),
                // amber from where the drone's own descent rule starts counting
                tone: (s: SensorSample): Tone => (s.wind_mps != null && s.wind_mps > WIND_ENTER_MPS ? 'warn' : 'ok'),
                minSpan: 4,
                include: [0],
                refs: [WIND_ENTER_MPS],
              } satisfies Channel,
            ]
          : []),
      ]
    : [
        {
          group: 'LIDAR',
          label: 'Nearest return',
          series: [{ key: 'lidar_nearest_m', color: '#5eebc0' }],
          value: (s) => (
            <>
              {fmt(s.lidar_nearest_m, 1)}
              <Unit>m</Unit>
            </>
          ),
          tone: (s) => (s.lidar_nearest_m != null && s.lidar_nearest_m < 1.0 && r.state !== 'RESCUE_DOCKING' && r.state !== 'REPAIRING' ? 'warn' : 'ok'),
          minSpan: 4,
          include: [0],
          clamp: { lo: 0, hi: 30 },
        },
        {
          group: 'LIDAR',
          label: 'Returns',
          series: [{ key: 'lidar_returns_pct', color: '#5eebc0' }],
          value: (s) => (
            <>
              {fmt(s.lidar_returns_pct, 0)}
              <Unit>%</Unit>
            </>
          ),
          tone: (s) => (s.lidar_returns_pct == null ? 'ok' : s.lidar_returns_pct < 35 ? 'crit' : s.lidar_returns_pct < 60 ? 'warn' : 'ok'),
          minSpan: 20,
          clamp: { lo: 0, hi: 100 },
        },
      ]
  const cam: Channel = {
    group: 'CAM',
    label: 'Confidence',
    series: [{ key: 'camera_conf', color: '#8ec5ff' }],
    value: (s) => fmt(s.camera_conf, 2),
    tone: (s) => (s.camera_conf == null ? 'ok' : s.camera_conf < 0.4 ? 'crit' : s.camera_conf < 0.7 ? 'warn' : 'ok'),
    minSpan: 0.3,
    clamp: { lo: 0, hi: 1 },
  }
  const drive: Channel[] = scout
    ? [
        {
          group: 'ODOM',
          label: 'Ground speed',
          series: [
            { key: 'v_cmd', color: C.haze, dashed: true },
            { key: 'v_ground', color: C.bone },
          ],
          value: (s) => (
            <>
              {fmt(s.v_ground, 1)}
              <Unit>m/s</Unit>
            </>
          ),
          minSpan: 2,
          include: [0],
        },
      ]
    : [
        {
          group: 'ODOM',
          label: (
            <span className="block leading-tight">
              Speed
              <span className="block text-[10.5px] text-dim">
                <span className="text-haze">cmd</span>, <span className="text-hb">wheel</span>, <span className="text-bone">ground</span>
              </span>
            </span>
          ),
          tall: true,
          series: [
            { key: 'v_cmd', color: C.haze, dashed: true },
            { key: 'v_wheel', color: C.hb },
            { key: 'v_ground', color: C.bone },
          ],
          value: (s) => (
            <span className="text-[13px]">
              <span className="text-haze">{fmt(s.v_cmd, 2)}</span> <span className="text-hb">{fmt(s.v_wheel, 2)}</span> {fmt(s.v_ground, 2)}
            </span>
          ),
          // commanded but not delivered by the wheels: the drivetrain fault signature
          tone: (s) => (s.v_cmd != null && s.v_wheel != null && s.v_cmd > 0.15 && s.v_wheel < 0.5 * s.v_cmd ? 'crit' : 'ok'),
          minSpan: 0.4,
          include: [0],
        },
        {
          group: 'ODOM',
          label: 'Wheel slip',
          series: [{ key: 'slip', color: C.bone }],
          value: (s) => fmt(s.slip, 2),
          tone: (s) => ((s.slip ?? 0) > 0.6 ? 'crit' : (s.slip ?? 0) > 0.3 ? 'warn' : 'ok'),
          minSpan: 0.15,
          include: [0],
          clamp: { lo: 0, hi: 1 },
        },
      ]
  const power: Channel[] = [
    scout
      ? {
          group: 'PWR',
          label: 'Hover load',
          series: [{ key: 'load_w', color: '#f2dc6b' }],
          value: (s) => (
            <>
              {fmt(s.load_w, 0)}
              <Unit>W</Unit>
            </>
          ),
          minSpan: 30,
          include: [0],
        }
      : {
          group: 'PWR',
          label: 'Solar panel',
          series: [{ key: 'solar_w', color: '#f2dc6b' }],
          value: (s) => (
            <>
              {fmt(s.solar_w, 0)}
              <Unit>W</Unit>
            </>
          ),
          minSpan: 40,
          include: [0],
        },
    {
      group: 'PWR',
      label: 'Net power',
      series: [{ key: 'net_w', color: '#f2dc6b' }],
      value: (s) => (
        <span style={{ color: s.net_w == null ? undefined : s.net_w >= 0 ? C.o3 : C.nad }}>
          {fmtSigned(s.net_w, 0)}
          <Unit>W</Unit>
        </span>
      ),
      minSpan: 40,
      include: [0],
      refs: [0],
    },
  ]
  return [...imu, ...range, cam, ...drive, ...power]
}

const signedDeg = (v: number | null) => (v == null ? '—' : `${v < 0 ? '−' : '+'}${Math.abs(v).toFixed(1)}°`)

/** Live instrument readings of the selected robot with a 20 s trace per channel. */
export function SensorReadings() {
  const [sel, robot] = useSelected()
  const gravity = useStore((s) => s.telemetry?.mission.config?.gravity_g ?? s.lastConfig?.gravity_g ?? 1)
  const hist = sel ? sensorHistory(sel) : []
  const last = hist.length ? hist[hist.length - 1] : null
  const tEnd = last?.t ?? 0
  const ts = hist.map((h) => h.t)

  return (
    <Panel
      title={`Live sensors${sel ? `, ${robotName(sel, robot?.name)}` : ''}`}
      right={<span className="num text-[11.5px] whitespace-nowrap text-dim">15 Hz, last {SPAN} s</span>}
      bodyClass="flex flex-col"
    >
      {!robot || !last ? (
        <p className="px-4 py-6 text-center text-[13px] text-dim">Sensor readings appear once the selected robot reports telemetry.</p>
      ) : (
        <>
          <HealthStrip robot={robot} hist={hist} ts={ts} tEnd={tEnd} />
          <ModeStrip robot={robot} />
          <dl className="grid grid-cols-[44px_minmax(0,1fr)_auto_minmax(64px,0.95fr)] items-center gap-x-2 px-3 pt-1 pb-2" aria-label="Sensor channels">
            {channels(robot, gravity).map((ch, k, all) => {
              const first = k === 0 || all[k - 1].group !== ch.group
              const tone = ch.tone?.(last)
              return (
                <Row key={k} first={first && k > 0} tall={ch.tall}>
                  <span className="flex">{first ? <SensorChip tag={ch.group} /> : null}</span>
                  <dt className="truncate text-[12px] text-haze">{ch.label}</dt>
                  <dd className="font-semi-cond num text-right text-[15.5px] leading-none font-[700] whitespace-nowrap" style={{ color: toneOf(tone) }}>
                    {ch.value(last)}
                  </dd>
                  <Spark ch={ch} hist={hist} ts={ts} tEnd={tEnd} tone={tone} />
                </Row>
              )
            })}
          </dl>
        </>
      )}
    </Panel>
  )
}

/** One grid row; a hairline separates channel groups. */
function Row({ first, tall, children }: { first: boolean; tall?: boolean; children: ReactNode }) {
  return <div className={`col-span-4 grid ${tall ? 'h-[32px]' : 'h-[23px]'} grid-cols-subgrid items-center ${first ? 'border-t border-rule/60' : ''}`}>{children}</div>
}

function Spark({ ch, hist, ts, tEnd, tone, H = 18 }: { ch: Channel; hist: readonly SensorSample[]; ts: number[]; tEnd: number; tone?: Tone; H?: number }) {
  const W = 120
  const cols = ch.series.map((s) => hist.map((h) => h[s.key]))
  const range = autoRange(cols.flat(), ch.minSpan, ch.include, ch.clamp)
  if (!range) return <span className="text-right text-[11px] text-dim">no data</span>
  return (
    <svg viewBox={`0 0 ${W} ${H}`} preserveAspectRatio="none" className="h-[18px] w-full overflow-visible" aria-hidden>
      {tone === 'crit' && <rect x={0} y={0} width={W} height={H} fill={C.ha} fillOpacity={0.1} />}
      {(ch.refs ?? []).map((v) => (
        <line key={v} x1={0} x2={W} y1={sparkY(v, H, range)} y2={sparkY(v, H, range)} stroke={C.rule2} strokeDasharray="2 3" strokeWidth={1} vectorEffect="non-scaling-stroke" />
      ))}
      {ch.series.map((s, k) =>
        sparkSegments(ts, cols[k], tEnd, SPAN, W, H, range).map((pts, j) => (
          <polyline
            key={`${s.key}${j}`}
            points={pts}
            fill="none"
            stroke={s.color}
            strokeWidth={s.dashed ? 1 : 1.3}
            strokeDasharray={s.dashed ? '3 2' : undefined}
            strokeLinejoin="round"
            vectorEffect="non-scaling-stroke"
          />
        )),
      )}
    </svg>
  )
}

/** Modes the robot entered on its own readings (§10.3): a rover's battery conservation, the drone's wind descent. */
function ModeStrip({ robot }: { robot: RobotTelemetry }) {
  const pwr = powerSave(robot)
  const wind = windReading(robot)
  if (!pwr && !wind?.descent) return null
  return (
    <div className="flex flex-col gap-1 border-b border-rule/70 px-3 py-1.5 text-[11.5px] text-haze">
      {pwr && (
        <p className="flex min-w-0 items-center gap-1.5">
          <Badge color={sensorColor('PWR')}>POWER SAVE</Badge>
          <span className="num truncate">
            solar <span className="text-bone">{fmt(pwr.solar_w, 0)} W</span>: battery conservation, slower drive
          </span>
        </p>
      )}
      {wind?.descent && (
        <p className="flex min-w-0 items-center gap-1.5">
          <Badge color={sensorColor('WIND')}>WIND DESCENT</Badge>
          <span className="num truncate">
            {wind.mps != null ? (
              <>
                wind <span className="text-bone">{fmt(wind.mps, 1)} m/s</span> from drift:{' '}
              </>
            ) : null}
            flying lower, out of the shear
          </span>
        </p>
      )}
    </div>
  )
}

const HEALTH_CH: Channel = { group: 'HEALTH', label: 'Health', series: [{ key: 'health_pct', color: C.bone }], value: () => null, minSpan: 20, include: [100, 0], clamp: { lo: 0, hi: 100 }, refs: [15, 60] }
const BATT_CH: Channel = { group: 'PWR', label: 'Battery', series: [{ key: 'battery_pct', color: '#f2dc6b' }], value: () => null, minSpan: 4, clamp: { lo: 0, hi: 100 } }

/** Health and battery at a glance, with fault codes from the motor controllers. */
function HealthStrip({ robot, hist, ts, tEnd }: { robot: RobotTelemetry; hist: readonly SensorSample[]; ts: number[]; tEnd: number }) {
  const h = robot.health
  if (!h) {
    return <p className="border-b border-rule/70 px-3 py-2.5 text-[12px] text-dim">Health and power telemetry not reported by this backend.</p>
  }
  const faults = h.faults ?? []
  const ht = healthTone(h.pct, h.critical)
  const bt = batteryTone(h.battery_pct)
  const tte = timeToEmptyH(h.battery_wh, h.net_w)
  const charging = (h.net_w ?? 0) > 0.5
  return (
    <div className={`border-b border-rule/70 px-3 pt-2 pb-2 ${ht === 'crit' ? 'bg-ha/10' : ''}`}>
      <div className="grid grid-cols-2 gap-3">
        <div className="min-w-0">
          <div className="flex items-baseline justify-between gap-2">
            <span className="text-[12px] text-haze">Health</span>
            {ht === 'crit' && <span className="font-cond animate-blink text-[12px] font-[800] tracking-[0.04em] text-ha">CRITICAL</span>}
          </div>
          <div className="flex items-end gap-2">
            <span className="font-cond num text-[30px] leading-[0.95] font-[850]" style={{ color: ht === 'ok' ? C.bone : TONE_COLOR[ht] }}>
              {fmt(h.pct, 0)}
              <span className="text-[16px] font-[650] text-haze">%</span>
            </span>
            <span className="mb-0.5 min-w-0 flex-1">
              <Spark ch={HEALTH_CH} hist={hist} ts={ts} tEnd={tEnd} />
            </span>
          </div>
          <div className="num mt-1 text-[11.5px] text-haze">
            structure <span style={{ color: toneOf(healthTone(h.structure_pct)) }}>{fmt(h.structure_pct, 0)} %</span>, drive{' '}
            <span style={{ color: toneOf(healthTone(h.drive_pct)) }}>{fmt(h.drive_pct, 0)} %</span>
          </div>
        </div>
        <div className="min-w-0">
          <div className="flex items-baseline justify-between gap-2">
            <span className="text-[12px] text-haze">Battery</span>
            <span className="num text-[11.5px] text-dim">
              {fmt(h.battery_wh, 0)} / {fmt(h.capacity_wh, 0)} Wh
            </span>
          </div>
          <div className="flex items-end gap-2">
            <span className="font-cond num text-[30px] leading-[0.95] font-[850]" style={{ color: bt === 'ok' ? C.bone : TONE_COLOR[bt] }}>
              {fmt(h.battery_pct, 1)}
              <span className="text-[16px] font-[650] text-haze">%</span>
            </span>
            <span className="mb-0.5 min-w-0 flex-1">
              <Spark ch={BATT_CH} hist={hist} ts={ts} tEnd={tEnd} />
            </span>
          </div>
          <div className="num mt-1 truncate text-[11.5px] text-haze">
            {charging ? (
              <span className="text-o3">charging {fmtSigned(h.net_w, 0)} W</span>
            ) : tte != null ? (
              <>
                <span className="text-nad">{fmtSigned(h.net_w, 0)} W</span>, empty in <span className="text-bone">{fmtHours(tte)}</span>
              </>
            ) : (
              'power balanced'
            )}
          </div>
        </div>
      </div>
      <div className="mt-1.5 flex min-h-[20px] flex-wrap items-center gap-1">
        <SensorChip tag="DIAG" />
        {faults.length === 0 ? (
          <span className="ml-1 text-[11.5px] text-dim">{robot.type === 'scout' ? 'Flight controller nominal' : 'Motor controllers nominal'}</span>
        ) : (
          faults.map((f) => (
            <span key={f} title={faultLabel(f)} className="rounded-[2px] border border-ha/60 bg-ha/15 px-1 py-[2px] font-mono text-[10px] leading-none text-ha [font-stretch:82%]">
              {f}
            </span>
          ))
        )}
        {(h.shocks ?? 0) > 0 && (
          <span className="num ml-auto text-[11px] whitespace-nowrap text-dim">
            {h.shocks} shock{h.shocks === 1 ? '' : 's'}, peak <span className={(h.peak_g ?? 0) > 3 ? 'text-ha' : 'text-haze'}>{fmt(h.peak_g, 1)} g</span>
          </span>
        )}
      </div>
    </div>
  )
}
