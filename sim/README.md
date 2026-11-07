# Redshift simulation backend

A fleet of rovers explores an unknown super-Earth at 2 g. Each rover only knows what its own
sensors tell it. The rovers build a shared map from LiDAR, pick where to go next, learn how steep
a slope they can really climb, find and sample science targets, and drive home to charge. They
make every decision on their own; nothing is scripted. The world can also change while they work
(rockfalls, soft sand, dust storms, motor faults) and they have to notice and cope. The web UI
(in `../web`) shows the world filling in. The message format is in `../PROTOCOL.md`.

The brain has two tiers. Fast hand-written rules decide everything, 5 times a second, and keep the
rovers safe. A small Gemma model running on your own machine gives advice beside them: which goal
to pick, what an odd sensor reading means (with the camera picture), and what the operator's typed
messages mean. The rules check every piece of advice. See "Two-tier brain" below and `GEMMA.md`.

## What each module does

| File | What it does |
|---|---|
| `redshift/config.py` | All settings in one place. Reads the env vars below. |
| `redshift/terrain.py` | Makes the ground: plains, hills, ridges, craters (each with a gentle way in), rocks, and 6 to 9 science targets in clusters. Same seed, same world. |
| `redshift/world.py` | Builds the MuJoCo physics model: the ground as height-map tiles, rocks, the lander pad, target markers, hidden rockfall boulders, and the rovers (4 wheels, skid steering, front axle that rocks so all wheels touch, an IMU). |
| `redshift/mapping.py` | The shared fleet map. Turns LiDAR hits into heights and slopes for each grid cell, notices when mapped ground changes, keeps a "soft ground" layer, and finds frontiers (edges of the known area). |
| `redshift/planner.py` | A* path planning on the map. Steep cells, obstacles and soft ground are off limits. The search can stop when its time is up and carry on in the next cycle. |
| `redshift/rover.py` | One rover's brain: sensors, driving, battery, and its state (exploring, investigating, returning, charging, recovering, idle, fault). |
| `redshift/fleet.py` | Team decisions: who goes to which frontier, the shared slope limit, the target list, soft ground reports, who takes over from a broken rover, and the event log. |
| `redshift/hazards.py` | Changes the world on request: rockfall, sand, dust storm, rover fault. Only the UI sees this list; the rovers must find out for themselves. |
| `redshift/sim.py` | Runs it all: steps the physics, runs the rovers' decisions 5 times per sim second, and builds the messages for the UI. Also a command line runner that can script hazards. |
| `redshift/camera.py` | Live pictures from each rover's mast camera (MuJoCo renders), for the UI and for Gemma. |
| `redshift/gemma.py` | Talks to the local Gemma model (Ollama), or a fake model for demos and tests: prompts, answer checks, timeouts, a keyword parser for typed commands. |
| `redshift/advisor.py` | Tier 2: asks Gemma at the right moments, applies its advice when it is safe, runs the operator chat and the questions to the operator. |
| `redshift/server.py` | The web server (FastAPI): WebSocket `/ws`, `GET /api/health`, `GET /api/truth`, `POST /api/reset`, `POST /api/hazard`, `POST /api/chat`, `POST /api/answer`, and the camera images from `camera.py`. Starts Gemma, then runs the sim in the background at `SIM_SPEED` times real time. |
| `scripts/climb_test.py` | Drives a rover up ramps of known angle to check how steep it can climb. |
| `scripts/gemma_bench.py` | Measures Gemma's speed and answers against a real Ollama. |
| `tests/test_smoke.py` | Runs 400 sim seconds with no hazards and checks the results and every message. |
| `tests/test_hazards.py` | Runs every hazard in turn and checks how the rovers react, plus the v2 message fields and `/api/hazard`. |
| `tests/test_camera.py` | Camera renders and the camera endpoint. |
| `tests/test_gemma.py` | `gemma.py` against a stub Ollama server. |
| `tests/test_brain.py` | Tier 2 with the fake model: goal switches, anomaly checks, questions, every chat command, Gemma offline, v3 messages, speed and budget. |

## How to run

Install (Python 3.10 to 3.13):

```bash
cd sim
pip install -r requirements.txt
```

Server with the 3D MuJoCo viewer (on your own machine, not in Docker):

```bash
python -m redshift.server --viewer            # or VIEWER=1 python -m redshift.server
mjpython -m redshift.server --viewer          # on macOS the viewer needs mjpython
```

Server only (this is what Docker runs):

```bash
uvicorn redshift.server:app --host 0.0.0.0 --port 8000
GEMMA_FAKE=1 uvicorn redshift.server:app --port 8000      # fake Gemma: the whole pipeline without Ollama
```

Talk to the rovers (the UI's chat box does the same), and answer their questions:

```bash
curl -X POST localhost:8000/api/chat -d '{"text": "rover 2, check out the area north of the lander"}'
curl -X POST localhost:8000/api/chat -d '{"text": "go to 10, -5", "rover": "r1"}'
curl -X POST localhost:8000/api/answer -d '{"id": "q1", "option": "reroute"}'
```

Change the world while it runs:

```bash
curl -X POST localhost:8000/api/hazard -d '{"type": "rockfall"}'
curl -X POST localhost:8000/api/hazard -d '{"type": "sand", "rover": "r1"}'
curl -X POST localhost:8000/api/hazard -d '{"type": "dust_storm"}'
curl -X POST localhost:8000/api/hazard -d '{"type": "rover_fault", "rover": "r3"}'
```

The answer is `{"ok": true, "hazard": {...}}`, or HTTP 400 with a reason (unknown type, nowhere
sensible to put it right now, or it would leave no working rover). Without `"rover"` the sim
picks the exploring rover with the longest route ahead (or a random working one for a fault).

Headless, as fast as possible, with a summary and the event log at the end. Hazards can be
scripted as `TYPE@SECONDS[:ROVER]` (retried for up to 60 s if there is nowhere to put it yet):

```bash
python -m redshift.sim --seconds 900 --seed 7
python -m redshift.sim --seconds 400 --hazard rockfall@60 --hazard sand@120:r1 \
       --hazard dust_storm@200 --hazard rover_fault@260:r3 --ticks
```

`--ticks` also builds the UI messages and reports their size. `--model-clock` makes runs repeat
exactly (see "Edge compute budget"). `--gemma` attaches Gemma (from the `GEMMA_*` variables) and
`--say` scripts operator messages; the chat is printed at the end:

```bash
GEMMA_FAKE=1 python -m redshift.sim --gemma --seconds 200 --hazard rockfall@60 \
       --say "30:rover 2, go check out the area north of the lander" --say "45:r1 go to 10, -5"
```

Tests (about 2 minutes):

```bash
python -m pytest -q tests/        # add -s to see the event log
python tests/test_smoke.py        # same checks without pytest
python scripts/climb_test.py      # ramp test for the rover tuning
```

## Environment variables

| Var | Default | Meaning |
|---|---|---|
| `SEED` | 42 | terrain and target layout |
| `NUM_ROVERS` | 3 | rovers in the fleet (1 to 8) |
| `SIM_SPEED` | 3.0 | sim seconds per wall second |
| `MAP_SIZE` | 80 | map width in metres |
| `GRAVITY_G` | 2.0 | gravity in Earth g |
| `VIEWER` | 0 | 1 = open the native MuJoCo viewer |
| `BRAIN_BUDGET_MS` | 40 | compute budget per rover per decision cycle (ms) |
| `AUTO_RESET_S` | 0 | off by default. If set, once exploration is complete and every rover is parked, wait this many seconds, then start a new random planet (server only) |
| `GEMMA_ENABLED` | 1 | 0 = rules only (no Gemma, no questions; typed commands still work through the keyword parser) |
| `GEMMA_URL` | `http://host.docker.internal:11434` | Ollama; `http://localhost:11434` is tried next |
| `GEMMA_MODEL` | `gemma4:e2b` | falls back to `gemma4:e4b`, then `gemma3:4b`, if not installed |
| `GEMMA_FAKE` | 0 | 1 = fake model (fixed rules, 0.5 s delay): demos and tests without Ollama |
| `GEMMA_BUDGET_MS` | 3000 | stated latency budget per Gemma call (answers after `GEMMA_TIMEOUT_S`, 4 s, are dropped) |
| `GEMMA_ASK_EVERY_S` | 20 | a rover asks Gemma about its goal at most this often (sim s), or when it reaches a goal |
| `OPERATOR_TIMEOUT_S` | 20 | wall seconds a question waits for the operator before the rover takes the safe default |
| `OPERATOR_ASK_BELOW` | 0.75 | Gemma confidence below this makes the rover ask the operator |
| `OPERATOR_ASK_FIRST` | 1 | the first anomaly after start or reset always asks the operator (reliable demo trigger) |

More Gemma settings (timeouts, tokens, model loading) are in `GEMMA.md`.

## Sensors

Everything a rover knows comes from these. Each tick message carries what they read (`sensors`)
and up to 200 points of the latest LiDAR sweep (`scan`).

- **LiDAR** on a 0.75 m mast: 1008 rays, 8 m range, 5 sweeps a second. Hits build the shared map.
  `lidar_range` is how far the rover believes it can see, `lidar_hits` the returns in the sweep.
- **Spectrometer**: LiDAR hits on science targets within 8 m. `spectro` is the nearest one.
- **IMU**: a MuJoCo `framequat` (attitude, like an AHRS chip) and `gyro` on the chassis, with a
  little noise. Pitch, roll, tilt, heading and yaw rate all come from it.
- **Wheel encoders**: wheel joint speeds give `wheel_speed`. Body speed over the ground gives
  `ground_speed`. `slip` = 1 - ground speed / wheel speed (when the wheels turn faster than
  0.2 m/s), smoothed over about a second.
- **Motor feedback**: torque per motor gives `motor_load`; the motor controller's status shows a
  fault. **Wheel load** (contact force per wheel, like strain gauges) is read only while slip is
  suspected.
- **Bumper**: three short rays at the nose.
- **Position** is taken from the simulator, standing in for GPS-like localisation (no SLAM).

## Hazards and how the rovers handle them

The hazard list (`hazards` in every message) is for the UI only. The rovers never read it.

| Hazard | What happens in the world | How a rover finds out | What it does |
|---|---|---|---|
| Rockfall | 3 to 5 boulders drop from 5 m onto a rover's route 4 to 7 m ahead, in a wall with gaps too narrow to pass. The rover can always still get round and home. | LiDAR. The map watches mapped ground: when the highest return near a cell stands well above what was mapped, in 3 separate sweeps, the cell is rewritten. One noisy ray changes nothing; rough ground and rock edges are ignored so the map does not flicker. | Slows to 0.3 m/s at once, says what it saw ("r2 sees new rocks blocking its path 5 m ahead (LiDAR, 6 map cells changed)"), plans round them and says how much longer the new route is. The boulders become impassable for everyone. |
| Sand | A hidden soft patch (2.5 m radius) on gentle ground 3 to 5 m ahead. Inside it the wheels lose grip and the rover sinks in: low wheel friction, extra rolling resistance, and drag on the body (high when pushing into fresh sand, low when backing out along its own ruts). | Wheel slip: wheels turning, rover hardly moving, slip over 60% for 1.5 s, on gentle ground, with at least two wheels spinning that are on the ground and working hard. That rules out slopes (own stall logic), turns, and a rover hung up on a rock (its spinning wheels hang in the air). | Stops, reports "wheels spinning but not moving (slip 85%). Soft ground", marks a generous soft zone that every rover now avoids, backs out along its own trail, and plans round. Near known soft ground it trusts weaker slip evidence. Failsafe: if still stuck after 30 s it digs itself out. |
| Dust storm | 90 s. LiDAR range drops from 8 m to 3 m with some lost returns and noise. Dust also covers the lander's solar panels: charging runs at 30%. | Its own scans: no returns from beyond about 3 m for 3 sweeps in a row. Docked rovers see their battery filling slowly. | Becomes "unsure": slows to 0.3 m/s briefly, then drives at 0.4 m/s with gentle turns, stays on mapped ground (unknown ground costs much more to plan through), prefers near frontiers and makes no far trips to targets. A rover charging slowly raises its return margin from 15% to 25%. Each change is logged once, and again when it clears. |
| Rover fault | One rover's motor controller drops out for 90 s. The brakes lock the wheels. State `fault`. | Motor feedback (controller status). | It tells the fleet: its frontier and any target it was going to sample are released at once, the others re-plan, and one may take over its area. Then "r3 rebooted. Back online." and it carries on. A request that would leave no working rover is refused. |

Hazards stay until `/api/reset`. Exploration-complete and the auto reset keep working (a faulted
rover is not parked, so there is no reset while one is down).

**Uncertainty.** Anything perception did not expect (new obstacles on the path, sudden wheel slip,
LiDAR returns collapsing) goes through one method, `Rover.on_anomaly(t, kind, details)`. It stores
the anomaly in `rover.anomaly` (and `rover.anomaly_log`), slows the rover at once, before any
re-planning, and logs what was seen. Other code can listen by adding a function to
`rover.anomaly_hooks`; it is called with `(rover, anomaly)`. The anomaly is a dict like
`{"t": 61.4, "kind": "new_obstacle", "type": "new_obstacle", "at": [2.0, -28.0], "dist": 4.0, "cells": 5}`
(`kind` is `new_obstacle`, `slip` or `low_visibility`; `type` uses the names `new_obstacle`,
`wheel_slip`, `low_visibility`).

## Two-tier brain (rules + Gemma)

| | Tier 1: rules | Tier 2: Gemma (advice) |
|---|---|---|
| Runs | 5 times per sim second, every rover | one model call at a time, on its own thread |
| Budget | 40 ms per rover per cycle | 3000 ms per call (timeout 4 s) |
| Decides | everything: map, A*, slope limit, battery and return, soft ground, stalls, faults, every safety reaction | nothing on its own: it suggests, the rules check |

Where Gemma helps:

1. **Goal choice.** When a rover picks a new frontier (at most every 20 s, or when it reaches one),
   it drives to the rules' pick straight away. Meanwhile the rules plan two other frontiers (A*
   path length, steepest slope on the way, battery left after the trip home), maybe a found science
   target and the trip home, and Gemma picks one with a short reason that quotes those numbers. If it
   picks another option that is still valid, the rover switches and logs one event, for example
   `r2 switching to frontier 14 m east (Gemma, 0.3 s): 'f2: slope max 9 deg, battery after return 51%'`.
   If it agrees, that is logged at most once a minute per rover.
2. **Anomalies.** New rocks, wheel slip, LiDAR range collapsing: the rules react at once (slow down,
   back out, reroute). Then the rover's camera picture and sensor numbers go to Gemma, which says what
   it is, how sure it is and what to do. Safe extras are applied (wait, go slowly, back off); unsafe
   ones are refused, for example "proceed slowly" on soft ground (`Safety rule wins: ...`).
3. **Questions to the operator.** If Gemma is unsure (confidence under 0.75, or it says
   `ask_operator`), or it is the first anomaly since start or reset, the rover asks in the chat with
   2 or 3 safe options and its camera picture, and creeps at 0.25 m/s along its safe route while it
   waits. No answer in 20 s: it takes the safest option and says so.
4. **Typed commands.** Gemma (or the keyword parser if Gemma is offline) turns the operator's text
   into a command; the rules check it and run it or refuse with numbers ("r1: can't, battery 22%
   would not get me there and back"). One reply per command. Commands: go to x, y; explore a
   direction (for 3 minutes); sample a target; return home; hold; resume; avoid an area (a circle that
   is off limits for the whole fleet); anything unclear gets a question back. After a command is done
   the rover carries on on its own.

**Hold** shows as state `idle` with goal `null` and a chat note ("r1: holding at (12, -4), say resume
to continue"). A held rover still watches its battery: if it would not get home otherwise, it leaves
the hold and says so.

**Demo a question:** start the sim (with Ollama, or `GEMMA_FAKE=1`), wait until the rovers are
driving, then trigger a rockfall (UI hazard button, or
`curl -X POST localhost:8000/api/hazard -d '{"type": "rockfall"}'`). Within a few seconds the rover
that sees the new rocks asks "New rocks block my path 3 m ahead ... Gemma thinks: rock (confidence
0.80). What should I do?" with Go around / Approach slowly / Wait and its camera picture. Answer in
the UI (or `POST /api/answer`), or wait 20 s for the safe default. Only the first anomaly after
start or reset asks for sure; after a reset it asks again. `OPERATOR_ASK_BELOW=1.01` makes every
anomaly ask (one question per rover per minute at most).

## Edge compute budget

Each rover states a budget: 40 ms of compute per 200 ms decision cycle (`BRAIN_BUDGET_MS`). The
"brain" is what a real rover computer would run: turning LiDAR hits into the map, frontier logic,
path planning and the state machine. It does not include the simulated sensor hardware (the ray
casts) or the physics. Work shared by the fleet (frontiers, reachability) is done by one rover
per cycle, taking turns, and counts toward that rover's time. The rules' share of tier 2 (planning
Gemma's candidate goals and operator trips) counts too and also plans in slices. Gemma's own time
does not: the model runs on another thread with its own budget (3000 ms per call).

Planning is anytime: when a plan uses up the cycle's share of the budget, A* stops and carries on
next cycle. Meanwhile the rover keeps its current route, or waits if something blocks it just
ahead. A rover that goes over budget several times in 10 s logs a warning and plans in smaller
slices for a while. Each tick reports `brain` (last, average and peak ms over 10 s, and cycles
over budget).

Measured on the dev machine (2 shared CPUs, 3 rovers, 400 sim s, wall clock):

| Run | avg ms | p95 ms | max ms | over 40 ms |
|---|---|---|---|---|
| no hazards, seeds 42 / 7 / 123 | 1.2 to 1.5 | 2.4 to 3.8 | 6 to 30 | 0 |
| all four hazards, seeds 42 / 7 / 123 | 1.1 to 1.3 | 2.2 to 3.6 | 8 to 30 | 0 |

The whole sim still runs about 14 to 15 times faster than real time with 3 rovers, hazards and
UI messages, and about 13 times with the fake Gemma attached. A tick message is about 12 to 13 KB
(most of it the LiDAR points).

The budget is wall clock time, so on a busy machine a plan may be split at a different moment and
two runs can drift apart. `--model-clock` (or `plan_clock="model"` in `Config`) counts A*
expansions instead (3 us each, measured here), which makes runs repeat exactly; the tests use it.

## How the rovers decide

- **Seeing.** Each rover's LiDAR hits go into one shared map of heights and slopes. Rocks show up
  as steep spots. If mapped ground changes (a rockfall), the map notices and updates.
- **Where to go.** A frontier is the edge between known, drivable ground and the unknown. Each
  rover scores the frontiers: big and close is good, turning around is bad, and a frontier another
  rover is already heading for is much worse. So the rovers spread out. It also checks it has
  enough battery to get there and back. Then A* plans a path around steep ground. While driving it
  keeps checking the next 12 m of its route against the map and re-plans if anything blocks it.
- **Learning the slope limit.** The fleet starts by guessing it can climb 30 deg. If a rover
  stalls nose up on a steady slope, with its motors maxed out and gravity alone needing most of
  their torque, the fleet lowers the limit to 2 deg under that slope ("slope 21 deg with motors at
  96%"). Every rover replans with the new limit straight away. The limit never drops below the
  steepest climb a rover has already managed. A stall caused by a rock or by losing grip does not
  count. The real limit at 2 g is about 20.5 deg (see `scripts/climb_test.py`).
- **Science.** If a LiDAR ray hits a target within 8 m, the rover reports it. If it is free and has
  the battery, it drives over, waits a few seconds to take a sample, and marks it sampled. Busy
  rovers log the find and another rover can pick it up. Targets come in clusters, so frontiers near
  a sampled target get a bonus.
- **Battery.** Driving uses power (more on slopes and in tight turns), and the electronics use a
  little all the time. Each rover keeps checking how much charge it needs to get home and goes
  back with a 15% margin (25% while the lander charges slowly). It charges on the lander pad, then
  heads out again.
- **Trouble.** If a rover stops moving it works out why: a rock (mark it and go round), a slope
  (learn), lost grip (back off and try another line), soft ground (back out and mark it), or a
  rover in the way (back off). If it tips over it rights itself. If steep ground boxes it in, it
  heads for the lander, allowing slightly steeper ground, until a frontier is reachable again.
- **Done.** When no frontier can be reached from the lander, exploration is complete and every
  rover parks at the lander and the sim stays as it is until someone presses Reset. (Optional:
  set `AUTO_RESET_S`, e.g. 60, to land the fleet on a new random planet that long after
  exploration finishes, so an unattended demo keeps going.)

Every decision is logged as a short event that the UI shows live, with the sensor reading behind
it and, where it makes sense, the place (`at`) so the UI can fly the camera there.
