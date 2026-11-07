# Gemma in Redshift: a small local model beside the rules

## What it does

The rovers already drive themselves with fast hand-written rules and A* path planning. Gemma
adds a second opinion that runs **on your own machine** (no cloud, no link to Earth, just like a
real rover). It does three jobs:

1. **Choose.** About every 20 s per rover, the rules list a few options (the best frontiers, a
   science target, go home, wait). Gemma picks one and says why in one short sentence that
   quotes real sensor numbers, for example "f3: slope max 6 deg and battery after return 66%".
2. **Interpret.** When something odd happens (a new obstacle, wheel slip, LiDAR range dropping
   in dust), Gemma looks at the rover's camera picture and its sensor numbers and says what it
   thinks it is (rock, soft ground, dust...), how sure it is (0 to 1), and what to do (reroute,
   back out, go slowly, wait). If it is not sure (below 0.55) it answers **ask_operator**, so the
   rover asks you instead of guessing.
3. **Understand you.** You type "rover 2 go to 10, -5" or "send the orange one home" and Gemma
   turns it into a command the sim can check and run. If the message is unclear it asks a short
   question back. A simple keyword parser does the same job when Gemma is offline.

## The two tiers

```
 Tier 1: rules (5 Hz, every 200 ms)        Tier 2: Gemma (one background thread)
 ------------------------------------      -------------------------------------
 LiDAR map, A*, slope limit, battery  -->  candidates, anomaly + camera, operator text
 always has an answer, never waits    <--  answer in ~1 s: choice / action / command
 keeps driving with its own choice         + confidence + reason quoting sensor numbers
```

- The rules **never wait** for Gemma. Asking and checking for an answer take about 0.1 ms.
- If Gemma is slow (no answer in 4 s), offline, still loading, or answers something invalid,
  the rule-based choice simply stands. For typed commands the keyword parser answers instead.
- Every accepted Gemma answer must quote at least one number the rover actually measured
  (the code checks this), so every decision can be traced back to sensor input.

## Install on Windows (one time)

1. Install Ollama from <https://ollama.com/download> (Windows installer). It runs in the tray
   and listens on `http://localhost:11434`. If you already have it, update to the latest
   version: Gemma 4 needs a recent Ollama.
2. Open PowerShell and download the model (several GB, one time):
   ```powershell
   ollama pull gemma4:e2b
   ollama list                      # gemma4:e2b should be listed
   ollama run gemma4:e2b "Say hi"   # quick check, then type /bye
   ```
   If `ollama pull gemma4:e2b` fails because your Ollama is too old and you cannot update yet,
   `gemma3:4b` works too: the code uses `gemma4:e2b` if installed, otherwise the first installed
   of `gemma4:e4b` and `gemma3:4b` (stats show which one is in use).
3. Measure it (from the `sim` folder, no simulation needed):
   ```powershell
   python scripts/gemma_bench.py --url http://localhost:11434 --model gemma4:e2b --n 10
   ```
   It prints one block between two `8<` lines. Paste that block back to the team. To test with
   a real rover camera picture, grab one from the running sim first:
   ```powershell
   curl.exe -o r1.jpg http://localhost:8080/api/camera/r1.jpg     # Docker (via the web container)
   curl.exe -o r1.jpg http://localhost:8000/api/camera/r1.jpg     # sim running natively
   python scripts/gemma_bench.py --image r1.jpg --n 10
   ```
   Without `--image` the bench uses a synthetic test picture.

## How the sim finds Ollama

- **In Docker** (Docker Desktop on Windows or macOS): containers reach programs on your PC at
  `host.docker.internal`, so the default `GEMMA_URL` is `http://host.docker.internal:11434`.
- **If it shows offline from Docker on Windows:** set the Windows environment variable
  `OLLAMA_HOST=0.0.0.0` (in PowerShell: `setx OLLAMA_HOST 0.0.0.0`), quit Ollama from the tray
  icon and start it again, and allow it through the Windows firewall if asked.
- **On Linux Docker:** the compose service also needs
  `extra_hosts: ["host.docker.internal:host-gateway"]`, and Ollama must listen on `0.0.0.0`.
- **Running the sim natively** (no Docker): nothing to do. If `host.docker.internal` does not
  answer, the code tries `http://localhost:11434` automatically.

The code checks every 15 s with a cheap `GET /api/tags` whether Ollama is up and the model is
installed. As soon as it is, it sends one tiny warm-up request (1 output token, `keep_alive`
30 minutes) so the model is loaded before the first real decision. The first load takes about
30 s; during that time the status is `warming` and the rules carry on alone.

## Settings (environment variables)

| Var | Default | Meaning |
|---|---|---|
| `GEMMA_ENABLED` | 1 | 0 = Gemma off (no threads; every request gets the rule fallback at once) |
| `GEMMA_URL` | `http://host.docker.internal:11434` | Ollama address; `http://localhost:11434` is tried next |
| `GEMMA_MODEL` | `gemma4:e2b` | model to use if installed |
| `GEMMA_FALLBACK_MODELS` | `gemma4:e2b,gemma4:e4b,gemma3:4b` | used in this order if `GEMMA_MODEL` is not installed |
| `GEMMA_TIMEOUT_S` | 4 | an answer later than this is thrown away (rule choice stands) |
| `GEMMA_BUDGET_MS` | 3000 | stated latency budget per call; stats count calls over it |
| `GEMMA_FAKE` | 0 | 1 = fake model: fixed rules and fake delay, to demo the pipeline without Ollama |
| `GEMMA_FAKE_LATENCY_MS` | 500 | delay of the fake model |
| `GEMMA_NUM_PREDICT` | 96 | max output tokens per answer (commands use at least 128) |
| `GEMMA_THINK` | 0 | 1 = let Gemma 4 think before answering (slower, needs more tokens) |
| `GEMMA_KEEP_ALIVE` | 30m | how long Ollama keeps the model in memory between calls |
| `GEMMA_MIN_CONFIDENCE` | 0.55 | below this an interpretation becomes ask_operator, a command becomes unclear |

## Latency budget (edge device)

- The rule brain keeps its own budget of 40 ms per rover per 200 ms cycle. Gemma is **not** in
  that loop; it runs beside it.
- Gemma's stated budget is **3000 ms per call** (`GEMMA_BUDGET_MS`), with a hard timeout of 4 s.
  Prompts are kept small (about 300 to 360 tokens without the picture) and answers short
  (under 100 tokens, JSON only, temperature 0, thinking off).
- Measured on the team laptop (RTX 4070, Ollama 0.16.3, `gemma3:4b`): first call 31.9 s (model
  load, now done by the warm-up), then about 1.1 s per decision with 42 output tokens. Not yet
  measured: `gemma4:e2b` and calls with a camera picture; run the bench to fill this in.
- Load: with 3 rovers asking every 20 sim seconds at `SIM_SPEED=3`, that is one choice call
  every 2.2 s of wall time on average, plus the odd anomaly or command. At about 1.1 s per call
  this fits. Only one call runs at a time; a newer question from the same rover replaces an
  older one that is still waiting, and typed commands go first, then anomalies, then choices.

## When Gemma is slow or offline

| Situation | What happens | Result |
|---|---|---|
| Ollama not running / unreachable | answered at once, nothing sent | fallback, error `gemma offline: ...` |
| Model not installed | answered at once | fallback, error says `ollama pull ...` |
| Model still loading | answered at once | fallback, error `gemma warming` |
| No answer within 4 s | call dropped | fallback, error `timeout ...`; warm-up repeats on the next check |
| Waited over 4 s in the queue | dropped before sending | fallback, error `stale ...` |
| Answer invalid (bad JSON, unknown id or action, no sensor number quoted, made-up coordinates) | answer rejected | fallback, error `invalid answer: ...` |
| Confidence below 0.55 | answer kept | interpretation: action `ask_operator` + `question`; command: `unclear` + `question` |

A fallback for a choice carries the rules' own pick (`result.choice`), for an interpretation the
safe default action (`result.action`), and for a command the keyword parser's command.

## Integration interface (for the orchestrator)

Everything is in `redshift/gemma.py` (standard library only). The planner is thread safe, holds
no sim state, and all calls below return in well under 1 ms.

```python
from redshift.gemma import get_planner
gp = get_planner()      # one shared planner for the process (reads GEMMA_* env vars)
```

### 1. Choices (where the rover picks its next goal)

Hook: where `Fleet.choose_goal(t, rover)` has scored the frontiers for a rover in the exploring
state. Every 20 sim seconds per rover (not every tick):

```python
cands = [  # ids must be unique; kind in frontier | target | return | wait
    {"id": "f1", "kind": "frontier", "summary": "frontier 9 m north, 34 cells, slope max 12 deg, battery after return 58%"},
    {"id": "f2", "kind": "frontier", "summary": "frontier 21 m east, 80 cells, slope max 18 deg, battery after return 41%"},
    {"id": "t3", "kind": "target",   "summary": "ice target 6.2 m north-east, slope max 8 deg, battery after return 47%"},
    {"id": "home", "kind": "return", "summary": "lander 31 m south-west, battery on arrival 60%"},
]
perception = {**rover.sensors_msg(), "battery": rover.battery, "state": rover.state,
              "slope_limit_deg": fleet.slope_limit, "home_distance": dist_home}
rover.gemma_ticket = gp.submit_choice(rover.id, t, perception, cands, default=rule_pick_id)
# ...the rover drives to rule_pick_id straight away, as today
```

Then, each tick, check for the answer:

```python
res = gp.poll(rover.gemma_ticket) if rover.gemma_ticket else None
if res is not None:
    rover.gemma_ticket = None
    if res.ok and res.choice != current_goal_id and t - res.t < 10.0 and still_valid(res.choice):
        switch_goal(res.choice)                      # plan the path as usual; the rules stay in charge
        fleet.log(t, "info", res.trace(), rover.id)  # "r1 Gemma picked f2 (conf 0.82, 1100 ms): ..."
```

Put numbers with units in the summaries: Gemma quotes them, and the check accepts only reasons
that contain a number from the readings or the summaries. Check `res.t` against the current sim
time before acting: in fast headless runs the sim can be many seconds ahead of the answer.

### 2. Interpretations (when something odd happens)

Hooks in `rover.py`: `_on_stall` (new obstacle), `_on_soft_ground` (wheel slip),
`_check_visibility` (dust / low visibility), optionally `_check_flip`. The rules react first,
exactly as today; then ask Gemma:

```python
anomaly = {"type": "wheel_slip", "slip": 0.46, "duration_s": 2.4, "detail": "wheels spinning, not moving"}
tk = gp.submit_interpretation(rover.id, t, perception, anomaly, jpeg,
                              ["reroute", "back_out", "proceed_slowly", "wait"], default="wait")
```

Use anomaly types `new_obstacle`, `wheel_slip`, `low_visibility`, `unexpected_tilt` (any name
works; these read well). `jpeg` may be `None`. To take the picture inside the sim thread (sim lock
held), render in the calling thread, NOT through `camera.render_jpeg` (that waits on the HTTP
render thread, which needs the same lock, so it would deadlock):

```python
from redshift import camera
fr = camera._local_renderer().frame(None, sim, rover.id, "fpv", 256, 192)
jpeg = camera.encode_jpeg(fr.rgb)        # about 15 to 30 ms with OSMesa: only at anomalies
```

(A public helper in camera.py for this would be cleaner; camera.py is not mine to edit.)

When the answer arrives (`res = gp.poll(tk)`):
- `res.ok` and `res.action` in reroute / back_out / proceed_slowly / wait: apply it if it differs
  from what the rules did and is still safe; log `res.trace()`.
- `res.action == "ask_operator"`: show `res.question` to the operator with a timeout (for
  example 30 s) and a safe default (wait or proceed_slowly); slow down meanwhile.
- `not res.ok`: keep the rules' reaction (`res.action` is your default).

`res.classification` is one of rock, steep_slope, soft_ground, dust, other_rover, crater_edge,
clear, unknown; `res.confidence` is 0 to 1; `res.model_action` is what Gemma picked before the
low-confidence rule.

### 3. Operator commands (typed chat)

Wire a chat box (WebSocket message or `POST /api/command {"text": ...}`) to:

```python
ctx = {"rovers": [{"id": r.id, "name": r.name, "color": r.color, "x": r.x, "y": r.y,
                   "state": r.state, "battery": r.battery} for r in sim.rovers],
       "targets": [ts.to_msg() for ts in fleet.targets if ts.status is not None],   # found ones only
       "lander": [0.0, 0.0]}
tk = gp.submit_command(t, text, ctx)
res = gp.poll(tk)          # later: None while Gemma is thinking (about 1 to 2 s)
```

`res.command` is always set (Gemma's when `res.ok`, the keyword parser's otherwise):

| `res.command["type"]` | fields |
|---|---|
| `go_to` | `rover`, `x`, `y` |
| `explore_direction` | `rover`, `direction` (north, south, east, west, north-east, north-west, south-east, south-west) |
| `sample_target` | `rover`, `target_id` |
| `return_home`, `hold`, `resume` | `rover` |
| `avoid_area` | `x`, `y`, `radius` (m, default 3) |
| `unclear` | `question` |

`rover` is `"r1"`..`"rN"`, `"all"`, or `None` when the message named no rover (the sim decides,
for example the nearest free rover for go_to and all rovers for hold). Names, numbers and
colours ("rover 2", "the orange one") are resolved from `ctx`. Show `res.ack` (one short
sentence) or, for `unclear`, `res.question`. Safety is the sim's job: check slope and battery
rules before running a command and refuse with a message if needed. For an instant preview
before Gemma answers, `parse_command(text, ctx)` gives the parser's reading synchronously.

### 4. UI and housekeeping

- `gp.stats()` for a status badge: `model` (in use), `requested_model`, `online`, `state`
  (disabled, starting, offline, no_model, warming, ready), `warming`, `calls`, `ok`, `fallbacks`,
  `asked_operator`, `avg_ms`, `p95_ms`, `last_ms`, `budget_ms`, `over_budget`, `timeout_ms`,
  `warmup_ms`, `queued`, `in_flight`, `last_error`, `detail`.
- `gp.recent(10)`: the last results as dicts, for a decision log panel.
- `gp.cancel()` on a sim reset (drops questions about the old world).
- Tests and fast headless runs: set `GEMMA_ENABLED=0`, or `GEMMA_FAKE=1` to exercise the
  pipeline with the fake model. With the default settings and no Ollama, everything still works:
  the status is `offline` and every request gets the fallback at once.

## Files

| File | What |
|---|---|
| `redshift/gemma.py` | planner, Ollama and fake backends, prompts, JSON schemas, validation, command parser |
| `tests/test_gemma.py` | 34 tests against a stub Ollama server on port 8400 (`python -m pytest -q tests/test_gemma.py`) |
| `scripts/gemma_bench.py` | latency and quality bench against a real Ollama |
