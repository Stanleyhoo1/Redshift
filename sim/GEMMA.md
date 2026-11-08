# Gemma in Redshift: a small local model beside the rules

## What it does

The rovers drive themselves with fast hand-written rules and A* path planning. Gemma adds a
second opinion that runs **on your own machine** (no cloud, no link to Earth, just like a real
rover). It does five jobs:

1. **Choose.** When a rover picks a new goal (at most every 20 s, or when it reaches one), the rules
   list a few safe options: two or three frontiers with the A* path length, the steepest slope on
   the way and the battery left after the trip home, maybe a science target, maybe going home to
   charge. Gemma picks one and says why in one short sentence that quotes those numbers.
2. **Interpret.** When something odd happens (new rocks on the path, wheel slip, LiDAR range
   dropping in dust), Gemma looks at the rover's camera picture and its sensor numbers and says what
   it thinks it is (rock, soft ground, dust...), how sure it is (0 to 1) and what to do.
3. **Understand you.** You type "rover 2 go to 10, -5" or "send the orange one home" and Gemma
   turns it into a command the rules can check and run. If the message is unclear it asks back.
   A keyword parser does the same job when Gemma is offline.
4. **Describe what moves.** When the LiDAR tracker sees something moving on its own, Gemma looks at
   the camera picture and the track numbers (distance, speed, size) and says whether it is a
   lifeform, gives a short label ("possible lifeform, six legs, about 1.2 m") and what to do:
   approach (to watch from about 6 m), observe from here, or ignore.
5. **Answer you.** You ask "rover 2, what have you found?" and the rover answers in one to three
   sentences, written by Gemma from a fact sheet the sim builds from that rover's own memory. Every
   number and id in the answer is checked against the fact sheet; anything else (or no answer in 3 s)
   and the rule-based answer is used.

## The two tiers

```
 Tier 1: rules (5 Hz, every 200 ms)        Tier 2: Gemma (one model call at a time, own thread)
 ------------------------------------      ----------------------------------------------------
 LiDAR map, A*, slope limit, battery  -->  candidate goals, anomaly + camera, operator text
 always has an answer, never waits    <--  answer in about 0.3 to 0.4 s: choice / action / command
 checks every answer before using it       + confidence + reason quoting sensor numbers
```

What the **rules still decide**, always: the map, every path (A*), the slope limit, when to go
home (battery), soft ground, stalls, faults, and every safety reaction. Gemma never drives the
rover; it can only pick among options the rules made, or suggest a safe extra (wait, go slowly,
back off) on top of what the rules already did.

- The rules **never wait** for Gemma. Asking and checking for an answer take about 0.1 ms.
- If Gemma is slow (no answer in 4 s), offline, still loading, or answers something invalid, the
  rules' choice simply stands. Typed commands then go through the keyword parser. A lifeform
  sighting then follows the rules (approach carefully), and the first one still asks the operator.
- Every accepted Gemma answer must quote at least one number the rover actually measured (the
  code checks this), so every decision can be traced back to sensor input. Events that came from
  Gemma carry `"source": "gemma"`; the ones from typed commands carry `"source": "operator"`.

## Where Gemma is used (redshift/advisor.py)

| Moment | The rules do at once | Gemma is asked | The answer |
|---|---|---|---|
| Rover picks a new frontier | drive to the rules' pick (the rover never waits) | pick one of 2 to 5 options | another valid option: the rover switches and logs ONE event, e.g. `r2 switching to frontier 14 m east (Gemma, 0.3 s): 'f2: slope max 9 deg, battery after return 51%'`. Same as the rules: logged at most once a minute per rover |
| New rocks on the path | slow to 0.3 m/s, replan around them | camera + numbers: what is it, what to do | reroute: agrees (and gives the rocks a little extra room). Wait or go slowly: applied. Low confidence: question to the operator |
| Wheel slip (soft ground) | stop, mark the soft zone for everyone, back out | same | back out: agrees. "Proceed slowly": refused, `Safety rule wins: no driving on into soft ground` |
| LiDAR range drops (dust) | slow down, then 0.4 m/s on mapped ground only | same | go slowly: agrees. Wait: stands still until the dust clears (at most 30 s) |
| LiDAR gets no returns ahead, camera sees blue (`no_returns`) | slow down, mark the lake for the fleet, replan around it | camera + numbers (missing rays, distance, share of blue pixels) | "liquid" + reroute: agrees (and marks the water if the rules had not). A hole the camera says is not liquid is a drop-off: the rules just slow down and keep to mapped ground, Gemma is not asked |
| Something moves (a LiDAR track) | stop and turn to face it | camera + track numbers: lifeform or not, short label, action | approach: follow at about 6 m (never closer than 4 m). Observe: watch from here. Ignore (not a lifeform): back to exploring. Unsure (or the first sighting): question to the operator |
| Operator types a message | post it in the chat | turn it into a command (or say it is a question, with its topic) | the rules check it (A* within the slope limit, battery there and back) and run it, or refuse with numbers |
| Operator asks a question | build the fact sheet and the rule answer | write the reply from the fact sheet only | every number and id must be in the fact sheet, at most 3 sentences: posted as `kind: "reply"`, `source: "gemma"`; otherwise (or after 3 s) the rule answer, `source: "rules"` |

The camera picture is grabbed by a small worker thread through the camera's render thread
(never from the sim thread, which holds the sim lock). Gemma's anomaly prompt says: soft ground or
wheel slip means back out, new rocks mean reroute, dust or low visibility means go slowly.

### Questions to the operator

A rover asks you (protocol `requests` plus a chat message) when Gemma says `ask_operator`, when its
confidence is under 0.75 (`OPERATOR_ASK_BELOW`), or for the **first anomaly after start or reset**
(`OPERATOR_ASK_FIRST=1`, so a demo always shows one). The question quotes the sensor numbers and
Gemma's guess, offers only safe options, and shows the rover's camera:

```
r2: New rocks block my path 3 m ahead where my map showed open ground (LiDAR: 4 map cells changed,
about 0.9 m tall). Gemma thinks: rock (confidence 0.80). What should I do?
[Go around] (default)  [Approach slowly]  [Wait]
```

While it waits the rover holds position. It only moves if staying put is the danger: backing out of
soft ground or liquid, heading home on a low battery, or making room for a creature that came closer
than the planetary protection distance. Your answer is applied ("r2: going around, thanks."). By default
the question waits for you (`OPERATOR_TIMEOUT_S` 0, `expires_in` null). With `OPERATOR_TIMEOUT_S` above 0
the rover takes the safest option after that many wall seconds and says so ("r2: no reply in 20 s, taking
the safe option: go around."). A new command for that rover also settles its question. If Gemma is offline the rules still ask about the first
anomaly, without Gemma's guess. At most one open question per rover, two for the fleet, and one
question per rover per minute.

**How to demo it:** start the sim (with Ollama, or `GEMMA_FAKE=1`), wait until the rovers drive,
then press the rockfall button in the UI (or `curl -X POST localhost:8000/api/hazard -d
'{"type": "rockfall"}'`). The rover that sees the rocks asks within a few seconds. Answer in the UI
(`POST /api/answer {"id": "q1", "option": "reroute"}`) or let it time out. After a reset the next
first anomaly asks again. To make every anomaly ask, set `OPERATOR_ASK_BELOW=1.01`.

### Lifeform questions

The first lifeform sighting after start or reset always asks (and later ones only when Gemma's
confidence is under 0.75):

```
r3: Possible lifeform 8 m north (LiDAR track k2, 0.3 m/s, about 1.2 m). Gemma thinks: possible
lifeform, six legs, about 1.2 m (confidence 0.80). Follow it or keep exploring?
[Follow it]  [Observe from here] (default)  [Keep exploring]
```

While it waits, the rover watches from where it is. "Observe from here" is the safe default: the
rover never gets closer. With `OPERATOR_TIMEOUT_S` above 0 it takes that when the time is up. Demo: `curl -X POST
localhost:8000/api/hazard -d '{"type": "lifeform"}'` (or click a spot with x and y).

### Goal choice with samples

Target options quote the spectrometer's value estimate and the free tubes, so Gemma can weigh them:
`sample t3 (mineral outcrop) 12 m north-east, value 8/10 (estimate), 2 tubes free, path 14 m, slope
max 9 deg, battery after return 51%`. With full tubes, "return to the lander to cache the samples"
is an option.

### Questions (protocol v5)

The command step sees the question ("question" with a topic: status, exploration, finds, samples,
battery, hazards, lifeforms, water, location, why, unknown); small-model mix-ups are corrected by the
keyword rules (a message starting with a question word is a question even if Gemma made an order of
it, "go to 10, 5?" is still an order). Who answers: the rover named (or picked in the UI); the fleet
when nobody or everybody is asked; each rover for per-rover topics addressed to everyone ("everyone,
report battery").

Each rover keeps a small mission memory (`memory.py`, updated where things happen): mapped area by
compass sector and distance band around the lander, distance driven, targets found and sampled (with
the on-board analysis), hazards met and how it handled them, lifeforms observed, notable recent log
lines, its goal and the reason. `answers.py` turns that into short first person facts, most relevant
first, and Gemma gets only those (a prompt of about 150 to 250 tokens):

```
You are rover r2.
Operator asks: "what have you explored?"
Facts:
- I've mapped about 1,904 m2 (13.2% of the 14,400 m2 map), mostly north-east of the lander, out past 35 m
- I've driven 410 m
- I'm at (31, 12), 33 m east of the lander
Answer the question.
```

Answer: `{"text": "..."}`. Checks: 1 to 3 sentences, at most 60 words; every number must match a fact
(simple rounding allowed: 1,904 as 1,900, 0.64 as 64%); every id (r2, t4, k1) must be in the facts;
an em dash is replaced. A failed check, a slow answer (over `GEMMA_BUDGET_MS`, 3 s) or Gemma offline:
the rule-based answer (`answers.rule_answer`) is posted instead. Asked about something outside its
memory, the rover says it does not know. The fake model (`GEMMA_FAKE=1`) answers by rewording the two
most relevant facts, so its answers are grounded too.

### Typed commands

`POST /api/chat {"text": "...", "rover": "r2"}` (`rover` optional: the rover selected in the UI).

| Command | What the rules do |
|---|---|
| go to x, y | A* within the slope limit, then battery there and back with the margin; drives there (state `exploring`, goal = the point), then explores again. Refuses outside the map, in soft ground, on blocked or too steep ground, without a path, or without the battery ("r1: can't, battery 22% would not get me there and back (needs about 41% with the 15% margin).") |
| explore north (or any of 8 directions) | for 3 minutes that rover prefers frontiers that way (and only picks up targets that way) |
| sample t4 | checks the target is known and free and the battery is enough, then goes to sample it |
| go to the water source, investigate the lake for signs of life, survey the spires, take a look at t7 | `go_to_place` (protocol v6): the place is resolved from fleet knowledge only; one match (or nearest, biggest, north, the second one, near rover 2): a safe spot next to it through the go to checks, and with investigate a 30 s survey and a report; several: a `choose_place` request; none: what the fleet knows |
| return home | goes back to the lander and charges (a rover backing out of trouble finishes that first) |
| hold | parks where it is: state `idle`, goal `null`, chat note "say resume to continue". Not allowed while it is returning on low battery. A held rover leaves the hold by itself if its battery would not get it home otherwise |
| resume | back to full autonomy (cancels holds, trips, directions and operator returns) |
| avoid x, y radius r | that circle is off limits for the whole fleet, mapped or not (refused over the lander pad or a rover) |
| anything unclear | a question back |

**Place commands (protocol v6).** Polite orders ("can we investigate ...", "could you check ...", "let's
survey ...", "I want you to ...") are commands; real questions ("can you see any water?") stay questions,
in Gemma's reading and in the keyword cross-check. Gemma's `go_to_place` is checked like every command:
place, qualifier and near_rover from fixed lists, ids that exist; anything invalid and the keyword parser
decides. The operator's own place words win over Gemma's place, and a Gemma `go_to` without coordinates in
the message (often the rover's own position, which used to give "Where should r1 go?") becomes the place
command. A place command goes to the rover the text or the UI names; Gemma's rover guess is ignored.

No rover named: go to and sample pick the nearest rover that can do it, a place command the closest free
rover (not faulted, not returning on a low battery, not charging below 50%), explore picks the free rover
with the most charge, hold, resume and return mean all rovers. Each command gets exactly one reply.
Gemma's small-model mistakes are caught: a "go to" the rover's own position, or a direction with no
coordinates ("go check out the area north of the lander"), becomes "explore north"; echoed "ack:"
prefixes are removed. The prompt includes the lander position (0, 0) and the map edges.

## Install Ollama on Windows (one time)

1. Install Ollama from <https://ollama.com/download> (Windows installer). It runs in the tray and
   listens on `http://localhost:11434`. Gemma 4 needs a recent Ollama.
2. Open PowerShell and download the model (several GB, one time):
   ```powershell
   ollama pull gemma4:e2b
   ollama list                      # gemma4:e2b should be listed
   ollama run gemma4:e2b "Say hi"   # quick check, then type /bye
   ```
   If your Ollama cannot run Gemma 4 yet, `gemma3:4b` works too: the code uses `gemma4:e2b` if
   installed, otherwise the first installed of `gemma4:e4b` and `gemma3:4b`.
3. Measure it (from the `sim` folder, no simulation needed):
   ```powershell
   python scripts/gemma_bench.py --url http://localhost:11434 --model gemma4:e2b --n 10
   curl.exe -o r1.jpg http://localhost:8080/api/camera/r1.jpg     # a real rover picture (Docker)
   python scripts/gemma_bench.py --image r1.jpg --n 10
   ```

## Run with Ollama

**Natively** (no Docker):

```bash
cd sim
uvicorn redshift.server:app --port 8000          # finds Ollama at localhost:11434 by itself
```

**With Docker** (`docker compose up --build` in the project folder): the compose file passes
`GEMMA_URL=http://host.docker.internal:11434` and maps `host.docker.internal` to your machine, so
the sim container reaches the Ollama running on your PC.

- **If the planner shows `offline` in Docker:** Ollama only listens on localhost by default. Make it
  listen on all interfaces: set `OLLAMA_HOST=0.0.0.0` for the Ollama service and restart it.
  - Windows: `setx OLLAMA_HOST 0.0.0.0`, quit Ollama from the tray icon, start it again, allow it
    through the firewall if asked.
  - Ollama inside WSL (systemd): `sudo systemctl edit ollama`, add `[Service]` and
    `Environment="OLLAMA_HOST=0.0.0.0"`, then `sudo systemctl restart ollama`. Without systemd:
    `OLLAMA_HOST=0.0.0.0 ollama serve`.
- The sim log shows what happens: `gemma planner warming: gemma4:e2b`, then
  `gemma planner ready: gemma4:e2b, warm-up 39.0 s` (or `offline` with the reason).
- **No Ollama at all:** `GEMMA_FAKE=1 docker compose up --build` (or `GEMMA_FAKE=1` for a native run)
  uses the fake model: fixed rules and a 0.5 s delay, the whole pipeline without a GPU.
  The fake cannot see the picture: it calls a hole "liquid" when the rules' camera check found blue,
  and something moving a "possible lifeform" when it moves on its own and is animal sized.

**Warm-up.** The server creates the planner when it starts, before the rovers need it. It checks
Ollama (`GET /api/tags`, every 15 s) and sends one tiny warm-up request so the model is loaded
(about 39 s the first time on the team laptop) and kept for 30 minutes. Until then the planner
state is `warming` and the rovers run on the rules alone. A reset keeps the warm model and only
drops questions still queued about the old world.

## Budgets (edge device)

| What | Budget | Measured (RTX 4070, Ollama 0.40.15, gemma4:e2b, gemma.py's prompts) |
|---|---|---|
| Rule brain, per rover per 200 ms cycle | 40 ms (`BRAIN_BUDGET_MS`) | about 1.3 ms average, p95 about 3 ms, with the fake Gemma attached (2 shared CPUs) |
| One Gemma call | 3000 ms (`GEMMA_BUDGET_MS`), dropped after 4 s | goal choice about 0.3 s, anomaly with a 320 x 240 camera JPEG about 0.4 s; answers to questions have a 3 s limit (prompt about 150 to 250 tokens, up to 128 output tokens; not measured here: Ollama is not reachable from the dev container, expect about 0.3 to 1.5 s warm) |
| Model load | in the background at server start | about 39 s, once |

- Gemma is **not** in the 40 ms rule budget. It has its own budget, and its stats are in every tick
  (`planner`: calls, ok, fallbacks, avg, p95 and last ms, calls over budget).
- The rules' share of tier 2 (planning the candidate goals and operator trips) does count toward
  the 40 ms and is planned in slices like every other plan.
- Load: 3 rovers ask about every 20 to 30 sim seconds; at `SIM_SPEED=3` that is one choice call
  every 2 to 3 wall seconds at about 0.3 s each, plus the odd anomaly, sighting or command. One call
  runs at a time; typed commands go first, then answers to questions, then anomalies, then lifeform
  descriptions, then choices. A question costs two calls (reading it, then writing the answer).
- Prompts are small (under 400 tokens plus the picture, for the default 3 rover fleet), answers short
  (JSON only, temperature 0, thinking off).
- Gravity (v7): the planner is shared across resets, so each request carries its world's gravity. Choices,
  interpretations and descriptions state it in the system prompt ("an unknown super-Earth (gravity 2.5 g,
  ..."; at 1 g and below "an unknown planet (gravity 1 g, ..."); commands and answers only name the world
  ("a super-Earth", at 1 g and below "a planet"), since they may only quote their own inputs. At exactly
  2 g every prompt is byte for byte as before v7.

## Settings (environment variables)

| Var | Default | Meaning |
|---|---|---|
| `GEMMA_ENABLED` | 1 | 0 = Gemma off: rules only, no questions; typed commands use the keyword parser |
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
| `GEMMA_MIN_CONFIDENCE` | 0.55 | below this Gemma's answer itself becomes ask_operator (commands: unclear) |
| `GEMMA_ASK_EVERY_S` | 20 | a rover asks about its goal at most this often (sim s), or when it reaches a goal |
| `OPERATOR_TIMEOUT_S` | 0 | 0 = questions wait for the operator; above 0: wall seconds before the safe default |
| `OPERATOR_ASK_BELOW` | 0.75 | Gemma confidence below this: ask the operator |
| `OPERATOR_ASK_FIRST` | 1 | the first anomaly after start or reset always asks the operator |

## When Gemma is slow or offline

| Situation | What happens | Result |
|---|---|---|
| Ollama not running / unreachable | answered at once, nothing sent | rules alone; planner `offline` |
| Model not installed | answered at once | rules alone; the log says `ollama pull ...` |
| Model still loading | answered at once | rules alone; planner `warming` |
| No answer within 4 s | call dropped | rules' choice stands; warm-up repeats on the next check |
| Answer invalid (bad JSON, unknown id or action, no sensor number quoted, made-up coordinates) | answer rejected | rules' choice stands (commands: the keyword parser) |
| Confidence below 0.55 | answer kept | anomaly: ask_operator; command: a question back (a question is always answered) |
| Answer to a question invalid or later than 3 s | answer dropped | the rule-based answer is posted |

## gemma.py interface (for reference)

`redshift/gemma.py` (standard library only) is thread safe, holds no sim state, and every call below
returns in well under 1 ms. `advisor.py` is its only user in the sim.

```python
from redshift.gemma import get_planner
gp = get_planner()                     # the shared planner (GEMMA_* env vars); the server makes it at start
tk = gp.submit_choice("r1", t, perception, candidates, default="f1")
tk = gp.submit_interpretation("r1", t, perception, anomaly, jpeg_or_None,
                              ["reroute", "back_out", "proceed_slowly", "wait"], default="back_out")
tk = gp.submit_command(t, "rover 2 go to 10, -5", fleet_context)   # rovers, targets, lander, bounds
tk = gp.submit_answer("r2", t, "what have you found?", "finds", facts, default=rule_text)
tk = gp.submit_choice("r1", t, perception, candidates, default="f1", gravity_g=1.0)   # v7, see below
res = gp.poll(tk)                      # None while pending, then a Result; never blocks
gp.stats(); gp.cancel()                # status badge; drop queued questions (on reset)
```

Every `submit_*` call (and every `make_*_request`) takes a keyword `gravity_g` (default 2.0): the world's
gravity in Earth g, written into that request's prompt as described under "Gravity" above. The advisor
passes `cfg.gravity_g` unless it is exactly 2 g, so at 2 g the calls and prompts are the same as before
v7.

`res.ok` is True only for a valid Gemma answer; otherwise `res.source == "fallback"` and the result
carries the caller's default (or the keyword parser's command). Choices: `res.choice`, `res.reason`,
`res.confidence`. Interpretations: `res.classification` (rock, steep_slope, soft_ground, dust, liquid,
other_rover, crater_edge, clear, unknown), `res.action`, `res.model_action`. Descriptions
(`gp.submit_description(rover, t, perception, observation, jpeg)`): `res.classification` (lifeform,
rock, other_rover, dust, unknown), `res.label`, `res.action` (approach, observe, ignore, or
ask_operator when unsure). Commands:
`res.command` (`go_to`, `go_to_place` with place, target_id, qualifier, ordinal, near_rover, investigate, `explore_direction`, `sample_target`, `return_home`, `hold`, `resume`,
`avoid_area`, `question` with `topic`, `unclear` with `question`). Answers: `res.answer` (Gemma's
checked text, or the caller's rule answer in a fallback).

## Files

| File | What |
|---|---|
| `redshift/gemma.py` | planner, Ollama and fake backends, prompts, JSON schemas, answer checks, command parser |
| `redshift/advisor.py` | tier 2 wiring: goal choice, anomaly checks with the camera, operator questions, chat commands and answers |
| `redshift/memory.py`, `redshift/answers.py` | each rover's mission memory; fact sheets, the facts for Gemma, rule-based answers |
| `redshift/places.py` | place commands: known places from fleet knowledge, qualifiers, safe goals, labels, chat picks |
| `tests/test_gemma.py` | gemma.py against a stub Ollama server on port 8400 |
| `tests/test_brain.py` | tier 2 in the running sim with the fake model and the keyword parser |
| `tests/test_v5.py` | questions: intent, answer checks, grounded rule answers, routing, `reply_to` |
| `tests/test_v6.py` | place commands: polite orders, place parsing and Gemma's checked answers, resolver, `choose_place`, survey report, no expiry |
| `scripts/gemma_bench.py` | latency and quality bench against a real Ollama |
