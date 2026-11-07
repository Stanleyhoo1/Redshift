"""Tier 2 of the two-tier brain: wires the local Gemma planner (gemma.py) into the rule-based rovers,
and runs the operator chat and clarification questions (PROTOCOL.md, version 3).

Tier 1, the rules (rover.py, fleet.py, planner.py), decide everything 5 times per sim second and keep
the rovers safe. Tier 2 only advises, on other threads, and the rules check every piece of advice:

1. Goal choice. When the rules pick a new frontier (and at most every GEMMA_ASK_EVERY_S sim seconds,
   or when a goal is reached), the rover drives to the rules' pick at once (the provisional goal).
   Meanwhile this module plans up to two other frontiers (A*, inside the rover's own compute budget,
   a slice per cycle), maybe a found target and the trip home, and asks Gemma to pick one. If Gemma
   picks another option that is still valid, the rover switches and logs ONE event quoting Gemma's
   reason (source "gemma"). Agreement is logged at most once a minute per rover.
2. Anomalies. A hook in rover.anomaly_hooks hands each anomaly to a small worker thread, which grabs
   the rover's camera frame (camera.py render thread; never from the sim thread) and asks Gemma what
   it is. The rules have already reacted (slow down, back out, reroute). Gemma's answer is logged;
   a safe action that adds something (wait, go slowly, back off) is applied; it can never undo a
   safety reaction (for example never drive on into soft ground).
3. Questions. If Gemma says ask_operator, its confidence is low (OPERATOR_ASK_BELOW), or it is the
   first anomaly after start or reset (OPERATOR_ASK_FIRST, a reliable demo trigger), the rover opens
   a request with 2 to 3 safe options and creeps along its safe route while it waits. The answer
   (POST /api/answer) or, after OPERATOR_TIMEOUT_S wall seconds, the safe default is applied.
4. Operator chat. POST /api/chat text goes to Gemma (or the keyword parser when Gemma is offline);
   the rules check the command (A* within the slope limit, battery there and back) and either run
   it or refuse with numbers. One reply per command in the chat.
5. Lifeforms. When a rover starts observing a moving LiDAR track, its camera frame and the track
   numbers go to Gemma, which says whether it is a lifeform, gives a short label and an action. The
   first sighting after start or reset (and later ones only when Gemma is unsure) asks the operator:
   follow it, observe from here (the default: it never gets closer) or keep exploring.

Nothing here ever waits for the model in the 5 Hz loop: submits and polls take microseconds.
"""
from __future__ import annotations

import math
import queue
import threading
import time
import traceback
from collections import OrderedDict

from .fleet import OVERLAP_RADIUS, compass
from .gemma import GemmaConfig, GemmaPlanner, parse_command
from .planner import PENDING
from .terrain import KIND_LABEL

EM_DASH = chr(0x2014)
PARKED = ("idle", "charging", "fault")
DRIVING = ("exploring", "investigating", "returning")

INTERP_OPTIONS = ["reroute", "back_out", "proceed_slowly", "wait"]
RULE_REACTION = {"new_obstacle": "reroute", "slip": "back_out", "low_visibility": "proceed_slowly",
                 "no_returns": "reroute"}
ANOMALY_NAME = {"new_obstacle": "new rocks on the path", "slip": "wheel slip", "low_visibility": "low visibility",
                "no_returns": "a hole in the LiDAR returns"}
# options offered to the operator (only safe ones), first = default = the safest
QUESTION_OPTIONS = {
    "new_obstacle": [("reroute", "Go around"), ("proceed_slowly", "Approach slowly"), ("wait", "Wait")],
    "slip": [("back_out", "Back out and go around"), ("wait", "Stop and wait")],
    "low_visibility": [("wait", "Wait for it to clear"), ("proceed_slowly", "Go on slowly")],
    "no_returns": [("reroute", "Mark it and go around"), ("proceed_slowly", "Approach slowly for a closer look"),
                   ("wait", "Wait")],
}
LIFE_OPTIONS = [("follow", "Follow it"), ("observe_here", "Observe from here"), ("keep_exploring", "Keep exploring")]
LIFE_DEFAULT = "observe_here"     # the safest: it never gets closer
DIRS = {"north": (0.0, 1.0), "south": (0.0, -1.0), "east": (1.0, 0.0), "west": (-1.0, 0.0),
        "north-east": (0.7071, 0.7071), "north-west": (-0.7071, 0.7071),
        "south-east": (0.7071, -0.7071), "south-west": (-0.7071, -0.7071)}

CRAWL_V = 0.25            # m/s while a question is open (the slip and stall detectors need > 0.2)
WAIT_S = 6.0              # "wait" after new rocks or slip (sim s)
DUST_WAIT_S = 30.0        # "wait" in low visibility: until it clears, at most this long (sim s)
SLOW_S = 10.0             # "proceed slowly": this long at caution speed (sim s)
EXPLORE_BIAS_S = 180.0    # "explore north" lasts this long (sim s)
AGREE_EVERY_S = 60.0      # at most one "Gemma agrees" event per rover in this time (sim s)
CHOICE_PLAN_S = 4.0       # longest time spent planning the alternatives before asking anyway (sim s)
CHOICE_STALE_S = 20.0     # an answer about a choice older than this (sim s) is ignored
GO_PLAN_S = 10.0          # an operator go_to whose plan takes longer than this (sim s) is refused
MAX_CHAT = 400


def _clean(text: str) -> str:
    return " ".join(str(text).replace(EM_DASH, "-").split())


def _quote(reason: str) -> str:
    return "'" + _clean(reason).strip("'\" ") + "'"


def path_stats(fmap, pts: list) -> tuple:
    """(length m, steepest mapped slope deg) along a path of (x, y) points."""
    length, smax = 0.0, 0.0
    known, slope = fmap.known, fmap.slope
    for (ax, ay), (bx, by) in zip(pts, pts[1:]):
        seg = math.hypot(bx - ax, by - ay)
        n = max(1, int(seg / 0.5))
        for k in range(n + 1):
            i, j = fmap.ij(ax + (bx - ax) * k / n, ay + (by - ay) * k / n)
            if known[j, i]:
                smax = max(smax, float(slope[j, i]))
        length += seg
    return length, smax


class _Choice:
    """One rover's pending question to Gemma about where to go next."""

    def __init__(self, t: float, rule_goal: tuple):
        self.t0 = t
        self.rule_goal = rule_goal
        self.cands: list = []      # dicts: id, kind, goal, path, size, tidx, label, summary
        self.to_plan: list = []    # candidates still waiting for an A* path
        self.stage = "plan"        # plan -> wait
        self.ticket = None
        self.wall = 0.0


class Advisor:
    """Tier 2 for one Simulation. planner: a GemmaPlanner (None = disabled: rules only, the chat
    still works through the keyword parser). frame_source(rover_id) -> JPEG bytes or None, called on
    the worker thread only. wait_for_gemma: tests only, block until each Gemma answer is in so runs
    repeat exactly (never in the server)."""

    def __init__(self, sim, planner: GemmaPlanner | None = None, frame_source=None, clock=time.monotonic,
                 wait_for_gemma: bool = False):
        self.sim = sim
        self.cfg = sim.cfg
        self.fleet = sim.fleet
        self.planner = planner if planner is not None else GemmaPlanner(GemmaConfig(enabled=False), start=False)
        self.frame_source = frame_source
        self.clock = clock
        self.wait_for_gemma = wait_for_gemma
        self.chat: list = []
        self.chat_seq = 0
        self._chat_cursor = 0
        self.requests: OrderedDict = OrderedDict()   # open questions: id -> dict
        self.closed: OrderedDict = OrderedDict()     # id -> "answered" | "expired"
        self.req_seq = 0
        self.commands: list = []                     # operator messages waiting for their parse
        self.go_jobs: dict = {}                      # rover id -> operator go_to being planned
        self.after_sand: dict = {}                   # rover id -> wait (s) once it is out of the sand
        self.home_after: set = set()                 # rovers to send home once their recovery ends
        self.choices: dict = {}                      # rover id -> _Choice
        self.last_ask: dict = {}                     # rover id -> sim t of the last goal question
        self.last_agree: dict = {}
        self.last_question: dict = {}
        self.first_pending = bool(self.cfg.operator_ask_first)
        self.first_life_pending = True               # the first lifeform sighting always asks the operator
        self._done: list = []                        # (job, ticket) from the worker
        self._mu = threading.Lock()
        self._jobs: queue.Queue = queue.Queue()
        self._worker: threading.Thread | None = None
        self._stop = threading.Event()
        self.errors = 0
        self.frames = 0                              # camera frames sent to Gemma
        for r in sim.rovers:
            r.advisor = self
            r.anomaly_hooks.append(self._on_anomaly)

    # ================================================================== basics
    @property
    def active(self) -> bool:
        """Tier 2 is switched on (Gemma may still be offline or warming)."""
        return bool(self.planner.cfg.enabled)

    def _rover(self, rid):
        return next((r for r in self.sim.rovers if r.id == rid), None)

    def close(self) -> None:
        self._stop.set()
        if self._worker is not None:
            self._jobs.put(None)
            self._worker.join(1.0)
            self._worker = None

    def _error(self, where: str) -> None:
        self.errors += 1
        if self.errors <= 5:
            print(f"[advisor] {where} failed:\n{traceback.format_exc()}", flush=True)

    def _wait(self, ticket: int):
        end = time.perf_counter() + self.planner.cfg.timeout_s + 2.0
        while time.perf_counter() < end:
            res = self.planner.poll(ticket)
            if res is not None:
                return res
            time.sleep(0.001)
        return None

    def _log(self, t, level, msg, rover=None, at=None, source=None):
        self.fleet.log(t, level, _clean(msg), rover, at=at, source=source)

    # ================================================================== protocol
    def planner_msg(self) -> dict:
        s = self.planner.stats()
        st = {"ready": "ready", "warming": "warming", "starting": "warming", "disabled": "disabled"}.get(
            s["state"], "offline")
        return {"model": s["model"], "state": st, "online": st == "ready", "calls": s["calls"], "ok": s["ok"],
                "fallbacks": s["fallbacks"], "avg_ms": s["avg_ms"], "p95_ms": s["p95_ms"],
                "last_ms": s["last_ms"], "budget_ms": s["budget_ms"], "over_budget": s["over_budget"]}

    def say(self, frm: str, t: float, text: str, kind: str = "info", to="operator", source="rules",
            request_id=None) -> str:
        """Add a chat message (from "operator", a rover id or "fleet"). Returns its id."""
        self.chat_seq += 1
        cid = f"c{self.chat_seq}"
        text = _clean(text)
        # The UI already shows who is speaking: drop a leading "r2:" or "r2 " from rover messages.
        if frm not in ("operator", "fleet"):
            for prefix in (f"{frm}: ", f"{frm} "):
                if text.startswith(prefix):
                    text = text[len(prefix):]
                    text = text[:1].upper() + text[1:]
                    break
        self.chat.append({"id": cid, "t": round(float(t), 1), "from": frm, "to": to, "text": text[:400],
                          "kind": kind, "source": source, "request_id": request_id, "_seq": self.chat_seq})
        if len(self.chat) > MAX_CHAT:
            del self.chat[:100]
        return cid

    @staticmethod
    def _public(m: dict) -> dict:
        return {k: v for k, v in m.items() if k != "_seq"}

    def new_chat(self) -> list:
        """Chat messages since the last call (for tick)."""
        out = [self._public(m) for m in self.chat if m["_seq"] > self._chat_cursor]
        if self.chat:
            self._chat_cursor = max(self._chat_cursor, self.chat[-1]["_seq"])
        return out

    def init_chat(self) -> list:
        """The last 50 messages already handed out in ticks (the next tick carries the rest)."""
        return [self._public(m) for m in self.chat if m["_seq"] <= self._chat_cursor][-50:]

    def requests_msg(self) -> list:
        now = self.clock()
        return [{"id": q["id"], "rover": q["rover"], "t": q["t"], "question": q["question"],
                 "options": [{"id": o, "label": lab} for o, lab in q["options"]], "default": q["default"],
                 "expires_in": round(max(0.0, q["deadline"] - now), 1), "image": q["image"]}
                for q in self.requests.values()]

    def fleet_context(self) -> dict:
        half = (self.cfg.nx - 1) * self.cfg.res / 2.0
        return {"rovers": [{"id": r.id, "name": r.name, "color": r.color, "x": round(r.x, 1), "y": round(r.y, 1),
                            "state": r.state, "battery": round(r.battery, 2)} for r in self.sim.rovers],
                "targets": self.fleet.discovered_targets(), "lander": [0.0, 0.0],
                "bounds": [-half, -half, half, half]}

    # ================================================================== per cycle
    def update(self, t: float) -> None:
        """Sim thread, after every rover's decision cycle: operator commands, Gemma's anomaly
        answers and question timeouts. Never blocks."""
        try:
            if self.commands:
                self._poll_commands(t)
            if self._done:
                self._poll_interpretations(t)
            if self.requests:
                now = self.clock()
                for q in [q for q in self.requests.values() if now >= q["deadline"]]:
                    self._expire(q, t)
        except Exception:                  # noqa: BLE001 (tier 2 must never stop the sim)
            self._error("update")

    def rover_tick(self, rover, t: float) -> None:
        """Called at the end of each rover's decision cycle (inside its compute budget): hold,
        operator go_to planning, and the A* planning for Gemma's goal candidates."""
        try:
            if rover.pause_dust and not rover.degraded:
                rover.pause_until, rover.pause_dust = t, False
            if rover.state != "recovering":
                wait = self.after_sand.pop(rover.id, None)
                if wait:
                    rover.pause_until = t + wait
                if rover.id in self.home_after:
                    self.home_after.discard(rover.id)
                    if rover.state in ("exploring", "investigating"):
                        rover._go_home(t, "operator")
            if rover.held and (rover.state in ("exploring", "investigating", "observing") or (
                    rover.state == "returning" and rover.return_reason not in ("battery", "range", "complete"))):
                self._enter_hold(rover, t)
            if rover.id in self.go_jobs:
                self._plan_go(rover, t)
            elif rover.id in self.choices:
                self._choice_step(rover, t)
        except Exception:                  # noqa: BLE001
            self.choices.pop(rover.id, None)
            self._error("rover_tick")

    @staticmethod
    def _budget_left(rover) -> bool:
        deadline, cap = rover.plan_limits()
        if cap is not None:
            return cap > 300
        return deadline is None or deadline - time.perf_counter() > 0.002

    # ================================================================== 1. goal choice
    def on_goal_chosen(self, rover, t: float, goal, path, size: int, reached: bool) -> None:
        """The rules just picked `goal` (the rover drives there now). Maybe ask Gemma."""
        if not self.planner.online or rover.id in self.choices or rover.state != "exploring":
            return
        if rover.op_goal is not None or rover.held or (rover.dir_bias is not None and t < rover.dir_bias[2]):
            return                         # the operator's wishes come before Gemma's
        if not reached and t - self.last_ask.get(rover.id, -1e9) < self.cfg.gemma_ask_every_s:
            return
        f = self.fleet
        job = _Choice(t, tuple(goal))
        plen, smax = path_stats(f.map, [(rover.x, rover.y)] + list(path))
        job.cands.append({"id": "f1", "kind": "frontier", "goal": tuple(goal), "path": list(path), "size": size,
                          "plen": plen, "smax": smax})
        alts = []
        for (_u, x, y, sz, _key) in rover.goal_options:
            if math.hypot(x - goal[0], y - goal[1]) < 6.0 or any(math.hypot(x - a[0], y - a[1]) < 6.0 for a in alts):
                continue
            if any(o is not rover and o.goal is not None and o.state in ("exploring", "investigating")
                   and math.hypot(x - o.goal[0], y - o.goal[1]) < OVERLAP_RADIUS for o in f.rovers):
                continue
            alts.append((x, y, sz))
            if len(alts) == 2:
                break
        for k, (x, y, sz) in enumerate(alts):
            c = {"id": f"f{k + 2}", "kind": "frontier", "goal": (x, y), "path": None, "size": sz}
            job.cands.append(c)
            job.to_plan.append(c)
        k, d = f.unclaimed_target(t, rover)
        if k is not None and not rover.degraded and d < 45.0 and rover.tube_verdict(f.targets[k]) is None:
            ts = f.targets[k]
            need = rover.energy_for_distance(1.3 * d + 1.3 * math.hypot(ts.spec.x, ts.spec.y))
            if rover.battery > need + rover.battery_margin + 0.05:
                c = {"id": ts.spec.id, "kind": "target", "goal": (ts.spec.x, ts.spec.y), "path": None, "tidx": k}
                job.cands.append(c)
                job.to_plan.append(c)
        if rover.battery < 0.6 or (rover.samples and rover.free_tubes() <= 0):
            job.cands.append({"id": "home", "kind": "return", "goal": rover.dock, "path": []})
        if len(job.cands) < 2:
            return                         # nothing to choose between
        self.choices[rover.id] = job
        self.last_ask[rover.id] = t

    def _choice_step(self, rover, t: float) -> None:
        job = self.choices[rover.id]
        if job.stage == "plan":
            if rover.state != "exploring" or rover.goal is None or \
                    math.hypot(rover.goal[0] - job.rule_goal[0], rover.goal[1] - job.rule_goal[1]) > 1.0:
                del self.choices[rover.id]     # the rules moved on (target, return, new goal)
                return
            while job.to_plan and t - job.t0 <= CHOICE_PLAN_S:
                if not self._budget_left(rover):
                    return                     # out of compute this cycle: carry on next cycle
                c = job.to_plan[0]
                path = self.fleet.plan(t, rover, c["goal"])
                if path is PENDING:
                    return
                job.to_plan.pop(0)
                if path is None:
                    c["dead"] = True
                else:
                    c["path"] = path
                    c["plen"], c["smax"] = path_stats(self.fleet.map, [(rover.x, rover.y)] + list(path))
            self._submit_choice(rover, job, t)
            return
        res = self.planner.poll(job.ticket)
        if res is None:
            if self.clock() - job.wall > self.planner.cfg.timeout_s + 2.0:
                del self.choices[rover.id]
            return
        del self.choices[rover.id]
        self._apply_choice(rover, job, res, t)

    def _summaries(self, rover, job) -> list:
        out = []
        for c in job.cands:
            if c.get("dead") or (c["kind"] in ("frontier", "target") and c.get("path") is None):
                continue
            gx, gy = c["goal"]
            d = math.hypot(gx - rover.x, gy - rover.y)
            where = f"{d:.0f} m {compass(gx - rover.x, gy - rover.y)}"
            if c["kind"] == "return":
                hd = rover.home_dist if rover.home_dist is not None else 1.5 * d
                arr = (rover.battery - rover.energy_for_distance(hd)) * 100.0
                full = len(rover.samples)
                c["label"] = "the lander to recharge" + (f" and cache {full} samples" if full else "")
                c["summary"] = (f"lander {where}, battery on arrival {arr:.0f}%, then recharge"
                                + (f", cache {full} samples ({rover.free_tubes()} tubes free now)" if full else ""))
            else:
                after = (rover.battery - rover.energy_for_distance(c["plen"] + 1.3 * math.hypot(gx, gy))) * 100.0
                tail = f"path {c['plen']:.0f} m, slope max {c['smax']:.0f} deg, battery after return {after:.0f}%"
                if c["kind"] == "target":
                    ts = self.fleet.targets[c["tidx"]]
                    c["label"] = f"sample {ts.spec.id} ({KIND_LABEL[ts.spec.kind]}) {where}"
                    c["summary"] = (f"{KIND_LABEL[ts.spec.kind]} {ts.spec.id} {where}, value {ts.spec.est_value}/10 "
                                    f"(estimate), {rover.free_tubes()} tubes free, {tail}")
                else:
                    c["label"] = f"frontier {where}"
                    c["summary"] = f"frontier {where}, {c['size']} cells, {tail}"
            out.append({"id": c["id"], "kind": c["kind"], "summary": c["summary"]})
        return out

    def _perception(self, rover) -> dict:
        p = rover.sensors_msg()
        if p.get("spectro") is None:
            p.pop("spectro", None)
        p["battery"] = round(rover.battery, 2)
        p["state"] = rover.state
        return p

    def _submit_choice(self, rover, job, t: float) -> None:
        cands = self._summaries(rover, job)
        if len(cands) < 2:
            del self.choices[rover.id]
            return
        perception = dict(self._perception(rover), slope_limit_deg=round(self.fleet.slope_limit, 1))
        if rover.home_dist is not None:
            perception["home_distance"] = round(rover.home_dist, 1)
        job.ticket = self.planner.submit_choice(rover.id, t, perception, cands, default="f1")
        job.stage, job.wall = "wait", self.clock()
        if self.wait_for_gemma:
            self._wait(job.ticket)

    def _apply_choice(self, rover, job, res, t: float) -> None:
        if not res.ok or t - job.t0 > CHOICE_STALE_S:
            return                         # the rules' pick stands
        if rover.state != "exploring" or rover.op_goal is not None or rover.held or rover.goal is None:
            return
        if math.hypot(rover.goal[0] - job.rule_goal[0], rover.goal[1] - job.rule_goal[1]) > 1.0:
            return                         # the rover already moved on to something else
        c = next((c for c in job.cands if c["id"] == res.choice), None)
        if c is None:
            return
        lat = res.latency_ms / 1000.0
        why = _quote(res.reason)
        if c["id"] == "f1":
            if t - self.last_agree.get(rover.id, -1e9) >= AGREE_EVERY_S:
                self.last_agree[rover.id] = t
                self._log(t, "info", f"{rover.id} Gemma agrees with {c['label']} ({lat:.1f} s): {why}",
                          rover.id, at=c["goal"], source="gemma")
            return
        f = self.fleet
        gx, gy = c["goal"]
        if c["kind"] == "frontier":
            still = any(abs(fx - gx) < 3.0 and abs(fy - gy) < 3.0 for (fx, fy, _s, _i) in f.frontiers)
            taken = any(o is not rover and o.goal is not None and o.state in ("exploring", "investigating")
                        and math.hypot(gx - o.goal[0], gy - o.goal[1]) < OVERLAP_RADIUS for o in f.rovers)
            need = rover.energy_for_distance(c["plen"] + 1.3 * math.hypot(gx, gy)) + rover.battery_margin
            if not still or taken or need > rover.battery:
                return
            rover.goal, rover.goal_size, rover.goal_t = (gx, gy), c["size"], t
            rover._set_path(t, c["path"])
            rover.plan_t = -1e9            # re-plan from where the rover is now (the rules' A*)
            rover.sticky_goal = (gx, gy)
            rover.next_goal_eval = t + 30.0
            self._log(t, "info", f"{rover.id} switching to {c['label']} (Gemma, {lat:.1f} s): {why}", rover.id,
                      at=(gx, gy), source="gemma")
        elif c["kind"] == "target":
            ts = f.targets[c["tidx"]]
            if ts.status != "found" or ts.investigator is not None or not rover.can_investigate(t, ts):
                return
            ts.status, ts.investigator = "investigating", rover.id
            rover.start_investigation(t, c["tidx"])
            self._log(t, "info", f"{rover.id} switching to {c['label']} (Gemma, {lat:.1f} s): {why}", rover.id,
                      at=(gx, gy), source="gemma")
        elif c["kind"] == "return":
            rover._go_home(t, "gemma")
            self._log(t, "info", f"{rover.id} returning to recharge early (Gemma, {lat:.1f} s): {why}", rover.id,
                      at=rover.dock, source="gemma")

    # ================================================================== 2. anomalies
    def _on_anomaly(self, rover, a: dict) -> None:
        """Anomaly hook (sim thread, sim lock held): hand the job over, never wait here."""
        if not self.active or rover.state in PARKED:
            return
        kind = a["kind"]
        first = self.first_pending and kind != "no_returns"   # water: Gemma looks, the operator only if Gemma is unsure
        if first:
            self.first_pending = False
        facts = dict(a)
        job = {"rover": rover.id, "kind": kind, "t": float(a["t"]), "at": tuple(a["at"]), "first": first,
               "anomaly": self._anomaly_payload(rover, facts), "perception": self._perception(rover),
               "facts": facts, "wall": self.clock(), "image": False}
        if not self.planner.online:
            if first:                      # Gemma offline or warming: the rules ask on their own
                with self._mu:
                    self._done.append((job, None))
            return
        if self.wait_for_gemma:
            self._run_job(job)
            return
        if self._worker is None:
            self._worker = threading.Thread(target=self._worker_loop, name="advisor-eyes", daemon=True)
            self._worker.start()
        self._jobs.put(job)

    def _anomaly_payload(self, rover, a: dict) -> dict:
        """What Gemma is told about the anomaly (numbers with units). a: a copy of the rover's anomaly
        record; the obstacle height read from the map is added to it for the operator question."""
        kind = a["kind"]
        if kind == "new_obstacle":
            fm = self.fleet.map
            i, j = fm.ij(a["at"][0], a["at"][1])
            k = 3
            win = (slice(max(j - k, 0), j + k + 1), slice(max(i - k, 0), i + k + 1))
            hs = fm.h[win][fm.known[win]]
            height = float(hs.max() - hs.min()) if hs.size else 0.0
            a["height"] = round(height, 1)
            return {"type": "new_obstacle", "distance": a.get("dist", 0.0), "cells_changed": a.get("cells", 0),
                    "height": round(height, 1), "detail": "rocks on the planned path where the map showed open ground"}
        if kind == "slip":
            return {"type": "wheel_slip", "slip": a.get("slip", rover.slip), "wheel_speed": a.get("wheel_speed"),
                    "ground_speed": a.get("ground_speed"), "detail": "wheels spinning, rover barely moving"}
        if kind == "no_returns":
            out = {"type": "no_returns", "distance": a.get("dist", 0.0), "rays_missing": a.get("rays", 0),
                   "detail": "no LiDAR returns where flat ground was expected straight ahead"}
            if a.get("camera_liquid_pct") is not None:
                out["camera_liquid_pct"] = a["camera_liquid_pct"]
            if a.get("edge_spread") is not None:
                out["edge_spread"] = a["edge_spread"]
            return out
        return {"type": "low_visibility", "lidar_range": a.get("lidar_range", rover.lidar_eff),
                "nominal_range": self.cfg.lidar_range, "detail": "no LiDAR returns from far away"}

    def _worker_loop(self) -> None:
        while not self._stop.is_set():
            try:
                job = self._jobs.get(timeout=0.5)
            except queue.Empty:
                continue
            if job is None:
                return
            try:
                self._run_job(job)
            except Exception:              # noqa: BLE001
                self._error("anomaly job")

    def _run_job(self, job: dict) -> None:
        """Worker thread (or inline in tests): camera frame, then ask Gemma."""
        jpeg = None
        if self.frame_source is not None:
            try:
                jpeg = self.frame_source(job["rover"])
            except Exception:              # noqa: BLE001 (no camera: Gemma gets the numbers only)
                jpeg = None
        job["image"] = bool(jpeg)
        if jpeg:
            self.frames += 1
        if job.get("type") == "describe":
            tk = self.planner.submit_description(job["rover"], job["t"], job["perception"], job["obs"], jpeg)
        else:
            tk = self.planner.submit_interpretation(job["rover"], job["t"], job["perception"], job["anomaly"], jpeg,
                                                    INTERP_OPTIONS, default=RULE_REACTION[job["kind"]])
        if self.wait_for_gemma:
            self._wait(tk)
        with self._mu:
            self._done.append((job, tk))

    def _poll_interpretations(self, t: float) -> None:
        with self._mu:
            items = list(self._done)
        for item in items:
            job, tk = item
            res = None
            if tk is not None:
                res = self.planner.poll(tk)
                if res is None and self.clock() - job["wall"] < self.planner.cfg.timeout_s + 12.0:
                    continue               # still thinking (or the camera is slow)
            with self._mu:
                self._done.remove(item)
            if job.get("type") == "describe":
                self._on_description(job, res, t)
            else:
                self._on_interpretation(job, res, t)

    def _on_interpretation(self, job: dict, res, t: float) -> None:
        rover = self._rover(job["rover"])
        if rover is None:
            return
        kind, rid = job["kind"], rover.id
        ok = res is not None and res.ok
        wants = job["first"] or (ok and (res.action == "ask_operator" or res.confidence < self.cfg.operator_ask_below))
        ask = wants and self._can_ask(rover, t, job["first"])
        if ok:
            job["gemma_class"] = res.classification
            label = res.classification.replace("_", " ")
            src = "camera and sensors" if job["image"] else "sensors"
            head = (f"{rid} Gemma ({src}, {res.latency_ms / 1000.0:.1f} s): "
                    f"{'maybe ' if res.action == 'ask_operator' else ''}{label}, confidence {res.confidence:.2f}. "
                    f"{_quote(res.reason)}")
            if ask:
                tail = "Asking the operator; creeping along the safe route meanwhile."
            elif res.action == "ask_operator":
                tail = "Keeping the rule reaction."
            else:
                phrase, mode = self._apply_action(rover, kind, res.action, t, job)
                tail = {"agree": "Agrees with the rules: ", "override": "Safety rule wins: ",
                        "noted": ""}.get(mode, "Doing it: ")
                tail = (tail + phrase + ".") if tail else phrase[0].upper() + phrase[1:] + "."
            self._log(t, "info", f"{head} {tail}", rid, at=job["at"], source="gemma")
        elif res is not None and res.error and not res.error.startswith(("gemma ", "cancelled", "superseded")):
            self._log(t, "info", f"{rid} no usable answer from Gemma ({res.error.split(':')[0]}). "
                                 f"Keeping the rule reaction.", rid, at=job["at"])
        if ask:
            self._open_request(rover, kind, job, res if ok else None, t, tried=res is not None)

    def _apply_action(self, rover, kind: str, action: str, t: float, job: dict) -> tuple:
        """Apply a SAFE action on top of the rules' reaction. Returns (phrase, mode) with mode
        "agree" (same as the rules), "apply" (adds something safe) or "override" (refused)."""
        facts = job["facts"]
        if rover.state in PARKED:
            return "parked now, nothing to change", "noted"
        in_sand = rover.state == "recovering" and rover.rec_phase == "sand"
        if kind == "new_obstacle":
            if action == "reroute":
                self._keep_clear(job["at"])
                rover.plan_t = -1e9
                return "going around", "agree"
            if action == "proceed_slowly":
                rover.crawl_until, rover.crawl_v = t + SLOW_S, self.cfg.caution_v
                return f"approaching slowly ({self.cfg.caution_v:.1f} m/s for {SLOW_S:.0f} s) on the safe route", "apply"
            if action == "wait":
                rover.pause_until = t + WAIT_S
                return f"waiting {WAIT_S:.0f} s, then going around", "apply"
            if rover.state in DRIVING:     # back_out
                rover._start_recovery(t, job["at"])
                return "backing off, then going around", "apply"
            return "staying clear of it", "agree"
        if kind == "slip":
            slip = f"slip {facts.get('slip', 0.0) * 100:.0f}%"
            if action in ("back_out", "reroute"):
                return "backing out, then going around", "agree"
            if action == "proceed_slowly":
                return f"no driving on into soft ground, backing out ({slip})", "override"
            if in_sand:
                self.after_sand[rover.id] = WAIT_S
                return f"backing out of the soft ground first ({slip}), then waiting {WAIT_S:.0f} s", "apply"
            rover.pause_until = t + WAIT_S
            return f"waiting {WAIT_S:.0f} s", "apply"
        if kind == "no_returns":
            at = job["at"]
            if action == "reroute":
                if job.get("gemma_class") == "liquid" and rover.mark_water_here(t, at, "gemma"):
                    return "marking it as water for the fleet and going around", "apply"
                if self.fleet.water_zone_at(at[0], at[1], margin=0.5) is not None:
                    return "it is marked as water, going around", "agree"
                return "keeping off it and staying on mapped ground", "agree"
            if action == "proceed_slowly":
                rover.crawl_until, rover.crawl_v = t + SLOW_S, self.cfg.caution_v
                return (f"approaching slowly ({self.cfg.caution_v:.1f} m/s for {SLOW_S:.0f} s) on mapped ground; the "
                        f"water sensor stops me if it is liquid"), "apply"
            rover.pause_until = t + WAIT_S
            return f"waiting {WAIT_S:.0f} s", "apply"
        if action == "wait":               # low visibility
            rover.pause_until, rover.pause_dust = t + DUST_WAIT_S, True
            return f"waiting up to {DUST_WAIT_S:.0f} s for the dust to clear", "apply"
        mode = "agree" if action in ("proceed_slowly", "reroute") else "override"
        return f"going on at {self.cfg.degraded_v_max:.1f} m/s on mapped ground", mode

    def _keep_clear(self, at) -> None:
        """'Go around': give the new obstacle a little extra room in the shared map (never on a rover)."""
        x, y = at
        if all(math.hypot(r.x - x, r.y - y) > 1.2 for r in self.sim.rovers):
            self.fleet.map.mark_blocked(x, y, 0.6)
            self.fleet.invalidate_costs()

    # ================================================================== 3. questions
    def _can_ask(self, rover, t: float, first: bool) -> bool:
        if rover.state in PARKED or any(q["rover"] == rover.id for q in self.requests.values()):
            return False
        if len(self.requests) >= self.cfg.max_open_questions:
            return False
        return first or t - self.last_question.get(rover.id, -1e9) >= self.cfg.question_cooldown_s

    def _question(self, rover, kind: str, job: dict, res, tried: bool) -> str:
        a = job["facts"]
        if kind == "new_obstacle":
            q = (f"New rocks block my path {a.get('dist', 0.0):.0f} m ahead where my map showed open ground "
                 f"(LiDAR: {a.get('cells', 0)} map cells changed, about {a.get('height', 0.0):.1f} m tall).")
        elif kind == "slip":
            q = (f"My wheels spin but I barely move (slip {a.get('slip', 0.0) * 100:.0f}%, ground speed "
                 f"{a.get('ground_speed', 0.0):.2f} m/s). Backing out now.")
        elif kind == "no_returns":
            cam = (f" Camera: {a['camera_liquid_pct']:.0f}% of the lower view is blue."
                   if a.get("camera_liquid_pct") is not None else "")
            q = (f"My LiDAR gets no returns {a.get('dist', 0.0):.0f} m ahead where ground should be "
                 f"({a.get('rays', 0)} rays).{cam}")
        else:
            q = f"My LiDAR only reaches {a.get('lidar_range', 0.0):.0f} m (normally {self.cfg.lidar_range:.0f} m)."
        if res is not None:
            q += f" Gemma thinks: {res.classification.replace('_', ' ')} (confidence {res.confidence:.2f})."
        elif tried:
            q += " Gemma could not tell."
        return q + " What should I do?"

    def _open_request(self, rover, kind: str, job: dict, res, t: float, tried: bool) -> None:
        self.req_seq += 1
        qid = f"q{self.req_seq}"
        opts = QUESTION_OPTIONS[kind]
        q = {"id": qid, "rover": rover.id, "t": round(t, 1), "question": self._question(rover, kind, job, res, tried),
             "options": opts, "default": opts[0][0], "deadline": self.clock() + self.cfg.operator_timeout_s,
             "image": f"/api/camera/{rover.id}.jpg?view=fpv", "kind": kind, "job": job}
        self.requests[qid] = q
        self.last_question[rover.id] = t
        rover.crawl_until, rover.crawl_v = math.inf, CRAWL_V
        src = "gemma" if res is not None else "rules"
        self.say(rover.id, t, f"{rover.id}: {q['question']}", "question", source=src, request_id=qid)
        if res is None:
            why = "first anomaly since landing" if job["first"] else "not sure what it sees"
            self._log(t, "warn", f"{rover.id} {ANOMALY_NAME[kind]}: {why}, asking the operator. Creeping at "
                                 f"{CRAWL_V:.2f} m/s meanwhile.", rover.id, at=job["at"])

    def _close(self, qid: str, status: str, t: float):
        q = self.requests.pop(qid)
        self.closed[qid] = status
        while len(self.closed) > 200:
            self.closed.popitem(last=False)
        rover = self._rover(q["rover"])
        if rover is not None:
            rover.crawl_until = t
        return q, rover

    def answer(self, qid: str, option: str) -> str:
        """POST /api/answer. Returns "ok", "unknown", "closed" or "bad_option"."""
        q = self.requests.get(qid)
        if q is None:
            return "closed" if qid in self.closed else "unknown"
        labels = dict(q["options"])
        if option not in labels:
            return "bad_option"
        t = self.sim.t
        q, rover = self._close(qid, "answered", t)
        rid = q["rover"]
        self.say("operator", t, labels[option], "answer", to=rid, source=None, request_id=qid)
        if rover is None:
            return "ok"
        if q["kind"] == "lifeform":
            phrase = self._apply_life(rover, option, t, q)
        else:
            phrase, _ = self._apply_action(rover, q["kind"], option, t, q["job"])
        self.say(rid, t, f"{rid}: {phrase}, thanks.", "ack", request_id=qid)
        self._log(t, "info", f"{rid} operator chose '{labels[option]}': {phrase}.", rid, at=q["job"]["at"],
                  source="operator")
        return "ok"

    def _expire(self, q: dict, t: float) -> None:
        q, rover = self._close(q["id"], "expired", t)
        rid, label = q["rover"], dict(q["options"])[q["default"]]
        if rover is not None:
            if q["kind"] == "lifeform":
                self._apply_life(rover, q["default"], t, q)
            else:
                self._apply_action(rover, q["kind"], q["default"], t, q["job"])
        secs = self.cfg.operator_timeout_s
        self.say(rid, t, f"{rid}: no reply in {secs:.0f} s, taking the safe option: {label.lower()}.", "info",
                 request_id=q["id"])
        self._log(t, "info", f"{rid} no operator reply in {secs:.0f} s. Taking the safe option: {label.lower()}.",
                  rid, at=q["job"]["at"])

    # ================================================================== 5. lifeforms
    def on_sighting(self, rover, tr, t: float) -> bool:
        """A rover started observing a moving track. Hand Gemma the camera frame and the numbers; the
        answer (or the operator) decides follow / observe from here / stop. Returns True if tier 2
        will decide, False to leave it to the rules (tier 2 switched off)."""
        if not self.active:
            return False
        first = self.first_life_pending
        self.first_life_pending = False
        px, py = tr.predict(t)
        d = math.hypot(px - rover.x, py - rover.y)
        bearing = math.degrees(math.atan2(math.sin(math.atan2(py - rover.y, px - rover.x) - rover.yaw),
                                          math.cos(math.atan2(py - rover.y, px - rover.x) - rover.yaw)))
        obs = {"distance": round(d, 1), "speed": round(tr.speed, 2), "size": round(tr.size, 1),
               "bearing_deg": round(bearing, 0), "seen_for_s": round(t - tr.t0, 1)}
        job = {"type": "describe", "rover": rover.id, "track": tr.id, "t": float(t), "at": (px, py), "first": first,
               "obs": obs, "perception": self._perception(rover), "wall": self.clock(), "image": False}
        if not self.planner.online:
            with self._mu:
                self._done.append((job, None))      # rules alone: ask about the first one, follow the rest
            return True
        if self.wait_for_gemma:
            self._run_job(job)
            return True
        if self._worker is None:
            self._worker = threading.Thread(target=self._worker_loop, name="advisor-eyes", daemon=True)
            self._worker.start()
        self._jobs.put(job)
        return True

    def _on_description(self, job: dict, res, t: float) -> None:
        rover = self._rover(job["rover"])
        f = self.fleet
        tr = f.tracker.get(job["track"])         # follows a merge into an older track
        if rover is None or rover.obs is None or tr is None or rover.obs["track"] != tr.id:
            return                               # the observation already ended
        ok = res is not None and res.ok
        unsure = ok and (res.action == "ask_operator" or res.confidence < self.cfg.operator_ask_below)
        ask = (job["first"] or unsure) and self._can_ask(rover, t, job["first"])
        sd = self.cfg.observe_standoff
        if ok:
            tr.label = res.label
            life = res.classification == "lifeform"
            if life and not unsure:
                rover.obs["life"] = True
                f.confirm_lifeform(tr, t)
            src = "camera" if job["image"] else "track numbers"
            head = (f"{rover.id} Gemma ({src}, {res.latency_ms / 1000.0:.1f} s): {res.label} "
                    f"(confidence {res.confidence:.2f}). {_quote(res.reason)}")
            if ask:
                tail = "Asking the operator; watching from here meanwhile."
            elif res.action == "ignore" and not life:
                tail = "Not a lifeform. Resuming exploration."
            elif res.action == "approach" or (life and res.action != "observe"):
                tail = f"Approaching to {sd:.0f} m to observe."
            else:
                tail = "Watching from here."
            f.log(t, "discovery" if life else "info", f"{head} {tail}", rover.id, at=job["at"], source="gemma")
        elif res is not None and res.error and not res.error.startswith(("gemma ", "cancelled", "superseded")):
            f.log(t, "info", f"{rover.id} no usable answer from Gemma about track {tr.id} "
                             f"({res.error.split(':')[0]}). The rules decide.", rover.id, at=job["at"])
        if ask:
            self._open_life_request(rover, tr, job, res if ok else None, t)
            rover.set_obs_mode(t, "here")
        elif ok and res.action == "ignore" and res.classification != "lifeform":
            rover.set_obs_mode(t, "stop")
        elif ok and res.action == "observe" or unsure:
            rover.set_obs_mode(t, "here")
        else:
            rover.set_obs_mode(t, "follow")

    def _open_life_request(self, rover, tr, job: dict, res, t: float) -> None:
        from .fleet import compass
        self.req_seq += 1
        qid = f"q{self.req_seq}"
        px, py = tr.predict(t)
        d = math.hypot(px - rover.x, py - rover.y)
        guess = f" Gemma thinks: {res.label} (confidence {res.confidence:.2f})." if res is not None else ""
        question = (f"Possible lifeform {d:.0f} m {compass(px - rover.x, py - rover.y)} (LiDAR track {tr.id}, "
                    f"{tr.speed:.1f} m/s, about {tr.size:.1f} m).{guess} Follow it or keep exploring?")
        q = {"id": qid, "rover": rover.id, "t": round(t, 1), "question": question, "options": LIFE_OPTIONS,
             "default": LIFE_DEFAULT, "deadline": self.clock() + self.cfg.operator_timeout_s,
             "image": f"/api/camera/{rover.id}.jpg?view=fpv", "kind": "lifeform", "job": job}
        self.requests[qid] = q
        self.last_question[rover.id] = t
        self.say(rover.id, t, f"{rover.id}: {question}", "question", source="gemma" if res is not None else "rules",
                 request_id=qid)
        if res is None:
            self.fleet.log(t, "info", f"{rover.id} first lifeform sighting: asking the operator. Watching from here "
                                      f"meanwhile.", rover.id, at=(px, py))

    def _apply_life(self, rover, option: str, t: float, q: dict) -> str:
        job = q["job"]
        tr = self.fleet.tracker.get(job["track"])
        if rover.obs is None or tr is None or rover.obs["track"] != tr.id:
            return "the observation already ended"
        sd = self.cfg.observe_standoff
        if option == "follow":
            rover.set_obs_mode(t, "follow")
            return f"following it at about {sd:.0f} m, never closer than {self.cfg.observe_min_dist:.0f} m"
        if option == "keep_exploring":
            for k in rover.obs["seen"]:
                tr = self.fleet.tracker.get(k)
                if tr is not None:
                    tr.ignored = True
            rover.set_obs_mode(t, "stop")
            return "back to exploring, leaving it alone"
        rover.set_obs_mode(t, "here")
        return "watching from here, I will not get closer"

    # ================================================================== 4. operator chat
    def operator_message(self, text: str, rover_hint: str | None = None) -> str:
        """POST /api/chat: post the message now, interpret it in the background. Returns the chat id."""
        text = " ".join(str(text).split())[:300]
        t = self.sim.t
        ctx = self.fleet_context()
        preview = parse_command(text, ctx)["command"].get("rover")
        cid = self.say("operator", t, text, "command", to=rover_hint or preview or None, source=None)
        tk = self.planner.submit_command(t, text, ctx)
        self.commands.append({"ticket": tk, "cid": cid, "hint": rover_hint, "t": t, "wall": self.clock()})
        if self.wait_for_gemma:
            self._wait(tk)
        return cid

    def _poll_commands(self, t: float) -> None:
        for c in list(self.commands):
            res = self.planner.poll(c["ticket"])
            if res is None:
                if self.clock() - c["wall"] < self.planner.cfg.timeout_s + 3.0:
                    continue
                self.commands.remove(c)
                self.say("fleet", t, "Sorry, that message got lost. Please send it again.", "info")
                continue
            self.commands.remove(c)
            self._execute(res, c, t)

    def _reply(self, rid, t, text, kind, src):
        self.say(rid or "fleet", t, text, kind, source=src)

    def _execute(self, res, c: dict, t: float) -> None:
        cmd = dict(res.command or {"type": "unclear", "question": res.question or "Sorry?"})
        src = "gemma" if res.ok else "rules"
        typ = cmd["type"]
        rid = cmd.get("rover") or c["hint"]
        ids = [r.id for r in self.sim.rovers]
        if rid not in ids and rid != "all":
            rid = None
        if typ == "unclear":
            frm = rid if rid in ids else "fleet"
            text = res.question or cmd.get("question") or "Sorry, I did not understand."
            self._reply(frm, t, (f"{frm}: " if frm != "fleet" else "") + text, "question", src)
        elif typ == "go_to":
            self._cmd_go_to(rid, float(cmd["x"]), float(cmd["y"]), t, src)
        elif typ == "explore_direction":
            self._cmd_explore(rid, cmd["direction"], t, src)
        elif typ == "sample_target":
            self._cmd_sample(rid, cmd["target_id"], t, src)
        elif typ in ("return_home", "hold", "resume"):
            getattr(self, "_cmd_" + typ)(rid, t, src)
        elif typ == "avoid_area":
            self._cmd_avoid(float(cmd["x"]), float(cmd["y"]), float(cmd.get("radius") or 3.0), t, src)

    # ---- helpers for commands
    def _targets_for(self, rid) -> list:
        return list(self.sim.rovers) if rid in (None, "all") else [r for r in self.sim.rovers if r.id == rid]

    def _busy_reason(self, r) -> str | None:
        """Why this rover cannot take a new job right now (safety rules), or None."""
        if r.state == "fault":
            return "my motors are offline (rebooting)"
        if r.state == "returning" and r.return_reason in ("battery", "range"):
            return f"battery {r.battery * 100:.0f}%, I am returning to charge"
        return None

    def _cancel_go(self, rid: str, t: float, why: str) -> None:
        """A go_to still being planned is replaced by a newer command: say so (one reply each)."""
        job = self.go_jobs.pop(rid, None)
        if job is not None:
            x, y = job["goal"]
            self._reply(rid, t, f"{rid}: dropped the trip to ({x:g}, {y:g}), {why}.", "info", "rules")

    def _go_need(self, r, x: float, y: float, dist: float) -> float:
        return r.energy_for_distance(dist + 1.3 * math.hypot(x, y)) + r.battery_margin

    def _free_up(self, r, t: float, why: str) -> None:
        """The operator gave this rover a job: stop holding, drop other plans."""
        r.held = False
        r.op_goal = None
        r.dir_bias = None
        r.sticky_goal = None
        r.pause_until = min(r.pause_until, t)
        self._cancel_go(r.id, t, why)
        self.choices.pop(r.id, None)
        if r.state == "observing":
            r.end_observing(t, "operator")
        if r.target_idx is not None:
            r._drop_target(t, why)
        if r.state in ("idle", "charging") or (r.state == "returning" and r.return_reason not in ("battery", "range")):
            r.state = "exploring"
            r.goal, r.path = None, []

    def _cmd_go_to(self, rid, x: float, y: float, t: float, src: str) -> None:
        half = (self.cfg.nx - 1) * self.cfg.res / 2.0
        name = rid or "fleet"
        pre = f"{rid}: " if rid else ""
        if rid == "all":
            self._reply("fleet", t, f"One rover at a time for go to, please. Which one should go to ({x:g}, {y:g})?",
                        "question", src)
            return
        if abs(x) > half - 2.0 or abs(y) > half - 2.0:
            self._reply(name, t, f"{pre}can't, ({x:g}, {y:g}) is outside the map (x and y from {-half + 2.0:g} "
                                 f"to {half - 2.0:g} m).", "refuse", "rules")
            return
        z = self.fleet.soft_zone_at(x, y)
        if z is not None:
            self._reply(name, t, f"{pre}can't, ({x:g}, {y:g}) is in soft ground {z[3]} found.", "refuse", "rules")
            return
        fm = self.fleet.map
        i, j = fm.ij(x, y)
        if fm.blocked[j, i] or (fm.known[j, i] and fm.slope[j, i] > self.fleet.slope_limit):
            self._reply(name, t, f"{pre}can't, ({x:g}, {y:g}) is blocked or steeper than the "
                                 f"{self.fleet.slope_limit:.0f} deg slope limit (map: {fm.slope[j, i]:.0f} deg).",
                        "refuse", "rules")
            return
        if rid is None:                    # nearest rover that can do it
            ok = [r for r in self.sim.rovers if self._busy_reason(r) is None and r.state != "recovering"
                  and r.battery > self._go_need(r, x, y, 1.3 * math.hypot(x - r.x, y - r.y))]
            if not ok:
                self._reply("fleet", t, f"No rover can go to ({x:g}, {y:g}) now (battery or motors).", "refuse", "rules")
                return
            rid = min(ok, key=lambda r: math.hypot(x - r.x, y - r.y)).id
        r = self._rover(rid)
        busy = self._busy_reason(r)
        if busy:
            self._reply(rid, t, f"{rid}: can't, {busy}.", "refuse", "rules")
            return
        d = 1.3 * math.hypot(x - r.x, y - r.y)
        need = self._go_need(r, x, y, d)
        if need > r.battery:
            self._reply(rid, t, f"{rid}: can't, battery {r.battery * 100:.0f}% would not get me there and back "
                                f"(needs about {need * 100:.0f}% with the {r.battery_margin * 100:.0f}% margin).",
                        "refuse", "rules")
            return
        self._cancel_go(rid, t, f"new destination ({x:g}, {y:g})")
        self.go_jobs[rid] = {"goal": (x, y), "t0": t, "src": src}

    def _plan_go(self, rover, t: float) -> None:
        job = self.go_jobs[rover.id]
        rid, (x, y) = rover.id, job["goal"]
        busy = self._busy_reason(rover)
        if busy:
            del self.go_jobs[rid]
            self._reply(rid, t, f"{rid}: can't, {busy}.", "refuse", "rules")
            return
        if rover.state == "recovering":
            return                         # finish the safety manoeuvre first
        path = self.fleet.plan(t, rover, (x, y), max_exp=30000)
        if path is PENDING:
            if t - job["t0"] > GO_PLAN_S:
                del self.go_jobs[rid]
                self._reply(rid, t, f"{rid}: can't, no path to ({x:g}, {y:g}) found in time.", "refuse", "rules")
            return
        del self.go_jobs[rid]
        if path is None:
            self._reply(rid, t, f"{rid}: can't, no safe path to ({x:g}, {y:g}) within the "
                                f"{self.fleet.slope_limit:.0f} deg slope limit.", "refuse", "rules")
            return
        plen, smax = path_stats(self.fleet.map, [(rover.x, rover.y)] + list(path))
        need = self._go_need(rover, x, y, plen)
        if need > rover.battery:
            self._reply(rid, t, f"{rid}: can't, battery {rover.battery * 100:.0f}% would not get me there and back "
                                f"({plen:.0f} m there, needs about {need * 100:.0f}% with the margin).",
                        "refuse", "rules")
            return
        self._free_up(rover, t, "the operator sent it elsewhere")
        rover.state = "exploring"
        rover.op_goal, rover.op_fails = (x, y), 0
        rover.goal, rover.goal_t = (x, y), t
        rover._set_path(t, path)
        after = (rover.battery - need + rover.battery_margin) * 100.0
        self._reply(rid, t, f"{rid}: heading to ({x:g}, {y:g}), {plen:.0f} m. Max slope on the way {smax:.0f} deg, "
                            f"battery after return about {after:.0f}%.", "ack", job["src"])
        self._log(t, "info", f"{rid} heading to ({x:.0f}, {y:.0f}) for the operator, {plen:.0f} m, max slope "
                             f"{smax:.0f} deg.", rid, at=(x, y), source="operator")

    def _cmd_explore(self, rid, direction: str, t: float, src: str) -> None:
        if rid is None:                    # the working rover with the most charge
            free = [r for r in self.sim.rovers if self._busy_reason(r) is None]
            if not free:
                self._reply("fleet", t, "No rover is free to explore right now.", "refuse", "rules")
                return
            rid = max(free, key=lambda r: r.battery).id
        dx, dy = DIRS[direction]
        lines = []
        for r in self._targets_for(rid):
            busy = self._busy_reason(r)
            if busy:
                lines.append((r.id, f"can't, {busy}", "refuse"))
                continue
            self._free_up(r, t, f"exploring {direction} for the operator")
            r.dir_bias = (dx, dy, t + EXPLORE_BIAS_S)
            r.next_goal_eval = r.next_goal_try = t
            best = None
            for (fx, fy, _s, _i) in self.fleet.usable_frontiers(t):
                d = math.hypot(fx - r.x, fy - r.y)
                if d > 2.5 and ((fx - r.x) * dx + (fy - r.y) * dy) / d >= 0.7:
                    best = d if best is None else min(best, d)
            tail = (f"nearest frontier that way {best:.0f} m" if best is not None
                    else f"no open frontier to the {direction} yet, I will prefer that side")
            lines.append((r.id, f"exploring {direction} for the next {EXPLORE_BIAS_S / 60:.0f} minutes, {tail}", "ack"))
            self._log(t, "info", f"{r.id} prefers frontiers to the {direction} for {EXPLORE_BIAS_S / 60:.0f} min "
                                 f"(operator).", r.id, source="operator")
        self._reply_many(rid, lines, t, src)

    def _reply_many(self, rid, lines: list, t: float, src: str) -> None:
        """One chat reply per command: from the rover, or from the fleet for several rovers."""
        if not lines:
            return
        if len(lines) == 1 and rid != "all":
            r_id, text, kind = lines[0]
            self._reply(r_id, t, f"{r_id}: {text}.", kind, src if kind != "refuse" else "rules")
            return
        kind = "ack" if any(k == "ack" for _, _, k in lines) else "refuse"
        self._reply("fleet", t, " ".join(f"{r_id}: {text}." for r_id, text, _ in lines), kind,
                    src if kind == "ack" else "rules")

    def _cmd_sample(self, rid, tid: str, t: float, src: str) -> None:
        f = self.fleet
        k = next((n for n, ts in enumerate(f.targets) if ts.spec.id == tid and ts.status is not None), None)
        if k is None:
            known = ", ".join(ts.spec.id for ts in f.targets if ts.status is not None) or "none yet"
            self._reply("fleet", t, f"I don't know target {tid}. Known targets: {known}.", "question", src)
            return
        ts = f.targets[k]
        if rid == "all":
            rid = None
        if ts.status == "sampled":
            self._reply(rid or "fleet", t, f"{rid + ': ' if rid else ''}{tid} is already sampled.", "refuse", "rules")
            return
        if ts.investigator is not None and ts.investigator != rid:
            self._reply(rid or "fleet", t, f"{rid + ': ' if rid else ''}{ts.investigator} is already sampling {tid}.",
                        "refuse", "rules")
            return
        tx, ty = ts.spec.x, ts.spec.y

        def need(r):
            d = math.hypot(tx - r.x, ty - r.y)
            return r.energy_for_distance(1.3 * d + 1.3 * math.hypot(tx, ty)) + r.battery_margin + 0.05

        if rid is None:
            ok = [r for r in f.rovers if self._busy_reason(r) is None and r.state != "recovering"
                  and r.battery > need(r) and r.free_tubes() > 0]
            if not ok:
                self._reply("fleet", t, f"No rover can reach {tid} and get back with a free sample tube right now.",
                            "refuse", "rules")
                return
            rid = min(ok, key=lambda r: math.hypot(tx - r.x, ty - r.y)).id
        r = self._rover(rid)
        busy = self._busy_reason(r) or ("I am backing out of trouble, ask again in a moment"
                                        if r.state == "recovering" else None)
        if busy:
            self._reply(rid, t, f"{rid}: can't, {busy}.", "refuse", "rules")
            return
        d = math.hypot(tx - r.x, ty - r.y)
        if r.free_tubes() <= 0 and r.target_idx != k:
            self._reply(rid, t, f"{rid}: can't, all {r.capacity} sample tubes are full. They go to the lander cache "
                                f"on my next return (say 'return home').", "refuse", "rules")
            return
        if r.battery < need(r):
            self._reply(rid, t, f"{rid}: can't, battery {r.battery * 100:.0f}% would not get me to {tid} ({d:.0f} m) "
                                f"and back.", "refuse", "rules")
            return
        if r.target_idx == k:
            self._reply(rid, t, f"{rid}: already on my way to sample {tid}, {d:.0f} m.", "ack", src)
            return
        self._free_up(r, t, "the operator asked for another target")
        r.state = "exploring"
        ts.status, ts.investigator = "investigating", rid
        r.start_investigation(t, k)
        self._reply(rid, t, f"{rid}: going to sample {tid} ({KIND_LABEL[ts.spec.kind]}), {d:.0f} m away, "
                            f"{r.free_tubes()} of {r.capacity} tubes free.", "ack", src)
        self._log(t, "info", f"{rid} heading to sample {tid} for the operator, {d:.0f} m.", rid, at=(tx, ty),
                  source="operator")

    def _cmd_return_home(self, rid, t: float, src: str) -> None:
        lines = []
        for r in self._targets_for(rid):
            dd = math.hypot(r.x - r.dock[0], r.y - r.dock[1])
            if r.state == "fault":
                lines.append((r.id, "can't, my motors are offline (rebooting)", "refuse"))
            elif r.state == "charging" or (r.state == "idle" and dd < 1.6):
                r.held = False
                lines.append((r.id, f"already at the lander ({r.battery * 100:.0f}%)", "ack"))
            elif r.state == "returning":
                lines.append((r.id, f"already returning, {dd:.0f} m to go", "ack"))
            elif r.state == "recovering":
                self.home_after.add(r.id)
                lines.append((r.id, f"backing out of trouble first, then returning, {dd:.0f} m", "ack"))
            else:
                self._free_up(r, t, "returning for the operator")
                r.state = "exploring"
                r._go_home(t, "operator")
                lines.append((r.id, f"returning to the lander, {dd:.0f} m", "ack"))
                self._log(t, "info", f"{r.id} returning to the lander for the operator, {dd:.0f} m.", r.id,
                          at=r.dock, source="operator")
        self._reply_many(rid, lines, t, src)

    def _enter_hold(self, r, t: float) -> None:
        r._drop_target(t, "holding for the operator")
        self._cancel_go(r.id, t, "holding for the operator")
        self.choices.pop(r.id, None)
        r.op_goal = None
        r.sticky_goal = None
        r.goal, r.path = None, []
        r.state = "idle"
        r._set_cmd(0.0, 0.0)

    def _cmd_hold(self, rid, t: float, src: str) -> None:
        lines = []
        for r in self._targets_for(rid):
            if r.state == "fault":
                lines.append((r.id, "already stopped (motors offline)", "ack"))
                continue
            if r.state == "returning" and r.return_reason in ("battery", "range"):
                lines.append((r.id, f"can't hold, battery {r.battery * 100:.0f}% and I must get home", "refuse"))
                continue
            r.held = True
            r.dir_bias = None
            if r.state == "recovering":
                lines.append((r.id, "backing out of trouble first, then holding", "ack"))
            elif r.state == "charging":
                lines.append((r.id, f"holding at the lander, charging ({r.battery * 100:.0f}%)", "ack"))
            else:
                if r.state != "idle":
                    self._enter_hold(r, t)
                lines.append((r.id, f"holding at ({r.x:.0f}, {r.y:.0f}), say resume to continue", "ack"))
            self._log(t, "info", f"{r.id} holding for the operator.", r.id, at=(r.x, r.y), source="operator")
        self._reply_many(rid, lines, t, src)

    def _cmd_resume(self, rid, t: float, src: str) -> None:
        """Back to full autonomy: cancel holds, operator trips, directions and operator returns."""
        lines = []
        for r in self._targets_for(rid):
            ordered_home = r.state == "returning" and r.return_reason in ("operator", "gemma")
            was = r.held or r.op_goal is not None or r.dir_bias is not None or r.state == "idle" or ordered_home
            r.held = False
            r.op_goal = None
            r.dir_bias = None
            r.pause_until = min(r.pause_until, t)
            self._cancel_go(r.id, t, "resuming exploration")
            if r.state == "fault":
                lines.append((r.id, "my motors are offline, I will carry on after the reboot", "ack"))
                continue
            if self.fleet.complete:
                lines.append((r.id, "exploration is complete, staying at the lander", "ack"))
                continue
            if r.state == "idle" or ordered_home:
                at_dock = math.hypot(r.x - r.dock[0], r.y - r.dock[1]) < 1.6
                r.state = "charging" if at_dock and r.battery < 0.99 else "exploring"
                r.goal, r.path = None, []
                r.next_goal_try = r.next_goal_eval = t
            if r.state == "returning":
                lines.append((r.id, f"returning to charge first (battery {r.battery * 100:.0f}%)", "ack"))
            elif r.state == "charging":
                lines.append((r.id, f"charging first ({r.battery * 100:.0f}%), then exploring", "ack"))
            else:
                lines.append((r.id, "resuming exploration" if was else "already exploring", "ack"))
        self._reply_many(rid, lines, t, src)

    def _cmd_avoid(self, x: float, y: float, radius: float, t: float, src: str) -> None:
        half = (self.cfg.nx - 1) * self.cfg.res / 2.0
        radius = max(0.5, min(30.0, radius))
        if abs(x) > half or abs(y) > half:
            self._reply("fleet", t, f"Can't, ({x:g}, {y:g}) is outside the map.", "refuse", "rules")
            return
        if math.hypot(x, y) < radius + self.cfg.lander_radius + 0.5:
            self._reply("fleet", t, f"Can't, a {radius:g} m circle at ({x:g}, {y:g}) would cover the lander pad "
                                    f"({math.hypot(x, y):.1f} m from it).", "refuse", "rules")
            return
        inside = [(r.id, math.hypot(r.x - x, r.y - y)) for r in self.sim.rovers
                  if math.hypot(r.x - x, r.y - y) < radius + 0.8]
        if inside:
            rid, d = inside[0]
            self._reply("fleet", t, f"Can't, {rid} is inside that area ({d:.1f} m from its centre). Move it out first.",
                        "refuse", "rules")
            return
        self.fleet.map.mark_avoid(x, y, radius)
        self.fleet.invalidate_costs()
        self.fleet.frontiers_t = -1e9
        self.fleet.refresh_reach()
        for r in self.sim.rovers:
            r.plan_t = -1e9
        self._reply("fleet", t, f"Marked ({x:g}, {y:g}) radius {radius:g} m as off limits. All rovers will plan "
                                f"around it.", "ack", src)
        self._log(t, "info", f"Operator marked ({x:.0f}, {y:.0f}) radius {radius:g} m as off limits for all rovers.",
                  None, at=(x, y), source="operator")
