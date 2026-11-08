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
6. Questions (protocol v5). When the operator's message is a question ("rover 2, what have you
   found?"), the addressed rover (or the fleet) builds a fact sheet from its own memory
   (answers.py) and Gemma writes a short reply from it; any number or id not in the fact sheet, a
   slow answer (over the 3 s budget) or Gemma offline means the rule-based answer is used. Replies
   are kind "reply" with reply_to = the operator message id (acks and refusals carry reply_to too).

7. Places (protocol v6). "go to the water source", "can we investigate the lake for signs of life": the
   place is resolved from fleet knowledge only (places.py). One match (or a qualifier that picks one):
   the named rover, or the closest free one, goes through the same go_to path (every safety check), and
   with "investigate" surveys on arrival (rover.py: stops, turns slowly through 360 deg for about 30 s
   with the LiDAR tracker on) and reports what moved and what it saw (kind "info", reply_to the command).
   Several matches: a choose_place request with numbered options carrying x, y; the rover holds until
   the operator picks one (POST /api/answer, or a chat message such as "the second one").

8. Protocol v7. Area focus: "explore around the lake", "go to the vent and look around", "explore there",
   "explore north of the lander", and "investigate X" (survey first): the rover concentrates on the unknown
   ground in a circle (rover.focus, fleet.choose_goal) and reports what it mapped and found there
   (focus_report). Chat memory (chatmem.py): the last messages and what they talked about, so "go back
   there", "the second one", "that lake", "do that again" and "same for rover 3" resolve, and Gemma gets a few
   lines of it. Lifeforms: a herd nobody asked about yet is followed at the standoff while the question is
   open (no holding still); hazards, water and a low battery end the follow, with a chat line.

9. Protocol v8. Autonomy mode (fleet.mode): "autonomous" (default) never waits for the operator: where a rover would
   ask (an anomaly, a lifeform sighting, several places matching a command) Gemma decides (the rules when Gemma is
   offline, the safest option below 0.5 confidence); each decision is an event with a `decision` object plus a chat
   line. "supervised" asks every such decision, with Gemma's pick marked as a suggestion. Fleet priorities
   ("prioritize the gaps", "the north", "water", "sampling", "back to normal") weigh every rover's goal choice
   (fleet.priority_bonus) and are managed here (plan per rover, reassignment, done). Unexplored gaps are a place kind.
   An operator focus on a mapped area is a re-survey (rover.py). Area questions ("what did we find there?") get a
   fact sheet scoped to that place (answers.area_facts).

Requests never expire by default (OPERATOR_TIMEOUT_S 0): the rover holds until the operator answers,
except for the safety moves in Rover._may_move_while_asking (a lifeform question is answered while following).
Above 0 the old countdown and safe default apply. A new operator job for a rover drops that rover's open
question (the operator decided), and so does a battery that says go home (on_battery_return).

Nothing here ever waits for the model in the 5 Hz loop: submits and polls take microseconds.
"""
from __future__ import annotations

import math
import queue
import re
import threading
import time
import traceback
from collections import OrderedDict

import numpy as np

from . import places as P
from . import gaps as G
from .answers import (PER_ROVER_TOPICS, area_answer, area_facts, fact_lines, fleet_sheet, rover_sheet, rule_answer,
                      why_text)
from .chatmem import ChatMemory, Ref
from .fleet import MODES, OVERLAP_RADIUS, PRIORITY_WATER_M, compass
from .gemma import (_MENTIONED_AFTER_RE, _MENTIONED_BEFORE_RE, _PLACE_VERB_RE, _SAMPLE_WORD_RE, GemmaConfig,
                    GemmaPlanner, _fleet_context, _place_words, _resolve_rover, parse_command, question_intent,
                    wants_each)
from .planner import PENDING
from .rover import SURVEY_S
from .terrain import KIND_LABEL

EM_DASH = chr(0x2014)
PARKED = ("idle", "charging", "fault", "disabled", "towed", "wrecked", "repairing")
DRIVING = ("exploring", "investigating", "returning")
HURT = ("disabled", "towed", "wrecked")       # protocol v9: out of action

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
LIFE_OPTIONS = [("follow", "Keep following"), ("observe_here", "Observe from here"), ("keep_exploring", "Keep exploring")]
LIFE_DEFAULT = "observe_here"     # the safest: it never gets closer
ASKED_HERD_M = 15.0       # protocol v7: a sighting this close to a herd already asked about is that herd
ACTIONS = ("go_to", "go_to_place", "explore_direction", "sample_target", "return_home", "hold", "resume",
           "avoid_area")   # commands "do that again" can repeat
# a question that refers back to a place ("what did you find there?", "anything around it?")
_QREF_RE = re.compile(r"\b(?:find|found|see|saw|seen|spot|spotted|map|mapped|explor\w*|at|over|around|near|from|"
                      r"in|out|anything|something|nothing|left)\s+there\b|\bthere\s*[?.!]*$"
                      r"|\bthat\s+(?:place|spot|area|location|site)\b|\bthe\s+same\s+(?:place|spot|area)\b"
                      r"|\b(?:around|near|at|by|next\s+to|about)\s+it\b")
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
FREE_CHARGE = 0.5         # a rover charging below this battery is not free for a place command
REQUEST_KIND = {"lifeform": "lifeform", "choose_place": "choose_place", "rescue": "rescue"}   # else: "anomaly"
# protocol v8: decisions a rover makes on its own (autonomous mode), or Gemma's suggestion (supervised)
CONF_FLOOR = 0.5          # below this Gemma confidence the safest option is taken
DET_OVERRIDE = 0.9        # an onboard detection this sure of a different class wins over Gemma's obstacle label
DET_WAIT_S = 1.5          # sim s: a new-obstacle interpretation waits this long for the detector's look ...
DET_WAIT_WALL_S = 1.0     # ... and the camera frame for Gemma this long (wall s) for a detection of that moment
DETECTOR_WHAT = {"spire": "a fallen spire", "boulder": "a boulder", "rock": "a rock"}


def detector_overrides(det: dict, gemma_class: str | None) -> bool:
    """True if a confident detector class (rock, boulder, spire) contradicts Gemma's label for the obstacle."""
    if det is None or det.get("conf", 0.0) < DET_OVERRIDE:
        return False
    if det["label"] in ("rock", "boulder"):
        return gemma_class != "rock"
    return True                   # spire: Gemma has no spire class, so any label of its own contradicts it
SAFEST = {k: v[0][0] for k, v in QUESTION_OPTIONS.items()}
DECISION_LABEL = {"tow": "tow it home", "salvage": "salvage parts", "leave": "leave it",
                  "reroute": "go around", "proceed_slowly": "approach slowly", "wait": "wait",
                  "back_out": "back out and go around", "follow": "keep following", "observe_here": "observe from here",
                  "keep_exploring": "keep exploring"}
OPTION_FIX = {("new_obstacle", "back_out"): "reroute", ("slip", "reroute"): "back_out",
              ("low_visibility", "reroute"): "proceed_slowly", ("low_visibility", "back_out"): "wait",
              ("no_returns", "back_out"): "reroute"}      # Gemma's action as one of the operator's options
LIFE_FROM_GEMMA = {"approach": "follow", "observe": "observe_here", "ignore": "keep_exploring"}
DECISION_CHAT_S = 30.0    # the same decision by the same rover is said in chat at most this often (sim s)
PRIORITY_CHECK_S = 2.0    # a fleet priority is checked (counts, reassignment, done) this often (sim s)
_WE_RE = re.compile(r"\b(?:we|us|our|ours|the fleet|everyone|everybody|y'?all)\b")
_MODEQ_RE = re.compile(r"\b(?:mode|autonom\w*|supervis\w*|priorit\w*|who decides)\b")
# an area question: "anything near the lake?", "what's at the vent?", "what's around rover 2?"
_AREA_PREP_RE = re.compile(r"\b(?:near|at|around|by|next to|close to|beside|over at|from|in|on)\s+(?:the\s+|that\s+|this\s+)?$")
_NEAR_ROVER_Q_RE = re.compile(r"\b(?:near|around|by|close to|next to|beside)\s+(?:rover\s*(\d+)|r(\d+)|(?:the\s+)?lander)\b")


def _dur(s: float) -> str:
    """'2 min' for 120 s, '45 s' for 45 s."""
    return f"{s / 60.0:.0f} min" if s >= 90.0 else f"{s:.0f} s"


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
        self.answers: list = []                      # questions waiting for Gemma's reply (or the budget)
        self._reply_to = None                        # operator message the replies being written answer
        self._request_id = None                      # request the replies being written answer (choose_place)
        self.n_answers = {"gemma": 0, "rules": 0}
        self.go_jobs: dict = {}                      # rover id -> operator go_to being planned
        self.after_sand: dict = {}                   # rover id -> wait (s) once it is out of the sand
        self.home_after: set = set()                 # rovers to send home once their recovery ends
        self.choices: dict = {}                      # rover id -> _Choice
        self.last_ask: dict = {}                     # rover id -> sim t of the last goal question
        self.last_agree: dict = {}
        self.last_question: dict = {}
        self.first_pending = bool(self.cfg.operator_ask_first)
        self.asked_herds: list = []                  # protocol v7: herds asked about {"ids", "x", "y", "t", "qid"}
        self.mem = ChatMemory()                      # protocol v7: what the chat talked about (references)
        self.place_jobs: list = []                   # protocol v8: Gemma deciding (or suggesting) between places
        self.said_decision: dict = {}                # (rover, kind, choice) -> sim t it was last said in chat
        self._prio_next = 0.0                        # next fleet priority check (sim s)
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

    def _gravity_kw(self) -> dict:
        """Gemma is told this world's gravity (v7). Only passed when it is not 2 g: at 2 g the call and the
        prompt stay exactly as before (tests wrap submit_choice with the old signature). Exactly 2 g only:
        a world at 2.04 g (GRAVITY_G=2.04) is told 2.04."""
        gv = float(self.cfg.gravity_g)
        return {} if gv == 2.0 else {"gravity_g": gv}

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

    def _log(self, t, level, msg, rover=None, at=None, source=None, decision=None):
        self.fleet.log(t, level, _clean(msg), rover, at=at, source=source, decision=decision)

    @property
    def supervised(self) -> bool:
        """Protocol v8: the operator is asked every decision (otherwise the rovers decide and log it)."""
        return self.fleet.mode == "supervised"

    # ================================================================== protocol
    def planner_msg(self) -> dict:
        s = self.planner.stats()
        st = {"ready": "ready", "warming": "warming", "starting": "warming", "disabled": "disabled"}.get(
            s["state"], "offline")
        return {"model": s["model"], "state": st, "online": st == "ready", "calls": s["calls"], "ok": s["ok"],
                "fallbacks": s["fallbacks"], "avg_ms": s["avg_ms"], "p95_ms": s["p95_ms"],
                "last_ms": s["last_ms"], "budget_ms": s["budget_ms"], "over_budget": s["over_budget"]}

    def say(self, frm: str, t: float, text: str, kind: str = "info", to="operator", source="rules",
            request_id=None, reply_to=None) -> str:
        """Add a chat message (from "operator", a rover id or "fleet"). Returns its id. reply_to: the
        operator message this answers (protocol v5)."""
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
                          "kind": kind, "source": source, "request_id": request_id, "reply_to": reply_to,
                          "_seq": self.chat_seq})
        if len(self.chat) > MAX_CHAT:
            del self.chat[:100]
        self._remember(frm, text[:400], kind, reply_to, request_id)
        return cid

    def _remember(self, frm: str, text: str, kind: str, reply_to, request_id) -> None:
        """Chat memory (protocol v7): the message, the rover talking with the operator, and the targets it
        named (one: "there"; several: a list for "the second one")."""
        self.mem.add_msg(frm, text)
        if frm == "operator":
            return
        if frm != "fleet" and (reply_to is not None or request_id is not None or kind in ("question", "reply", "ack")):
            self.mem.rover = frm
        refs = [self._target_ref(tid, frm if frm != "fleet" else None)
                for tid in dict.fromkeys(re.findall(r"\bt\d+\b", text.lower()))]
        refs = [r for r in refs if r is not None]
        if len(refs) == 1:
            self.mem.mention(refs[0])
        elif refs:
            self.mem.mention_list(refs)

    def _target_ref(self, tid: str, rover=None):
        ts = next((ts for ts in self.fleet.targets if ts.spec.id == tid and ts.status is not None), None)
        if ts is None:
            return None
        p = P.Place(ts.spec.id, ts.spec.kind, ts.spec.kind, float(ts.spec.x), float(ts.spec.y), t=ts.found_t,
                    standoff=2.5)
        return self._place_ref(p, rover)

    def _place_ref(self, p, rover=None, label: str | None = None) -> Ref:
        return Ref(p.kind, float(p.x), float(p.y), label or p.dest(), id=p.id, rover=rover, t=self.sim.t, place=p)

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
        """Open requests (protocol v3, v6): kind on every one, options may carry x, y, expires_in null when
        the request waits for the operator (never Infinity: the JSON must stay valid)."""
        now = self.clock()
        out = []
        for q in self.requests.values():
            opts = []
            for o in q["options"]:
                e = {"id": o[0], "label": o[1]}
                if len(o) >= 4:
                    e["x"], e["y"] = round(float(o[2]), 1), round(float(o[3]), 1)
                if q.get("suggest") == o[0]:
                    e["suggested"] = True            # protocol v8: Gemma's pick
                mt = (q.get("meta") or {}).get(o[0])
                if mt is not None:                   # protocol v9: the rules' verdict on a rescue option
                    e["safe"], e["note"] = bool(mt["safe"]), mt["note"]
                opts.append(e)
            dl = q.get("deadline")
            e = {"id": q["id"], "rover": q["rover"], "kind": REQUEST_KIND.get(q["kind"], "anomaly"), "t": q["t"],
                 "question": q["question"], "options": opts, "default": q["default"],
                 "expires_in": None if dl is None else round(max(0.0, dl - now), 1), "image": q["image"]}
            if q.get("about"):
                e["about"] = q["about"]              # protocol v9: the rover a rescue request is about
            out.append(e)
        return out

    def _deadline(self):
        """Wall clock deadline for a new request, or None (OPERATOR_TIMEOUT_S 0: wait for the operator)."""
        secs = float(self.cfg.operator_timeout_s or 0.0)
        return self.clock() + secs if secs > 0.0 and math.isfinite(secs) else None

    @staticmethod
    def _labels(q: dict) -> dict:
        return {o[0]: o[1] for o in q["options"]}

    def fleet_context(self) -> dict:
        half = (self.cfg.nx - 1) * self.cfg.res / 2.0
        return {"rovers": [{"id": r.id, "name": r.name, "color": r.color, "x": round(r.x, 1), "y": round(r.y, 1),
                            "state": r.state, "battery": round(r.battery, 2)} for r in self.sim.rovers],
                "targets": self.fleet.discovered_targets(), "lander": [0.0, 0.0],
                "bounds": [-half, -half, half, half], "chat": self.mem.context()}

    # ================================================================== per cycle
    def update(self, t: float) -> None:
        """Sim thread, after every rover's decision cycle: operator commands, Gemma's anomaly
        answers and question timeouts. Never blocks."""
        try:
            if self.commands:
                self._poll_commands(t)
            if self.answers:
                self._poll_answers(t)
            if self._done:
                self._poll_interpretations(t)
            if self.place_jobs:
                self._poll_place_jobs(t)
            if self.fleet.priority is not None and t >= self._prio_next:
                self._prio_next = t + PRIORITY_CHECK_S
                self._priority_tick(t)
            if self.requests:
                now = self.clock()
                for q in [q for q in self.requests.values() if q.get("deadline") is not None and now >= q["deadline"]]:
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
    @staticmethod
    def _goal_why(c: dict, base: dict | None) -> str:
        """Why Gemma's pick beats the rules' pick, from the real numbers only, so every comparison the
        log shows is true. Empty when there is no measurable advantage."""
        if not base or c.get("kind") != "frontier":
            return ""
        bits = []
        sz, bsz = int(c.get("size") or 0), int(base.get("size") or 0)
        if bsz and sz >= bsz * 1.25 and sz - bsz >= 3:
            bits.append(f"more unknown ground at its edge ({sz} vs {bsz} cells)")
        pl, bpl = c.get("plen"), base.get("plen")
        if pl is not None and bpl is not None and bpl > 0 and pl <= bpl * 0.8 and bpl - pl >= 3:
            bits.append(f"a shorter drive ({pl:.0f} vs {bpl:.0f} m)")
        sm, bsm = c.get("smax"), base.get("smax")
        if sm is not None and bsm is not None and bsm - sm >= 3:
            bits.append(f"a gentler route (steepest {sm:.0f} vs {bsm:.0f} deg)")
        return ", ".join(bits)

    def on_goal_chosen(self, rover, t: float, goal, path, size: int, reached: bool) -> None:
        """The rules just picked `goal` (the rover drives there now). Maybe ask Gemma."""
        if not self.planner.online or rover.id in self.choices or rover.state != "exploring":
            return
        if rover.op_goal is not None or rover.held or (rover.dir_bias is not None and t < rover.dir_bias[2]):
            return                         # the operator's wishes come before Gemma's
        if rover.focus is not None and rover.focus["reason"] == "operator":
            return                         # ... and so does an area the operator asked it to explore
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
        job.ticket = self.planner.submit_choice(rover.id, t, perception, cands, default="f1", **self._gravity_kw())
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
            base = next((b for b in job.cands if b["id"] == "f1"), None)
            adv = self._goal_why(c, base)
            if not adv:                    # no measurable advantage: state the facts only, never a false comparison
                pl, sm = c.get("plen"), c.get("smax")
                adv = (f"Gemma's pick: {int(c.get('size') or 0)} cells of unknown edge"
                       + (f", {pl:.0f} m drive" if pl is not None else "")
                       + (f", steepest {sm:.0f} deg" if sm is not None else ""))
            rover.goal, rover.goal_size, rover.goal_t = (gx, gy), c["size"], t
            rover._set_path(t, c["path"])
            rover.plan_t = -1e9            # re-plan from where the rover is now (the rules' A*)
            rover.sticky_goal = (gx, gy)
            rover.next_goal_eval = t + 30.0
            rover.mem.on_goal(t, "gemma", adv)
            self._log(t, "info", f"{rover.id} switching to {c['label']} (Gemma, {lat:.1f} s): {adv}.", rover.id,
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
            with self._mu:                 # Gemma offline or warming: the rules decide (v8) or ask (supervised)
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
            if a.get("camera_water") is not None:          # protocol v8: the onboard detector saw water there
                out["camera_water"] = a["camera_water"]
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
        if job.get("type") != "describe" and not self.wait_for_gemma:
            # the onboard detector's result for the same moment goes with the frame: wait briefly for a
            # detection made at or after the anomaly (the camera thread runs the detector)
            vb = getattr(self._rover(job["rover"]), "vision", None)
            end = time.monotonic() + DET_WAIT_WALL_S
            while (vb is not None and vb.latest is not None and vb.latest.t < job["t"] - 0.1
                   and time.monotonic() < end and not self._stop.is_set()):
                time.sleep(0.05)
        self._camera_evidence(job)
        if job.get("type") == "describe":
            kw = self._gravity_kw()
            if getattr(self.cfg, "life_variety", False):      # v7: the label may name a kind it sees
                from .gemma import KIND_LOOKS
                kw["kinds"] = tuple(KIND_LOOKS)
            tk = self.planner.submit_description(job["rover"], job["t"], job["perception"], job["obs"], jpeg, **kw)
        else:
            tk = self.planner.submit_interpretation(job["rover"], job["t"], job["perception"], job["anomaly"], jpeg,
                                                    INTERP_OPTIONS, default=RULE_REACTION[job["kind"]],
                                                    **self._gravity_kw())
        if self.wait_for_gemma:
            self._wait(tk)
        with self._mu:
            self._done.append((job, tk))

    def _camera_evidence(self, job: dict) -> None:
        """Protocol v8: the onboard detector's objects near the anomaly (or the moving track) go to Gemma as an
        extra reading, e.g. "boulder 0.78 at 4.2 m". Worker thread: only reads the rover's latest result."""
        rover = self._rover(job["rover"])
        vb = getattr(rover, "vision", None)
        if vb is None:
            return
        try:
            seen = vb.describe_near(job.get("at"), self.sim.t)
        except Exception:                  # noqa: BLE001 (evidence is optional)
            seen = None
        if seen:
            (job["obs"] if job.get("type") == "describe" else job["anomaly"])["camera_sees"] = seen
            if "facts" in job:
                job["facts"]["camera_sees"] = seen
        if job.get("type") != "describe":
            det = self._detector_for(job, self.sim.t)
            if det is not None:
                job["detector"] = det
                job["anomaly"]["camera_detector"] = f"{det['label']} {det['conf']:.2f}"

    def _detector_for(self, job: dict, t: float):
        """The onboard detector's strongest obstacle class (rock, boulder, spire) at this anomaly, or None:
        the rover's own camera rule result (rover.anomaly["camera"]) or a fresh detection near the place."""
        rover = self._rover(job["rover"])
        if rover is None or job.get("kind") != "new_obstacle":
            return None
        a = getattr(rover, "anomaly", None)
        if isinstance(a, dict) and a.get("t") == job["t"] and isinstance(a.get("camera"), dict):
            return {"label": a["camera"]["label"], "conf": float(a["camera"]["conf"])}
        vb = getattr(rover, "vision", None)
        res = vb.fresh(t) if vb is not None else None
        if res is None or res.t < job["t"] - 0.5 or job.get("at") is None:
            return None
        ax, ay = job["at"]
        cands = [o for o in res.objects if o["label"] in ("rock", "boulder", "spire")
                 and math.hypot(o["x"] - ax, o["y"] - ay) < 3.0]
        if not cands:
            return None
        o = max(cands, key=lambda c: c["conf"])
        return {"label": o["label"], "conf": float(o["conf"])}

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
            if job.get("kind") == "new_obstacle" and job.get("type") != "describe" and t - job["t"] < DET_WAIT_S:
                vb = getattr(self._rover(job["rover"]), "vision", None)
                if (vb is not None and vb.latest is not None and vb.latest.t < job["t"] - 0.1
                        and self._detector_for(job, t) is None):
                    continue               # the detector has not seen this moment yet: its class comes first
            with self._mu:
                self._done.remove(item)
            if job.get("type") == "describe":
                self._on_description(job, res, t)
            else:
                self._on_interpretation(job, res, t)

    def _on_interpretation(self, job: dict, res, t: float) -> None:
        """Gemma's look at an anomaly (or None: Gemma offline or no answer). Supervised (protocol v8): every anomaly
        becomes a question with Gemma's pick as the suggestion. Autonomous: the rover decides at once (Gemma, the
        rules when Gemma is offline, the safest option below 0.5 confidence) and logs the decision."""
        rover = self._rover(job["rover"])
        if rover is None:
            return
        kind, rid = job["kind"], rover.id
        ok = res is not None and res.ok
        ask = self.supervised and self._can_ask(rover, t, True, every=True)
        pending = [q for q in self.requests.values() if q["rover"] == rid] if self.supervised and not ask else []
        if pending:
            # supervised: the operator is already deciding for this rover (often the same rockfall seen again
            # from where it holds). Do not decide behind their back; the safety rules keep it safe meanwhile.
            self._log(t, "info", f"{rid} also sees: {self._anomaly_text(kind, job['facts'])}. Still waiting for your "
                                 f"answer to its open question; the safety rules keep it safe meanwhile.", rid,
                      at=job["at"])
            return
        choice = by = conf = why = None
        phrase = mode = None
        if not ask:
            choice, by, conf, why = self._anomaly_choice(kind, res if ok else None)
            if by == "gemma" or choice != RULE_REACTION[kind]:
                phrase, mode = self._apply_action(rover, kind, choice, t, job)
                if mode == "override":              # safety rule wins over Gemma's pick
                    why = f"Gemma said {DECISION_LABEL.get(choice, choice)} ({conf:.2f}); safety rule wins: {phrase}"
                    choice, by = RULE_REACTION[kind], "safety rule"
        if ok:
            job["gemma_class"] = res.classification
            label = res.classification.replace("_", " ")
            src = "camera and sensors" if job["image"] else "sensors"
            det = self._detector_for(job, t) or job.get("detector")
            if det is not None:
                job["facts"]["camera_detector"] = f"{det['label']} {det['conf']:.2f}"
            if det is not None and detector_overrides(det, res.classification):
                # a confident onboard detection of a different class wins over Gemma's label
                what = DETECTOR_WHAT.get(det["label"], det["label"])
                job["gemma_class"] = "rock"
                head = (f"{rid} Gemma ({src}, {res.latency_ms / 1000.0:.1f} s) said {label} "
                        f"({res.confidence:.2f}). Camera detector: {det['label']} {det['conf']:.2f}, so {what}.")
            else:
                head = (f"{rid} Gemma ({src}, {res.latency_ms / 1000.0:.1f} s): "
                        f"{'maybe ' if res.action == 'ask_operator' else ''}{label}, confidence {res.confidence:.2f}. "
                        f"{_quote(res.reason)}")
            if ask:
                tail = "Asking the operator; holding position meanwhile."
            elif phrase is None:
                tail = "Keeping the rule reaction."
            else:
                tail = {"agree": "Agrees with the rules: ", "override": "Safety rule wins: ",
                        "noted": ""}.get(mode, "Doing it: ")
                tail = (tail + phrase + ".") if tail else phrase[0].upper() + phrase[1:] + "."
            self._log(t, "info", f"{head} {tail}", rid, at=job["at"], source="gemma")
        elif res is not None and res.error and not res.error.startswith(("gemma ", "cancelled", "superseded")):
            self._log(t, "info", f"{rid} no usable answer from Gemma ({res.error.split(':')[0]}). "
                                 f"Keeping the rule reaction.", rid, at=job["at"])
        if ask:
            self._open_request(rover, kind, job, res if ok else None, t, tried=res is not None)
            return
        busy = " (the operator has another question open)" if self.supervised else ""
        self._decided(rover, kind, self._anomaly_text(kind, job["facts"]), [o for o, _ in QUESTION_OPTIONS[kind]],
                      choice, by, conf, (why or "") + busy, job["at"], t)

    def _anomaly_choice(self, kind: str, res) -> tuple:
        """(choice, by, confidence, why) for an anomaly the rover decides itself (protocol v8)."""
        if res is None:
            return RULE_REACTION[kind], "rules", None, "Gemma offline or no answer: the rules' reaction"
        act = res.action if res.action != "ask_operator" else (res.model_action or "ask_operator")
        if res.confidence < CONF_FLOOR or act == "ask_operator":
            return SAFEST[kind], "rules", res.confidence, (f"Gemma unsure ({res.confidence:.2f}, maybe "
                                                           f"{res.classification.replace('_', ' ')}): the safest option")
        return act, "gemma", res.confidence, _clean(res.reason).strip("'\" ")

    @staticmethod
    def _anomaly_text(kind: str, a: dict) -> str:
        """The decision's question in a few words, with the numbers ("New rocks 4 m ahead, 0.6 m tall")."""
        if kind == "new_obstacle":
            if a.get("shape") == "long":
                return f"A long, low obstacle {a.get('dist', 0.0):.0f} m ahead, {a.get('length', 0.0):.1f} m long"
            return f"New rocks {a.get('dist', 0.0):.0f} m ahead, {a.get('height', 0.0):.1f} m tall"
        if kind == "slip":
            return f"Wheel slip {a.get('slip', 0.0) * 100:.0f}%, ground speed {a.get('ground_speed', 0.0):.2f} m/s"
        if kind == "no_returns":
            cam = f", camera {a['camera_liquid_pct']:.0f}% blue" if a.get("camera_liquid_pct") is not None else ""
            return f"No LiDAR returns {a.get('dist', 0.0):.0f} m ahead{cam}"
        return f"LiDAR range down to {a.get('lidar_range', 0.0):.0f} m"

    def _decided(self, rover, kind: str, question: str, options: list, choice: str, by: str, conf, why: str, at,
                 t: float, label: str | None = None, chat: bool = True, request_id=None, reply_to=None,
                 force: bool = False) -> None:
        """Protocol v8: a decision the rover made instead of asking: an event with a `decision` object, readable
        on its own ("r2 decided (Gemma 0.82): go around. New rocks 4 m ahead, 0.6 m tall."), and a chat line (the
        same decision by the same rover at most once in DECISION_CHAT_S)."""
        rid = rover.id
        label = label or DECISION_LABEL.get(choice, choice.replace("_", " "))
        opts = list(options) + ([choice] if choice not in options else [])
        dec = {"question": question, "options": opts, "choice": choice, "by": "gemma" if by == "gemma" else "rules",
               "confidence": None if conf is None else round(float(conf), 2), "why": _clean(why)[:160]}
        who = f"Gemma {conf:.2f}" if by == "gemma" else by
        text = f"decided ({who}): {label}. {question}."
        self._log(t, "info", f"{rid} {text}", rid, at=at, source="gemma" if by == "gemma" else None, decision=dec)
        key = (rid, kind, choice)
        if chat and (force or t - self.said_decision.get(key, -1e9) >= DECISION_CHAT_S):
            self.said_decision[key] = t
            self.say(rid, t, text[:1].upper() + text[1:], "info", source="gemma" if by == "gemma" else "rules",
                     request_id=request_id, reply_to=reply_to)

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
    def _can_ask(self, rover, t: float, first: bool, every: bool = False) -> bool:
        """every (supervised, protocol v8): no cooldown, and up to one open question per rover."""
        if rover.state in PARKED or any(q["rover"] == rover.id for q in self.requests.values()):
            return False
        cap = max(self.cfg.max_open_questions, len(self.sim.rovers)) if every else self.cfg.max_open_questions
        if sum(1 for q in self.requests.values() if q["kind"] != "choose_place") >= cap:
            return False
        return first or every or t - self.last_question.get(rover.id, -1e9) >= self.cfg.question_cooldown_s

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
        if a.get("camera_sees") and kind in ("new_obstacle", "no_returns"):
            q += f" Camera: {a['camera_sees']}."
        if res is not None:
            q += f" Gemma thinks: {res.classification.replace('_', ' ')} (confidence {res.confidence:.2f})."
            sug = self._suggest(kind, res)
            if sug is not None:
                q += f" Gemma suggests: {self._labels_of(kind)[sug].lower()} ({res.confidence:.2f})."
        elif tried:
            q += " Gemma could not tell."
        return q + " What should I do?"

    @staticmethod
    def _labels_of(kind: str) -> dict:
        return {o: lab for o, lab in QUESTION_OPTIONS[kind]}

    @staticmethod
    def _suggest(kind: str, res):
        """Gemma's pick as one of the operator's options (protocol v8 suggestion), or None."""
        if res is None:
            return None
        act = res.action if res.action != "ask_operator" else res.model_action
        if res.confidence < CONF_FLOOR or act in (None, "ask_operator"):
            return SAFEST[kind]
        act = OPTION_FIX.get((kind, act), act)
        return act if act in dict(QUESTION_OPTIONS[kind]) else None

    def _open_request(self, rover, kind: str, job: dict, res, t: float, tried: bool) -> None:
        self.req_seq += 1
        qid = f"q{self.req_seq}"
        opts = QUESTION_OPTIONS[kind]
        job["res"] = res                  # Gemma's look, for a decision if the mode switches to autonomous
        q = {"id": qid, "rover": rover.id, "t": round(t, 1), "question": self._question(rover, kind, job, res, tried),
             "options": opts, "default": opts[0][0], "deadline": self._deadline(),
             "image": f"/api/camera/{rover.id}.jpg?view=fpv", "kind": kind, "job": job,
             "suggest": self._suggest(kind, res)}
        self.requests[qid] = q
        self.last_question[rover.id] = t
        rover.question_hold = True        # stay put until the operator answers (safety moves still allowed)
        src = "gemma" if res is not None else "rules"
        self.say(rover.id, t, f"{rover.id}: {q['question']}", "question", source=src, request_id=qid)
        if res is None:
            why = "first anomaly since landing" if job["first"] else "not sure what it sees"
            self._log(t, "warn", f"{rover.id} {ANOMALY_NAME[kind]}: {why}, asking the operator. Holding position "
                                 f"meanwhile.", rover.id, at=job["at"])

    def _close(self, qid: str, status: str, t: float):
        q = self.requests.pop(qid)
        self.closed[qid] = status
        while len(self.closed) > 200:
            self.closed.popitem(last=False)
        rover = self._rover(q["rover"])
        if rover is not None:
            rover.crawl_until = t
            # a lifeform question never holds the rover: it follows the herd meanwhile (protocol v7)
            rover.question_hold = any(o["rover"] == rover.id and o["kind"] != "lifeform"
                                      for o in self.requests.values())
            if rover.obs is not None and rover.obs.get("asking") == qid:
                rover.obs["asking"] = None
        return q, rover

    def on_battery_return(self, rover, t: float) -> None:
        """Rover hook (protocol v7): its battery says go home now. Whatever it was waiting on cannot wait: each
        open question closes (superseded) with a chat line saying why, and a trip still being planned drops."""
        b = rover.battery * 100.0
        for q in [q for q in self.requests.values() if q["rover"] == rover.id]:
            self._close(q["id"], "superseded", t)
            if q["kind"] == "choose_place":
                tail = "Send the command again once I've charged."
            else:
                tail = "I'll ask again if it still matters."
                if q["kind"] == "lifeform":       # that herd may be asked about again on a later sighting
                    self.asked_herds = [h for h in self.asked_herds if h.get("qid") != q["id"]]
            if q["kind"] == "rescue":
                tail = "Another rover will take over the rescue."
            self.say(rover.id, t, f"Battery {b:.0f}%, I can't wait any longer: heading back to the lander. {tail}",
                     "info", request_id=q["id"], reply_to=q["job"].get("cid") if q["kind"] == "choose_place" else None)
            self._log(t, "info", f"{rover.id} closed its question {q['id']}: battery {b:.0f}%, returning to the "
                                 f"lander.", rover.id, at=(rover.x, rover.y))
        if rover.id in self.go_jobs:
            self._cancel_go(rover.id, t, f"battery {b:.0f}%, returning to the lander")

    def on_out_of_action(self, rover, t: float) -> None:
        """Protocol v9: the rover can no longer drive. Its open questions close (it can't act on them) with a chat line,
        and a trip still being planned drops."""
        for q in [q for q in self.requests.values() if q["rover"] == rover.id]:
            self._close(q["id"], "superseded", t)
            if q["kind"] == "lifeform":
                self.asked_herds = [h for h in self.asked_herds if h.get("qid") != q["id"]]
            self.say(rover.id, t, f"I can't drive any more ({rover.health.short()}), so I can't act on this question.",
                     "info", request_id=q["id"], reply_to=q["job"].get("cid") if q["kind"] == "choose_place" else None)
        self._cancel_go(rover.id, t, "I can't drive any more")
        self.choices.pop(rover.id, None)

    def open_rescue_request(self, helper, rec: dict, opts: list, default: str, text: str, suggest, conf,
                            t: float, meta: dict | None = None) -> str:
        """Protocol v9 (supervised): ask the operator what to do with a rover that can't drive (tow, salvage, leave).
        Every option carries the rules' verdict (safe, note); an unsafe one cannot be picked. The helper waits by
        the rover."""
        self.req_seq += 1
        qid = f"q{self.req_seq}"
        at = (rec["at"][0], rec["at"][1])
        q = {"id": qid, "rover": helper.id, "t": round(t, 1), "question": text, "options": opts, "default": default,
             "deadline": self._deadline(), "image": f"/api/camera/{helper.id}.jpg?view=fpv", "kind": "rescue",
             "job": {"at": at, "rescue": rec["rover"], "cid": rec.get("cid")}, "suggest": suggest,
             "suggest_conf": conf, "meta": meta or {}, "about": rec["rover"]}
        self.requests[qid] = q
        self.last_question[helper.id] = t
        self.say(helper.id, t, text, "question", source="gemma" if suggest is not None else "rules", request_id=qid)
        return qid

    def _apply_rescue(self, q: dict, option: str, t: float, by: str = "operator") -> str:
        resc = getattr(self.sim, "rescue", None)
        rec = resc.active(q["job"]["rescue"]) if resc is not None else None
        if rec is None:
            return "that rescue is over"
        rec["qid"] = None
        why = resc.apply_plan(rec, option, t, by=by)
        return f"can't: {why}" if why else DECISION_LABEL.get(option, option)

    def answer(self, qid: str, option: str) -> str:
        """POST /api/answer. Returns "ok", "unknown", "closed" or "bad_option"."""
        return self._answer(qid, option, echo=True)

    def _answer(self, qid: str, option: str, echo: bool = True) -> str:
        """Apply the operator's answer. echo: post the chosen option's label as the operator's chat line
        (an answer typed in the chat is already there)."""
        q = self.requests.get(qid)
        if q is None:
            return "closed" if qid in self.closed else "unknown"
        labels = self._labels(q)
        if option not in labels:
            return "bad_option"
        if q["kind"] == "rescue" and not (q.get("meta") or {}).get(option, {}).get("safe", True):
            return "unsafe"                          # protocol v9: the rules marked it unsafe (the request stays open)
        t = self.sim.t
        q, rover = self._close(qid, "answered", t)
        rid = q["rover"]
        if echo:
            self.say("operator", t, labels[option], "answer", to=rid, source=None, request_id=qid)
        if q["kind"] == "choose_place":
            at = next(((o[2], o[3]) for o in q["options"] if o[0] == option and len(o) >= 4), q["job"]["at"])
            self._log(t, "info", f"{rid} operator chose '{labels[option]}'.", rid, at=at, source="operator")
            self._apply_place_choice(rover, q, option, t)
            return "ok"
        if q["kind"] == "rescue":
            phrase = self._apply_rescue(q, option, t)
            self._log(t, "info", f"{rid} operator chose '{labels[option]}' for {q['job']['rescue']}: {phrase}.", rid,
                      at=q["job"]["at"], source="operator")
            return "ok"
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
        if q.get("default") is None:       # no automatic choice: it waits for the operator
            q["deadline"] = None
            return
        q, rover = self._close(q["id"], "expired", t)
        rid, label = q["rover"], self._labels(q)[q["default"]]
        if q["kind"] == "choose_place":
            self._apply_place_choice(rover, q, q["default"], t, quiet=True)
        elif q["kind"] == "rescue":
            self._apply_rescue(q, q["default"], t, by="rules")
        elif rover is not None:
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
        """A rover started observing a moving track. Hand Gemma the camera frame and the numbers. A herd
        nobody asked about yet (protocol v7): the rover follows it at the standoff at once and asks the operator
        in the same moment (keep following, observe from here, keep exploring); Gemma's look only adds a label.
        A herd already asked about: Gemma's answer decides follow / observe from here / stop. Returns True if
        tier 2 will decide, False to leave it to the rules (tier 2 switched off)."""
        if not self.active:
            return False
        px, py = tr.predict(t)
        d = math.hypot(px - rover.x, py - rover.y)
        bearing = math.degrees(math.atan2(math.sin(math.atan2(py - rover.y, px - rover.x) - rover.yaw),
                                          math.cos(math.atan2(py - rover.y, px - rover.x) - rover.yaw)))
        obs = {"distance": round(d, 1), "speed": round(tr.speed, 2), "size": round(tr.size, 1),
               "bearing_deg": round(bearing, 0), "seen_for_s": round(t - tr.t0, 1)}
        job = {"type": "describe", "rover": rover.id, "track": tr.id, "t": float(t), "at": (px, py), "first": False,
               "obs": obs, "perception": self._perception(rover), "wall": self.clock(), "image": False, "qid": None}
        rec = self._herd_record(tr, px, py)
        if self.supervised and self._can_ask(rover, t, True, every=True):   # v8: every sighting asks
            if rec is None:
                rec = {"ids": set(rover.obs["seen"]) | {tr.id}, "x": px, "y": py, "t": t, "qid": None}
                self.asked_herds.append(rec)
            rec["qid"] = job["qid"] = self._open_life_request(rover, tr, job, None, t)
            rover.follow_while_asking(t, job["qid"])
        if rec is not None and rover.obs is not None:
            rover.obs["herd"] = rec
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

    def _herd_record(self, tr, px: float, py: float):
        """The herd already asked about that this track belongs to (same track id, a track merged into one of
        its ids, or a re-acquisition near where it was last seen), or None."""
        ids = {tr.id}
        for new, old in self.fleet.tracker.merged.items():
            if old == tr.id:
                ids.add(new)
        for h in self.asked_herds:
            if ids & h["ids"] or math.hypot(px - h["x"], py - h["y"]) < ASKED_HERD_M:
                return h
        return None

    def _on_description(self, job: dict, res, t: float) -> None:
        rover = self._rover(job["rover"])
        f = self.fleet
        tr = f.tracker.get(job["track"])         # follows a merge into an older track
        if rover is None or rover.obs is None or tr is None or rover.obs["track"] != tr.id:
            return                               # the observation already ended
        ok = res is not None and res.ok
        unsure = ok and (res.action == "ask_operator" or res.confidence < self.cfg.operator_ask_below)
        sd = self.cfg.observe_standoff
        if job.get("qid") is not None:           # following while the operator decides: Gemma only describes it
            if ok:
                tr.label = res.label
                life = res.classification == "lifeform"
                if life and not unsure:
                    rover.obs["life"] = True
                    f.confirm_lifeform(tr, t)
                q = self.requests.get(job["qid"])
                if q is not None:                # the open question now carries Gemma's guess and suggestion
                    q["job"]["res"] = res
                    q["suggest"] = self._life_choice(res)[0]
                    q["question"] = self._life_question(q["job"]["ask"], res, q["suggest"])
                src = "camera" if job["image"] else "track numbers"
                f.log(t, "discovery" if life else "info",
                      f"{rover.id} Gemma ({src}, {res.latency_ms / 1000.0:.1f} s): {res.label} (confidence "
                      f"{res.confidence:.2f}). {_quote(res.reason)} Following at about {sd:.0f} m while the operator "
                      f"decides.", rover.id, at=job["at"], source="gemma")
            elif res is not None and res.error and not res.error.startswith(("gemma ", "cancelled", "superseded")):
                f.log(t, "info", f"{rover.id} no usable answer from Gemma about track {tr.id} "
                                 f"({res.error.split(':')[0]}). Following while the operator decides.", rover.id,
                      at=job["at"])
            return
        # nobody asks (autonomous, protocol v8; or no room for another question): the rover decides and logs it
        choice, by, conf, why = self._life_choice(res if ok else None)
        if ok:
            tr.label = res.label
            life = res.classification == "lifeform"
            if life and not unsure:
                rover.obs["life"] = True
                f.confirm_lifeform(tr, t)
            src = "camera" if job["image"] else "track numbers"
            head = (f"{rover.id} Gemma ({src}, {res.latency_ms / 1000.0:.1f} s): {res.label} "
                    f"(confidence {res.confidence:.2f}). {_quote(res.reason)}")
            tail = {"keep_exploring": "Not a lifeform. Resuming exploration.", "observe_here": "Watching from here.",
                    "follow": f"Approaching to {sd:.0f} m to observe."}[choice]
            f.log(t, "discovery" if life else "info", f"{head} {tail}", rover.id, at=job["at"], source="gemma")
        elif res is not None and res.error and not res.error.startswith(("gemma ", "cancelled", "superseded")):
            f.log(t, "info", f"{rover.id} no usable answer from Gemma about track {tr.id} "
                             f"({res.error.split(':')[0]}). The rules decide.", rover.id, at=job["at"])
        rover.set_obs_mode(t, {"keep_exploring": "stop", "observe_here": "here", "follow": "follow"}[choice])
        busy = " (the operator has another question open)" if self.supervised else ""
        o = job["obs"]
        what = (f"Possible lifeform {o['distance']:.0f} m away, {o['speed']:.1f} m/s, about {o['size']:.1f} m"
                + (f" ({res.label})" if ok and res.label else ""))
        self._decided(rover, "lifeform", what, [k for k, _ in LIFE_OPTIONS], choice, by, conf, (why or "") + busy,
                      job["at"], t)

    def _life_choice(self, res) -> tuple:
        """(option, by, confidence, why) for a lifeform sighting the rover decides itself (protocol v8): Gemma's
        action, the rules (follow carefully) when Gemma is offline, observe from here below 0.5 confidence."""
        if res is None or not res.ok:
            return "follow", "rules", None, "Gemma offline or no answer: the rules follow carefully at the standoff"
        act = res.action if res.action != "ask_operator" else res.model_action
        if res.confidence < CONF_FLOOR or act in (None, "ask_operator"):
            return "observe_here", "rules", res.confidence, f"Gemma unsure ({res.confidence:.2f}): the safest option"
        if act == "ignore" and res.classification == "lifeform":
            act = "approach"
        return LIFE_FROM_GEMMA.get(act, "observe_here"), "gemma", res.confidence, _clean(res.reason).strip("'\" ")

    def _life_question(self, a: dict, res=None, suggest: str | None = None) -> str:
        """The lifeform question (protocol v7): what was seen, Gemma's guess and suggestion (v8) once it has looked,
        and that the rover follows at the standoff while it waits."""
        guess = f" Gemma thinks: {res.label} (confidence {res.confidence:.2f})." if res is not None else ""
        if res is not None and suggest:
            guess += f" Gemma suggests: {dict(LIFE_OPTIONS)[suggest].lower()} ({res.confidence:.2f})."
        return (f"Possible lifeform {a['d']:.0f} m {a['dir']} (LiDAR track {a['track']}, {a['speed']:.1f} m/s, about "
                f"{a['size']:.1f} m).{guess} I'm following at about {self.cfg.observe_standoff:.0f} m (never closer "
                f"than {self.cfg.observe_min_dist:.0f} m) while I wait, in case it wanders off. Keep following, "
                f"observe from here, or keep exploring?")

    def _open_life_request(self, rover, tr, job: dict, res, t: float) -> str:
        """A lifeform question. It never holds the rover (no question_hold): the rover follows the herd at the
        standoff until the answer comes (Rover.follow_while_asking). Returns the request id."""
        self.req_seq += 1
        qid = f"q{self.req_seq}"
        px, py = tr.predict(t)
        d = math.hypot(px - rover.x, py - rover.y)
        job["ask"] = {"d": d, "dir": compass(px - rover.x, py - rover.y), "track": tr.id, "speed": tr.speed,
                      "size": tr.size}
        question = self._life_question(job["ask"], res)
        q = {"id": qid, "rover": rover.id, "t": round(t, 1), "question": question, "options": LIFE_OPTIONS,
             "default": LIFE_DEFAULT, "deadline": self._deadline(),
             "image": f"/api/camera/{rover.id}.jpg?view=fpv", "kind": "lifeform", "job": job}
        self.requests[qid] = q
        self.last_question[rover.id] = t
        self.say(rover.id, t, f"{rover.id}: {question}", "question", source="gemma" if res is not None else "rules",
                 request_id=qid)
        sd = self.cfg.observe_standoff
        herd = P.Place(tr.id, "lifeform", "herd", px, py, size=1, t=t, standoff=sd, track=tr.id, live=True)
        self.mem.mention(self._place_ref(herd, rover.id, "the herd"))
        self.fleet.log(t, "info", f"{rover.id} new herd (track {tr.id}): following at about {sd:.0f} m and asking "
                                  f"the operator.", rover.id, at=(px, py))
        return qid

    def _apply_life(self, rover, option: str, t: float, q: dict) -> str:
        job = q["job"]
        tr = self.fleet.tracker.get(job["track"])
        if option == "keep_exploring":       # leave this herd alone, also when it is seen again (fleet)
            px, py = tr.predict(t) if tr is not None else job["at"]
            self.fleet.dismissed_herds.append((float(px), float(py), t))
        if rover.obs is None or tr is None or rover.obs["track"] != tr.id:
            return "the observation already ended"
        o = rover.obs
        o["asking"] = None
        o["t_lim"] = t                       # the observation time limit counts from the answer
        sd = self.cfg.observe_standoff
        if option == "follow":
            o["x0"], o["y0"] = rover.x, rover.y      # and the leash from here
            rover.set_obs_mode(t, "follow")
            return f"following it at about {sd:.0f} m, never closer than {self.cfg.observe_min_dist:.0f} m"
        if option == "keep_exploring":
            for k in o["seen"]:
                tr = self.fleet.tracker.get(k)
                if tr is not None:
                    tr.ignored = True
            rover.set_obs_mode(t, "stop")
            return "back to exploring, leaving it alone"
        rover.set_obs_mode(t, "here")
        return "watching from here, I will not get closer"

    def on_follow_ended(self, rover, qid: str, why: str, t: float, o: dict) -> None:
        """Rover hook (protocol v7): a follow while the operator decides ended before the answer (safety wins:
        hazard, water, lost it, motors, a hold). The herd record moves to where it was last seen; the open
        question closes with a chat line saying why. A low battery closes it in on_battery_return instead."""
        rec = o.get("herd")
        tr = self.fleet.tracker.get(o["track"])
        if rec is not None:
            rec["ids"] |= set(o["seen"])
            if tr is not None:
                rec["x"], rec["y"] = tr.predict(t)
        q = self.requests.get(qid)
        if q is None or why == "battery":
            return
        self._close(qid, "superseded", t)
        reason = {"hazard": o.get("hazard", "a hazard"), "water": "my water sensor got wet",
                  "lost": "I lost sight of it", "leash": f"it led me {self.cfg.observe_leash:.0f} m from where I first "
                                                         f"saw it", "not_life": "it is not a lifeform"}.get(why)
        if reason is None:                   # interrupted: what the rover is doing now
            reason = {"fault": "my motor controller dropped out", "idle": "I'm holding for the operator",
                      "returning": "I'm returning to the lander", "recovering": "I'm backing out of trouble"}.get(
                rover.state, "something else came up")
        self.say(rover.id, t, f"Stopped following the herd: {reason}. Closing my question; I'm keeping my distance.",
                 "info", request_id=qid)
        self._log(t, "info", f"{rover.id} stopped following track {o['track']} ({reason}) and closed its question "
                             f"{qid}.", rover.id, at=(rover.x, rover.y))

    # ================================================================== 4. operator chat
    def operator_message(self, text: str, rover_hint: str | None = None) -> str:
        """POST /api/chat: post the message now, interpret it in the background. Returns the chat id."""
        text = " ".join(str(text).split())[:300]
        t = self.sim.t
        ctx = self.fleet_context()
        picked = self._chat_pick(text, rover_hint, ctx)
        if picked is not None:             # "the second one": answers the open choose_place request
            q, option = picked
            cid = self.say("operator", t, text, "answer", to=q["rover"], source=None, request_id=q["id"])
            self._answer(q["id"], option, echo=False)
            return cid
        preview = parse_command(text, ctx)["command"].get("rover")
        cid = self.say("operator", t, text, "command", to=rover_hint or preview or None, source=None)
        tk = self.planner.submit_command(t, text, ctx, **self._gravity_kw())
        self.commands.append({"ticket": tk, "cid": cid, "hint": rover_hint, "t": t, "wall": self.clock(), "text": text})
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
                self.say("fleet", t, "Sorry, that message got lost. Please send it again.", "info", reply_to=c["cid"])
                continue
            self.commands.remove(c)
            self._execute(res, c, t)

    def _reply(self, rid, t, text, kind, src):
        self.say(rid or "fleet", t, text, kind, source=src, reply_to=self._reply_to, request_id=self._request_id)

    def _execute(self, res, c: dict, t: float) -> None:
        self._reply_to = c["cid"]
        try:
            self._execute_cmd(res, c, t)
        finally:
            self._reply_to = None

    def _execute_cmd(self, res, c: dict, t: float) -> None:
        cmd = dict(res.command or {"type": "unclear", "question": res.question or "Sorry?"})
        src = "gemma" if res.ok else "rules"
        typ = cmd["type"]
        # a place command goes to the rover the text (or the UI) names; a small model's guess does not count
        rid = self._who(None if typ in ("go_to_place", "repeat") else cmd.get("rover"), c.get("hint"), c.get("text"))
        ids = [r.id for r in self.sim.rovers]
        if rid not in ids and rid != "all":
            rid = None
        named = rid
        prev = self.mem.command
        ref = None
        # protocol v7: chat references ("do that again", "same for rover 3", "go back there", "the second one")
        if typ == "repeat":
            if prev is None:
                frm = rid if rid in ids else "fleet"
                self._reply(frm, t, (f"{frm}: " if frm != "fleet" else "") + "Which command should I repeat? I don't "
                                    "have an earlier one yet.", "question", src)
                return
            cmd, ref = dict(prev["cmd"]), prev.get("ref")
            typ = cmd["type"]
            if rid is None:
                rid = prev.get("rid")
        elif typ == "go_to_place" and (cmd.get("place") == "there" or cmd.get("qualifier") == "mentioned"):
            ref, question = self._resolve_ref(cmd)
            if ref is None and question:
                frm = rid if rid in ids else "fleet"
                self._reply(frm, t, (f"{frm}: " if frm != "fleet" else "") + question, "question", src)
                return
            if ref is None:                  # "that lake" but no lake was mentioned: resolve it as usual
                cmd["qualifier"] = "none"
        if ref is not None and named is None and c.get("hint") not in ids and typ != "repeat":
            rid = self._ref_rover(ref) or rid           # the rover the chat was about, if it is free
        if rid in ids:
            self.mem.rover = rid
        if typ in ACTIONS:
            self.mem.set_command(cmd, rid, ref, c.get("text") or "")
            self.mem.command["cid"] = c["cid"]
        if typ == "unclear":
            frm = rid if rid in ids else "fleet"
            text = res.question or cmd.get("question") or "Sorry, I did not understand."
            self._reply(frm, t, (f"{frm}: " if frm != "fleet" else "") + text, "question", src)
        elif typ == "go_to":
            x, y = float(cmd["x"]), float(cmd["y"])
            if cmd.get("explore"):
                self._cmd_area(rid, x, y, f"({x:g}, {y:g})", t, src)
            else:
                self._cmd_go_to(rid, x, y, t, src)
                self.mem.mention(Ref("point", x, y, f"({x:g}, {y:g})", rover=rid, t=t))
        elif typ == "go_to_place" and ref is not None:
            self._cmd_ref(rid, ref, cmd, t, src, c, prev)
        elif typ == "go_to_place":
            self._cmd_place(rid, cmd, t, src, c)
        elif typ == "explore_direction":
            self._cmd_explore(rid, cmd["direction"], t, src)
        elif typ == "sample_target":
            self._cmd_sample(rid, cmd["target_id"], t, src)
        elif typ in ("return_home", "hold", "resume"):
            getattr(self, "_cmd_" + typ)(rid, t, src)
        elif typ == "avoid_area":
            self._cmd_avoid(float(cmd["x"]), float(cmd["y"]), float(cmd.get("radius") or 3.0), t, src)
        elif typ == "question":
            self._ask(cmd, c, t, res)
        elif typ == "prioritize":
            self._cmd_prioritize(cmd, t, src, c)
        elif typ == "set_mode":
            self._cmd_mode(cmd, t, src, c)
        elif typ == "rescue":
            self._cmd_rescue(cmd, t, src, c)

    def _cmd_rescue(self, cmd: dict, t: float, src: str, c: dict) -> None:
        """Protocol v9: "rover 1, go help rover 2", "tow rover 2 home", "salvage rover 3", "leave it"."""
        plan = cmd.get("plan") or "help"
        target = cmd.get("target")
        helper = cmd.get("rover") if cmd.get("rover") not in (None, "all") else None
        ids = [r.id for r in self.sim.rovers]
        if target not in ids:
            target = None
        if helper not in ids:
            helper = None
        hr = self._rover(helper)
        if hr is not None and target is None and hr.state in HURT:
            helper, target = None, helper     # "rover 2, leave it" said to the rover in trouble: it is the target
        if helper is not None and helper == target:
            self._reply(helper, t, f"{helper}: I can't help myself. Which rover should help?", "question", src)
            return
        ok, text = self.sim.rescue.order(plan, target, helper, t, reply_to=self._reply_to)
        frm = helper or (self.sim.rescue.active(target)["helper"] if target and self.sim.rescue.active(target)
                         and self.sim.rescue.active(target).get("helper") else None) or "fleet"
        if not ok:
            self._reply(frm, t, (text or "I can't do that."), "refuse", "rules")
        elif text:
            self._reply(frm, t, text, "ack", src)

    def _who(self, named, hint, text) -> str | None:
        """Who a message is for. The text itself decides when it names a rover or everyone; otherwise
        the rover picked in the UI (hint) does. Small models often answer "all" (or guess a rover) for
        a message that names nobody, so the model's pick only counts when there is no hint."""
        ids = [r.id for r in self.sim.rovers]
        try:
            ctx = _fleet_context(self.fleet_context())
            said, _rest, _q = _resolve_rover(" ".join(str(text or "").lower().split()), ctx["rovers"])
        except Exception:
            said = None
        if said in ids or said == "all":
            return said
        if hint in ids:
            return hint
        return named if (named in ids or named == "all") else None

    # ================================================================== 6. questions from the operator
    def _ask(self, cmd: dict, c: dict, t: float, res) -> None:
        """Answer an operator question: who answers (one rover, each rover, or the fleet), a fact sheet
        each, the rule answer at once, Gemma's reply when it comes (within the budget)."""
        ids = [r.id for r in self.sim.rovers]
        topic = cmd.get("topic") or "status"
        named = self._who(cmd.get("rover"), c.get("hint"), c.get("text"))
        text = res.text or c.get("text") or ""
        ref = self._question_ref(c.get("text") or text)      # protocol v7: "what did you find there?"
        low = str(c.get("text") or text).lower()
        if ref is not None and named not in ids and _WE_RE.search(low):
            speakers = ["fleet"]            # protocol v8: "what did we find there?" is the fleet at that place
        elif named in ids:
            speakers = [named]
        elif named == "all":
            speakers = list(ids) if (wants_each(text) and topic in PER_ROVER_TOPICS) else ["fleet"]
        elif c.get("hint") in ids:
            speakers = [c["hint"]]
        elif ref is not None and ref.rover in ids:
            speakers = [ref.rover]          # "you" is the rover that went there or talked about it
        else:
            speakers = ["fleet"]
        if named in ids:
            self.mem.rover = named
        budget = self.planner.cfg.budget_ms / 1000.0
        for sp in speakers:
            r = self._rover(sp)
            sheet = rover_sheet(r) if r is not None else fleet_sheet(self.sim)
            facts = fact_lines(sheet, topic)
            rules = rule_answer(sheet, topic)
            context = None
            if ref is not None and topic != "battery":
                where = ref.label if "(" in ref.label else f"{ref.label} at ({ref.x:.0f}, {ref.y:.0f})"
                if topic in ("unknown", "status", "location"):
                    topic = "finds"           # "what's at the vent?": what is known there
                if topic == "why":
                    line = self._why_there(ref, r)
                    facts = [line] + facts[:4]
                    rules = self._sentence(line)
                else:                        # protocol v8: only what is known at that place
                    facts = area_facts(self.sim, ref.x, ref.y, ref.label, ref.r, ref.kind)
                    rules = area_answer(facts)
                context = [f"'there' or 'it' means {where}, from the chat; answer about that place only"]
            elif _MODEQ_RE.search(low):       # "are you in autonomous mode?", "what's the priority?"
                lines = self._mode_facts()
                facts, rules = lines + facts[:3], self._sentence("; ".join(lines))
                topic = "status"
            elif topic == "exploration" and r is None:
                extra = self._gap_fact(t)
                if extra and re.search(r"\b(?:left|remain\w*|still|unexplored|unmapped|gaps?|holes?|missing)\b", low):
                    facts = [extra] + facts[:5]       # "what's left to explore?": the gaps and frontiers first
                    rules = self._sentence(extra) + " " + rules
                elif extra:
                    facts = facts[:5] + [extra]
                    rules = rules + " " + self._sentence(extra)
            job = {"speaker": sp, "cid": c["cid"], "rules": rules, "topic": topic, "wall": self.clock(),
                   "ticket": None, "budget": budget}
            if self.planner.online:
                job["ticket"] = self.planner.submit_answer(sp, t, text, topic, facts, rules, timeout_s=budget,
                                                           context=context, **self._gravity_kw())
                self.answers.append(job)
                if self.wait_for_gemma:
                    self._wait(job["ticket"])
            else:
                self._post_answer(job, None, t)
        if self.answers and self.wait_for_gemma:
            self._poll_answers(t)

    def _poll_answers(self, t: float) -> None:
        for job in list(self.answers):
            res = self.planner.poll(job["ticket"])
            if res is None and self.clock() - job["wall"] < job["budget"]:
                continue                   # still within the latency budget
            self.answers.remove(job)
            self._post_answer(job, res, t)

    def _post_answer(self, job: dict, res, t: float) -> None:
        ok = res is not None and res.ok and bool(res.answer)
        text = res.answer if ok else job["rules"]
        src = "gemma" if ok else "rules"
        self.n_answers[src] += 1
        self.say(job["speaker"], t, text, "reply", source=src, reply_to=job["cid"])

    # ---- helpers for commands
    def _targets_for(self, rid) -> list:
        return list(self.sim.rovers) if rid in (None, "all") else [r for r in self.sim.rovers if r.id == rid]

    def _busy_reason(self, r) -> str | None:
        """Why this rover cannot take a new job right now (safety rules), or None."""
        why = self._v9_busy(r)
        if why is not None:
            return why
        if r.state == "fault":
            return "my motors are offline (rebooting)"
        if r.state == "returning" and r.return_reason in ("battery", "range"):
            return f"battery {r.battery * 100:.0f}%, I am returning to charge"
        return None

    def _v9_busy(self, r, allow_rescuing: bool = False) -> str | None:
        """Protocol v9: why a rover out of action, being repaired, towing or helping can't take a job, or None."""
        st = r.state
        if st == "disabled":
            return f"I can't drive ({r.health.short()}); I'm waiting for help"
        if st == "towed":
            return "I'm being towed to the lander"
        if st == "wrecked":
            return f"I'm a wreck at ({r.x:.0f}, {r.y:.0f}), out of action"
        if st == "repairing":
            return f"I'm being repaired at the lander (health {r.health.hp()})"
        resc = getattr(self.sim, "rescue", None)
        rec = resc.by_helper(r.id) if resc is not None else None
        if st == "towing" and rec is not None:
            return f"I'm towing {rec['rover']} home; say 'leave it' to stop"
        if st == "rescuing" and rec is not None and not allow_rescuing:
            return f"I'm helping {rec['rover']}; send another rover to help it, or say 'leave it'"
        return None

    def _cancel_go(self, rid: str, t: float, why: str) -> None:
        """A go_to still being planned is replaced by a newer command: say so (one reply each)."""
        job = self.go_jobs.pop(rid, None)
        if job is not None:
            x, y = job["goal"]
            self.say(rid, t, f"{rid}: dropped the trip to ({x:g}, {y:g}), {why}.", "info", source="rules",
                     reply_to=job.get("cid"))

    def _go_need(self, r, x: float, y: float, dist: float) -> float:
        return r.energy_for_distance(dist + 1.3 * math.hypot(x, y)) + r.battery_margin

    def _free_up(self, r, t: float, why: str) -> None:
        """The operator gave this rover a job: stop holding, drop other plans (and its open question: the
        operator has decided; otherwise a rover waiting for an answer would never move again)."""
        self._supersede(r, t)
        if r.survey is not None:
            r.end_survey(t, "new command")
        if r.focus is not None:
            r.end_focus(t, "new command", quiet=r.focus["reason"] == "interest")
        r.pending_interest = None
        r.held = False
        r.op_goal = None
        r.op_task = None
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

    def _cmd_go_to(self, rid, x: float, y: float, t: float, src: str, extra: dict | None = None) -> None:
        """extra: more fields for the go job (a place command: its label, the task on arrival, other goals)."""
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
        self.go_jobs[rid] = dict({"goal": (x, y), "t0": t, "src": src, "cid": self._reply_to,
                                  "qid": self._request_id}, **(extra or {}))

    def _plan_go(self, rover, t: float) -> None:
        job = self.go_jobs[rover.id]
        prev = (self._reply_to, self._request_id)
        self._reply_to, self._request_id = job.get("cid"), job.get("qid")
        try:
            self._plan_go_job(rover, job, t)
        finally:
            self._reply_to, self._request_id = prev

    def _plan_go_job(self, rover, job: dict, t: float) -> None:
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
        if path is None and job.get("alts"):
            job["goal"], job["t0"] = job["alts"].pop(0), t      # a place: try the next safe spot next to it
            return
        del self.go_jobs[rid]
        if path is None:
            where = f"{job['place']}" if job.get("place") else f"({x:g}, {y:g})"
            self._reply(rid, t, f"{rid}: can't, no safe path to {where} within the "
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
        rover.op_task = job.get("task")
        rover.goal, rover.goal_t = (x, y), t
        rover._set_path(t, path)
        after = (rover.battery - need + rover.battery_margin) * 100.0
        if job.get("place"):
            task = job.get("task") or {}
            fs = task.get("focus")
            tail = f" I'll survey there for about {SURVEY_S:.0f} s." if task.get("investigate") else (
                f" I'll explore around it for about {_dur(self.cfg.focus_s)}." if fs else "")
            if fs and rover.mostly_mapped(fs["x"], fs["y"], fs["r"]):     # protocol v8: a re-survey
                kn, unk = self.fleet.map.area_cells(fs["x"], fs["y"], fs["r"])
                rs = (f"it is already {100.0 * kn / max(kn + unk, 1):.0f}% mapped, so I'll re-survey it for about "
                      f"{_dur(self.cfg.focus_s)}: a loop through it, rescanning for changes and anything missed.")
                tail = (f" I'll survey there for about {SURVEY_S:.0f} s; {rs}" if task.get("investigate")
                        else " " + rs[0].upper() + rs[1:])
            self._reply(rid, t, f"{rid}: {job.get('announce') or ''}Heading to {job['place']} at ({x:g}, {y:g}), "
                                f"{plen:.0f} m. Max slope on the way {smax:.0f} deg, battery after return about "
                                f"{after:.0f}%.{tail}", "ack", job["src"])
            self._log(t, "info", f"{rid} heading to {job['place']} for the operator, {plen:.0f} m, max slope "
                                 f"{smax:.0f} deg.", rid, at=(x, y), source="operator")
            return
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
        ref = self._target_ref(tid, rid)
        if ref is not None:
            self._note_ref(ref)
        self._reply(rid, t, f"{rid}: going to sample {tid} ({KIND_LABEL[ts.spec.kind]}), {d:.0f} m away, "
                            f"{r.free_tubes()} of {r.capacity} tubes free.", "ack", src)
        self._log(t, "info", f"{rid} heading to sample {tid} for the operator, {d:.0f} m.", rid, at=(tx, ty),
                  source="operator")

    def _cmd_return_home(self, rid, t: float, src: str) -> None:
        lines = []
        for r in self._targets_for(rid):
            dd = math.hypot(r.x - r.dock[0], r.y - r.dock[1])
            why = self._v9_busy(r, allow_rescuing=True)
            if why is not None:
                lines.append((r.id, f"can't, {why}", "refuse"))
                continue
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
            why = self._v9_busy(r)
            if why is not None:
                lines.append((r.id, f"can't hold, {why}", "refuse"))
                continue
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
            if r.focus is not None and r.focus["reason"] == "operator":
                was = True
                r.end_focus(t, "resume")
            r.held = False
            r.op_goal = None
            r.dir_bias = None
            r.pause_until = min(r.pause_until, t)
            self._cancel_go(r.id, t, "resuming exploration")
            why = self._v9_busy(r)
            if why is not None:
                lines.append((r.id, f"not now, {why}", "ack"))
                continue
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
        if rid in (None, "all") and len(lines) > 1 and not any("resuming" in txt for _, txt, _ in lines):
            other = [f"{r_id} {txt}" for r_id, txt, _ in lines if txt != "already exploring"]
            self._reply("fleet", t, "Nothing to resume: every rover is already exploring on its own"
                        + (f" ({'; '.join(other)})" if other else "") + ".", "ack", src)
            return
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

    # ================================================================== 7. places (protocol v6)
    def _supersede(self, r, t: float) -> None:
        """The operator gave this rover a new job: its open questions are settled (closed, no reply)."""
        for q in [q for q in self.requests.values() if q["rover"] == r.id]:
            self._close(q["id"], "superseded", t)
            self._log(t, "info", f"{r.id} dropped its open question {q['id']}: the operator gave it a new command.",
                      r.id, at=(r.x, r.y))

    def _free_rovers(self) -> list:
        """Rovers free for a place command: not faulted, not returning on a low battery, not charging with a
        low battery. Those without an open question of their own come first."""
        free = [r for r in self.sim.rovers if self._busy_reason(r) is None
                and not (r.state == "charging" and r.battery < FREE_CHARGE)]
        calm = [r for r in free if not any(q["rover"] == r.id for q in self.requests.values())]
        return calm or free

    def _cmd_place(self, rid, cmd: dict, t: float, src: str, c: dict) -> None:
        """'go to the water source', 'rover 3, survey the spires', 'can we investigate the lake again'."""
        ids = [r.id for r in self.sim.rovers]
        place, tid = cmd.get("place"), cmd.get("target_id")
        inv = bool(cmd.get("investigate"))
        explore, side = bool(cmd.get("explore")), cmd.get("direction")      # protocol v7: an area to explore
        if side not in DIRS:
            side = None
        sample = bool(_SAMPLE_WORD_RE.search(str(c.get("text") or "").lower()))
        named = rid if rid in ids else None
        if place == "lander" and not inv and not explore:   # "go to the lander": home (and charge), one rover
            if named is None:
                free = self._free_rovers()
                if free:
                    named = min(free, key=lambda r_: math.hypot(r_.x - r_.dock[0], r_.y - r_.dock[1])).id
            self._cmd_return_home(named or "all", t, src)
            return
        cands = P.candidates(self.sim, place, tid, t)
        if not cands:
            text = self._no_gaps_text(t) if place == "gaps" else P.nothing_known(self.sim, place, tid, t,
                                                                                 fleet=named is None)
            self._reply(named or "fleet", t, (f"{named}: " if named else "") + text, "info", "rules")
            return
        qual, ordinal = cmd.get("qualifier") or "none", cmd.get("ordinal")
        near_xy = None
        if qual == "near_rover":
            nr = cmd.get("near_rover")
            ro = self._rover(nr)
            near_xy = (0.0, 0.0) if nr == "lander" else ((ro.x, ro.y) if ro is not None else None)
        announce = ""
        if named:
            r = self._rover(named)
            busy = self._busy_reason(r)
            if busy:
                self._reply(named, t, f"{named}: can't, {busy}.", "refuse", "rules")
                return
        else:
            free = self._free_rovers()
            if not free:
                self._reply("fleet", t, "No rover is free to go there right now (motors offline, returning on a low "
                                        "battery, or charging).", "refuse", "rules")
                return

            def dist(r_, p):
                return math.hypot(p.x - r_.x, p.y - r_.y)

            if qual == "nearest":
                r, p0 = min(((r_, p) for r_ in free for p in cands), key=lambda rp: dist(*rp))
                cands = [p0]
            else:
                if qual not in ("none", "ordinal"):
                    left, ok = P.apply_qualifier(cands, qual, ordinal, ref=(0.0, 0.0), near_xy=near_xy)
                    if ok:
                        cands = left
                able = [r_ for r_ in free if r_.battery > self._go_need(r_, cands[0].x, cands[0].y,
                                                                        1.3 * dist(r_, cands[0]))]
                pool = (able or free) if len(cands) == 1 else free
                r = min(pool, key=lambda r_: min(dist(r_, p) for p in cands))
            announce = f"I'm the closest free rover ({min(dist(r, p) for p in cands):.0f} m). "
        left, decisive = P.apply_qualifier(cands, qual, ordinal, ref=(r.x, r.y), near_xy=near_xy)
        if decisive and len(left) == 1:
            self._go_place(r, left[0], inv, t, src, announce=announce, sample=sample, explore=explore, side=side)
            return
        note = ""
        if qual in ("biggest", "smallest", "newest"):
            note = "I can't tell them apart by that. "
        elif qual == "ordinal" and ordinal:
            note = f"There is no number {ordinal}. "
        kind = place if tid is None else "target"
        if not self.supervised:            # protocol v8: Gemma decides which one (the rules when offline)
            self._decide_place(r, cands, kind, inv, t, src, announce + note, sample, explore, side, c.get("text"))
            return
        qid = self._open_place_request(r, cands, kind, inv, t, src, announce + note, sample, explore=explore,
                                       side=side)
        self._ask_place_suggestion(r, qid, c.get("text"), t)

    def _go_place(self, r, p, inv: bool, t: float, src: str, announce: str = "", sample: bool = False,
                  explore: bool = False, side: str | None = None) -> None:
        """Send rover r to a safe spot next to place p through the operator go_to path (all its checks). With
        explore (or investigate, after the survey) it then explores around the place (protocol v7 focus); with a
        side ("explore north of the lander") it explores the area that way of it at once."""
        if sample and p.kind in P.TARGET_KINDS:
            self._cmd_sample(r.id, p.id, t, src)        # "sample the vent": the instrument, not just a visit
            return
        p = P.refresh(self.sim, p, t)
        label = "the herd's area" if p.kind == "lifeform" else p.dest()
        self._note_ref(self._place_ref(p, r.id, "the herd" if p.kind == "lifeform" else None))
        if p.kind == "gaps" and side is None:      # protocol v8: an unexplored gap is an area to explore
            rad = max(6.0, min(20.0, (p.zones[0][2] if p.zones else 6.0) + 3.0))
            self._cmd_area(r.id, p.x, p.y, label, t, src, radius=rad, announce=announce, mode="explore",
                           what=f" (about {p.size:.0f} m2 of unknown ground)")
            return
        if side is not None and explore:
            dx, dy = DIRS[side]
            off = self.cfg.focus_offset
            where = "the lander" if p.kind == "lander" else (f"the {p.sub}" if p.kind == "water" else label)
            self._cmd_area(r.id, p.x + off * dx, p.y + off * dy, f"the area {side} of {where}", t, src,
                           announce=announce)
            return
        goals = P.approach_goals(self.sim, p, (r.x, r.y), t)
        focus = None
        if explore or inv:
            rad = self.cfg.focus_r if p.kind != "water" else max(self.cfg.focus_r, 0.5 * (p.size or 0.0) + 8.0)
            focus = {"x": round(p.x, 1), "y": round(p.y, 1), "r": round(rad, 1), "label": label}
        if not goals:
            if explore and not inv:          # no mapped spot next to it yet: explore the area around it directly
                self._cmd_area(r.id, p.x, p.y, label, t, src, radius=focus["r"], announce=announce)
                return
            self._reply(r.id, t, f"{r.id}: can't, I know no safe, mapped ground next to {p.dest()} yet "
                                 f"({p.noun().lower()} at ({p.x:.0f}, {p.y:.0f})).", "refuse", "rules")
            return
        task = {"label": p.dest(), "investigate": bool(inv), "cid": self._reply_to, "qid": self._request_id,
                "at": (round(p.x, 1), round(p.y, 1)), "kind": p.kind, "noun": p.noun(), "focus": focus}
        self._cmd_go_to(r.id, goals[0][0], goals[0][1], t, src,
                        extra={"place": p.dest(), "task": task, "alts": list(goals[1:]), "announce": announce})

    def _note_ref(self, ref: Ref) -> None:
        """A place command resolved to this place: it is "there" now, and what "do that again" repeats."""
        self.mem.mention(ref)
        cmd = self.mem.command
        if cmd is not None and cmd.get("cid") is not None and cmd.get("cid") == self._reply_to:
            cmd["ref"] = ref

    def _ref_rover(self, ref: Ref):
        """The rover a reference was about (or the one the operator talked with last), if it is free."""
        for rid in (ref.rover, self.mem.rover):
            r = self._rover(rid) if rid else None
            if r is not None and self._busy_reason(r) is None and not (r.state == "charging" and r.battery < FREE_CHARGE):
                return r.id
        return None

    def _resolve_ref(self, cmd: dict):
        """(Ref, None), or (None, a question back), or (None, None) when "that lake" has no match in the chat
        (then it resolves like "the lake")."""
        if cmd.get("place") == "there":
            if cmd.get("qualifier") == "ordinal" and cmd.get("ordinal"):
                ref = self.mem.pick(int(cmd["ordinal"]))
                if ref is not None:
                    return ref, None
                n = len(self.mem.cands)
                if n == 0:
                    return None, "Which one do you mean? We haven't talked about a list of places yet."
                return None, f"The last list had only {n}. Which one do you mean?"
            if self.mem.there is not None:
                return self.mem.there, None
            return None, ("Where do you mean? We haven't talked about a place yet. Name it (the lake, t4) or give "
                          "x, y.")
        return self.mem.last_of(cmd.get("place") or ""), None

    def _fresh_place(self, ref: Ref, t: float):
        """The place behind a reference as the fleet knows it now (a lake may have grown, a herd moved)."""
        p = ref.place
        if p is None:
            return None
        cur = [c for c in P.candidates(self.sim, p.kind if p.kind != "target" else "target", None, t)
               if c.kind == p.kind and (c.id == p.id or math.hypot(c.x - p.x, c.y - p.y) < 10.0)]
        return min(cur, key=lambda c: math.hypot(c.x - p.x, c.y - p.y)) if cur else p

    def _cmd_ref(self, rid, ref: Ref, cmd: dict, t: float, src: str, c: dict, prev) -> None:
        """A place command that names a chat reference ("go back there", "the second one", "that lake")."""
        text = str(c.get("text") or "").lower()
        inv, explore = bool(cmd.get("investigate")), bool(cmd.get("explore"))
        sample = bool(_SAMPLE_WORD_RE.search(text))
        bare = not (inv or explore or sample or _PLACE_VERB_RE.search(text))
        if bare and prev is not None and prev["cmd"].get("type") == "go_to_place":
            inv, explore = bool(prev["cmd"].get("investigate")), bool(prev["cmd"].get("explore"))   # "the second one"
        if rid == "all":
            rid = None
        announce = ""
        if rid is None:
            free = self._free_rovers()
            if not free:
                self._reply("fleet", t, "No rover is free to go there right now (motors offline, returning on a low "
                                        "battery, or charging).", "refuse", "rules")
                return
            r = min(free, key=lambda r_: math.hypot(ref.x - r_.x, ref.y - r_.y))
            announce = f"I'm the closest free rover ({math.hypot(ref.x - r.x, ref.y - r.y):.0f} m). "
        else:
            r = self._rover(rid)
            busy = self._busy_reason(r)
            if busy:
                self._reply(rid, t, f"{rid}: can't, {busy}.", "refuse", "rules")
                return
        p = self._fresh_place(ref, t)
        if p is not None:
            if p.kind == "lander" and not inv and not explore:
                self._cmd_return_home(r.id, t, src)
                return
            self._go_place(r, p, inv, t, src, announce=announce, sample=sample, explore=explore,
                           side=cmd.get("direction") if cmd.get("direction") in DIRS else None)
        elif explore or inv or (ref.kind == "area" and not self._passable(ref.x, ref.y, t)):
            self._cmd_area(r.id, ref.x, ref.y, ref.label, t, src, radius=ref.r, announce=announce)
        else:
            self._cmd_go_to(r.id, ref.x, ref.y, t, src)
            self._note_ref(ref)

    def _passable(self, x: float, y: float, t: float) -> bool:
        fm = self.fleet.map
        i, j = fm.ij(x, y)
        return bool(fm.known[j, i] and math.isfinite(self.fleet.cost(t)[j, i]))

    def _cmd_area(self, rid, x: float, y: float, label: str, t: float, src: str, radius: float | None = None,
                  announce: str = "", mode: str | None = None, what: str = "") -> None:
        """Explore an area now (protocol v7 focus): no known spot to drive to first, so the rover's frontier
        choice takes it into the circle (unknown ground inside first, then the nearest edge) and it stays."""
        half = (self.cfg.nx - 1) * self.cfg.res / 2.0
        rad = float(radius or self.cfg.focus_r)
        if rid == "all":
            self._reply("fleet", t, f"One rover at a time for that, please. Which one should explore around {label}?",
                        "question", src)
            return
        if abs(x) > half - 2.0 or abs(y) > half - 2.0:
            x, y = max(-half + 2.0, min(half - 2.0, x)), max(-half + 2.0, min(half - 2.0, y))
            if label.startswith("("):
                self._reply(rid or "fleet", t, f"{rid + ': ' if rid else ''}can't, {label} is outside the map (x and y "
                                               f"from {-half + 2.0:g} to {half - 2.0:g} m).", "refuse", "rules")
                return

        def need(r_):
            return r_.energy_for_distance(1.3 * math.hypot(x - r_.x, y - r_.y) + 1.3 * math.hypot(x, y)) + \
                r_.battery_margin + 0.05

        if rid is None:
            free = [r_ for r_ in self._free_rovers() if r_.state != "recovering"]
            able = [r_ for r_ in free if r_.battery > need(r_)]
            if not able:
                self._reply("fleet", t, f"No rover can explore around {label} and get back right now (battery or "
                                        f"motors).", "refuse", "rules")
                return
            r = min(able, key=lambda r_: math.hypot(x - r_.x, y - r_.y))
            announce = announce or f"I'm the closest free rover ({math.hypot(x - r.x, y - r.y):.0f} m). "
        else:
            r = self._rover(rid)
        busy = self._busy_reason(r)
        if busy:
            self._reply(r.id, t, f"{r.id}: can't, {busy}.", "refuse", "rules")
            return
        if r.battery < need(r):
            self._reply(r.id, t, f"{r.id}: can't, battery {r.battery * 100:.0f}% would not get me to {label} and back "
                                 f"(needs about {need(r) * 100:.0f}% with the margin).", "refuse", "rules")
            return
        kn, unk = self.fleet.map.area_cells(x, y, rad)
        self._free_up(r, t, "exploring an area for the operator")
        r.state = "exploring"
        fc = r.start_focus(t, x, y, rad, "operator", label, self.cfg.focus_s, cid=self._reply_to,
                           qid=self._request_id, mode=mode)
        d = math.hypot(x - r.x, y - r.y)
        where = "I'm in it" if d <= rad else f"it is {d:.0f} m {compass(x - r.x, y - r.y)} of me"
        if fc["mode"] == "resurvey":            # protocol v8: already mapped: a loop through it, rescanning
            self._reply(r.id, t, f"{r.id}: {announce}{label[:1].upper() + label[1:]} is already "
                                 f"{100.0 * kn / max(kn + unk, 1):.0f}% mapped, so I'll re-survey it for about "
                                 f"{_dur(self.cfg.focus_s)}: a loop through it within {rad:.0f} m, rescanning for "
                                 f"changes and anything missed; {where}.", "ack", src)
        else:
            self._reply(r.id, t, f"{r.id}: {announce}Exploring around {label}{what} for about "
                                 f"{_dur(self.cfg.focus_s)}, within {rad:.0f} m; {where}. About "
                                 f"{100.0 * unk / max(kn + unk, 1):.0f}% of that area is still unmapped.", "ack", src)
        self._log(t, "info", f"{r.id} exploring around {label} for the operator (within {rad:.0f} m).", r.id,
                  at=(x, y), source="operator")
        self._note_ref(Ref("area", float(x), float(y), label, rover=r.id, t=t, r=rad))

    def _open_place_request(self, r, cands: list, place: str, inv: bool, t: float, src: str, prefix: str,
                            sample: bool, explore: bool = False, side: str | None = None) -> str:
        """Several places match: ask which one (numbered options with x, y for the map, plus Never mind).
        The rover holds position meanwhile. Returns the request id."""
        for q in [q for q in self.requests.values() if q["rover"] == r.id and q["kind"] == "choose_place"]:
            self._close(q["id"], "superseded", t)
        self.req_seq += 1
        qid = f"q{self.req_seq}"
        frm = (r.x, r.y)
        shown = self._rank_places(cands, place, frm)[:P.MAX_OPTIONS]
        opts = [(f"p{k + 1}", P.option_label(p, frm), round(p.x, 1) + 0.0, round(p.y, 1) + 0.0)
                for k, p in enumerate(shown)]
        opts.append(("cancel", "Never mind"))
        more = f" These are the {len(shown)} closest." if len(cands) > len(shown) else ""
        question = f"{prefix}I know {P.count_noun(place, len(cands))}. Which one?{more}"
        dl = self._deadline()
        self.requests[qid] = {
            "id": qid, "rover": r.id, "t": round(t, 1), "question": question, "options": opts,
            "default": "cancel" if dl is not None else None, "deadline": dl, "image": None, "kind": "choose_place",
            "job": {"at": frm, "cands": {f"p{k + 1}": p for k, p in enumerate(shown)}, "place": place,
                    "investigate": bool(inv), "cid": self._reply_to, "src": src, "sample": sample,
                    "explore": bool(explore), "side": side}}
        r.question_hold = True             # stay put until the operator picks (safety moves still allowed)
        self.say(r.id, t, question, "question", source=src, request_id=qid, reply_to=self._reply_to)
        self.mem.mention_list([self._place_ref(p, r.id) for p in shown])     # "the second one" picks from these
        self._log(t, "info", f"{r.id} knows {P.count_noun(place, len(cands))}: asking the operator which one. "
                             f"Holding position meanwhile.", r.id, at=frm, source="operator")
        return qid

    def _apply_place_choice(self, rover, q: dict, option: str, t: float, quiet: bool = False) -> None:
        job = q["job"]
        prev = (self._reply_to, self._request_id)
        self._reply_to, self._request_id = job.get("cid"), q["id"]
        try:
            if rover is None:
                return
            if option == "cancel":
                if not quiet:
                    self._reply(rover.id, t, f"{rover.id}: OK, never mind. Carrying on with my own plan.", "ack",
                                "rules")
                return
            p = job["cands"].get(option)
            if p is None:
                return
            busy = self._busy_reason(rover)
            if busy:
                self._reply(rover.id, t, f"{rover.id}: can't, {busy}.", "refuse", "rules")
                return
            self._go_place(rover, p, job["investigate"], t, job["src"], sample=job.get("sample", False),
                           explore=job.get("explore", False), side=job.get("side"), announce=job.get("announce", ""))
        finally:
            self._reply_to, self._request_id = prev

    def _chat_pick(self, text: str, hint, ctx: dict):
        """(request, option) if the operator's chat message clearly picks an option of an open choose_place
        request ("the second one", "p2", "the one near the lander", "the lake", "never mind"), else None."""
        opens = [q for q in self.requests.values() if q["kind"] == "choose_place"]
        if not opens:
            return None
        low = " ".join(str(text or "").lower().split())
        rovers = _fleet_context(ctx)["rovers"]
        if not low or len(low.split()) > 12 or question_intent(low, rovers)[0] == "strong":
            return None
        ids = [r.id for r in self.sim.rovers]
        said, _rest, _q = _resolve_rover(low, rovers)
        if said in ids:
            pool = [q for q in opens if q["rover"] == said]
        elif hint in ids:
            pool = [q for q in opens if q["rover"] == hint] or opens
        else:
            pool = opens
        if not pool:
            return None
        q = pool[-1]
        parsed = parse_command(text, ctx)["command"]
        if parsed["type"] in ("go_to", "explore_direction", "sample_target", "return_home", "hold", "resume",
                              "avoid_area"):
            return None                    # a new order, not an answer
        if parsed["type"] == "go_to_place" and parsed.get("place") not in (q["job"]["place"], "there"):
            return None                    # "the second one" (a reference, v7) still picks from the open list
        r = self._rover(q["rover"])
        frm = (r.x, r.y) if r is not None else q["job"]["at"]
        opt = P.pick_option(low, q["job"]["cands"], frm, {o.id: (o.x, o.y) for o in self.sim.rovers})
        return (q, opt) if opt is not None else None

    def on_place_arrived(self, rover, task: dict, t: float) -> None:
        """Rover hook: reached the spot next to a place the operator named (the survey has already started
        when the command said investigate)."""
        fs = task.get("focus")
        mins = _dur(self.cfg.focus_s)
        rs = rover.focus is not None and rover.focus.get("mode") == "resurvey" and not task.get("investigate")
        if rs:                               # protocol v8
            msg = (f"{rover.id}: arrived at {task['label']}. Re-surveying it for about {mins}: a loop through it "
                   f"within {fs['r']:.0f} m, rescanning for changes and anything missed.")
        elif task.get("investigate"):
            msg = (f"{rover.id}: arrived at {task['label']}. Surveying for about {SURVEY_S:.0f} s: turning slowly "
                   f"through 360 deg while my LiDAR watches for movement."
                   + (f" Then I'll explore around it for about {mins}." if fs else ""))
        elif fs:
            msg = (f"{rover.id}: arrived at {task['label']}. Exploring around it for about {mins}, within "
                   f"{fs['r']:.0f} m.")
        else:
            msg = f"{rover.id}: arrived at {task['label']}. Back to exploring."
        self.say(rover.id, t, msg, "info", reply_to=task.get("cid"), request_id=task.get("qid"))
        self._log(t, "info", f"{rover.id} reached {task['label']} for the operator.", rover.id,
                  at=(rover.x, rover.y), source="operator")

    def survey_report(self, rover, sv: dict, t: float, why: str) -> None:
        """Rover hook: a survey ended. Report in chat (kind info, reply_to the command) from the rover's own
        tracker, spectrometer, map and memory: what moved and what it saw. Only measured numbers."""
        f = self.fleet
        dur = max(0.0, t - sv["t0"])
        turned = math.degrees(sv["turned"])
        label = sv["label"]
        if why == "done":
            sweep = "a full circle" if turned >= 340.0 else f"about {turned:.0f} deg"
            head = f"Survey of {label} done: {dur:.0f} s, turned {sweep}."
        elif why == "observing":
            head = f"Surveyed {label} for {dur:.0f} s."
        else:
            reason = {"returning": (f"battery {rover.battery * 100:.0f}%, returning to the lander"
                                    if rover.return_reason in ("battery", "range") else "returning to the lander"),
                      "idle": "holding for the operator", "recovering": "backing out of trouble",
                      "new command": "a new command", "fault": "motor fault"}.get(why, why)
            head = f"Survey of {label} stopped after {dur:.0f} s ({reason})."
        parts = [head]
        tracks = sorted(sv["tracks"].values(), key=lambda e: e["dist"])
        if tracks:
            bits = []
            for e in tracks[:2]:
                what = e["label"] if e["label"] and e["label"] != "moving object" else "moving object"
                bits.append(f"{e['id']} ({what}, about {e['size']:.1f} m) {e['dist']:.0f} m {e['dir']}, "
                            f"up to {e['speed']:.1f} m/s")
            more = f", and {len(tracks) - 2} more" if len(tracks) > 2 else ""
            parts.append("Movement: " + "; ".join(bits) + more + ".")
            if why == "observing" and sv.get("observing"):
                parts.append(f"Stopping to watch {sv['observing']}.")
        else:
            parts.append(f"Nothing moved within my {rover.lidar_eff:.0f} m LiDAR range.")
        seen = []
        for k, dmin in sorted(sv["targets"].items(), key=lambda kv: kv[1])[:3]:
            ts = f.targets[k]
            seen.append(f"{ts.spec.id} ({KIND_LABEL[ts.spec.kind]}) "
                        f"{math.hypot(ts.spec.x - rover.x, ts.spec.y - rover.y):.0f} m "
                        f"{compass(ts.spec.x - rover.x, ts.spec.y - rover.y)}")
        if seen:
            parts.append("Spectrometer: " + ", ".join(seen) + ".")
        gap = f.water_zone_gap(rover.x, rover.y)
        if gap < 10.0:
            z = min(f.water_zones, key=lambda z: math.hypot(rover.x - z[0], rover.y - z[1]) - z[2])
            parts.append(f"The water's edge is about {max(gap, 0.0):.0f} m {compass(z[0] - rover.x, z[1] - rover.y)} "
                         f"of me.")
        traces = [ts for k, ts in enumerate(f.targets) if ts.status is not None and ts.spec.kind == "trace"
                  and k not in sv["targets"] and math.hypot(ts.spec.x - rover.x, ts.spec.y - rover.y) < 15.0]
        for ts in traces[:1]:
            parts.append(f"Organic trace {ts.spec.id} (found earlier) is "
                         f"{math.hypot(ts.spec.x - rover.x, ts.spec.y - rover.y):.0f} m "
                         f"{compass(ts.spec.x - rover.x, ts.spec.y - rover.y)}.")
        if not tracks:
            herds = [(hx, hy, n) for hx, hy, n in f.lifeform_herds if math.hypot(hx - rover.x, hy - rover.y) < 25.0]
            if herds:
                hx, hy, n = min(herds, key=lambda h: math.hypot(h[0] - rover.x, h[1] - rover.y))
                who = f"a herd of {n}" if n > 1 else "a creature"
                parts.append(f"The fleet saw {who} about {math.hypot(hx - rover.x, hy - rover.y):.0f} m "
                             f"{compass(hx - rover.x, hy - rover.y)} earlier.")
        new_m2 = (rover.mem.mapped_cells - sv["mapped0"]) * self.cfg.res ** 2
        if new_m2 >= 1.0:
            parts.append(f"Mapped {new_m2:.0f} m2 of new ground.")
        if sv.get("then_focus"):
            parts.append(f"Now exploring around it for about {_dur(self.cfg.focus_s)}.")
        text = parts[0]
        for ptxt in parts[1:]:
            if len(text) + 1 + len(ptxt) > 400:
                break
            text += " " + ptxt
        self.say(rover.id, t, text, "info", reply_to=sv.get("cid"), request_id=sv.get("qid"))
        self._log(t, "discovery" if tracks else "info", f"{rover.id} {parts[0][0].lower() + parts[0][1:]} "
                  f"{parts[1]}", rover.id, at=(rover.x, rover.y), source="operator")
        at = sv.get("at") or (rover.x, rover.y)
        self.mem.add_report(Ref("area", float(at[0]), float(at[1]), label, rover=rover.id, t=t, r=10.0), text,
                            rover.id, t, mention=why != "new command")

    # ================================================================== 8. protocol v7: area focus reports
    def focus_report(self, rover, fc: dict, s: dict, t: float, why: str, quiet: bool = False) -> None:
        """Rover hook: a focus ended. What it mapped there (area), found (targets, water, spires, herds) and how
        long, from the rover's own memory (rover.focus_summary: measured numbers only). An operator focus
        reports in chat (kind info, reply_to the command); the rover's own curiosity in the event log."""
        label, rid = s["label"], rover.id
        found = [f"{d['id']} ({KIND_LABEL.get(d['kind'], d['kind'])})" for d in s["targets"]]
        sampled = [d["id"] for d in s["samples"]]
        for w in s["water"][:1]:
            found.append(f"water about {2.0 * w['r']:.0f} m across")
        if s["spires"]:
            n = len(s["spires"])
            found.append(f"{n} spire{'s' if n != 1 else ''}")
        for e in s["life"][-1:]:
            found.append((f"a herd of {e['n']}" if e["n"] > 1 else "a creature") + f" (watched {e['dur']:.0f} s)")
        resurvey = s.get("mode") == "resurvey"
        if not s["there"]:
            head = f"Didn't reach {label}"
        elif resurvey:                       # protocol v8: what the loop through it found or changed
            laps = s.get("laps", 0)
            head = (f"Re-surveyed {label} for {s['dur']:.0f} s: drove {s.get('driven', 0)} m through it"
                    + (f" ({laps} lap{'s' if laps != 1 else ''})" if laps else "")
                    + ", rescanning with LiDAR and the spectrometer")
            news = [f"{n} (missed on the first pass)" for n in s.get("new_found") or []]
            if s.get("changes"):
                news.insert(0, f"{s['changes']} map samples changed since it was mapped")
            for e in s["life"][-1:]:
                news.append((f"a herd of {e['n']}" if e["n"] > 1 else "a creature") + f" (watched {e['dur']:.0f} s)")
            found = news
        elif why == "explored" and s["dur"] < 3.0 and s["mapped"] < 1:
            head = f"Around {label} everything within {s['r']} m is already mapped ({s['pct']}%)"
        else:
            head = (f"Explored around {label} for {s['dur']:.0f} s: mapped {s['mapped']} m2 of new ground; the area "
                    f"within {s['r']} m is now {s['pct']}% mapped")
        body = ("Found " + ", ".join(found) + (f"; sampled {', '.join(sampled)}" if sampled else "") + ".") if found \
            else ("Nothing new found there." if s["there"] else "")
        if resurvey and s["there"]:
            body = ("Changes: " + "; ".join(found) + (f". Sampled {', '.join(sampled)}" if sampled else "") + ".") \
                if found else ("No changes since it was mapped, and nothing new found"
                               + (f"; sampled {', '.join(sampled)}" if sampled else "") + ".")
        tail = {"time": "Back to normal exploration.",
                "explored": "Nothing unknown left nearby. Back to normal exploration.",
                "blocked": f"Couldn't reach the rest of the unknown ground there (steep or blocked); "
                           f"{max(0, 100 - int(round(float(s.get('pct') or 0))))}% of it is still unmapped. "
                           f"Back to normal exploration.",
                "battery": f"Stopped: battery {rover.battery * 100:.0f}%, returning to the lander.",
                "unreached": f"Gave up after {t - fc['t0']:.0f} s: no way into that area yet. Back to normal exploration.",
                "held": "Stopped: holding for the operator.", "new command": "Stopped for a new command.",
                "resume": "Stopped: resuming normal exploration.", "replaced": "Stopped: a new area to explore.",
                "returning": "Stopped: returning to the lander.", "charging": "Stopped: charging at the lander.",
                "idle": "Stopped.", "noloop": "No safe loop through it on mapped ground. Back to normal exploration."
                }.get(why, "Back to normal exploration.")
        text = f"{head}. {body} {tail}".replace("  ", " ").replace(" .", ".")
        if resurvey and s["there"] and s["dur"] < 10.0 and why not in ("time",) and not found:
            reason = {"new command": "a new command came in", "held": "holding for the operator",
                      "battery": f"battery {rover.battery * 100:.0f}%, returning to the lander",
                      "replaced": "a new area to explore", "resume": "resuming normal exploration",
                      "returning": "returning to the lander", "charging": "charging at the lander",
                      "noloop": "no safe loop through it on mapped ground"}.get(why, why)
            text = f"Stopped re-surveying {label} after {s['dur']:.0f} s: {reason}."
        ref = Ref("area", s["x"], s["y"], label, rover=rid, t=t, r=s["r"])
        if fc["reason"] == "operator":
            if not quiet:
                self.say(rid, t, text, "info", reply_to=fc.get("cid"), request_id=fc.get("qid"))
                self._log(t, "discovery" if found else "info", f"{rid} {head[0].lower() + head[1:]}.", rid,
                          at=(s["x"], s["y"]), source="operator")
            self.mem.add_report(ref, text, rid, t, s, mention=why not in ("new command", "replaced", "resume", "held"))
            return
        if quiet:
            return
        what = ("found " + ", ".join(found)) if found else "nothing new"
        self._log(t, "discovery" if found else "info",
                  f"{rid} done looking around {label}: {s['dur']:.0f} s, mapped {s['mapped']} m2 of new ground, "
                  f"{what}.", rid, at=(s["x"], s["y"]))

    # ================================================================== 9. protocol v8: deciding between places
    @staticmethod
    def _rank_places(cands: list, place: str, frm: tuple) -> list:
        """The rules' order: unexplored gaps by size and distance (gaps.score), anything else nearest first."""
        if place == "gaps":
            return sorted(cands, key=lambda p: -G.score({"m2": p.size or 0.0, "x": p.x, "y": p.y}, frm[0], frm[1]))
        return sorted(cands, key=lambda p: math.hypot(p.x - frm[0], p.y - frm[1]))

    def _no_gaps_text(self, t: float) -> str:
        f = self.fleet
        allg = f.refresh_gaps(t, max_age=1.0)
        stuck = sum(1 for g in allg if not g["frontiers"])
        n_fr = len(f.usable_frontiers(t))
        head = "I don't see any unexplored gaps inside the map that a rover can reach"
        head += f" ({stuck} more sit{'s' if stuck == 1 else ''} in steep ground nobody can drive to)." if stuck else "."
        tail = (f" The unknown ground left is beyond the {n_fr} frontier{'s' if n_fr != 1 else ''} at the edge of the "
                f"map; the rovers are exploring those.") if n_fr else " Exploration is nearly done."
        return head + tail

    def _place_candidates(self, r, shown: list) -> list:
        frm = (r.x, r.y)
        return [{"id": f"p{k + 1}", "kind": "place", "summary": P.option_label(p, frm)} for k, p in enumerate(shown)]

    def _place_job(self, r, shown: list, kind: str, n: int, inv: bool, t: float, src: str, prefix: str, sample: bool,
                   explore: bool, side, text) -> dict:
        return {"rover": r.id, "cands": {f"p{k + 1}": p for k, p in enumerate(shown)}, "place": kind, "n": n,
                "investigate": bool(inv), "cid": self._reply_to, "src": src, "sample": sample, "explore": bool(explore),
                "side": side, "prefix": prefix, "t": t, "wall": self.clock(), "ticket": None, "qid": None,
                "text": str(text or "")[:120]}

    def _submit_place(self, r, job: dict, shown: list, t: float) -> None:
        """Ask Gemma to pick one of the places (a choice request with kind "place" candidates)."""
        perception = dict(self._perception(r), operator_asked=job["text"] or "go there")
        job["ticket"] = self.planner.submit_choice(r.id, t, perception, self._place_candidates(r, shown), default="p1")
        self.choices.pop(r.id, None)       # this rover's goal question would replace it in Gemma's queue
        self.last_ask[r.id] = t
        self.place_jobs.append(job)
        if self.wait_for_gemma:
            self._wait(job["ticket"])
            self._poll_place_jobs(t)

    def _decide_place(self, r, cands: list, kind: str, inv: bool, t: float, src: str, prefix: str, sample: bool,
                      explore: bool, side, text) -> None:
        """Autonomous: several places match and nothing picks one: Gemma picks (the rules' order otherwise: the
        best gap, or the nearest place). The ack says which one and who picked it."""
        shown = self._rank_places(cands, kind, (r.x, r.y))[:P.MAX_OPTIONS]
        job = self._place_job(r, shown, kind, len(cands), inv, t, src, prefix, sample, explore, side, text)
        if self.planner.online and len(shown) > 1:
            self._submit_place(r, job, shown, t)
        else:
            self._resolve_place_job(job, None, t)

    def _ask_place_suggestion(self, r, qid: str, text, t: float) -> None:
        """Supervised: Gemma's pick for an open choose_place request, shown as a suggestion when it comes."""
        q = self.requests.get(qid)
        if q is None or not self.planner.online:
            return
        shown = [q["job"]["cands"][o[0]] for o in q["options"] if o[0] in q["job"]["cands"]]
        job = self._place_job(r, shown, q["job"]["place"], len(shown), False, t, "gemma", "", False, False, None, text)
        job["qid"] = qid
        self._submit_place(r, job, shown, t)

    def _poll_place_jobs(self, t: float) -> None:
        for job in list(self.place_jobs):
            res = self.planner.poll(job["ticket"])
            if res is None and self.clock() - job["wall"] < self.planner.cfg.timeout_s + 2.0:
                continue
            self.place_jobs.remove(job)
            if job["qid"] is not None:
                self._place_suggestion(job, res)
            else:
                self._resolve_place_job(job, res, t)

    def _place_suggestion(self, job: dict, res) -> None:
        q = self.requests.get(job["qid"])
        if q is None or res is None or not res.ok or res.confidence < CONF_FLOOR or res.choice not in job["cands"]:
            return
        q["suggest"], q["suggest_conf"] = res.choice, res.confidence
        label = self._labels(q).get(res.choice, res.choice)
        q["question"] += f" Gemma suggests: {label[:1].lower() + label[1:]} ({res.confidence:.2f})."

    def _resolve_place_job(self, job: dict, res, t: float) -> None:
        r = self._rover(job["rover"])
        if r is None:
            return
        ok = res is not None and res.ok and res.choice in job["cands"]
        what = "the best gap for size and distance" if job["place"] == "gaps" else "the nearest one"
        if ok and res.confidence >= CONF_FLOOR:
            choice, by, conf, why = res.choice, "gemma", res.confidence, _clean(res.reason).strip("'\" ")
        elif ok:
            choice, by, conf, why = "p1", "rules", res.confidence, f"Gemma unsure ({res.confidence:.2f}): {what}"
        else:
            choice, by, conf, why = "p1", "rules", None, f"{what} (Gemma offline or no answer)" if len(
                job["cands"]) > 1 else "the only one"
        p = job["cands"][choice]
        label = P.option_label(p, (r.x, r.y))
        noun = P.count_noun(job["place"], job["n"])
        who = f"Gemma picked it, {conf:.2f}" if by == "gemma" else "my rules picked it"
        lab = label[:1].lower() + label[1:]
        job["announce"] = job["prefix"] + (f"I know {noun}; going for the {lab} ({who}). " if job["n"] > 1 else "")
        self._decided(r, "place", f"Which of {noun} for '{job['text']}'", list(job["cands"]), choice, by, conf, why,
                      (p.x, p.y), t, label=label, chat=False)
        prev = (self._reply_to, self._request_id)
        self._reply_to = job["cid"]
        try:
            self._apply_place_choice(r, {"id": None, "job": job}, choice, t)
        finally:
            self._reply_to, self._request_id = prev

    # ================================================================== 10. protocol v8: autonomy mode
    def _cmd_mode(self, cmd: dict, t: float, src: str, c: dict) -> None:
        mode = cmd.get("mode")
        text = str(c.get("text") or "").lower()
        once = mode == "autonomous" and self.supervised and self.requests and not re.search(
            r"\b(?:from now on|mode|always|autonom\w*|switch|any ?more|from here on|stop asking|don'?t ask|robots?)\b",
            text)
        if once:                            # "you decide": settle the open questions now, stay supervised
            n = len(self.requests)
            self._reply("fleet", t, f"OK, deciding the {n} open question{'s' if n != 1 else ''} myself. Still in "
                                    f"supervised mode: say 'you decide from now on' to switch.", "ack", src)
            self._decide_open(t)
            return
        self.set_mode(mode, t, src=src, chat=True)

    def set_mode(self, mode: str, t: float | None = None, src: str = "rules", chat: bool = False) -> str:
        """Protocol v8: "autonomous" or "supervised" (POST /api/mode, or the chat). Switching to autonomous decides
        every open question at once. Returns "ok", or "bad" for an unknown mode."""
        if mode not in MODES:
            return "bad"
        t = self.sim.t if t is None else t
        frm = "fleet"
        if self.fleet.mode == mode:
            text = ("Already in autonomous mode: the rovers decide on their own and log every decision."
                    if mode == "autonomous" else "Already in supervised mode: I ask you before every decision.")
            if chat:
                self._reply(frm, t, text, "ack", src)
            return "ok"
        self.fleet.mode = mode
        n = len(self.requests)
        if mode == "autonomous":
            text = ("Autonomous mode: the rovers decide on their own (Gemma, or the rules when Gemma is offline) and "
                    "log every decision here and in the event log." + (f" Deciding the {n} open question"
                                                                        f"{'s' if n != 1 else ''} now." if n else ""))
        else:
            text = ("Supervised mode: I'll ask you before every decision (anomalies, lifeform sightings, which place), "
                    "with Gemma's pick marked as a suggestion. A rover holds while it waits, unless staying is the "
                    "danger.")
        if chat:
            self._reply(frm, t, text, "ack", src)
        else:
            self.say(frm, t, text, "info")
        self._log(t, "info", f"Autonomy mode: {mode} (operator).", None, source="operator")
        if mode == "autonomous" and n:
            self._decide_open(t)
        return "ok"

    def _decide_open(self, t: float) -> int:
        """Every open question is decided by its rover now (Gemma's look or suggestion, else the rules) and logged."""
        n = 0
        for q in list(self.requests.values()):
            q, rover = self._close(q["id"], "decided", t)
            n += 1
            if rover is None:
                continue
            job = q["job"]
            if q["kind"] == "choose_place":
                cands = job["cands"]
                choice = q.get("suggest") if q.get("suggest") in cands else next(iter(cands))
                by = "gemma" if q.get("suggest") in cands else "rules"
                conf = q.get("suggest_conf") if by == "gemma" else None
                p = cands[choice]
                why = "Gemma's suggestion" if by == "gemma" else "the first option (nearest, or the best gap)"
                self._decided(rover, "place", f"Which of {P.count_noun(job['place'], len(cands))}", list(cands), choice,
                              by, conf, why, (p.x, p.y), t, label=P.option_label(p, (rover.x, rover.y)), force=True,
                              request_id=q["id"], reply_to=job.get("cid"))
                self._apply_place_choice(rover, q, choice, t)
            elif q["kind"] == "rescue":
                resc = self.sim.rescue
                rec = resc.active(job["rescue"])
                if rec is None or rec.get("opts") is None:
                    continue
                order = resc.rules_order(rec, self._rover(job["rescue"]))
                sug = q.get("suggest")
                choice = sug if sug in order else order[0]
                by = "gemma" if sug in order else "rules"
                conf = q.get("suggest_conf") if by == "gemma" else None
                why = "Gemma's suggestion" if by == "gemma" else resc._rules_why(rec, order)
                rec["qid"] = None
                resc._decided(rec, self._rover(job["rescue"]), rover, order, choice, by, conf, why, t)
                resc.apply_plan(rec, choice, t, by=by)
            elif q["kind"] == "lifeform":
                choice, by, conf, why = self._life_choice(job.get("res"))
                self._decided(rover, "lifeform", self._life_question(job["ask"]).split(")")[0] + ")", [o for o, _ in
                              LIFE_OPTIONS], choice, by, conf, why, job.get("at", (rover.x, rover.y)), t, force=True,
                              request_id=q["id"])
                self._apply_life(rover, choice, t, q)
            else:
                choice, by, conf, why = self._anomaly_choice(q["kind"], job.get("res"))
                phrase, mode = self._apply_action(rover, q["kind"], choice, t, job)
                if mode == "override":
                    choice, by = RULE_REACTION[q["kind"]], "safety rule"
                self._decided(rover, q["kind"], self._anomaly_text(q["kind"], job["facts"]),
                              [o for o, _ in QUESTION_OPTIONS[q["kind"]]], choice, by, conf, why, job["at"], t,
                              force=True, request_id=q["id"])
        return n

    # ================================================================== 11. protocol v8: fleet priorities
    _LATER = {"reboot": ("joins after its reboot", "join after their reboots"),
              "hold": ("is holding (say resume)", "are holding (say resume)"),
              "task": ("finishes its current task first", "finish their current tasks first"),
              "dir": ("keeps to the direction you gave it first", "keep to the directions you gave them first")}

    def _joins_later(self, r, t: float):
        """Why a rover joins a fleet priority later (its own task, charging, a fault), or None if it joins now."""
        if r.state == "fault":
            return "reboot"
        if self._busy_reason(r) is not None or (r.state == "charging" and r.battery < FREE_CHARGE):
            return f"joins after charging (battery {r.battery * 100:.0f}%)"
        if r.held:
            return "hold"
        if r.op_goal is not None or self.go_jobs.get(r.id) or (r.focus is not None and r.focus["reason"] == "operator"):
            return "task"
        if r.dir_bias is not None and t < r.dir_bias[2]:
            return "dir"
        return None

    def _later_text(self, t: float) -> str:
        """'r1 and r3 finish their current tasks first; r2 joins after charging (battery 27%)', or ''."""
        groups: dict = {}
        for r in self.sim.rovers:
            why = self._joins_later(r, t)
            if why:
                groups.setdefault(why, []).append(r.id)
        out = []
        for why, ids in groups.items():
            one, many = self._LATER.get(why, (why, why))
            out.append(f"{' and '.join(ids)} {one if len(ids) == 1 else many}")
        return "; ".join(out)

    def _wake(self, r, t: float) -> None:
        """A rover that joins a priority picks a new goal now."""
        if r.focus is not None and r.focus["reason"] == "interest":
            r.end_focus(t, "replaced", quiet=True)
        r.pending_interest = None
        r.sticky_goal = None
        r.next_goal_eval = r.next_goal_try = t
        self.choices.pop(r.id, None)

    def _frontiers_where(self, t: float, p: dict) -> list:
        f = self.fleet
        kind = p["kind"]
        out = []
        for fr in f.usable_frontiers(t):
            x, y = fr[0], fr[1]
            if kind == "direction" and f.in_direction(x, y, p["direction"]):
                out.append(fr)
            elif kind == "water" and f.water_zones and f.water_zone_gap(x, y) < PRIORITY_WATER_M:
                out.append(fr)
            elif kind == "water" and not f.water_zones and p.get("low_h") is not None and \
                    float(f.map.h.ravel()[fr[3]]) <= p["low_h"]:
                out.append(fr)
        return out

    def _cmd_prioritize(self, cmd: dict, t: float, src: str, c: dict) -> None:
        f = self.fleet
        kind = cmd.get("priority")
        old = f.priority
        if kind == "none":
            if old is None:                  # nothing to drop: "back to normal" means normal exploration
                self._cmd_resume(None, t, src)
                return
            f.priority = None
            self._reply("fleet", t, f"Back to normal exploration: dropped the priority on {old['label']}. Every rover "
                                    f"picks its own frontiers again.", "ack", src)
            self._log(t, "info", f"Fleet priority on {old['label']} cleared (operator).", None, source="operator")
            return
        f.priority = None
        rep = f"Replacing the priority on {old['label']}. " if old is not None else ""
        p = {"kind": kind, "since": t, "cid": self._reply_to, "assign": {}, "labs": {}, "direction": None}
        later = self._later_text(t)
        free = [r for r in self.sim.rovers if self._joins_later(r, t) is None]
        if kind == "gaps":
            gaps = f.explorable_gaps(t, max_age=0.5)
            if not gaps:
                self._reply("fleet", t, rep + self._no_gaps_text(t) + " Nothing to prioritize.", "info", "rules")
                return
            p.update(label="the unexplored gaps inside the map", count=len(gaps))
            f.priority = p
            plan = self._assign_gaps(p, gaps, free, t)
            groups: dict = {}
            for rid, g, d in plan:
                groups.setdefault(g["lab"], (g, []))[1].append((rid, d))
            bits = []
            for k, (g, who) in enumerate(groups.values()):
                ids = " and ".join(rid for rid, _ in who)
                dist = " and ".join(f"{d:.0f}" for _, d in who)
                verb = ("takes" if len(who) == 1 else "take") if k == 0 else ""
                bits.append(f"{ids} {verb + ' ' if verb else ''}the one at ({g['x']:.0f}, {g['y']:.0f}) "
                            f"({g['m2']:.0f} m2, {dist} m away)")
            stuck = sum(1 for g in f.gaps if not g["frontiers"])
            head = (f"Prioritizing {len(gaps)} unexplored gap{'s' if len(gaps) != 1 else ''} inside the map"
                    + (f" ({stuck} more sit{'s' if stuck == 1 else ''} in steep ground nobody can reach)" if stuck else ""))
            text = head + (": " + ", ".join(bits) + "." if bits else ".")
        elif kind == "direction":
            d = cmd.get("direction") or "north"
            p.update(direction=d, label=f"the {d} of the map")
            frs = self._frontiers_where(t, p)
            if not frs:
                self._reply("fleet", t, rep + f"No reachable frontier is left to the {d} (seen from the lander), so "
                                              f"there is nothing to prioritize there.", "info", "rules")
                return
            p["count"] = len(frs)
            f.priority = p
            bits = self._nearest_bits(free, frs)
            text = (f"Prioritizing the {d}: {len(frs)} reachable frontier{'s' if len(frs) != 1 else ''} that way (seen "
                    f"from the lander)" + (": " + ", ".join(bits) + "." if bits else "."))
        elif kind == "water":
            if f.water_zones:
                n_b = len(P.water_bodies(f.water_zones))
                p.update(label="the ground next to the known water")
                frs = self._frontiers_where(t, p)
                if not frs:
                    self._reply("fleet", t, rep + f"The ground around the {n_b} known bod{'ies' if n_b != 1 else 'y'} "
                                                  f"of water is already mapped (no frontier within "
                                                  f"{PRIORITY_WATER_M:.0f} m). Nothing to prioritize; we keep looking "
                                                  f"for more water as we explore.", "info", "rules")
                    return
                head = (f"Prioritizing water: {len(frs)} frontier{'s' if len(frs) != 1 else ''} within "
                        f"{PRIORITY_WATER_M:.0f} m of the {n_b} known bod{'ies' if n_b != 1 else 'y'} of water")
            else:
                hs = f.map.h[f.map.known]
                p.update(label="low ground, where water collects", low_h=float(np.percentile(hs, 25)) if hs.size else 0.0)
                frs = self._frontiers_where(t, p)
                head = (f"No water found yet, so I'm prioritizing low ground, where liquid collects: {len(frs)} frontier"
                        f"{'s' if len(frs) != 1 else ''} below {p['low_h']:.1f} m. Once we find water, its edge comes "
                        f"first")
            p["count"] = len(frs)
            f.priority = p
            bits = self._nearest_bits(free, frs)
            text = head + (": " + ", ".join(bits) + "." if bits else ".")
        else:                                # sampling
            waiting = [k for k, ts in enumerate(f.targets) if ts.status in ("found", "investigating")]
            if not waiting:
                done = sum(1 for ts in f.targets if ts.status == "sampled")
                self._reply("fleet", t, rep + f"No found target is waiting to be sampled ({done} sampled so far). "
                                              f"Nothing to prioritize.", "info", "rules")
                return
            p.update(label="sampling the found targets", count=len(waiting))
            f.priority = p
            bits = self._assign_samples(free, t)
            text = (f"Prioritizing sampling: {len(waiting)} found target{'s' if len(waiting) != 1 else ''} wait"
                    + (": " + ", ".join(bits) if bits else "") + ". Rovers now fetch found targets from up to 90 m.")
        for r in free:
            self._wake(r, t)
        text = rep + text + (f" {later}." if later else "") + " Say 'back to normal' to stop."
        self._reply("fleet", t, text, "ack", src)
        self._log(t, "info", f"Fleet priority: {p['label']} ({p['count']} left, operator).", None, source="operator")

    def _nearest_bits(self, free: list, frs: list) -> list:
        out = []
        for r in free:
            fr = min(frs, key=lambda fr_: math.hypot(fr_[0] - r.x, fr_[1] - r.y))
            out.append(f"{r.id} starts with one {math.hypot(fr[0] - r.x, fr[1] - r.y):.0f} m "
                       f"{compass(fr[0] - r.x, fr[1] - r.y)}")
        return out

    def _assign_gaps(self, p: dict, gaps: list, rovers: list, t: float) -> list:
        """Each rover its best gap (size and distance), a different one while there are enough. [(rid, gap, m)]."""
        by_lab = {g["lab"]: g for g in gaps}
        pairs = sorted(((G.score(g, r.x, r.y), r.id, g["lab"]) for r in rovers for g in gaps), reverse=True)
        labs, taken = {}, set()
        for _s, rid, lab in pairs:            # greedy: the best (rover, gap) pairs first, one gap per rover
            if rid in labs or lab in taken:
                continue
            labs[rid] = lab
            taken.add(lab)
        for r in rovers:                       # more rovers than gaps: the rest share their best one
            if r.id not in labs:
                labs[r.id] = max(gaps, key=lambda g: G.score(g, r.x, r.y))["lab"]
        plan = []
        for r in rovers:
            g = by_lab[labs[r.id]]
            p["labs"][r.id], p["assign"][r.id] = g["lab"], (g["x"], g["y"])
            plan.append((r.id, g, math.hypot(g["x"] - r.x, g["y"] - r.y)))
        return plan

    def _assign_samples(self, rovers: list, t: float) -> list:
        """Free rovers with a free tube and the battery start on the nearest found target nobody is sampling."""
        f = self.fleet
        out = []
        for r in rovers:
            if r.target_idx is not None:
                out.append(f"{r.id} is already on {f.targets[r.target_idx].spec.id}")
                continue
            if r.state != "exploring" or r.free_tubes() <= 0:
                if r.free_tubes() <= 0:
                    out.append(f"{r.id} caches its full tubes first")
                continue
            best = None
            for k, ts in enumerate(f.targets):
                if ts.status != "found" or ts.investigator is not None or not r.can_investigate(t, ts):
                    continue
                d = math.hypot(ts.spec.x - r.x, ts.spec.y - r.y)
                need = r.energy_for_distance(1.3 * d + 1.3 * math.hypot(ts.spec.x, ts.spec.y)) + r.battery_margin + 0.05
                if r.battery > need and (best is None or d < best[1]):
                    best = (k, d)
            if best is None:
                continue
            k, d = best
            ts = f.targets[k]
            ts.status, ts.investigator = "investigating", r.id
            r.start_investigation(t, k)
            out.append(f"{r.id} takes {ts.spec.id} ({KIND_LABEL[ts.spec.kind]}, {d:.0f} m)")
            self._log(t, "info", f"{r.id} heading to sample {ts.spec.id} (fleet priority: sampling), {d:.0f} m.", r.id,
                      at=(ts.spec.x, ts.spec.y), source="operator")
        return out

    def _priority_tick(self, t: float) -> None:
        """Every few seconds while a fleet priority is set: counts, reassignment, and the end ("Done: ...")."""
        f = self.fleet
        p = f.priority
        kind, done = p["kind"], None
        if kind == "gaps":
            gaps = f.explorable_gaps(t)
            p["count"] = len(gaps)
            p["places"] = [{"x": g["x"], "y": g["y"], "r": g["r"]} for g in gaps[:24]]
            if not gaps:
                done = (f"Done: no unexplored gaps left inside the map that a rover can reach "
                        f"({f.map.explored_pct():.0f}% of the map explored). Back to normal exploration.")
            else:
                self._reassign_gaps(p, gaps, t)
        elif kind in ("direction", "water"):
            if kind == "water" and f.water_zones and p.get("low_h") is not None:
                p["low_h"], p["label"] = None, "the ground next to the known water"
                self._log(t, "info", "Fleet priority: water found, now the ground next to it comes first.", None)
            p["count"] = len(self._frontiers_where(t, p))
            if p["count"] == 0 and (kind == "direction" or f.water_zones):
                done = (f"Done with the {p['direction']}: no reachable frontier left that way." if kind == "direction"
                        else "Done: no unexplored ground left next to the known water.") + " Back to normal exploration."
        else:
            p["count"] = sum(1 for ts in f.targets if ts.status in ("found", "investigating"))
            if p["count"] == 0:
                done = "Done: every found target is sampled. Back to normal exploration."
        if done:
            f.priority = None
            self.say("fleet", t, done, "info", reply_to=p.get("cid"))
            self._log(t, "info", f"Fleet priority on {p['label']} done.", None)

    def _reassign_gaps(self, p: dict, gaps: list, t: float) -> None:
        """Keep each rover on its gap while it is there; a finished gap's rover takes the best one left; a rover
        that became free (charged, task done) joins."""
        f = self.fleet
        labs, assign = {}, {}
        for rid, (x, y) in p["assign"].items():
            lab = G.gap_at(f.gap_grid, f.map, x, y)
            g = next((g for g in gaps if g["lab"] == lab), None) or next(
                (g for g in gaps if math.hypot(g["x"] - x, g["y"] - y) <= g["r"] + 4.0), None)
            if g is not None:
                labs[rid], assign[rid] = g["lab"], (g["x"], g["y"])
        for r in self.sim.rovers:
            if r.id in labs or self._joins_later(r, t) is not None:
                continue
            busy = set(labs.values())
            pool = [g for g in gaps if g["lab"] not in busy] or gaps
            g = max(pool, key=lambda g_: G.score(g_, r.x, r.y))
            labs[r.id], assign[r.id] = g["lab"], (g["x"], g["y"])
            was = p["assign"].get(r.id)
            what = f"finished the gap at ({was[0]:.0f}, {was[1]:.0f}); next" if was else "joins the gaps priority"
            self._log(t, "info", f"{r.id} {what}: the one at ({g['x']:.0f}, {g['y']:.0f}) ({g['m2']:.0f} m2).", r.id,
                      at=(g["x"], g["y"]), source="operator")
            self._wake(r, t)
        p["labs"], p["assign"] = labs, assign

    # ---- references in questions ("what did you find there?")
    def _question_ref(self, text: str):
        """The place a question refers back to, from the chat memory, or None."""
        low = " ".join(str(text or "").lower().split())
        kind, span = _place_words(low)
        if kind is not None and kind != "lander" and (_MENTIONED_AFTER_RE.match(low[span[1]:])
                                                      or _MENTIONED_BEFORE_RE.search(low[:span[0]])):
            ref = self.mem.last_of(kind)
            if ref is not None:
                return ref
        if _QREF_RE.search(low):
            return self.mem.there
        m = _NEAR_ROVER_Q_RE.search(low)          # protocol v8: "anything interesting around rover 2?"
        if m:
            rid = f"r{m.group(1) or m.group(2)}" if (m.group(1) or m.group(2)) else None
            ro = self._rover(rid) if rid else None
            if ro is not None:
                return Ref("area", float(ro.x), float(ro.y), f"{ro.id}'s position", rover=ro.id, t=self.sim.t)
            if rid is None:
                return Ref("lander", 0.0, 0.0, "the lander", t=self.sim.t)
        if kind is not None and kind not in ("lander", "gaps") and _AREA_PREP_RE.search(low[:span[0]]):
            ref = self.mem.last_of(kind)          # "anything near the lake?", "what's at the vent?"
            if ref is not None:
                return ref
            cands = P.candidates(self.sim, kind, None, self.sim.t)
            if cands:
                ro = self._rover(self.mem.rover)
                frm = (ro.x, ro.y) if ro is not None else (0.0, 0.0)
                p = min(cands, key=lambda c: math.hypot(c.x - frm[0], c.y - frm[1]))
                rr = 0.5 * (p.size or 0.0) + 10.0 if p.kind == "water" else None
                return Ref(p.kind, float(p.x), float(p.y), p.dest(), id=p.id, t=self.sim.t, place=p, r=rr)
        return None

    def _why_there(self, ref: Ref, rover) -> str:
        """Why this rover (or the fleet) went to the place the chat talked about: the operator's command that sent
        it there, or its own reason (protocol v8)."""
        where = ref.label if ref.label.startswith("(") else f"{ref.label} at ({ref.x:.0f}, {ref.y:.0f})"
        cmd = self.mem.command
        cr = cmd.get("ref") if cmd else None
        if cr is not None and math.hypot(cr.x - ref.x, cr.y - ref.y) < 12.0 and (
                rover is None or cmd.get("rid") in (None, rover.id) or cr.rover == rover.id):
            who = "me" if rover is not None else (cr.rover or "a rover")
            return f"why {where}: you asked {who} to: '{cmd['text']}'"
        if rover is not None:
            s = rover.mem.focus_near(ref.x, ref.y, 15.0)
            if s is not None and s.get("reason") == "interest":
                return f"why {where}: I found {s.get('what') or 'something new'} there and looked around it for more"
            return f"why {where}: " + why_text(rover)
        return f"why {where}: the rover that went there chose it for its own goal"

    def _mode_facts(self) -> list:
        f = self.fleet
        mode = ("the fleet is in autonomous mode: the rovers decide on their own and log every decision"
                if f.mode == "autonomous" else "the fleet is in supervised mode: the operator is asked before every "
                                               "decision, with Gemma's pick as a suggestion")
        p = f.priority
        prio = (f"fleet priority: {p['label']} ({p.get('count', 0)} left)" if p is not None
                else "no fleet priority is set")
        return [mode, prio]

    def _gap_fact(self, t: float) -> str | None:
        gaps = self.fleet.explorable_gaps(t)
        n_fr = len(self.fleet.usable_frontiers(t))
        if not gaps:
            return f"{n_fr} frontier{'s' if n_fr != 1 else ''} left at the edge of the known map, no unexplored gaps inside it"
        g = max(gaps, key=lambda g_: g_["m2"])
        return (f"{n_fr} frontier{'s' if n_fr != 1 else ''} left at the edge of the known map, and {len(gaps)} unexplored "
                f"gap{'s' if len(gaps) != 1 else ''} inside it, the biggest about {g['m2']:.0f} m2 at "
                f"({g['x']:.0f}, {g['y']:.0f})")

    @staticmethod
    def _sentence(line: str) -> str:
        s = line[:1].upper() + line[1:]
        if len(s) > 355:
            s = s[:352].rsplit(" ", 1)[0] + "..."
        return s if s.endswith((".", "!", "?")) else s + "."
