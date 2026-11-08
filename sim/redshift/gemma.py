"""Gemma advisory layer: a small LOCAL model as a slow second opinion for the rule-based rovers.

Two tiers
---------
Tier 1, the rules (rover.py, fleet.py, planner.py): 5 Hz, hand-written logic plus A*. They keep
the rover safe and always have an answer.

Tier 2, Gemma (this module): a small local vision-language model (Gemma 4 E2B served by Ollama,
or Gemma 3 4B on older Ollama) on ONE background thread beside the rules. It
  (a) picks between candidate actions the rules generated (frontiers, a target, return, wait),
      about every 20 s per rover, with a short reason that quotes sensor readings,
  (b) interprets anomalies (new obstacle, wheel slip, low visibility) from the camera image plus
      the sensor numbers: a classification, a confidence and one action from a fixed list, or
      "ask_operator" when it is not sure (confidence below 0.55), and
  (c) turns the operator's typed messages ("rover 2 go to 10, -5", "can we investigate the lake
      for signs of life": go_to_place, protocol v6) into structured commands, or "unclear" with a
      clarifying question. A regex parser does the same when Gemma is
      offline or answers something invalid, and
  (d) describes something the LiDAR tracker saw moving (camera image plus track numbers): a
      lifeform or not, a short label ("possible lifeform, six legs, about 1.2 m"), what to do
      (approach to observe, observe from here, ignore) and a confidence, and
  (e) answers the operator's questions ("rover 2, what have you found?"): the command step says
      the message is a question (with its topic), then Gemma writes a 1 to 3 sentence reply from a
      fact sheet the sim built from the rover's own memory (answers.py). Every number in the reply
      must appear in that fact sheet and every id must be one it lists, or the rule-based answer
      is used instead.

The rules never wait on Gemma. submit_choice(), submit_interpretation(), submit_command() and
poll() return at once (tens of microseconds). If Gemma is disabled, offline, still loading, slow
(no answer within GEMMA_TIMEOUT_S) or answers something invalid, the result is a fallback
(ok=False, source="fallback"): for choices and interpretations it carries the caller's own
default (the rule-based choice stands); for commands it carries the regex parser's command.

Interface (GEMMA.md shows where the rovers call it)
---------
    from redshift.gemma import get_planner
    gp = get_planner()                                   # one shared planner, reads GEMMA_* env vars
    tk = gp.submit_choice("r1", t, perception, candidates, default="f1")
    tk = gp.submit_interpretation("r1", t, perception, anomaly, jpeg_or_None,
                                  ["reroute", "back_out", "proceed_slowly", "wait"])
    tk = gp.submit_command(t, "rover 2 go to 10, -5", fleet_context)
    tk = gp.submit_description("r3", t, perception, observation, jpeg_or_None)
    tk = gp.submit_answer("r2", t, "what have you found?", "finds", fact_lines, default=rule_text)
    res = gp.poll(tk)          # None while pending, then a Result; never blocks
    gp.stats()                 # {"model", "online", "state", "calls", "ok", "fallbacks", "avg_ms", ...}

Validation (every answer, before the caller sees it)
----------
- the answer must be JSON with every field; choice must be one of the candidate ids, action one
  of the options, classification one of the classes, confidence a number from 0 to 1;
- the reason (cut to 25 words) must contain at least one number that appears in the input the
  model was given (a sensor reading or a number from a candidate summary; 0.41 may be written
  as 41%). An answer that cites nothing it saw is rejected, so every accepted decision is
  traceable to perceived input;
- an interpretation with confidence below GEMMA_MIN_CONFIDENCE (0.55) becomes action
  "ask_operator" (the model's own pick is kept in model_action, and Result.question holds a
  short question for the human). A command below it becomes "unclear" with a question;
- command coordinates must come from the message or the fleet data (no invented places), rover
  and target ids must exist;
- an answer (kind "answer") is 1 to 3 sentences, at most 60 words; every number in it must match a
  number in the fact lines it was given (simple rounding allowed: 7.83 as 7.8, 1904 as 1,900, 0.64
  as 64%) and every id (r2, t4, k1, ...) must appear in them.
Rejected answers become fallback results with the reason in Result.error.

Env vars
--------
GEMMA_ENABLED      1          0 = off: no threads, every submit returns a fallback at once
GEMMA_URL          http://host.docker.internal:11434   Ollama; http://localhost:11434 is tried next
GEMMA_MODEL        gemma4:e2b used if installed, else the first installed of GEMMA_FALLBACK_MODELS
GEMMA_FALLBACK_MODELS gemma4:e2b,gemma4:e4b,gemma3:4b
GEMMA_TIMEOUT_S    4          an answer later than this is dropped (fallback)
GEMMA_BUDGET_MS    3000       stated latency budget per call (stats count calls over it)
GEMMA_FAKE         0          1 = FakeBackend: deterministic rules + fake latency, no model needed
GEMMA_FAKE_LATENCY_MS 500     latency of the fake backend
GEMMA_NUM_PREDICT  96         max output tokens per answer (commands use at least 128)
GEMMA_THINK        0          1 = let Gemma 4 think before answering (slower; needs more tokens)
GEMMA_KEEP_ALIVE   30m        how long Ollama keeps the model loaded between calls
GEMMA_MIN_CONFIDENCE 0.55     interpretations below this become ask_operator, commands unclear

Standard library only (urllib, json, threading, base64). Proxies are bypassed: Ollama is local.
"""
from __future__ import annotations

import base64
import colorsys
import http.client
import itertools
import json
import math
import os
import re
import socket
import threading
import time
import urllib.error
import urllib.request
from collections import OrderedDict, deque
from dataclasses import asdict, dataclass, field

__all__ = [
    "ACTIONS", "CANDIDATE_KINDS", "CLASSES", "COMMANDS", "DIRECTIONS", "GemmaConfig", "Result", "Request",
    "GemmaPlanner", "OllamaBackend", "FakeBackend", "GemmaError", "GemmaTimeout", "GemmaOffline",
    "make_choice_request", "make_interpretation_request", "make_command_request", "make_description_request",
    "make_answer_request", "build_prompt", "DESCRIBE_CLASSES", "DESCRIBE_ACTIONS", "TOPICS", "question_intent",
    "question_topic", "wants_each", "PLACES", "QUALIFIERS", "parse_place", "near_reference",
    "build_schema", "validate", "run_request", "parse_command", "cited_numbers", "render_readings",
    "colour_name", "latency_summary", "estimate_tokens", "get_planner", "planner_from_env", "close_planner",
]

DEFAULT_URL = "http://host.docker.internal:11434"    # Ollama on the host, seen from Docker Desktop
LOCAL_URL = "http://localhost:11434"                 # Ollama when the sim runs natively
DEFAULT_MODEL = "gemma4:e2b"
FALLBACK_MODELS = ("gemma4:e2b", "gemma4:e4b", "gemma3:4b")

CANDIDATE_KINDS = ("frontier", "target", "return", "wait")
ACTIONS = ("reroute", "back_out", "proceed_slowly", "wait", "ask_operator")
CLASSES = ("rock", "steep_slope", "soft_ground", "dust", "liquid", "other_rover", "crater_edge", "clear", "unknown")
DESCRIBE_CLASSES = ("lifeform", "rock", "other_rover", "dust", "unknown")
DESCRIBE_ACTIONS = ("approach", "observe", "ignore", "ask_operator")
MAX_LABEL_WORDS = 10
COMMANDS = ("go_to", "go_to_place", "explore_direction", "sample_target", "return_home", "hold", "resume",
            "avoid_area", "repeat", "question", "unclear")
TOPICS = ("status", "history", "exploration", "finds", "samples", "battery", "hazards", "lifeforms", "water",
          "location", "why", "unknown")
DIRECTIONS = ("north", "south", "east", "west", "north-east", "north-west", "south-east", "south-west")
# protocol v6: commands that name a place the fleet knows ("go to the water source"); v7: "there" is the place
# the chat talked about last (resolved by the advisor's chat memory), "mentioned" picks the one the chat named
PLACES = ("water", "spire", "mineral", "ice", "vent", "trace", "target", "lifeform", "hazard", "lander", "there")
QUALIFIERS = ("nearest", "farthest", "biggest", "smallest", "newest") + DIRECTIONS + ("near_rover", "ordinal",
                                                                                      "mentioned", "none")

MAX_REASON_WORDS = 25
MAX_CANDIDATES = 6
MAX_STR = 160          # longest string kept in a prompt (candidate summaries, anomaly details)
MAX_LIST = 12          # longest list kept in a prompt (long arrays such as LiDAR points are dropped)
MAX_TEXT = 300         # longest operator message
SKIP_KEYS = {"scan", "path", "image", "images", "jpeg"}
MAX_RESULTS = 512      # results kept for poll(); older tickets are forgotten
MAX_LATENCIES = 200    # window for avg / p95
COMMAND_NUM_PREDICT = 160
ANSWER_NUM_PREDICT = 128
MAX_ANSWER_WORDS = 60
MAX_ANSWER_CHARS = 360
MAX_ANSWER_SENTENCES = 3
MAX_FACT_LINES = 8
DEFAULT_AVOID_RADIUS = 3.0
MAX_CHAT_LINES = 3     # protocol v7: recent chat lines in a command prompt ...
MAX_CHAT_LINE = 90     # ... each cut to this many characters
MAX_CHAT_REFS = 200    # ... plus one line of resolved references (last place, last list, last rover)
MAX_CONTEXT_LINES = 2  # context lines in an answer prompt (what "there" or "it" means)

SYSTEM_BASE = (
    "You advise an autonomous rover exploring an unknown super-Earth (gravity 2 g, no link to Earth). "
    "Decide only from the rover's own sensor readings listed below. "
)
SYSTEM_CHOICE = SYSTEM_BASE + (
    "Pick the best candidate: explore safely, keep enough battery to return, avoid steep or soft ground. "
    "Reply with JSON only: reason, choice (one candidate id), confidence (0 to 1). "
    "The reason has at most 25 words and must quote at least one reading exactly as written, number and unit, "
    "never a placeholder. Example: "
    '{"reason": "f2: slope max 13 deg is climbable and battery after return 47% is safe.", '
    '"choice": "f2", "confidence": 0.8}'
)
SYSTEM_INTERPRET = SYSTEM_BASE + (
    "An anomaly was flagged. Use the camera image (if any) and the readings. "
    "Classify the cause, pick one option, give a confidence from 0 to 1. "
    "Soft ground or wheel slip means back_out. New rocks on the path mean reroute. "
    "Dust or low visibility means proceed_slowly. "
    "No LiDAR returns plus blue in the camera means liquid: reroute. "
    "If unsure, give a low confidence and choose ask_operator. "
    "Reply with JSON only: classification, reason, action, confidence. "
    "The reason (max 25 words) must quote a reading exactly as written, number and unit. "
    'Example reason: "slip 52% while ground speed 0.05 m/s: the wheels spin in soft ground."'
)
SYSTEM_COMMAND = (
    "Turn an operator message into one command for a rover fleet. "
    "Commands: go_to (x, y); go_to_place (place, target_id, qualifier, ordinal, near_rover, investigate, explore): "
    "a known place, like the water, a vent, the herd, t7; explore_direction (direction); sample_target (target_id); "
    "return_home, hold, resume; avoid_area (x, y, radius); repeat: the last command again (do that again, same "
    "for rover 3); question (topic): asks for information; unclear (ask one short question). Polite requests to "
    "act are commands. rover: an id, all, or none if not named (rover 2 is r2, the orange one the orange rover). "
    "investigate: investigate, survey, look for life. explore: stay and map around it (explore around, look "
    "around). place there: the place talked about last (there, it, the second one); qualifier mentioned: that "
    "lake, the vent you mentioned. A direction without place or coordinates is explore_direction. Use only "
    "numbers and ids from the message or lists, never a rover position; 0 or none if unused. ack: one short "
    "sentence. JSON only."
)

SYSTEM_DESCRIBE = SYSTEM_BASE + (
    "Its LiDAR tracker saw something moving on its own. Use the camera image and the track readings. "
    "Say if it is a lifeform, give a short label (max 10 words: what it looks like, its size), pick an "
    "action: approach (observe from a safe distance), observe (from here) or ignore, and a confidence "
    "from 0 to 1 (low if you cannot tell). Reply with JSON only: classification, label, reason, action, "
    "confidence. The reason (max 25 words) must quote a reading exactly as written, number and unit. "
    'Example: {"classification": "lifeform", "label": "possible lifeform, six legs, about 1.2 m", '
    '"reason": "speed 0.3 m/s, size 1.2 m, legs in the camera.", '
    '"action": "approach", "confidence": 0.8}'
)

SYSTEM_ANSWER = (
    "You are a rover exploring a super-Earth, answering your human operator in a chat. "
    "Use only the facts listed. Answer the question in 1 to 3 short sentences, in the first person "
    "(I, my), or we when you speak for the whole fleet. Copy numbers exactly as written in the facts; "
    "never add numbers, places, ids or events that are not in the facts. "
    "If the facts do not answer the question, say plainly that you do not know that. "
    'Reply with JSON only: {"text": "your answer"}'
)

EM_DASH = chr(0x2014)     # banned in all project text; replaced if the model writes one


# ====================================================================== config
def _env_str(name: str, default: str) -> str:
    v = os.environ.get(name)
    return default if v is None or not v.strip() else v.strip()


def _env_float(name: str, default: float) -> float:
    try:
        return float(os.environ.get(name, default))
    except (TypeError, ValueError):
        return default


def _env_bool(name: str, default: bool) -> bool:
    v = os.environ.get(name)
    if v is None or not v.strip():
        return default
    return v.strip().lower() in ("1", "true", "yes", "on")


@dataclass(frozen=True)
class GemmaConfig:
    url: str = DEFAULT_URL
    fallback_urls: tuple = (LOCAL_URL,)
    model: str = DEFAULT_MODEL
    fallback_models: tuple = FALLBACK_MODELS
    enabled: bool = True
    timeout_s: float = 4.0
    budget_ms: float = 3000.0
    fake: bool = False
    fake_latency_ms: float = 500.0
    num_predict: int = 96
    think: bool = False
    keep_alive: str = "30m"
    min_confidence: float = 0.55
    temperature: float = 0.0
    check_every_s: float = 15.0      # online check period (GET /api/tags)
    check_timeout_s: float = 1.5
    warmup_timeout_s: float = 120.0  # the first load of the model into memory takes about 30 s
    max_queue: int = 8               # queued requests (one per rover and kind, plus operator commands)
    max_wait_s: float | None = None  # a request that waited longer in the queue is dropped (default timeout_s)

    @property
    def urls(self) -> tuple:
        out = []
        for u in (self.url,) + tuple(self.fallback_urls):
            u = (u or "").strip().rstrip("/")
            if u and u not in out:
                out.append(u)
        return tuple(out)

    @property
    def queue_wait_s(self) -> float:
        return self.timeout_s if self.max_wait_s is None else self.max_wait_s

    @classmethod
    def from_env(cls, **overrides) -> "GemmaConfig":
        fb = tuple(m.strip() for m in _env_str("GEMMA_FALLBACK_MODELS", ",".join(FALLBACK_MODELS)).split(",")
                   if m.strip())
        cfg = cls(
            url=_env_str("GEMMA_URL", DEFAULT_URL),
            model=_env_str("GEMMA_MODEL", DEFAULT_MODEL),
            fallback_models=fb,
            enabled=_env_bool("GEMMA_ENABLED", True),
            timeout_s=max(0.05, _env_float("GEMMA_TIMEOUT_S", 4.0)),
            budget_ms=_env_float("GEMMA_BUDGET_MS", 3000.0),
            fake=_env_bool("GEMMA_FAKE", False),
            fake_latency_ms=max(0.0, _env_float("GEMMA_FAKE_LATENCY_MS", 500.0)),
            num_predict=int(_env_float("GEMMA_NUM_PREDICT", 96)),
            think=_env_bool("GEMMA_THINK", False),
            keep_alive=_env_str("GEMMA_KEEP_ALIVE", "30m"),
            min_confidence=_env_float("GEMMA_MIN_CONFIDENCE", 0.55),
        )
        return cls(**{**asdict(cfg), **overrides}) if overrides else cfg


# ====================================================================== data
@dataclass
class Request:
    """One question for Gemma. Built (and the inputs snapshotted) at submit time."""
    ticket: int
    kind: str                        # "choice" | "interpretation" | "command" | "describe"
    rover_id: str
    t: float                         # sim time of the submit
    payload: dict                    # compact data the prompt is made from
    inputs: list                     # every number the model is shown: what a reason may cite
    ids: tuple = ()                  # choice: candidate ids; command: rover ids
    options: tuple = ()              # interpretation: allowed actions (always ends with ask_operator)
    classes: tuple = ()              # interpretation: allowed classifications
    image_b64: str | None = None
    default: str | None = None       # the caller's rule-based pick, returned in fallbacks
    text: str | None = None          # command: the operator's message
    target_ids: tuple = ()           # command: known target ids
    coords: list = field(default_factory=list)   # command: signed numbers a coordinate may use
    parsed: dict | None = None       # command: the regex parser's answer (the fallback)
    num_predict: int | None = None
    timeout_s: float | None = None   # per request limit (answers: the latency budget); default GEMMA_TIMEOUT_S
    allowed_ids: tuple = ()          # answer: ids (r2, t4, ...) the reply may mention
    submitted: float = field(default_factory=time.perf_counter)

    @property
    def key(self) -> tuple:
        if self.kind == "command":                 # operator commands never replace each other
            return ("operator", "command", self.ticket)
        if self.kind == "answer":                  # nor do answers to different questions
            return (self.rover_id, "answer", self.ticket)
        return (self.rover_id, self.kind)


@dataclass
class Result:
    """What poll() returns. Dict-style access works too: res["choice"], res.get("reason")."""
    ok: bool                         # True only for a valid Gemma answer
    kind: str                        # "choice" | "interpretation" | "command"
    source: str                      # "gemma" | "fallback"
    rover_id: str = ""               # "operator" for commands
    t: float = 0.0                   # sim time of the submit (compare with now before acting)
    ticket: int = 0
    choice: str | None = None        # choice: candidate id (the caller's default in a fallback)
    classification: str | None = None
    action: str | None = None        # interpretation: action to take (the caller's default in a fallback)
    model_action: str | None = None  # what Gemma picked before the low-confidence rule
    command: dict | None = None      # command: {"type": "go_to", "rover": "r2", "x": 10.0, "y": -5.0}
    model_command: dict | None = None    # Gemma's command before the low-confidence rule made it unclear
    label: str | None = None         # describe: short description ("possible lifeform, six legs, about 1.2 m")
    ack: str | None = None           # command: one short sentence for the operator
    answer: str | None = None        # answer: the reply text (the caller's rule answer in a fallback)
    text: str | None = None          # command: the operator's message
    confidence: float = 0.0
    reason: str = ""                 # max 25 words, cites at least one input number
    cited: list = field(default_factory=list)    # the numbers in reason that match the input
    question: str | None = None      # ask_operator / unclear: short question for the human
    latency_ms: float = 0.0
    error: str | None = None         # why this is a fallback
    tokens_in: int | None = None
    tokens_out: int | None = None
    model: str = ""

    def to_dict(self) -> dict:
        return asdict(self)

    def __getitem__(self, key: str):
        return getattr(self, key)

    def get(self, key: str, default=None):
        return getattr(self, key, default)

    def trace(self) -> str:
        """One line for the event log."""
        ms = f"{self.latency_ms:.0f} ms"
        if self.kind == "command":
            who = "Gemma" if self.ok else "parser"
            typ = (self.command or {}).get("type", "unclear")
            say = self.question if typ == "unclear" else self.ack
            extra = f", Gemma fallback: {self.error}" if not self.ok and self.error else ""
            return f"Operator: {self.text!r} -> {typ} ({who}, conf {self.confidence:.2f}{extra}). {say}"
        if self.kind == "answer":
            who = "Gemma" if self.ok else f"rules, Gemma fallback: {self.error}"
            return f"{self.rover_id} answer ({who}, {ms}): {self.answer}"
        if not self.ok:
            keep = self.choice if self.kind == "choice" else self.action
            return f"{self.rover_id} Gemma fallback ({self.error}); keeping rule choice {keep}."
        if self.kind == "choice":
            return f"{self.rover_id} Gemma picked {self.choice} (conf {self.confidence:.2f}, {ms}): {self.reason}"
        if self.action == "ask_operator":
            return (f"{self.rover_id} Gemma unsure (conf {self.confidence:.2f}, {ms}), maybe "
                    f"{self.classification}; asking operator. {self.reason}")
        return (f"{self.rover_id} Gemma: {self.classification} -> {self.action} "
                f"(conf {self.confidence:.2f}, {ms}): {self.reason}")


class GemmaError(Exception):
    """The model or server answered with an error."""


class GemmaTimeout(GemmaError):
    """No answer in time."""


class GemmaOffline(GemmaError):
    """The server cannot be reached."""


# ====================================================================== prompt inputs
def compact(obj, depth: int = 0):
    """Small, JSON-safe copy for a prompt: floats rounded, None dropped, long lists cut, LiDAR
    points and images dropped. Also snapshots the caller's dicts (they may change after submit)."""
    if depth > 4:
        return None
    if obj is None or isinstance(obj, bool):
        return obj
    if isinstance(obj, int):
        return obj
    if isinstance(obj, float):
        if not math.isfinite(obj):
            return None
        return round(obj, 2) if abs(obj) < 100 else round(obj, 1)
    if isinstance(obj, str):
        return obj[:MAX_STR]
    if isinstance(obj, dict):
        out = {}
        for k, v in obj.items():
            if str(k) in SKIP_KEYS:
                continue
            v = compact(v, depth + 1)
            if v is None or v == {} or v == []:
                continue
            out[str(k)] = v
        return out
    if isinstance(obj, (list, tuple)):
        if len(obj) > MAX_LIST and all(isinstance(v, (int, float)) for v in obj):
            return None                       # a raw array (scan points): useless to a small model
        return [v for v in (compact(v, depth + 1) for v in list(obj)[:MAX_LIST]) if v is not None]
    if hasattr(obj, "item"):                  # numpy scalar
        try:
            return compact(obj.item(), depth)
        except Exception:                     # noqa: BLE001
            pass
    return str(obj)[:MAX_STR]


# ---- readings as named values with units ("battery 64%, pitch 3.1 deg"), so the model quotes
# real numbers instead of inventing labels
_PCT_KEYS = {"battery", "slip", "motor_load", "charge", "battery_after_return"}
_DEG_KEYS = {"pitch", "roll", "tilt", "heading", "slope", "slope_max", "slope_limit", "max_slope"}
_MPS_KEYS = {"wheel_speed", "ground_speed", "speed", "v"}
_M_KEYS = {"lidar_range", "dist", "distance", "range", "home_distance", "home_dist", "radius", "nominal_range",
           "visibility", "x", "y", "z", "size", "height", "edge_spread", "length"}
_S_KEYS = {"duration", "age", "stalled_for", "elapsed"}
_SUFFIX_UNITS = (("_deg", "deg"), ("_mps", "m/s"), ("_pct", "%"), ("_m", "m"), ("_s", "s"))


def _label_unit(key: str) -> tuple:
    k = key.lower()
    for suf, unit in _SUFFIX_UNITS:
        if k.endswith(suf) and len(k) > len(suf):
            return k[:-len(suf)], unit
    for keys, unit in ((_DEG_KEYS, "deg"), (_MPS_KEYS, "m/s"), (_M_KEYS, "m"), (_S_KEYS, "s")):
        if k in keys:
            return k, unit
    return k, ""


def _fmt_reading(key: str, v) -> str:
    label, unit = _label_unit(str(key))
    label = label.replace("_", " ")
    if isinstance(v, bool):
        return f"{label} {'yes' if v else 'no'}"
    if isinstance(v, (int, float)):
        if str(key).lower() in _PCT_KEYS and 0.0 <= v <= 1.0:
            return f"{label} {v * 100:.0f}%"
        if unit == "%":
            return f"{label} {v:g}%"
        return f"{label} {v:g}" + (f" {unit}" if unit else "")
    if isinstance(v, dict):
        return f"{label} (" + ", ".join(_fmt_reading(k2, v2) for k2, v2 in v.items()) + ")"
    if isinstance(v, (list, tuple)):
        return f"{label} [" + ", ".join(f"{x:g}" if isinstance(x, (int, float)) and not isinstance(x, bool)
                                        else str(x) for x in v) + "]"
    return f"{label} {v}"


def render_readings(d: dict) -> str:
    """{"battery": 0.64, "pitch": 3.1} -> "battery 64%, pitch 3.1 deg"."""
    return ", ".join(_fmt_reading(k, v) for k, v in (d or {}).items())


# a number not glued to a letter, digit, underscore or dot on its left: "f2", "r1", "t3" are ids
_NUM_RE = re.compile(r"(?<![A-Za-z0-9_.])[-+]?\d+(?:\.\d+)?")
# keys whose numbers are not inputs a reason may cite or a coordinate may use (ids, the map edges)
_NO_NUMBERS_KEYS = {"rover", "id", "kind", "options", "classes", "type", "name", "colour", "color", "bounds",
                    "lines"}     # lines: recent chat text (v7); only the context's place coordinates count


def numbers_in_text(text: str) -> list:
    """[(as written, value)] for each number in text."""
    return [(m.group(0), float(m.group(0))) for m in _NUM_RE.finditer(text or "")]


def input_numbers(payload, key: str = "") -> list:
    """Every number in a payload (values, and numbers inside strings such as summaries)."""
    out = []
    if key in _NO_NUMBERS_KEYS:
        return out
    if isinstance(payload, bool) or payload is None:
        return out
    if isinstance(payload, (int, float)):
        out.append(float(payload))
    elif isinstance(payload, str):
        out.extend(v for _, v in numbers_in_text(payload))
    elif isinstance(payload, dict):
        for k, v in payload.items():
            out.extend(input_numbers(v, str(k)))
    elif isinstance(payload, (list, tuple)):
        for v in payload:
            out.extend(input_numbers(v, key))
    return out


def _decimals(s: str) -> int:
    return len(s.split(".", 1)[1]) if "." in s else 0


def _matches(written: str, r: float, w: float) -> bool:
    """Is `written` (value r) a citation of input value w? Allows rounding to the precision the
    model wrote (7.83 as 7.8, or as 8 when w >= 2), a fraction as a percent (0.41 as 41), a percent
    as a fraction, and a dropped minus sign."""
    r = abs(r)
    d = _decimals(written)
    aw = abs(w)
    forms = [aw]
    if 0 < aw <= 1.0:
        forms.append(aw * 100.0)
    if 1.0 < aw <= 100.0:
        forms.append(aw / 100.0)
    for c in forms:
        if r == 0.0 or c == 0.0:
            if r == 0.0 and c == 0.0:
                return True
            continue
        if abs(r - c) <= 1e-9 + 0.01 * c:
            return True
        if d > 0 and abs(c - r) <= 0.5 * 10.0 ** (-d) + 1e-9:
            return True
        if d == 0 and c >= 2.0 and abs(c - r) <= 0.5 + 1e-9:
            return True
    return False


_UNIT_RE = re.compile(r"\s?(%|m/s\b|deg\b|m\b|s\b)")


def cited_numbers(reason: str, inputs: list) -> list:
    """The numbers in reason that match a number from the input, as written, with the unit that
    follows them if any ("41%", "12 deg", "0.58 m/s")."""
    out = []
    for m in _NUM_RE.finditer(reason or ""):
        s = m.group(0)
        if not any(_matches(s, float(s), w) for w in inputs):
            continue
        u = _UNIT_RE.match(reason, m.end())
        if u:
            s += u.group(1) if u.group(1) == "%" else " " + u.group(1)
        if s not in out:
            out.append(s)
    return out


def estimate_tokens(text: str) -> int:
    """Rough, conservative token count for these prompts (about 3.5 characters a token)."""
    return int(math.ceil(len(text) / 3.5))


def _clean_text(text: str, max_words: int = MAX_REASON_WORDS) -> str:
    text = " ".join(str(text).replace(EM_DASH, " - ").split())
    words = text.split(" ")
    if len(words) > max_words:
        text = " ".join(words[:max_words]).rstrip(",;:") + "..."
    return text


# ---- colours: the sim sends hex colours, people say "the orange one"
_COLOUR_SYNONYMS = {"teal": "cyan", "turquoise": "cyan", "aqua": "cyan", "violet": "purple", "magenta": "pink",
                    "gold": "yellow", "golden": "yellow", "lime": "green", "peach": "orange", "gray": "grey",
                    "amber": "orange", "navy": "blue", "crimson": "red", "scarlet": "red"}
_COLOURS = ("red", "orange", "yellow", "green", "cyan", "blue", "purple", "pink", "grey", "white", "black")
_COLOUR_RE = re.compile(r"\b(" + "|".join(_COLOURS + tuple(_COLOUR_SYNONYMS)) + r")\b")


def colour_name(c) -> str | None:
    """'#ff7a45' -> 'orange'. Plain names pass through."""
    if not isinstance(c, str) or not c.strip():
        return None
    s = c.strip().lower()
    if s in _COLOURS:
        return s
    if s in _COLOUR_SYNONYMS:
        return _COLOUR_SYNONYMS[s]
    s = s.lstrip("#")
    if len(s) == 3:
        s = "".join(ch * 2 for ch in s)
    if len(s) != 6 or any(ch not in "0123456789abcdef" for ch in s):
        return None
    r, g, b = (int(s[i:i + 2], 16) / 255.0 for i in (0, 2, 4))
    h, sat, val = colorsys.rgb_to_hsv(r, g, b)
    if val < 0.15:
        return "black"
    if sat < 0.15:
        return "white" if val > 0.85 else "grey"
    deg = h * 360.0
    for top, name in ((15, "red"), (40, "orange"), (70, "yellow"), (160, "green"), (195, "cyan"),
                      (255, "blue"), (300, "purple"), (345, "pink"), (361, "red")):
        if deg < top:
            return name
    return "red"


# ====================================================================== requests
def make_choice_request(ticket: int, rover_id: str, t: float, perception: dict, candidates: list,
                        default: str | None = None) -> Request:
    """Raises ValueError on bad input (the planner turns that into a fallback, it never raises)."""
    if not isinstance(candidates, (list, tuple)) or not candidates:
        raise ValueError("no candidates")
    cands, ids = [], []
    for c in list(candidates)[:MAX_CANDIDATES]:
        if not isinstance(c, dict) or not c.get("id"):
            raise ValueError(f"candidate without id: {c!r}"[:120])
        cid = str(c["id"])
        if cid in ids:
            raise ValueError(f"duplicate candidate id {cid}")
        kind = str(c.get("kind", "frontier"))
        if kind not in CANDIDATE_KINDS:
            raise ValueError(f"candidate {cid}: kind {kind!r} not in {CANDIDATE_KINDS}")
        ids.append(cid)
        cands.append({"id": cid, "kind": kind, "summary": " ".join(str(c.get("summary", "")).split())[:MAX_STR]})
    if default is not None and str(default) not in ids:
        raise ValueError(f"default {default!r} is not a candidate id")
    payload = {"rover": str(rover_id), "sensors": compact(perception or {}) or {}, "candidates": cands}
    readings = render_readings(payload["sensors"])
    return Request(ticket=ticket, kind="choice", rover_id=str(rover_id), t=float(t), payload=payload,
                   inputs=input_numbers(payload) + input_numbers(readings), ids=tuple(ids),
                   default=str(default) if default is not None else ids[0])


def make_interpretation_request(ticket: int, rover_id: str, t: float, perception: dict, anomaly: dict,
                                image_jpeg: bytes | None, options: list,
                                default: str | None = None, classes: list | None = None) -> Request:
    opts = []
    for o in (options or ()):
        o = str(o)
        if o not in ACTIONS:
            raise ValueError(f"option {o!r} not in {ACTIONS}")
        if o not in opts:
            opts.append(o)
    if not [o for o in opts if o != "ask_operator"]:
        raise ValueError("no options besides ask_operator")
    if "ask_operator" not in opts:
        opts.append("ask_operator")       # always allowed: low confidence must be able to ask
    cls = tuple(str(c) for c in (classes or CLASSES))
    if "unknown" not in cls:
        cls = cls + ("unknown",)
    if default is None:
        default = next((o for o in ("proceed_slowly", "wait") if o in opts), opts[0])
    elif default not in opts:
        raise ValueError(f"default {default!r} not in options")
    if image_jpeg is not None and not isinstance(image_jpeg, (bytes, bytearray, memoryview)):
        raise ValueError("image_jpeg must be bytes or None")
    payload = {"rover": str(rover_id), "sensors": compact(perception or {}) or {},
               "anomaly": compact(anomaly or {}) or {}, "options": opts, "classes": list(cls)}
    img = base64.b64encode(bytes(image_jpeg)).decode("ascii") if image_jpeg else None
    rendered = render_readings(payload["sensors"]) + " " + render_readings(payload["anomaly"])
    return Request(ticket=ticket, kind="interpretation", rover_id=str(rover_id), t=float(t), payload=payload,
                   inputs=input_numbers(payload) + input_numbers(rendered), options=tuple(opts), classes=cls,
                   image_b64=img, default=default)


def make_description_request(ticket: int, rover_id: str, t: float, perception: dict, observation: dict,
                             image_jpeg: bytes | None, default: str = "observe") -> Request:
    """Something moving: what is it? observation: track numbers (distance, speed, size, ...)."""
    if default not in DESCRIBE_ACTIONS:
        raise ValueError(f"default {default!r} not in {DESCRIBE_ACTIONS}")
    if image_jpeg is not None and not isinstance(image_jpeg, (bytes, bytearray, memoryview)):
        raise ValueError("image_jpeg must be bytes or None")
    payload = {"rover": str(rover_id), "sensors": compact(perception or {}) or {},
               "observation": compact(observation or {}) or {}, "options": list(DESCRIBE_ACTIONS),
               "classes": list(DESCRIBE_CLASSES)}
    img = base64.b64encode(bytes(image_jpeg)).decode("ascii") if image_jpeg else None
    rendered = render_readings(payload["sensors"]) + " " + render_readings(payload["observation"])
    return Request(ticket=ticket, kind="describe", rover_id=str(rover_id), t=float(t), payload=payload,
                   inputs=input_numbers(payload) + input_numbers(rendered), options=DESCRIBE_ACTIONS,
                   classes=DESCRIBE_CLASSES, image_b64=img, default=default)


def _fleet_context(ctx: dict) -> dict:
    """Small copy of the fleet context for the command prompt: rovers (id, name, colour name,
    position, state, battery), known targets, lander, map size."""
    ctx = ctx if isinstance(ctx, dict) else {}
    rovers = []
    for r in list(ctx.get("rovers") or [])[:8]:
        if not isinstance(r, dict) or not r.get("id"):
            continue
        e = {"id": str(r["id"])}
        if r.get("name"):
            e["name"] = str(r["name"])[:40]
        col = colour_name(r.get("colour", r.get("color")))
        if col:
            e["colour"] = col
        for k in ("x", "y", "state", "battery"):
            if r.get(k) is not None:
                e[k] = compact(r[k])
        rovers.append(e)
    targets = []
    for tg in list(ctx.get("targets") or [])[:MAX_LIST]:
        if not isinstance(tg, dict) or not tg.get("id"):
            continue
        e = {"id": str(tg["id"])}
        for k in ("kind", "x", "y", "status"):
            if tg.get(k) is not None:
                e[k] = compact(tg[k])
        targets.append(e)
    out = {"rovers": rovers, "targets": targets}
    lander = ctx.get("lander")
    if isinstance(lander, dict):
        lander = [lander.get("x", 0.0), lander.get("y", 0.0)]
    if isinstance(lander, (list, tuple)) and len(lander) >= 2:
        out["lander"] = [compact(float(lander[0])), compact(float(lander[1]))]
    if isinstance(ctx.get("map_size"), (int, float)):
        out["map_size"] = compact(ctx["map_size"])
    b = ctx.get("bounds")            # [xmin, ymin, xmax, ymax] in m
    if isinstance(b, (list, tuple)) and len(b) == 4 and all(isinstance(v, (int, float)) for v in b):
        out["bounds"] = [compact(float(v)) for v in b]
    elif isinstance(out.get("map_size"), (int, float)):
        h = out["map_size"] / 2.0
        out["bounds"] = [compact(-h), compact(-h), compact(h), compact(h)]
    chat = ctx.get("chat")           # protocol v7: a few lines of chat memory (advisor.ChatMemory.context)
    if isinstance(chat, dict):
        lines = [" ".join(str(x).replace(EM_DASH, ",").split())[:MAX_CHAT_LINE]
                 for x in list(chat.get("lines") or [])[-MAX_CHAT_LINES:] if str(x).strip()]
        refs = " ".join(str(chat.get("refs") or "").replace(EM_DASH, ",").split())[:MAX_CHAT_REFS]
        if lines or refs:
            out["chat"] = {"lines": lines, "refs": refs}
    return out


def make_command_request(ticket: int, t: float, text: str, fleet_context: dict) -> Request:
    text = " ".join(str(text or "").split())[:MAX_TEXT]
    ctx = _fleet_context(fleet_context)
    rover_ids = tuple(r["id"] for r in ctx["rovers"])
    target_ids = tuple(tg["id"] for tg in ctx["targets"])
    parsed = parse_command(text, ctx)
    _, cleaned, _ = _resolve_rover(text.lower(), ctx["rovers"])
    coords = [v for _, v in numbers_in_text(_TARGET_RE.sub(" ", cleaned))] + input_numbers(ctx)
    payload = {"message": text, "fleet": ctx}
    return Request(ticket=ticket, kind="command", rover_id="operator", t=float(t), payload=payload,
                   inputs=coords, ids=rover_ids, text=text, target_ids=target_ids, coords=coords, parsed=parsed,
                   num_predict=COMMAND_NUM_PREDICT)


_ID_RE = re.compile(r"\b([rtkqwsh])(\d+)\b")


def make_answer_request(ticket: int, speaker: str, t: float, question: str, topic: str, facts: list,
                        default: str, timeout_s: float | None = None, context: list | None = None) -> Request:
    """An operator question to answer from fact lines (answers.fact_lines). speaker: a rover id, or
    "fleet". default: the rule-based answer, returned in a fallback. context (protocol v7): up to two short
    lines from the chat memory saying what a reference in the question means ("there" is the pond's shore);
    the facts must still hold every number the answer uses."""
    q = " ".join(str(question or "").split())[:MAX_TEXT]
    if not q:
        raise ValueError("empty question")
    lines = [" ".join(str(x).replace(EM_DASH, ",").split())[:2 * MAX_STR] for x in (facts or []) if str(x).strip()]
    lines = lines[:MAX_FACT_LINES]
    topic = topic if topic in TOPICS else "unknown"
    ids = {str(speaker)}
    for ln in lines:
        ids.update(m.group(0) for m in _ID_RE.finditer(ln))
    payload = {"speaker": str(speaker), "question": q, "topic": topic, "facts": lines}
    ctx = [" ".join(str(x).replace(EM_DASH, ",").split())[:MAX_STR] for x in (context or []) if str(x).strip()]
    if ctx:
        payload["context"] = ctx[:MAX_CONTEXT_LINES]
    return Request(ticket=ticket, kind="answer", rover_id=str(speaker), t=float(t), payload=payload,
                   inputs=[v for ln in lines for _, v in numbers_in_text(_strip_thousands(ln))], text=q,
                   default=str(default or ""), num_predict=ANSWER_NUM_PREDICT, timeout_s=timeout_s,
                   allowed_ids=tuple(sorted(ids)))


def _answer_prompt(req: Request) -> str:
    p = req.payload
    who = ("You speak for the whole rover fleet." if p["speaker"] == "fleet" else f"You are rover {p['speaker']}.")
    facts = "\n".join(f"- {ln}" for ln in p["facts"]) or "- (no facts)"
    ctx = "".join(f"Chat context: {ln}\n" for ln in p.get("context") or [])
    return (f"{who}\nOperator asks: {json.dumps(p['question'], ensure_ascii=False)}\n{ctx}Facts:\n{facts}\n"
            f"Answer the question.")


def _command_prompt(req: Request) -> str:
    ctx = req.payload["fleet"]
    lines = [f"Operator message: {json.dumps(req.text, ensure_ascii=False)}"]
    rv = []
    for r in ctx["rovers"]:
        bits = [r["id"]]
        if r.get("name"):
            bits.append(json.dumps(r["name"]))
        if r.get("colour"):
            bits.append(r["colour"])
        if "x" in r and "y" in r:
            bits.append(f"at ({r['x']:g}, {r['y']:g})")
        if r.get("state"):
            bits.append(str(r["state"]))
        if isinstance(r.get("battery"), (int, float)):
            bits.append(_fmt_reading("battery", r["battery"]))
        rv.append(" ".join(bits))
    lines.append("Rovers: " + ("; ".join(rv) if rv else "unknown"))
    tg = []
    for t_ in ctx["targets"]:
        bits = [t_["id"]] + [str(t_[k]) for k in ("kind", "status") if t_.get(k)]
        if "x" in t_ and "y" in t_:
            bits.append(f"at ({t_['x']:g}, {t_['y']:g})")
        tg.append(" ".join(bits))
    lines.append("Known targets: " + ("; ".join(tg) if tg else "none"))
    where = []
    if "lander" in ctx:
        where.append(f"Lander at ({ctx['lander'][0]:g}, {ctx['lander'][1]:g}).")
    if "bounds" in ctx:
        x0, y0, x1, y1 = ctx["bounds"]
        if (x0, x1) == (y0, y1):
            where.append(f"Map x and y {x0:g} to {x1:g} m, +x east, +y north.")
        else:
            where.append(f"Map x {x0:g} to {x1:g} m, y {y0:g} to {y1:g} m, +x east, +y north.")
    if where:
        lines.append(" ".join(where))
    ch = ctx.get("chat")
    if ch:
        if ch.get("lines"):
            lines.append("Recent chat: " + " | ".join(ch["lines"]))
        if ch.get("refs"):
            lines.append("Context: " + ch["refs"])
    lines.append("Give one command.")
    return "\n".join(lines)


def build_prompt(req: Request) -> tuple:
    """(system prompt, user text). Kept small: under about 400 tokens without the image."""
    if req.kind == "command":
        return SYSTEM_COMMAND, _command_prompt(req)
    if req.kind == "answer":
        return SYSTEM_ANSWER, _answer_prompt(req)
    p = req.payload
    head = f"Rover {p['rover']} readings: {render_readings(p['sensors']) or 'none'}"
    if req.kind == "choice":
        cands = "\n".join(f"{c['id']} ({c['kind']}): {c['summary']}" for c in p["candidates"])
        return SYSTEM_CHOICE, (f"{head}\nCandidates:\n{cands}\n"
                               "Pick one candidate id. Quote at least one reading above with its number and unit.")
    if req.kind == "describe":
        ob = dict(p["observation"])
        lines = [head, f"Moving object (LiDAR track): {render_readings(ob) or 'no numbers'}",
                 "Camera image attached." if req.image_b64 else "No camera image.",
                 "Classes: " + ", ".join(p["classes"]),
                 "Actions: " + ", ".join(o for o in p["options"] if o != "ask_operator"),
                 "Describe it and pick one action. Quote at least one reading above with its number and unit."]
        return SYSTEM_DESCRIBE, "\n".join(lines)
    an = dict(p["anomaly"])
    typ = str(an.pop("type", "anomaly"))
    lines = [head, f"Anomaly: {typ}" + (f": {render_readings(an)}" if an else ""),
             "Camera image attached." if req.image_b64 else "No camera image.",
             "Classes: " + ", ".join(p["classes"]),
             "Options: " + ", ".join(p["options"]),
             "Classify the cause and pick one option. Quote at least one reading above with its number and unit."]
    return SYSTEM_INTERPRET, "\n".join(lines)


def build_schema(req: Request) -> dict:
    """JSON schema for Ollama's `format`: constrains the output to valid ids and options. For
    choices the reason comes first so the model states the evidence before it commits."""
    if req.kind == "choice":
        props = {"reason": {"type": "string"},
                 "choice": {"type": "string", "enum": list(req.ids)},
                 "confidence": {"type": "number"}}
    elif req.kind == "interpretation":
        props = {"classification": {"type": "string", "enum": list(req.classes)},
                 "reason": {"type": "string"},
                 "action": {"type": "string", "enum": list(req.options)},
                 "confidence": {"type": "number"}}
    elif req.kind == "answer":
        props = {"text": {"type": "string"}}
    elif req.kind == "describe":
        props = {"classification": {"type": "string", "enum": list(req.classes)},
                 "label": {"type": "string"},
                 "reason": {"type": "string"},
                 "action": {"type": "string", "enum": [o for o in req.options if o != "ask_operator"]},
                 "confidence": {"type": "number"}}
    else:   # flat on purpose: unions (anyOf) are not reliably supported by every grammar engine
        rover = ({"type": "string", "enum": list(req.ids) + ["all", "none"]} if req.ids else {"type": "string"})
        target = ({"type": "string", "enum": list(req.target_ids) + ["none"]} if req.target_ids
                  else {"type": "string"})
        near = ({"type": "string", "enum": list(req.ids) + ["lander", "none"]} if req.ids else {"type": "string"})
        props = {"command": {"type": "string", "enum": list(COMMANDS)},
                 "rover": rover,
                 "place": {"type": "string", "enum": list(PLACES) + ["none"]},
                 "qualifier": {"type": "string", "enum": list(QUALIFIERS)},
                 "ordinal": {"type": "number"},
                 "near_rover": near,
                 "investigate": {"type": "boolean"},
                 "explore": {"type": "boolean"},
                 "direction": {"type": "string", "enum": list(DIRECTIONS) + ["none"]},
                 "x": {"type": "number"}, "y": {"type": "number"}, "radius": {"type": "number"},
                 "target_id": target,
                 "topic": {"type": "string", "enum": list(TOPICS) + ["none"]},
                 "question": {"type": "string"},
                 "ack": {"type": "string"},
                 "confidence": {"type": "number"}}
    return {"type": "object", "properties": props, "required": list(props)}


def operator_question(req: Request, classification: str, confidence: float, reason: str) -> str:
    what = str(req.payload.get("anomaly", {}).get("type", "anomaly")).replace("_", " ")
    opts = ", ".join(o for o in req.options if o != "ask_operator")
    return (f"{req.rover_id} {what}: Gemma is unsure (confidence {confidence:.2f}), maybe "
            f"{classification.replace('_', ' ')}. {reason} Choose: {opts}.")


# ====================================================================== operator command parser
_WORD_NUMS = {"one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6, "seven": 7, "eight": 8,
              "first": 1, "second": 2, "third": 3, "fourth": 4, "fifth": 5, "sixth": 6, "seventh": 7, "eighth": 8}
_NUMWORD = r"\d+|" + "|".join(_WORD_NUMS)
_ROVERN_RE = re.compile(rf"\b(?:rover|robot|unit|bot)\s*(?:#|no\.?|number)?\s*({_NUMWORD})\b"
                        rf"|\b({'|'.join(k for k in _WORD_NUMS if not k.isdigit())})\s+(?:rover|robot)\b")
_RID_RE = re.compile(r"\br(\d+)\b")
_ALL_RE = re.compile(r"\b(?:all rovers|all of you|all|everyone|everybody|every rover|whole fleet|the fleet|fleet)\b")
_TARGET_RE = re.compile(r"\b(?:target\s*(?:#|no\.?|number)?\s*|t)(\d+)\b")
_DIR_RE = re.compile(r"\b(north|south)[\s-]?(east|west)\b|\b(north|south|east|west)(?:wards?)?\b|\b(ne|nw|se|sw)\b")
_XY_RE = re.compile(r"\bx\s*[=:]?\s*(-?\d+(?:\.\d+)?)\s*[,;]?\s*(?:and\s+)?y\s*[=:]?\s*(-?\d+(?:\.\d+)?)")
_PAIR_RE = re.compile(r"(?<![\w.])(-?\d+(?:\.\d+)?)\s*[,;]\s*(-?\d+(?:\.\d+)?)")
_RADIUS_RE = re.compile(r"\bradius\s*(?:of\s*)?(\d+(?:\.\d+)?)|(\d+(?:\.\d+)?)\s*(?:m|meters?|metres?)\b")
_AVOID_RE = re.compile(r"\b(avoid|stay away|keep away|keep out|stay out|no[- ]go|keep clear|steer clear)\b")
_SAMPLE_RE = re.compile(r"\b(sample|samples|investigate|analy[sz]e|inspect|check|study|scan|visit|grab|go|drive|"
                        r"head|send|get)\b")
_RETURN_RE = re.compile(r"\b(return|returns|go home|come home|come back|head home|head back|go back|recall|dock|"
                        r"home|rtb|back to (?:the )?(?:lander|base))\b")
_RESUME_RE = re.compile(r"\b(resume|continue|carry on|go on|unpause|un-pause|keep going|keep exploring|restart|"
                        r"proceed|start exploring|explore again)\b")
_HOLD_RE = re.compile(r"\b(hold|stop|halt|freeze|pause|wait|stay|park|brake|standby|stand by)\b")
_GO_RE = re.compile(r"\b(go|goes|drive|move|head|navigate|travel|send|get|to)\b")
_EXPLORE_RE = re.compile(r"\b(explore|exploring|scout|survey|search|map)\b")
_HELP = ("Sorry, I did not understand. Try: 'rover 2 go to 10, -5', 'all return', 'r1 explore north', "
         "'hold', 'resume' or 'sample t4'.")


# ---- questions ("rover 2, what have you found?", "everyone, report battery", "status")
_SUBJ = r"(?:you|u|we|they|it|anyone|anybody|everyone|the fleet|the rovers?|any|r\d+|rover\s*\w+)"
_CMD_VERBS = (r"go|drive|head|move|return|come|hold|stop|halt|freeze|wait|park|resume|continue|carry on|keep|"
              r"sample|explore|scout|survey|avoid|stay|check|investigate|inspect|send|get|navigate|travel|recall|dock|"
              r"proceed|turn|back|visit|revisit|scan|search|examine|study|observe|watch|approach|swing by|stop by|"
              r"look (?:at|around|for|into|over)|have a look|take a look|take a closer look")
_CMD_VERBS_Q = _CMD_VERBS            # "can you go home?" / "can we investigate the lake" are orders, not questions
_Q_START_RE = re.compile(r"^(?:what|whats|what's|where|wheres|where's|why|how|hows|how's|which|who|when|is|are|am|"
                         r"was|were|any|anything|seen|seeing|found|got)\b"
                         rf"|^(?:do|does|did|have|has|had)\s+{_SUBJ}\b"
                         rf"|^(?:can|could)\s+{_SUBJ}\b(?!\s+(?:please\s+)?(?:also\s+|now\s+|maybe\s+)?(?:{_CMD_VERBS_Q})\b)")
_Q_TELL_RE = re.compile(r"\b(?:tell me|let me know|give me (?:a |an |your )?(?:update|report|status|summary|rundown)|"
                        r"report(?!\s+(?:back|to|in)\b)|status|update me|summary|sitrep|what'?s up|"
                        r"how are (?:you|things)|how is it going|how'?s it going)\b")
# polite orders: "can we investigate ...", "could you check ...", "let's survey ...", "I want you to go ..."
_POLITE_SUBJ = (r"(?:you|u|we|y'?all|you guys|someone|somebody|one of you|a rover|the rovers?|the fleet|"
                r"rover\s*\w+|r\d+)")
_CMD_PREFIX_RE = re.compile(
    rf"^(?:please\s+)?(?:can|could|would|will|shall)\s+(?:{_POLITE_SUBJ}\s+)?(?:please\s+)?(?:also\s+|now\s+|maybe\s+)?"
    rf"(?:{_CMD_VERBS})\b"
    r"|^(?:how about|what about|why don'?t you|why not)\b"
    rf"|^(?:let'?s|lets|let us)\s+(?:also\s+|now\s+)?(?:{_CMD_VERBS})\b"
    rf"|^(?:i|we)\s*(?:'d|would)?\s+(?:want|like|need)\s+(?:you\s+|one of you\s+|a rover\s+|someone\s+)?to\s+"
    rf"(?:{_CMD_VERBS})\b"
    rf"|^(?:you|we)\s+(?:should|must|need to|have to|could)\s+(?:{_CMD_VERBS})\b")
_IMPERATIVE_RE = re.compile(rf"^(?:please\s+)?(?:{_CMD_VERBS}|map)\b")
_LEAD_RE = re.compile(r"^(?:[\s,.:;!\-]|hey\b|hi\b|ok\b|okay\b|so\b|and\b|also\b|quick question\b|question\b)+")
_EACH_RE = re.compile(r"\b(?:each|every rover|everyone|everybody|all of you|all rovers|each of you|every one of you)\b")
_TOPIC_RES = (   # first match wins
    ("battery", re.compile(r"batter|charg|\bpower|energy|juice|\bfuel")),
    ("lifeforms", re.compile(r"\blife|alive|creature|animal|organism|grazer|\bherd|alien(?!\s+spire)|living|biolog|"
                             r"moving thing|something mov|anything mov")),
    ("water", re.compile(r"water|\blake|river|liquid|\bpond|\bpool|\bsea\b|\bwet\b")),
    ("samples", re.compile(r"sample|\btube|cache|drill|\bcore|scoop|swab|analys|analyz|science score")),
    ("why", re.compile(r"\bwhy\b|reason|what are you doing|what're you doing|where are you (?:going|heading|headed)|"
                       r"where (?:to|next)\b|\bplan\b|\bplans\b|\bgoal|heading|headed|destination|\bnext\b|purpose")),
    ("location", re.compile(r"where are you|where is|where'?s|wheres|position|location|coordinates|"
                            r"how far (?:are you |is it )?(?:from|away)|distance (?:to|from) (?:the )?(?:lander|base|home)")),
    ("hazards", re.compile(r"problem|issue|trouble|hazard|danger|wrong|rock ?fall|\brocks?\b|boulder|\bsand|stuck|"
                           r"\bdust|storm|fault|obstacle|\brisk|damage|broke|spire|incident|difficult|scary")),
    ("finds", re.compile(r"\bfind|found|discover|target|interesting|science|\bspot|detect|result")),
    ("history", re.compile(r"done so far|have you done|did you do|you('?ve| have) done|been up to|been doing|"
                           r"what have you (?:achieved|accomplished)|\brecap|\bhistory\b|\blogs?\b|mission report|"
                           r"your day\b|summar|everything you")),
    ("exploration", re.compile(r"explor|mapped|\bmap\b|cover|\barea|region|\bbeen\b|visited|driven|\bdrive|travel|"
                               r"distance|how far|how much ground|progress|percent")),
    ("status", re.compile(r"status|how are you|how'?s it going|how is it going|update|report|sitrep|\bok\b|okay|"
                          r"alright|all right|doing\b|\bstate\b|summary|overview|what'?s up|health|condition")),
)


def question_topic(text: str) -> str:
    """'how's your battery?' -> 'battery'. One of TOPICS ('unknown' when nothing matches)."""
    low = " ".join(str(text or "").lower().split())
    for topic, rx in _TOPIC_RES:
        if rx.search(low):
            return topic
    return "unknown"


def _q_core(text: str, rovers: list) -> tuple:
    """(rover, the message with rover mentions and filler words removed)."""
    low = " ".join(str(text or "").lower().split())
    rover, rest, _ = _resolve_rover(low, rovers)
    return rover, _LEAD_RE.sub("", " ".join(rest.split())).strip()


def question_intent(text: str, rovers: list | None = None) -> tuple:
    """Does the message ask for information? Returns (strength, topic): strength "strong" (starts
    with a question word, or 'tell me', 'report', 'status'), "weak" (only a trailing '?'), "command"
    (starts with an order, also when politely asked: 'can you go home?') or None (no sign either way)."""
    _, core = _q_core(text, list(rovers or []))
    topic = question_topic(core)
    if not core:
        return None, topic
    if _CMD_PREFIX_RE.match(core) or _IMPERATIVE_RE.match(core):
        return "command", topic
    if _Q_START_RE.match(core) or _Q_TELL_RE.search(core):
        return "strong", topic
    if core.endswith("?"):
        return "weak", topic
    return None, topic


def wants_each(text: str) -> bool:
    """'everyone, report battery': addressed to every rover one by one."""
    return bool(_EACH_RE.search(" ".join(str(text or "").lower().split())))


# ---- places (protocol v6): "go to the water source", "can we investigate the lake for signs of life"
_PLACE_RES = tuple((k, re.compile(rx)) for k, rx in (
    ("hazard", r"\b(?:fallen spires?|rock ?falls?|boulders?|new rocks|landslides?|soft (?:ground|sand|spot|patch)|"
               r"sand(?: ?pits?| patch)?|quicksand|sinkholes?|hazards?|danger(?:ous)? (?:zone|area|spot)s?)\b"),
    ("water", r"\b(?:water(?: ?sources?| ?holes?)?|lakes?|ponds?|pools?|rivers?|streams?|creeks?|liquids?|"
              r"shore(?:lines?)?s?|seas?|puddles?)\b"),
    ("spire", r"\b(?:spires?|crystals?|towers?|pillars?|obelisks?|monoliths?)\b"),
    ("vent", r"\b(?:vents?|geysers?|fumaroles?)\b"),
    ("ice", r"\b(?:ice|icy|frost)\b"),
    ("mineral", r"\b(?:minerals?|outcrops?|veins?|ores?)\b"),
    ("trace", r"\b(?:organics?|organic traces?|traces?(?! of)|residues?|biofilms?)\b"),
    ("lifeform", r"\b(?:herds?|creatures?|animals?|lifeforms?|life forms?|grazers?|aliens?(?! spires?)|organisms?|"
                 r"wildlife|beasts?)\b"),
    ("target", r"\b(?:targets?|science sites?|sample sites?|discover(?:y|ies))\b"),
    ("lander", r"\b(?:lander|landing (?:site|pad)|home base|base)\b"),
))
_PLACE_VERB_RE = re.compile(r"\b(?:go|goes|going|head\w*|drive|move|navigate|travel|visit\w*|revisit\w*|return\w*|"
                            r"come back|get back|back to|investigat\w*|check\w*|look\w*|inspect\w*|survey\w*|scan\w*|"
                            r"search\w*|explor\w*|examin\w*|stud(?:y|ies)|observ\w*|watch\w*|approach\w*|send|"
                            r"swing by|stop by|take|get (?:to|over|closer)|bring|reach|find)\b")
_INVESTIGATE_RE = re.compile(r"\b(?:investigat\w*|survey\w*|look(?:ing)? for|"
                             r"search\w*|scan\w*|examin\w*|check(?:ing)? (?:for|around)|"
                             r"stud(?:y|ying)|observ\w*|watch\w*|signs? of|traces? of|evidence of)\b")
# protocol v7: stay and explore an area ("explore around the lake", "go to the vent and look around")
_EXPLORE_AREA_RE = re.compile(r"\b(?:explor\w*|look(?:ing)? around|have a look around|stick around|"
                              r"map (?:out )?(?:the area )?around)\b")
_EXPLORE_SIDE_RE = re.compile(r"\bexplor\w*\s+(?:the\s+(?:area|ground|region|land|terrain)\s+)?(?:to\s+the\s+)?"
                              r"(north|south)[\s-]?(east|west)(?:ern)?\s+(?:side\s+)?of\b"
                              r"|\bexplor\w*\s+(?:the\s+(?:area|ground|region|land|terrain)\s+)?(?:to\s+the\s+)?"
                              r"(north|south|east|west)(?:ern)?\s+(?:side\s+)?of\b")
# chat references: the place talked about last ("go back there", "investigate it", "that spot")
_THERE_RE = re.compile(r"\b(?:over\s+)?there\b(?!\s+(?:is|are|was|were|be)\b)"
                       r"|\bthat\s+(?:place|spot|area|location|site|one)\b"
                       r"|\b(?:the\s+)?same\s+(?:place|spot|area|location|site)\b"
                       r"|\bwhere\s+you\s+(?:were|went|just\s+were)\b"
                       r"|\b(?:to|at|around|near|investigat\w*|survey\w*|explor\w*|check(?:\s+out)?|look\s+at|"
                       r"inspect\w*|visit|revisit|approach|sample|study|watch|observe|follow)\s+it\b")
_LIST_PICK_RE = re.compile(r"\b(?:the\s+)?(first|second|third|fourth|fifth|sixth|1st|2nd|3rd|4th|5th|6th|last)\s+"
                           r"(?:one|option|place|spot|target|site)\b")
_MENTIONED_AFTER_RE = re.compile(r"^\s*(?:that\s+|which\s+)?(?:you|we|r\d+|rover\s*\w+)\s+(?:just\s+)?(?:mentioned|found|saw|"
                                 r"spotted|reported|told me about|talked about|were talking about|flagged|marked|"
                                 r"listed|noted)\b")
_MENTIONED_BEFORE_RE = re.compile(r"\b(?:that|this|the\s+same|those|these)\s+(?:\w+\s+)?$")
_REPEAT_RE = re.compile(r"^(?:(?:can|could|would|will)\s+(?:you|we|u)\s+)?(?:please\s+)?(?:"
                        r"(?:do|try)\s+(?:that|it|this|the\s+same)(?:\s+thing)?(?:\s+again)?"
                        r"|(?:and\s+)?(?:the\s+)?same\s*(?:thing\s*)?(?:again|for|with|goes\s+for|here|please)?"
                        r"(?:\s+(?:you|him|her|them|that\s+one))?"
                        r"|repeat(?:\s+(?:that|it|this|the\s+last(?:\s+(?:command|order|one))?))?"
                        r"|again|one\s+more\s+time|ditto|you\s+too|too|as\s+well|also"
                        r"|(?:you\s+)?do\s+(?:it|that|the\s+same)\s+too)"
                        r"(?:\s+(?:too|as\s+well|again|please))*\s*[.!?]*$")
_SIGNS_RE = re.compile(r"\b(?:for\s+)?(?:any\s+|some\s+)?(?:signs?|traces?|evidence|hints?|indications?)\s+of\s+"
                       r"(?:any\s+)?(\w+(?: forms?)?)")
_LOOKFOR_RE = re.compile(r"\b(?:look|looking|search|searching|check|checking|scan|scanning|hunt|hunting)\s+"
                         r"(?:out\s+)?for\s+(?:any\s+|some\s+|more\s+)?(\w+(?: forms?)?)")
_LIFE_WORDS = re.compile(r"^(?:life|life forms?|lifeforms?|creatures?|animals?|movement|activity|organisms?|biology|"
                         r"aliens?|grazers?|herds?|living)$")
_NEAR_RE = re.compile(r"\b(?:near(?:est)?(?: to)?|close to|closest to|next to|beside)\s+(?:the\s+)?"
                      r"(?:(lander|landing site|base|home)\b"
                      rf"|(?:rover|robot|unit)\s*(?:#|no\.?|number)?\s*({_NUMWORD})\b|r(\d+)\b"
                      r"|(" + "|".join(_COLOURS + tuple(_COLOUR_SYNONYMS)) + r")\s+(?:rover|robot|one)\b)")
_ORDINAL_RE = re.compile(r"\b(first|second|third|fourth|fifth|sixth|1st|2nd|3rd|4th|5th|6th)\b"
                         r"|\b(?:number|no\.?|#|option)\s*(\d)\b")
_ORDINALS = {"first": 1, "second": 2, "third": 3, "fourth": 4, "fifth": 5, "sixth": 6,
             "1st": 1, "2nd": 2, "3rd": 3, "4th": 4, "5th": 5, "6th": 6}
_QUAL_RES = tuple((k, re.compile(rx)) for k, rx in (
    ("nearest", r"\b(?:nearest|closest|nearby|close by|near (?:me|you|here))\b"),
    ("farthest", r"\b(?:farthest|furthest|most distant|remotest|last)\b"),
    ("biggest", r"\b(?:biggest|largest|bigger|larger|big|large|main|major|huge)\b"),
    ("smallest", r"\b(?:smallest|smaller|small|little|tiny|minor)\b"),
    ("newest", r"\b(?:newest|latest|most recent|recent(?:ly found)?|new|just found|last found)\b"),
))
_DIR_ADJ_RE = re.compile(r"\b(north|south)[\s-]?(east|west)(?:ern)?\b|\b(north|south|east|west)(?:ern|wards?)?\b")
_SAMPLE_WORD_RE = re.compile(r"\b(?:sample|samples|sampling|drill|scoop|swab|sniff|collect|grab a sample|take a sample)\b")
_PLACE_SYNONYMS = {"lake": "water", "pond": "water", "river": "water", "stream": "water", "liquid": "water",
                   "herd": "lifeform", "herds": "lifeform", "creature": "lifeform", "creatures": "lifeform",
                   "animal": "lifeform", "life": "lifeform", "lifeforms": "lifeform", "grazer": "lifeform",
                   "spires": "spire", "crystal": "spire", "vents": "vent", "minerals": "mineral", "outcrop": "mineral",
                   "traces": "trace", "organic": "trace", "targets": "target", "base": "lander", "hazards": "hazard",
                   "rockfall": "hazard", "sand": "hazard"}
_QUAL_SYNONYMS = {"closest": "nearest", "near": "nearest", "furthest": "farthest", "largest": "biggest",
                  "big": "biggest", "small": "smallest", "latest": "newest", "new": "newest", "recent": "newest",
                  "northeast": "north-east", "northwest": "north-west", "southeast": "south-east",
                  "southwest": "south-west", "near rover": "near_rover", "nearrover": "near_rover"}


def near_reference(low: str, rovers: list | None = None):
    """'the lake near rover 2' -> ("r2", span); 'the one near the lander' -> ("lander", span); else (None, None)."""
    m = _NEAR_RE.search(low or "")
    if not m:
        return None, None
    if m.group(1):
        return "lander", m.span()
    if m.group(2):
        tok = m.group(2)
        return f"r{int(tok) if tok.isdigit() else _WORD_NUMS[tok]}", m.span()
    if m.group(3):
        return f"r{m.group(3)}", m.span()
    col = colour_name(m.group(4))
    match = [r["id"] for r in (rovers or []) if r.get("colour", colour_name(r.get("color"))) == col]
    return (match[0] if len(match) == 1 else None), m.span()


def _place_words(text: str) -> tuple:
    """(place kind, span) of the first place word in the text, or (None, None)."""
    best = None
    for kind, rx in _PLACE_RES:
        m = rx.search(text)
        if m and (best is None or m.start() < best[1][0]):
            best = (kind, m.span())
    return best if best else (None, None)


def _qualifier(text: str) -> tuple:
    """(qualifier, ordinal) from the words: ordinal > nearest/farthest > direction > size > newest."""
    m = _ORDINAL_RE.search(text)
    if m:
        return "ordinal", (_ORDINALS[m.group(1)] if m.group(1) else int(m.group(2)))
    for kind in ("nearest", "farthest"):
        if dict(_QUAL_RES)[kind].search(text):
            return kind, None
    d = _DIR_ADJ_RE.search(text)
    if d:
        return (f"{d.group(1)}-{d.group(2)}" if d.group(1) else d.group(3)), None
    for kind in ("biggest", "smallest", "newest"):
        if dict(_QUAL_RES)[kind].search(text):
            return kind, None
    return "none", None


def parse_place(low: str, rest: str, rover, target, rovers: list, polite: bool = False) -> dict | None:
    """A go_to_place command from the operator's words, or None if the message names no known kind of place
    with an action. low: the whole message (lower case); rest: rover mentions (and a target id) blanked."""
    text = rest
    life = False
    inv = bool(_INVESTIGATE_RE.search(text))
    for rx in (_SIGNS_RE, _LOOKFOR_RE):
        for m in list(rx.finditer(text)):
            obj = m.group(1)
            inv = True
            if _LIFE_WORDS.match(obj):
                life = True
                text = _blank(text, [m.span()])
    near, span = near_reference(low, rovers)
    if span is not None:
        text = _blank(text, [(a, b) for a, b in [span] if b <= len(text)])
    side = _EXPLORE_SIDE_RE.search(text)          # v7: "explore north of the lander": an area on that side
    side_dir = None
    if side:
        side_dir = f"{side.group(1)}-{side.group(2)}" if side.group(1) else side.group(3)
    place, pspan = _place_words(text)
    if place is None and target:
        place = "target"
    if place is None and life:
        place = "lifeform"
    pick = None
    if place is None:                             # v7: a chat reference ("go back there", "the second one")
        pick = _LIST_PICK_RE.search(text)
        if pick is not None or _THERE_RE.search(text):
            place = "there"
    if place is None:
        return None
    explore = bool(_EXPLORE_AREA_RE.search(text)) or side is not None
    if place == "lander" and side is None:
        if _DIR_RE.search(text) or (_RETURN_RE.search(text) and not inv and not explore):
            return None                   # "north of the lander" (explore), "back to the lander" (return home)
    mentioned = place != "there" and pspan is not None and bool(
        _MENTIONED_AFTER_RE.match(text[pspan[1]:]) or _MENTIONED_BEFORE_RE.search(text[:pspan[0]]))
    if not (inv or polite or explore or pick is not None or mentioned or _PLACE_VERB_RE.search(text)
            or (place == "there" and _SAMPLE_WORD_RE.search(text))):
        return None
    qtext = _blank(text, [pspan]) if pspan else text
    if side is not None:
        qtext = _blank(qtext, [side.span()])      # the direction names the side, not which one
    if place == "there":
        if pick is not None:
            w = pick.group(1)
            qual, ordinal = "ordinal", (-1 if w == "last" else _ORDINALS[w])
        else:
            qual, ordinal = "none", None
    elif mentioned:
        qual, ordinal = "mentioned", None         # "that lake", "the vent you mentioned"
    else:
        qual, ordinal = _qualifier(qtext)
    if qual == "none" and near is not None:
        qual = "near_rover"
    if target and place in ("mineral", "ice", "vent", "trace", "target"):
        qual, ordinal = "none", None
    out = {"type": "go_to_place", "rover": rover, "place": place,
           "target_id": target if place in ("mineral", "ice", "vent", "trace", "target") else None,
           "qualifier": qual, "ordinal": ordinal, "near_rover": near if qual == "near_rover" else None,
           "investigate": inv}
    if explore:
        out["explore"] = True
    if side_dir:
        out["direction"] = side_dir
    return out


def _ask(rover, topic: str, conf: float) -> dict:
    cmd = {"type": "question", "rover": rover, "topic": topic if topic in TOPICS else "unknown"}
    return {"command": cmd, "ack": "", "question": None, "confidence": conf}


def _strip_thousands(text: str) -> str:
    """'1,904 m2' -> '1904 m2' (a comma between digits followed by exactly three digits)."""
    return re.sub(r"(?<=\d),(?=\d{3}(?!\d))", "", text or "")


def _norm_direction(m) -> str:
    if m.group(1):
        return f"{m.group(1)}-{m.group(2)}"
    if m.group(3):
        return m.group(3)
    return {"ne": "north-east", "nw": "north-west", "se": "south-east", "sw": "south-west"}[m.group(4)]


def _blank(text: str, spans: list) -> str:
    chars = list(text)
    for a, b in spans:
        for i in range(a, b):
            chars[i] = " "
    return "".join(chars)


def _resolve_rover(low: str, rovers: list) -> tuple:
    """(rover id | "all" | None, text with the rover mentions blanked out, question or None). A rover named
    as a reference ("the lake near rover 2") is not the addressee: that phrase is blanked first."""
    ids = [r["id"] for r in rovers]
    m_near = _NEAR_RE.search(low)
    if m_near and not m_near.group(1):
        low = _blank(low, [m_near.span()])
    found, spans = [], []
    for m in _ROVERN_RE.finditer(low):
        tok = m.group(1) or m.group(2)
        n = int(tok) if tok.isdigit() else _WORD_NUMS[tok]
        found.append(f"r{n}")
        spans.append(m.span())
    for m in _RID_RE.finditer(low):
        found.append(f"r{m.group(1)}")
        spans.append(m.span())
    for r in rovers:
        name = str(r.get("name") or "").lower().strip()
        if name and not re.fullmatch(r"(rover|robot)\s*\d+", name):
            m = re.search(r"\b" + re.escape(name) + r"\b", low)
            if m:
                found.append(r["id"])
                spans.append(m.span())
    if not found:
        for m in _COLOUR_RE.finditer(low):
            col = colour_name(m.group(1))
            match = [r["id"] for r in rovers if r.get("colour") == col]
            if len(match) == 1:
                found.append(match[0])
                spans.append(m.span())
            elif len(match) > 1:
                return None, _blank(low, spans + [m.span()]), f"Which {col} rover: {' or '.join(match)}?"
    uniq = list(dict.fromkeys(found))
    cleaned = _blank(low, spans)
    if not uniq:
        m = _ALL_RE.search(low)
        if m:
            return "all", _blank(low, [m.span()]), None
        return None, cleaned, None
    if len(uniq) > 1:
        return None, cleaned, f"One rover at a time, or say all: which one, {' or '.join(uniq)}?"
    if ids and uniq[0] not in ids:
        return None, cleaned, f"There is no rover {uniq[0]}. Rovers: {', '.join(ids)}."
    return uniq[0], cleaned, None


def _fmt_xy(x: float, y: float) -> str:
    return f"({x:g}, {y:g})"


def command_ack(cmd: dict) -> str:
    """Short confirmation sentence for a command dict."""
    typ, rover = cmd.get("type"), cmd.get("rover")
    who = "All rovers" if rover == "all" else rover
    if typ == "go_to":
        verb = "exploring around" if cmd.get("explore") else "going to"
        return (f"{who} {verb} {_fmt_xy(cmd['x'], cmd['y'])}." if who
                else f"{verb[:1].upper() + verb[1:]} {_fmt_xy(cmd['x'], cmd['y'])}.")
    if typ == "explore_direction":
        return f"{who} exploring {cmd['direction']}." if who else f"Exploring {cmd['direction']}."
    if typ == "sample_target":
        return f"{who} going to sample {cmd['target_id']}." if who else f"Going to sample {cmd['target_id']}."
    if typ == "go_to_place":
        what = cmd.get("target_id") or {"lifeform": "herd", "hazard": "hazard site", "there": "place we talked about"}.get(
            cmd["place"], cmd["place"])
        if cmd.get("direction") and cmd.get("explore"):
            what = f"area {cmd['direction']} of the {what}"
        verb = "investigating" if cmd.get("investigate") else ("exploring around" if cmd.get("explore") else "going to")
        return f"{who} {verb} the {what}." if who else f"{verb[:1].upper() + verb[1:]} the {what}."
    if typ == "repeat":
        return f"{who} doing the same." if who else "Repeating the last command."
    if typ == "return_home":
        return f"{who} returning to the lander." if who else "Returning to the lander."
    if typ == "hold":
        return f"{who} holding position." if who else "Holding position."
    if typ == "resume":
        return f"{who} resuming exploration." if who else "Resuming exploration."
    if typ == "avoid_area":
        return f"Avoiding the area around {_fmt_xy(cmd['x'], cmd['y'])}, radius {cmd['radius']:g} m."
    if typ == "question":
        return f"Answering a question about {cmd.get('topic', 'status')}."
    return cmd.get("question") or _HELP


def _unclear(question: str) -> dict:
    return {"command": {"type": "unclear", "question": question}, "ack": question, "question": question,
            "confidence": 0.0}


def _done(cmd: dict, conf: float = 0.9) -> dict:
    return {"command": cmd, "ack": command_ack(cmd), "question": None, "confidence": conf}


def parse_command(text: str, fleet_context: dict | None = None) -> dict:
    """Deterministic keyword / regex parser for operator messages (the fallback when Gemma is
    offline or invalid; also usable directly for an instant preview). Returns
        {"command": {"type": ..., ...}, "ack": str, "question": str | None, "confidence": float}
    with command one of
        {"type": "go_to", "rover", "x", "y"}          {"type": "explore_direction", "rover", "direction"}
        {"type": "sample_target", "rover", "target_id"}
        {"type": "return_home" | "hold" | "resume", "rover"}     rover: "r1".."rN", "all" or None
        {"type": "avoid_area", "x", "y", "radius"}    {"type": "unclear", "question"}
        {"type": "question", "rover", "topic"}        the operator asks for information (TOPICS)
        {"type": "go_to_place", "rover", "place", "target_id", "qualifier", "ordinal", "near_rover", "investigate"}
                                                      a place the fleet knows (PLACES, QUALIFIERS; protocol v6)
        {"type": "repeat", "rover"}                   the last command again ("do that again", "same for rover 3")
    Protocol v7: go_to and go_to_place carry "explore": True to stay and explore around the place ("explore
    around the lake", "go to 10, -5 and look around"), and "direction" for an area on that side of it ("explore
    north of the lander"). place "there" is the place the chat talked about last (qualifier "ordinal": an entry
    of the last list, -1 the last one); qualifier "mentioned" picks the one the chat named ("that lake").
    fleet_context: {"rovers": [{"id", "name", "color", "x", "y"}], "targets": [{"id", ...}], ...}
    (raw or already compacted)."""
    ctx = _fleet_context(fleet_context)       # idempotent: raw or already compacted
    rovers = ctx["rovers"]
    target_ids = [tg["id"] for tg in ctx["targets"]]
    low = " ".join(str(text or "").lower().split())
    if not low:
        return _unclear(_HELP)
    rover, rest, question = _resolve_rover(low, rovers)
    if question:
        return _unclear(question)
    if _REPEAT_RE.match(_LEAD_RE.sub("", " ".join(rest.split())).strip(" ,;:")):
        return _done({"type": "repeat", "rover": rover})       # v7: "do that again", "same for rover 3"
    q_strength, q_topic = question_intent(low, rovers)
    if q_strength == "strong":
        return _ask(rover, q_topic, 0.9)

    tm = _TARGET_RE.search(rest)
    target = f"t{tm.group(1)}" if tm else None
    if tm:
        rest = _blank(rest, [tm.span()])
    xy = _XY_RE.search(rest) or _PAIR_RE.search(rest)
    pair = (float(xy.group(1)), float(xy.group(2))) if xy else None
    dm = _DIR_RE.search(rest)

    if _AVOID_RE.search(rest):
        rest2 = _blank(rest, [xy.span()]) if xy else rest
        rm = _RADIUS_RE.search(rest2)
        radius = float(rm.group(1) or rm.group(2)) if rm else DEFAULT_AVOID_RADIUS
        if pair is None:
            nums = [v for _, v in numbers_in_text(_blank(rest2, [rm.span()]) if rm else rest2)]
            if len(nums) >= 2:
                pair = (nums[0], nums[1])
        if pair is None:
            return _unclear("Which area should the rovers avoid? Give x, y and a radius, like 'avoid 12, -4 radius 3'.")
        return _done({"type": "avoid_area", "x": pair[0], "y": pair[1], "radius": max(0.5, min(30.0, radius))})
    looking = target and re.search(r"\b(?:look|survey|observe|watch)", rest) and not _SAMPLE_WORD_RE.search(rest)
    if target and not looking and (_SAMPLE_RE.search(rest) or not rest.strip() or rover is not None):
        if target_ids and target not in target_ids:
            return _unclear(f"I do not know target {target}. Known targets: {', '.join(target_ids)}.")
        return _done({"type": "sample_target", "rover": rover, "target_id": target})
    if pair is None:                  # protocol v6: a place the fleet knows ("go to the water source")
        if target and target_ids and target not in target_ids:
            return _unclear(f"I do not know target {target}. Known targets: {', '.join(target_ids)}.")
        place = parse_place(low, rest, rover, target, rovers)
        if place is not None:
            return _done(place)
    if _RETURN_RE.search(rest):
        return _done({"type": "return_home", "rover": rover})
    if _RESUME_RE.search(rest):
        return _done({"type": "resume", "rover": rover})
    if _HOLD_RE.search(rest):
        return _done({"type": "hold", "rover": rover})
    explore = bool(_EXPLORE_AREA_RE.search(rest))           # v7: "explore around 10, -5"
    if pair is not None:
        return _done(dict({"type": "go_to", "rover": rover, "x": pair[0], "y": pair[1]},
                          **({"explore": True} if explore else {})))
    if dm:
        return _done({"type": "explore_direction", "rover": rover, "direction": _norm_direction(dm)})
    if _EXPLORE_RE.search(rest):
        return _done({"type": "resume", "rover": rover})
    nums = [v for _, v in numbers_in_text(rest)]
    if len(nums) == 2 and (_GO_RE.search(rest) or explore):
        return _done(dict({"type": "go_to", "rover": rover, "x": nums[0], "y": nums[1]},
                          **({"explore": True} if explore else {})))
    if q_strength == "weak" or (q_strength is None and q_topic != "unknown"):    # "battery?", "any water"
        return _ask(rover, q_topic, 0.7)
    return _unclear(_HELP)


# ====================================================================== validation
def fallback(req: Request, error: str, latency_ms: float = 0.0, model: str = "") -> Result:
    r = Result(ok=False, kind=req.kind, source="fallback", rover_id=req.rover_id, t=req.t, ticket=req.ticket,
               latency_ms=round(latency_ms, 1), error=error, model=model)
    if req.kind == "choice":
        r.choice = req.default
    elif req.kind in ("interpretation", "describe"):
        r.action = req.default
    elif req.kind == "answer":
        r.answer = req.default
        r.text = req.text
    else:
        p = req.parsed or _unclear(_HELP)
        r.command = dict(p["command"])
        r.ack = p["ack"]
        r.question = p["question"]
        r.confidence = p["confidence"]
        r.text = req.text
    return r


def _parse_json(content: str):
    s = (content or "").strip()
    if s.startswith("```"):
        s = s.strip("`")
        if s[:4].lower() == "json":
            s = s[4:]
    try:
        return json.loads(s)
    except (ValueError, TypeError):
        pass
    a, b = s.find("{"), s.rfind("}")
    if 0 <= a < b:
        try:
            return json.loads(s[a:b + 1])
        except (ValueError, TypeError):
            pass
    raise ValueError("bad JSON")


def _confidence(v) -> float:
    if isinstance(v, bool) or v is None:
        raise ValueError("missing confidence")
    try:
        c = float(v)
    except (TypeError, ValueError):
        raise ValueError(f"confidence {v!r} is not a number") from None
    if 1.0 < c <= 100.0:
        c /= 100.0                          # "85" meant 85%
    if not (0.0 <= c <= 1.0) or not math.isfinite(c):
        raise ValueError(f"confidence {v!r} not in 0..1")
    return c


def _coord_ok(v: float, coords: list) -> bool:
    """A coordinate must be a number from the message or the fleet data (sign matters)."""
    return any(abs(v - w) <= (0.51 if abs(w) >= 2 else 0.051) for w in coords)


def _number(data: dict, key: str) -> float:
    v = data.get(key)
    if isinstance(v, bool) or v is None:
        raise ValueError(f"missing {key}")
    try:
        f = float(v)
    except (TypeError, ValueError):
        raise ValueError(f"{key} {v!r} is not a number") from None
    if not math.isfinite(f):
        raise ValueError(f"{key} is not finite")
    return f


_LABEL_RE = re.compile(r"^\s*(?:ack|ok|question|answer|reply|response)\s*[:=\-]+\s*", re.I)
OWN_POSITION_M = 2.0       # a go_to this close to the rover's own position is not a real destination


def _strip_label(s: str) -> str:
    """'ack: Rover 2 is going north.' -> 'Rover 2 is going north.' (small models echo field names)."""
    s = str(s or "")
    for _ in range(2):
        s = _LABEL_RE.sub("", s)
    return s.strip()


def _fix_go_to(req: Request, cmd: dict, data: dict) -> dict:
    """Post-check a go_to. Small models sometimes answer 'go check out the area north of the lander'
    with go_to at the rover's own position (a number they saw in the fleet list). If the message names
    a direction and gives no coordinates, or the point is within 2 m of the rover itself, the operator
    meant explore_direction; without a direction the command is unclear."""
    if cmd.get("type") != "go_to":
        return cmd
    low = (req.text or "").lower()
    rovers = (req.payload.get("fleet") or {}).get("rovers", [])
    _, cleaned, _ = _resolve_rover(low, rovers)
    explicit = len(numbers_in_text(_TARGET_RE.sub(" ", cleaned))) >= 2
    d = str(data.get("direction") or "").strip().lower().replace(" ", "-").replace("_", "-")
    d = {"northeast": "north-east", "northwest": "north-west", "southeast": "south-east",
         "southwest": "south-west"}.get(d, d)
    if d not in DIRECTIONS:
        m = _DIR_RE.search(cleaned)
        d = _norm_direction(m) if m else None
    rv = cmd.get("rover")
    own = next((r for r in rovers if r.get("id") == rv and "x" in r and "y" in r), None)
    at_self = own is not None and math.hypot(cmd["x"] - own["x"], cmd["y"] - own["y"]) <= OWN_POSITION_M
    if d and (at_self or not explicit):
        return {"type": "explore_direction", "rover": rv, "direction": d}
    if at_self:
        who = rv or "the rover"
        return {"type": "unclear", "question": f"Where should {who} go? Give x, y or a direction."}
    return cmd


def _explicit_xy(req: Request) -> bool:
    low = (req.text or "").lower()
    _, cleaned, _ = _resolve_rover(low, (req.payload.get("fleet") or {}).get("rovers", []))
    return len(numbers_in_text(_TARGET_RE.sub(" ", cleaned))) >= 2


def _place_command(req: Request, data: dict, rv) -> dict:
    """Gemma's go_to_place, checked: every enum, ids that exist. Raises ValueError (the parser decides)."""
    def norm(v) -> str:
        s = str(v if v is not None else "").strip().lower().replace("_", " ")
        return " ".join(s.split())

    place = norm(data.get("place"))
    place = _PLACE_SYNONYMS.get(place, place)
    tid = str(data.get("target_id") or "").strip()
    if tid.lower() in ("", "none", "null", "0"):
        tid = None
    if tid is not None:
        if req.target_ids and tid not in req.target_ids:
            raise ValueError(f"target {tid!r} not in {list(req.target_ids)}")
        if not req.target_ids and not re.fullmatch(r"t\d+", tid):
            raise ValueError(f"target {tid!r} is not a target id")
    if place not in PLACES:
        if tid is None:
            raise ValueError(f"place {data.get('place')!r} not in {list(PLACES)}")
        place = "target"
    if place not in ("mineral", "ice", "vent", "trace", "target"):
        tid = None
    qual = norm(data.get("qualifier")) or "none"
    qual = _QUAL_SYNONYMS.get(qual, qual).replace(" ", "-") if qual not in ("near rover",) else "near_rover"
    if qual not in QUALIFIERS:
        raise ValueError(f"qualifier {data.get('qualifier')!r} not in {list(QUALIFIERS)}")
    if place == "there" and qual != "ordinal":
        qual = "none"                         # the place talked about last: nothing to choose between
    ordinal = None
    if qual == "ordinal":
        try:
            ordinal = int(float(data.get("ordinal")))
        except (TypeError, ValueError):
            ordinal = 0
        if not 1 <= ordinal <= 9:
            qual, ordinal = "none", None          # no usable number: ask which one
    near = norm(data.get("near_rover"))
    if near in ("", "none", "null", "0"):
        near = None
    elif near in ("lander", "base", "the lander"):
        near = "lander"
    elif near not in (req.ids or ()) and not (not req.ids and re.fullmatch(r"r\d+", near)):
        raise ValueError(f"near_rover {data.get('near_rover')!r} is not a rover id or lander")
    if qual == "near_rover" and near is None:
        qual = "none"
    inv = _flag(data.get("investigate"))
    out = {"type": "go_to_place", "rover": rv, "place": place, "target_id": tid, "qualifier": qual,
           "ordinal": ordinal, "near_rover": near if qual == "near_rover" else None, "investigate": inv}
    if _flag(data.get("explore")):
        out["explore"] = True
    return out


def _flag(v) -> bool:
    return v.strip().lower() in ("true", "yes", "1") if isinstance(v, str) else bool(v)


def _fix_place(req: Request, cmd: dict) -> dict:
    """Post-check place commands with the keyword parser. When the operator's words name a known kind of place
    (the parser made a go_to_place) and Gemma answered go_to without coordinates in the message (often the
    rover's own position), explore_direction, unclear, return_home or sample_target, the place command stands.
    Gemma's go_to_place keeps the place the operator's words name, and picks up a qualifier or investigate the
    parser saw. Explicit coordinates always mean go_to."""
    p = (req.parsed or {}).get("command") or {}
    typ = cmd.get("type")
    if p.get("type") == "go_to_place" and p.get("place") == "there" and typ != "go_to_place" and \
            not ((req.payload.get("fleet") or {}).get("chat") or {}).get("refs"):
        return cmd                        # "go there" with nothing in the chat to refer to: the model's reading
    if p.get("type") == "go_to_place":
        if typ == "go_to_place":
            out = dict(cmd, place=p["place"])
            if p.get("target_id"):
                out["target_id"] = p["target_id"]
            if out.get("place") not in ("mineral", "ice", "vent", "trace", "target"):
                out["target_id"] = None
            if (cmd.get("qualifier") in (None, "none") and p.get("qualifier") not in (None, "none")) or \
                    p["place"] == "there" or p.get("qualifier") == "mentioned":
                out.update(qualifier=p["qualifier"], ordinal=p.get("ordinal"), near_rover=p.get("near_rover"))
            out["investigate"] = bool(cmd.get("investigate")) or bool(p.get("investigate"))
            # v7: the operator's words decide whether it stays to explore and on which side
            out.pop("explore", None)
            out.pop("direction", None)
            if p.get("explore"):
                out["explore"] = True
            if p.get("direction"):
                out["direction"] = p["direction"]
            return out
        if typ in ("go_to", "explore_direction", "unclear", "return_home", "sample_target", "resume", "hold",
                   "repeat") and not _explicit_xy(req):
            return dict(p)
        return cmd
    if typ == "go_to_place" and p.get("type") == "go_to" and _explicit_xy(req):
        return dict(p)
    if typ == "go_to" and p.get("type") == "go_to":
        out = {k: v for k, v in cmd.items() if k != "explore"}
        if p.get("explore"):
            out["explore"] = True
        return out
    return cmd


def _fix_refs(req: Request, cmd: dict) -> dict:
    """Post-check 'repeat' (protocol v7) with the keyword parser: "do that again" and "same for rover 3" are a
    repeat whatever the model made of them; a model's repeat for a message that names a concrete order is that
    order."""
    p = (req.parsed or {}).get("command") or {}
    typ = cmd.get("type")
    if p.get("type") == "repeat" and typ != "repeat":
        return dict(p)
    if typ == "repeat" and p.get("type") not in (None, "repeat", "unclear", "question"):
        return dict(p)
    return cmd


def _fix_intent(req: Request, cmd: dict) -> dict:
    """Post-check command vs question with the keyword rules (small models mix them up): a message
    that starts with a question word ("did you sample t4?") is a question, not an order; one that
    starts with an order ("go to 10, 5?") is not a question. A question about an unknown topic takes
    the keyword topic when there is one."""
    rovers = (req.payload.get("fleet") or {}).get("rovers", [])
    strength, topic = question_intent(req.text or "", rovers)
    typ = cmd.get("type")
    if typ == "question":
        p = (req.parsed or {}).get("command") or {}
        if strength == "command" and p.get("type") not in (None, "unclear", "question"):
            return dict(p)
        if cmd.get("topic") in (None, "unknown") and topic != "unknown":
            cmd = dict(cmd, topic=topic)
        return cmd
    if strength == "strong" or (typ == "unclear" and (strength == "weak" or (strength is None and topic != "unknown"))):
        rv = cmd.get("rover")
        if rv is None:
            rv, _ = _q_core(req.text or "", rovers)
        return {"type": "question", "rover": rv, "topic": topic}
    return cmd


def _validate_answer(req: Request, data: dict, model: str) -> Result:
    """An answer to an operator question: plain text, 1 to 3 sentences, grounded in the fact lines."""
    def bad(msg: str) -> Result:
        return fallback(req, f"invalid answer: {msg}", model=model)

    text = data.get("text")
    if not isinstance(text, str) or not text.strip():
        return bad("missing text")
    text = " ".join(text.replace(EM_DASH, ", ").split())
    text = _strip_label(text)
    sp = req.payload.get("speaker", "")
    for prefix in (f"{sp}:", f"{sp} :"):
        if sp and text.lower().startswith(prefix.lower()):
            text = text[len(prefix):].strip()
    if text[:1] in "\"'" and text[-1:] in "\"'":
        text = text[1:-1].strip()
    if not text:
        return bad("empty text")
    words = text.split()
    if len(words) < 2:
        return bad("too short")
    if len(words) > MAX_ANSWER_WORDS or len(text) > MAX_ANSWER_CHARS:
        return bad(f"too long ({len(words)} words)")
    n_sent = len(re.findall(r"[.!?]+(?=\s|$)", text))
    if n_sent > MAX_ANSWER_SENTENCES:
        return bad(f"{n_sent} sentences")
    if "{" in text or "}" in text:
        return bad("looks like JSON, not a sentence")
    plain = _strip_thousands(text)
    for written, value in numbers_in_text(plain):
        if not any(_matches_rounded(written, value, w) for w in req.inputs):
            return bad(f"number {written} is not in the fact sheet")
    for m in _ID_RE.finditer(text.lower()):
        if m.group(0) not in req.allowed_ids:
            return bad(f"id {m.group(0)} is not in the fact sheet")
    if text[-1] not in ".!?":
        text += "."
    return Result(ok=True, kind="answer", source="gemma", rover_id=req.rover_id, t=req.t, ticket=req.ticket,
                  answer=text, text=req.text, confidence=1.0, model=model,
                  cited=cited_numbers(plain, req.inputs))


def _matches_rounded(written: str, r: float, w: float) -> bool:
    """_matches, plus rounding to tens, hundreds or thousands when written so (1904 as 1,900 or 2,000)."""
    if _matches(written, r, w):
        return True
    if "." in written:
        return False
    digits = written.lstrip("+-")
    zeros = len(digits) - len(digits.rstrip("0"))
    if zeros and digits.rstrip("0"):
        return abs(abs(r) - abs(w)) <= 0.5 * 10.0 ** zeros + 1e-9
    return False


def _validate_command(req: Request, data: dict, min_confidence: float, model: str) -> Result:
    def bad(msg: str) -> Result:
        return fallback(req, f"invalid answer: {msg}", model=model)

    typ = data.get("command")
    if typ not in COMMANDS:
        return bad(f"command {typ!r} not in {list(COMMANDS)}")
    try:
        conf = _confidence(data.get("confidence"))
    except ValueError as e:
        return bad(str(e))
    rv = data.get("rover")
    rv = None if rv is None else str(rv).strip()
    if rv is not None and rv.lower() in ("", "none", "null"):
        rv = None
    elif rv is not None and rv.lower() == "all":
        rv = "all"
    elif rv is not None and (rv not in req.ids if req.ids else not re.fullmatch(r"r\d+", rv)):
        return bad(f"rover {rv!r} not in {list(req.ids)}")
    try:
        if typ == "go_to":
            x, y = _number(data, "x"), _number(data, "y")
            if not (_coord_ok(x, req.coords) and _coord_ok(y, req.coords)):
                return bad(f"coordinates ({x:g}, {y:g}) are not in the message or the fleet data")
            cmd = {"type": typ, "rover": rv, "x": x, "y": y}
            if _flag(data.get("explore")):
                cmd["explore"] = True
        elif typ == "avoid_area":
            x, y = _number(data, "x"), _number(data, "y")
            if not (_coord_ok(x, req.coords) and _coord_ok(y, req.coords)):
                return bad(f"coordinates ({x:g}, {y:g}) are not in the message or the fleet data")
            try:
                radius = _number(data, "radius")
            except ValueError:
                radius = DEFAULT_AVOID_RADIUS
            if radius <= 0:
                radius = DEFAULT_AVOID_RADIUS
            cmd = {"type": typ, "x": x, "y": y, "radius": max(0.5, min(30.0, radius))}
        elif typ == "explore_direction":
            d = str(data.get("direction") or "").strip().lower().replace(" ", "-").replace("_", "-")
            d = {"northeast": "north-east", "northwest": "north-west", "southeast": "south-east",
                 "southwest": "south-west", "ne": "north-east", "nw": "north-west", "se": "south-east",
                 "sw": "south-west"}.get(d, d)
            if d not in DIRECTIONS:
                return bad(f"direction {d!r} not in {list(DIRECTIONS)}")
            cmd = {"type": typ, "rover": rv, "direction": d}
        elif typ == "sample_target":
            tid = str(data.get("target_id") or "").strip()
            if not tid or tid.lower() == "none":
                return bad("missing target_id")
            if req.target_ids and tid not in req.target_ids:
                return bad(f"target {tid!r} not in {list(req.target_ids)}")
            if not req.target_ids and not re.fullmatch(r"t\d+", tid):
                return bad(f"target {tid!r} is not a target id")
            cmd = {"type": typ, "rover": rv, "target_id": tid}
        elif typ in ("return_home", "hold", "resume", "repeat"):
            cmd = {"type": typ, "rover": rv}
        elif typ == "go_to_place":
            cmd = _place_command(req, data, rv)
        elif typ == "question":
            topic = str(data.get("topic") or "").strip().lower()
            cmd = {"type": "question", "rover": rv, "topic": topic if topic in TOPICS else "unknown"}
        else:
            q = _clean_text(data.get("question") or "", 30) or _HELP
            cmd = {"type": "unclear", "question": q}
    except ValueError as e:
        return bad(str(e))
    cmd = _fix_go_to(req, cmd, data)
    cmd = _fix_place(req, cmd)
    cmd = _fix_refs(req, cmd)
    cmd = _fix_intent(req, cmd)
    if cmd["type"] == "unclear":
        cmd["question"] = _strip_label(cmd["question"]) or _HELP
    ack = _strip_label(_clean_text(data.get("ack") or ""))
    if not ack or cmd["type"] != typ:
        ack = command_ack(cmd)
    if cmd["type"] == "unclear":
        ack = cmd["question"]
    res = Result(ok=True, kind="command", source="gemma", rover_id="operator", t=req.t, ticket=req.ticket,
                 command=cmd, ack=ack, text=req.text, confidence=round(conf, 3), model=model,
                 question=cmd.get("question"))
    if cmd["type"] not in ("unclear", "question") and conf < min_confidence:    # asking is always harmless
        q = f"Did you mean: {ack.rstrip('.')}?"
        res.model_command = cmd
        res.command = {"type": "unclear", "question": q}
        res.question = res.ack = q
    return res


def validate(req: Request, content: str, min_confidence: float = 0.55, model: str = "") -> Result:
    """Check a raw model answer against the request. Valid: Result(ok=True, source="gemma").
    Anything wrong: a fallback Result whose error says what."""
    def bad(msg: str) -> Result:
        return fallback(req, f"invalid answer: {msg}", model=model)

    if not (content or "").strip():
        return bad("empty")
    try:
        data = _parse_json(content)
    except ValueError:
        return bad(f"bad JSON {content.strip()[:60]!r}")
    if not isinstance(data, dict):
        return bad("not a JSON object")
    if req.kind == "command":
        return _validate_command(req, data, min_confidence, model)
    if req.kind == "answer":
        return _validate_answer(req, data, model)
    reason = data.get("reason")
    if not isinstance(reason, str) or not reason.strip():
        return bad("missing reason")
    reason = _clean_text(reason)
    try:
        conf = _confidence(data.get("confidence"))
    except ValueError as e:
        return bad(str(e))
    cited = cited_numbers(reason, req.inputs)
    if not cited:
        return bad(f"reason quotes no input reading: {reason[:80]!r}")
    res = Result(ok=True, kind=req.kind, source="gemma", rover_id=req.rover_id, t=req.t, ticket=req.ticket,
                 confidence=round(conf, 3), reason=reason, cited=cited, model=model)
    if req.kind == "choice":
        choice = data.get("choice")
        if not isinstance(choice, str) or not choice:
            return bad("missing choice")
        if choice not in req.ids:
            return bad(f"choice {choice!r} not in candidates {list(req.ids)}")
        res.choice = choice
        return res
    cls = data.get("classification")
    if not isinstance(cls, str) or not cls:
        return bad("missing classification")
    cls = cls.strip().lower().replace(" ", "_")
    if cls not in req.classes:
        return bad(f"classification {cls!r} not in {list(req.classes)}")
    if req.kind == "describe":
        label = data.get("label")
        if not isinstance(label, str) or not label.strip():
            return bad("missing label")
        action = data.get("action")
        if action not in req.options or action == "ask_operator":
            return bad(f"action {action!r} not in {[o for o in req.options if o != 'ask_operator']}")
        res.classification = cls
        res.label = _clean_text(label, MAX_LABEL_WORDS).rstrip(".")
        res.model_action = action
        res.action = "ask_operator" if conf < min_confidence else action
        return res
    action = data.get("action")
    if not isinstance(action, str) or not action:
        return bad("missing action")
    if action not in req.options:
        return bad(f"action {action!r} not in options {list(req.options)}")
    res.classification = cls
    res.model_action = action
    res.action = "ask_operator" if conf < min_confidence else action
    if res.action == "ask_operator":
        res.question = operator_question(req, cls, conf, reason)
    return res


# ====================================================================== backends
class Backend:
    """What GemmaPlanner needs from a model server."""
    name = "base"
    model = ""
    requested_model = ""
    url = ""

    def check(self, timeout: float) -> tuple:
        """("ready" | "no_model" | "offline", detail). Cheap: called every ~15 s. May switch
        self.model to an installed fallback model."""
        raise NotImplementedError

    def warmup(self, timeout: float) -> str | None:
        """Load the model into memory. Returns an error string, or None when done."""
        return None

    def chat(self, req: Request, system: str, user: str, schema: dict, timeout: float) -> tuple:
        """(raw answer text, meta dict). Raises GemmaTimeout, GemmaOffline or GemmaError."""
        raise NotImplementedError


class _ThinkRejected(GemmaError):
    pass


def _find_model(model: str, names: list) -> str | None:
    """The installed name matching `model` ("gemma3" matches "gemma3:latest"), or None."""
    for n in names:
        if not n:
            continue
        if n == model:
            return n
        if ":" not in model and (n == model + ":latest" or n.split(":", 1)[0] == model):
            return n
        if model.endswith(":latest") and n == model.split(":", 1)[0]:
            return n
    return None


class OllamaBackend(Backend):
    """Ollama HTTP API: GET /api/tags (online check and model choice), POST /api/chat (answers
    with a JSON schema `format`). Uses the requested model if installed, else the first installed
    fallback model."""
    name = "ollama"

    def __init__(self, urls, model: str = DEFAULT_MODEL, num_predict: int = 96, keep_alive: str = "30m",
                 think: bool | None = False, temperature: float = 0.0, fallback_models=FALLBACK_MODELS):
        self.urls = tuple(urls) if isinstance(urls, (list, tuple)) else (str(urls),)
        self.urls = tuple(u.rstrip("/") for u in self.urls)
        self.url = self.urls[0]
        self.requested_model = model
        self.model = model
        self.fallback_models = tuple(fallback_models or ())
        self.num_predict = int(num_predict)
        self.keep_alive = keep_alive
        self.think = think
        self.temperature = temperature
        self._send_think = think is not None
        # Ollama is local: never route it through HTTP(S)_PROXY
        self._opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))

    @classmethod
    def from_config(cls, cfg: GemmaConfig) -> "OllamaBackend":
        return cls(cfg.urls, cfg.model, cfg.num_predict, cfg.keep_alive, cfg.think, cfg.temperature,
                   cfg.fallback_models)

    # ---- HTTP
    def _request(self, url: str, path: str, body: dict | None, timeout: float) -> dict:
        data = None if body is None else json.dumps(body).encode("utf-8")
        req = urllib.request.Request(url + path, data=data, method="GET" if body is None else "POST",
                                     headers={"Content-Type": "application/json"})
        try:
            with self._opener.open(req, timeout=max(0.01, timeout)) as resp:
                raw = resp.read()
        except urllib.error.HTTPError as e:
            try:
                text = e.read()[:400].decode("utf-8", "replace")
                msg = str(json.loads(text).get("error", text))
            except Exception:                 # noqa: BLE001
                msg = str(e.reason)
            if e.code == 400 and "think" in msg.lower():
                raise _ThinkRejected(msg) from None
            if e.code == 404:
                raise GemmaError(f"HTTP 404: {msg} (run: ollama pull {self.model})") from None
            raise GemmaError(f"HTTP {e.code}: {msg}"[:200]) from None
        except urllib.error.URLError as e:
            if isinstance(e.reason, (TimeoutError, socket.timeout)):
                raise GemmaTimeout("timeout") from None
            raise GemmaOffline(f"{url}: {e.reason}") from None
        except (TimeoutError, socket.timeout):
            raise GemmaTimeout("timeout") from None
        except (ConnectionError, OSError) as e:
            raise GemmaOffline(f"{url}: {e}") from None
        except http.client.HTTPException as e:
            raise GemmaError(f"bad HTTP response: {type(e).__name__}") from None
        try:
            out = json.loads(raw.decode("utf-8"))
        except (ValueError, UnicodeDecodeError):
            raise GemmaError("server sent bad JSON") from None
        if not isinstance(out, dict):
            raise GemmaError("server sent a non-object")
        return out

    def _post_chat(self, body: dict, timeout: float) -> dict:
        """POST /api/chat; if the server rejects the think field, drop it for good and retry."""
        deadline = time.perf_counter() + timeout
        if self._send_think:
            body["think"] = bool(self.think)
        try:
            return self._request(self.url, "/api/chat", body, timeout)
        except _ThinkRejected:
            self._send_think = False          # older server, or a model without a thinking switch
            body.pop("think", None)
            left = deadline - time.perf_counter()
            if left < 0.05:
                raise GemmaTimeout("timeout") from None
            return self._request(self.url, "/api/chat", body, left)

    # ---- Backend API
    def check(self, timeout: float) -> tuple:
        errors = []
        order = (self.url,) + tuple(u for u in self.urls if u != self.url)   # last good URL first
        for url in order:
            try:
                data = self._request(url, "/api/tags", None, timeout)
            except GemmaError as e:
                errors.append(str(e) if url in str(e) else f"{url}: {e}")
                continue
            self.url = url
            names = [m.get("name") or m.get("model") for m in data.get("models", []) if isinstance(m, dict)]
            pick = None
            for want in (self.requested_model,) + self.fallback_models:
                pick = _find_model(want, names)
                if pick:
                    break
            if pick:
                if pick != self.model:
                    self.model = pick
                    self._send_think = self.think is not None
                note = "" if pick == _find_model(self.requested_model, names) else \
                    f" (using {pick}: {self.requested_model} is not installed)"
                return "ready", url + note
            have = ", ".join(n for n in names if n)[:120] or "none"
            return "no_model", (f"none of {self.requested_model}, {', '.join(self.fallback_models)} at {url} "
                                f"(have: {have}). Run: ollama pull {self.requested_model}")
        return "offline", "; ".join(errors)[:300]

    def warmup(self, timeout: float) -> str | None:
        """One tiny request (1 output token) so the model is loaded and kept for keep_alive; the
        first real decision then does not pay the ~30 s load."""
        body = {"model": self.model, "messages": [{"role": "user", "content": "Reply OK."}], "stream": False,
                "options": {"temperature": 0, "num_predict": 1}, "keep_alive": self.keep_alive}
        try:
            self._post_chat(body, timeout)
            return None
        except GemmaError as e:
            return f"warmup of {self.model} failed: {e}"

    def chat(self, req: Request, system: str, user: str, schema: dict, timeout: float) -> tuple:
        msg = {"role": "user", "content": user}
        if req.image_b64:
            msg["images"] = [req.image_b64]
        body = {"model": self.model, "messages": [{"role": "system", "content": system}, msg],
                "stream": False, "format": schema,
                "options": {"temperature": self.temperature,
                            "num_predict": max(self.num_predict, req.num_predict or 0)},
                "keep_alive": self.keep_alive}
        data = self._post_chat(body, timeout)
        if data.get("error"):
            raise GemmaError(str(data["error"])[:200])
        m = data.get("message") or {}
        content = m.get("content") or ""
        if not content.strip() and m.get("thinking"):
            raise GemmaError("empty answer: the model spent its tokens thinking (set GEMMA_THINK=0)")
        meta = {k: data.get(k) for k in ("total_duration", "load_duration", "prompt_eval_count",
                                         "prompt_eval_duration", "eval_count", "eval_duration")}
        return content, meta


class FakeBackend(Backend):
    """Deterministic stand-in for the model (tests, and GEMMA_FAKE=1 demos without Ollama).
    Same answer format as Gemma, so the whole pipeline (queue, validation, stats) is exercised.
    latency_s: fixed delay, or a list of delays used in turn."""
    name = "fake"

    def __init__(self, latency_s=0.5, online: bool = True, model: str = "fake-gemma"):
        self.latency_s = latency_s
        self.online = online
        self.model = self.requested_model = model
        self.url = "fake://"
        self._n = 0

    def check(self, timeout: float) -> tuple:
        return ("ready", "fake backend") if self.online else ("offline", "fake backend set offline")

    def _delay(self) -> float:
        lat = self.latency_s
        if isinstance(lat, (list, tuple)):
            lat = lat[self._n % len(lat)] if lat else 0.0
        self._n += 1
        return max(0.0, float(lat))

    def chat(self, req: Request, system: str, user: str, schema: dict, timeout: float) -> tuple:
        if not self.online:
            raise GemmaOffline("fake backend set offline")
        d = self._delay()
        if d > timeout:
            time.sleep(timeout)
            raise GemmaTimeout("timeout")
        time.sleep(d)
        if req.kind == "choice":
            ans = self._choice(req)
        elif req.kind == "interpretation":
            ans = self._interpret(req)
        elif req.kind == "describe":
            ans = self._describe(req)
        elif req.kind == "answer":
            ans = self._answer(req)
        else:
            ans = self._command(req)
        return json.dumps(ans), {"prompt_eval_count": estimate_tokens(system + user), "eval_count": 40}

    def _choice(self, req: Request) -> dict:
        sens = req.payload.get("sensors", {})
        cands = req.payload["candidates"]
        bat = sens.get("battery")
        frac = None
        if isinstance(bat, (int, float)) and not isinstance(bat, bool):
            frac = bat / 100.0 if bat > 1.0 else float(bat)
        by_kind = {}
        for c in cands:
            by_kind.setdefault(c["kind"], []).append(c)

        def cells(c):
            m = re.search(r"(\d+)\s*cells", c["summary"])
            return int(m.group(1)) if m else 0

        if frac is not None and frac < 0.3 and "return" in by_kind:
            pick, why = by_kind["return"][0], "battery low"
        elif "target" in by_kind:
            pick, why = by_kind["target"][0], "science target in reach"
        elif "frontier" in by_kind:
            pick, why = max(by_kind["frontier"], key=cells), "largest frontier"
        else:
            pick, why = cands[0], "only safe option"
        bits = [why]
        if frac is not None:
            bits.append(f"battery {frac * 100:.0f}%")
        # quote the summary's numbers that matter (cells, slope, battery after return or on arrival)
        parts = [p.strip() for p in pick["summary"].split(",")]
        keep = [p for p in parts if re.search(r"cells|slope|battery|value|tubes", p)] or parts[:2]
        bits.extend(keep[:3])
        return {"reason": ", ".join(bits), "choice": pick["id"], "confidence": 0.8}

    def _interpret(self, req: Request) -> dict:
        sens = req.payload.get("sensors", {})
        anom = req.payload.get("anomaly", {})
        typ = str(anom.get("type", "")).lower()
        lr = sens.get("lidar_range")
        liq = anom.get("camera_liquid_pct")
        spread = anom.get("edge_spread")
        if "no_return" in typ:
            seen_liquid = (isinstance(liq, (int, float)) and liq >= 3.0) or (
                liq is None and isinstance(spread, (int, float)) and spread < 0.12)
            if seen_liquid:
                cls, act, conf, keys = "liquid", "reroute", 0.85, ("camera_liquid_pct", "distance", "rays_missing")
            else:
                cls, act, conf, keys = "crater_edge", "reroute", 0.6, ("distance", "rays_missing")
        elif "slip" in typ or "sand" in typ:
            cls, act, conf, keys = "soft_ground", "back_out", 0.8, ("slip", "ground_speed")
        elif "obstacle" in typ or "stall" in typ or "rock" in typ:
            cls, act, conf, keys = "rock", "reroute", 0.8, ("distance", "motor_load", "ground_speed")
        elif "vis" in typ or "dust" in typ:
            slow = isinstance(lr, (int, float)) and lr >= 3.0
            cls, act, conf, keys = "dust", "proceed_slowly" if slow else "wait", 0.75, ("lidar_range", "lidar_hits")
        elif "tilt" in typ or "slope" in typ:
            cls, act, conf, keys = "steep_slope", "back_out", 0.7, ("pitch", "roll")
        else:
            cls, act, conf, keys = "unknown", "wait", 0.35, ("pitch", "slip", "motor_load")
        if act not in req.options:
            act = next((o for o in ("wait", "proceed_slowly", "reroute", "back_out") if o in req.options),
                       req.options[0])
        bits = []
        for k in keys:
            v = anom.get(k, sens.get(k))
            if isinstance(v, (int, float)) and not isinstance(v, bool):
                bits.append(_fmt_reading(k, v))
        if not bits:                          # quote any reading at all
            for k, v in list(anom.items()) + list(sens.items()):
                if isinstance(v, (int, float)) and not isinstance(v, bool):
                    bits.append(_fmt_reading(k, v))
                    break
        label = cls.replace("_", " ")
        return {"classification": cls, "reason": f"Looks like {label}: " + ", ".join(bits[:2]),
                "action": act, "confidence": conf}

    def _describe(self, req: Request) -> dict:
        """The fake cannot see legs: it goes by the track numbers (moving on its own, animal sized)."""
        ob = req.payload.get("observation", {})
        sp, size = ob.get("speed"), ob.get("size")
        num = lambda v: isinstance(v, (int, float)) and not isinstance(v, bool)    # noqa: E731
        bits = [_fmt_reading(k, ob[k]) for k in ("speed", "size", "distance") if num(ob.get(k))]
        if num(sp) and num(size) and sp >= 0.08 and 0.4 <= size <= 2.5:
            return {"classification": "lifeform", "label": f"possible lifeform, about {size:.1f} m, moving on its own",
                    "reason": "Moves by itself and is animal sized: " + ", ".join(bits[:2]),
                    "action": "approach", "confidence": 0.8}
        return {"classification": "unknown", "label": "moving object, unclear",
                "reason": "Cannot tell from " + (", ".join(bits[:2]) or "the readings"),
                "action": "observe", "confidence": 0.4}

    def _command(self, req: Request) -> dict:
        p = parse_command(req.text or "", req.payload.get("fleet"))
        c = p["command"]
        return {"command": c["type"], "rover": c.get("rover") or "none", "place": c.get("place") or "none",
                "qualifier": c.get("qualifier") or "none", "ordinal": c.get("ordinal") or 0,
                "near_rover": c.get("near_rover") or "none", "investigate": bool(c.get("investigate")),
                "explore": bool(c.get("explore")), "direction": c.get("direction", "none"),
                "x": c.get("x", 0), "y": c.get("y", 0), "radius": c.get("radius", 0),
                "target_id": c.get("target_id") or "none", "topic": c.get("topic", "none"),
                "question": c.get("question", ""), "ack": p["ack"], "confidence": 0.85}

    _LEADS = {"status": "Status: ", "history": "My log: ", "exploration": "Mapping so far: ", "finds": "Finds: ", "samples": "Samples: ",
              "battery": "Power check: ", "hazards": "Hazards: ", "lifeforms": "Lifeforms: ", "water": "Water: ",
              "location": "Position: ", "why": "Plan: "}

    def _answer(self, req: Request) -> dict:
        """The fake cannot reason: it rewords the two most relevant fact lines (they come first) into
        sentences, which keeps every number and id grounded. Unknown topics: it says it does not know."""
        p = req.payload
        facts = [f for f in p.get("facts", []) if f]
        if p.get("topic") == "unknown" or not facts:
            return {"text": "I don't know that. I can only tell you what my own sensors and memory recorded."}
        out = []
        for f in facts[:2]:
            f = f.strip().rstrip(".")
            if f.lower().startswith("why: "):
                f = "that is because " + f[5:]
            out.append(f if _ID_RE.match(f) else f[:1].upper() + f[1:])
            out[-1] += "."
        text = self._LEADS.get(p.get("topic"), "") + " ".join(out)
        words = text.split()
        if len(words) > MAX_ANSWER_WORDS or len(text) > MAX_ANSWER_CHARS:
            text = self._LEADS.get(p.get("topic"), "") + out[0]
            if len(text.split()) > MAX_ANSWER_WORDS or len(text) > MAX_ANSWER_CHARS:
                cut = " ".join(text.split()[:MAX_ANSWER_WORDS - 2])[:MAX_ANSWER_CHARS - 4].rsplit(" ", 1)[0]
                text = cut.rstrip(",;:") + "..."
        return {"text": text}


# ====================================================================== one call
def run_request(backend: Backend, req: Request, cfg: GemmaConfig, meta_out: dict | None = None) -> tuple:
    """Ask the backend once, synchronously, and validate. Returns (Result, problem) where problem
    is None, "timeout", "offline", "error" or "invalid". Used by the planner's worker and the bench.
    meta_out (optional dict) receives the server's timing fields and the raw answer."""
    system, user = build_prompt(req)
    schema = build_schema(req)
    model = getattr(backend, "model", "")
    limit = min(cfg.timeout_s, req.timeout_s) if req.timeout_s else cfg.timeout_s
    t0 = time.perf_counter()
    try:
        content, meta = backend.chat(req, system, user, schema, limit)
        if meta_out is not None:
            meta_out.update(meta or {})
            meta_out["content"] = content
    except GemmaTimeout:
        ms = (time.perf_counter() - t0) * 1000.0
        return fallback(req, f"timeout: no answer in {ms:.0f} ms", ms, model), "timeout"
    except GemmaOffline as e:
        ms = (time.perf_counter() - t0) * 1000.0
        return fallback(req, f"offline: {e}"[:200], ms, model), "offline"
    except GemmaError as e:
        ms = (time.perf_counter() - t0) * 1000.0
        return fallback(req, f"error: {e}"[:200], ms, model), "error"
    except Exception as e:                    # noqa: BLE001 (the worker must never die)
        ms = (time.perf_counter() - t0) * 1000.0
        return fallback(req, f"error: {type(e).__name__}: {e}"[:200], ms, model), "error"
    ms = (time.perf_counter() - t0) * 1000.0
    if ms > limit * 1000.0:
        return fallback(req, f"late: answered in {ms:.0f} ms, over the {limit:g} s timeout", ms,
                        model), "timeout"
    res = validate(req, content, cfg.min_confidence, model)
    res.latency_ms = round(ms, 1)
    meta = meta or {}
    res.tokens_in = meta.get("prompt_eval_count")
    res.tokens_out = meta.get("eval_count")
    return res, (None if res.ok else "invalid")


def latency_summary(values, budget_ms: float) -> dict:
    """avg, p50, p95 (nearest rank), max and count over budget, in ms."""
    v = sorted(float(x) for x in values)
    if not v:
        return {"n": 0, "avg_ms": 0.0, "p50_ms": 0.0, "p95_ms": 0.0, "max_ms": 0.0, "over_budget": 0}

    def pct(p):
        return v[max(0, min(len(v) - 1, int(math.ceil(p * len(v))) - 1))]

    return {"n": len(v), "avg_ms": round(sum(v) / len(v), 1), "p50_ms": round(pct(0.50), 1),
            "p95_ms": round(pct(0.95), 1), "max_ms": round(v[-1], 1),
            "over_budget": sum(1 for x in v if x > budget_ms)}


# ====================================================================== planner
class GemmaPlanner:
    """Non-blocking front end for the 5 Hz rover loop. One worker thread asks the model, one
    monitor thread checks every ~15 s that Ollama is up and a usable model is installed (and
    warms it up once, so the first real decision does not pay the model load).

    Queue: at most one queued request per (rover, kind); a newer submit for the same rover and
    kind replaces the queued one (the old ticket resolves to a fallback, error "superseded").
    Operator commands never replace each other. Order: commands, then interpretations, then
    choices. One request is in flight at a time, so never more than one per rover. A request
    that waited longer than GEMMA_TIMEOUT_S in the queue is dropped (fallback). When the model
    is not ready (disabled, offline, model missing, warming), submits resolve to a fallback at once."""

    _PRIORITY = ("command", "answer", "interpretation", "describe", "choice")

    def __init__(self, config: GemmaConfig | None = None, backend: Backend | None = None, start: bool = True):
        self.cfg = config or GemmaConfig.from_env()
        if backend is None:
            backend = (FakeBackend(self.cfg.fake_latency_ms / 1000.0) if self.cfg.fake
                       else OllamaBackend.from_config(self.cfg))
        self.backend = backend
        self._lock = threading.Lock()
        self._cv = threading.Condition(self._lock)
        self._check_lock = threading.Lock()
        self._stop = threading.Event()
        self._pending: OrderedDict = OrderedDict()     # key -> Request, in arrival order
        self._inflight: Request | None = None
        self._results: OrderedDict = OrderedDict()     # ticket -> Result
        self._recent: deque = deque(maxlen=20)
        self._tickets = itertools.count(1)
        self._state = "starting" if self.cfg.enabled else "disabled"
        self._detail = ""
        self._warm_model: str | None = None            # model that finished warming up
        self._lat: deque = deque(maxlen=MAX_LATENCIES)
        self._n_calls = self._n_ok = self._n_fallback = self._n_over = self._n_ask = 0
        self._last_ms = 0.0
        self._warmup_ms: float | None = None
        self._last_error: str | None = None
        self._threads: list = []
        if self.cfg.enabled and start:
            self.start()

    # ---------------------------------------------------------------- lifecycle
    def start(self) -> None:
        if self._threads or not self.cfg.enabled:
            return
        for name, fn in (("gemma-worker", self._worker), ("gemma-monitor", self._monitor)):
            th = threading.Thread(target=fn, name=name, daemon=True)
            th.start()
            self._threads.append(th)

    def close(self, timeout: float = 1.0) -> None:
        self._stop.set()
        with self._cv:
            self._cv.notify_all()
        for th in self._threads:
            th.join(timeout)

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()

    @property
    def state(self) -> str:
        """disabled | starting | offline | no_model | warming | ready"""
        return self._state

    @property
    def online(self) -> bool:
        return self._state == "ready"

    def wait_ready(self, timeout: float = 5.0) -> bool:
        """Block until the model is ready (scripts and tests only; never from the 5 Hz loop)."""
        end = time.perf_counter() + timeout
        with self._cv:
            while self._state != "ready":
                left = end - time.perf_counter()
                if left <= 0 or self._state == "disabled":
                    return False
                self._cv.wait(min(left, 0.05))
            return True

    def _set_state(self, state: str, detail: str = "") -> None:
        with self._cv:
            self._state = state
            self._detail = detail
            if state not in ("ready", "warming") and detail:
                self._last_error = detail
            self._cv.notify_all()

    def check_now(self) -> str:
        """Run one online check now (and warm the model up if it just became reachable)."""
        if not self.cfg.enabled:
            return "disabled"
        with self._check_lock:
            try:
                state, detail = self.backend.check(self.cfg.check_timeout_s)
            except Exception as e:            # noqa: BLE001
                state, detail = "offline", f"check failed: {type(e).__name__}: {e}"
            if state != "ready":
                self._warm_model = None
                self._set_state(state, detail)
                return state
            if self._warm_model != self.backend.model:
                self._set_state("warming", f"loading {self.backend.model}")
                t0 = time.perf_counter()
                err = self.backend.warmup(self.cfg.warmup_timeout_s)
                if err:
                    self._set_state("offline", err)
                    return "offline"
                self._warmup_ms = round((time.perf_counter() - t0) * 1000.0, 1)
                self._warm_model = self.backend.model
            self._set_state("ready", detail)
            return "ready"

    def _monitor(self) -> None:
        while not self._stop.is_set():
            self.check_now()
            self._stop.wait(self.cfg.check_every_s)

    # ---------------------------------------------------------------- submit / poll
    def submit_choice(self, rover_id: str, t: float, perception: dict, candidates: list,
                      default: str | None = None) -> int:
        """Ask Gemma to pick one candidate. candidates: [{"id", "kind", "summary"}], kind in
        CANDIDATE_KINDS. default: the rules' own pick (default: the first candidate), returned
        as Result.choice in a fallback. Returns a ticket for poll(). Never blocks, never raises."""
        ticket = next(self._tickets)
        try:
            req = make_choice_request(ticket, rover_id, t, perception, candidates, default)
        except Exception as e:                # noqa: BLE001
            ids = [c.get("id") for c in candidates or [] if isinstance(c, dict)]
            stub = Request(ticket, "choice", str(rover_id), float(t or 0.0), {}, [],
                           default=default if default is not None else (ids[0] if ids else None))
            return self._resolve_now(stub, f"bad input: {e}")
        return self._enqueue(req)

    def submit_interpretation(self, rover_id: str, t: float, perception: dict, anomaly: dict,
                              image_jpeg: bytes | None, options: list, default: str | None = None,
                              classes: list | None = None) -> int:
        """Ask Gemma what an anomaly is. anomaly: {"type": "wheel_slip" | "new_obstacle" |
        "low_visibility" | ..., plus any numbers}. image_jpeg: a small fpv JPEG (320 x 240 is
        plenty) or None. options: actions from ACTIONS (ask_operator is always added). default:
        the action used in a fallback (default proceed_slowly, else wait, else the first option).
        Returns a ticket for poll(). Never blocks, never raises."""
        ticket = next(self._tickets)
        try:
            req = make_interpretation_request(ticket, rover_id, t, perception, anomaly, image_jpeg, options,
                                              default, classes)
        except Exception as e:                # noqa: BLE001
            opts = [o for o in (options or []) if o in ACTIONS]
            dflt = default if default in ACTIONS else next(
                (o for o in ("proceed_slowly", "wait") if o in opts), opts[0] if opts else "wait")
            stub = Request(ticket, "interpretation", str(rover_id), float(t or 0.0), {}, [], default=dflt)
            return self._resolve_now(stub, f"bad input: {e}")
        return self._enqueue(req)

    def submit_description(self, rover_id: str, t: float, perception: dict, observation: dict,
                           image_jpeg: bytes | None, default: str = "observe") -> int:
        """Ask Gemma what a moving LiDAR track is (lifeform or not, short label, action). observation:
        {"distance": m, "speed": m/s, "size": m, ...}. Returns a ticket. Never blocks, never raises."""
        ticket = next(self._tickets)
        try:
            req = make_description_request(ticket, rover_id, t, perception, observation, image_jpeg, default)
        except Exception as e:                # noqa: BLE001
            stub = Request(ticket, "describe", str(rover_id), float(t or 0.0), {}, [], default="observe")
            return self._resolve_now(stub, f"bad input: {e}")
        return self._enqueue(req)

    def submit_command(self, t: float, text: str, fleet_context: dict) -> int:
        """Turn an operator message into a command. fleet_context: {"rovers": [{"id", "name",
        "color", "x", "y", "state", "battery"}], "targets": [{"id", "kind", "x", "y", "status"}],
        "lander": [x, y]}. The result's command is Gemma's (ok=True) or the regex parser's
        (ok=False, source="fallback"); either way use Result.command and show Result.ack, or
        Result.question when the command is "unclear". Never blocks, never raises."""
        ticket = next(self._tickets)
        try:
            req = make_command_request(ticket, t, text, fleet_context)
        except Exception as e:                # noqa: BLE001
            stub = Request(ticket, "command", "operator", float(t or 0.0), {}, [], text=str(text)[:MAX_TEXT],
                           parsed=_unclear(_HELP))
            return self._resolve_now(stub, f"bad input: {type(e).__name__}: {e}")
        if not req.text:
            return self._resolve_now(req, "empty message")
        return self._enqueue(req)

    def submit_answer(self, speaker: str, t: float, question: str, topic: str, facts: list,
                      default: str, timeout_s: float | None = None, context: list | None = None) -> int:
        """Ask Gemma to answer an operator question from fact lines (answers.fact_lines). speaker: a
        rover id or "fleet". default: the rule-based answer (Result.answer in a fallback). timeout_s:
        latency limit for this call (default the budget, GEMMA_BUDGET_MS). context: chat memory lines (what
        "there" means; protocol v7). Never blocks, never raises."""
        ticket = next(self._tickets)
        lim = timeout_s if timeout_s is not None else self.cfg.budget_ms / 1000.0
        try:
            req = make_answer_request(ticket, speaker, t, question, topic, facts, default, lim, context)
        except Exception as e:                # noqa: BLE001
            stub = Request(ticket, "answer", str(speaker), float(t or 0.0), {}, [], default=str(default or ""),
                           text=str(question)[:MAX_TEXT])
            return self._resolve_now(stub, f"bad input: {e}")
        return self._enqueue(req)

    def poll(self, ticket: int) -> Result | None:
        """None while the request is queued or in flight; then its Result (kept for the last 512
        tickets, so polling twice is fine). Never blocks."""
        with self._lock:
            return self._results.get(ticket)

    def cancel(self, rover_id: str | None = None) -> int:
        """Drop queued requests (all, or one rover's), e.g. on a sim reset. Returns how many."""
        n = 0
        with self._cv:
            for key in [k for k in self._pending if rover_id is None or k[0] == rover_id]:
                self._deliver_locked(fallback(self._pending.pop(key), "cancelled"))
                n += 1
        return n

    def _resolve_now(self, req: Request, error: str) -> int:
        with self._lock:
            self._deliver_locked(fallback(req, error, model=self.backend.model))
        return req.ticket

    def _enqueue(self, req: Request) -> int:
        with self._cv:
            if self._state != "ready":
                why = f"gemma {self._state}" + (f": {self._detail}" if self._detail else "")
                self._deliver_locked(fallback(req, why[:200], model=self.backend.model))
                return req.ticket
            old = self._pending.get(req.key)
            if old is not None:
                self._deliver_locked(fallback(old, f"superseded by ticket {req.ticket}"))
                self._pending[req.key] = req           # same key: keeps its place in the queue
            elif len(self._pending) >= self.cfg.max_queue:
                self._deliver_locked(fallback(req, "queue full", model=self.backend.model))
                return req.ticket
            else:
                self._pending[req.key] = req
            self._cv.notify()
        return req.ticket

    def _deliver_locked(self, res: Result) -> None:
        self._results[res.ticket] = res
        while len(self._results) > MAX_RESULTS:
            self._results.popitem(last=False)
        self._recent.append(res)
        if res.ok:
            self._n_ok += 1
            if res.action == "ask_operator":
                self._n_ask += 1
        else:
            self._n_fallback += 1
            self._last_error = res.error

    # ---------------------------------------------------------------- worker
    def _next_locked(self) -> Request | None:
        for kind in self._PRIORITY:
            key = next((k for k, r in self._pending.items() if r.kind == kind), None)
            if key is not None:
                return self._pending.pop(key)
        return None

    def _worker(self) -> None:
        while not self._stop.is_set():
            with self._cv:
                req = self._next_locked()
                while req is None and not self._stop.is_set():
                    self._cv.wait(0.5)
                    req = self._next_locked()
                if req is None:
                    return
                self._inflight = req
                state = self._state
            try:
                res = self._process(req, state)
            except Exception as e:            # noqa: BLE001 (never lose a ticket)
                res = fallback(req, f"error: {type(e).__name__}: {e}"[:200])
            with self._cv:
                self._inflight = None
                self._deliver_locked(res)

    def _process(self, req: Request, state: str) -> Result:
        waited = time.perf_counter() - req.submitted
        if waited > self.cfg.queue_wait_s:
            return fallback(req, f"stale: waited {waited:.1f} s in the queue", model=self.backend.model)
        if state != "ready":
            return fallback(req, f"gemma {state}", model=self.backend.model)
        res, problem = run_request(self.backend, req, self.cfg)
        with self._lock:
            self._n_calls += 1
            self._lat.append(res.latency_ms)
            self._last_ms = res.latency_ms
            if res.latency_ms > self.cfg.budget_ms:
                self._n_over += 1
        if problem == "offline":
            self._warm_model = None
            self._set_state("offline", res.error or "offline")
        elif problem == "timeout":
            self._warm_model = None           # maybe the model was unloaded: warm up again on the next check
        return res

    # ---------------------------------------------------------------- UI
    def stats(self) -> dict:
        """For the UI. Latencies are wall-clock ms per model call over the last 200 calls.
        model is the model actually in use (it may be a fallback such as gemma3:4b)."""
        with self._lock:
            s = latency_summary(self._lat, self.cfg.budget_ms)
            return {
                "model": self.backend.model, "requested_model": self.backend.requested_model,
                "backend": self.backend.name, "url": self.backend.url,
                "online": self._state == "ready", "state": self._state, "warming": self._state == "warming",
                "enabled": self.cfg.enabled, "detail": self._detail,
                "calls": self._n_calls, "ok": self._n_ok, "fallbacks": self._n_fallback,
                "asked_operator": self._n_ask,
                "avg_ms": s["avg_ms"], "p95_ms": s["p95_ms"], "last_ms": round(self._last_ms, 1),
                "budget_ms": self.cfg.budget_ms, "timeout_ms": round(self.cfg.timeout_s * 1000.0, 1),
                "over_budget": self._n_over, "warmup_ms": self._warmup_ms,
                "queued": len(self._pending), "in_flight": self._inflight is not None,
                "last_error": self._last_error,
            }

    def recent(self, n: int = 10) -> list:
        """The last n results (newest last) as dicts, for a decision log panel."""
        with self._lock:
            return [r.to_dict() for r in list(self._recent)[-n:]]


# ====================================================================== shared instance
_SHARED: GemmaPlanner | None = None
_SHARED_LOCK = threading.Lock()


def planner_from_env(**overrides) -> GemmaPlanner:
    """A new planner configured from GEMMA_* env vars (keyword overrides win)."""
    return GemmaPlanner(GemmaConfig.from_env(**overrides))


def get_planner() -> GemmaPlanner:
    """The process-wide planner (created on first use). It holds no sim state, so it can live
    across sim resets; call cancel() on reset to drop queued questions about the old world."""
    global _SHARED
    with _SHARED_LOCK:
        if _SHARED is None:
            _SHARED = planner_from_env()
        return _SHARED


def close_planner(timeout: float = 1.0) -> None:
    """Stop the process-wide planner's threads and forget it (server shutdown)."""
    global _SHARED
    with _SHARED_LOCK:
        p, _SHARED = _SHARED, None
    if p is not None:
        p.close(timeout)
