"""Gemma 4 mission commander: prompt, tool schema, Ollama call and parsing. Standard library only.
Mirror of web/brain/agent.ts. The LLM only ever sees the brain's text summary (sensed map, predictions,
rover status) -- never the simulator's world state -- and may only answer with one of five tool calls.
"""
import json
import time
import urllib.request
import urllib.error

SYSTEM_PROMPT = (
    "You are the mission commander of an autonomous disaster-response rover. The rover's own safety layer "
    "handles immediate danger; you set the strategic goal. Mission: survey the area and place early-warning "
    "probes where they warn people about hazards, while keeping the rover alive. Coordinates are metres, map "
    "is -97..97 on both axes. Use only positions mentioned in the situation report. Call exactly one tool."
)


def _fn(name, desc, props, required):
    return {"type": "function", "function": {"name": name, "description": desc,
                                             "parameters": {"type": "object", "properties": props, "required": required}}}


_R = {"type": "string", "description": "one short sentence: why"}
_XY = {"x": {"type": "number"}, "y": {"type": "number"}}
TOOLS = [
    _fn("go_to_region", "Drive to a point (e.g. towards unexplored ground).", dict(_XY, reason=_R), ["x", "y", "reason"]),
    _fn("place_probe_at", "Drive to a point and drop an early-warning probe there.", dict(_XY, reason=_R), ["x", "y", "reason"]),
    _fn("retreat_to_high_ground", "Climb to the highest safe ground (floods).", {"reason": _R}, ["reason"]),
    _fn("hold_position", "Stay still for some seconds (e.g. while the ground shakes).",
        {"seconds": {"type": "number"}, "reason": _R}, ["seconds", "reason"]),
    _fn("survey_area", "Explore within a radius of a point.", dict(_XY, radius={"type": "number"}, reason=_R), ["x", "y", "radius", "reason"]),
]

#: event kinds that trigger an immediate call (otherwise at most every PERIOD seconds)
TRIGGER_EVENTS = ("new_hazard", "hazard_on_path", "battery", "probe_placed", "probe_warning")
PERIOD = 30.0
TIMEOUT = 60.0


def build_request(model, summary):
    return {
        "model": model, "stream": False, "think": False, "keep_alive": "30m",
        "options": {"num_predict": 200, "temperature": 0.2},
        "tools": TOOLS,
        "messages": [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": "Situation report:\n" + summary}],
    }


def parse_response(d):
    """-> {'name', 'args'} or raises ValueError. Accepts native tool_calls, or a JSON object in the text."""
    msg = d.get("message") or {}
    calls = msg.get("tool_calls") or []
    if calls:
        fn = calls[0].get("function") or {}
        args = fn.get("arguments")
        if isinstance(args, str):
            args = json.loads(args)
        if not isinstance(fn.get("name"), str) or not isinstance(args, dict):
            raise ValueError("malformed tool call")
        return {"name": fn["name"], "args": args}
    text = (msg.get("content") or "").strip()
    i, j = text.find("{"), text.rfind("}")
    if i >= 0 and j > i:
        o = json.loads(text[i:j + 1])
        name = o.get("name") or o.get("tool")
        args = o.get("arguments") or o.get("args") or o.get("parameters") or {}
        if isinstance(name, str) and isinstance(args, dict):
            return {"name": name, "args": args}
    raise ValueError("no tool call in output: %r" % text[:80])


def call_ollama(url, model, summary, timeout=TIMEOUT):
    """Blocking. Returns (tool_call or None, error or None, latency_s, raw)."""
    t0 = time.time()
    body = json.dumps(build_request(model, summary)).encode()
    try:
        req = urllib.request.Request(url.rstrip("/") + "/api/chat", data=body, headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=timeout) as r:
            d = json.loads(r.read())
    except (urllib.error.URLError, OSError, ValueError) as e:
        return None, "ollama error: %s" % (getattr(e, "reason", None) or e), time.time() - t0, None
    lat = time.time() - t0
    try:
        return parse_response(d), None, lat, d
    except (ValueError, TypeError) as e:
        return None, "unparsable output (%s)" % e, lat, d


class Trigger:
    """When to ask the LLM: on major events, otherwise at most every PERIOD s; one request in flight."""

    def __init__(self):
        self.last_call_t = -1e9
        self.pending = []

    def should_call(self, t, events, in_flight):
        self.pending.extend(e for e in events if e.get("kind") in TRIGGER_EVENTS)
        if in_flight:
            return False
        if self.pending or t - self.last_call_t >= PERIOD:
            self.last_call_t = t
            why = ", ".join(sorted(set(e["kind"] for e in self.pending))) or "periodic"
            self.pending = []
            return why
        return False
