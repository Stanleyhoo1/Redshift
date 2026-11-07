"""Tests for the Gemma advisory layer (redshift/gemma.py) against a stub Ollama server.

Run with `python -m pytest -q tests/test_gemma.py` or `python tests/test_gemma.py`.
The stub (http.server in a thread, port 8400) mimics Ollama's GET /api/tags, GET /api/version,
POST /api/show and POST /api/chat (with `format` JSON schemas). No model is needed.
"""
from __future__ import annotations

import base64
import contextlib
import gc
import io
import importlib.util
import json
import os
import re
import statistics
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from redshift import gemma as g  # noqa: E402
from redshift.gemma import FakeBackend, GemmaConfig, GemmaPlanner  # noqa: E402

PORT = 8400
STUB_URL = f"http://127.0.0.1:{PORT}"
DEAD_URL = "http://127.0.0.1:9"        # nothing listens here: connection refused at once
EM_DASH = chr(0x2014)

PERCEPTION = {"battery": 0.64, "state": "exploring", "x": 3.21, "y": -1.04, "lidar_range": 8.0, "lidar_hits": 812,
              "pitch": 3.1, "roll": -1.2, "wheel_speed": 0.62, "ground_speed": 0.58, "slip": 0.06,
              "motor_load": 0.41, "spectro": None, "slope_limit_deg": 20.0, "home_distance": 31.2,
              "scan": [0.1] * 900}
CANDIDATES = [
    {"id": "f1", "kind": "frontier", "summary": "frontier 9 m north, 34 cells, slope max 12 deg, battery after return 58%"},
    {"id": "f2", "kind": "frontier", "summary": "frontier 21 m east, 80 cells, slope max 18 deg, battery after return 41%"},
    {"id": "f3", "kind": "frontier", "summary": "frontier 6 m west, 8 cells, slope max 6 deg, battery after return 66%"},
    {"id": "home", "kind": "return", "summary": "lander 31 m south-west, battery on arrival 60%"},
]
SLIP = {"type": "wheel_slip", "slip": 0.46, "duration_s": 2.4, "detail": "wheels spinning, not moving"}
OPTIONS = ["reroute", "back_out", "proceed_slowly", "wait"]
FLEET = {"rovers": [{"id": "r1", "name": "Rover 1", "color": "#ff7a45", "x": 3.2, "y": -1.0, "state": "exploring",
                     "battery": 0.64},
                    {"id": "r2", "name": "Rover 2", "color": "#36cfc9", "x": -8.5, "y": 12.1, "state": "returning",
                     "battery": 0.31},
                    {"id": "r3", "name": "Rover 3", "color": "#9254de", "x": 15.0, "y": 4.4, "state": "investigating",
                     "battery": 0.82}],
         "targets": [{"id": "t4", "kind": "ice", "x": 12.0, "y": -4.5, "status": "found"}],
         "lander": [0.0, 0.0], "map_size": 80}


# ====================================================================== stub Ollama
class Stub:
    """Shared, mutable behaviour of the stub server. reset() before each test."""

    def __init__(self):
        self.lock = threading.Lock()
        self.reset()

    def reset(self):
        with getattr(self, "lock", threading.Lock()):
            self.models = ["gemma4:e2b", "llama3:latest"]
            self.tags_status = 200
            self.delay = 0.0              # seconds before answering a real chat
            self.warmup_delay = 0.0       # seconds before answering the warm-up
            self.reply = None             # callable(body) -> content string; None = schema-shaped default
            self.reject_think = False     # answer 400 "does not support thinking" when "think" is sent
            self.chats = []               # bodies of real chat requests (not warm-ups)
            self.warmups = []             # bodies of warm-up requests

    @staticmethod
    def default_reply(body) -> str:
        """A schema-shaped answer whose reason quotes the first number of the user message."""
        props = (body.get("format") or {}).get("properties", {})
        user = body["messages"][-1]["content"]
        nums = g.numbers_in_text(user)
        out = {}
        for k, spec in props.items():
            if "enum" in spec:
                out[k] = spec["enum"][0]
            elif spec.get("type") == "number":
                out[k] = 0.8 if k == "confidence" else 0
            else:
                out[k] = ""
        if "reason" in props:
            out["reason"] = f"reading {nums[0][0] if nums else 'none'} looks fine"
        return json.dumps(out)


STUB = Stub()


class Handler(BaseHTTPRequestHandler):
    def log_message(self, *a):          # quiet
        pass

    def _send(self, code: int, obj) -> None:
        data = json.dumps(obj).encode() if not isinstance(obj, bytes) else obj
        try:
            self.send_response(code)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)
        except (BrokenPipeError, ConnectionResetError):   # the client gave up (timeout test)
            pass

    def do_GET(self):
        if self.path == "/api/tags":
            if STUB.tags_status != 200:
                return self._send(STUB.tags_status, {"error": "stub tags failure"})
            return self._send(200, {"models": [{"name": m, "model": m} for m in STUB.models]})
        if self.path == "/api/version":
            return self._send(200, {"version": "0.0.0-stub"})
        self._send(404, {"error": "not found"})

    def do_POST(self):
        n = int(self.headers.get("Content-Length") or 0)
        body = json.loads(self.rfile.read(n) or b"{}")
        if self.path == "/api/show":
            return self._send(200, {"details": {"parameter_size": "stub", "quantization_level": "none"},
                                    "capabilities": ["completion", "vision"]})
        if self.path != "/api/chat":
            return self._send(404, {"error": "not found"})
        if STUB.reject_think and "think" in body:
            return self._send(400, {"error": f"\"{body.get('model')}\" does not support thinking"})
        if body.get("model") not in STUB.models:
            return self._send(404, {"error": f"model \"{body.get('model')}\" not found, try pulling it first"})
        warm = "format" not in body
        with STUB.lock:
            (STUB.warmups if warm else STUB.chats).append(body)
        time.sleep(STUB.warmup_delay if warm else STUB.delay)
        if warm:
            return self._send(200, {"model": body["model"], "message": {"role": "assistant", "content": "OK"},
                                    "done": True})
        content = (STUB.reply or Stub.default_reply)(body)
        if isinstance(content, bytes):
            return self._send(200, content)
        self._send(200, {"model": body["model"], "message": {"role": "assistant", "content": content},
                         "done": True, "total_duration": 9_000_000, "prompt_eval_count": 250, "eval_count": 30,
                         "eval_duration": 600_000_000})


_SERVER = None


def start_stub():
    global _SERVER
    if _SERVER is None:
        ThreadingHTTPServer.daemon_threads = True
        _SERVER = ThreadingHTTPServer(("127.0.0.1", PORT), Handler)
        threading.Thread(target=_SERVER.serve_forever, name="stub-ollama", daemon=True).start()
    return _SERVER


def stop_stub():
    global _SERVER
    if _SERVER is not None:
        _SERVER.shutdown()
        _SERVER.server_close()
        _SERVER = None


# ---- pytest fixtures (the plain runner at the bottom does the same)
try:
    import pytest

    @pytest.fixture(scope="module", autouse=True)
    def _stub_server():
        start_stub()
        yield
        stop_stub()

    @pytest.fixture(autouse=True)
    def _fresh_stub():
        STUB.reset()
        yield
except ImportError:                           # plain runner only
    pytest = None


# ====================================================================== helpers
def planner(url=STUB_URL, ready=True, **kw) -> GemmaPlanner:
    base = dict(url=url, fallback_urls=(), model="gemma4:e2b", timeout_s=2.0, check_every_s=3600.0,
                check_timeout_s=0.5, warmup_timeout_s=5.0)
    base.update(kw)
    p = GemmaPlanner(GemmaConfig(**base))
    if ready:
        assert p.wait_ready(5.0), f"planner not ready: {p.state} {p.stats()['detail']}"
    return p


def wait_result(p: GemmaPlanner, ticket: int, timeout: float = 5.0):
    end = time.perf_counter() + timeout
    while time.perf_counter() < end:
        r = p.poll(ticket)
        if r is not None:
            return r
        time.sleep(0.002)
    raise AssertionError(f"no result for ticket {ticket} in {timeout} s")


def answer(**fields):
    return lambda body: json.dumps(fields)


# ====================================================================== choices
def test_choice_ok_and_request_shape():
    STUB.reply = answer(reason="f2 has 80 cells and battery after return 41% is enough.", choice="f2",
                        confidence=0.82)
    with planner() as p:
        tk = p.submit_choice("r1", 12.5, PERCEPTION, CANDIDATES, default="f1")
        r = wait_result(p, tk)
    assert r.ok and r.source == "gemma" and r.kind == "choice", r
    assert r.choice == "f2" and r.confidence == 0.82 and r.rover_id == "r1" and r.t == 12.5
    assert "80" in r.cited and "41%" in r.cited
    assert r.tokens_in == 250 and r.tokens_out == 30 and r.latency_ms > 0
    assert "f2" in r.trace() and r["choice"] == "f2" and r.get("reason").startswith("f2")
    body = STUB.chats[-1]
    assert body["model"] == "gemma4:e2b" and body["stream"] is False and body["keep_alive"] == "30m"
    assert body["think"] is False
    assert body["options"]["temperature"] == 0 and body["options"]["num_predict"] == 96
    schema = body["format"]
    assert schema["properties"]["choice"]["enum"] == ["f1", "f2", "f3", "home"]
    assert set(schema["required"]) == {"reason", "choice", "confidence"}
    sys_msg, user_msg = body["messages"]
    assert sys_msg["role"] == "system" and "2 g" in sys_msg["content"] and "JSON only" in sys_msg["content"]
    assert "battery 64%" in user_msg["content"] and "pitch 3.1 deg" in user_msg["content"]
    assert "f2 (frontier): frontier 21 m east" in user_msg["content"]
    assert "scan" not in user_msg["content"] and "images" not in user_msg


def test_invalid_choice_falls_back_to_rule_pick():
    STUB.reply = answer(reason="f9 has battery 64%", choice="f9", confidence=0.9)
    with planner() as p:
        r = wait_result(p, p.submit_choice("r1", 1.0, PERCEPTION, CANDIDATES, default="f3"))
    assert not r.ok and r.source == "fallback" and r.choice == "f3"
    assert "not in candidates" in r.error


def test_bad_json_and_missing_fields_fall_back():
    cases = [
        (lambda b: "not json {", "bad JSON"),
        (lambda b: "", "empty"),
        (answer(reason="battery 64% is fine", choice="f1"), "missing confidence"),
        (answer(choice="f1", confidence=0.7), "missing reason"),
        (answer(reason="battery 64% is fine", confidence=0.7), "missing choice"),
        (answer(reason="battery 64% is fine", choice="f1", confidence=7000), "not in 0..1"),
        (lambda b: b'{"oops": ', "error"),                       # the server itself sends broken JSON
    ]
    with planner() as p:
        for reply, expect in cases:
            STUB.reply = reply
            r = wait_result(p, p.submit_choice("r1", 1.0, PERCEPTION, CANDIDATES, default="f1"))
            assert not r.ok and r.choice == "f1", (expect, r)
            assert expect in r.error, (expect, r.error)


def test_reason_must_quote_an_input_number():
    with planner() as p:
        STUB.reply = answer(reason="Lidar range (sensor 1) is sufficient and the slope is manageable.",
                            choice="f1", confidence=0.8)
        r = wait_result(p, p.submit_choice("r1", 1.0, PERCEPTION, CANDIDATES))
        assert not r.ok and "quotes no input reading" in r.error, r.error
        STUB.reply = answer(reason="f1 is only 99 m away", choice="f1", confidence=0.8)     # 99 is not in the input
        r = wait_result(p, p.submit_choice("r1", 1.0, PERCEPTION, CANDIDATES))
        assert not r.ok and "quotes no input reading" in r.error
        STUB.reply = answer(reason="slope max 12 deg is fine", choice="f1", confidence=0.8)
        r = wait_result(p, p.submit_choice("r1", 1.0, PERCEPTION, CANDIDATES))
        assert r.ok and r.cited == ["12 deg"]


def test_cited_number_matching():
    inputs = [0.41, 7.83, -1.2, 812.0, 20.0, 0.0]
    assert g.cited_numbers("battery 41%", inputs) == ["41%"]           # fraction as percent
    assert g.cited_numbers("range 7.8 m", inputs) == ["7.8 m"]          # rounded to what was written
    assert g.cited_numbers("range 8 m", inputs) == ["8 m"]              # rounded to an integer (>= 2)
    assert g.cited_numbers("roll 1.2 deg left", inputs) == ["1.2 deg"]    # sign dropped
    assert g.cited_numbers("812 hits, limit 20 deg", inputs) == ["812", "20 deg"]
    assert g.cited_numbers("f2 and r1 and t3", inputs) == []          # ids are not readings
    assert g.cited_numbers("battery 55%", inputs) == []
    assert g.cited_numbers("1 rock", [0.6]) == []                     # no loose rounding of small values
    assert g.cited_numbers("0 hits", inputs) == ["0"]


def test_long_reason_cut_and_em_dash_removed():
    long = "battery 64% " + EM_DASH + " " + " ".join(["word"] * 40)
    STUB.reply = answer(reason=long, choice="f1", confidence=0.8)
    with planner() as p:
        r = wait_result(p, p.submit_choice("r1", 1.0, PERCEPTION, CANDIDATES))
    assert r.ok and EM_DASH not in r.reason
    assert len(r.reason.split()) <= g.MAX_REASON_WORDS and r.reason.endswith("...")


def test_bad_input_never_raises():
    with planner() as p:
        r = p.poll(p.submit_choice("r1", 1.0, PERCEPTION, []))
        assert r is not None and not r.ok and "bad input" in r.error
        r = p.poll(p.submit_choice("r1", 1.0, PERCEPTION, [{"id": "f1", "kind": "teleport"}]))
        assert r is not None and r.choice == "f1" and "bad input" in r.error
        r = p.poll(p.submit_interpretation("r1", 1.0, PERCEPTION, SLIP, None, ["fly"]))
        assert r is not None and not r.ok and r.action == "wait"
        r = p.poll(p.submit_command(1.0, None, None))
        assert r is not None and r.command["type"] == "unclear"


# ====================================================================== interpretations
def test_interpretation_ok_with_image_base64():
    jpeg = b"\xff\xd8\xff\xe0fake-jpeg-bytes\xff\xd9"
    STUB.reply = answer(classification="soft_ground", reason="slip 46% while ground speed 0.58 m/s",
                        action="back_out", confidence=0.78)
    with planner() as p:
        r = wait_result(p, p.submit_interpretation("r2", 40.0, PERCEPTION, SLIP, jpeg, OPTIONS))
    assert r.ok and r.classification == "soft_ground" and r.action == "back_out" and r.model_action == "back_out"
    assert r.question is None and "46%" in r.cited
    body = STUB.chats[-1]
    user = body["messages"][-1]
    assert user["images"] == [base64.b64encode(jpeg).decode("ascii")]
    assert "Camera image attached." in user["content"] and "Anomaly: wheel_slip: slip 46%" in user["content"]
    props = body["format"]["properties"]
    assert props["action"]["enum"] == OPTIONS + ["ask_operator"]          # ask_operator always allowed
    assert "soft_ground" in props["classification"]["enum"]


def test_low_confidence_becomes_ask_operator():
    STUB.reply = answer(classification="rock", reason="motor load 41% and pitch 3.1 deg", action="reroute",
                        confidence=0.3)
    with planner() as p:
        r = wait_result(p, p.submit_interpretation("r2", 40.0, PERCEPTION, SLIP, None, OPTIONS))
        assert r.ok and r.source == "gemma"
        assert r.action == "ask_operator" and r.model_action == "reroute" and r.confidence == 0.3
        assert r.question and "unsure" in r.question and "back_out" in r.question
        assert p.stats()["asked_operator"] == 1
        assert "No camera image." in STUB.chats[-1]["messages"][-1]["content"]


def test_invalid_action_and_class_fall_back_to_default():
    with planner() as p:
        STUB.reply = answer(classification="rock", reason="slip 46%", action="self_destruct", confidence=0.9)
        r = wait_result(p, p.submit_interpretation("r2", 1.0, PERCEPTION, SLIP, None, OPTIONS, default="wait"))
        assert not r.ok and r.action == "wait" and "not in options" in r.error
        STUB.reply = answer(classification="alien", reason="slip 46%", action="wait", confidence=0.9)
        r = wait_result(p, p.submit_interpretation("r2", 1.0, PERCEPTION, SLIP, None, OPTIONS))
        assert not r.ok and r.action == "proceed_slowly" and "classification" in r.error


# ====================================================================== timing, offline, queue
def test_slow_response_times_out_to_fallback():
    STUB.delay = 1.5
    with planner(timeout_s=0.3) as p:
        t0 = time.perf_counter()
        tk = p.submit_choice("r1", 1.0, PERCEPTION, CANDIDATES, default="f2")
        submit_ms = (time.perf_counter() - t0) * 1000
        assert p.poll(tk) is None                        # pending, and polling did not block
        r = wait_result(p, tk, 3.0)
        st = p.stats()
    assert submit_ms < 5, submit_ms
    assert not r.ok and r.choice == "f2" and "timeout" in r.error
    assert 250 <= r.latency_ms < 1000, r.latency_ms
    assert st["calls"] == 1 and st["fallbacks"] == 1


def _timed(fn, n=200):
    """(sorted wall ms, sorted CPU ms of this thread) per call. Wall time includes the OS
    preempting this process (other programs on a busy machine); CPU time is what the call costs."""
    wall, cpu, gc_hits = [], [], [0]
    gc_runs = []
    cb = lambda phase, info: gc_runs.append(info["generation"]) if phase == "start" else None  # noqa: E731
    gc.callbacks.append(cb)
    try:
        for i in range(n):
            before = len(gc_runs)
            c0, t0 = time.thread_time(), time.perf_counter()
            fn(i)
            w, c = (time.perf_counter() - t0) * 1000.0, (time.thread_time() - c0) * 1000.0
            if len(gc_runs) > before and max(gc_runs[before:]) == 2:
                gc_hits[0] += 1        # a full CPython GC pause hit this call: not the planner's doing
                continue
            wall.append(w)
            cpu.append(c)
    finally:
        gc.callbacks.remove(cb)
    if gc_hits[0]:
        print(f"({gc_hits[0]} calls skipped: a full garbage collection ran inside them)")
    return sorted(wall), sorted(cpu)


def _check_fast(label, wall, cpu):
    p99 = wall[int(0.99 * len(wall)) - 1]
    print(f"{label} submit+poll: wall median {statistics.median(wall):.3f} ms, p99 {p99:.3f} ms, "
          f"max {wall[-1]:.3f} ms; CPU max {cpu[-1]:.3f} ms")
    assert p99 < 1.0, (label, p99)
    assert cpu[-1] < 2.0, (label, cpu[-1])      # no call does more than ~1 ms of work, ever


def test_offline_server_never_blocks():
    p = planner(url=DEAD_URL, ready=False)
    try:
        p.check_now()
        assert p.state == "offline" and not p.stats()["online"]
        jpeg = b"\xff\xd8" + bytes(30_000)           # a realistic 30 KB frame (base64 encoded at submit)

        def one(i):
            if i % 3 == 0:
                tk = p.submit_choice("r1", float(i), PERCEPTION, CANDIDATES, default="f1")
            elif i % 3 == 1:
                tk = p.submit_interpretation("r2", float(i), PERCEPTION, SLIP, jpeg, OPTIONS)
            else:
                tk = p.submit_command(float(i), "rover 2 go to 10, -5", FLEET)
            r = p.poll(tk)
            assert r is not None and r.source == "fallback" and "offline" in r.error

        _check_fast("offline", *_timed(one))
        r = p.poll(p.submit_command(1.0, "rover 2 go to 10, -5", FLEET))   # parser fallback is usable
        assert r.command == {"type": "go_to", "rover": "r2", "x": 10.0, "y": -5.0}
        assert p.stats()["calls"] == 0                 # nothing was sent anywhere
    finally:
        p.close()


def test_busy_model_never_blocks_the_caller():
    STUB.delay = 0.4
    with planner() as p:
        p.submit_choice("r9", 0.0, PERCEPTION, CANDIDATES)        # keeps the worker busy

        def one(i):
            p.poll(p.submit_choice(f"r{i % 3 + 1}", float(i), PERCEPTION, CANDIDATES))

        _check_fast("busy", *_timed(one))
        assert p.stats()["queued"] <= 3                # 200 submits for 3 rovers: one queued per rover


def test_queue_replaces_stale_requests_and_orders_by_priority():
    STUB.delay = 0.3
    with planner() as p:
        busy = p.submit_choice("r9", 0.0, PERCEPTION, CANDIDATES)
        time.sleep(0.1)                                 # r9's request is in flight now
        old = p.submit_choice("r1", 1.0, PERCEPTION, CANDIDATES)
        new = p.submit_choice("r1", 2.0, PERCEPTION, CANDIDATES)
        interp = p.submit_interpretation("r3", 3.0, PERCEPTION, SLIP, None, OPTIONS)
        cmd1 = p.submit_command(4.0, "r1 hold", FLEET)
        cmd2 = p.submit_command(5.0, "r2 hold", FLEET)       # commands never replace each other
        r_old = p.poll(old)
        assert r_old is not None and not r_old.ok and "superseded" in r_old.error
        for tk in (busy, new, interp, cmd1, cmd2):
            assert wait_result(p, tk, 5.0).ok
    order = []
    for b in STUB.chats:
        u = b["messages"][-1]["content"]
        order.append("cmd" if u.startswith("Operator") else re.search(r"Rover (r\d)", u).group(1))
    assert order == ["r9", "cmd", "cmd", "r3", "r1"], order    # commands, then interpretations, then choices
    assert len(STUB.chats) == 5                                # the superseded request was never sent


def test_queue_full_and_stale_requests():
    STUB.delay = 0.5
    with planner(max_queue=2, max_wait_s=0.2) as p:
        p.submit_choice("r9", 0.0, PERCEPTION, CANDIDATES)
        time.sleep(0.1)
        a = p.submit_choice("r1", 0.0, PERCEPTION, CANDIDATES)
        b = p.submit_choice("r2", 0.0, PERCEPTION, CANDIDATES)
        c = p.submit_choice("r3", 0.0, PERCEPTION, CANDIDATES)
        rc = p.poll(c)
        assert rc is not None and rc.error == "queue full"
        ra = wait_result(p, a)
        assert not ra.ok and "stale" in ra.error           # waited 0.4 s behind r9 with max_wait 0.2 s
        assert wait_result(p, b).error.startswith("stale")


def test_cancel_drops_queued_requests():
    STUB.delay = 0.3
    with planner() as p:
        p.submit_choice("r9", 0.0, PERCEPTION, CANDIDATES)
        time.sleep(0.05)
        tk = p.submit_choice("r1", 0.0, PERCEPTION, CANDIDATES)
        assert p.cancel() == 1
        assert p.poll(tk).error == "cancelled"


def test_stats_percentiles():
    s = g.latency_summary(list(range(1, 101)), budget_ms=90)
    assert s == {"n": 100, "avg_ms": 50.5, "p50_ms": 50.0, "p95_ms": 95.0, "max_ms": 100.0, "over_budget": 10}
    assert g.latency_summary([], 100)["p95_ms"] == 0.0
    assert g.latency_summary([5.0], 1)["p95_ms"] == 5.0
    STUB.delay = 0.05
    with planner(budget_ms=20.0) as p:
        for i in range(4):
            wait_result(p, p.submit_choice("r1", float(i), PERCEPTION, CANDIDATES))
        st = p.stats()
    assert st["model"] == "gemma4:e2b" and st["online"] and st["state"] == "ready"
    assert st["calls"] == 4 and st["ok"] == 4 and st["fallbacks"] == 0
    assert st["avg_ms"] >= 50 and st["p95_ms"] >= st["avg_ms"] * 0.9 and st["last_ms"] >= 50
    assert st["budget_ms"] == 20.0 and st["over_budget"] == 4
    for key in ("model", "online", "calls", "ok", "fallbacks", "avg_ms", "p95_ms", "last_ms", "budget_ms",
                "over_budget"):
        assert key in st


# ====================================================================== server and model handling
def test_model_fallback_selection_and_stats():
    STUB.models = ["llama3:latest", "gemma3:4b"]          # the user's machine today: no gemma4 yet
    STUB.reply = answer(reason="battery 64% is fine", choice="f1", confidence=0.7)
    with planner() as p:
        st = p.stats()
        assert st["model"] == "gemma3:4b" and st["requested_model"] == "gemma4:e2b"
        assert "gemma4:e2b is not installed" in st["detail"]
        r = wait_result(p, p.submit_choice("r1", 1.0, PERCEPTION, CANDIDATES))
        assert r.ok and r.model == "gemma3:4b"
        assert STUB.chats[-1]["model"] == "gemma3:4b" and STUB.warmups[-1]["model"] == "gemma3:4b"
        STUB.models = ["llama3:latest", "gemma3:4b", "gemma4:e2b"]   # the user pulls gemma4 later
        assert p.check_now() == "ready"
        assert p.stats()["model"] == "gemma4:e2b"
        assert STUB.warmups[-1]["model"] == "gemma4:e2b"            # warmed up again for the new model
    STUB.models = ["llama3:latest"]
    p = planner(ready=False)
    try:
        assert p.check_now() == "no_model"
        assert "ollama pull gemma4:e2b" in p.stats()["detail"]
        r = p.poll(p.submit_choice("r1", 1.0, PERCEPTION, CANDIDATES))
        assert r is not None and "no_model" in r.error
    finally:
        p.close()


def test_warmup_runs_once_and_shows_warming():
    STUB.warmup_delay = 0.6
    p = planner(ready=False)
    try:
        time.sleep(0.25)
        st = p.stats()
        assert st["state"] == "warming" and st["warming"] and not st["online"], st
        r = p.poll(p.submit_choice("r1", 1.0, PERCEPTION, CANDIDATES, default="f3"))
        assert r is not None and r.choice == "f3" and "warming" in r.error     # no waiting while it loads
        assert p.wait_ready(3.0)
        w = STUB.warmups
        assert len(w) == 1 and w[0]["keep_alive"] == "30m" and w[0]["options"]["num_predict"] == 1
        assert p.stats()["warmup_ms"] >= 500
        p.check_now()
        assert len(STUB.warmups) == 1                    # already warm: no second warm-up
    finally:
        p.close()


def test_think_field_rejected_is_dropped():
    STUB.reject_think = True
    STUB.reply = answer(reason="battery 64% is fine", choice="f1", confidence=0.7)
    with planner() as p:
        r = wait_result(p, p.submit_choice("r1", 1.0, PERCEPTION, CANDIDATES))
        assert r.ok
        assert "think" not in STUB.chats[-1] and "think" not in STUB.warmups[-1]


def test_url_fallback_to_second_address():
    with planner(url=DEAD_URL, fallback_urls=(STUB_URL,)) as p:
        assert p.stats()["url"] == STUB_URL
        assert wait_result(p, p.submit_choice("r1", 1.0, PERCEPTION, CANDIDATES)).ok


def test_server_going_down_marks_offline():
    with planner() as p:
        STUB.tags_status = 500
        assert p.check_now() == "offline"
        r = p.poll(p.submit_choice("r1", 1.0, PERCEPTION, CANDIDATES))
        assert r is not None and "offline" in r.error
        STUB.tags_status = 200
        assert p.check_now() == "ready"


def test_disabled_and_env_config(monkeypatch=None):
    env = {"GEMMA_ENABLED": "0", "GEMMA_URL": "http://example:1", "GEMMA_MODEL": "gemma3:4b",
           "GEMMA_TIMEOUT_S": "2.5", "GEMMA_BUDGET_MS": "1500", "GEMMA_FALLBACK_MODELS": "a:1, b:2"}
    old = {k: os.environ.get(k) for k in env}
    os.environ.update(env)
    try:
        cfg = GemmaConfig.from_env()
        assert (cfg.enabled, cfg.url, cfg.model, cfg.timeout_s, cfg.budget_ms) == \
            (False, "http://example:1", "gemma3:4b", 2.5, 1500.0)
        assert cfg.urls == ("http://example:1", "http://localhost:11434")
        assert cfg.fallback_models == ("a:1", "b:2")
        p = GemmaPlanner(cfg)
        assert p.state == "disabled" and not p._threads
        r = p.poll(p.submit_choice("r1", 1.0, PERCEPTION, CANDIDATES))
        assert r is not None and "disabled" in r.error
    finally:
        for k, v in old.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v
    d = GemmaConfig()
    assert d.url == "http://host.docker.internal:11434" and d.model == "gemma4:e2b"
    assert d.timeout_s == 4.0 and d.budget_ms == 3000.0 and d.enabled


def test_fake_backend_whole_pipeline():
    old = os.environ.get("GEMMA_FAKE")
    os.environ["GEMMA_FAKE"] = "1"
    try:
        p = g.planner_from_env(fake_latency_ms=10.0, check_every_s=3600.0)
    finally:
        if old is None:
            os.environ.pop("GEMMA_FAKE", None)
        else:
            os.environ["GEMMA_FAKE"] = old
    with p:
        assert p.wait_ready(2.0) and p.stats()["backend"] == "fake"
        low = dict(PERCEPTION, battery=0.2)
        r = wait_result(p, p.submit_choice("r1", 1.0, low, CANDIDATES))
        assert r.ok and r.choice == "home" and "20%" in r.cited
        r = wait_result(p, p.submit_choice("r1", 1.0, PERCEPTION, CANDIDATES))
        assert r.ok and r.choice == "f2"                                  # largest frontier
        r = wait_result(p, p.submit_interpretation("r2", 1.0, PERCEPTION, SLIP, None, OPTIONS))
        assert r.ok and (r.classification, r.action) == ("soft_ground", "back_out")
        dust = dict(PERCEPTION, lidar_range=2.0)
        r = wait_result(p, p.submit_interpretation("r2", 1.0, dust, {"type": "low_visibility"}, None, OPTIONS))
        assert r.ok and (r.classification, r.action) == ("dust", "wait")
        r = wait_result(p, p.submit_interpretation("r2", 1.0, PERCEPTION, {"type": "weird_noise"}, None, OPTIONS))
        assert r.ok and r.action == "ask_operator" and r.question
        r = wait_result(p, p.submit_command(1.0, "send the orange one back home", FLEET))
        assert r.ok and r.command == {"type": "return_home", "rover": "r1"}
        assert p.stats()["ok"] == 6


def test_prompt_size_under_budget():
    reqs = [g.make_choice_request(1, "r1", 0.0, PERCEPTION, CANDIDATES),
            g.make_interpretation_request(2, "r1", 0.0, PERCEPTION, SLIP, b"x", OPTIONS),
            g.make_command_request(3, 0.0, "rover 2 go to 10, -5", FLEET)]
    for req in reqs:
        system, user = g.build_prompt(req)
        n = g.estimate_tokens(system + user)
        assert n < 400, (req.kind, n)


# ====================================================================== operator commands
PARSER_CASES = [
    ("rover 2 go to 10, -5", {"type": "go_to", "rover": "r2", "x": 10.0, "y": -5.0}),
    ("all return", {"type": "return_home", "rover": "all"}),
    ("r1 explore north", {"type": "explore_direction", "rover": "r1", "direction": "north"}),
    ("hold", {"type": "hold", "rover": None}),
    ("resume", {"type": "resume", "rover": None}),
    ("sample t4", {"type": "sample_target", "rover": None, "target_id": "t4"}),
    ("Send the orange one back home", {"type": "return_home", "rover": "r1"}),
    ("everyone stop!", {"type": "hold", "rover": "all"}),
    ("r3 explore towards the north east", {"type": "explore_direction", "rover": "r3", "direction": "north-east"}),
    ("Rover two, head to x=4 y=-7", {"type": "go_to", "rover": "r2", "x": 4.0, "y": -7.0}),
    ("keep away from 12, -4, about 5 m", {"type": "avoid_area", "x": 12.0, "y": -4.0, "radius": 5.0}),
    ("avoid 3, 8", {"type": "avoid_area", "x": 3.0, "y": 8.0, "radius": 3.0}),
    ("the purple rover should investigate target 4", {"type": "sample_target", "rover": "r3", "target_id": "t4"}),
    ("teal rover, come back", {"type": "return_home", "rover": "r2"}),
    ("r2 go sw", {"type": "explore_direction", "rover": "r2", "direction": "south-west"}),
    ("r1 continue", {"type": "resume", "rover": "r1"}),
]


def test_parser_examples():
    for text, want in PARSER_CASES:
        p = g.parse_command(text, FLEET)
        assert p["command"] == want, (text, p["command"])
        assert p["ack"] and p["question"] is None and p["confidence"] > 0.5
    assert g.parse_command("rover 2 go to 10, -5", FLEET)["ack"] == "r2 going to (10, -5)."
    assert g.parse_command("all return", FLEET)["ack"] == "All rovers returning to the lander."


def test_parser_unclear_cases():
    for text, part in [("do the thing", "did not understand"), ("", "did not understand"),
                       ("r9 hold", "no rover r9"), ("r1 and r2 hold", "which one"),
                       ("avoid the crater", "Which area"), ("sample t9", "do not know target t9")]:
        p = g.parse_command(text, FLEET)
        assert p["command"]["type"] == "unclear" and part in p["question"], (text, p)
    two_orange = {"rovers": [{"id": "r1", "color": "#ff7a45"}, {"id": "r8", "color": "#ffc069"}]}
    p = g.parse_command("orange rover hold", two_orange)
    assert p["command"]["type"] == "unclear" and "r1 or r8" in p["question"]
    p = g.parse_command("rover 5 hold", {})                # no context: any rover id is accepted
    assert p["command"] == {"type": "hold", "rover": "r5"}


def test_colour_names():
    sim_colours = ["#ff7a45", "#36cfc9", "#9254de", "#fadb14", "#73d13d", "#f759ab", "#40a9ff", "#ffc069"]
    assert [g.colour_name(c) for c in sim_colours] == ["orange", "cyan", "purple", "yellow", "green", "pink",
                                                       "blue", "orange"]
    assert g.colour_name("teal") == "cyan" and g.colour_name("nonsense") is None


def test_command_gemma_path():
    STUB.reply = answer(command="go_to", rover="r2", direction="none", x=10, y=-5, radius=0, target_id="none",
                        question="", ack="Rover 2 heading to (10, -5).", confidence=0.9)
    with planner() as p:
        r = wait_result(p, p.submit_command(7.0, "rover 2 go to 10, -5", FLEET))
    assert r.ok and r.source == "gemma" and r.kind == "command" and r.rover_id == "operator"
    assert r.command == {"type": "go_to", "rover": "r2", "x": 10.0, "y": -5.0}
    assert r.ack == "Rover 2 heading to (10, -5)." and r.text == "rover 2 go to 10, -5"
    assert "go_to" in r.trace() and "Gemma" in r.trace()
    body = STUB.chats[-1]
    props = body["format"]["properties"]
    assert props["command"]["enum"] == list(g.COMMANDS)
    assert props["rover"]["enum"] == ["r1", "r2", "r3", "all", "none"]
    assert props["target_id"]["enum"] == ["t4", "none"]
    assert body["options"]["num_predict"] >= 128
    user = body["messages"][-1]["content"]
    assert 'Operator message: "rover 2 go to 10, -5"' in user
    assert 'r1 "Rover 1" orange at (3.2, -1)' in user and "t4 ice found at (12, -4.5)" in user


def test_command_gemma_resolves_names_and_other_commands():
    with planner() as p:
        cases = [
            (dict(command="return_home", rover="r1", direction="none", x=0, y=0, radius=0, target_id="none",
                  question="", ack="Orange rover r1 is heading home.", confidence=0.88),
             "bring the orange one home", {"type": "return_home", "rover": "r1"}),
            (dict(command="avoid_area", rover="none", direction="none", x=12, y=-4.5, radius=0, target_id="none",
                  question="", ack="Avoiding t4.", confidence=0.8),
             "keep everyone away from the ice at t4", {"type": "avoid_area", "x": 12.0, "y": -4.5, "radius": 3.0}),
            (dict(command="explore_direction", rover="r3", direction="northeast", x=0, y=0, radius=0,
                  target_id="none", question="", ack="r3 exploring north-east.", confidence=0.8),
             "r3 scout NE", {"type": "explore_direction", "rover": "r3", "direction": "north-east"}),
            (dict(command="unclear", rover="none", direction="none", x=0, y=0, radius=0, target_id="none",
                  question="Which rover do you mean?", ack="", confidence=0.9),
             "go there", {"type": "unclear", "question": "Which rover do you mean?"}),
        ]
        for reply, text, want in cases:
            STUB.reply = answer(**reply)
            r = wait_result(p, p.submit_command(1.0, text, FLEET))
            assert r.ok and r.command == want, (text, r)
        assert r.question == "Which rover do you mean?" and r.ack == r.question


def test_command_invalid_answers_use_the_parser():
    base = dict(command="go_to", rover="r2", direction="none", x=33, y=7, radius=0, target_id="none",
                question="", ack="ok", confidence=0.9)
    with planner() as p:
        for reply, part in [(base, "coordinates"),                              # invented place
                            (dict(base, x=10, y=5), "coordinates"),             # sign dropped
                            (dict(base, x=10, y=-5, rover="r7"), "rover"),
                            (dict(base, command="teleport"), "command"),
                            (dict(base, command="sample_target", target_id="t9"), "target")]:
            STUB.reply = answer(**reply)
            r = wait_result(p, p.submit_command(1.0, "rover 2 go to 10, -5", FLEET))
            assert not r.ok and r.source == "fallback" and part in r.error, (reply, r.error)
            assert r.command == {"type": "go_to", "rover": "r2", "x": 10.0, "y": -5.0}      # parser's answer
            assert r.ack == "r2 going to (10, -5)." and "parser" in r.trace()
        STUB.delay = 1.0
        p2 = planner(timeout_s=0.2)
        with p2:
            r = wait_result(p2, p2.submit_command(1.0, "all return", FLEET))
            assert not r.ok and "timeout" in r.error and r.command == {"type": "return_home", "rover": "all"}


def test_command_low_confidence_becomes_unclear():
    STUB.reply = answer(command="hold", rover="r3", direction="none", x=0, y=0, radius=0, target_id="none",
                        question="", ack="r3 holding position.", confidence=0.3)
    with planner() as p:
        r = wait_result(p, p.submit_command(1.0, "maybe the purple one should pause or something", FLEET))
    assert r.ok and r.command["type"] == "unclear"
    assert r.question == "Did you mean: r3 holding position?" and r.model_command == {"type": "hold", "rover": "r3"}


# ====================================================================== bench and style
def test_bench_runs_against_stub():
    spec = importlib.util.spec_from_file_location("gemma_bench", ROOT / "scripts" / "gemma_bench.py")
    bench = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(bench)
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        code = bench.main(["--url", STUB_URL, "--n", "2", "--timeout", "5"])
    out = buf.getvalue()
    assert code == 0, out
    assert "paste from here" in out and "to here" in out and "ollama    0.0.0-stub" in out
    assert re.search(r"^choice\s+2\s+\d/2", out, re.M) and re.search(r"^command\s+2", out, re.M)
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        assert bench.main(["--url", DEAD_URL, "--n", "1"]) == 2       # clear message, no traceback
    assert "Cannot reach Ollama" in buf.getvalue()


def test_no_em_dash_in_gemma_files():
    for rel in ("redshift/gemma.py", "tests/test_gemma.py", "scripts/gemma_bench.py", "GEMMA.md"):
        path = ROOT / rel
        if path.exists():
            assert EM_DASH not in path.read_text(encoding="utf-8"), rel


# ====================================================================== plain runner
if __name__ == "__main__":
    start_stub()
    failed = 0
    tests = [(n, f) for n, f in sorted(globals().items()) if n.startswith("test_") and callable(f)]
    for name, fn in tests:
        STUB.reset()
        t0 = time.perf_counter()
        try:
            fn()
            print(f"ok    {name} ({time.perf_counter() - t0:.2f} s)")
        except Exception as e:                # noqa: BLE001
            failed += 1
            print(f"FAIL  {name}: {type(e).__name__}: {e}")
    stop_stub()
    print(f"{len(tests) - failed}/{len(tests)} passed")
    sys.exit(1 if failed else 0)
