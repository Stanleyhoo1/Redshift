"""Protocol v9: distress, helpers, diagnosis, field repair, towing, salvage, leaving a wreck, and repair at the lander.

A rover whose health says it cannot drive (health.py) stops, logs its own diagnosis and sends a distress signal,
if its radio still reaches the lander or another rover. If it does not, the others notice its heartbeat went quiet
for HEARTBEAT_S and go to where it was last heard.

The best placed healthy rover becomes the helper: the closest one with the battery for the trip there, a tow home
and the margin (the rules check it). It drives over (state "rescuing", rescue phase "en_route"), looks the rover
over for DIAG_S with its camera and close up ("diagnosing") and reports. Field-fixable faults (a jammed wheel, a
tripped motor controller, dust on the optics) are fixed on the spot ("repairing"): if that is enough, both carry on.

Otherwise there are three options, and the rules mark which are safe:
  tow      a route home exists within the towing slope limit (the fleet limit minus TOW_SLOPE_MARGIN, and the
           helper's own) and the helper's battery covers it at TOW_ENERGY times the usual cost per metre
  salvage  the wreck still has a working part worth taking and the helper can get home afterwards
  leave    always safe
In autonomous mode Gemma picks among the safe options (a choice request with "rescue" candidates that quote the
numbers; the rules' order when Gemma is offline or unsure) and the decision is logged with a `decision` object; in
supervised mode a request of kind "rescue" asks the operator, with Gemma's pick marked as the suggestion.

Towing is physical: each rover has a rear and a front hitch site at the ends of a virtual tow bar, and the model
has an inactive equality constraint ("connect") for every helper/towed pair. The helper drives to a point ahead
of the disabled rover, turns to face away from it and backs up until the hitch points meet, then the constraint is
switched on (soft first, then stiff). The towed rover's drive actuators go passive (free rolling); the helper
drives home slowly with a stricter slope limit. If physical towing gets stuck (no progress, the hitch pulling
apart, the towed rover tipping), the fallback is kinematic: the towed rover's collisions are switched off and it
follows the hitch point like a trailer (set every physics substep). At the lander the rover is unhitched and
repaired (state "repairing"): field faults are cleared, parts below 30% are replaced with spares when the fleet
has one, the rest is serviced over time.

Salvage: the helper spends SALVAGE_S at the wreck and takes the best working parts (at most SALVAGE_MAX, at least
SALVAGE_MIN %) and the wreck's sample tubes. A part it needs itself is fitted at once (it counts as used); the
others count as fleet spares once delivered to the lander. A wreck that is salvaged or left stays where it is,
marked in fleet `wrecks`. Every claim (frontier, target, focus, open question) of a disabled rover is released at
once so the fleet keeps exploring.
"""
from __future__ import annotations

import math

import numpy as np

from .health import PART_NAME, PARTS, SPARE_KINDS, SPARE_NEED_BELOW, SPARE_OF
from .planner import PENDING, path_length

HEARTBEAT_S = 15.0         # no heartbeat for this long: the others go and look
ARRIVE_M = 2.7             # the helper is "there" this close to the disabled rover
APPROACH_M = 1.9           # it parks this far from it
DIAG_S = 20.0              # close inspection (camera and close up)
FIX_S = 15.0               # field repair
SALVAGE_S = 30.0           # taking parts off a wreck
SALVAGE_MAX = 2            # parts a helper can carry
SALVAGE_MIN = 60.0         # a part this good is worth taking
TOW_V = 0.3                # towing: top speed (m/s) ...
TOW_W = 0.3                # ... and turn rate (rad/s); no pivots in place (wide arcs instead)
TOW_TYRE_MU = 0.3          # the towed rover's tyres: dragged sideways in a turn they skid on the loose regolith
TOW_SLOPE_MARGIN = 4.0     # ... and this much below the fleet's slope limit
TOW_ENERGY = 2.2           # battery per metre while towing, times the helper's own
HITCH_PRE_M = 1.4          # the helper first goes this far beyond the hitch point ...
HITCH_OK_M = 0.18          # ... backs up until the hitch points are this close ...
HITCH_TIMEOUT_S = 40.0     # ... or gives up on a physical hitch after this long
TOW_STUCK_S = 12.0         # no progress for this long while towing: kinematic fallback
TOW_APART_M = 0.6          # the hitch points this far apart while hitched: the bar is not holding
REPAIR_RADIUS = 4.5        # a rover this close to the lander can be repaired there
REPAIR_RATE = 2.0          # service points per second at the lander
DONE_KEEP_S = 60.0         # finished rescues stay in fleet.rescues this long (for the UI)
RESCUE_STATES = ("disabled", "towed", "wrecked")
BUSY_STATES = ("disabled", "towed", "wrecked", "towing", "rescuing", "repairing", "fault")
PLAN_LABEL = {"tow": "tow it home", "salvage": "salvage parts", "leave": "leave it"}
OPTION_LABEL = {"tow": "Tow it home", "salvage": "Salvage parts", "leave": "Leave it"}


def _d(ax: float, ay: float, bx: float, by: float) -> float:
    return math.hypot(ax - bx, ay - by)


class _From:
    """A planning proxy: plan from (x, y) (the disabled rover's spot) inside the helper's compute budget."""

    def __init__(self, helper, x: float, y: float):
        self.helper, self.x, self.y = helper, x, y
        self.searches: dict = {}
        self.plan_fails: dict = {}
        self.degraded = False
        self.exp_used = 0

    def plan_limits(self):
        return self.helper.plan_limits()

    def limit_now(self):
        return self.helper.fleet.slope_limit


class RescueManager:
    def __init__(self, sim):
        self.sim = sim
        self.fleet = sim.fleet
        self.cfg = sim.cfg
        self.rescues: list = []            # dicts, see _new
        self.wrecks: list = []             # {"rover", "x", "y"}
        self.heard: dict = {r.id: (0.0, r.x, r.y, r.state) for r in sim.rovers}
        self.quiet_ok: dict = {}           # rover id -> t: known to be fine with a weak radio, no search until then
        self.kinematic: list = []          # [(helper, towed)] fallback tows (follow_step every substep)
        self._geoms = {}                   # rover id -> [(gid, contype, conaffinity)] while kinematic
        self.fleet.spares = {"drive": 2, "lidar": 1, "camera": 1, "power": 1}
        self.fleet.rescues_msg = self.rescues_msg
        self.fleet.wrecks = self.wrecks

    # ================================================================== lookups
    def _rover(self, rid):
        return next((r for r in self.sim.rovers if r.id == rid), None)

    def active(self, rid: str):
        """The open rescue of rover rid (the one in trouble), or None."""
        return next((x for x in self.rescues if x["rover"] == rid and x["phase"] != "done"), None)

    def by_helper(self, rid: str):
        return next((x for x in self.rescues if x["helper"] == rid and x["phase"] != "done"), None)

    def rescues_msg(self) -> list:
        return [{"rover": x["rover"], "helper": x["helper"], "phase": x["phase"], "plan": x["plan"]}
                for x in self.rescues]

    def comms_link(self, r) -> bool:
        """Does r's radio reach the lander or another working rover?"""
        rng = r.health.effects()["comms_m"]
        if math.hypot(r.x, r.y) <= rng:
            return True
        return any(o is not r and o.state != "wrecked" and _d(o.x, o.y, r.x, r.y) <= rng for o in self.sim.rovers)

    def _log(self, t, level, msg, rover=None, at=None, source=None, decision=None):
        self.fleet.log(t, level, msg, rover, at=at, source=source, decision=decision)

    def _say(self, rid, t, text, kind="info", reply_to=None, request_id=None, source="rules"):
        adv = getattr(self.sim, "advisor", None)
        if adv is not None:
            adv.say(rid, t, text, kind, source=source, reply_to=reply_to, request_id=request_id)

    # ================================================================== rover hooks
    def check_now(self, rover, t: float | None = None) -> None:
        """Health just changed (a demo hit, a physics hit): react now instead of at the next cycle."""
        t = self.sim.t if t is None else t
        if not rover.health.mobile() and rover.state not in RESCUE_STATES + ("repairing",):
            self.on_immobile(rover, t)

    def on_immobile(self, rover, t: float) -> None:
        """The rover cannot drive any more: stop, release every claim, log the diagnosis, call for help."""
        rid = rover.id
        was = rover.state
        # a helper that breaks down mid-rescue: its rescue needs someone else; a tow comes apart
        mine = self.by_helper(rid)
        if mine is not None:
            self._release_helper(mine, t, f"{rid} broke down itself")
        if was == "towing":
            self._unhitch_pair(rid, t)
        rover.release_all(t, "immobile")
        rover.state = "disabled"
        rover.cmd_v = rover.cmd_w = 0.0
        diag = rover.health.diagnosis()
        here = (rover.x, rover.y)
        self._log(t, "warn", f"{rid} immobile ({rover.health.status()}, health {rover.health.hp()}). {diag}", rid,
                  at=here)
        rover.mem.on_hazard("immobile", t, here, f"I could not drive any more: {rover.health.short()}",
                            handled="stopped and called for help")
        if self.active(rid) is not None:
            return
        if self.comms_link(rover):
            self._distress(rover, t)
        else:
            self._log(t, "warn", f"{rid} radio too weak to call for help (comms {rover.health.value('comms'):.0f}%, "
                                 f"range {rover.health.effects()['comms_m']:.0f} m). Waiting.", rid, at=here)

    def _distress(self, rover, t: float, relayed_by: str | None = None) -> dict:
        rid = rover.id
        here = (rover.x, rover.y)
        text = f"{rid}: immobile. {rover.health.short()[:1].upper() + rover.health.short()[1:]}. " \
               f"Requesting help at ({rover.x:.1f}, {rover.y:.1f})."
        rec = self._new(rid, t, here)
        self._log(t, "warn", f"Distress signal from {rid}: {text.split(': ', 1)[1]}", rid, at=here)
        self._say(rid, t, text, "info")
        self._pick_helper(rec, t)
        return rec

    def _new(self, rid: str, t: float, at, silent: bool = False) -> dict:
        old = [x for x in self.rescues if x["rover"] == rid and x["phase"] == "done"]
        for x in old:
            self.rescues.remove(x)
        rec = {"rover": rid, "helper": None, "phase": "en_route", "plan": None, "t0": t, "t_phase": t,
               "at": (float(at[0]), float(at[1])), "silent": silent, "contact": not silent, "goal": None,
               "fails": 0, "exclude": set(), "diag": None, "preset": None, "cid": None, "qid": None,
               "ticket": None, "opts": None, "route": None, "proxy": None, "wait_note": -1e9, "done_t": None,
               "hitch": None, "progress": None, "fixed": False}
        self.rescues.append(rec)
        return rec

    # ================================================================== helpers
    def helper_need(self, r, at) -> float:
        """Battery a helper needs: there, a tow home, and its margin."""
        d_to = 1.3 * _d(r.x, r.y, at[0], at[1])
        d_home = 1.3 * math.hypot(at[0], at[1])
        return r.energy_for_distance(d_to) + TOW_ENERGY * r.energy_for_distance(d_home) + r.battery_margin + 0.03

    def can_help(self, r, rec) -> str | None:
        """Why r cannot be the helper for rec, or None."""
        if r.id == rec["rover"]:
            return "it is the rover in trouble"
        if r.state in BUSY_STATES:
            return {"rescuing": "already helping another rover", "towing": "towing another rover",
                    "repairing": "being repaired at the lander", "fault": "motors offline (rebooting)"}.get(
                r.state, "out of action itself")
        if not r.health.mobile():
            return "can't drive itself"
        if r.state == "returning" and r.return_reason in ("battery", "range"):
            return f"returning to charge (battery {r.battery * 100:.0f}%)"
        need = self.helper_need(r, rec["at"])
        if r.battery < need:
            return f"battery {r.battery * 100:.0f}% is short of the {need * 100:.0f}% needed (there, a tow home, margin)"
        return None

    def _pick_helper(self, rec: dict, t: float, quiet: bool = False):
        best = None
        notes = []
        for r in self.sim.rovers:
            if r.id in rec["exclude"] or r.held:
                continue
            why = self.can_help(r, rec)
            if why is not None:
                notes.append(f"{r.id} {why}")
                continue
            d = _d(r.x, r.y, rec["at"][0], rec["at"][1])
            if best is None or d < best[0]:
                best = (d, r)
        if best is None:
            if t - rec["wait_note"] > 60.0 and not quiet:
                rec["wait_note"] = t
                self._log(t, "info", f"No rover can help {rec['rover']} yet ({'; '.join(notes) or 'none free'}). "
                                     f"Waiting for one.", rec["rover"], at=rec["at"])
            return None
        self.assign(rec, best[1], t)
        return best[1]

    def assign(self, rec: dict, r, t: float, operator: bool = False, reply_to=None) -> None:
        """r becomes rec's helper (replacing any other)."""
        old = self._rover(rec["helper"]) if rec["helper"] else None
        if old is not None and old is not r:
            self._release_helper(rec, t, f"{r.id} takes over", pick=False)
        adv = getattr(self.sim, "advisor", None)
        if adv is not None:
            adv._free_up(r, t, f"helping {rec['rover']}")
        r.release_all(t, "rescue", quiet=True)
        r.state = "rescuing"
        r.return_reason = None
        rec["helper"] = r.id
        rec["goal"] = None
        rec["fails"] = 0
        if rec["phase"] not in ("en_route",):
            rec["phase"] = "en_route"
        rec["t_phase"] = t
        d = _d(r.x, r.y, rec["at"][0], rec["at"][1])
        from .fleet import compass
        where = compass(rec["at"][0] - r.x, rec["at"][1] - r.y)
        what = "its last known position" if rec["silent"] and not rec["contact"] else rec["rover"]
        text = (f"{r.id}: going to help {rec['rover']}, {d:.0f} m {where}. Battery {r.battery * 100:.0f}%, enough for "
                f"the trip and a tow home.")
        self._log(t, "info", f"{r.id} is the helper for {rec['rover']}: heading to {what}, {d:.0f} m {where} "
                             f"(battery {r.battery * 100:.0f}%, needs {self.helper_need(r, rec['at']) * 100:.0f}%).",
                  r.id, at=rec["at"], source="operator" if operator else None)
        self._say(r.id, t, text, "ack" if operator else "info", reply_to=reply_to)

    def _release_helper(self, rec: dict, t: float, why: str, pick: bool = True) -> None:
        h = self._rover(rec["helper"]) if rec["helper"] else None
        if h is not None:
            if h.state == "towing":
                self._unhitch(h, self._rover(rec["rover"]), t)
            if h.state in ("rescuing", "towing"):
                h.state = "exploring"
                h.goal, h.path = None, []
                h.next_goal_try = t
            rec["exclude"].add(h.id)
            self._log(t, "warn", f"{h.id} stops helping {rec['rover']}: {why}.", h.id, at=(h.x, h.y))
            self._say(h.id, t, f"Stopping my rescue of {rec['rover']}: {why}."
                      + (" Looking for another rover to help." if pick else ""), "info")
        rec["helper"] = None
        if rec["phase"] != "done":
            rec["phase"] = "en_route"
            rec["t_phase"] = t
            rec["qid"] = None
            rec["ticket"] = None
            rec["opts"] = None
            rec["route"] = None
            rec["hitch"] = None
            if pick:
                self._pick_helper(rec, t)

    # ================================================================== per cycle
    def update(self, t: float) -> None:
        """Sim thread, once per decision cycle after the rovers: heartbeats, rescue phases, decisions."""
        rovers = self.sim.rovers
        for r in rovers:
            if self.comms_link(r):
                self.heard[r.id] = (t, r.x, r.y, r.state)
        for r in rovers:
            last = self.heard.get(r.id)
            if last is None or t - last[0] < HEARTBEAT_S or self.active(r.id) is not None:
                continue
            if last[3] in ("wrecked",) or t < self.quiet_ok.get(r.id, -1.0):
                continue
            self._silent(r, last, t)
        for rec in list(self.rescues):
            if rec["phase"] == "done":
                if t - (rec["done_t"] or t) > DONE_KEEP_S:
                    self.rescues.remove(rec)
                continue
            self._step(rec, t)

    def _silent(self, r, last, t: float) -> None:
        lt, lx, ly, _st = last
        rec = self._new(r.id, t, (lx, ly), silent=True)
        noticer = min((o for o in self.sim.rovers if o is not r and o.state not in RESCUE_STATES),
                      key=lambda o: _d(o.x, o.y, lx, ly), default=None)
        who = noticer.id if noticer is not None else "The fleet"
        self._log(t, "warn", f"{who} noticed: no heartbeat from {r.id} for {t - lt:.0f} s (last heard at "
                             f"({lx:.1f}, {ly:.1f})). Going to its last known position.",
                  noticer.id if noticer is not None else None, at=(lx, ly))
        self._pick_helper(rec, t)

    def _step(self, rec: dict, t: float) -> None:
        rover = self._rover(rec["rover"])
        if rover is None:
            return
        if rover.health.mobile() and rec["phase"] in ("en_route", "diagnosing") and not rec["silent"]:
            self._finish(rec, t, None, f"{rover.id} can drive again (health {rover.health.hp()}). Rescue called off.")
            return
        h = self._rover(rec["helper"]) if rec["helper"] else None
        if h is None:
            if rec["phase"] in ("en_route",):
                self._pick_helper(rec, t, quiet=False)
            return
        if h.state not in ("rescuing", "towing"):
            if h.state in ("exploring", "investigating", "observing") and h.health.mobile() \
                    and not h._need_return(t) and h.op_goal is None and not h.held:
                # side-tracked (a target, a creature, a curious detour): the stranded rover comes first
                self._log(t, "info", f"{h.id} drops what it was doing ({h.state}): {rec['rover']} comes first.",
                          h.id, at=(h.x, h.y))
                h.release_all(t, "rescue", quiet=True)
                h.state = "rescuing"
                h.goal, h.path, h.plan_t = None, [], -1e9
                rec["goal"] = None
            else:
                # the helper was taken off the job (low battery, a fault, an operator order)
                self._release_helper(rec, t, f"it is {h.state} now")
                return
        ph = rec["phase"]
        if ph == "en_route":
            self._en_route(rec, rover, h, t)
        elif ph == "diagnosing":
            if rec.get("diag") is None and t - rec["t_phase"] >= DIAG_S:
                self._diagnosed(rec, rover, h, t)
            elif rec.get("diag") is not None:
                self._decide_step(rec, rover, h, t)
        elif ph == "repairing" and t - rec["t_phase"] >= FIX_S:
            self._field_fixed(rec, rover, h, t)
        elif ph == "salvaging" and t - rec["t_phase"] >= SALVAGE_S:
            self._salvaged(rec, rover, h, t)
        elif ph == "towing":
            self._tow_watch(rec, rover, h, t)

    # ------------------------------------------------------------------ en route
    def _en_route(self, rec, rover, h, t: float) -> None:
        if h.battery < 0.06:
            self._release_helper(rec, t, f"battery {h.battery * 100:.0f}%")
            return
        if rec["silent"] and not rec["contact"]:
            rng = max(rover.health.effects()["comms_m"], 6.0)
            if _d(h.x, h.y, rover.x, rover.y) <= rng:
                rec["contact"] = True
                rec["at"] = (rover.x, rover.y)
                rec["goal"] = None
                self.heard[rover.id] = (t, rover.x, rover.y, rover.state)
                if rover.health.mobile() and rover.state not in RESCUE_STATES:
                    self.quiet_ok[rover.id] = t + 300.0
                    self._finish(rec, t, None, f"{h.id} found {rover.id}: it can drive, only its radio is weak (comms "
                                               f"{rover.health.value('comms'):.0f}%, range "
                                               f"{rover.health.effects()['comms_m']:.0f} m). Carrying on.")
                    return
                self._log(t, "warn", f"{h.id} found {rover.id} (its radio could not call for help): immobile. "
                                     f"{rover.health.short()}.", h.id, at=rec["at"])
            elif _d(h.x, h.y, rec["at"][0], rec["at"][1]) < 2.0:
                self._finish(rec, t, None, f"{h.id} reached {rover.id}'s last known position but it is not there "
                                           f"(it may be driving with a weak radio). Carrying on.")
                self.quiet_ok[rover.id] = t + 120.0
                return
        target = (rover.x, rover.y) if rec["contact"] else rec["at"]
        if _d(h.x, h.y, target[0], target[1]) <= ARRIVE_M and rec["contact"]:
            rec["phase"] = "diagnosing"
            rec["t_phase"] = t
            h.goal, h.path = None, []
            self._log(t, "info", f"{h.id} reached {rover.id}. Diagnosing it (camera and close inspection, about "
                                 f"{DIAG_S:.0f} s).", h.id, at=(rover.x, rover.y))
            self._say(h.id, t, f"Reached {rover.id}, looking it over.", "info")

    def approach_goal(self, rec, h, target):
        """A safe spot APPROACH_M from the target, on the helper's side."""
        base = math.atan2(h.y - target[1], h.x - target[0])
        for off in (0.0, 0.5, -0.5, 1.0, -1.0, 1.6, -1.6, 2.3, -2.3, math.pi):
            a = base + off
            gx, gy = target[0] + APPROACH_M * math.cos(a), target[1] + APPROACH_M * math.sin(a)
            if h._safe_cell(gx, gy) and self.fleet.water_zone_at(gx, gy, margin=0.8) is None:
                return gx, gy
        return target[0] + APPROACH_M * math.cos(base), target[1] + APPROACH_M * math.sin(base)

    def drive(self, h, t: float) -> bool:
        """Called in the helper's decision cycle (its compute budget). True if it set the motor command itself."""
        rec = self.by_helper(h.id)
        if rec is None:
            h.state = "exploring"
            h.goal, h.path = None, []
            h.next_goal_try = t
            return False
        rover = self._rover(rec["rover"])
        ph = rec["phase"]
        if ph == "en_route":
            target = (rover.x, rover.y) if rec["contact"] else rec["at"]
            if rec["goal"] is None or (rec["contact"] and _d(*rec["goal"], *target) > APPROACH_M + 1.0):
                rec["goal"] = self.approach_goal(rec, h, target)
            if _d(h.x, h.y, target[0], target[1]) <= ARRIVE_M and rec["contact"]:
                h._face(*target)
                return True
            return self._goto(h, rec, rec["goal"], t)
        if ph == "towing":
            return self._tow_drive(rec, rover, h, t)
        # diagnosing, repairing, salvaging, waiting for a decision: stand by the rover, camera on it
        h.goal, h.path = None, []
        if rover is not None:
            h._face(rover.x, rover.y)
        if ph == "diagnosing" and rec.get("diag") is not None and rec.get("opts") is None:
            self._plan_options(rec, rover, h, t)      # the tow route is planned inside the helper's budget
        return True

    def _goto(self, h, rec, goal, t: float, limit=None, max_exp: int = 30000) -> bool:
        h.goal = goal
        if not h._need_replan(t):
            return False
        path = self.fleet.plan(t, h, goal, max_exp=max_exp, limit=limit)
        h._after_replan(t, path)
        if path is PENDING:
            h._hold_if_blocked()
        elif path is None:
            rec["fails"] += 1
            if h.block_info is not None:
                h.path = []
            h.plan_t = t
            rec["goal"] = None
            if rec["fails"] >= 6:
                rec["fails"] = 0
                rec["unreachable"] = rec.get("unreachable", 0) + 1
                self._release_helper(rec, t, "no safe path to it")
                if rec["helper"] is None and rec["phase"] != "done" and rec["unreachable"] >= 1 and not any(
                        self.can_help(r, rec) is None and r.id not in rec["exclude"] and not r.held
                        for r in self.sim.rovers):
                    self._unreachable(rec, t)
        else:
            rec["fails"] = 0
            h._set_path(t, path)
        return False

    # ------------------------------------------------------------------ diagnosis and field repair
    def _diagnosed(self, rec, rover, h, t: float) -> None:
        hp = rover.health
        fix = hp.fixable()
        v = hp.values()
        words = {"jam": "jammed wheel", "trip": "tripped motor controller", "dirt": "dust on the optics"}
        broken = [p for p in PARTS if hp.value(p, field_fixed=True) < 95.0]
        bits = []
        for p in sorted(broken, key=lambda p_: hp.value(p_, field_fixed=True))[:4]:
            bits.append(f"{PART_NAME[p]} {hp.value(p, field_fixed=True):.0f}%")
        found = [f"{words.get(k, k)} ({PART_NAME[p]})" for p, k, _pts in fix]
        mobile_after = hp.mobile(field_fixed=True)
        if fix and mobile_after:
            verdict = "Fixable here."
        elif fix:
            verdict = "Part of it is fixable here, but it still won't drive after that."
        else:
            verdict = "Nothing I can fix here."
        text = (f"Diagnosed {rover.id}: " + (", ".join(found) + "; " if found else "")
                + ("damage: " + ", ".join(bits) if bits else "no lasting damage") + f". {verdict}")
        rec["diag"] = text
        self._log(t, "info", f"{h.id} {text[0].lower() + text[1:]}", h.id, at=(rover.x, rover.y))
        self._say(h.id, t, text, "info", reply_to=rec.get("cid"))
        if fix:
            rec["phase"] = "repairing"
            rec["t_phase"] = t
            names = ", ".join(words.get(k, k) for _p, k, _pts in fix)
            self._log(t, "info", f"{h.id} fixing {rover.id} in the field: {names} (about {FIX_S:.0f} s).", h.id,
                      at=(rover.x, rover.y))

    def _field_fixed(self, rec, rover, h, t: float) -> None:
        done = rover.health.field_fix()
        rec["fixed"] = True
        words = {"jam": "freed its jammed", "trip": "reset its tripped", "dirt": "cleaned its"}
        what = ", ".join(f"{words.get(k, 'fixed its')} {PART_NAME[p]}" + (" controller" if k == "trip" else "")
                         for p, k, _b, _a in done) or "found nothing to fix"
        if rover.health.mobile():
            msg = f"{h.id} {what}. {rover.id} can drive again (health {rover.health.hp()})."
            self._log(t, "info", msg, h.id, at=(rover.x, rover.y))
            self._say(h.id, t, msg.split(" ", 1)[1][0].upper() + msg.split(" ", 1)[1][1:], "info")
            rover.state = "exploring"
            rover.goal, rover.path = None, []
            rover.next_goal_try = t
            rover._replan_now(t)
            self._finish(rec, t, None, None)
            if rover.health.needs_repair():
                rover._go_home(t, "repair")
            return
        self._log(t, "info", f"{h.id} {what}, but {rover.id} still can't drive ({rover.health.short()}).", h.id,
                  at=(rover.x, rover.y))
        rec["phase"] = "diagnosing"
        rec["t_phase"] = t
        rec["diag"] = (rec["diag"] or "") + " Field repair done; it still can't drive."

    # ------------------------------------------------------------------ the decision: tow, salvage or leave
    def _plan_options(self, rec, rover, h, t: float) -> None:
        """Work out which options are safe (the tow route is an A* plan from the rover to the lander inside the
        helper's budget, possibly over several cycles)."""
        if rec["route"] is None:
            if rec["proxy"] is None:
                rec["proxy"] = _From(h, rover.x, rover.y)
            lim = self.tow_limit(h)
            path = self.fleet.plan(t, rec["proxy"], h.dock, limit=lim, max_exp=24000)
            if path is PENDING:
                return
            if path is None:
                rec["route"] = ("none", None, 0.0, lim)
            else:
                from .advisor import path_stats
                length, smax = path_stats(self.fleet.map, [(rover.x, rover.y)] + list(path))
                rec["route"] = ("ok", path, length, smax, lim)
        rec["opts"] = self.options(rec, rover, h)

    def tow_limit(self, h) -> float:
        lim = self.fleet.slope_limit
        cap = h.health.effects()["slope_cap"]
        if cap is not None:
            lim = min(lim, cap)
        return max(6.0, lim - TOW_SLOPE_MARGIN)

    def options(self, rec, rover, h) -> dict:
        """{"tow": {...}, "salvage": {...}, "leave": {...}}: each with ok (safe), why (if not) and summary."""
        out = {}
        hp = rover.health
        route = rec.get("route")
        spares = self.fleet.spares
        need = [SPARE_OF[p] for p in PARTS if p in SPARE_OF and hp.value(p, field_fixed=True) < SPARE_NEED_BELOW]
        have = all(spares.get(k, 0) >= need.count(k) for k in set(need))
        spare_txt = (f"lander repair needs {len(need)} spare{'s' if len(need) != 1 else ''} "
                     f"({', '.join(need)}), {'in stock' if have else 'not all in stock'}" if need else
                     "lander repair needs no spares")
        if route is None or route[0] != "ok":
            lim = route[3] if route is not None else self.tow_limit(h)
            out["tow"] = {"ok": False, "why": f"no route home within the {lim:.0f} deg towing slope limit",
                          "summary": ""}
        else:
            _st, _path, length, smax, lim = route
            cost = TOW_ENERGY * h.energy_for_distance(length * 1.15)
            after = h.battery - cost
            ok = after >= h.battery_margin
            out["tow"] = {"ok": ok, "why": None if ok else f"battery {h.battery * 100:.0f}% would leave "
                                                            f"{max(after, 0.0) * 100:.0f}% (needs {h.battery_margin * 100:.0f}%)",
                          "summary": f"tow {rover.id} home: route {length:.0f} m, slope max {smax:.0f} deg (tow limit "
                                     f"{lim:.0f} deg), {h.id} battery after {max(after, 0.0) * 100:.0f}%, {spare_txt}",
                          "length": length, "after": after, "spares_ok": have, "need": need}
        good = sorted((p for p in PARTS if p in SPARE_OF and hp.value(p) >= SALVAGE_MIN),
                      key=lambda p: (spares.get(SPARE_OF[p], 0), -hp.value(p)))[:SALVAGE_MAX]
        home = h.energy_for_distance(1.3 * math.hypot(h.x, h.y)) + h.battery_margin
        if not good:
            out["salvage"] = {"ok": False, "why": "no part on it is still worth taking", "summary": ""}
        else:
            ok = h.battery >= home
            parts = ", ".join(f"{PART_NAME[p]} {hp.value(p):.0f}%" for p in good)
            stock = ", ".join(f"{k} {spares.get(k, 0)}" for k in SPARE_KINDS)
            out["salvage"] = {"ok": ok, "why": None if ok else f"battery {h.battery * 100:.0f}% too low",
                              "summary": f"salvage {rover.id}: take {parts} ({SALVAGE_S:.0f} s here; spares now "
                                         f"{stock}), {h.id} battery {h.battery * 100:.0f}%",
                              "parts": good}
        out["leave"] = {"ok": True, "why": None,
                        "summary": f"leave {rover.id} here and keep exploring, {h.id} battery {h.battery * 100:.0f}%"}
        return out

    def rules_order(self, rec, rover) -> list:
        """The rules' preference among the safe options: tow a repairable rover home; salvage a wreck whose repair
        the spares cannot cover; leave when nothing else is safe."""
        opts = rec["opts"]
        wrecked = rover.health.status() == "wrecked"
        tow = opts["tow"]
        order = []
        if tow["ok"] and (not wrecked or tow.get("spares_ok")):
            order.append("tow")
        if opts["salvage"]["ok"]:
            order.append("salvage")
        if tow["ok"] and "tow" not in order:
            order.append("tow")
        order.append("leave")
        return order

    def _question(self, rec, rover) -> str:
        return (f"{rover.id} immobile at ({rover.x:.0f}, {rover.y:.0f}), health {rover.health.hp()} "
                f"({rover.health.status()}): {rover.health.short()}")

    def _decide_step(self, rec, rover, h, t: float) -> None:
        """Options known: apply an operator's order, or decide (autonomous) / ask (supervised)."""
        if rec.get("opts") is None or rec.get("decided"):
            return
        order = self.rules_order(rec, rover)
        if rec.get("preset"):
            plan, cid = rec["preset"], rec.get("cid")
            rec["preset"] = None
            if plan in order:
                self.apply_plan(rec, plan, t, by="operator", reply_to=cid)
            else:
                why = rec["opts"][plan]["why"] or "not possible"
                self._say(h.id, t, f"Can't {PLAN_LABEL[plan]} after all: {why}.", "refuse", reply_to=cid)
                rec["decided"] = False
            return
        adv = getattr(self.sim, "advisor", None)
        planner = getattr(adv, "planner", None)
        online = planner is not None and planner.online
        if rec.get("ticket") is None and online and len(order) > 1:
            cands = [{"id": o, "kind": "rescue", "summary": rec["opts"][o]["summary"]} for o in order]
            perc = dict(adv._perception(h), rover_hp=rover.health.hp(), rover_status=rover.health.status())
            rec["ticket"] = planner.submit_choice(h.id, t, perc, cands, default=order[0])
            rec["wall"] = adv.clock()
            if adv.wait_for_gemma:
                adv._wait(rec["ticket"])
        res = None
        if rec.get("ticket") is not None:
            res = planner.poll(rec["ticket"])
            if res is None and adv.clock() - rec["wall"] < planner.cfg.timeout_s + 2.0:
                if adv.supervised and rec.get("qid") is None:
                    self._ask(rec, rover, h, order, None, t)
                return
        ok = res is not None and res.ok and res.choice in order
        if adv is not None and adv.supervised:
            if rec.get("qid") is None:
                self._ask(rec, rover, h, order, res if ok else None, t)
            elif ok and res.confidence >= 0.5 and adv.requests.get(rec["qid"]) is not None and \
                    adv.requests[rec["qid"]].get("suggest") is None:
                q = adv.requests[rec["qid"]]
                q["suggest"], q["suggest_conf"] = res.choice, res.confidence
                q["question"] += f" Gemma suggests: {PLAN_LABEL[res.choice]} ({res.confidence:.2f})."
            return
        if ok and res.confidence >= 0.5:
            choice, by, conf, why = res.choice, "gemma", res.confidence, res.reason
        elif ok:
            choice, by, conf, why = order[0], "rules", res.confidence, f"Gemma unsure ({res.confidence:.2f}): the rules' pick"
        else:
            choice, by, conf, why = order[0], "rules", None, self._rules_why(rec, order)
        self._decided(rec, rover, h, order, choice, by, conf, why, t)
        self.apply_plan(rec, choice, t, by=by)

    def _rules_why(self, rec, order) -> str:
        o = rec["opts"]
        if order[0] == "tow":
            tw = o["tow"]
            return (f"repairable and a safe tow: route {tw['length']:.0f} m within the towing limit, helper battery "
                    f"after about {max(tw['after'], 0.0) * 100:.0f}%" + (", spares in stock" if tw.get("spares_ok") else ""))
        if order[0] == "salvage":
            miss = [k for k in ("tow",) if not o[k]["ok"]]
            return ("tow not safe (" + o["tow"]["why"] + "); " if miss else "a wreck the spares can't fix; ") + \
                "salvaging its working parts"
        return "no tow or salvage is safe: " + "; ".join(o[k]["why"] for k in ("tow", "salvage") if o[k]["why"])

    def _decided(self, rec, rover, h, order, choice, by, conf, why, t: float) -> None:
        adv = getattr(self.sim, "advisor", None)
        q = self._question(rec, rover)
        missing = [f"{k} not offered: {rec['opts'][k]['why']}" for k in ("tow", "salvage") if k not in order]
        full = why + (f" ({'; '.join(missing)})" if missing else "")
        if adv is not None:
            adv._decided(h, "rescue", q, list(order), choice, by, conf, full, (rover.x, rover.y), t,
                         label=PLAN_LABEL[choice], force=True)
        else:
            self._log(t, "info", f"{h.id} decided ({by}): {PLAN_LABEL[choice]}. {q}.", h.id, at=(rover.x, rover.y))

    def option_meta(self, rec, rover, h) -> tuple:
        """Protocol v9 request options with the rules' verdict: [(id, label)] (safe ones first, in the rules' order)
        and {id: {"safe", "note"}}."""
        o = rec["opts"]
        order = self.rules_order(rec, rover)
        ids = order + [k for k in ("tow", "salvage", "leave") if k not in order]
        labels, meta = [], {}
        for k in ids:
            ok = k in order
            if k == "tow":
                lab = f"Tow it home ({o['tow']['length']:.0f} m)" if o["tow"].get("length") else "Tow it home"
                note = (f"Route at most {rec['route'][3]:.0f} deg, {h.id} battery after tow about "
                        f"{max(o['tow']['after'], 0.0) * 100:.0f}%" if ok else o["tow"]["why"])
                if ok and o["tow"].get("need"):
                    note += f"; repair needs {len(o['tow']['need'])} spare(s)" + (
                        "" if o["tow"].get("spares_ok") else ", not all in stock")
            elif k == "salvage":
                parts = o["salvage"].get("parts") or []
                lab = f"Salvage parts ({', '.join(PART_NAME[p] for p in parts)})" if parts else "Salvage parts"
                note = (f"Takes {', '.join(f'{PART_NAME[p]} {rover.health.value(p):.0f}%' for p in parts)} into "
                        f"spares, about {SALVAGE_S:.0f} s" if ok else o["salvage"]["why"])
            else:
                lab, note = "Leave it", "The wreck stays where it is, marked on the map"
            labels.append((k, lab))
            meta[k] = {"safe": ok, "note": note[:120]}
        return labels, meta

    def _ask(self, rec, rover, h, order, res, t: float) -> None:
        adv = self.sim.advisor
        opts, meta = self.option_meta(rec, rover, h)
        missing = [f"{OPTION_LABEL[k].split(' ')[0]} is not safe: {rec['opts'][k]['why']}." for k in ("tow", "salvage")
                   if k not in order]
        text = (f"{rec['diag']} {' '.join(missing)} " if rec.get("diag") else "") + "What should I do with " + \
               f"{rover.id}?"
        sug = res.choice if res is not None and res.confidence >= 0.5 and res.choice in order else None
        if sug is not None:
            text += f" Gemma suggests: {PLAN_LABEL[sug]} ({res.confidence:.2f})."
        rec["qid"] = adv.open_rescue_request(h, rec, opts, order[0], text, sug,
                                             None if sug is None else res.confidence, t, meta=meta)

    # ------------------------------------------------------------------ carrying out a plan
    def apply_plan(self, rec, plan: str, t: float, by: str = "rules", reply_to=None) -> str | None:
        """Carry out tow / salvage / leave. Returns None, or why it can't (unsafe now)."""
        rover = self._rover(rec["rover"])
        h = self._rover(rec["helper"]) if rec["helper"] else None
        if plan != "leave" and (h is None or rec.get("opts") is None):
            return "the helper has not looked at it yet"
        if plan != "leave" and not rec["opts"][plan]["ok"]:
            return rec["opts"][plan]["why"]
        rec["decided"] = True
        rec["plan"] = plan
        adv = getattr(self.sim, "advisor", None)
        if rec.get("qid") is not None and adv is not None and rec["qid"] in adv.requests:
            adv._close(rec["qid"], "decided", t)
        rec["qid"] = None
        src = "operator" if by == "operator" else None
        if plan == "leave":
            self._leave(rec, rover, h, t, by, reply_to)
        elif plan == "salvage":
            rec["phase"] = "salvaging"
            rec["t_phase"] = t
            self._log(t, "info", f"{h.id} salvaging {rover.id}: taking "
                                 f"{', '.join(PART_NAME[p] for p in rec['opts']['salvage']['parts'])} (about "
                                 f"{SALVAGE_S:.0f} s).", h.id, at=(rover.x, rover.y), source=src)
            if by == "operator":
                self._say(h.id, t, f"Salvaging {rover.id}: taking "
                                   f"{', '.join(PART_NAME[p] for p in rec['opts']['salvage']['parts'])}.", "ack",
                          reply_to=reply_to)
        else:
            rec["phase"] = "towing"
            rec["t_phase"] = t
            rec["hitch"] = {"stage": "approach", "t0": t, "best": 1e9, "best_t": t}
            h.state = "towing"
            h.goal, h.path = None, []
            self._log(t, "info", f"{h.id} towing {rover.id} home: hitching up first.", h.id, at=(rover.x, rover.y),
                      source=src)
            if by == "operator":
                self._say(h.id, t, f"Towing {rover.id} home ({rec['opts']['tow']['length']:.0f} m), hitching up.",
                          "ack", reply_to=reply_to)
        return None

    def _unreachable(self, rec, t: float) -> None:
        """No capable rover can find a safe path to it (in a lake, boxed in by steep ground): the rules leave it."""
        rover = self._rover(rec["rover"])
        tried = sorted(rec["exclude"])
        why = f"no safe path to it for {', '.join(tried)}" if tried else "no rover can reach it"
        adv = getattr(self.sim, "advisor", None)
        q = self._question(rec, rover)
        who = self._rover(tried[0]) if tried else rover
        if adv is not None:
            adv._decided(who, "rescue", q, ["leave"], "leave", "rules", None, why, (rover.x, rover.y), t,
                         label=PLAN_LABEL["leave"], force=True)
        self._leave(rec, rover, None, t, "rules")

    def _leave(self, rec, rover, h, t: float, by: str, reply_to=None) -> None:
        self._mark_wreck(rover, t)
        msg = f"Leaving {rover.id} at ({rover.x:.0f}, {rover.y:.0f}), marked as a wreck. The fleet carries on without it."
        src = "operator" if by == "operator" else None
        if h is not None:
            self._log(t, "info", f"{h.id} {msg[0].lower() + msg[1:]}", h.id, at=(rover.x, rover.y), source=src)
        else:
            self._log(t, "info", msg, rover.id, at=(rover.x, rover.y), source=src)
        self._say(h.id if h is not None else "fleet", t, msg, "ack" if by == "operator" else "info", reply_to=reply_to)
        self._finish(rec, t, "leave", None)

    def _mark_wreck(self, rover, t: float) -> None:
        rover.state = "wrecked"
        rover.goal, rover.path = None, []
        for w in [w for w in self.wrecks if w["rover"] == rover.id]:
            self.wrecks.remove(w)
        self.wrecks.append({"rover": rover.id, "x": round(rover.x, 2), "y": round(rover.y, 2)})
        self.quiet_ok[rover.id] = 1e12           # a wreck's silence is expected

    def _finish(self, rec, t: float, plan, msg: str | None) -> None:
        h = self._rover(rec["helper"]) if rec["helper"] else None
        rec["phase"] = "done"
        rec["plan"] = plan if plan is not None else rec.get("plan")
        rec["done_t"] = t
        adv = getattr(self.sim, "advisor", None)
        if rec.get("qid") is not None and adv is not None and rec["qid"] in adv.requests:
            adv._close(rec["qid"], "decided", t)
        if msg:
            self._log(t, "info", msg, rec["helper"] or rec["rover"], at=rec["at"])
        if h is not None and h.state in ("rescuing", "towing"):
            h.state = "exploring"
            h.goal, h.path = None, []
            h.next_goal_try = t
            h._replan_now(t)
            if h._need_return(t):
                h._go_home(t, "battery")

    def _salvaged(self, rec, rover, h, t: float) -> None:
        taken = []
        for p in rec["opts"]["salvage"]["parts"]:
            cond = rover.health.strip(p)
            kind = SPARE_OF[p]
            mine = [q for q in PARTS if SPARE_OF.get(q) == kind and h.health.value(q) < SALVAGE_MIN]
            if mine:                               # the helper needs one itself: fitted on the spot (used now)
                q = min(mine, key=lambda q_: h.health.value(q_))
                before = h.health.value(q)
                h.health.install(q, cond)
                taken.append(f"{PART_NAME[p]} {cond:.0f}% (fitted to its own {PART_NAME[q]}, was {before:.0f}%)")
            else:
                h.carried_spares.append((kind, cond, rover.id))
                taken.append(f"{PART_NAME[p]} {cond:.0f}%")
        tubes = 0
        while rover.samples and h.free_tubes() > 0:
            k = rover.samples.pop(0)
            h.samples.append(k)
            self.fleet.targets[k].carried_by = h.id
            tubes += 1
        extra = f" and {tubes} sample tube{'s' if tubes != 1 else ''}" if tubes else ""
        self._mark_wreck(rover, t)
        msg = (f"{h.id} salvaged {rover.id}: took {', '.join(taken) or 'nothing usable'}{extra}. Spares count once "
               f"delivered to the lander. The wreck stays at ({rover.x:.0f}, {rover.y:.0f}).")
        self._log(t, "info", msg, h.id, at=(rover.x, rover.y))
        self._say(h.id, t, msg.split(" ", 1)[1][0].upper() + msg.split(" ", 1)[1][1:], "info")
        self._finish(rec, t, "salvage", None)

    def deliver(self, r, t: float) -> None:
        """r docked at the lander: salvaged parts it carries become fleet spares."""
        if not r.carried_spares:
            return
        kinds = [k for k, _c, _w in r.carried_spares]
        for k in kinds:
            self.fleet.spares[k] = self.fleet.spares.get(k, 0) + 1
        r.carried_spares = []
        stock = ", ".join(f"{k} {self.fleet.spares.get(k, 0)}" for k in SPARE_KINDS)
        self._log(t, "info", f"{r.id} delivered {len(kinds)} salvaged part{'s' if len(kinds) != 1 else ''} "
                             f"({', '.join(kinds)}) to the lander. Spares: {stock}.", r.id, at=r.dock)

    # ------------------------------------------------------------------ towing
    def _hitch_points(self, h, w):
        d = self.sim.data
        from .world import HITCH_ARM
        cx = self.cfg.chassis_half[0] + HITCH_ARM
        Rh = d.xmat[h.body].reshape(3, 3)
        Rw = d.xmat[w.body].reshape(3, 3)
        ph = d.xpos[h.body] + Rh @ np.array([-cx, 0.0, 0.0])
        pw = d.xpos[w.body] + Rw @ np.array([cx, 0.0, 0.0])
        return ph, pw

    def _tow_drive(self, rec, w, h, t: float) -> bool:
        hc = rec["hitch"]
        if hc is None:
            return False
        stage = hc["stage"]
        from .world import HITCH_ARM
        gap = 2.0 * (self.cfg.chassis_half[0] + HITCH_ARM)        # centre to centre when hitched
        if stage == "approach":
            # a point ahead of the disabled rover (its front hitch), facing the same way, then back up to it
            wy = w.yaw
            pre = gap + HITCH_PRE_M
            px, py = w.x + pre * math.cos(wy), w.y + pre * math.sin(wy)
            if t - hc["t0"] > HITCH_TIMEOUT_S or not (h._safe_cell(px, py) or _d(h.x, h.y, px, py) < 0.8):
                self._go_kinematic(rec, w, h, t, "no room to back up to its hitch" if t - hc["t0"] <= HITCH_TIMEOUT_S
                                   else "the hitch took too long")
                return True
            if _d(h.x, h.y, px, py) < 0.6:
                hc["stage"] = "align"
                h.goal, h.path = None, []
                return True
            rec["goal"] = (px, py)
            return self._goto(h, rec, (px, py), t)
        if stage == "align":
            e = (w.yaw - h.yaw + math.pi) % (2 * math.pi) - math.pi
            if abs(e) < 0.1:
                hc["stage"] = "back"
                hc["best"], hc["best_t"] = 1e9, t
                return True
            h.cmd_v, h.cmd_w = 0.0, max(-0.5, min(0.5, 1.6 * e))
            if t - hc["t0"] > HITCH_TIMEOUT_S:
                self._go_kinematic(rec, w, h, t, "the hitch took too long")
            return True
        if stage == "back":
            ph, pw = self._hitch_points(h, w)
            dist = float(np.hypot(ph[0] - pw[0], ph[1] - pw[1]))
            if dist < hc["best"] - 0.02:
                hc["best"], hc["best_t"] = dist, t
            if dist <= HITCH_OK_M:
                self._hitch(rec, w, h, t)
                return True
            if t - hc["best_t"] > 4.0 or t - hc["t0"] > HITCH_TIMEOUT_S:
                if dist < 0.45:
                    self._hitch(rec, w, h, t)        # close enough: the soft constraint pulls the bar in
                else:
                    self._go_kinematic(rec, w, h, t, f"could not back up to its hitch ({dist:.2f} m short)")
                return True
            # steer the rear hitch onto the towed rover's front hitch while reversing
            want = math.atan2(ph[1] - pw[1], ph[0] - pw[0])     # helper heading that points its rear at pw
            e = (want - h.yaw + math.pi) % (2 * math.pi) - math.pi
            h.cmd_v = -0.12 if dist > 0.4 else -0.07
            h.cmd_w = max(-0.4, min(0.4, 1.2 * e))
            return True
        # hitched: drive home slowly
        dock = h.dock
        if _d(w.x, w.y, 0.0, 0.0) <= REPAIR_RADIUS - 0.8 and _d(h.x, h.y, dock[0], dock[1]) < 2.6:
            self._arrived_home(rec, w, h, t)
            return True
        rec["goal"] = dock
        return self._goto(h, rec, dock, t, limit=self.tow_limit(h))

    def _hitch(self, rec, w, h, t: float) -> None:
        m, d = self.sim.model, self.sim.data
        eq = self.sim.world.tow_eq.get((h.id, w.id))
        if eq is None:
            self._go_kinematic(rec, w, h, t, "no hitch for this pair")
            return
        m.eq_solref[eq] = (0.25, 1.0)              # soft: pull the bar in gently ...
        d.eq_active[eq] = 1
        self._tyres(w, True)
        rec["hitch"].update({"stage": "tow", "eq": eq, "t_hitch": t, "stiff": False})
        rec["progress"] = (t, h.x, h.y, math.hypot(h.x, h.y))
        w.state = "towed"
        self._log(t, "info", f"{h.id} hitched {w.id}. Towing it home at {TOW_V:.1f} m/s (slope limit "
                             f"{self.tow_limit(h):.0f} deg).", h.id, at=(h.x, h.y))
        h.plan_t = -1e9
        h.path = []

    def _tow_watch(self, rec, w, h, t: float) -> None:
        hc = rec["hitch"]
        if hc is None or hc.get("stage") != "tow":
            return
        m, d = self.sim.model, self.sim.data
        if hc.get("eq") is not None and not hc["stiff"] and t - hc["t_hitch"] > 1.5:
            m.eq_solref[hc["eq"]] = (0.03, 1.0)    # ... then a stiff tow bar
            hc["stiff"] = True
        if h.battery < 0.04:
            self._unhitch(h, w, t)
            self._log(t, "warn", f"{h.id} battery flat while towing {w.id}. Unhitched; both stop here.", h.id,
                      at=(h.x, h.y))
            w.state = "disabled"
            h.state = "idle"
            self._release_helper(rec, t, "battery flat")
            return
        if not hc.get("kinematic"):
            ph, pw = self._hitch_points(h, w)
            apart = float(np.linalg.norm(ph - pw))
            if hc["stiff"] and (apart > TOW_APART_M or w.tilt > 45.0):
                why = f"the tow bar is pulling apart ({apart:.2f} m)" if apart > TOW_APART_M else \
                    f"{w.id} is tipping ({w.tilt:.0f} deg)"
                self._go_kinematic(rec, w, h, t, why)
                return
        p = rec["progress"]
        dist_home = math.hypot(h.x, h.y)
        if _d(h.x, h.y, p[1], p[2]) > 1.0 or dist_home < p[3] - 1.0:
            rec["progress"] = (t, h.x, h.y, dist_home)
        elif t - p[0] > TOW_STUCK_S:
            if not hc.get("kinematic"):
                self._go_kinematic(rec, w, h, t, f"no progress for {TOW_STUCK_S:.0f} s")
            elif t - p[0] > 3.0 * TOW_STUCK_S:
                self._unhitch(h, w, t)
                w.state = "disabled"
                self._log(t, "warn", f"{h.id} can't get {w.id} any further. Unhitched at ({w.x:.0f}, {w.y:.0f}).",
                          h.id, at=(w.x, w.y))
                rec["route"] = None
                rec["opts"] = None
                rec["decided"] = False
                rec["plan"] = None
                rec["phase"] = "diagnosing"
                rec["t_phase"] = t
                h.state = "rescuing"

    def _tyres(self, w, towed: bool) -> None:
        """While towed, the rover's tyres slide sideways more easily (loose regolith, free wheels)."""
        m = self.sim.model
        hz = self.sim.hazards
        for k in range(4):
            import mujoco
            g = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, f"{w.id}_wheel{k}")
            base = hz._base_fric.get(g, self.cfg.wheel_friction) if hz is not None else self.cfg.wheel_friction
            m.geom_friction[g][0] = TOW_TYRE_MU if towed else base

    def _go_kinematic(self, rec, w, h, t: float, why: str) -> None:
        """Fallback: the towed rover follows the hitch point like a trailer (collisions off, pose set each substep)."""
        hc = rec["hitch"]
        if hc.get("eq") is not None:
            self.sim.data.eq_active[hc["eq"]] = 0
        m = self.sim.model
        saved = []
        lo, hi = self.sim.world.rover_geom_range[w.id]
        for g in range(lo, hi + 1):
            saved.append((g, int(m.geom_contype[g]), int(m.geom_conaffinity[g])))
            m.geom_contype[g] = 0
            m.geom_conaffinity[g] = 0
        self._geoms[w.id] = saved
        self.kinematic.append((h, w))
        hc.update({"stage": "tow", "kinematic": True, "t_hitch": t, "stiff": True})
        rec["progress"] = (t, h.x, h.y, math.hypot(h.x, h.y))
        w.state = "towed"
        h.state = "towing"
        self._log(t, "info", f"{h.id} tows {w.id} on a rigid bar instead ({why}): it follows the hitch like a "
                             f"trailer.", h.id, at=(h.x, h.y))

    def follow_step(self, d) -> None:
        """Every physics substep while a kinematic tow runs: put the towed rover behind the helper's hitch."""
        from .world import HITCH_ARM, rover_spawn_height
        L = self.cfg.chassis_half[0] + HITCH_ARM
        ter = self.sim.terrain
        z0 = rover_spawn_height(self.cfg)
        for (h, w) in self.kinematic:
            a, b = h.qadr, w.qadr
            qw, qx, qy, qz = d.qpos[a + 3:a + 7]
            yaw = math.atan2(2.0 * (qw * qz + qx * qy), 1.0 - 2.0 * (qy * qy + qz * qz))
            hx = float(d.qpos[a]) - L * math.cos(yaw)
            hy = float(d.qpos[a + 1]) - L * math.sin(yaw)
            tx, ty = float(d.qpos[b]), float(d.qpos[b + 1])
            ang = math.atan2(hy - ty, hx - tx)
            nx_, ny_ = hx - L * math.cos(ang), hy - L * math.sin(ang)
            d.qpos[b:b + 3] = (nx_, ny_, ter.height_at(nx_, ny_) + z0)
            d.qpos[b + 3:b + 7] = (math.cos(ang / 2), 0.0, 0.0, math.sin(ang / 2))
            d.qvel[w.vadr:w.vadr + w.nv] = 0.0

    def _unhitch_pair(self, rid: str, t: float) -> None:
        rec = self.by_helper(rid)
        if rec is not None:
            self._unhitch(self._rover(rid), self._rover(rec["rover"]), t)

    def _unhitch(self, h, w, t: float) -> None:
        d, m = self.sim.data, self.sim.model
        if h is None or w is None:
            return
        eq = self.sim.world.tow_eq.get((h.id, w.id))
        if eq is not None:
            d.eq_active[eq] = 0
        self._tyres(w, False)
        if (h, w) in self.kinematic:
            self.kinematic.remove((h, w))
            for g, ct, ca in self._geoms.pop(w.id, []):
                m.geom_contype[g] = ct
                m.geom_conaffinity[g] = ca
            from .world import rover_spawn_height
            a = w.qadr
            x, y = float(d.qpos[a]), float(d.qpos[a + 1])
            d.qpos[a + 2] = self.sim.terrain.height_at(x, y) + rover_spawn_height(self.cfg) + 0.04
            d.qvel[w.vadr:w.vadr + w.nv] = 0.0
            import mujoco
            mujoco.mj_forward(m, d)

    def _arrived_home(self, rec, w, h, t: float) -> None:
        self._unhitch(h, w, t)
        w.read_pose()
        w.state = "repairing"
        w.repair_t0 = t
        w.repair_note = None
        how = "on the rigid bar" if rec["hitch"].get("kinematic") else "on the tow bar"
        msg = (f"{h.id} towed {w.id} to the lander ({how}, {t - rec['t_phase']:.0f} s). Unhitched: {w.id} is being "
               f"repaired.")
        self._log(t, "info", msg, h.id, at=(w.x, w.y))
        self._say(h.id, t, msg.split(" ", 1)[1][0].upper() + msg.split(" ", 1)[1][1:], "info")
        self._finish(rec, t, "tow", None)
        if h.state == "exploring":
            h._go_home(t, "battery" if h._need_return(t) else "dock")

    # ================================================================== repair at the lander
    def repair_step(self, r, t: float, dt: float) -> None:
        """A rover in state "repairing" (at the lander): charge and service it, with spares for broken parts."""
        r._set_cmd(0.0, 0.0)
        power = self.sim.hazards.charge_factor() if self.sim.hazards is not None else 1.0
        r.battery = min(1.0, r.battery + power * dt / self.cfg.charge_time_s)
        before = {p: r.health.value(p) for p in PARTS}
        ev = r.health.service(dt, self.fleet.spares, REPAIR_RATE)
        for kind, p, sk in ev:
            if kind == "spare":
                stock = ", ".join(f"{k} {self.fleet.spares.get(k, 0)}" for k in SPARE_KINDS)
                self._log(t, "info", f"{r.id} {PART_NAME[p]} replaced with a spare {sk} at the lander (was "
                                     f"{before[p]:.0f}%). Spares left: {stock}.", r.id, at=(r.x, r.y))
            elif getattr(r, "repair_note", None) != (p, sk):
                r.repair_note = (p, sk)
                self._log(t, "warn", f"{r.id} needs a spare {sk} for its {PART_NAME[p]} ({before[p]:.0f}%) and the "
                                     f"fleet has none. Waiting at the lander (salvage one from a wreck).", r.id,
                          at=(r.x, r.y))
        missing = any(k == "missing" for k, _p, _s in ev)
        if missing:
            return                                     # waits (charging) until a spare arrives
        if not r.health.worn():
            took = t - getattr(r, "repair_t0", t)
            self._log(t, "info", f"{r.id} repaired at the lander in {took:.0f} s (health {r.health.hp()}, "
                                 f"{r.health.status()}).", r.id, at=(r.x, r.y))
            r.repair_note = None
            self.fleet.cache_samples(t, r)
            if _d(r.x, r.y, r.dock[0], r.dock[1]) > 1.0:
                r.state = "exploring"
                r._go_home(t, "dock")
            elif r.battery < 0.995:
                r.state = "charging"
            else:
                r.state = "exploring"
                r.goal, r.path = None, []
                r.next_goal_try = t

    # ================================================================== operator orders (advisor.py)
    def order(self, plan: str, target, helper, t: float, reply_to=None) -> tuple:
        """"help", "tow", "salvage", "leave" for rover `target` (an id or None: the rover in trouble), by rover
        `helper` (an id or None: the rules pick). Returns (ok, text) for the chat reply (the caller posts it)."""
        rec = None
        if target is None:
            cands = [x for x in self.rescues if x["phase"] != "done"]
            if not cands:
                wr = [r for r in self.sim.rovers if r.state in RESCUE_STATES]
                if len(wr) == 1:
                    target = wr[0].id
                elif not wr:
                    return False, "No rover needs help right now."
                else:
                    return False, f"Which rover: {' or '.join(r.id for r in wr)}?"
            elif len(cands) == 1:
                rec = cands[0]
                target = rec["rover"]
            else:
                return False, f"Which rover: {' or '.join(x['rover'] for x in cands)}?"
        tr = self._rover(target)
        if tr is None:
            return False, f"There is no rover {target}."
        rec = rec or self.active(target)
        if rec is None:
            if tr.state not in RESCUE_STATES:
                return False, (f"{target} doesn't need help: health {tr.health.hp()} ({tr.health.status()}), "
                               f"{tr.state}.")
            if tr.state == "towed":
                return False, f"{target} is already being towed."
            rec = self._new(target, t, (tr.x, tr.y))
        h = self._rover(helper) if helper else None
        if helper and h is None:
            return False, f"There is no rover {helper}."
        if plan == "leave":
            hh = self._rover(rec["helper"]) if rec["helper"] else None
            if hh is not None and rec["phase"] == "towing":
                self._unhitch(hh, tr, t)
            self._leave(rec, tr, hh, t, "operator", reply_to)
            return True, None
        if h is not None and rec.get("helper") != h.id:
            why = self.can_help(h, rec)
            if why is not None:
                return False, f"{h.id} can't help {target}: {why}."
            self.assign(rec, h, t, operator=True, reply_to=reply_to)
            if plan == "help":
                return True, None
        elif rec.get("helper") is None:
            if self._pick_helper(rec, t, quiet=True) is None:
                return False, (f"No rover can help {target} now: " + "; ".join(
                    f"{r.id} {self.can_help(r, rec)}" for r in self.sim.rovers if r.id != target) + ".")
            if plan == "help":
                return True, None
        elif plan == "help":
            hh = self._rover(rec["helper"])
            far = f", {_d(hh.x, hh.y, *rec['at']):.0f} m to go" if hh is not None and rec["phase"] == "en_route" else ""
            return True, (f"{rec['helper']}: already on it, {rec['phase'].replace('_', ' ')} to {target}{far}. "
                          f"{target} comes first.")
        # tow or salvage
        if rec["phase"] == "diagnosing" and rec.get("opts") is not None and not rec.get("decided"):
            why = self.apply_plan(rec, plan, t, by="operator", reply_to=reply_to)
            if why is not None:
                return False, f"Can't {PLAN_LABEL[plan]}: {why}."
            return True, None
        if rec["phase"] in ("towing", "salvaging") and rec.get("plan") == plan:
            return False, f"Already doing that: {rec['helper']} is {rec['phase']} {target}."
        if rec["phase"] in ("towing", "salvaging"):
            return False, f"{rec['helper']} is already {rec['phase']} {target}; say 'leave it' to stop."
        rec["preset"] = plan
        rec["cid"] = reply_to
        hid = rec["helper"]
        return True, (f"{hid}: will {PLAN_LABEL[plan]} once I've reached {target} and looked it over"
                      f" ({rec['phase'].replace('_', ' ')} now).")
