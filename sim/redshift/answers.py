"""Answers to the operator's questions (PROTOCOL.md version 5): fact sheets and rule-based replies.

A question ("rover 2, what have you explored?") gets a short answer built ONLY from what the rover
itself knows: its state, battery and position, and its mission memory (memory.py). The fleet answers
from the shared map and every rover's memory.

    sheet = rover_sheet(rover)            # or fleet_sheet(sim): a compact dict, built on the sim thread
    text = rule_answer(sheet, "battery")  # templated answer (always available, no model needed)
    lines = fact_lines(sheet, "battery")  # the facts Gemma may use, most relevant first (short clauses)

Gemma (gemma.py, kind "answer") writes the reply from fact_lines only; every number it writes must
appear in those lines, or the rule answer is used instead. Topics: status, exploration, finds,
samples, battery, hazards, lifeforms, water, location, why, unknown (outside what it knows: it says
so plainly).

Building a sheet reads a few dozen fields (well under a millisecond); it happens only when someone
asks, never in the 5 Hz loop.
"""
from __future__ import annotations

import math
import re

from .gemma import TOPICS
from .memory import BANDS, compass
from .terrain import KIND_LABEL
PER_ROVER_TOPICS = ("battery", "location", "why", "status", "history", "health")   # "everyone, report battery": each rover replies
MAX_ANSWER_CHARS = 360
KIND_SHORT = {"mineral": "mineral", "ice": "ice", "vent": "vent", "trace": "organic trace"}
INSTRUMENT = {"drill": "drill", "scoop": "scoop", "sniffer": "gas sniffer", "swab": "swab"}


# ====================================================================== formatting helpers
def _n(v: float) -> str:
    """Whole number with thousands separators: 1904.3 -> '1,904'."""
    return f"{int(round(v)):,}"


def _xy(x: float, y: float) -> str:
    return f"({int(round(x))}, {int(round(y))})"


def _list(items: list, n: int = 4) -> str:
    items = [str(i) for i in items if i]
    if not items:
        return ""
    if len(items) > n:
        return ", ".join(items[:n]) + f" and {len(items) - n} more"
    if len(items) == 1:
        return items[0]
    return ", ".join(items[:-1]) + " and " + items[-1]


_IDLEAD_RE = re.compile(r"^[rtkqwsh]\d+\b")


def _cap(s: str) -> str:
    if not s or _IDLEAD_RE.match(s):          # "r2 is ..." stays lower case
        return s
    return s[:1].upper() + s[1:]


def _shorten(p: str, limit: int) -> str:
    """Cut one over-long sentence at a clause boundary ('; ' or ', ') and mark it with '...'."""
    if len(p) <= limit:
        return p
    cut = p[:max(limit - 3, 1)]               # room for the "..."
    for sep in ("; ", ", "):
        k = cut.rfind(sep)
        if k > limit // 3:
            return cut[:k] + "..."
    return cut.rsplit(" ", 1)[0] + "..."


def _sentences(*parts) -> str:
    out, used = [], 0
    for p in parts:
        p = (p or "").strip()
        if not p:
            continue
        room = MAX_ANSWER_CHARS - used - (1 if out else 0)
        if room < 40:
            break
        p = _shorten(_cap(p), room - 1)
        if p[-1] not in ".!?":
            p += "."
        out.append(p)
        used += len(p) + (1 if len(out) > 1 else 0)
    return " ".join(out)


def _where(x: float, y: float) -> str:
    d = math.hypot(x, y)
    if d < 3.0:
        return "at the lander"
    return f"{d:.0f} m {compass(x, y)} of the lander"


def _areas(sectors: list, bands: list) -> str:
    """'mostly north-east and east of the lander, out past 35 m'."""
    if not sectors:
        return ""
    top = [name for name, share in sectors[:3] if share >= 0.18] or [sectors[0][0]]
    lead = "mostly " if len(top) < len(sectors) else "all around, mostly "
    txt = lead + _list(top[:3]) + " of the lander"
    if bands:
        far = [b for b, share in bands if share >= 0.1]
        last = far[-1] if far else bands[-1][0]
        reach = {"near": f"within {BANDS[0][1]:.0f} m", "middle": f"out to about {BANDS[1][1]:.0f} m",
                 "far": f"out past {BANDS[2][0]:.0f} m"}[last]
        txt += f", {reach}"
    return txt


# ====================================================================== what a rover is doing and why
def _target_bits(rover, tidx):
    ts = rover.fleet.targets[tidx]
    d = math.hypot(ts.spec.x - rover.x, ts.spec.y - rover.y)
    return ts, d, compass(ts.spec.x - rover.x, ts.spec.y - rover.y)


def _rescue_rec(rover):
    resc = getattr(rover.sim, "rescue", None)
    if resc is None:
        return None, None
    return resc.active(rover.id), resc.by_helper(rover.id)


def doing_text(rover) -> str:
    """What the rover is doing right now, first person, without a leading 'I'm'."""
    st, t = rover.state, rover.sim.t
    dd = math.hypot(rover.x - rover.dock[0], rover.y - rover.dock[1])
    mine, helping = _rescue_rec(rover)
    if st == "disabled":                  # protocol v9
        who = (f"; {mine['helper']} is {'on its way' if mine['phase'] == 'en_route' else 'with me'}"
               if mine is not None and mine.get("helper") else "")
        return f"stuck at {_xy(rover.x, rover.y)}: I can't drive ({rover.health.short()}), waiting for help{who}"
    if st == "towed":
        return f"being towed to the lander by {mine['helper'] if mine and mine.get('helper') else 'another rover'}"
    if st == "wrecked":
        return f"out of action, left as a wreck at {_xy(rover.x, rover.y)}"
    if st == "repairing":
        return f"being repaired at the lander (health {rover.health.hp()})"
    if st in ("rescuing", "towing") and helping is not None:
        other = next((r for r in rover.sim.rovers if r.id == helping["rover"]), None)
        d = math.hypot(other.x - rover.x, other.y - rover.y) if other is not None else 0.0
        if st == "towing":
            return f"towing {helping['rover']} to the lander, {math.hypot(rover.x, rover.y):.0f} m to go"
        what = {"en_route": f"driving to help {helping['rover']}, {d:.0f} m away",
                "diagnosing": f"looking {helping['rover']} over to see what is wrong",
                "repairing": f"fixing {helping['rover']} in the field",
                "salvaging": f"taking the working parts off {helping['rover']}"}
        return what.get(helping["phase"], f"helping {helping['rover']}")
    if st == "fault":
        return "standing still: my motor controller is rebooting"
    if st == "charging":
        return f"charging at the lander ({rover.battery * 100:.0f}%)"
    if st == "idle":
        if rover.held:
            return f"holding at {_xy(rover.x, rover.y)} for the operator"
        if rover.battery <= 0.0:
            return "stopped in the field with a flat battery"
        if dd < 1.6:
            return "parked at the lander" + (", exploration is complete" if rover.fleet.complete else "")
        return f"idle at {_xy(rover.x, rover.y)}"
    if st == "recovering":
        return {"sand": "backing out of soft ground along my own tracks",
                "water": "backing out of liquid along my own tracks"}.get(rover.rec_phase,
                                                                          "backing off to try another line")
    if st == "observing" and rover.obs is not None:
        tr = rover.fleet.tracker.get(rover.obs["track"])
        if tr is not None:
            px, py = tr.predict(t)
            d = math.hypot(px - rover.x, py - rover.y)
            return f"watching something that moves, {d:.0f} m {compass(px - rover.x, py - rover.y)}, from a safe distance"
        return "watching something that moves, from a safe distance"
    if st == "investigating" and rover.target_idx is not None:
        ts, d, dirn = _target_bits(rover, rover.target_idx)
        if rover.sampling is not None:
            inst = INSTRUMENT.get(rover.sampling["instrument"], rover.sampling["instrument"])
            return f"sampling {ts.spec.id} ({KIND_LABEL[ts.spec.kind]}) with my {inst}"
        return f"driving to sample {ts.spec.id} ({KIND_LABEL[ts.spec.kind]}), {d:.0f} m {dirn}"
    if st == "returning":
        why = {"battery": "to recharge", "range": "to recharge", "cache": "to cache my samples",
               "complete": "because exploration is complete", "operator": "as the operator asked",
               "gemma": "to recharge early", "regroup": "to find a way around steep ground",
               "repair": "for a repair", "dock": "to my parking spot"}.get(
            rover.return_reason, "")
        return f"returning to the lander {why}, {dd:.0f} m to go".replace("  ", " ")
    # exploring
    extra = ""
    if t < rover.pause_until:
        extra = ", waiting a moment first"
    elif t < rover.crawl_until:
        extra = ", creeping slowly while I wait for your answer"
    if rover.op_goal is not None:
        gx, gy = rover.op_goal
        return f"driving to {_xy(gx, gy)} for the operator, {math.hypot(gx - rover.x, gy - rover.y):.0f} m away{extra}"
    fc = getattr(rover, "focus", None)
    if fc is not None:                    # protocol v7: an area it concentrates on
        left = rover.focus_msg()["ends_in"]
        return f"exploring around {fc['label']} ({left:.0f} s left){extra}"
    if rover.goal is not None:
        gx, gy = rover.goal
        d = math.hypot(gx - rover.x, gy - rover.y)
        return f"driving to a frontier {d:.0f} m {compass(gx - rover.x, gy - rover.y)}{extra}"
    return "choosing where to go next" + extra


def why_text(rover) -> str:
    """The reason for what it is doing (first person clause)."""
    st, t, cfg = rover.state, rover.sim.t, rover.cfg
    if st in ("disabled", "towed", "wrecked", "repairing"):      # protocol v9
        return {"disabled": f"my self test says: {rover.health.diagnosis()}",
                "towed": "I can't drive, so another rover is taking me home for a repair",
                "wrecked": "I was damaged beyond field repair and the fleet carries on without me",
                "repairing": "the lander replaces broken parts with spares and services the rest"}[st]
    if st in ("rescuing", "towing"):
        _m, helping = _rescue_rec(rover)
        if helping is not None:
            return (f"{helping['rover']} can't drive and I was the best placed rover with enough battery to help"
                    if st == "rescuing" else f"{helping['rover']} can't drive and the lander can repair it")
    if rover.state in ("exploring", "returning") and rover.return_reason == "repair" and st == "returning":
        return f"my health is {rover.health.hp()} ({rover.health.short()}), so I'm going home for a repair"
    need = rover.need_home * 100.0
    margin = rover.battery_margin * 100.0
    if st == "fault":
        return "my motor controller stopped responding; it reboots by itself and I'll carry on after that"
    if st == "charging":
        return f"my battery needed charging; I'll head out again when it is full (now {rover.battery * 100:.0f}%)"
    if st == "idle":
        if rover.held:
            return "the operator told me to hold; say resume and I'll carry on"
        if rover.fleet.complete:
            return "no reachable frontier is left, so exploration is complete"
        return "I'm waiting for enough charge or a reachable frontier"
    if st == "recovering":
        if rover.rec_phase == "sand":
            return "my wheels spun in soft ground, so I'm backing out along my own tracks to plan around it"
        if rover.rec_phase == "water":
            return "my belly water sensor got wet, so I'm backing out along my own tracks"
        return "I stopped making progress, so I'm backing off to try another line"
    if st == "observing":
        return (f"something moved on its own; I'm watching it from at least {cfg.observe_min_dist:.0f} m "
                f"(planetary protection) for at most {cfg.observe_max_s:.0f} s")
    if st == "investigating" and rover.target_idx is not None:
        ts = rover.fleet.targets[rover.target_idx]
        return (f"the spectrometer flagged {ts.spec.id} as a possible {KIND_LABEL[ts.spec.kind]} with an estimated "
                f"value of {ts.spec.est_value}/10, and I have {rover.free_tubes()} free sample tubes")
    if st == "returning":
        r = rover.return_reason
        if r in ("battery", "range"):
            return (f"my battery is {rover.battery * 100:.0f}% and I need about {need:.0f}% to get home plus a "
                    f"{margin:.0f}% margin")
        if r == "cache":
            return f"my sample tubes are full ({len(rover.samples)} of {rover.capacity}), so I'm taking them to the lander cache"
        if r == "complete":
            return "no reachable frontier is left"
        if r == "operator":
            return "the operator told me to return"
        if r == "gemma":
            return "Gemma advised recharging early"
        if r == "regroup":
            return "steep ground boxed me in, and the way to new frontiers runs back past the lander"
        return "I'm heading home"
    if rover.op_goal is not None:
        return "the operator sent me there; I'll explore again on arrival"
    fc = getattr(rover, "focus", None)
    if fc is not None:
        if fc["reason"] == "operator":
            return f"the operator asked me to explore around {fc['label']}"
        return f"I found {fc.get('what') or 'something new'} there, so I'm looking around it for more"
    b = rover.dir_bias
    if b is not None and t < b[2]:
        dirn = compass(b[0], b[1])
        return f"the operator asked me to explore {dirn} (for {b[2] - t:.0f} s more)"
    note = rover.mem.goal_note
    if note is not None and note[1] == "gemma" and rover.goal is not None and rover.goal == rover.sticky_goal:
        return f"Gemma picked it: {note[2]}"
    if rover.goal is not None:
        return (f"it was the best frontier for size and distance ({rover.goal_size} cells of unknown ground at its "
                f"edge), with enough battery to get there and back")
    return "I'm looking for the best reachable frontier"


def _problem_now(rover):
    st, t = rover.state, rover.sim.t
    hs = rover.health.status()
    if st in ("disabled", "towed", "wrecked"):                   # protocol v9
        return f"I can't drive: {rover.health.short()}"
    if hs != "ok":
        return f"my health is {rover.health.hp()} ({hs}: {rover.health.short()})"
    if st == "fault":
        return "my motor controller is offline (rebooting)"
    if st == "recovering" and rover.rec_phase == "sand":
        return "I'm backing out of soft ground"
    if st == "recovering" and rover.rec_phase == "water":
        return "I'm backing out of liquid"
    if rover.degraded:
        return f"my LiDAR only reaches {rover.lidar_eff:.0f} m (dust), so I'm driving slowly on mapped ground"
    if rover.battery <= 0.0:
        return "my battery is flat"
    if st == "returning" and rover.return_reason in ("battery", "range"):
        return f"my battery is getting low ({rover.battery * 100:.0f}%), so I'm heading home"
    if t < rover.crawl_until:
        return "I asked the operator a question and I'm creeping slowly until it is answered"
    return None


# ====================================================================== fact sheets
def rover_sheet(rover) -> dict:
    """Compact facts about one rover, from its own state and memory. Call it on the sim thread."""
    sim, f, cfg, mem = rover.sim, rover.fleet, rover.cfg, rover.mem
    t = sim.t
    map_m2 = ((cfg.nx - 1) * cfg.res) ** 2
    st = rover.state
    targets = [{"id": d["id"], "kind": d["kind"], "est": d["est"]} for d in mem.targets]
    samples = [{"id": d["id"], "kind": d["kind"], "value": d["value"], "analysis": d["analysis"],
                "instrument": d["instrument"]} for d in mem.samples]
    charge_in = None
    if st == "charging":
        cf = sim.hazards.charge_factor() if sim.hazards is not None else 1.0
        charge_in = (1.0 - rover.battery) * cfg.charge_time_s / max(cf, 0.05)
    return {
        "speaker": rover.id, "name": rover.name, "t": round(t), "state": st,
        "doing": doing_text(rover), "why": why_text(rover), "problem": _problem_now(rover),
        "x": round(rover.x, 1), "y": round(rover.y, 1), "where": _where(rover.x, rover.y),
        "battery": round(rover.battery * 100.0), "need": round(rover.need_home * 100.0),
        "margin": round(rover.battery_margin * 100.0),
        "return_below": round((rover.need_home + rover.battery_margin) * 100.0),
        "charge_in": None if charge_in is None else round(charge_in),
        "slow_charge": bool(rover.slow_charge),
        "mapped": round(mem.mapped_m2()), "map_m2": round(map_m2),
        "map_pct": round(100.0 * mem.mapped_m2() / map_m2, 1),
        "areas": _areas(mem.sectors(), mem.bands()), "driven": round(rover.odometer),
        "targets": targets, "n_targets": mem.n_targets,
        "samples": samples, "n_samples": mem.n_samples,
        "carrying": [f.targets[k].spec.id for k in rover.samples], "capacity": rover.capacity,
        "cached": mem.cached,
        "hazards": [dict(h) for h in list(mem.hazards)[-4:]], "n_hazards": mem.n_hazards,
        "life": [dict(x) for x in list(mem.life)[-3:]], "n_herds": mem.n_herds, "sightings": mem.sightings,
        "water": [dict(w) for w in mem.water[-3:]],
        "recent": [m for (_t, _lvl, m) in list(mem.events)[-3:]],
        "lidar": round(float(rover.lidar_eff), 1), "slope_limit": round(f.slope_limit),
        "health": _health_bits(rover),                          # protocol v9
    }


def _health_bits(rover) -> dict:
    h = rover.health
    mine, helping = _rescue_rec(rover)
    resc = None
    if mine is not None:
        resc = (f"{mine['helper']} is helping me ({mine['phase'].replace('_', ' ')})" if mine.get("helper")
                else "I've called for help and no rover is free yet")
    elif helping is not None:
        resc = f"I'm helping {helping['rover']} ({helping['phase'].replace('_', ' ')})"
    lh = h.last_hit
    return {"hp": h.hp(), "status": h.status(), "short": h.short(), "diagnosis": h.diagnosis(), "rescue": resc,
            "last_hit": None if lh is None else f"{lh['cause']} at {lh['t']:.0f} s"
            + (f" ({lh['shock_g']:.1f} g)" if lh.get("shock_g") else ""), "state": rover.state}


def fleet_sheet(sim) -> dict:
    """Compact facts about the whole fleet: the shared map and every rover's memory."""
    f, cfg = sim.fleet, sim.cfg
    map_m2 = ((cfg.nx - 1) * cfg.res) ** 2
    mapped = f.map.known_count * cfg.res * cfg.res
    sec = None
    for r in sim.rovers:
        sec = r.mem.sector_cells.copy() if sec is None else sec + r.mem.sector_cells
    from .memory import SECTORS
    sectors, bands = [], []
    if sec is not None and sec.sum() > 0:
        by_sec = sec.reshape(8, 3).sum(axis=1)
        tot = float(by_sec.sum())
        sectors = sorted(((SECTORS[k], by_sec[k] / tot) for k in range(8) if by_sec[k] > 0), key=lambda x: -x[1])
        by_band = sec.reshape(8, 3).sum(axis=0)
        bands = [(BANDS[k][2], by_band[k] / float(by_band.sum())) for k in range(3) if by_band[k] > 0]
    found = [ts for ts in f.targets if ts.status is not None]
    sampled = [ts for ts in found if ts.status == "sampled"]
    hazards = []
    for r in sim.rovers:
        for h in r.mem.hazards:
            hazards.append(dict(h, rover=r.id))
    hazards.sort(key=lambda h: h["t"])
    grouped = []
    for h in hazards:
        g = grouped[-1] if grouped else None
        if g is not None and g["kind"] == h["kind"] and h["t"] - g["t"] < 15.0 and h["rover"] not in g["rover"]:
            g["rover"] = g["rover"] + " and " + h["rover"] if " and " not in g["rover"] else \
                g["rover"].replace(" and ", ", ") + " and " + h["rover"]
            continue
        grouped.append(dict(h))
    hazards = grouped
    lives = []
    for r in sim.rovers:
        for x in r.mem.life:
            lives.append(dict(x, rover=r.id))
    lives.sort(key=lambda x: x["t"])
    return {
        "speaker": "fleet", "t": round(sim.t), "complete": bool(f.complete),
        "explored_pct": round(f.map.explored_pct(), 1), "mapped": round(mapped), "map_m2": round(map_m2),
        "areas": _areas(sectors, bands), "driven": round(sum(r.odometer for r in sim.rovers)),
        "rovers": [{"id": r.id, "state": r.state, "battery": round(r.battery * 100.0), "doing": _third(doing_text(r)),
                    "hp": r.health.hp(), "hstatus": r.health.status(), "short": r.health.short(),
                    "why": _third(why_text(r)), "x": round(r.x, 1), "y": round(r.y, 1), "where": _where(r.x, r.y),
                    "problem": _third(_problem_now(r) or "") or None, "hazards": r.mem.n_hazards}
                   for r in sim.rovers],
        "n_found": len(found), "n_total": len(f.targets),
        "found": [{"id": ts.spec.id, "kind": ts.spec.kind, "status": ts.status, "by": ts.found_by} for ts in found],
        "sampled": [{"id": ts.spec.id, "kind": ts.spec.kind, "value": ts.value, "analysis": ts.analysis,
                     "cached": bool(ts.cached), "carried_by": ts.carried_by} for ts in sampled],
        "cached": int(f.samples_cached), "science": int(f.science_score),
        "carried": int(sum(len(r.samples) for r in sim.rovers)),
        "water": [{"x": round(z[0], 1), "y": round(z[1], 1), "r": round(z[2], 1), "by": z[3]} for z in f.water_zones],
        "lifeforms": int(f.lifeforms_seen), "herds": len(f.lifeform_herds), "life": lives[-3:],
        "hazards": hazards[-5:], "n_hazards": sum(r.mem.n_hazards for r in sim.rovers),
        "slope_limit": round(f.slope_limit), "slope_initial": round(float(cfg.slope_limit_initial_deg)),
        "spares": dict(getattr(f, "spares", {}) or {}), "wrecks": [dict(w) for w in getattr(f, "wrecks", [])],
        "rescues": f.rescues_msg() if hasattr(f, "rescues_msg") else [],
    }


# ====================================================================== clauses (shared by rules, Gemma and the fake)
def _target_list(items: list, n: int = 5) -> str:
    return _list([f"{d['id']} {KIND_SHORT.get(d['kind'], d['kind'])}" for d in items], n)


def _sample_list(items: list, n: int = 3, analysis: bool = True) -> str:
    out = []
    for d in items:
        a = f", {d['analysis'][:1].lower() + d['analysis'][1:]}" if analysis and d.get("analysis") else ""
        out.append(f"{d['id']} {KIND_SHORT.get(d['kind'], d['kind'])} (value {d['value']}/10{a})")
    return _list(out, n)


def _third(s: str) -> str:
    """First person memory text in the third person, for the fleet's answers."""
    s = re.sub(r"\bmyself\b", "itself", s)
    s = re.sub(r"\bmy\b", "its", s)
    s = re.sub(r"\bI am\b|\bI'm\b", "it is", s)
    s = re.sub(r"\bI'll\b", "it will", s)
    s = re.sub(r"\bI've\b", "it has", s)
    s = re.sub(r"\bI was\b", "it was", s)
    s = re.sub(r"\bI have\b", "it has", s)
    s = re.sub(r"\bI\b", "it", s)
    return s.replace("your answer", "the operator's answer")


def _water_groups(zones: list) -> list:
    """Water zones of one lake (extended piece by piece as rovers drive along the shore) as one:
    [(x, y, size across in m, how many zones)]."""
    groups = []
    for z in zones:
        for g in groups:
            if math.hypot(z["x"] - g["x"], z["y"] - g["y"]) < g["r"] + z["r"] + 4.0:
                n = g["n"]
                g["x"], g["y"] = (g["x"] * n + z["x"]) / (n + 1), (g["y"] * n + z["y"]) / (n + 1)
                g["r"] = max(g["r"], z["r"], 0.5 * math.hypot(z["x"] - g["x"], z["y"] - g["y"]) + z["r"])
                g["n"] = n + 1
                break
        else:
            groups.append({"x": z["x"], "y": z["y"], "r": z["r"], "n": 1})
    return [(g["x"], g["y"], 2.0 * g["r"], g["n"]) for g in groups]


def _hazard_clause(h: dict, who: bool = False) -> str:
    out = f"{h['what']} at {h['t']:.0f} s: {h['handled']}"
    if h.get("outcome"):
        out += f", {h['outcome']}"
    if who:
        out = f"{h['rover']} met " + _third(out)
    return out


def _life_clause(x: dict, who: bool = False) -> str:
    what = "a herd" if x["n"] > 1 else "a creature"
    if x.get("label"):
        what += f" ({x['label']})"
    s = (f"watched {what} for {x['dur']:.0f} s at {_xy(*x['at'])}: {x['n']} individual{'s' if x['n'] != 1 else ''}, "
         f"average speed {x['speed']:.1f} m/s")
    if not x["life"]:
        s += " (not confirmed as alive)"
    return (f"{x['rover']} " + s) if who else s


def fact_lines(sheet: dict, topic: str) -> list:
    """The facts Gemma may use for this topic: short first person clauses (a rover) or 'we' clauses
    (the fleet), the most relevant first. Every number Gemma writes must come from these."""
    fleet = sheet["speaker"] == "fleet"
    c = _fleet_clauses(sheet) if fleet else _rover_clauses(sheet)
    order = {
        "status": ["doing", "battery", "problem", "where", "progress", "recent"],
        "history": ["progress", "targets", "samples", "hazards", "life", "water", "doing"],
        "exploration": ["mapped", "driven", "where"],
        "finds": ["targets", "samples", "life", "water"],
        "samples": ["samples", "tubes", "targets"],
        "battery": ["battery", "charge", "doing"],
        "hazards": (["hazards", "problem", "slope"] if sheet.get("n_hazards") else ["problem", "hazards", "slope"]),
        "lifeforms": ["life"],
        "water": ["water"],
        "location": ["where", "doing"],
        "why": ["doing", "why", "battery"],
        "unknown": ["doing", "battery", "where", "mapped", "targets", "samples", "hazards", "life", "water"],
        "health": ["health", "diagnosis", "rescue", "last_hit", "spares", "wrecks"],
    }.get(topic, ["doing", "battery", "where"])
    out = []
    for key in order:
        v = c.get(key)
        if v:
            out.append(v)
    return out[:7]


def _rover_clauses(s: dict) -> dict:
    c = {"doing": f"I'm {s['doing']}", "why": f"why: {s['why']}",
         "where": f"I'm at {_xy(s['x'], s['y'])}, {s['where']}"}
    if s["state"] == "charging":
        c["battery"] = f"my battery is at {s['battery']}% and charging"
        if s.get("charge_in") is not None:
            c["charge"] = f"I'll be full in about {s['charge_in']} s" + (
                " (the lander's panels are dusty, charging is slow)" if s.get("slow_charge") else "")
    else:
        c["battery"] = (f"my battery is at {s['battery']}%; I head home below about {s['return_below']}% "
                        f"(about {s['need']}% to get back plus a {s['margin']}% margin)")
    if s.get("problem"):
        c["problem"] = f"right now {s['problem']}"
    elif s["n_hazards"] == 0:
        c["problem"] = "no problems so far"
    else:
        c["problem"] = "no problems right now"
    c["mapped"] = (f"I've mapped about {_n(s['mapped'])} m2 ({s['map_pct']:g}% of the {_n(s['map_m2'])} m2 map)"
                   + (f", {s['areas']}" if s.get("areas") else ""))
    c["driven"] = f"I've driven {_n(s['driven'])} m"
    c["progress"] = f"I've mapped about {_n(s['mapped'])} m2 and driven {_n(s['driven'])} m"
    if s["n_targets"]:
        c["targets"] = f"I found {s['n_targets']} target{'s' if s['n_targets'] != 1 else ''}: {_target_list(s['targets'])}"
    else:
        c["targets"] = "I haven't found any science targets yet"
    if s["n_samples"]:
        c["samples"] = f"I sampled {s['n_samples']}: {_sample_list(s['samples'])}"
    else:
        c["samples"] = "I haven't taken any samples yet"
    used = len(s["carrying"])
    c["tubes"] = (f"{used} of my {s['capacity']} sample tubes are full"
                  + (f" ({_list(s['carrying'])})" if used else "")
                  + (f"; I've delivered {s['cached']} to the lander cache" if s["cached"] else ""))
    if s["n_hazards"]:
        hz = "; ".join(_hazard_clause(h) for h in s["hazards"][-3:])
        c["hazards"] = f"I met {s['n_hazards']} hazard{'s' if s['n_hazards'] != 1 else ''}, latest: {hz}"
    else:
        c["hazards"] = "I haven't met any hazards"
    if s["life"]:
        n = s["n_herds"]
        c["life"] = ((f"I confirmed {n} herd{'s' if n != 1 else ''} of lifeforms; " if n else "")
                     + "I " + "; I ".join(_life_clause(x) for x in s["life"][-2:]))
    elif s["sightings"]:
        c["life"] = f"I saw something moving {s['sightings']} time{'s' if s['sightings'] != 1 else ''} but have not finished observing it"
    else:
        c["life"] = "I haven't seen anything move on its own (no lifeforms seen)"
    groups = _water_groups(s["water"])
    if groups:
        c["water"] = "I identified water " + _list([f"about {d:.0f} m across near {_xy(x, y)}"
                                                     for x, y, d, _k in groups]) + "; every rover now routes around it"
    else:
        c["water"] = "I haven't found any water"
    if s.get("recent"):
        c["recent"] = "latest log: " + s["recent"][-1]
    c["slope"] = f"the fleet's learned slope limit is {s['slope_limit']} deg"
    hb = s.get("health")
    if hb:                                                       # protocol v9
        c["health"] = (f"my health is {hb['hp']} ({hb['status']})" + (f": {hb['short']}" if hb["status"] != "ok"
                                                                       else ", all parts normal"))
        if hb["status"] != "ok":
            c["diagnosis"] = f"my self test says: {hb['diagnosis']}"
        if hb.get("rescue"):
            c["rescue"] = hb["rescue"]
        if hb.get("last_hit"):
            c["last_hit"] = f"my last hit: {hb['last_hit']}"
    return c


def _fleet_clauses(s: dict) -> dict:
    rov = s["rovers"]
    c = {"doing": "; ".join(f"{r['id']} is {r['doing']}" for r in rov),
         "why": "; ".join(f"{r['id']}: {r['why']}" for r in rov),
         "battery": "batteries: " + ", ".join(f"{r['id']} {r['battery']}% ({r['state']})" for r in rov),
         "where": "; ".join(f"{r['id']} is at {_xy(r['x'], r['y'])}, {r['where']}" for r in rov)}
    with_p = [r for r in rov if r.get("problem")]
    if with_p and len(with_p) == len(rov) > 1 and len({r["problem"] for r in rov}) == 1:
        probs = [f"every rover: {rov[0]['problem']}"]
    else:
        probs = [f"{r['id']}: {r['problem']}" for r in with_p]
    c["problem"] = ("right now " + "; ".join(probs)) if probs else (
        "no problems so far" if s["n_hazards"] == 0 else "no problems right now")
    c["mapped"] = (f"we've mapped about {_n(s['mapped'])} m2 ({s['explored_pct']:g}% of the {_n(s['map_m2'])} m2 map)"
                   + (f", {s['areas']}" if s.get("areas") else "") + (", and exploration is complete" if s["complete"] else ""))
    c["driven"] = f"together we've driven {_n(s['driven'])} m"
    c["progress"] = f"we've explored {s['explored_pct']:g}% of the map in {s['t']} s"
    if s["n_found"]:
        c["targets"] = f"we found {s['n_found']} of {s['n_total']} targets: {_target_list(s['found'], 6)}"
    else:
        c["targets"] = f"we haven't found any of the {s['n_total']} science targets yet"
    if s["sampled"]:
        c["samples"] = (f"we sampled {len(s['sampled'])}: {_sample_list(s['sampled'], 3, analysis=False)}; "
                        f"{s['cached']} in the lander cache, {s['carried']} still carried, science score {s['science']}")
    else:
        c["samples"] = "we haven't taken any samples yet"
    c["tubes"] = f"{s['cached']} samples in the lander cache and {s['carried']} carried by rovers"
    if s["n_hazards"]:
        c["hazards"] = (f"we met {s['n_hazards']} hazard{'s' if s['n_hazards'] != 1 else ''}, latest: "
                        + "; ".join(_hazard_clause(h, who=True) for h in s["hazards"][-3:]))
    else:
        c["hazards"] = "we haven't met any hazards"
    if s["lifeforms"]:
        c["life"] = (f"we confirmed {s['lifeforms']} lifeform{'s' if s['lifeforms'] != 1 else ''} in {s['herds']} "
                     f"herd{'s' if s['herds'] != 1 else ''}"
                     + (": " + "; ".join(_life_clause(x, who=True) for x in s["life"][-2:]) if s["life"] else ""))
    else:
        c["life"] = "we haven't confirmed any lifeforms yet"
    groups = _water_groups(s["water"])
    if groups:
        n = len(groups)
        c["water"] = (f"we identified {n} bod{'ies' if n != 1 else 'y'} of water, " + _list(
            [f"about {d:.0f} m across near {_xy(x, y)}" for x, y, d, _k in groups], 3)
            + f"; every rover routes around {'them' if n != 1 else 'it'}")
    else:
        c["water"] = "we haven't found any water"
    c["slope"] = (f"our slope limit is {s['slope_limit']} deg (we started at {s['slope_initial']} deg and learned it "
                  f"from stalls)" if s["slope_limit"] != s["slope_initial"] else
                  f"our slope limit is still {s['slope_limit']} deg")
    if rov and "hp" in rov[0]:                                   # protocol v9
        c["health"] = "health: " + "; ".join(f"{r['id']} {r['hp']} ({r['hstatus']}" + (f", {r['short']}" if
                                             r["hstatus"] != "ok" else "") + ")" for r in rov)
    if s.get("rescues"):
        c["rescue"] = "rescues: " + "; ".join(
            f"{x['helper'] or 'nobody yet'} for {x['rover']} ({x['phase'].replace('_', ' ')}"
            + (f", {x['plan']}" if x.get("plan") else "") + ")" for x in s["rescues"])
    if s.get("spares"):
        c["spares"] = "spare parts at the lander: " + ", ".join(f"{k} {v}" for k, v in s["spares"].items())
    if s.get("wrecks"):
        c["wrecks"] = "wrecks: " + ", ".join(f"{w['rover']} at {_xy(w['x'], w['y'])}" for w in s["wrecks"])
    return c


# ====================================================================== rule-based answers
def rule_answer(sheet: dict, topic: str) -> str:
    """A short, natural answer from the fact sheet alone (no model). Always grounded."""
    topic = topic if topic in TOPICS else "unknown"
    if sheet["speaker"] == "fleet":
        return _fleet_answer(sheet, topic)
    return _rover_answer(sheet, topic)


def _rover_answer(s: dict, topic: str) -> str:
    c = _rover_clauses(s)
    if topic == "history":
        parts = [c["progress"] + (f", {s['areas']}" if s.get("areas") else "")]
        finds = c["targets"]
        if s["n_samples"]:
            finds += f", and sampled {s['n_samples']}"
        parts.append(finds)
        extra = []
        if s["n_hazards"]:
            extra.append(f"handled {s['n_hazards']} hazard{'s' if s['n_hazards'] != 1 else ''}")
        if s["n_herds"]:
            extra.append(f"watched {s['n_herds']} herd{'s' if s['n_herds'] != 1 else ''} of lifeforms")
        if s["water"]:
            x, y, _d, _k = _water_groups(s["water"])[0]
            extra.append(f"found water near {_xy(x, y)}")
        if extra:
            parts.append("I also " + _list(extra))
        parts.append("Now " + c["doing"])
        return _sentences(*parts)
    if topic == "status":
        tail = c["problem"] if s.get("problem") else ("all systems normal" if s["n_hazards"] == 0 else
                                                      f"all systems normal now ({s['n_hazards']} hazards handled so far)")
        bat = f"battery {s['battery']}%" + (" and charging" if s["state"] == "charging" else "")
        return _sentences(c["doing"], f"{bat}, {len(s['carrying'])} of {s['capacity']} sample tubes full, {tail}")
    if topic == "exploration":
        return _sentences(c["mapped"], c["driven"])
    if topic == "finds":
        parts = [c["targets"]]
        if s["n_samples"]:
            parts.append(f"I sampled {s['n_samples']}: " + _sample_list(s["samples"], 3, analysis=False))
        extra = []
        if s["n_herds"]:
            extra.append(f"watched {s['n_herds']} herd{'s' if s['n_herds'] != 1 else ''} of lifeforms")
        if s["water"]:
            x, y, _d, _k = _water_groups(s["water"])[0]
            extra.append(f"found water near {_xy(x, y)}")
        if extra:
            parts.append("I also " + _list(extra))
        return _sentences(*parts)
    if topic == "samples":
        if not s["n_samples"]:
            return _sentences(c["samples"], c["tubes"])
        return _sentences(f"I sampled {s['n_samples']}: {_sample_list(s['samples'], 2)}", c["tubes"])
    if topic == "battery":
        if s["state"] == "charging":
            return _sentences(c["battery"], c.get("charge", ""))
        if s["state"] == "returning" and "recharge" in s["doing"]:
            return _sentences(c["battery"], f"I'm {s['doing']}")
        return _sentences(c["battery"])
    if topic == "hazards":
        if not s["n_hazards"]:
            return _sentences(c["problem"] if s.get("problem") else "No problems so far",
                              "I haven't met any hazards" if not s.get("problem") else "")
        last = s["hazards"][-2:]
        lead = f"right now {s['problem']}" if s.get("problem") else ""
        return _sentences(lead, f"I met {s['n_hazards']} hazard{'s' if s['n_hazards'] != 1 else ''} so far, latest: "
                          + "; ".join(_hazard_clause(h) for h in last))
    if topic == "lifeforms":
        if s["life"]:
            x = s["life"][-1]
            return _sentences(("Yes, " if x["life"] else "") + "I " + _life_clause(x),
                              f"{s['n_herds']} confirmed herd{'s' if s['n_herds'] != 1 else ''} so far" if s["n_herds"] > 1 else "")
        return _sentences(c["life"])
    if topic == "water":
        return _sentences(c["water"])
    if topic == "location":
        return _sentences(c["where"], c["doing"])
    if topic == "why":
        return _sentences(c["doing"], f"Why: {s['why']}")
    if topic == "health":                                        # protocol v9
        return _sentences(c.get("health", ""), c.get("diagnosis", ""), c.get("rescue", ""))
    return _sentences("I can't answer that: I only know what my own sensors and memory recorded",
                      "Ask me about my status, exploration, finds, samples, battery, hazards, lifeforms, water, "
                      "position or why I'm going somewhere")


def _fleet_answer(s: dict, topic: str) -> str:
    c = _fleet_clauses(s)
    rov = s["rovers"]
    if topic == "history":
        return _sentences(c["mapped"], c["targets"], c["samples"] if s["sampled"] else "",
                          c["hazards"] if s["n_hazards"] else "", c["life"] if s["lifeforms"] else "")
    if topic == "status":
        brief = "; ".join(f"{r['id']} {r['state']} ({r['battery']}%)" for r in rov)
        tail = c["problem"] if any(r.get("problem") for r in rov) else ""
        return _sentences(c["progress"], brief, tail)
    if topic == "exploration":
        return _sentences(c["mapped"], c["driven"])
    if topic == "finds":
        extra = []
        if s["lifeforms"]:
            extra.append(f"confirmed {s['lifeforms']} lifeform{'s' if s['lifeforms'] != 1 else ''}")
        if s["water"]:
            n = len(_water_groups(s["water"]))
            extra.append(f"found {n} bod{'ies' if n != 1 else 'y'} of water")
        samp = (f"sampled {len(s['sampled'])} and cached {s['cached']} (science score {s['science']})"
                if s["sampled"] else "")
        return _sentences(c["targets"], ("We " + samp) if samp else "", ("We also " + _list(extra)) if extra else "")
    if topic == "samples":
        return _sentences(c["samples"])
    if topic == "battery":
        return _sentences(c["battery"])
    if topic == "hazards":
        if not s["n_hazards"]:
            return _sentences(c["problem"], "we haven't met any hazards" if c["problem"].startswith("no") else "")
        last = s["hazards"][-2:]
        lead = c["problem"] if c["problem"].startswith("right now") else ""
        return _sentences(lead, f"we met {s['n_hazards']} hazard{'s' if s['n_hazards'] != 1 else ''}, latest: "
                          + "; ".join(_hazard_clause(h, who=True) for h in last))
    if topic == "lifeforms":
        return _sentences(c["life"])
    if topic == "water":
        return _sentences(c["water"])
    if topic == "location":
        return _sentences(c["where"])
    if topic == "health":                                        # protocol v9
        return _sentences(c.get("health", ""), c.get("rescue", ""), c.get("spares", ""), c.get("wrecks", ""))
    if topic == "why":
        return _sentences("Each rover picks its own goal", *(
            f"{r['id']} is {r['doing']}: {re.split(r' [(]|; |, with', r['why'])[0]}" for r in rov))
    return _sentences("I can't answer that from what the fleet has sensed",
                      "Ask about status, exploration, finds, samples, battery, hazards, lifeforms, water, positions "
                      "or why a rover is going somewhere")


# ====================================================================== area questions (protocol v8)
AREA_R = 15.0          # "what did we find there?": what the fleet knows within about this far of the place


def area_facts(sim, x: float, y: float, label: str, r: float | None = None, kind: str | None = None) -> list:
    """Fact lines about one place only ("what did we find there?", "anything near the lake?", "what's at the
    vent?"): targets and samples, water, spires, herds and hazards within about r of it, and what a rover explored
    or re-surveyed there, from fleet knowledge. Neutral wording, so a rover and the fleet can both use them.
    kind: the place's kind (for the caller's records; water at the place itself is described as "there")."""
    from .places import water_bodies
    f = sim.fleet
    rad = float(max(AREA_R, r or 0.0))
    where = label if "(" in label else f"{label} at {_xy(x, y)}"
    near = [ts for ts in f.targets if ts.status is not None and math.hypot(ts.spec.x - x, ts.spec.y - y) <= rad]
    near.sort(key=lambda ts: math.hypot(ts.spec.x - x, ts.spec.y - y))
    bits = []
    for ts in near[:5]:
        if ts.status == "sampled":
            st = f"sampled by {ts.carried_by or 'the fleet'}" if not ts.cached else "sampled and cached"
            st += f", value {ts.value}/10" if ts.value is not None else ""
        elif ts.status == "investigating":
            st = f"{ts.investigator} is sampling it"
        else:
            st = f"found by {ts.found_by}, not sampled yet"
        bits.append(f"{ts.spec.id} {KIND_SHORT.get(ts.spec.kind, ts.spec.kind)} ({st})")
    lines = [(f"targets known within {rad:.0f} m of {where}: " + _list(bits, 5)) if bits else
             f"no science targets known within {rad:.0f} m of {where}"]
    for wb in water_bodies(f.water_zones):
        dd = math.hypot(wb.x - x, wb.y - y)
        if dd <= rad + 0.5 * (wb.size or 0.0):
            if dd < 0.5 * (wb.size or 0.0) + 4.0:
                lines.append(f"the {wb.sub} there is about {wb.size:.0f} m across")
            else:
                lines.append(f"a {wb.sub} about {wb.size:.0f} m across, its middle {dd:.0f} m from it")
            break
    spires = [k for k, v in f.spires_seen.items() if math.hypot(v[2] - x, v[3] - y) <= rad]
    if spires:
        lines.append(f"{len(spires)} spire{'s' if len(spires) != 1 else ''} stand{'s' if len(spires) == 1 else ''} "
                     f"within {rad:.0f} m of it")
    herd = None
    for r_ in sim.rovers:
        for e in r_.mem.life:
            if e.get("at") and math.hypot(e["at"][0] - x, e["at"][1] - y) <= rad + 5.0:
                herd = (r_.id, e)
    if herd is not None:
        rid, e = herd
        what = f"a herd of {e['n']}" if e["n"] > 1 else "a creature"
        lines.append(f"{rid} watched {what} {math.hypot(e['at'][0] - x, e['at'][1] - y):.0f} m from it for "
                     f"{e['dur']:.0f} s")
    else:
        for hx, hy, n in f.lifeform_herds:
            dd = math.hypot(hx - x, hy - y)
            if dd <= rad + 5.0:
                lines.append(f"{'a herd of ' + str(n) if n > 1 else 'a creature'} was seen about {dd:.0f} m from it")
                break
    hz = []
    for z in f.soft_zones:
        dd = math.hypot(z[0] - x, z[1] - y)
        if dd <= rad:
            hz.append(f"soft ground found by {z[3]} {dd:.0f} m from it")
    for r_ in sim.rovers:
        for h in r_.mem.hazards:
            if h["kind"] in ("rocks", "spire", "dropoff") and h.get("at") and \
                    math.hypot(h["at"][0] - x, h["at"][1] - y) <= rad:
                what = {"rocks": "new rocks", "spire": "a fallen spire", "dropoff": "a drop-off"}[h["kind"]]
                hz.append(f"{what} met by {r_.id} {math.hypot(h['at'][0] - x, h['at'][1] - y):.0f} m from it")
    if hz:
        lines.append("hazards there: " + _list(list(dict.fromkeys(hz)), 3))
    for r_ in sim.rovers:
        s = r_.mem.focus_near(x, y, max(12.0, rad - 3.0))
        if s is None or not s.get("there"):
            continue
        if s.get("mode") == "resurvey":
            ch = s.get("changes", 0)
            new = s.get("new_found") or []
            what = (f"{ch} map samples changed" if ch else "no changes since it was mapped") + (
                f", newly found {_list(new)}" if new else "")
            lines.append(f"{r_.id} re-surveyed it for {s['dur']:.0f} s ({s.get('driven', 0)} m driven): {what}")
        else:
            lines.append(f"{r_.id} explored around it for {s['dur']:.0f} s and mapped {s['mapped']} m2 of new ground")
        break
    kn, unk = f.map.area_cells(x, y, rad)
    lines.append(f"the ground within {rad:.0f} m of it is {round(100.0 * kn / max(kn + unk, 1))}% mapped")
    return lines[:7]


def area_answer(lines: list) -> str:
    """The rule-based answer from area_facts lines."""
    return _sentences(*lines)
