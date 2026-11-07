"""Empirical rover check: drive up planar heightfield ramps of known angle, and turn in place.

Run from sim/:  python scripts/climb_test.py [--torque 7.2] [--mu 0.8] [--g 2.0] [--wy 0.40]
Prints, per ramp angle, how far up the ramp the rover got in 12 s. With the defaults (2 g) the
rover climbs 20 deg at full speed, stalls at 21 deg and slides back at 22 deg.
"""
from __future__ import annotations

import argparse
import math
import os
import sys
from dataclasses import replace

import mujoco
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from redshift.config import Config  # noqa: E402
from redshift.world import physics_options_xml, rover_actuator_xml, rover_body_xml, rover_spawn_height  # noqa: E402


def ramp_model(cfg: Config, angle_deg: float, yaw: float = 0.0, start_x: float = -4.0, on_ramp: bool = False):
    res = 0.25
    nx, ny = 121, 41  # x from -10 to 20, y from -5 to 5
    xs = -10 + np.arange(nx) * res
    t = math.tan(math.radians(angle_deg))
    ramp_start = 0.0 if not on_ramp else -10.0
    h = np.where(xs > ramp_start, (xs - ramp_start) * t, 0.0)
    H = np.tile(h, (ny, 1))
    lo, hi = H.min() - 0.01, H.max() + 0.01
    rz = hi - lo
    hx, hy = (nx - 1) * res / 2, (ny - 1) * res / 2
    sx = start_x
    gz = (sx - ramp_start) * t if sx > ramp_start else 0.0
    z0 = rover_spawn_height(cfg) + 0.02
    pitch = math.atan(t) if on_ramp else 0.0
    body = rover_body_xml(cfg, "r1", sx, 0.0, gz + z0 / math.cos(pitch), yaw, "#ff7a45")
    if on_ramp:
        # orient the rover with the slope: yaw then pitch (nose up along +x when yaw=0)
        cy, sy = math.cos(yaw / 2), math.sin(yaw / 2)
        qy = np.array([cy, 0, 0, sy])
        # slope rotation about world y axis by -pitch (nose up for +x)
        qp = np.array([math.cos(-pitch / 2), 0, math.sin(-pitch / 2), 0])
        q = np.zeros(4)
        mujoco.mju_mulQuat(q, qp, qy)
        old = body.split('quat="')[1].split('"')[0]
        body = body.replace(f'quat="{old}"', f'quat="{q[0]:.6f} {q[1]:.6f} {q[2]:.6f} {q[3]:.6f}"', 1)
    xml = f"""
<mujoco>
  <compiler angle="radian" autolimits="true"/>
  {physics_options_xml(cfg)}
  <asset><hfield name="ramp" nrow="{ny}" ncol="{nx}" size="{hx} {hy} {rz} 0.5"/></asset>
  <worldbody>
    <geom type="hfield" hfield="ramp" pos="{-10 + hx} 0 {lo}" friction="{cfg.terrain_friction} 0.005 0.0001"/>
    {body}
  </worldbody>
  <actuator>{rover_actuator_xml(cfg, "r1")}</actuator>
</mujoco>"""
    m = mujoco.MjModel.from_xml_string(xml)
    m.hfield_data[:] = ((H - lo) / rz).astype(np.float32).ravel()
    d = mujoco.MjData(m)
    mujoco.mj_forward(m, d)
    return m, d


def wheel_dofs(m):
    return [int(m.jnt_dofadr[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, f"r1_w{k}")]) for k in range(4)]


def drive(m, d, cfg, v, w, seconds):
    r = cfg.wheel_radius
    wl = (v - w * cfg.track_eff) / r
    wr = (v + w * cfg.track_eff) / r
    d.ctrl[:] = [wl, wl, wr, wr]
    steps = int(seconds / m.opt.timestep)
    energy = 0.0
    dofs = wheel_dofs(m)
    for _ in range(steps):
        mujoco.mj_step(m, d)
        energy += float(np.sum(np.abs(d.actuator_force * d.qvel[dofs]))) * m.opt.timestep
    return energy


def tilt_deg(d):
    R = d.xmat[1].reshape(3, 3)
    return math.degrees(math.acos(max(-1.0, min(1.0, R[2, 2]))))


def yaw_of(d):
    R = d.xmat[1].reshape(3, 3)
    return math.atan2(R[1, 0], R[0, 0])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--torque", type=float, default=None)
    ap.add_argument("--mu", type=float, default=None)
    ap.add_argument("--g", type=float, default=2.0)
    ap.add_argument("--wx", type=float, default=None)
    ap.add_argument("--wy", type=float, default=None)
    ap.add_argument("--cone", default=None)
    ap.add_argument("--impratio", type=float, default=None)
    args = ap.parse_args()
    cfg = Config(gravity_g=args.g)
    if args.torque is not None:
        cfg = replace(cfg, motor_torque=args.torque)
    if args.mu is not None:
        cfg = replace(cfg, wheel_friction=args.mu)
    if args.wx is not None:
        cfg = replace(cfg, wheel_x=args.wx)
    if args.wy is not None:
        cfg = replace(cfg, wheel_y=args.wy)
    if args.cone:
        cfg = replace(cfg, cone=args.cone)
    if args.impratio is not None:
        cfg = replace(cfg, impratio=args.impratio)
    print(f"gravity {cfg.gravity_g} g, torque cap {cfg.motor_torque} N m, wheel mu {cfg.wheel_friction}")

    # flat driving and power
    m, d = ramp_model(cfg, 0.0)
    drive(m, d, cfg, cfg.v_max, 0.0, 2.0)
    x0 = d.qpos[0]
    e = drive(m, d, cfg, cfg.v_max, 0.0, 5.0)
    print(f"flat: speed {(d.qpos[0] - x0) / 5:.2f} m/s, motor power {e / 5:.1f} W, energy/m {e / max(1e-6, d.qpos[0] - x0):.1f} J")

    # turning in place
    for ang in (0.0, 10.0, 15.0, 18.0):
        m, d = ramp_model(cfg, ang, yaw=math.pi / 2, start_x=2.0, on_ramp=True)
        drive(m, d, cfg, 0.0, 0.0, 1.0)
        y0 = yaw_of(d)
        p0 = d.qpos[:2].copy()
        drive(m, d, cfg, 0.0, cfg.w_max, 3.0)
        dy = (yaw_of(d) - y0 + math.pi) % (2 * math.pi) - math.pi
        print(f"turn in place on {ang:4.1f} deg: yaw rate {math.degrees(dy) / 3:6.1f} deg/s, drift {np.linalg.norm(d.qpos[:2] - p0):.2f} m")

    # holding on a slope with brakes (target 0, brake torque = 3x drive)
    for ang in (20, 24, 26, 28, 30, 32):
        m, d = ramp_model(cfg, ang, start_x=1.0, on_ramp=True)
        m.actuator_forcerange[:, 0] = -3 * cfg.motor_torque
        m.actuator_forcerange[:, 1] = 3 * cfg.motor_torque
        drive(m, d, cfg, 0.0, 0.0, 0.5)
        x0 = d.qpos[0]
        drive(m, d, cfg, 0.0, 0.0, 5.0)
        print(f"  hold on {ang} deg with brakes: slid {(x0 - d.qpos[0]):.2f} m in 5 s")
    # climbing
    print("climb (start on ramp, 12 s at full throttle):")
    for ang in (10, 14, 16, 18, 19, 20, 21, 22, 23, 24, 26, 28, 32):
        m, d = ramp_model(cfg, ang, start_x=1.0, on_ramp=True)
        drive(m, d, cfg, 0.0, 0.0, 0.5)
        x0 = d.qpos[0]
        e = drive(m, d, cfg, cfg.v_max, 0.0, 12.0)
        dist = (d.qpos[0] - x0) / math.cos(math.radians(ang))
        wheel = float(np.mean(np.abs(d.qvel[wheel_dofs(m)]))) * cfg.wheel_radius
        print(f"  {ang:4.1f} deg: up-slope {dist:6.2f} m, wheel surface speed {wheel:.2f} m/s, tilt {tilt_deg(d):5.1f}, "
              f"power {e / 12:.0f} W")


if __name__ == "__main__":
    main()
