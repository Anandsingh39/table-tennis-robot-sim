"""Extra credit E0: can an OpenArm with a paddle return a ball over the net?

1. Required paddle state for cooperative rally shots: the ball is hit at the juggling hit centre and
   must land in the middle of the far half (flight times rally_feasibility.flight_times); the incoming
   ball is the mirror-image shot after a bounce on our half. M6's inverse formula gives the paddle
   normal (tilt toward the opponent) and normal speed.
2. Reachability of forward-tilted paddle poses for the juggling mount and a pre-tilted mount.
3. Joint speeds: juggling logs and a tracked rally stroke vs the motor speed limits; torque usage.

Writes results/e0_feasibility.md. Usage: python scripts/rally_feasibility.py
"""

from __future__ import annotations

import copy
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import mujoco
import numpy as np
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tt.aero import AeroParams, rk4_step  # noqa: E402
from tt.config import RESULTS, load_config  # noqa: E402
from tt.contact import table_restitution_from_rule  # noqa: E402
from tt.juggle_law import paddle_command  # noqa: E402


def fly(p, pos, vel, dt, t_max, table=None, stop=None):
    """RK4 flight; table = (x0, x1, z_contact, e_t) adds one vertical table bounce. Returns final state."""
    y = np.r_[pos, vel].astype(float)
    t, bounced = 0.0, False
    while t < t_max - 1e-12:
        h = min(dt, t_max - t)
        y = rk4_step(y, np.zeros(3), p, h)
        t += h
        if table and not bounced and y[5] < 0 and y[2] < table[2] and table[0] <= y[0] <= table[1]:
            y[2], y[5], bounced = table[2], -table[3] * y[5], True
        if stop and stop(y, bounced):
            break
    return y


def required_shots(cfg):
    rc = cfg["rally_feasibility"]
    p = AeroParams.from_config(cfg)
    tc, r, dt = cfg["table"], cfg["ball"]["radius"], rc["dt"]
    x0, L, H = tc["near_edge_x"], tc["length"], tc["top_height"]
    e_t = table_restitution_from_rule(cfg["contact"]["table_rule_drop_height"],
                                      cfg["contact"]["table_rule_rebound_height"], p.k_d)
    hit = np.asarray(cfg["hit"]["center"], float)
    rows = []
    for T in rc["flight_times"]:
        def shoot(start, land):
            return least_squares(lambda v: fly(p, start, v, dt, T)[:3] - np.r_[land, H + r],
                                 np.r_[(land - start[:2]) / T, 3.0]).x
        v_out = shoot(hit, np.array([x0 + 0.75 * L, 0.0]))
        opp = np.array([2 * x0 + L - hit[0], hit[1], hit[2]])
        v_opp = shoot(opp, np.array([x0 + 0.25 * L, 0.0]))
        y_in = fly(p, opp, v_opp, dt, 5.0, table=(x0, x0 + L, H + r, e_t),
                   stop=lambda y, b: b and y[0] <= hit[0])
        n, s = paddle_command(y_in[3:], v_out, cfg["contact"]["paddle_restitution"])
        rows.append(dict(T=T, v_out=v_out, z_in=y_in[2], v_in=y_in[3:], tilt=np.degrees(np.arccos(n[2])), speed=s))
    return rows


def _reach_job(args):
    alpha, tilt = args
    from tt.kinematics import solve_paddle_pose_cfg
    from tt.model import build_model

    cfg = load_config()
    rc = cfg["rally_feasibility"]
    qx, qy, qz, qw = Rotation.from_euler("y", 90 + alpha, degrees=True).as_quat()
    cfg["paddle"]["flange_quat"] = [qw, qx, qy, qz]
    m, ids = build_model(cfg)
    q_nom = np.asarray(cfg["ready_pose"]["q_nominal"], float)
    n = Rotation.from_euler("y", tilt, degrees=True).apply([0.0, 0.0, 1.0])
    seeds = [q_nom + np.asarray(s, float) for s in rc["ik_seed_offsets"]]
    pts = []
    for z in np.arange(*rc["z_range"], rc["step"]):
        for x in np.arange(*rc["x_range"], rc["step"]):
            c = np.array([x, 0.0, z])
            best = -np.inf
            for s0 in seeds:
                q, pp, nn = solve_paddle_pose_cfg(m, ids, cfg, c, n, q0=s0)
                if np.linalg.norm(pp - c) < 1e-3 and np.degrees(np.arccos(np.clip(nn @ n, -1, 1))) < 0.5:
                    best = max(best, np.min(np.minimum(q - ids.q_lo, ids.q_hi - q)))
            if best >= rc["min_joint_margin"]:
                pts.append((x, z))
    return alpha, tilt, pts


def rally_stroke(cfg):
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from tracking_test import run_stroke

    rc = cfg["rally_feasibility"]["stroke"]
    c = copy.deepcopy(cfg)
    c["hit"]["center"] = rc["hit_center"]
    c["planner"]["approach_time"] = rc["approach_time"]
    res, L, ids = run_stroke(c, {"name": "rally", "offset": [0, 0, 0], "tilt_deg": [0.0, rc["tilt_deg"]],
                                 "speed": rc["speed"]})
    before = L["t"] < res["t_hit"]
    frac = np.abs(L["tau"]) / ids.tau_max
    return res, np.abs(L["qd"]).max(axis=0), frac[before].max(axis=0), frac[~before].max(axis=0)


def main() -> None:
    cfg = load_config()
    rc = cfg["rally_feasibility"]
    rated, vmax = np.asarray(cfg["robot"]["joint_speed_rated"]), np.asarray(cfg["robot"]["joint_speed_max"])
    lines = ["# E0 rally feasibility", ""]

    lines += ["## 1. Required paddle state for cooperative rally shots", "",
              f"Hit at the juggling hit centre {cfg['hit']['center']}; land mid far half; incoming = mirror shot after a bounce on our half.",
              "", "| flight time [s] | v_out [m/s] | incoming ball at hit x: z [m], v_in [m/s] | paddle tilt toward opponent [deg] | paddle normal speed [m/s] |",
              "|---|---|---|---|---|"]
    for r in required_shots(cfg):
        lines.append(f"| {r['T']:.1f} | {np.round(r['v_out'], 2)} (|v| {np.linalg.norm(r['v_out']):.2f}) | "
                     f"{r['z_in']:.2f}, {np.round(r['v_in'], 2)} | {r['tilt']:.0f} | {r['speed']:.2f} |")

    lines += ["", "## 2. Reachable hit points for forward-tilted paddle poses", "",
              f"Grid x {rc['x_range']}, z {rc['z_range']}, step {rc['step']} m, y = 0; reachable = IK within 1 mm / 0.5 deg "
              f"with every joint >= {rc['min_joint_margin']} rad from its limits.", "",
              "| paddle tilt [deg] | " + " | ".join(f"mount pre-tilt {a} deg" for a in rc["mount_pretilts"]) + " |",
              "|---|" + "---|" * len(rc["mount_pretilts"])]
    jobs = [(a, t) for a in rc["mount_pretilts"] for t in rc["tilts"]]
    with ProcessPoolExecutor(rc["workers"]) as ex:
        res = {(a, t): pts for a, t, pts in ex.map(_reach_job, jobs)}
    for t in rc["tilts"]:
        cells = []
        for a in rc["mount_pretilts"]:
            pts = res[(a, t)]
            zs = [z for _, z in pts]
            cells.append(f"{len(pts)} pts" + (f" (z {min(zs):.2f}–{max(zs):.2f} m)" if pts else ""))
        lines.append(f"| {t} | " + " | ".join(cells) + " |")

    lines += ["", "## 3. Joint speeds and torques", ""]
    for name in ("juggle_nominal", "juggle_noisy"):
        f = RESULTS / f"{name}.npz"
        if f.exists():
            qd = np.abs(np.load(f)["qd"]).max(axis=0)
            lines.append(f"- {name}: max |q̇| {np.round(qd, 2)} rad/s = {np.round(100 * qd / rated).astype(int)} % of rated")
    st, qd, tau_app, tau_ft = rally_stroke(cfg)
    s = rc["stroke"]
    lines += [f"- tracked rally stroke ({s['tilt_deg']} deg tilt, {s['speed']} m/s at {s['hit_center']}): impact error "
              f"{1e3 * st['pos_err']:.2f} mm / {st['normal_err']:.2f} deg / {1e3 * st['timing_err']:+.2f} ms",
              f"  - max |q̇| {np.round(qd, 2)} rad/s = {np.round(100 * qd / rated).astype(int)} % of rated, "
              f"{np.round(100 * qd / vmax).astype(int)} % of max",
              f"  - peak torque before impact {np.round(100 * tau_app).astype(int)} %, after impact (follow-through) "
              f"{np.round(100 * tau_ft).astype(int)} %; planned trajectory min joint margin {st['stats']['min_margin']:.2f} rad"]
    out = RESULTS / "e0_feasibility.md"
    out.write_text("\n".join(lines) + "\n")
    print(out.read_text())


if __name__ == "__main__":
    main()
