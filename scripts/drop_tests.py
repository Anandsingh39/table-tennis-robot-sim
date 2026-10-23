"""Phase 3: validate the analytic contact model (M3) in the MuJoCo scene.

  1. Ball dropped on the arm's static horizontal paddle (arm holding the ready pose):
     rebound ratio vs e_r^2 (drag off) and vs the closed-form drag prediction (drag on);
     bounce times vs the analytic vacuum sequence (event-count / no-double-impact test).
  2. Ball dropped on the table from 0.30 m (drag on): rebound vs the ITTF 0.23 m.
  3. Prescribed (mocap) paddle, moving and/or tilted: outgoing velocity measured from the
     trajectory vs the Section 6 formula; and the Section 6 inverse (juggle law) reaching a
     desired v_out.

Usage:
    python scripts/drop_tests.py [--video results/phase3.mp4 --scenario case3 --slowmo 4] [--snapshot PNG] [--viewer]
    scenarios: paddle, table, case0..caseN (prescribed paddle cases from the config)
"""

from __future__ import annotations

import argparse
import copy
import sys
from pathlib import Path

import mujoco
import numpy as np
from scipy.spatial.transform import Rotation

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tt.aero import AeroParams, BallAero, rk4_rollout  # noqa: E402
from tt.config import RESULTS, load_config  # noqa: E402
from tt.contact import (ContactModel, SurfaceState, arm_paddle_state, paddle_surface,  # noqa: E402
                        table_surface)
from tt.juggle_law import frictionless_rebound, paddle_command  # noqa: E402
from tt.render import simulate, snapshot  # noqa: E402
from tt.sim import make_sim  # noqa: E402


def drag_rebound(h_drop: float, e: float, k: float) -> float:
    """Rebound height after a vertical drop with quadratic drag (closed form; k = 0 -> e^2 h)."""
    if k == 0:
        return e ** 2 * h_drop
    return float(np.log1p(-e ** 2 * np.expm1(-2 * k * h_drop)) / (2 * k))


def vacuum_bounce_times(h: float, e: float, g: float, t_end: float) -> np.ndarray:
    t, v, out = np.sqrt(2 * h / g), np.sqrt(2 * g * h), []
    while t < t_end:
        out.append(t)
        v *= e
        t += 2 * v / g
    return np.array(out)


def run(sim, aero, contact, duration, pre_step=None, video=None, viewer=False, slowmo=1.0, cfg=None):
    log = {"t": [], "p": [], "v": []}

    def control(m, d):
        sim.hold_arm()
        contact.step(m, d)
        aero.apply(d)
        p, v, _ = aero.ball_state(d)
        log["t"].append(d.time)
        log["p"].append(p)
        log["v"].append(v)

    simulate(sim.m, sim.d, duration, control, pre_step=pre_step, cfg=cfg or sim.cfg, video=video, viewer=viewer,
             slowmo=slowmo)
    return {k: np.array(v) for k, v in log.items()}


def drop(cfg, surface: str, drag: bool, **render):
    sim = make_sim(cfg)
    ck = cfg["checks"]
    params = AeroParams.from_config(cfg, drag_coefficient=None if drag else 0.0)
    aero = BallAero(sim.m, sim.ids, params)
    paddle = paddle_surface(cfg, arm_paddle_state(sim.ids))
    table = table_surface(cfg, params)
    contact = ContactModel(sim.m, sim.ids, aero, [paddle, table], cfg["contact"]["rewind_to_impact_time"])
    r = cfg["ball"]["radius"]
    if surface == "paddle":
        z_c = sim.p_ready[2] + cfg["paddle"]["blade_thickness"] / 2 + r      # ball centre height at contact
        xy = sim.p_ready[:2]
    else:
        tc = cfg["table"]
        z_c = tc["top_height"] + r
        xy = np.array([tc["near_edge_x"] + 1.0, 0.3])
    p0 = np.array([xy[0], xy[1], z_c + ck["drop_height"]])
    aero.set_ball_state(sim.m, sim.d, p0, np.zeros(3))
    mujoco.mj_forward(sim.m, sim.d)
    if render.get("snapshot"):
        snapshot(sim.m, sim.d, render["rcfg"], render["snapshot"])
    log = run(sim, aero, contact, ck["drop_duration"], video=render.get("video"), viewer=render.get("viewer", False),
              slowmo=render.get("slowmo", 1.0), cfg=render.get("rcfg"))
    ev = [e for e in contact.events if e.surface == surface]
    t_ev = np.array([e.t for e in ev])
    i0 = np.searchsorted(log["t"], t_ev[0])
    i1 = np.searchsorted(log["t"], t_ev[1]) if len(t_ev) > 1 else len(log["t"])
    h_reb = log["p"][i0:i1, 2].max() - z_c
    e = paddle.restitution if surface == "paddle" else table.restitution
    return dict(log=log, z_c=z_c, t_ev=t_ev, h_reb=h_reb, e=e, k=params.k_d, n_events=len(contact.events),
                surfaces=[x.surface for x in contact.events])


def tilt_matrix(tilt_deg) -> np.ndarray:
    return Rotation.from_euler("xy", tilt_deg, degrees=True).as_matrix()


def prescribed_case(cfg, case: dict, rewind: bool | None = None, **render):
    """Drop the ball onto a mocap paddle moving at constant velocity; return measured vs predicted v_out."""
    sim = make_sim(cfg, prescribed_paddle=True)
    m, d, ids = sim.m, sim.d, sim.ids
    ck = cfg["checks"]
    params = AeroParams.from_config(cfg)
    aero = BallAero(m, ids, params)
    e = cfg["contact"]["paddle_restitution"]
    h_slab = cfg["paddle"]["blade_thickness"] / 2 + cfg["ball"]["radius"]
    c_star = np.array(ck["prescribed_paddle_pos"], float)
    dt = m.opt.timestep

    # Ball falls from rest; v_in at the hit depends only on the fall height (vertical flight).
    # The impact is placed a fraction of a step after a grid point, so detection happens with
    # real penetration and the impact-time rewind is exercised.
    x_grid, _ = rk4_rollout([0, 0, 0], [0, 0, 0], [0, 0, 0], params, dt, int(1.0 / dt))
    k = int(np.searchsorted(-x_grid[:, 2], case["drop_height"]))
    t_star = (k + ck["impact_step_phase"]) * dt
    x_fall, v_fall = rk4_rollout([0, 0, 0], [0, 0, 0], [0, 0, 0], params, t_star / k, k)   # ends exactly at t_star
    x_fall, v_fall = x_fall[-1], v_fall[-1]
    if "v_out_des" in case:            # Section 6 inverse: choose n and the paddle normal speed
        v_in = v_fall
        n, s = paddle_command(v_in, np.array(case["v_out_des"]), e)
        R = Rotation.align_vectors([n], [[0, 0, 1]])[0].as_matrix()
        v_p = s * n
    else:
        R = tilt_matrix(case["tilt_deg"])
        n = R[:, 2]
        v_p = np.array(case["paddle_vel"], float)
    p_hit = c_star + h_slab * n
    p0 = p_hit - x_fall                             # so that the ball is at p_hit at t_star
    center = lambda t: c_star + v_p * (t - t_star)  # noqa: E731
    quat = np.roll(Rotation.from_matrix(R).as_quat(), 1)   # scipy (x,y,z,w) -> MuJoCo (w,x,y,z)

    def state(m, d):
        return SurfaceState(center=center(d.time), R=R, v=v_p, w=np.zeros(3))

    def move_mocap(m, d):
        d.mocap_pos[ids.prescribed_mocap] = center(d.time)
        d.mocap_quat[ids.prescribed_mocap] = quat

    surf = paddle_surface(cfg, state)
    contact = ContactModel(m, ids, aero, [surf], cfg["contact"]["rewind_to_impact_time"] if rewind is None else rewind)
    aero.set_ball_state(m, d, p0, np.zeros(3))
    move_mocap(m, d)
    mujoco.mj_forward(m, d)
    if render.get("snapshot"):
        snapshot(m, d, render["rcfg"], render["snapshot"])
    log = run(sim, aero, contact, t_star + 0.25, pre_step=move_mocap, video=render.get("video"),
              viewer=render.get("viewer", False), slowmo=render.get("slowmo", 1.0), cfg=render.get("rcfg"))
    assert len(contact.events) == 1, f"{case['name']}: {len(contact.events)} impacts"
    ev = contact.events[0]
    # Independent check (does not use the contact model's internals): by construction the ball reaches
    # p_hit at t_star with v_in = v_fall[k]. The Section 6 formula gives v_out; fly (p_hit, v_out) forward
    # with RK4 for impact_check_delay and compare with the MuJoCo ball at that time.
    v_in = v_fall
    v_out_pred = np.array(case["v_out_des"]) if "v_out_des" in case else frictionless_rebound(v_in, n, v_p, e)
    delay = ck["impact_check_delay"]
    j = int(np.searchsorted(log["t"], t_star + delay))           # logged step at/after t_star + delay
    n_after = int(round(delay / dt))
    x_rk, v_rk = rk4_rollout(p_hit, v_out_pred, [0, 0, 0], params, (log["t"][j] - t_star) / n_after, n_after)
    rel_err = np.linalg.norm(log["v"][j] - v_rk[-1]) / np.linalg.norm(v_rk[-1])
    pos_err = np.linalg.norm(log["p"][j] - x_rk[-1])
    return dict(name=case["name"], v_in=v_in, v_out=log["v"][j], v_out_pred=v_rk[-1], rel_err=rel_err,
                pos_err=pos_err, t_err=ev.t - t_star, n=n, v_p=v_p, log=log)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", type=Path, default=None)
    ap.add_argument("--scenario", default="paddle", help="paddle | table | case<i> (rendered by --video etc.)")
    ap.add_argument("--video", type=Path, default=None)
    ap.add_argument("--snapshot", type=Path, default=None)
    ap.add_argument("--viewer", action="store_true")
    ap.add_argument("--slowmo", type=float, default=1.0)
    ap.add_argument("--log", type=Path, default=RESULTS / "phase3_contact.npz")
    ap.add_argument("--plot", type=Path, default=RESULTS / "phase3_contact.png")
    args = ap.parse_args()
    np.set_printoptions(suppress=True)
    cfg = load_config(args.config)
    ck = cfg["checks"]
    g = -cfg["physics"]["gravity"][2]

    def render_for(name, lookat=None):
        if args.scenario != name:
            return {}
        rcfg = copy.deepcopy(cfg)
        if lookat is not None:
            rcfg["render"]["lookat"] = list(lookat)
        return dict(video=args.video, snapshot=args.snapshot, viewer=args.viewer, slowmo=args.slowmo, rcfg=rcfg)

    checks, logs = {}, {}
    # 1. paddle drops
    pv = drop(cfg, "paddle", drag=False, **render_for("paddle_vacuum"))
    pd = drop(cfg, "paddle", drag=True, **render_for("paddle"))
    ratio_v = pv["h_reb"] / ck["drop_height"]
    ratio_d = pd["h_reb"] / ck["drop_height"]
    pred_d = drag_rebound(ck["drop_height"], pd["e"], pd["k"]) / ck["drop_height"]
    t_an = vacuum_bounce_times(ck["drop_height"], pv["e"], g, ck["drop_duration"])
    n_cmp = min(len(t_an), len(pv["t_ev"]))
    t_err = np.abs(pv["t_ev"][:n_cmp] - t_an[:n_cmp]).max()
    checks["paddle drop (vacuum): ratio vs e^2"] = abs(ratio_v / pv["e"] ** 2 - 1) < ck["drop_ratio_tol"]
    checks["paddle drop (drag): ratio vs drag prediction"] = abs(ratio_d / pred_d - 1) < ck["drop_ratio_tol"]
    checks["paddle bounce count = analytic, no double impacts"] = (
        len(pv["t_ev"]) == len(t_an) and set(pv["surfaces"]) == {"paddle"} and t_err < ck["event_time_tol"])
    # 2. table drop
    tl = table_surface_lookat(cfg)
    td = drop(cfg, "table", drag=True, **render_for("table", tl))
    tv = drop(cfg, "table", drag=False)
    checks["table drop 0.30 m -> 0.23 m (drag)"] = (
        abs(td["h_reb"] / cfg["contact"]["table_rule_rebound_height"] - 1) < ck["table_rebound_tol"])
    checks["table drop (vacuum): ratio vs e_t^2"] = abs(tv["h_reb"] / ck["drop_height"] / tv["e"] ** 2 - 1) < ck["drop_ratio_tol"]
    # 3. prescribed paddle
    cases = []
    for i, case in enumerate(ck["prescribed_paddle_cases"]):
        res = prescribed_case(cfg, case, **render_for(f"case{i}", ck["prescribed_paddle_pos"]))
        res["rel_err_no_rewind"] = prescribed_case(cfg, case, rewind=False)["rel_err"]
        cases.append(res)
        checks[f"prescribed: {case['name']}"] = res["rel_err"] < ck["impact_velocity_tol"]

    print(f"paddle e_r = {pv['e']:.3f} (e^2 = {pv['e'] ** 2:.4f}); table e_t = {td['e']:.4f} from ITTF rule with drag "
          f"(drag-free shortcut would be {np.sqrt(0.23 / 0.30):.4f})")
    print(f"paddle drop {ck['drop_height']} m, vacuum: rebound {pv['h_reb'] * 1e3:.2f} mm, ratio {ratio_v:.4f} "
          f"vs e^2 {pv['e'] ** 2:.4f} ({(ratio_v / pv['e'] ** 2 - 1) * 100:+.2f}%)")
    print(f"paddle drop {ck['drop_height']} m, drag:   rebound {pd['h_reb'] * 1e3:.2f} mm, ratio {ratio_d:.4f} "
          f"vs prediction {pred_d:.4f} ({(ratio_d / pred_d - 1) * 100:+.2f}%)")
    print(f"paddle bounces in {ck['drop_duration']} s: {len(pv['t_ev'])} simulated vs {len(t_an)} analytic, "
          f"max time error {t_err * 1e3:.3f} ms")
    print(f"table drop {ck['drop_height']} m, drag: rebound {td['h_reb'] * 1e3:.2f} mm (ITTF 230 mm, "
          f"{(td['h_reb'] / 0.23 - 1) * 100:+.2f}%); vacuum ratio {tv['h_reb'] / ck['drop_height']:.4f} vs e_t^2 "
          f"{tv['e'] ** 2:.4f}")
    print(f"prescribed paddle ({ck['impact_check_delay'] * 1e3:.0f} ms after impact; MuJoCo vs RK4 flight of the "
          f"Section 6 v_out):")
    print("  case                   v_in [m/s]                v MuJoCo                  v predicted            "
          "vel err   pos err    t_impact err  vel err w/o rewind")
    for c in cases:
        print(f"  {c['name']:22s} {np.array2string(c['v_in'], precision=3):25s} {np.array2string(c['v_out'], precision=4):25s}"
              f" {np.array2string(c['v_out_pred'], precision=4):23s} {c['rel_err'] * 100:.4f}%  "
              f"{c['pos_err'] * 1e3:.4f} mm  {c['t_err'] * 1e3:+.3f} ms     {c['rel_err_no_rewind'] * 100:.3f}%")
    for k, ok in checks.items():
        print(f"  [{'PASS' if ok else 'FAIL'}] {k}")

    for name, res in (("paddle_vac", pv), ("paddle_drag", pd), ("table_drag", td), ("table_vac", tv)):
        logs.update({f"{name}_t": res["log"]["t"], f"{name}_z": res["log"]["p"][:, 2] - res["z_c"],
                     f"{name}_tev": res["t_ev"]})
    for i, c in enumerate(cases):
        logs.update({f"case{i}_t": c["log"]["t"], f"case{i}_p": c["log"]["p"], f"case{i}_v": c["log"]["v"],
                     f"case{i}_vout": c["v_out"], f"case{i}_vpred": c["v_out_pred"]})
    args.log.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.log, **logs)
    plot(cfg, pv, pd, td, t_an, cases, args.plot)
    print(f"log: {args.log}\nplot: {args.plot}" + (f"\nvideo: {args.video}" if args.video else ""))
    sys.exit(0 if all(checks.values()) else 1)


def table_surface_lookat(cfg):
    tc = cfg["table"]
    return [tc["near_edge_x"] + 1.0, 0.3, tc["top_height"] + 0.2]


def plot(cfg, pv, pd, td, t_an, cases, path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    ck = cfg["checks"]
    fig, ax = plt.subplots(2, 2, figsize=(12, 8))
    a = ax[0, 0]
    a.plot(pv["log"]["t"], 1e3 * (pv["log"]["p"][:, 2] - pv["z_c"]), label="drag off")
    a.plot(pd["log"]["t"], 1e3 * (pd["log"]["p"][:, 2] - pd["z_c"]), label="drag on")
    a.axhline(1e3 * pv["e"] ** 2 * ck["drop_height"], color="C0", ls=":", label="e$^2$ h (drag off)")
    for t in t_an:
        a.axvline(t, color="k", lw=0.5, alpha=0.4)
    a.set(title=f"ball on static paddle, e_r = {pv['e']}", xlabel="time [s]", ylabel="height above contact [mm]")
    a.legend(fontsize=8)
    a = ax[0, 1]
    a.plot(td["log"]["t"], 1e3 * (td["log"]["p"][:, 2] - td["z_c"]), label="drag on")
    a.axhline(1e3 * cfg["contact"]["table_rule_rebound_height"], color="r", ls="--", label="ITTF 230 mm")
    a.set(title=f"ball on table, e_t = {td['e']:.3f} (from ITTF rule)", xlabel="time [s]",
          ylabel="height above contact [mm]")
    a.legend(fontsize=8)
    a = ax[1, 0]
    n_cmp = min(len(t_an), len(pv["t_ev"]))
    a.plot(range(1, n_cmp + 1), 1e3 * (pv["t_ev"][:n_cmp] - t_an[:n_cmp]), "o-")
    a.axhline(0, color="k", lw=0.5)
    a.set(title=f"bounce times vs analytic (drag off): {len(pv['t_ev'])} sim / {len(t_an)} analytic",
          xlabel="bounce #", ylabel="time error [ms]")
    a = ax[1, 1]
    x = np.arange(len(cases))
    a.bar(x - 0.2, [100 * c["rel_err"] for c in cases], 0.4, label="with impact-time rewind")
    a.bar(x + 0.2, [100 * c["rel_err_no_rewind"] for c in cases], 0.4, label="without rewind")
    a.axhline(100 * ck["impact_velocity_tol"], color="r", ls="--", label="1% limit")
    a.set_xticks(range(len(cases)), [c["name"] for c in cases], rotation=20, ha="right", fontsize=8)
    a.set(title=f"moving/tilted paddle: velocity {1e3 * ck['impact_check_delay']:.0f} ms after impact\n"
                "|MuJoCo - RK4(Section 6 v_out)| / |RK4|", ylabel="relative error [%] (log)", yscale="log")
    a.legend(fontsize=8)
    for a in ax.flat:
        a.grid(alpha=0.3)
    fig.suptitle("Phase 3: analytic contact model")
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=110)


if __name__ == "__main__":
    main()
