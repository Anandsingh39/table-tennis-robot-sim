"""Phase 4: plan and track a paddle stroke with computed-torque control.

For each stroke in tracking.strokes: the arm rests in the ready pose, then a quintic joint
trajectory (tt.planner) brings the paddle centre to p* with normal n* and velocity v* = speed n*
at the commanded impact time t*, follows through and returns to the ready pose. The computed-
torque controller (tt.control) tracks it.

Acceptance at t*: paddle position error < max_pos_err, normal error < max_normal_err_deg,
timing error (when the paddle centre actually crosses the plane through p* normal to v*) <
max_timing_err, and torques never exceed the motor limits.

Usage:
    python scripts/tracking_test.py [--video results/phase4_stroke.mp4 --stroke 0 --slowmo 4] [--snapshot PNG] [--viewer]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import mujoco
import numpy as np
from scipy.spatial.transform import Rotation

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tt.config import RESULTS, load_config  # noqa: E402
from tt.control import computed_torque  # noqa: E402
from tt.kinematics import paddle_fk, paddle_pose  # noqa: E402
from tt.log_utils import Logger  # noqa: E402
from tt.planner import impact_joint_state, plan_stroke, trajectory_stats  # noqa: E402
from tt.render import simulate, snapshot  # noqa: E402
from tt.sim import make_sim  # noqa: E402


def run_stroke(cfg, stroke, video=None, viewer=False, slowmo=1.0, snapshot_path=None):
    sim = make_sim(cfg)
    m, d, ids = sim.m, sim.d, sim.ids
    pl, cc, tr = cfg["planner"], cfg["control"], cfg["tracking"]
    dt = m.opt.timestep
    # park the ball on the floor, out of the way (no ball in this test)
    d.qpos[ids.ball_qpos:ids.ball_qpos + 3] = [-1.0, -1.0, cfg["ball"]["radius"]]
    mujoco.mj_forward(m, d)

    p_star = np.asarray(cfg["hit"]["center"], float) + stroke["offset"]
    n_star = Rotation.from_euler("xy", stroke["tilt_deg"], degrees=True).apply([0.0, 0.0, 1.0])
    v_star = stroke["speed"] * n_star
    t_start = tr["settle_time"]
    t_hit = t_start + pl["approach_time"]
    scratch = mujoco.MjData(m)
    q_hit, qd_hit, info = impact_joint_state(m, scratch, ids, cfg, p_star, n_star, v_star, sim.q_ready)
    zeros = np.zeros(len(q_hit))
    traj = plan_stroke(t_start, sim.q_ready, zeros, zeros, t_hit, q_hit, qd_hit, pl["return_time"], sim.q_ready)
    stats = trajectory_stats(traj, ids, t_start, dt)
    kp, kd = cc["ct_kp"], cc["ct_kd"]
    log = Logger()

    def control(m, d):
        q_d, qd_d, qdd_d = traj(d.time)
        tau, sat = computed_torque(m, d, ids, q_d, qd_d, qdd_d, kp, kd, cc["friction_ff"], cc["friction_smoothing"])
        d.ctrl[ids.arm_act] = tau
        p, n = paddle_pose(m, d, ids)
        p_d, n_d, _, _ = paddle_fk(m, scratch, ids, q_d)       # desired paddle pose = FK(q_d)
        log.log(t=d.time, q=d.qpos[ids.arm_qpos], qd=d.qvel[ids.arm_dof], q_d=q_d, qd_d=qd_d, qdd_d=qdd_d,
                tau=tau, sat=sat, p=p, n=n, p_d=p_d, n_d=n_d, ncon=d.ncon)

    if snapshot_path:
        snapshot(m, d, cfg, snapshot_path)
    simulate(m, d, traj.t_end + tr["settle_time"], control, cfg=cfg, video=video, viewer=viewer, slowmo=slowmo)
    L = log.arrays()

    t = L["t"]
    k = int(np.argmin(np.abs(t - t_hit)))
    pos_err = np.linalg.norm(L["p"][k] - p_star)
    normal_err = np.rad2deg(np.arccos(np.clip(L["n"][k] @ n_star, -1, 1)))
    v_meas = (L["p"][k + 1] - L["p"][k - 1]) / (t[k + 1] - t[k - 1])
    s = (L["p"] - p_star) @ (v_star / np.linalg.norm(v_star))       # signed distance past the impact plane
    win = np.arange(k - 50, k + 50)
    j = win[np.nonzero((s[win[:-1]] < 0) & (s[win[1:]] >= 0))[0]]
    t_cross = t[j[0]] - s[j[0]] * (t[j[0] + 1] - t[j[0]]) / (s[j[0] + 1] - s[j[0]]) if len(j) else np.nan
    res = dict(name=stroke["name"], pos_err=pos_err, normal_err=normal_err, timing_err=t_cross - t_hit,
               v_meas=v_meas, v_star=v_star, speed_err=np.linalg.norm(v_meas - v_star),
               tau_usage=np.abs(L["tau"] / ids.tau_max).max(axis=0), saturated=bool(L["sat"].any()),
               max_ncon=int(L["ncon"].max()), info=info, stats=stats, t_hit=t_hit, q_hit=q_hit, qd_hit=qd_hit)
    return res, L, ids


def plot(res, L, ids, path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    t = L["t"]
    fig, ax = plt.subplots(5, 1, figsize=(10, 13), sharex=True)
    labels = [f"j{i + 1}" for i in range(L["q"].shape[1])]
    ax[0].plot(t, L["q_d"], lw=2, alpha=0.35)
    ax[0].set_prop_cycle(None)
    ax[0].plot(t, L["q"], "--", lw=1)
    ax[0].set_ylabel("q (thick: desired) [rad]")
    ax[0].legend(labels, ncol=7, fontsize=7)
    ax[1].plot(t, np.rad2deg(L["q_d"] - L["q"]))
    ax[1].set_ylabel("tracking error [deg]")
    ax[2].plot(t, L["qd"])
    ax[2].set_ylabel("q̇ [rad/s]")
    ax[3].plot(t, 100 * L["tau"] / ids.tau_max)
    ax[3].axhline(100, color="r", ls="--", lw=1)
    ax[3].axhline(-100, color="r", ls="--", lw=1)
    ax[3].set_ylabel("τ [% of limit]")
    e_p = 1e3 * np.linalg.norm(L["p"] - L["p_d"], axis=1)
    e_n = np.rad2deg(np.arccos(np.clip(np.sum(L["n"] * L["n_d"], axis=1), -1, 1)))
    ax[4].plot(t, e_p, label="|p - FK(q_d)| [mm]")
    ax[4].plot(t, e_n, label="normal error [deg]")
    ax[4].set_ylabel("task-space error")
    ax[4].set_xlabel("time [s]")
    ax[4].legend(fontsize=8)
    for a in ax:
        a.axvline(res["t_hit"], color="k", ls=":", lw=1)
        a.grid(alpha=0.3)
    fig.suptitle(f"Phase 4 stroke '{res['name']}': at impact pos err {res['pos_err'] * 1e3:.3f} mm, "
                 f"normal err {res['normal_err']:.3f}°, timing err {res['timing_err'] * 1e3:+.3f} ms")
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=100)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", type=Path, default=None)
    ap.add_argument("--stroke", type=int, default=0, help="stroke index rendered by --video/--snapshot/--viewer")
    ap.add_argument("--video", type=Path, default=None)
    ap.add_argument("--snapshot", type=Path, default=None)
    ap.add_argument("--viewer", action="store_true")
    ap.add_argument("--slowmo", type=float, default=1.0)
    args = ap.parse_args()
    np.set_printoptions(precision=3, suppress=True)
    cfg = load_config(args.config)
    tr = cfg["tracking"]

    ok_all = True
    for i, stroke in enumerate(tr["strokes"]):
        sel = i == args.stroke
        res, L, ids = run_stroke(cfg, stroke, video=args.video if sel else None, viewer=args.viewer and sel,
                                 slowmo=args.slowmo, snapshot_path=args.snapshot if sel else None)
        np.savez_compressed(RESULTS / f"phase4_tracking_{i}.npz", **L)
        plot(res, L, ids, RESULTS / f"phase4_tracking_{i}.png")
        checks = {
            f"position error {res['pos_err'] * 1e3:.3f} mm < {tr['max_pos_err'] * 1e3:.0f} mm": res["pos_err"] < tr["max_pos_err"],
            f"normal error {res['normal_err']:.3f} deg < {tr['max_normal_err_deg']} deg": res["normal_err"] < tr["max_normal_err_deg"],
            f"timing error {res['timing_err'] * 1e3:+.3f} ms, |.| < {tr['max_timing_err'] * 1e3:.0f} ms":
                abs(res["timing_err"]) < tr["max_timing_err"],
            f"torque within limits (peak {100 * res['tau_usage'].max():.0f}% of limit, no saturation)": not res["saturated"],
        }
        ok_all &= all(checks.values())
        print(f"\nstroke '{res['name']}': impact at t* = {res['t_hit']:.3f} s")
        print(f"  impact IK: pos err {res['info']['ik_pos_err'] * 1e3:.4f} mm, normal err "
              f"{res['info']['ik_normal_err_deg']:.4f} deg, |J q̇* - v*| {res['info']['task_vel_err']:.2e} m/s")
        print(f"  q* = {res['q_hit']}  q̇* = {res['qd_hit']}")
        print(f"  paddle velocity at impact: measured {res['v_meas']} vs commanded {res['v_star']} "
              f"(err {res['speed_err'] * 1e3:.1f} mm/s)")
        print(f"  torque usage per joint [%]: {100 * res['tau_usage']}")
        print(f"  planned trajectory: min joint margin {res['stats']['min_margin']:.3f} rad, "
              f"max |q̇| {res['stats']['max_qd']} rad/s; max contacts during run {res['max_ncon']}")
        for k, ok in checks.items():
            print(f"  [{'PASS' if ok else 'FAIL'}] {k}")
        print(f"  plot: results/phase4_tracking_{i}.png")
    if args.video:
        print(f"video: {args.video}")
    sys.exit(0 if ok_all else 1)


if __name__ == "__main__":
    main()
