"""Phase 1: build the scene, move the arm to the IK ready pose, and hold it.

Controller: tau = qfrc_bias + Kp (q_ready - q) - Kd qdot (gravity comp + joint PD).
Acceptance: paddle-center drift < checks.hold_max_drift over checks.hold_duration.

The ball is parked above the paddle by default (ball-paddle contact arrives in Phase 3);
--drop-ball lets it fall (it passes through the paddle and lands on the floor).

Usage:
    python scripts/view_scene.py --snapshot results/phase1_scene.png --video results/phase1_hold.mp4
    python scripts/view_scene.py --viewer            # on a machine with a display (mjpython on macOS)
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import mujoco
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tt.config import RESULTS, load_config  # noqa: E402
from tt.control import gravity_comp_pd  # noqa: E402
from tt.kinematics import paddle_pose  # noqa: E402
from tt.log_utils import Logger  # noqa: E402
from tt.render import simulate, snapshot  # noqa: E402
from tt.sim import make_sim  # noqa: E402


def setup(cfg: dict):
    sim = make_sim(cfg)
    m, d, ids = sim.m, sim.d, sim.ids
    d.qpos[ids.ball_qpos:ids.ball_qpos + 3] = sim.p_ready + [0, 0, cfg["ball"]["init_offset_above_paddle"]]
    mujoco.mj_forward(m, d)
    return m, d, ids, sim.q_ready


def run_hold(cfg: dict, duration: float, video=None, viewer=False, drop_ball=False):
    m, d, ids, q_ready = setup(cfg)
    kp = np.array(cfg["control"]["hold_kp"])
    kd = np.array(cfg["control"]["hold_kd"])
    p0, n0 = paddle_pose(m, d, ids)
    ball_q0 = d.qpos[ids.ball_qpos:ids.ball_qpos + 7].copy()
    log = Logger()

    def park_ball(m, d):
        d.qpos[ids.ball_qpos:ids.ball_qpos + 7] = ball_q0
        d.qvel[ids.ball_dof:ids.ball_dof + 6] = 0.0

    def control(m, d):
        # Between mj_step1 and mj_step2: kinematics, qfrc_bias and d.time all refer to the same state.
        d.ctrl[ids.arm_act] = gravity_comp_pd(m, d, ids, q_ready, kp, kd)
        p, n = paddle_pose(m, d, ids)
        log.log(t=d.time, q=d.qpos[ids.arm_qpos], qd=d.qvel[ids.arm_dof], tau=d.ctrl[ids.arm_act],
                paddle_pos=p, paddle_normal=n, ncon=d.ncon)

    simulate(m, d, duration, control, pre_step=None if drop_ball else park_ball,
             cfg=cfg, video=video, viewer=viewer)
    return m, d, ids, q_ready, p0, n0, log.arrays()


def gravity_usage(m, ids, q):
    d = mujoco.MjData(m)
    d.qpos[ids.arm_qpos] = q
    mujoco.mj_forward(m, d)
    return d.qfrc_bias[ids.arm_dof].copy()


def plot(logs: dict, ids, p0, q_ready, path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    t = logs["t"]
    fig, ax = plt.subplots(3, 1, figsize=(8, 8), sharex=True)
    ax[0].plot(t, 1e3 * np.linalg.norm(logs["paddle_pos"] - p0, axis=1))
    ax[0].set_ylabel("paddle drift [mm]")
    ax[1].plot(t, np.rad2deg(logs["q"] - q_ready))
    ax[1].set_ylabel("q - q_ready [deg]")
    ax[2].plot(t, logs["tau"] / ids.tau_max * 100)
    ax[2].set_ylabel("torque [% of limit]")
    ax[2].set_xlabel("time [s]")
    ax[1].legend([f"j{i + 1}" for i in range(len(q_ready))], ncol=7, fontsize=7)
    for a in ax:
        a.grid(alpha=0.3)
    fig.suptitle("Phase 1: hold ready pose (gravity comp + PD)")
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=120)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", type=Path, default=None)
    ap.add_argument("--duration", type=float, default=None, help="default: checks.hold_duration")
    ap.add_argument("--video", type=Path, default=None, help="write an MP4 of the run")
    ap.add_argument("--snapshot", type=Path, default=None, help="write a PNG of the initial scene")
    ap.add_argument("--viewer", action="store_true", help="interactive viewer (needs a display)")
    ap.add_argument("--drop-ball", action="store_true", help="let the ball fall instead of parking it")
    ap.add_argument("--log", type=Path, default=RESULTS / "phase1_hold.npz")
    ap.add_argument("--plot", type=Path, default=RESULTS / "phase1_hold.png")
    args = ap.parse_args()

    cfg = load_config(args.config)
    duration = args.duration or cfg["checks"]["hold_duration"]

    if args.snapshot:
        m, d, _, _ = setup(cfg)
        print(f"snapshot: {snapshot(m, d, cfg, args.snapshot)}")

    m, d, ids, q_ready, p0, n0, logs = run_hold(cfg, duration, args.video, args.viewer, args.drop_ball)
    np.savez_compressed(args.log, **logs)
    plot(logs, ids, p0, q_ready, args.plot)

    drift = np.linalg.norm(logs["paddle_pos"] - p0, axis=1).max()
    tilt = np.rad2deg(np.arccos(np.clip(logs["paddle_normal"] @ n0, -1, 1))).max()
    g = gravity_usage(m, ids, q_ready)
    ck = cfg["checks"]
    rp = cfg["ready_pose"]
    ik_pos_err = np.linalg.norm(p0 - rp["paddle_pos"])
    ik_n_err = np.rad2deg(np.arccos(np.clip(n0 @ np.array(rp["paddle_normal"]), -1, 1)))

    print(f"model: nq={m.nq} nv={m.nv} nu={m.nu}  paddle mass={m.body_subtreemass[ids.paddle_body]:.3f} kg  "
          f"ball mass={m.body_mass[ids.ball_body] * 1e3:.1f} g")
    print("q_ready [rad]:", np.array2string(q_ready, precision=4))
    print("margin to joint limits [rad]:", np.array2string(np.minimum(q_ready - ids.q_lo, ids.q_hi - q_ready), precision=3))
    print("gravity torque at ready pose [% of limit]:", np.array2string(np.abs(g) / ids.tau_max * 100, precision=1))
    print(f"ready pose IK: pos err {ik_pos_err * 1e3:.4f} mm, normal err {ik_n_err:.4f} deg")
    print(f"hold {duration:.1f} s: max paddle drift {drift * 1e3:.4f} mm, max normal change {tilt:.4f} deg, "
          f"max |tau| {np.abs(logs['tau'] / ids.tau_max).max() * 100:.1f}% of limit, max contacts {logs['ncon'].max()}")
    checks = {
        "IK position": ik_pos_err < ck["ready_pose_pos_tol"],
        "IK normal": ik_n_err < ck["ready_pose_normal_tol_deg"],
        "hold drift": drift < ck["hold_max_drift"],
    }
    for k, ok in checks.items():
        print(f"  [{'PASS' if ok else 'FAIL'}] {k}")
    print(f"log: {args.log}\nplot: {args.plot}" + (f"\nvideo: {args.video}" if args.video else ""))
    sys.exit(0 if all(checks.values()) else 1)


if __name__ == "__main__":
    main()
