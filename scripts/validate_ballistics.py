"""Phase 2: validate the ball flight model (M2) in MuJoCo.

For each initial condition in checks.ballistic_cases:
  1. MuJoCo run, aero force via xfrc_applied, half-step (leapfrog) state handling;
  2. the same with the naive state handling (qvel read as v(t)) — shows why the half step matters;
  3. an independent RK4 integration of the same ODE (tt.aero.rk4_rollout);
  4. with drag and Magnus off: MuJoCo vs the analytic parabola x0 + v0 t + g t^2 / 2.
Acceptance: |MuJoCo - RK4| < checks.ballistic_max_error over checks.ballistic_duration for every case.

Usage:
    python scripts/validate_ballistics.py [--video results/phase2_case3.mp4 --case 2] [--snapshot PNG] [--viewer]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tt.aero import AeroParams, BallAero, rk4_rollout  # noqa: E402
from tt.config import RESULTS, load_config  # noqa: E402
from tt.render import simulate, snapshot  # noqa: E402
from tt.sim import make_sim  # noqa: E402


def run_mujoco(cfg, params: AeroParams, case: dict, duration: float, half_step=True, video=None,
               viewer=False, snapshot_path=None):
    sim = make_sim(cfg)
    m, d = sim.m, sim.d
    aero = BallAero(m, sim.ids, params, half_step=half_step)
    aero.set_ball_state(m, d, case["pos"], case["vel"], case["spin"])
    if snapshot_path:
        import mujoco

        mujoco.mj_forward(m, d)
        snapshot(m, d, cfg, snapshot_path)
    t, pos, vel = [], [], []

    def control(m, d):
        sim.hold_arm()
        aero.apply(d)
        p, v, _ = aero.ball_state(d)
        t.append(d.time)
        pos.append(p)
        vel.append(v)

    # one extra step so that the last logged state is at t = duration
    simulate(m, d, duration + m.opt.timestep, control, cfg=cfg, video=video, viewer=viewer)
    return np.array(t), np.array(pos), np.array(vel)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", type=Path, default=None)
    ap.add_argument("--case", type=int, default=2, help="case index rendered by --video/--snapshot/--viewer")
    ap.add_argument("--video", type=Path, default=None)
    ap.add_argument("--snapshot", type=Path, default=None)
    ap.add_argument("--viewer", action="store_true")
    ap.add_argument("--log", type=Path, default=RESULTS / "phase2_ballistics.npz")
    ap.add_argument("--plot", type=Path, default=RESULTS / "phase2_ballistics.png")
    args = ap.parse_args()

    cfg = load_config(args.config)
    ck = cfg["checks"]
    T = ck["ballistic_duration"]
    dt = cfg["physics"]["timestep"]
    n = int(round(T / dt))
    g = np.array(cfg["physics"]["gravity"])
    cm_test = ck["ballistic_test_magnus_coefficient"]
    p_full = AeroParams.from_config(cfg, magnus_coefficient=cm_test)
    p_vac = AeroParams.from_config(cfg, drag_coefficient=0.0, magnus_coefficient=0.0)
    print(f"k_d = {p_full.k_d:.4f} 1/m (C_d = {cfg['aero']['drag_coefficient']}), "
          f"k_m = {p_full.k_m:.5f} (test C_m = {cm_test}), dt = {dt} s")

    logs, rows, ok_all = {}, [], True
    for i, case in enumerate(ck["ballistic_cases"]):
        render = i == args.case
        t, x_mj, v_mj = run_mujoco(cfg, p_full, case, T, video=args.video if render else None,
                                   viewer=args.viewer and render, snapshot_path=args.snapshot if render else None)
        _, x_naive, _ = run_mujoco(cfg, p_full, case, T, half_step=False)
        x_rk, v_rk = rk4_rollout(case["pos"], case["vel"], case["spin"], p_full, dt, n)
        _, x_vac, _ = run_mujoco(cfg, p_vac, case, T)
        x_par = np.asarray(case["pos"]) + np.outer(t, case["vel"]) + 0.5 * np.outer(t ** 2, g)

        err = np.linalg.norm(x_mj - x_rk, axis=1)
        err_naive = np.linalg.norm(x_naive - x_rk, axis=1)
        err_v = np.linalg.norm(v_mj - v_rk, axis=1)
        err_par = np.linalg.norm(x_vac - x_par, axis=1)
        drag_effect = np.linalg.norm(x_rk[-1] - x_par[-1])
        ok = err.max() < ck["ballistic_max_error"] and x_mj[:, 2].min() > cfg["ball"]["radius"]
        ok_all &= ok
        rows.append((case["name"], err.max(), err_v.max(), err_naive.max(), err_par.max(), drag_effect, ok))
        logs.update({f"c{i}_t": t, f"c{i}_mujoco": x_mj, f"c{i}_naive": x_naive, f"c{i}_rk4": x_rk,
                     f"c{i}_vac_mujoco": x_vac, f"c{i}_parabola": x_par})

    args.log.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.log, **logs)
    plot(cfg, logs, rows, args.plot)

    print(f"\nmax errors over {T:.1f} s           pos vs RK4   vel vs RK4   naive pos    no-drag vs parabola   drag effect")
    for name, e, ev, en, ep, de, ok in rows:
        print(f"  [{'PASS' if ok else 'FAIL'}] {name:14s}  {e * 1e3:9.4f} mm  {ev * 1e3:7.3f} mm/s  {en * 1e3:8.3f} mm"
              f"   {ep * 1e3:12.6f} mm      {de * 1e3:7.1f} mm")
    print(f"log: {args.log}\nplot: {args.plot}" + (f"\nvideo: {args.video}" if args.video else ""))
    sys.exit(0 if ok_all else 1)


def plot(cfg, logs, rows, path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    k = len(rows)
    fig, ax = plt.subplots(2, k, figsize=(5 * k, 8))
    for i, row in enumerate(rows):
        t = logs[f"c{i}_t"]
        a = ax[0, i]
        # horizontal axis: x for lobs, time for a (purely) vertical toss
        vertical = np.ptp(logs[f"c{i}_rk4"][:, 0]) < 0.01
        h = (lambda x: t) if vertical else (lambda x: x[:, 0])
        a.plot(h(logs[f"c{i}_rk4"]), logs[f"c{i}_rk4"][:, 2], "k-", lw=3, alpha=0.3, label="RK4 reference")
        a.plot(h(logs[f"c{i}_mujoco"]), logs[f"c{i}_mujoco"][:, 2], "C0--", label="MuJoCo (drag+Magnus)")
        a.plot(h(logs[f"c{i}_parabola"]), logs[f"c{i}_parabola"][:, 2], "C2:", label="vacuum parabola")
        a.set_title(row[0])
        a.set_xlabel("time [s]" if vertical else "x [m]")
        a.set_ylabel("z [m]")
        a.grid(alpha=0.3)
        a.legend(fontsize=8)
        b = ax[1, i]
        floor = 1e-9  # avoid log(0) at t = 0
        b.semilogy(t, 1e3 * np.linalg.norm(logs[f"c{i}_mujoco"] - logs[f"c{i}_rk4"], axis=1) + floor,
                   label="MuJoCo (half-step) vs RK4")
        b.semilogy(t, 1e3 * np.linalg.norm(logs[f"c{i}_naive"] - logs[f"c{i}_rk4"], axis=1) + floor,
                   label="MuJoCo (naive qvel) vs RK4")
        b.semilogy(t, 1e3 * np.linalg.norm(logs[f"c{i}_vac_mujoco"] - logs[f"c{i}_parabola"], axis=1) + floor,
                   label="no drag: MuJoCo vs parabola (exactly 0; drawn at 1e-9)")
        b.axhline(1e3 * cfg["checks"]["ballistic_max_error"], color="r", ls="--", lw=1, label="1 mm limit")
        b.set_xlabel("time [s]")
        b.set_ylabel("position error [mm]")
        b.grid(alpha=0.3, which="both")
        b.legend(fontsize=8)
    fig.suptitle("Phase 2: ball flight model validation")
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=110)


if __name__ == "__main__":
    main()
