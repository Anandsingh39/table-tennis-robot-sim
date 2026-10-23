"""Phase 4: choose the hit-plane centre (and so the ready pose) from a workspace scan.

For every grid point c = (x, y, z) with the paddle facing up, the point is feasible when
  * IK reaches c with normal up, and with the normal tilted by max_tilt about x and y (both signs),
  * the ready pose (c + ready offset) is reachable,
  * every IK solution keeps all joints >= min_joint_margin from their limits,
  * no arm link above the hit plane is within ball_column_clearance of the ball's vertical path.
Score = Yoshikawa manipulability sqrt(det(J J^T)) of the 5x7 paddle task Jacobian (centre velocity +
tilt rates) at the upright pose: larger = paddle velocity/tilt reachable in all directions with smaller q_dot.

Usage:
    python scripts/workspace_scan.py [--plot results/phase4_workspace.png]
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
from tt.kinematics import manipulability, solve_paddle_pose_cfg, task_jacobian  # noqa: E402
from tt.model import build_model  # noqa: E402


def evaluate(m, ids, cfg, c, q_seed, d):
    ws = cfg["workspace_scan"]
    # nominal paddle normal: vertical (juggling) or tilted toward +x about y (rally); variations of
    # ±max_tilt are applied about the world x and y axes on top of it
    R_nom = Rotation.from_euler("y", ws.get("nominal_tilt_deg", 0.0), degrees=True)
    up = R_nom.apply([0.0, 0, 1])
    normals = [up] + [(Rotation.from_euler(ax, s * ws["max_tilt_deg"], degrees=True) * R_nom).apply([0.0, 0, 1])
                      for ax in "xy" for s in (1, -1)]
    ready_n = np.asarray(cfg["ready_pose"]["paddle_normal"], float)
    targets = [(c, n) for n in normals] + [(c + np.asarray(cfg["ready_pose"]["offset_from_hit_center"]), ready_n)]
    sols, margin = [], np.inf
    for p_t, n_t in targets:
        q, p, n = solve_paddle_pose_cfg(m, ids, cfg, p_t, n_t, q0=q_seed)
        ok = (np.linalg.norm(p - p_t) < ws["ik_tol_pos"]
              and np.rad2deg(np.arccos(np.clip(n @ n_t, -1, 1))) < ws["ik_tol_normal_deg"])
        if not ok:
            return None
        margin = min(margin, np.min(np.minimum(q - ids.q_lo, ids.q_hi - q)))
        sols.append(q)
    q_up = sols[0]
    J = task_jacobian(m, d, ids, q_up)          # also leaves d at q_up for the clearance check
    arm_bodies = [m.body(f"{cfg['robot']['arm_prefix']}link{i}").id for i in range(1, 7)] + [ids.ee_body]
    clear = min((np.linalg.norm(d.xpos[b, :2] - c[:2]) for b in arm_bodies if d.xpos[b, 2] > c[2]), default=np.inf)
    return dict(q=q_up, w=manipulability(J), margin=margin, clearance=clear,
                feasible=margin >= ws["min_joint_margin"] and clear >= ws["ball_column_clearance"])


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", type=Path, default=None)
    ap.add_argument("--plot", type=Path, default=RESULTS / "phase4_workspace.png")
    ap.add_argument("--log", type=Path, default=RESULTS / "phase4_workspace.npz")
    args = ap.parse_args()
    cfg = load_config(args.config)
    ws = cfg["workspace_scan"]
    m, ids = build_model(cfg)
    d = mujoco.MjData(m)
    xs = np.arange(ws["x_range"][0], ws["x_range"][1] + 1e-9, ws["step"])
    zs = np.arange(ws["z_range"][0], ws["z_range"][1] + 1e-9, ws["step"])
    W = np.full((len(zs), len(xs)), np.nan)
    feas = np.zeros_like(W, dtype=bool)
    margin = np.full_like(W, np.nan)
    best = None
    q_nom = np.asarray(cfg["ready_pose"]["q_nominal"], float)
    for iz, z in enumerate(zs):
        for ix, x in enumerate(xs):
            c = np.array([x, ws["y"], z])
            r = evaluate(m, ids, cfg, c, q_nom, d)
            if r is None:
                continue
            W[iz, ix], feas[iz, ix], margin[iz, ix] = r["w"], r["feasible"], r["margin"]
            if r["feasible"] and (best is None or r["w"] > best[1]["w"]):
                best = (c, r)
    np.savez_compressed(args.log, x=xs, z=zs, manipulability=W, feasible=feas, margin=margin)

    print(f"grid {len(xs)} x {len(zs)}: reachable (all tilts) {np.isfinite(W).sum()}, feasible {feas.sum()}")
    if best is None:
        print("no feasible point")
        sys.exit(1)
    c, r = best
    print(f"best hit centre: {np.array2string(c, precision=3)}  manipulability {r['w']:.4f}  "
          f"min joint margin {r['margin']:.3f} rad  ball-column clearance {r['clearance']:.3f} m")
    print(f"  q_up = {np.array2string(r['q'], precision=3)}")
    print(f"  current config hit.center = {cfg['hit']['center']}")
    plot(xs, zs, W, feas, c, cfg, args.plot)
    print(f"plot: {args.plot}")


def plot(xs, zs, W, feas, c, cfg, path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(7, 6))
    ext = [xs[0] - 0.0125, xs[-1] + 0.0125, zs[0] - 0.0125, zs[-1] + 0.0125]
    im = ax.imshow(np.where(feas, W, np.nan), origin="lower", extent=ext, cmap="viridis", aspect="equal")
    ax.imshow(np.where(np.isfinite(W) & ~feas, 1.0, np.nan), origin="lower", extent=ext, cmap="Greys",
              vmin=0, vmax=3, aspect="equal")
    fig.colorbar(im, label="manipulability of paddle task (feasible points)")
    ax.plot(c[0], c[2], "r*", ms=15, label=f"chosen hit centre z_h = {c[2]:.3f} m")
    base = cfg["robot"]["base_pos"]
    ax.plot(base[0], base[2], "ks", label="arm base (shoulder)")
    ax.axhline(cfg["table"]["top_height"], color="C0", ls="--", lw=1, label="table top")
    ax.set(xlabel="x [m]", ylabel="z [m]",
           title=f"hit-plane scan (y = {cfg['workspace_scan']['y']}, tilts ±{cfg['workspace_scan']['max_tilt_deg']:.0f}°)\n"
                 "grey = reachable but too close to limits / arm in ball path")
    ax.legend(fontsize=8, loc="lower left")
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=120)


if __name__ == "__main__":
    main()
