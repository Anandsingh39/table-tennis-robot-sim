"""Extra credit E1: one arm returns launched balls over the net with the MPC swing.

Runs seeds 0..n-1 in parallel (one ball per episode, launcher seeded per episode) and reports:
outcome counts (in / net / long / short / wide / no_hit), landing error vs the target, impact accuracy
(paddle centre vs planned, impact time vs planned, outgoing velocity vs planned), MPC solve times,
torque usage, joint-limit margins. Writes results/e1_summary.md, results/e1_landing.png, results/e1_runs.npz.

Usage:
    python scripts/rally_return.py [--balls 100] [--video results/e1_return.mp4 --video-seeds 0 1 2 --slowmo 1]
"""

from __future__ import annotations

import os

os.environ.setdefault("OMP_NUM_THREADS", "1")

import argparse  # noqa: E402
import subprocess  # noqa: E402
import sys  # noqa: E402
import tempfile  # noqa: E402
import time  # noqa: E402
from concurrent.futures import ProcessPoolExecutor  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tt.config import RALLY_CONFIG, RESULTS, load_config  # noqa: E402
from tt.mpc import solve_stat_rows  # noqa: E402


def episode(args):
    cfg_path, seed = args
    from tt.rally import run_return

    cfg = load_config(cfg_path)
    t0 = time.time()
    r = run_return(cfg, seed)
    L = r.logs
    ok = [s for s in r.strikes if s.adopted]          # plans that were executed (exact, or best effort)
    out = dict(seed=seed, outcome=r.outcome, wall=time.time() - t0,
               landing=np.full(2, np.nan) if r.landing is None else r.landing, target=r.target, z_net=r.z_at_net,
               n_solves=len(r.strikes), n_ok=sum(s.success for s in r.strikes),
               solve_log=[(s.success, s.best_effort, s.adopted, s.status, s.impact_err) for s in r.strikes],
               solve_times=np.array([s.solve_time for s in r.strikes]),
               peak_tau=np.abs(L["tau"]).max(axis=0) / r.tau_max, saturated=bool(L["sat"].any()),
               min_margin=float(np.min(np.minimum(L["q"] - r.q_lo, r.q_hi - L["q"]))),
               max_qd=np.abs(L["qd"]).max(axis=0), launch_speed=float(np.linalg.norm(r.launch.vel)))
    if r.paddle_events and ok:
        e, s = r.paddle_events[0], ok[-1]
        k = int(np.searchsorted(L["t"], e.t))
        out.update(hit=True, dt_hit=e.t - s.t_hit, c_err=float(np.linalg.norm(L["paddle_p"][k] - s.c_star)),
                   n_err=float(np.degrees(np.arccos(np.clip(L["paddle_n"][k] @ s.n, -1, 1)))),
                   vout_err=float(np.linalg.norm(e.v_out - s.v_out)), T=s.T,
                   tilt=float(np.degrees(np.arccos(np.clip(s.n[2], -1, 1)))), speed=s.speed)
    else:
        out.update(hit=bool(r.paddle_events))
    return out


def render(cfg_path, seeds, path: Path, slowmo: float):
    """One MP4 per seed (rally camera), concatenated with ffmpeg."""
    from tt.rally import run_return

    cfg = load_config(cfg_path)
    with tempfile.TemporaryDirectory() as tmp:
        parts = []
        for s in seeds:
            part = Path(tmp) / f"ball_{s}.mp4"
            run_return(cfg, s, video=part, slowmo=slowmo)
            parts.append(part)
        lst = Path(tmp) / "list.txt"
        lst.write_text("".join(f"file '{p}'\n" for p in parts))
        path.parent.mkdir(parents=True, exist_ok=True)
        subprocess.run(["ffmpeg", "-v", "error", "-y", "-f", "concat", "-safe", "0", "-i", str(lst), "-c", "copy",
                        str(path)], check=True)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", type=Path, default=RALLY_CONFIG)
    ap.add_argument("--balls", type=int, default=None)
    ap.add_argument("--workers", type=int, default=None)
    ap.add_argument("--video", type=Path, default=None)
    ap.add_argument("--video-seeds", type=int, nargs="*", default=[0, 1, 2])
    ap.add_argument("--slowmo", type=float, default=1.0)
    args = ap.parse_args()
    cfg = load_config(args.config)
    ec = cfg["e1"]
    n = args.balls or ec["balls"]
    t0 = time.time()
    with ProcessPoolExecutor(args.workers or ec["workers"]) as ex:
        runs = list(ex.map(episode, [(args.config, s) for s in range(n)]))
    print(f"{n} balls in {time.time() - t0:.0f} s")

    outcomes = {}
    for r in runs:
        outcomes[r["outcome"]] = outcomes.get(r["outcome"], 0) + 1
    hits = [r for r in runs if r["hit"] and "c_err" in r]
    ins = [r for r in runs if r["outcome"] == "in"]
    land_err = np.array([np.linalg.norm(r["landing"] - r["target"]) for r in ins])
    st = np.concatenate([r["solve_times"] for r in runs])
    rows = [
        ("balls", n), ("outcomes", ", ".join(f"{k}: {v}" for k, v in sorted(outcomes.items()))),
        ("hit rate (paddle contact)", f"{100 * len([r for r in runs if r['hit']]) / n:.0f}%"),
        ("return rate (lands on opponent half)", f"{100 * len(ins) / n:.0f}%"),
        ("landing error vs target: mean / RMS / p95 / max [mm]",
         f"{1e3 * land_err.mean():.1f} / {1e3 * np.sqrt(np.mean(land_err ** 2)):.1f} / "
         f"{1e3 * np.percentile(land_err, 95):.1f} / {1e3 * land_err.max():.1f}" if len(land_err) else "–"),
        ("impact: paddle centre vs plan, RMS [mm]", f"{1e3 * np.sqrt(np.mean([r['c_err'] ** 2 for r in hits])):.2f}" if hits else "–"),
        ("impact: normal error, RMS [deg]", f"{np.sqrt(np.mean([r['n_err'] ** 2 for r in hits])):.3f}" if hits else "–"),
        ("impact: time vs plan, RMS [ms]", f"{1e3 * np.sqrt(np.mean([r['dt_hit'] ** 2 for r in hits])):.2f}" if hits else "–"),
        ("outgoing ball velocity vs plan, RMS [mm/s]", f"{1e3 * np.sqrt(np.mean([r['vout_err'] ** 2 for r in hits])):.1f}" if hits else "–"),
        ("shot: flight time T [s] (min / median / max)", f"{min(r['T'] for r in hits)} / {np.median([r['T'] for r in hits])} / {max(r['T'] for r in hits)}" if hits else "–"),
        ("shot: paddle tilt [deg], normal speed [m/s] (median)",
         f"{np.median([r['tilt'] for r in hits]):.0f}, {np.median([r['speed'] for r in hits]):.2f}" if hits else "–"),
        ("ball height over the net, median [m]", f"{np.nanmedian([r['z_net'] for r in runs]):.2f}"),
        ("incoming ball launch speed, median [m/s]", f"{np.median([r['launch_speed'] for r in runs]):.2f}"),
        ("MPC solves per ball (mean)", f"{np.mean([r['n_solves'] for r in runs]):.1f} ({sum(r['n_solves'] - r['n_ok'] for r in runs)} failed in total)"),
        *solve_stat_rows([x for r in runs for x in r["solve_log"]]),
        ("MPC solve time: median / p95 / max [ms]", f"{1e3 * np.median(st):.0f} / {1e3 * np.percentile(st, 95):.0f} / {1e3 * st.max():.0f}"),
        ("peak torque, worst ball [% of limit] per joint", str(np.round(100 * np.max([r["peak_tau"] for r in runs], axis=0)).astype(int))),
        ("balls with torque saturation", str(sum(r["saturated"] for r in runs))),
        ("max joint speed, worst ball [% of rated]",
         str(np.round(100 * np.max([r["max_qd"] for r in runs], axis=0) / np.asarray(cfg["robot"]["joint_speed_rated"])).astype(int))),
        ("min joint-limit margin over all balls [rad]", f"{min(r['min_margin'] for r in runs):.3f}"),
    ]
    md = ["# E1: single-arm return with MPC swing", "",
          f"Config: config/rally.yaml (pre-tilted paddle mount, hit zone {cfg['hit']['center']}); true ball state; "
          f"launcher = stand-in opponent (config launcher:). Target: {np.round(runs[0]['target'], 3)} on the opponent half.", "",
          "| metric | value |", "|---|---|"] + [f"| {a} | {b} |" for a, b in rows]
    (RESULTS / "e1_summary.md").write_text("\n".join(md) + "\n")
    print("\n".join(md))
    np.savez_compressed(RESULTS / "e1_runs.npz", outcome=[r["outcome"] for r in runs],
                        landing=np.array([r["landing"] for r in runs]), solve_times=st,
                        target=runs[0]["target"])
    plot(cfg, runs, st, RESULTS / "e1_landing.png")
    if args.video:
        render(args.config, args.video_seeds, args.video, args.slowmo)
        print(f"video: {args.video}")


def plot(cfg, runs, st, path):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    tc = cfg["table"]
    x0, L, W = tc["near_edge_x"], tc["length"], tc["width"]
    fig, ax = plt.subplots(1, 2, figsize=(13, 5))
    a = ax[0]
    a.add_patch(plt.Rectangle((x0, -W / 2), L, W, fill=False, color="b"))
    a.axvline(x0 + L / 2, color="k", lw=2, label="net")
    lp = np.array([r["landing"] for r in runs if r["outcome"] == "in"])
    if len(lp):
        a.plot(lp[:, 0], lp[:, 1], ".", label="landing points")
    t = runs[0]["target"]
    a.plot(t[0], t[1], "r*", ms=14, label="target")
    a.set(xlabel="x [m]", ylabel="y [m]", aspect="equal", title="E1 landing points (opponent half right of the net)")
    a.legend(fontsize=8)
    a = ax[1]
    a.hist(1e3 * st, bins=40)
    a.set(xlabel="MPC solve time [ms]", ylabel="count", title=f"MPC solve times ({len(st)} solves)")
    for a in ax:
        a.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=110)


if __name__ == "__main__":
    main()
