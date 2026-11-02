"""Extra credit E2: two OpenArms rally (cooperative lobs, MPC swings).

Runs e2.rallies seeded rallies (serve = launcher ball to robot A) in parallel, up to e2.max_hits hits each,
and reports rally length (hits), how rallies ended, shot statistics (bounce point past the net, crossing height
at the receiver, paddle tilt/speed), MPC solve times, torque usage. Writes results/e2_summary.md and
results/e2_rally.png; --video renders one rally.

Usage:
    python scripts/two_robot_rally.py [--rallies 20] [--video results/e2_rally.mp4 --video-seed 0 --video-hits 12]
"""

from __future__ import annotations

import os

os.environ.setdefault("OMP_NUM_THREADS", "1")

import argparse  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402
from concurrent.futures import ProcessPoolExecutor  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tt.config import REPO, RESULTS, load_config  # noqa: E402
from tt.mpc import solve_stat_rows  # noqa: E402

RALLY2_CONFIG = REPO / "config" / "rally2.yaml"


def rally(args):
    cfg_path, seed, max_hits = args
    from tt.rally2 import run_rally

    cfg = load_config(cfg_path)
    t0 = time.time()
    r = run_rally(cfg, seed, max_hits)
    x_mid = cfg["table"]["near_edge_x"] + cfg["table"]["length"] / 2
    st = np.concatenate([np.asarray(v, float) for v in r.solve_times.values()])
    L = r.logs
    spin_pairs = []                                  # (commanded top(+)/back(-)spin, achieved, along the topspin axis)
    for who, e in r.hits:
        sh = [x for x in r.shots[who] if abs(x.t_hit - e.t) < 0.05]
        if sh:
            d = np.array([e.v_out[0], e.v_out[1], 0.0])
            axis = np.cross([0.0, 0.0, 1.0], d / np.linalg.norm(d))
            spin_pairs.append((sh[-1].spin_cmd, float(e.w_out @ axis)))
    return dict(seed=seed, hits=len(r.hits), fault=r.fault, by=r.fault_by, wall=time.time() - t0,
                spin_pairs=spin_pairs, downgrades=sum((r.spin_downgrades or {}).values()),
                hit_y=[e.pos[1] for _, e in r.hits],
                solve_log=[x for k in sorted(r.solve_logs or {}) for x in r.solve_logs[k]],
                rail_range=[float(L[f"q_{k}"][:, 0].min()) for k in "AB"] + [float(L[f"q_{k}"][:, 0].max()) for k in "AB"]
                if cfg["robot"].get("rail", {}).get("enabled", False) else None,
                sim_time=float(L["t"][-1]),
                bounce_dists=[abs(b.pos[0] - x_mid) for b in r.bounces],
                hit_z=[e.pos[2] for _, e in r.hits], solve_times=st,
                tilts=[s.tilt_off_deg for k in r.shots for s in r.shots[k]],
                speeds=[s.speed for k in r.shots for s in r.shots[k]],
                peak_tau=np.max([np.abs(L[f"tau_{k}"]).max(axis=0) for k in "AB"], axis=0) / r.tau_max,
                saturated=bool(L["sat_A"].any() or L["sat_B"].any()),
                min_margin=float(min(np.min(np.minimum(L[f"q_{k}"] - r.q_lo, r.q_hi - L[f"q_{k}"])) for k in "AB")))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", type=Path, default=RALLY2_CONFIG)
    ap.add_argument("--rallies", type=int, default=None)
    ap.add_argument("--max-hits", type=int, default=None)
    ap.add_argument("--workers", type=int, default=None)
    ap.add_argument("--video", type=Path, default=None)
    ap.add_argument("--video-seed", type=int, default=0)
    ap.add_argument("--video-hits", type=int, default=12)
    ap.add_argument("--slowmo", type=float, default=1.0)
    ap.add_argument("--skip-stats", action="store_true", help="only render the video")
    ap.add_argument("--tag", default="e2", help="output names results/<tag>_summary.md, <tag>_rally.png")
    args = ap.parse_args()
    cfg = load_config(args.config)
    ec = cfg["e2"]
    if not args.skip_stats:
        n, cap = args.rallies or ec["rallies"], args.max_hits or ec["max_hits"]
        t0 = time.time()
        with ProcessPoolExecutor(args.workers or ec["workers"]) as ex:
            runs = list(ex.map(rally, [(args.config, s, cap) for s in range(n)]))
        print(f"{n} rallies in {time.time() - t0:.0f} s")
        hits = np.array([r["hits"] for r in runs])
        ends = {}
        for r in runs:
            key = r["fault"] + (f" ({r['by']})" if r["by"] else "")
            ends[key] = ends.get(key, 0) + 1
        bd = np.concatenate([r["bounce_dists"] for r in runs])
        hz = np.concatenate([r["hit_z"] for r in runs])
        st = np.concatenate([r["solve_times"] for r in runs])
        rows = [
            ("rallies", n), ("hit cap per rally", cap),
            ("rally length (hits): mean / median / min / max", f"{hits.mean():.1f} / {np.median(hits):.0f} / {hits.min()} / {hits.max()}"),
            ("rallies reaching the cap", f"{(hits >= cap).sum()} / {n}"),
            ("how rallies ended", ", ".join(f"{k}: {v}" for k, v in sorted(ends.items()))),
            ("total hits simulated", int(hits.sum())),
            ("bounce distance past the net: mean ± std [m]", f"{bd.mean():.3f} ± {bd.std():.3f}"),
            ("ball height at impact: mean ± std [m]", f"{hz.mean():.3f} ± {hz.std():.3f}"),
            ("lateral hit position y: min / max / std [m]", "{:.2f} / {:.2f} / {:.2f}".format(
                *(lambda y: (y.min(), y.max(), y.std()))(np.concatenate([r["hit_y"] for r in runs])))),
            *spin_rows(runs),
            *([("spin shots played flat by the reachable-spin gate", str(sum(r["downgrades"] for r in runs)))]
              if any(r["downgrades"] for r in runs) else []),
            ("rail travel used: min / max [m]", "–" if runs[0]["rail_range"] is None else
             f"{min(min(r['rail_range'][:2]) for r in runs):.2f} / {max(max(r['rail_range'][2:]) for r in runs):.2f}"),
            ("paddle normal vs nominal: median / max [deg]", f"{np.median(np.concatenate([r['tilts'] for r in runs])):.1f} / {np.max(np.concatenate([r['tilts'] for r in runs])):.1f}"),
            ("paddle normal speed: median [m/s]", f"{np.median(np.concatenate([r['speeds'] for r in runs])):.2f}"),
            *solve_stat_rows([x for r in runs for x in r["solve_log"]]),
            ("MPC solve time: median / p95 / max [ms]", f"{1e3 * np.median(st):.0f} / {1e3 * np.percentile(st, 95):.0f} / {1e3 * st.max():.0f}"),
            ("peak torque, worst rally [% of limit] per joint", str(np.round(100 * np.max([r["peak_tau"] for r in runs], axis=0)).astype(int))),
            ("rallies with torque saturation", str(sum(r["saturated"] for r in runs))),
            ("min joint-limit margin [rad]", f"{min(r['min_margin'] for r in runs):.3f}"),
            ("simulated time / wall time per rally (mean) [s]", f"{np.mean([r['sim_time'] for r in runs]):.1f} / {np.mean([r['wall'] for r in runs]):.0f}"),
        ]
        md = [f"# {args.tag}: two-robot cooperative rally", "",
              f"Config: {cfg_label(args.config)} (two OpenArms, pre-tilted paddle mounts, cooperative lob shots, MPC swings, "
              "true ball state). Serve: launcher ball to robot A. Legal hit = after exactly one bounce on the receiver's half.", "",
              "| metric | value |", "|---|---|"] + [f"| {a} | {b} |" for a, b in rows]
        (RESULTS / f"{args.tag}_summary.md").write_text("\n".join(md) + "\n")
        print("\n".join(md))
        plot(runs, cap, RESULTS / f"{args.tag}_rally.png")
    if args.video:
        from tt.rally2 import run_rally

        r = run_rally(cfg, args.video_seed, args.video_hits, video=args.video, slowmo=args.slowmo)
        print(f"video: {args.video} ({len(r.hits)} hits, ended by '{r.fault}')")


def cfg_label(path: Path) -> str:
    """Config path relative to the repo, or absolute for a config kept elsewhere (e.g. a comparison run)."""
    p = path.resolve()
    return str(p.relative_to(REPO)) if p.is_relative_to(REPO) else str(p)


def spin_rows(runs):
    pairs = np.array([p for r in runs for p in r["spin_pairs"]]).reshape(-1, 2)
    if not len(pairs) or not np.any(pairs[:, 0]):
        return []
    rows = []
    for c in sorted(set(pairs[:, 0])):
        sel = pairs[:, 0] == c
        kind = "topspin" if c > 0 else ("backspin" if c < 0 else "flat")
        rows.append((f"{kind} shots (commanded {c:+.0f} rad/s): count, achieved mean ± std [rad/s]",
                     f"{sel.sum()}, {pairs[sel, 1].mean():+.1f} ± {pairs[sel, 1].std():.1f}"))
    return rows


def plot(runs, cap, path):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(1, 3, figsize=(16, 4.5))
    hits = [r["hits"] for r in runs]
    ax[0].hist(hits, bins=np.arange(0, cap + 2) - 0.5)
    ax[0].set(xlabel="hits in the rally", ylabel="rallies", title=f"Rally length ({len(runs)} rallies, cap {cap})")
    ax[1].hist(np.concatenate([r["bounce_dists"] for r in runs]), bins=40)
    ax[1].set(xlabel="bounce distance past the net [m]", ylabel="bounces", title="Where the ball bounces")
    ax[2].hist(1e3 * np.concatenate([r["solve_times"] for r in runs]), bins=40)
    ax[2].set(xlabel="MPC solve time [ms]", ylabel="solves", title="MPC solve times")
    for a in ax:
        a.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=110)


if __name__ == "__main__":
    main()
