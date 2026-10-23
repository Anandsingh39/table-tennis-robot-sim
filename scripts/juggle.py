"""Phases 5–6: juggle the ball on the paddle (single run).

Phase 5 uses the true ball state. Acceptance (steady state, after juggle.transient_bounces):
>= max_bounces consecutive bounces, |apex error| < max_apex_err, impacts within max_lateral of the centre.

Usage:
    python scripts/juggle.py [--seed 0] [--max-bounces 200] [--video results/juggle_nominal.mp4 --video-duration 20]
                             [--slowmo 1] [--snapshot PNG] [--viewer]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tt.config import RESULTS, load_config  # noqa: E402
from tt.metrics import juggle_metrics  # noqa: E402
from tt.sim import run_juggle  # noqa: E402


def plot(run, mt, cfg, path: Path, title: str) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    jc = cfg["juggle"]
    fig, ax = plt.subplots(2, 2, figsize=(12, 8))
    a = ax[0, 0]
    t, L = run.logs["t"], run.logs
    t_end = min(t[-1], 6.0)
    sel = t <= t_end
    a.plot(t[sel], L["ball_p"][sel, 2], label="ball centre z")
    a.plot(t[sel], L["paddle_p"][sel, 2], label="paddle centre z")
    a.axhline(run.z_impact + jc["apex_height"], color="r", ls=":", lw=1, label="target apex")
    a.axhline(run.z_impact, color="k", ls=":", lw=1, label="impact height")
    a.set(xlabel="time [s]", ylabel="z [m]", title=f"first {t_end:.0f} s")
    a.legend(fontsize=8)
    a = ax[0, 1]
    k = np.arange(1, len(mt["apex"]) + 1)
    a.plot(k, 1e3 * mt["apex_err"], ".-")
    a.axhspan(-1e3 * jc["max_apex_err"], 1e3 * jc["max_apex_err"], color="g", alpha=0.15, label="±1 cm")
    a.axvline(jc["transient_bounces"] + 0.5, color="k", ls=":", lw=1)
    a.set(xlabel="bounce #", ylabel="apex error [mm]", title=f"apex height error (target {jc['apex_height']} m)")
    a.legend(fontsize=8)
    a = ax[1, 0]
    xy = 1e3 * mt["impact_xy"]
    a.plot(xy[:, 0], xy[:, 1], "-", color="0.8", lw=0.8)
    sc = a.scatter(xy[:, 0], xy[:, 1], c=np.arange(len(xy)), s=10, cmap="viridis")
    fig.colorbar(sc, ax=a, label="bounce #")
    r = 1e3 * jc["max_lateral"]
    a.add_patch(plt.Circle((0, 0), r, fill=False, color="r", ls="--"))
    a.set(xlabel="x - c [mm]", ylabel="y - c [mm]", title="impact points relative to the centre", aspect="equal")
    a = ax[1, 1]
    a.plot(np.arange(1, len(mt["timing"]) + 1), 1e3 * mt["timing"], ".-")
    a.set(xlabel="bounce #", ylabel="actual - planned impact time [ms]", title="impact timing")
    for a in ax.flat:
        a.grid(alpha=0.3)
    fig.suptitle(title)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=110)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", type=Path, default=None)
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--max-bounces", type=int, default=None)
    ap.add_argument("--video", type=Path, default=None)
    ap.add_argument("--video-duration", type=float, default=20.0, help="seconds of sim time to record")
    ap.add_argument("--slowmo", type=float, default=1.0)
    ap.add_argument("--snapshot", type=Path, default=None)
    ap.add_argument("--viewer", action="store_true")
    ap.add_argument("--tag", default="nominal", help="output name: results/juggle_<tag>.{npz,png}")
    args = ap.parse_args()
    np.set_printoptions(precision=3, suppress=True)
    cfg = load_config(args.config)
    jc = cfg["juggle"]
    seed = jc["seed"] if args.seed is None else args.seed
    max_b = args.max_bounces or jc["max_bounces"]

    run = run_juggle(cfg, seed, max_b, video=args.video, video_duration=args.video_duration, viewer=args.viewer,
                     slowmo=args.slowmo)
    mt = juggle_metrics(run, cfg)
    out = RESULTS / f"juggle_{args.tag}"
    np.savez_compressed(out.with_suffix(".npz"), **run.logs, impact_t=[e.t for e in run.events],
                        impact_pos=np.array([e.pos for e in run.events]), apex=mt["apex"], timing=mt["timing"])
    plot(run, mt, cfg, out.with_suffix(".png"), f"juggling ({args.tag}, seed {seed}): {mt['bounces']} bounces, "
         f"{'stopped at max' if run.failure is None else 'failure: ' + run.failure}")
    if args.snapshot:
        print("(snapshot: use --video; a still of the start is the first video frame)")

    print(f"seed {seed}: {mt['bounces']} bounces ({'reached max' if run.failure is None else run.failure}), "
          f"sim time {run.logs['t'][-1]:.1f} s")
    print(f"steady state (bounce > {jc['transient_bounces']}): apex error mean {1e3 * mt['apex_err_mean']:+.2f} mm, "
          f"std {1e3 * mt['apex_err_std']:.2f} mm, max |.| {1e3 * mt['apex_err_max']:.2f} mm")
    print(f"  impact distance from centre: RMS {1e3 * mt['lateral_rms']:.2f} mm, max {1e3 * mt['lateral_max']:.2f} mm")
    print(f"  impact timing (actual - planned): RMS {1e3 * mt['timing_rms']:.3f} ms")
    print(f"  peak torque per joint [% of limit]: {100 * mt['peak_tau_frac']}  saturated: {mt['saturated']}")
    checks = {
        f">= {max_b} bounces": mt["bounces"] >= max_b,
        f"apex error < {1e3 * jc['max_apex_err']:.0f} mm": mt["apex_err_max"] < jc["max_apex_err"],
        f"impacts within {1e3 * jc['max_lateral']:.0f} mm of centre": mt["lateral_max"] < jc["max_lateral"],
    }
    for k, ok in checks.items():
        print(f"  [{'PASS' if ok else 'FAIL'}] {k}")
    print(f"log: {out.with_suffix('.npz')}\nplot: {out.with_suffix('.png')}" + (f"\nvideo: {args.video}" if args.video else ""))
    sys.exit(0 if all(checks.values()) else 1)


if __name__ == "__main__":
    main()
