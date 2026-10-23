"""Phases 5–6: juggle the ball on the paddle (single run).

Phase 5 (--true-state): the juggler sees the simulator's ball state. Acceptance (steady state, after
juggle.transient_bounces): >= max_bounces bounces, |apex error| < max_apex_err, impacts within max_lateral.
Phase 6 (default, sensing.ball_source = estimate): noisy 200 Hz position measurements -> EKF; additionally
reports steady-state estimation RMSE, impact-point prediction error vs time-to-impact, bounce detection.

Usage:
    python scripts/juggle.py --true-state --tag nominal --video results/juggle_nominal.mp4     # Phase 5
    python scripts/juggle.py --tag noisy --video results/juggle_noisy.mp4                      # Phase 6, sigma = 2 mm
    options: --seed N --max-bounces N --noise SIGMA_M --latency S --dropout P --video-duration S --slowmo X --viewer
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tt.config import RESULTS, load_config  # noqa: E402
from tt.metrics import estimator_metrics, juggle_metrics  # noqa: E402
from tt.sim import run_juggle  # noqa: E402


def plot_estimator(run, em, cfg, path: Path, title: str) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    L, t = run.logs, run.logs["t"]
    fig, ax = plt.subplots(2, 2, figsize=(12, 8))
    sel = t <= min(t[-1], 2.5)
    a = ax[0, 0]
    a.plot(t[sel], L["ball_p"][sel, 2], "k-", lw=2, alpha=0.3, label="true z")
    a.plot(t[sel], L["meas"][sel, 2], ".", ms=2, label="measured z")
    a.plot(t[sel], L["est_p"][sel, 2], "-", lw=1, label="EKF z")
    a.set(xlabel="time [s]", ylabel="z [m]", title="ball height: truth, measurements, estimate")
    a.legend(fontsize=8)
    a = ax[0, 1]
    a.plot(t[sel], 1e3 * (L["est_v"][sel] - L["ball_v"][sel]))
    a.set(xlabel="time [s]", ylabel="velocity error [mm/s]", title="EKF velocity error (x, y, z)", ylim=(-300, 300))
    a.legend(["x", "y", "z"], fontsize=8)
    a = ax[1, 0]
    c = em["tti_centers"]
    a.plot(1e3 * c, 1e3 * em["pred_err_median"], "o-", label="median")
    a.plot(1e3 * c, 1e3 * em["pred_err_p90"], "s--", label="90th percentile")
    a.axvline(1e3 * cfg["estimator"]["report_time_to_impact"], color="k", ls=":", lw=1)
    a.set(xlabel="time to impact [ms]", ylabel="predicted impact point error (xy) [mm]",
          title="impact-point prediction error vs time to impact")
    a.legend(fontsize=8)
    a = ax[1, 1]
    if len(em["detect_delays"]):
        a.hist(1e3 * em["detect_delays"], bins=20)
    a.set(xlabel="bounce detection delay [ms]", ylabel="count",
          title=f"bounce detection: {em['missed']} missed, {em['false_detections']} false")
    for a in ax.flat:
        a.grid(alpha=0.3)
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(path, dpi=110)


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
    ap.add_argument("--true-state", action="store_true", help="juggle from the simulator's ball state (Phase 5)")
    ap.add_argument("--noise", type=float, default=None, help="override sensing.pos_noise_std [m]")
    ap.add_argument("--latency", type=float, default=None, help="override sensing.latency [s]")
    ap.add_argument("--dropout", type=float, default=None, help="override sensing.dropout_prob")
    ap.add_argument("--tag", default=None, help="output name: results/juggle_<tag>.{npz,png} (default nominal/noisy)")
    args = ap.parse_args()
    np.set_printoptions(precision=3, suppress=True)
    cfg = load_config(args.config)
    jc, sc = cfg["juggle"], cfg["sensing"]
    if args.true_state:
        sc["ball_source"] = "true"
    for key, val in (("pos_noise_std", args.noise), ("latency", args.latency), ("dropout_prob", args.dropout)):
        if val is not None:
            sc[key] = val
    est = sc["ball_source"] == "estimate"
    args.tag = args.tag or ("noisy" if est else "nominal")
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

    print(f"ball source: {'EKF estimate' if est else 'true state'}"
          + (f" (sigma {1e3 * sc['pos_noise_std']:.1f} mm, {sc['rate']:.0f} Hz, latency {1e3 * sc['latency']:.0f} ms, "
             f"dropout {100 * sc['dropout_prob']:.0f}%)" if est else ""))
    print(f"seed {seed}: {mt['bounces']} bounces ({'reached max' if run.failure is None else run.failure}), "
          f"sim time {run.logs['t'][-1]:.1f} s")
    if est:
        ec = cfg["estimator"]
        bins = np.arange(ec["tti_bin"] / 2, 0.6 + 1e-9, ec["tti_bin"])     # bin centres at multiples of tti_bin
        em = estimator_metrics(run, cfg, bins)
        i_rep = int(np.argmin(np.abs(em["tti_centers"] - ec["report_time_to_impact"])))
        print(f"  EKF steady state (> {1e3 * ec['settle_after_bounce']:.0f} ms after impacts): velocity RMSE per axis "
              f"{1e3 * em['vel_rmse']} mm/s (|.| {1e3 * em['vel_rmse_norm']:.1f}), position RMSE {1e3 * em['pos_rmse']} mm")
        print(f"  impact-point prediction error at {1e3 * em['tti_centers'][i_rep]:.0f} ms before impact: median "
              f"{1e3 * em['pred_err_median'][i_rep]:.2f} mm, 90% {1e3 * em['pred_err_p90'][i_rep]:.2f} mm; "
              f"impact-time error median {1e3 * em['pred_time_err_median'][i_rep]:.2f} ms")
        print(f"  bounce detection: delay median {1e3 * np.median(em['detect_delays']):.1f} ms, max "
              f"{1e3 * em['detect_delays'].max():.1f} ms; missed {em['missed']}, false {em['false_detections']}")
        plot_estimator(run, em, cfg, out.with_name(out.name + "_estimator.png"),
                       f"EKF ({args.tag}, sigma {1e3 * sc['pos_noise_std']:.1f} mm, seed {seed})")
        np.savez_compressed(out.with_name(out.name + "_estimator.npz"), tti=em["tti_centers"],
                            pred_err_median=em["pred_err_median"], pred_err_p90=em["pred_err_p90"],
                            pred_rows=em["pred_rows"], detect_delays=em["detect_delays"])
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
