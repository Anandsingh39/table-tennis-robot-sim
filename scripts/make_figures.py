"""Phase 7: figures and tables from the Monte Carlo results (results/mc/*/seed_*.npz).

Outputs:
  results/mc_survival.png      fraction of runs still juggling after k bounces, per condition
  results/mc_apex_hist.png     steady-state apex-height error histograms (noise sweep)
  results/mc_dispersion.png    steady-state impact points relative to the centre (noise sweep)
  results/mc_vs_sigma.png      metrics vs measurement noise sigma
  results/mc_summary.md/.csv   per-condition table
  results/replication_table.md our numbers next to the papers' reported results (config/literature.yaml)

Usage:
    python scripts/make_figures.py
"""

from __future__ import annotations

import csv
import sys
from pathlib import Path

import numpy as np
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tt.config import REPO, RESULTS, load_config  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
from monte_carlo import MC_DIR, apply_overrides, slug  # noqa: E402

NOISE_CONDS = ["sigma 0 mm", "sigma 1 mm", "sigma 2 mm", "sigma 5 mm", "sigma 10 mm"]


def load_condition(name: str) -> list[dict]:
    runs = []
    for f in sorted((MC_DIR / slug(name)).glob("seed_*.npz")):
        with np.load(f, allow_pickle=False) as d:
            runs.append({k: d[k] for k in d.files})
    return runs


def summarise(name: str, runs: list[dict], cfg: dict) -> dict:
    jc = cfg["juggle"]
    s = jc["transient_bounces"]
    cap = cfg["monte_carlo"]["max_bounces"]
    bounces = np.array([int(r["bounces"]) for r in runs])
    apex_err = np.concatenate([r["apex_err"][s:] for r in runs]) if runs else np.array([])
    lateral = np.concatenate([r["lateral"][s:] for r in runs]) if runs else np.array([])
    timing = np.concatenate([r["timing"][s:] for r in runs]) if runs else np.array([])
    out = dict(condition=name, runs=len(runs), success_rate=float(np.mean(bounces >= cap)) if runs else np.nan,
               bounces_mean=float(np.mean(bounces)) if runs else np.nan,
               bounces_min=int(bounces.min()) if runs else 0,
               apex_err_mean_mm=1e3 * float(np.mean(apex_err)) if apex_err.size else np.nan,
               apex_err_std_mm=1e3 * float(np.std(apex_err)) if apex_err.size else np.nan,
               apex_abs_p95_mm=1e3 * float(np.percentile(np.abs(apex_err), 95)) if apex_err.size else np.nan,
               lateral_rms_mm=1e3 * float(np.sqrt(np.mean(lateral ** 2))) if lateral.size else np.nan,
               lateral_p95_mm=1e3 * float(np.percentile(lateral, 95)) if lateral.size else np.nan,
               timing_rms_ms=1e3 * float(np.sqrt(np.mean(timing ** 2))) if timing.size else np.nan,
               peak_tau_pct=100 * float(max(r["peak_tau_frac"].max() for r in runs)) if runs else np.nan,
               any_saturation=bool(any(bool(r["saturated"]) for r in runs)),
               failures=", ".join(sorted({str(r["failure"]) for r in runs if str(r["failure"])})) or "-")
    if runs and "vel_rmse_norm" in runs[0]:
        out.update(vel_rmse_mm_s=1e3 * float(np.mean([r["vel_rmse_norm"] for r in runs])) / np.sqrt(3),
                   pred_err_100ms_mm=1e3 * float(np.nanmedian([r["pred_err_at_report"] for r in runs])),
                   missed=int(sum(int(r["missed"]) for r in runs)),
                   false_detections=int(sum(int(r["false_detections"]) for r in runs)))
    return out


def fig_survival(conds, data, cap, path):
    import matplotlib.pyplot as plt

    groups = [("noise / latency", [c for c in conds if "e_r" not in c and "C_d" not in c and "PD" not in c]),
              ("model mismatch (nominal e_r 0.8, C_d 0.47)", ["sigma 2 mm"] + [c for c in conds if "e_r" in c or "C_d" in c]),
              ("juggling law", ["sigma 2 mm", "sigma 10 mm"] + [c for c in conds if "PD" in c])]
    fig, ax = plt.subplots(1, len(groups), figsize=(19, 5), sharey=True)
    k = np.arange(0, cap + 1)
    for a, (title, names) in zip(ax, groups):
        for name in names:
            b = np.array([int(r["bounces"]) for r in data[name]])
            if b.size:
                a.step(k, [(b >= kk).mean() for kk in k], where="post", label=f"{name} (n={b.size})")
        a.set(xlabel="bounce k", title=title, ylim=(-0.02, 1.05))
        a.grid(alpha=0.3)
        a.legend(fontsize=7)
    ax[0].set_ylabel("fraction of runs with >= k bounces")
    fig.suptitle(f"Survival: consecutive bounces (cap {cap})")
    fig.tight_layout()
    fig.savefig(path, dpi=110)


def fig_apex_hist(data, cfg, path):
    import matplotlib.pyplot as plt

    s = cfg["juggle"]["transient_bounces"]
    fig, ax = plt.subplots(figsize=(8, 5))
    for name in ["true state"] + NOISE_CONDS:
        e = np.concatenate([r["apex_err"][s:] for r in data[name]]) if data[name] else np.array([])
        if e.size:
            ax.hist(1e3 * e, bins=np.linspace(-40, 40, 161), histtype="step", density=True, label=name)
    ax.axvspan(-1e3 * cfg["juggle"]["max_apex_err"], 1e3 * cfg["juggle"]["max_apex_err"], color="g", alpha=0.1)
    ax.set(xlabel="apex height error [mm]", ylabel="density", title="Steady-state apex height error (all runs)")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=110)


def fig_dispersion(data, cfg, path):
    import matplotlib.pyplot as plt

    s = cfg["juggle"]["transient_bounces"]
    names = ["true state"] + NOISE_CONDS
    fig, ax = plt.subplots(1, len(names), figsize=(3.2 * len(names), 3.6), sharex=True, sharey=True)
    lim = 1e3 * cfg["juggle"]["fail_radius"] / 2
    for a, name in zip(ax, names):
        xy = np.concatenate([r["impact_xy"][s:] for r in data[name]]) if data[name] else np.zeros((0, 2))
        a.plot(1e3 * xy[:, 0], 1e3 * xy[:, 1], ".", ms=1, alpha=0.3)
        a.add_patch(plt.Circle((0, 0), 1e3 * cfg["juggle"]["max_lateral"], fill=False, color="r", ls="--"))
        rms = 1e3 * np.sqrt(np.mean(np.sum(xy ** 2, axis=1))) if len(xy) else np.nan
        a.set(title=f"{name}\nRMS {rms:.1f} mm", xlabel="x - c [mm]", xlim=(-lim, lim), ylim=(-lim, lim), aspect="equal")
        a.grid(alpha=0.3)
    ax[0].set_ylabel("y - c [mm]")
    fig.suptitle("Impact points relative to the juggling centre (steady state, all runs; dashed = ±3 cm)")
    fig.tight_layout()
    fig.savefig(path, dpi=110)


def fig_vs_sigma(rows, path):
    import matplotlib.pyplot as plt

    r = {x["condition"]: x for x in rows}
    sig = [0, 1, 2, 5, 10]
    names = NOISE_CONDS
    panels = [("success_rate", "success rate (200 bounces)"), ("apex_err_std_mm", "apex error std [mm]"),
              ("lateral_rms_mm", "impact RMS from centre [mm]"), ("vel_rmse_mm_s", "EKF velocity RMSE per axis [mm/s]"),
              ("pred_err_100ms_mm", "impact-point prediction error\n100 ms before impact [mm]")]
    fig, ax = plt.subplots(1, len(panels), figsize=(4 * len(panels), 3.8))
    for a, (key, label) in zip(ax, panels):
        a.plot(sig, [r[n].get(key, np.nan) for n in names], "o-")
        if key in r.get("true state", {}):
            a.axhline(r["true state"][key], color="k", ls=":", lw=1, label="true state")
            a.legend(fontsize=8)
        a.set(xlabel="measurement noise sigma [mm]", ylabel=label)
        a.grid(alpha=0.3)
    fig.suptitle("Juggling performance vs measurement noise (50 seeds per point)")
    fig.tight_layout()
    fig.savefig(path, dpi=110)


def fmt(v, nd=2):
    if isinstance(v, float):
        return "–" if np.isnan(v) else f"{v:.{nd}f}"
    return str(v)


def write_tables(rows, cfg, lit):
    cols = ["condition", "runs", "success_rate", "bounces_mean", "bounces_min", "apex_err_mean_mm", "apex_err_std_mm",
            "apex_abs_p95_mm", "lateral_rms_mm", "lateral_p95_mm", "timing_rms_ms", "peak_tau_pct", "vel_rmse_mm_s",
            "pred_err_100ms_mm", "missed", "false_detections", "failures"]
    with open(RESULTS / "mc_summary.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)
    hdr = ["condition", "runs", "success", "mean bounces", "apex err mean ± std [mm]", "|apex err| p95 [mm]",
           "impact RMS / p95 [mm]", "timing RMS [ms]", "peak τ [%]", "EKF v RMSE [mm/s/axis]",
           "pred. err @100 ms [mm]", "missed / false bounce det.", "failure modes"]
    lines = ["# Monte Carlo summary (Phase 7)", "",
             f"{cfg['monte_carlo']['n_seeds']} seeds per condition, cap {cfg['monte_carlo']['max_bounces']} bounces. "
             f"Steady state = bounces after the first {cfg['juggle']['transient_bounces']}. Baseline: sigma 2 mm, 200 Hz, "
             "no latency, nominal = true model.", "",
             "| " + " | ".join(hdr) + " |", "|" + "---|" * len(hdr)]
    for r in rows:
        lines.append("| " + " | ".join([
            r["condition"], str(r["runs"]), f"{100 * r['success_rate']:.0f}%", fmt(r["bounces_mean"], 1),
            f"{fmt(r['apex_err_mean_mm'])} ± {fmt(r['apex_err_std_mm'])}", fmt(r["apex_abs_p95_mm"]),
            f"{fmt(r['lateral_rms_mm'], 1)} / {fmt(r['lateral_p95_mm'], 1)}", fmt(r["timing_rms_ms"], 3),
            fmt(r["peak_tau_pct"], 0), fmt(r.get("vel_rmse_mm_s", np.nan), 1), fmt(r.get("pred_err_100ms_mm", np.nan)),
            f"{r.get('missed', '–')} / {r.get('false_detections', '–')}", r["failures"]]) + " |")
    (RESULTS / "mc_summary.md").write_text("\n".join(lines) + "\n")

    # replication table
    rb = {x["condition"]: x for x in rows}.get("sigma 2 mm", {})
    rt = {x["condition"]: x for x in rows}.get("true state", {})
    jc, sc = cfg["juggle"], cfg["sensing"]
    n, j = lit["nakashima2008"], lit["ji2024"]
    ours_pos = (f"impact RMS {fmt(rb.get('lateral_rms_mm', np.nan), 1)} mm, p95 {fmt(rb.get('lateral_p95_mm', np.nan), 1)} mm "
                f"(true state: {fmt(rt.get('lateral_rms_mm', np.nan), 1)} mm)")
    ours_h = (f"apex error {fmt(rb.get('apex_err_mean_mm', np.nan))} ± {fmt(rb.get('apex_err_std_mm', np.nan))} mm, "
              f"|err| p95 {fmt(rb.get('apex_abs_p95_mm', np.nan))} mm = "
              f"{fmt(rb.get('apex_abs_p95_mm', np.nan) / (10 * jc['apex_height']), 2)}% of the {100 * jc['apex_height']:.0f} cm target")
    ours_dur = (f"{100 * rb.get('success_rate', np.nan):.0f}% of {rb.get('runs', 0)} runs reached the "
                f"{cfg['monte_carlo']['max_bounces']}-bounce cap (~{cfg['monte_carlo']['max_bounces'] * 0.57:.0f} s each)")
    table = [
        ("Platform", "sim: OpenArm v2 (7 DOF, torque control) in MuJoCo", n["robot"], j["robot"]),
        ("Paddle", "disk r = 75 mm, 0.15 kg", n["paddle"], j["paddle"]),
        ("Ball sensing", f"sim: position + N(0, ({1e3 * sc['pos_noise_std']:.0f} mm)²) at {sc['rate']:.0f} Hz, EKF", n["sensing"], j["sensing"]),
        ("Ball / impact model", "gravity + quadratic drag; impulse, e = 0.8 (assumed)", n["ball_model"], j["ball_model"]),
        ("Control law", "impact-targeting (mirror-law style) + IK + quintic + computed torque", n["control_law"], j["control_law"]),
        ("Target", f"apex {100 * jc['apex_height']:.0f} cm above the hit plane; centre of the hit plane", n["target"], j["target"]),
        ("Duration / length", ours_dur, n["duration"], j["duration"]),
        ("Hitting-position error", ours_pos, n["position_error"], j["position_error"]),
        ("Height error", ours_h, n["height_error"], j["height_error"]),
    ]
    out = ["# Replication table (Phase 7)", "",
           "Our numbers: Monte Carlo baseline (sigma 2 mm). Paper numbers are quoted from the papers (config/literature.yaml).",
           "Differences to keep in mind: the papers are real robots with vision (latency, calibration error, ball spin,",
           "unknown restitution); ours is a simulation with a specified Gaussian noise model and a known (or deliberately",
           "mismatched) ball model.", "",
           "| | This project (sim) | Nakashima et al. 2008 [3] | Ji et al. 2024 [7] |", "|---|---|---|---|"]
    out += [f"| {a} | {b} | {c} | {d} |" for a, b, c, d in table]
    out += ["", "Sources:", f"- [3] {n['cite']}", f"- [7] {j['cite']}"]
    (RESULTS / "replication_table.md").write_text("\n".join(out) + "\n")


def main() -> None:
    import matplotlib

    matplotlib.use("Agg")
    cfg = load_config()
    lit = yaml.safe_load(open(REPO / "config" / "literature.yaml"))
    conds = [c["name"] for c in cfg["monte_carlo"]["conditions"]]
    data = {c: load_condition(c) for c in conds}
    rows = [summarise(c, data[c], apply_overrides(cfg, next(x for x in cfg["monte_carlo"]["conditions"]
                                                            if x["name"] == c)["overrides"])) for c in conds]
    cap = cfg["monte_carlo"]["max_bounces"]
    fig_survival(conds, data, cap, RESULTS / "mc_survival.png")
    fig_apex_hist(data, cfg, RESULTS / "mc_apex_hist.png")
    fig_dispersion(data, cfg, RESULTS / "mc_dispersion.png")
    fig_vs_sigma(rows, RESULTS / "mc_vs_sigma.png")
    write_tables(rows, cfg, lit)
    print((RESULTS / "mc_summary.md").read_text())
    print("figures: results/mc_survival.png, mc_apex_hist.png, mc_dispersion.png, mc_vs_sigma.png")
    print("tables: results/mc_summary.md/.csv, results/replication_table.md")


if __name__ == "__main__":
    main()
