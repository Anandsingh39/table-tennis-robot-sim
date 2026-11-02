"""RL add-on: train a shot strategy by self-play with the cross-entropy method (evolution strategies, cf. [11]),
then evaluate it in head-to-head matches against the hand-designed cooperative strategy and a random one.

The MPC swing (E1/E2) executes every shot; the strategy only chooses where the ball should go (tt/strategy.py).
Training: mean parameters mu (initialised at the cooperative strategy), std sigma. Each generation, `population`
candidates mu + sigma * eps play `points_per_candidate` points against the CURRENT MEAN policy (half as robot A,
half as robot B; common seeds within a generation). Fitness = (points won - points lost) / points. mu, sigma <-
mean, std of the elite candidates. Outputs in results/rl/.

Usage:
    nohup python scripts/selfplay_es.py > results/rl/train.log 2>&1 &          # train + evaluate
    python scripts/selfplay_es.py --eval-only [--video results/rl_point.mp4]
"""

from __future__ import annotations

import os

os.environ.setdefault("OMP_NUM_THREADS", "1")

import argparse  # noqa: E402
import json  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402
from concurrent.futures import ProcessPoolExecutor  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tt.config import REPO, RESULTS, load_config  # noqa: E402
from tt.mpc import solve_stat_rows  # noqa: E402

CFG = REPO / "config" / "selfplay.yaml"
OUT = RESULTS / "rl"


def _policy(spec, cfg, seed):
    from tt.strategy import make_policy

    kind, theta = spec
    return make_policy(kind, cfg, theta, seed)


def job(args):
    """Play one point; spec_a / spec_b = (kind, theta) for robots A / B."""
    from tt.strategy import play_point

    cfg_path, seed, spec_a, spec_b, max_hits = args
    cfg = load_config(cfg_path)
    t0 = time.time()
    try:
        res = play_point(cfg, seed, _policy(spec_a, cfg, seed + 11), _policy(spec_b, cfg, seed + 13), max_hits)
    except Exception as exc:          # a failed serve solve etc.: count as a void point
        res = dict(seed=seed, winner=None, fault=f"error: {exc}", by=None, hits=0, decisions={})
    res["wall"] = time.time() - t0
    return res


def match(ex, cfg_path, spec_x, spec_y, n_points, seed0, max_hits):
    """X vs Y over n_points (X as A for even points, as B for odd). Returns X's wins, losses, draws, hits."""
    jobs = []
    for j in range(n_points):
        a, b = (spec_x, spec_y) if j % 2 == 0 else (spec_y, spec_x)
        jobs.append((cfg_path, seed0 + j, a, b, max_hits))
    res = list(ex.map(job, jobs))
    w = l = dr = 0
    for j, r in enumerate(res):
        x_side = "A" if j % 2 == 0 else "B"
        if r["winner"] is None:
            dr += 1
        elif r["winner"] == x_side:
            w += 1
        else:
            l += 1
    return w, l, dr, [r["hits"] for r in res], res


def train(ex, cfg, cfg_path):
    from tt.strategy import N_ACT, N_OBS, cooperative_theta

    sp = cfg["selfplay"]
    rng = np.random.default_rng(0)
    mu = cooperative_theta(cfg)
    sigma = np.full(N_ACT * (N_OBS + 1), sp["init_std"])
    hist = []
    for g in range(sp["generations"]):
        t0 = time.time()
        eps = rng.standard_normal((sp["population"], mu.size))
        cands = mu + sigma * eps
        K = sp["points_per_candidate"]
        jobs, owner = [], []
        for i, th in enumerate(cands):
            for j in range(K):
                a, b = (("linear", th), ("linear", mu)) if j % 2 == 0 else (("linear", mu), ("linear", th))
                jobs.append((cfg_path, 100_000 * (g + 1) + j, a, b, sp["max_hits"]))
                owner.append((i, "A" if j % 2 == 0 else "B"))
        res = list(ex.map(job, jobs))
        fit = np.zeros(len(cands))
        for (i, side), r in zip(owner, res):
            if r["winner"] is not None:
                fit[i] += 1 if r["winner"] == side else -1
        fit /= K
        n_el = max(2, int(sp["elite_frac"] * len(cands)))
        elite = np.argsort(fit)[::-1][:n_el]
        mu = cands[elite].mean(axis=0)
        sigma = np.maximum(cands[elite].std(axis=0), sp["min_std"])
        hits = np.mean([r["hits"] for r in res])
        hist.append(dict(gen=g, fit_mean=float(fit.mean()), fit_best=float(fit.max()), fit_elite=float(fit[elite].mean()),
                         sigma=float(sigma.mean()), hits=float(hits), wall=time.time() - t0, mu=mu.tolist()))
        print(f"gen {g:2d}: fitness mean {fit.mean():+.2f} elite {fit[elite].mean():+.2f} best {fit.max():+.2f}, "
              f"sigma {sigma.mean():.3f}, mean hits/point {hits:.1f}, {time.time() - t0:.0f} s", flush=True)
        (OUT / "train_history.json").write_text(json.dumps(hist))
        np.save(OUT / "theta.npy", mu)
    return mu, hist


def evaluate(ex, cfg, cfg_path, theta, video=None, tag="rl"):
    from tt.strategy import cooperative_theta

    sp = cfg["selfplay"]
    n, cap = sp["eval_points"], sp["max_hits"]
    coop, rand, rl = ("cooperative", None), ("random", None), ("linear", theta)
    rows, mpc_rows = [], []
    for name, x, y in (("cooperative vs cooperative (baseline)", coop, coop),
                       ("RL strategy vs cooperative", rl, coop),
                       ("RL strategy vs random", rl, rand),
                       ("random vs cooperative", rand, coop)):
        w, l, dr, hits, res = match(ex, cfg_path, x, y, n, 9_000_000, cap)
        rows.append((name, w, l, dr, np.mean(hits)))
        solves = [e for r in res for log in r.get("solve_logs", {}).values() for e in log]
        mpc_rows.append((name, solve_stat_rows(solves)))
        print(f"{name}: X won {w}, lost {l}, draws (cap {cap} hits) {dr}; mean hits/point {np.mean(hits):.1f}", flush=True)
    from tt.strategy import make_policy

    pol = make_policy("linear", cfg, theta)
    from tt.strategy import action_bounds

    lo, hi = action_bounds(cfg)
    md = ["# RL add-on: self-play shot strategy (CEM / evolution strategies) on top of the MPC rally", "",
          f"Config: {Path(cfg_path).resolve().relative_to(REPO)} (E2 + rail, no spin). {n} points per match, sides alternate, point cap {cap} hits "
          "(draw). The MPC executes every shot; only the shot choice differs.", "",
          "| match (X vs Y) | X wins | X losses | draws | win rate of decided points | mean hits/point |", "|---|---|---|---|---|---|"]
    for name, w, l, dr, h in rows:
        md.append(f"| {name} | {w} | {l} | {dr} | {100 * w / max(w + l, 1):.0f}% | {h:.1f} |")
    md += ["", f"Learned action at a neutral observation (ball centred, rails centred): y_c, z_c, d_b = "
           f"{np.round(pol(np.zeros(len(theta) // 3 - 1)), 3).tolist()} (cooperative: [0, {cfg['strategy']['coop_z']}, "
           f"{cfg['strategy']['coop_d']}]; ranges {lo.tolist()} .. {hi.tolist()})"]
    md += ["", "MPC solves per match (both robots):", ""] + [
        f"- {name}: " + "; ".join(f"{k}: {v}" for k, v in mr) for name, mr in mpc_rows]
    (RESULTS / f"{tag}_summary.md").write_text("\n".join(md) + "\n")
    print("\n".join(md))
    if video:
        from tt.rally2 import run_rally
        from tt.strategy import make_policy as mp

        r = run_rally(cfg, 9_000_000, cap, video=video, strategies={"A": mp("linear", cfg, theta),
                                                                     "B": mp("cooperative", cfg)})
        print(f"video: {video} ({len(r.hits)} hits, '{r.fault}' by {r.fault_by})")


def plot(hist, path):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    g = [h["gen"] for h in hist]
    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    ax[0].plot(g, [h["fit_mean"] for h in hist], "o-", label="population mean")
    ax[0].plot(g, [h["fit_elite"] for h in hist], "s-", label="elite mean")
    ax[0].axhline(0, color="k", lw=0.5)
    ax[0].set(xlabel="generation", ylabel="fitness vs current mean policy", title="Self-play training (CEM)")
    ax[0].legend(fontsize=8)
    ax[1].plot(g, [h["hits"] for h in hist], "o-")
    ax[1].set(xlabel="generation", ylabel="mean hits per point", title="Rally length during training")
    for a in ax:
        a.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=110)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", type=Path, default=CFG)
    ap.add_argument("--eval-only", action="store_true")
    ap.add_argument("--video", type=Path, default=None)
    ap.add_argument("--tag", default="rl", help="evaluation summary results/<tag>_summary.md")
    args = ap.parse_args()
    cfg = load_config(args.config)
    OUT.mkdir(parents=True, exist_ok=True)
    with ProcessPoolExecutor(cfg["selfplay"]["workers"]) as ex:
        if args.eval_only:
            theta = np.load(OUT / "theta.npy")
        else:
            theta, hist = train(ex, cfg, args.config)
            plot(hist, RESULTS / "rl_training.png")
        evaluate(ex, cfg, args.config, theta, args.video, args.tag)


if __name__ == "__main__":
    main()
