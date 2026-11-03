"""Self-play v2, step 4: evaluate a trained PPO strategy in the FULL rally (MuJoCo + MPC, CPU) and next to the
game model's prediction for the same matches.

Matches of selfplay.eval_points points with the evaluation seeds of scripts/selfplay_es.py
(9,000,000 + j), sides alternating, point cap selfplay.max_hits (draw). Writes results/selfplay_ppo_<tag>_summary.md.

Usage:
    python scripts/eval_selfplay_ppo.py --policy results/selfplay_gpu/ppo_r1/policy.pt --model results/selfplay_gpu/game_model_r1.pt --tag r1
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
sys.path.insert(0, str(Path(__file__).resolve().parent))

from tt.config import REPO, RESULTS, load_config  # noqa: E402
from tt.mpc import solve_stat_rows  # noqa: E402

CFG = REPO / "config" / "selfplay_ppo.yaml"


def model_prediction(model_path, policy_path, cfg, opponents, n=8192, seed=0):
    import torch

    from tt.game_model import load_model
    from tt.ppo_policy import load_actor_critic
    from tt.selfplay_ppo import League, evaluate_in_model

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    model = load_model(model_path, dev)
    ac = load_actor_critic(policy_path, dev)
    so = torch.as_tensor(np.asarray(model.extra["serve_obs"], np.float32), device=dev)
    sr = torch.as_tensor(np.asarray(model.extra["serve_r"], np.float32), device=dev)
    league = League(ac, cfg, np.load(RESULTS / "rl" / "theta.npy"), float(np.median(model.extra["serve_r"])), dev)
    return evaluate_in_model(model, league, cfg, so, sr, opponents, np.random.default_rng(seed), n)


def fault_rates(res) -> tuple:
    """Errors per ball received for X and Y (X is robot A in even-numbered points): (errors, balls) each.
    A player receives the serve (robot A) and every opponent hit, except the last hit of a point that ended at the cap
    or by that hitter's own fault (net / out / short: the ball never had to be played). An error is any fault by the
    player while playing a ball it received (missed, double bounce, or its own shot into the net / out / short)."""
    f = {"X": 0, "Y": 0}
    balls = {"X": 0, "Y": 0}
    for j, r in enumerate(res):
        if str(r.get("fault", "")).startswith("error"):
            continue
        side = {"A": "X", "B": "Y"} if j % 2 == 0 else {"A": "Y", "B": "X"}
        hits = r.get("hit_log", [])
        balls[side["A"]] += 1                                          # the serve
        for who, _ in hits:
            balls[side["B" if who == "A" else "A"]] += 1
        last = hits[-1][0] if hits else None
        if last is not None and (r.get("fault") == "max hits" or r.get("by") == last):
            balls[side["B" if last == "A" else "A"]] -= 1              # that ball was never played
        if r.get("by") in ("A", "B"):
            f[side[r["by"]]] += 1
    return (f["X"], balls["X"]), (f["Y"], balls["Y"])


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", type=Path, default=CFG)
    ap.add_argument("--policy", type=Path, required=True)
    ap.add_argument("--model", type=Path, default=None, help="game model for the predicted results column")
    ap.add_argument("--tag", required=True)
    ap.add_argument("--points", type=int, default=None)
    ap.add_argument("--workers", type=int, default=None)
    args = ap.parse_args()
    import selfplay_es as se

    cfg = load_config(args.config)
    sp = cfg["selfplay"]
    n, cap = args.points or sp["eval_points"], sp["max_hits"]
    theta = np.load(RESULTS / "rl" / "theta.npy")
    ppo = ("ppo", str(args.policy.resolve()))
    matches = [("PPO vs cooperative", ppo, ("cooperative", None), "cooperative"),
               ("PPO vs random", ppo, ("random", None), "random"),
               ("PPO vs CEM (earlier strategy)", ppo, ("linear", theta), "cem"),
               ("cooperative vs cooperative (baseline)", ("cooperative", None), ("cooperative", None), None)]
    pred = model_prediction(args.model, args.policy, cfg, [m[3] for m in matches if m[3]]) if args.model else {}
    rows, mpc_rows = [], []
    t0 = time.time()
    with ProcessPoolExecutor(args.workers or sp["workers"]) as ex:
        for name, x, y, key in matches:
            w, l, dr, hits, res = se.match(ex, args.config, x, y, n, 9_000_000, cap)
            p = pred.get(key)
            fx, fy = fault_rates(res)
            rows.append((name, w, l, dr, float(np.mean(hits)), p, fx, fy))
            mpc_rows.append((name, solve_stat_rows([e for r in res for log in r.get("solve_logs", {}).values() for e in log])))
            print(f"{name}: {w}-{l}, {dr} draws, {np.mean(hits):.1f} hits/point "
                  f"({time.time() - t0:.0f} s)", flush=True)
    md = [f"# Self-play v2 (PPO + league, GPU game model): full-rally evaluation, {args.tag}", "",
          f"Policy: {args.policy}. Config: {args.config.resolve().relative_to(REPO)} (rail rally, current MPC with "
          f"best-effort swings). {n} points per match with the scripts/selfplay_es.py evaluation seeds, sides alternate, "
          f"point cap {cap} hits (draw). Model column: the game model's prediction for the same match "
          f"(deterministic policies, 8192 points).", "",
          "| match (X vs Y) | X wins | X losses | draws | mean hits/point | faults per ball received: X / Y | "
          "game model: win / draw / loss |", "|---|---|---|---|---|---|---|"]
    for name, w, l, dr, h, p, fx, fy in rows:
        pm = f"{p['win']:.2f} / {p['draw']:.2f} / {p['loss']:.2f}" if p else "–"
        fr = f"{fx[0]}/{fx[1]} ({100 * fx[0] / max(fx[1], 1):.1f}%) / {fy[0]}/{fy[1]} ({100 * fy[0] / max(fy[1], 1):.1f}%)"
        md.append(f"| {name} | {w} | {l} | {dr} | {h:.1f} | {fr} | {pm} |")
    md += ["", "MPC solves per match (both robots):", ""] + [
        f"- {name}: " + "; ".join(f"{k}: {v}" for k, v in mr) for name, mr in mpc_rows]
    (RESULTS / f"selfplay_ppo_{args.tag}_summary.md").write_text("\n".join(md) + "\n")
    (RESULTS / f"selfplay_ppo_{args.tag}_eval.json").write_text(json.dumps(dict(rows=rows), default=str))
    print("\n".join(md))


if __name__ == "__main__":
    main()
