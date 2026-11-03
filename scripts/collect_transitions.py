"""Self-play v2, step 1: record real stroke-level transitions from full rally points (MuJoCo + MPC, CPU).

Each point: robots A and B each get an exploration policy (tt.strategy.ExplorePolicy; kind drawn from collect.mix:
uniform random shots, the cooperative shot + noise, or the CEM strategy + noise; random rail recovery targets).
Transitions (tt.selfplay_data) are saved in shards of collect.shard_points points under
results/selfplay_gpu/<tag>/shard_XXXX.npz; existing shards are skipped, so an interrupted run resumes.
A `--policy PATH` adds a trained PPO policy (with exploration) to the mix (later data rounds).

Usage:
    nohup python scripts/collect_transitions.py --points 4000 --tag round1 > results/selfplay_gpu/collect_round1.log 2>&1 &
"""

from __future__ import annotations

import os

os.environ.setdefault("OMP_NUM_THREADS", "1")

import argparse  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402
import zlib  # noqa: E402
from concurrent.futures import ProcessPoolExecutor  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tt.config import REPO, RESULTS, load_config  # noqa: E402

CFG = REPO / "config" / "selfplay_ppo.yaml"
OUT = RESULTS / "selfplay_gpu"


def point(args):
    cfg_path, seed, max_hits, policy_path = args
    from tt.selfplay_data import point_transitions
    from tt.strategy import make_policy, play_point

    cfg = load_config(cfg_path)
    rng = np.random.default_rng(seed)
    mix = dict(cfg["collect"]["mix"])
    if policy_path:
        mix["explore_ppo"] = mix.get("explore_ppo", 0.4)
    kinds, probs = list(mix), np.array(list(mix.values()), float)
    theta = np.load(RESULTS / "rl" / "theta.npy")
    pols, names = [], []
    for k in range(2):
        kind = kinds[rng.choice(len(kinds), p=probs / probs.sum())]
        if kind == "explore_ppo":
            from tt.ppo_policy import load_policy
            pol = load_policy(policy_path, cfg, explore=True, seed=seed * 2 + k)
        else:
            pol = make_policy(kind, cfg, theta if kind == "explore_cem" else None, seed=seed * 2 + k)
        pols.append(pol)
        names.append(kind)
    t0 = time.time()
    try:
        res = play_point(cfg, seed, pols[0], pols[1], max_hits)
    except Exception as exc:          # a failed serve solve etc.: no transitions from this point
        return seed, names, [], f"error: {exc}", time.time() - t0
    return seed, names, point_transitions(res), res["fault"], time.time() - t0


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", type=Path, default=CFG)
    ap.add_argument("--points", type=int, required=True)
    ap.add_argument("--tag", required=True)
    ap.add_argument("--policy", type=Path, default=None, help="trained PPO policy to add to the exploration mix")
    ap.add_argument("--workers", type=int, default=None)
    args = ap.parse_args()
    from tt.selfplay_data import stack

    cfg = load_config(args.config)
    cc = cfg["collect"]
    out = OUT / args.tag
    out.mkdir(parents=True, exist_ok=True)
    n_shards = -(-args.points // cc["shard_points"])
    seed0 = cc["seed0"] + 1_000_000 * (zlib.crc32(args.tag.encode()) % 1000)     # a separate seed block per tag
    with ProcessPoolExecutor(args.workers or cc["workers"]) as ex:
        for s in range(n_shards):
            path = out / f"shard_{s:04d}.npz"
            if path.exists():
                continue
            t0 = time.time()
            seeds = [seed0 + s * cc["shard_points"] + j for j in range(cc["shard_points"])]
            jobs = [(args.config, sd, cc["max_hits"], str(args.policy) if args.policy else "") for sd in seeds]
            trans, seed_of, kinds_of, faults = [], [], [], []
            for seed, names, tr, fault, wall in ex.map(point, jobs):
                trans += tr
                seed_of += [seed] * len(tr)
                kinds_of += ["|".join(names)] * len(tr)
                faults.append(fault)
            np.savez_compressed(path, **stack(trans, seed=seed_of, kinds=kinds_of))
            n_err = sum(str(f).startswith("error") for f in faults)
            print(f"shard {s + 1}/{n_shards}: {len(trans)} transitions from {len(seeds)} points "
                  f"({n_err} errors) in {time.time() - t0:.0f} s", flush=True)


if __name__ == "__main__":
    main()
