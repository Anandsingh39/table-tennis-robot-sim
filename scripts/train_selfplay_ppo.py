"""Self-play v2, step 3: PPO with a league of opponents in the learned game model (GPU).

Each iteration plays ppo.envs points in parallel in the game model (tt.selfplay_ppo.play): the learner (current
policy, sampling) on a random side, the opponent drawn per point from ppo.league (current policy, frozen past
snapshots, cooperative, random, the earlier CEM strategy), then one PPO update. A snapshot joins the league every
ppo.snapshot_every iterations. Every ppo.eval_every iterations the deterministic policy is evaluated in the model
against the fixed opponents and the first snapshot. Outputs in results/selfplay_gpu/ppo_<tag>/.

Usage:
    python scripts/train_selfplay_ppo.py --model results/selfplay_gpu/game_model_r1.pt --tag r1
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tt.config import REPO, RESULTS, load_config  # noqa: E402
from tt.game_model import load_model  # noqa: E402
from tt.ppo_policy import ActorCritic, save_policy  # noqa: E402
from tt.selfplay_ppo import FIXED, League, evaluate_in_model, play, ppo_update  # noqa: E402

CFG = REPO / "config" / "selfplay_ppo.yaml"
OUT = RESULTS / "selfplay_gpu"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", type=Path, default=CFG)
    ap.add_argument("--model", type=Path, required=True)
    ap.add_argument("--tag", required=True)
    ap.add_argument("--iterations", type=int, default=None)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    cfg = load_config(args.config)
    pc = cfg["ppo"]
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)
    out = OUT / f"ppo_{args.tag}"
    out.mkdir(parents=True, exist_ok=True)

    model = load_model(args.model, dev)
    serve_obs = torch.as_tensor(np.asarray(model.extra["serve_obs"], np.float32), device=dev)
    serve_r = torch.as_tensor(np.asarray(model.extra["serve_r"], np.float32), device=dev)
    rail_ready = float(np.median(model.extra["serve_r"]))
    ac = ActorCritic(model.obs_mean.cpu().numpy(), model.obs_std.cpu().numpy(), model.act_lo.cpu().numpy(),
                     model.act_hi.cpu().numpy(), pc["hidden"]).to(dev)
    opt = torch.optim.Adam(ac.parameters(), lr=pc["lr"])
    league = League(ac, cfg, np.load(RESULTS / "rl" / "theta.npy"), rail_ready, dev)
    hist = []
    n_it = args.iterations or pc["iterations"]
    t0 = time.time()
    for it in range(n_it):
        kinds = league.sample_kinds(pc["envs"], pc["league"], rng)
        ro = play(model, league, "self", kinds, serve_obs, serve_r, cfg, rng, record=True, gae_lambda=pc["gae_lambda"])
        st = ppo_update(ac, opt, ro, cfg)
        res = ro.result.cpu().numpy()
        groups = {g: res[np.array([k.startswith("snap") if g == "past" else k == g for k in ro.kinds])]
                  for g in ("self", "past") + FIXED}
        row = dict(it=it, time=time.time() - t0, **st, spread=ro.p_lose_spread,
                   train_win={g: float((v > 0).mean()) if len(v) else None for g, v in groups.items()})
        if (it + 1) % pc["snapshot_every"] == 0:
            league.add_snapshot()
        if it == 0 or (it + 1) % pc["eval_every"] == 0 or it == n_it - 1:
            opps = list(FIXED) + (["snap0"] if league.snaps else [])
            row["eval"] = evaluate_in_model(model, league, cfg, serve_obs, serve_r, opps, rng, pc["eval_envs"])
            ev = row["eval"]
            print(f"it {it + 1:5d} {row['time']:6.0f}s  entropy {st['entropy']:+.2f}  clip {st['clip_frac']:.2f}  "
                  f"P(lose) spread {ro.p_lose_spread:.3f}  | win/draw/loss vs " +
                  "  ".join(f"{k} {ev[k]['win']:.2f}/{ev[k]['draw']:.2f}/{ev[k]['loss']:.2f}" for k in opps), flush=True)
            save_policy(ac, out / "policy.pt", extra=dict(it=it + 1, model=str(args.model), cfg=str(args.config)))
            (out / "history.json").write_text(json.dumps(hist + [row]))
        hist.append(row)
    save_policy(ac, out / "policy.pt", extra=dict(it=n_it, model=str(args.model), cfg=str(args.config)))
    (out / "history.json").write_text(json.dumps(hist))
    print(f"done: {n_it} iterations in {time.time() - t0:.0f} s on {dev}; policy {out / 'policy.pt'}")


if __name__ == "__main__":
    main()
