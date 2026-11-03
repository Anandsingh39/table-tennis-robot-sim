"""Self-play v2, step 2: fit the GPU game model (tt.game_model) to the recorded transitions and check it on held-out points.

Split by point seed (no point contributes to both training and validation). Each ensemble member is trained on its own
bootstrap resample. Loss = cross-entropy of the stroke outcome + Gaussian NLL of the next observation (only for
"continue" strokes). Early stopping on the validation loss. Report (results/selfplay_gpu/game_model_<tag>.md):
outcome log-loss / accuracy vs the class-prior baseline, calibration of P(lose), next-observation error vs the
predict-the-mean baseline.

Usage:
    python scripts/train_game_model.py --data round1 [--data round2 ...] --tag r1
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
from tt.game_model import GameModel, save_model  # noqa: E402
from tt.selfplay_data import N_OBS, OUTCOMES  # noqa: E402
from tt.strategy import action_bounds  # noqa: E402

CFG = REPO / "config" / "selfplay_ppo.yaml"
OUT = RESULTS / "selfplay_gpu"


def load_shards(tags) -> dict:
    parts = [np.load(f, allow_pickle=False) for t in tags for f in sorted((OUT / t).glob("shard_*.npz"))]
    if not parts:
        raise SystemExit(f"no shards for {tags}")
    keys = ["obs", "act", "r_recv", "outcome", "next_obs", "seed", "fallback", "hits_before"]
    d = {k: np.concatenate([p[k] for p in parts]) for k in keys}
    serve = d["hits_before"] == 0                    # first decision of each point: robot A receiving the serve
    keep = d["outcome"] >= 0
    out = {k: v[keep] for k, v in d.items()}
    out["serve_obs"], out["serve_r"] = d["obs"][serve], d["r_recv"][serve]
    return out


def act_bounds_all(cfg):
    lo, hi = action_bounds(cfg)
    r_lo, r_hi = cfg["strategy"]["recover_range"]
    return np.append(lo, r_lo), np.append(hi, r_hi)


def batch_loss(model, obs, act, r, outc, nxt, cont):
    """obs etc. [E, B, ...] (one minibatch per member). Returns (total, ce, nll) averaged over members."""
    x = model.inputs(obs, act, r)
    logits, mean, log_std = model(x)
    ce = torch.nn.functional.cross_entropy(logits.reshape(-1, 3), outc.reshape(-1), reduction="none").view(outc.shape)
    y = (nxt - model.obs_mean) / model.obs_std
    nll = (0.5 * ((y - mean) / log_std.exp()) ** 2 + log_std).sum(-1)
    nll = torch.where(cont, nll, torch.zeros_like(nll))
    n_cont = cont.float().sum(-1).clamp(min=1)
    return ce.mean(-1), (nll.sum(-1) / n_cont)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", type=Path, default=CFG)
    ap.add_argument("--data", action="append", required=True)
    ap.add_argument("--tag", required=True)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    cfg = load_config(args.config)
    mc = cfg["model"]
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)

    d = load_shards(args.data)
    serve_obs, serve_r = d.pop("serve_obs"), d.pop("serve_r")
    seeds = np.unique(d["seed"])
    val_seeds = rng.choice(seeds, int(round(mc["val_frac"] * len(seeds))), replace=False)
    is_val = np.isin(d["seed"], val_seeds)
    tr, va = {k: v[~is_val] for k, v in d.items()}, {k: v[is_val] for k, v in d.items()}
    n_tr, n_va = len(tr["obs"]), len(va["obs"])
    cont_tr = tr["outcome"] == 0
    obs_mean = np.concatenate([tr["obs"], tr["next_obs"][cont_tr]]).mean(0)
    obs_std = np.concatenate([tr["obs"], tr["next_obs"][cont_tr]]).std(0) + 1e-3
    lo, hi = act_bounds_all(cfg)
    E = mc["members"]
    all_obs = np.concatenate([d["obs"], d["next_obs"][d["outcome"] == 0]])
    model = GameModel(E, mc["hidden"], mc["layers"], lo, hi, obs_mean, obs_std,
                      max(abs(x) for x in cfg["strategy"]["recover_range"]), all_obs.min(0), all_obs.max(0)).to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=mc["lr"], weight_decay=mc["weight_decay"])

    def tensors(s):
        nx = np.nan_to_num(s["next_obs"])
        return [torch.as_tensor(a, dtype=torch.float32, device=dev) for a in (s["obs"], s["act"], s["r_recv"], nx)] + \
               [torch.as_tensor(s["outcome"], dtype=torch.long, device=dev)]

    T_tr, T_va = tensors(tr), tensors(va)
    boot = torch.as_tensor(rng.integers(0, n_tr, (E, n_tr)), device=dev)         # bootstrap index per member
    B = mc["batch"]
    best, best_state, bad, hist = np.inf, None, 0, []
    t0 = time.time()
    for epoch in range(mc["max_epochs"]):
        model.train()
        perm = boot[:, torch.randperm(n_tr, device=dev)]
        for i in range(0, n_tr, B):
            idx = perm[:, i:i + B]                                             # [E, b]
            obs, act, r, nxt, outc = (t[idx] for t in T_tr)
            ce, nll = batch_loss(model, obs, act, r, outc, nxt, outc == 0)
            loss = (ce + mc["w_next"] * nll).sum()
            opt.zero_grad()
            loss.backward()
            opt.step()
        model.eval()
        with torch.no_grad():
            idx = torch.arange(n_va, device=dev).expand(E, -1)
            obs, act, r, nxt, outc = (t[idx] for t in T_va)
            ce, nll = batch_loss(model, obs, act, r, outc, nxt, outc == 0)
            v = float((ce + mc["w_next"] * nll).mean())
        hist.append(dict(epoch=epoch, val=v, val_ce=float(ce.mean()), val_nll=float(nll.mean())))
        if v < best - 1e-4:
            best, bad = v, 0
            best_state = {k: t.detach().clone() for k, t in model.state_dict().items()}
        else:
            bad += 1
            if bad >= mc["patience"]:
                break
    model.load_state_dict(best_state)
    model.eval()
    train_time = time.time() - t0

    # ---- held-out evaluation (ensemble mean) ----
    with torch.no_grad():
        obs, act, r, nxt, outc = T_va
        logits, mean, log_std = model(model.inputs(obs, act, r))
        p = torch.softmax(logits, -1).mean(0).cpu().numpy()
        mu = (mean.mean(0) * model.obs_std + model.obs_mean).cpu().numpy()
    y = va["outcome"]
    prior = np.bincount(tr["outcome"], minlength=3) / n_tr
    ll = -np.mean(np.log(np.clip(p[np.arange(n_va), y], 1e-9, 1)))
    ll0 = -np.mean(np.log(np.clip(prior[y], 1e-9, 1)))
    acc = float(np.mean(p.argmax(1) == y))
    acc0 = float(np.mean(prior.argmax() == y))
    p_lose = p[:, 1]
    bins = np.linspace(0, 1, 6)
    calib = []
    for a, b in zip(bins[:-1], bins[1:]):
        sel = (p_lose >= a) & (p_lose < b if b < 1 else p_lose <= b)
        if sel.sum():
            calib.append((f"{a:.1f}-{b:.1f}", int(sel.sum()), float(p_lose[sel].mean()), float((y[sel] == 1).mean())))
    cont = y == 0
    err = mu[cont] - va["next_obs"][cont]
    rmse = np.sqrt((err ** 2).mean(0))
    rmse0 = va["next_obs"][cont].std(0)
    names = ["impact y", "impact z", "v_x", "v_y", "v_z", "own rail", "opp rail"]

    path = OUT / f"game_model_{args.tag}.pt"
    report = dict(tag=args.tag, data=args.data, n_train=n_tr, n_val=n_va, points=len(seeds), epochs=len(hist),
                  train_time_s=train_time, device=dev, outcome_prior=prior.tolist(),
                  val_logloss=ll, val_logloss_prior=ll0, val_acc=acc, val_acc_prior=acc0, calibration=calib,
                  next_rmse=rmse.tolist(), next_rmse_baseline=rmse0.tolist())
    save_model(model, path, extra=dict(report=report, serve_obs=serve_obs.tolist(), serve_r=serve_r.tolist()))
    md = [f"# Game model {args.tag}", "",
          f"Data: {', '.join(args.data)} - {len(seeds)} points, {n_tr} training / {n_va} held-out strokes "
          f"(split by point). Ensemble of {E}, {mc['layers']}x{mc['hidden']}, {len(hist)} epochs, {train_time:.0f} s on {dev}.", "",
          "| stroke outcome | class share (train) |", "|---|---|"] + \
         [f"| {o} | {100 * q:.1f}% |" for o, q in zip(OUTCOMES, prior)] + \
         ["", "| held-out metric | model | baseline (class prior) |", "|---|---|---|",
          f"| outcome log-loss | {ll:.3f} | {ll0:.3f} |", f"| outcome accuracy | {100 * acc:.1f}% | {100 * acc0:.1f}% |", "",
          "Calibration of P(lose):", "", "| predicted | strokes | mean predicted | observed |", "|---|---|---|---|"] + \
         [f"| {a} | {n} | {pm:.2f} | {po:.2f} |" for a, n, pm, po in calib] + \
         ["", "Next observation (continue strokes), RMSE in feature units:", "",
          "| feature | model | baseline (std) |", "|---|---|---|"] + \
         [f"| {nm} | {a:.3f} | {b:.3f} |" for nm, a, b in zip(names, rmse, rmse0)]
    (OUT / f"game_model_{args.tag}.md").write_text("\n".join(md) + "\n")
    (OUT / f"game_model_{args.tag}_history.json").write_text(json.dumps(hist))
    print("\n".join(md))
    print(f"saved {path}")


if __name__ == "__main__":
    main()
