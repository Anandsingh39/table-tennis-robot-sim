"""Self-play v2: PPO with a league of opponents in the learned game model, batched on the GPU.

Game (one point per environment, thousands in parallel): robot A decides first (it receives the serve; serve states
are drawn from the recorded data). At each stroke the decider picks an action; the game model (tt.game_model, random
ensemble member per stroke) samples the outcome:
    continue  -> the opponent decides next from the sampled observation; the decider's recovery target becomes the
                 receiver's rail target in the model input,
    lose      -> the decider loses the point,   opp_fault -> the decider wins,
    after `max_hits` continues the point is a draw (as the full-rally match cap).
One side is the learner (random side per point); the other is drawn from the league: the current policy, frozen
past snapshots, and fixed baselines (cooperative, random, the earlier CEM strategy). Reward +1 / -1 / 0 for the
learner at the end of the point; GAE with gamma = 1 over the learner's own decisions; standard clipped PPO.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass

import numpy as np
import torch

from tt.game_model import GameModel
from tt.ppo_policy import ActorCritic
from tt.strategy import N_ACT as N_SHOT, N_OBS as N_STRAT_OBS

FIXED = ("cooperative", "random", "cem")


class League:
    """Opponent policies, each mapping observations [n, 7] -> u in (0, 1)^4 (torch, on the training device)."""

    def __init__(self, ac: ActorCritic, cfg: dict, cem_theta: np.ndarray, rail_ready: float, device):
        self.ac, self.snaps = ac, []
        lo, hi = ac.act_lo, ac.act_hi
        sc = cfg["strategy"]
        coop = torch.tensor([0.0, sc["coop_z"], sc["coop_d"], rail_ready], device=device)
        self.u_coop = ((coop - lo) / (hi - lo)).clamp(1e-4, 1 - 1e-4)
        self.u_ready = float(self.u_coop[3])
        W = torch.as_tensor(np.asarray(cem_theta, np.float32).reshape(N_SHOT, N_STRAT_OBS + 1), device=device)
        self.cem_W, self.cem_b = W[:, :N_STRAT_OBS], W[:, N_STRAT_OBS]
        self.device = device

    def add_snapshot(self):
        snap = copy.deepcopy(self.ac).eval()
        for p in snap.parameters():
            p.requires_grad_(False)
        self.snaps.append(snap)

    def names(self):
        return ["self"] + list(FIXED) + [f"snap{i}" for i in range(len(self.snaps))]

    def act(self, kind: str, obs, deterministic=False):
        n = obs.shape[0]
        if kind == "self":
            return self.ac.act(obs, deterministic)
        if kind.startswith("snap"):
            return self.snaps[int(kind[4:])].act(obs, deterministic)
        if kind == "cooperative":
            return self.u_coop.expand(n, -1).clone()
        if kind == "random":                      # uniform shot, ready-pose recovery (as the full-rally RandomPolicy)
            u = torch.rand(n, 4, device=self.device)
            u[:, 3] = self.u_ready
            return u
        if kind == "cem":                         # linear tanh policy of the earlier CEM self-play (3 shot actions)
            u = torch.empty(n, 4, device=self.device)
            u[:, :3] = (torch.tanh(obs @ self.cem_W.T + self.cem_b) + 1) / 2
            u[:, 3] = self.u_ready
            return u.clamp(1e-4, 1 - 1e-4)
        raise ValueError(kind)

    def sample_kinds(self, n: int, mix: dict, rng: np.random.Generator) -> np.ndarray:
        """Opponent kind per environment: 'self', 'past' (uniform over snapshots; 'self' if none yet), fixed baselines."""
        keys = list(mix)
        p = np.array([mix[k] for k in keys], float)
        pick = rng.choice(len(keys), n, p=p / p.sum())
        kinds = np.array([keys[i] for i in pick], dtype=object)
        past = kinds == "past"
        if past.any():
            kinds[past] = [f"snap{i}" for i in rng.integers(len(self.snaps), size=past.sum())] if self.snaps else "self"
        return kinds


@dataclass
class Rollout:
    obs: torch.Tensor          # [M, 7] learner decisions
    u: torch.Tensor            # [M, 4]
    logp: torch.Tensor         # [M]
    adv: torch.Tensor          # [M]
    ret: torch.Tensor          # [M]
    result: torch.Tensor       # [N] learner's point result (+1 / 0 / -1)
    kinds: np.ndarray          # [N] opponent kind per point
    strokes: torch.Tensor      # [N] strokes per point
    p_lose_spread: float       # mean ensemble spread of P(lose) over the learner's decisions (model-exploitation monitor)


def play(model: GameModel, league: League, learner_kind: str, opp_kinds: np.ndarray, serve_obs, serve_r, cfg: dict,
         rng: np.random.Generator, deterministic=False, record=True, gae_lambda=0.95):
    """Play N points (N = len(opp_kinds)) to the end in the game model. The learner acts with
    league.act(learner_kind, ...) ('self' = the policy being trained). Returns a Rollout (advantages only if record)."""
    dev = serve_obs.device
    n = len(opp_kinds)
    max_hits = cfg["ppo"]["max_hits"]
    learner_A = torch.as_tensor(rng.random(n) < 0.5, device=dev)
    learner_turn = learner_A.clone()                         # A decides first (receives the serve)
    k = torch.as_tensor(rng.integers(len(serve_obs), size=n), device=dev)
    obs, r = serve_obs[k].clone(), serve_r[k].clone()
    hits = torch.zeros(n, dtype=torch.long, device=dev)
    done = torch.zeros(n, dtype=torch.bool, device=dev)
    result = torch.zeros(n, device=dev)
    strokes = torch.zeros(n, dtype=torch.long, device=dev)
    T = max_hits + 2
    B_obs = torch.zeros(T, n, obs.shape[1], device=dev)
    B_u = torch.zeros(T, n, 4, device=dev)
    B_logp = torch.zeros(T, n, device=dev)
    B_v = torch.zeros(T, n, device=dev)
    B_m = torch.zeros(T, n, dtype=torch.bool, device=dev)
    last_t = torch.full((n,), -1, dtype=torch.long, device=dev)
    spread = []
    kinds_t = np.asarray(opp_kinds)
    for t in range(T):
        active = ~done
        if not bool(active.any()):
            break
        u = torch.zeros(n, 4, device=dev)
        lm = active & learner_turn
        om = active & ~learner_turn
        if bool(lm.any()):
            o = obs[lm]
            if record:
                with torch.no_grad():
                    d = league.ac.dist(o)
                    ul = d.sample().clamp(1e-4, 1 - 1e-4)
                    B_logp[t, lm] = d.log_prob(ul).sum(-1)
                    B_v[t, lm] = league.ac.value(o)
            else:
                ul = league.act(learner_kind, o, deterministic)
            u[lm] = ul
            B_obs[t, lm], B_u[t, lm], B_m[t, lm] = o, ul, True
            last_t[lm] = t
        if bool(om.any()):
            om_np = om.cpu().numpy()
            for kind in np.unique(kinds_t[om_np]):
                sel = om & torch.as_tensor(kinds_t == kind, device=dev)
                u[sel] = league.act(kind, obs[sel], deterministic)
        act = league.ac.to_action(u)
        ia = active.nonzero().squeeze(-1)
        outcome, nxt = model.sample(obs[ia], act[ia], r[ia])
        if bool(lm.any()):
            pm, ps = model.outcome_probs(obs[lm], act[lm], r[lm])
            spread.append(float(ps[:, 1].mean()))
        strokes[ia] += 1
        dec_learner = learner_turn[ia]
        cont = outcome == 0
        hits[ia] += cont.long()
        end = ~cont
        draw = cont & (hits[ia] >= max_hits)
        decider_wins = outcome == 2
        learner_wins = torch.where(dec_learner, decider_wins, ~decider_wins)
        res = torch.where(learner_wins, 1.0, -1.0)
        result[ia[end]] = res[end]
        done[ia[end | draw]] = True
        go = ia[cont & ~draw]
        keep = cont & ~draw
        obs[go] = nxt[keep].clamp(model.obs_lo, model.obs_hi)
        r[go] = act[ia[keep], 3]                              # the decider becomes the receiver
        learner_turn[go] = ~learner_turn[go]
    if not record:
        return Rollout(None, None, None, None, None, result, kinds_t, strokes, float(np.mean(spread)) if spread else 0.0)
    # GAE (gamma = 1) over each point's learner decisions; reward only at the learner's last decision
    adv = torch.zeros(T, n, device=dev)
    next_v = torch.zeros(n, device=dev)
    next_adv = torch.zeros(n, device=dev)
    for t in reversed(range(T)):
        m = B_m[t]
        is_last = m & (last_t == t)
        v = B_v[t]
        target = torch.where(is_last, result, next_v)
        a = target - v + torch.where(is_last, torch.zeros_like(v), gae_lambda * next_adv)
        adv[t] = torch.where(m, a, torch.zeros_like(a))
        next_v = torch.where(m, v, next_v)
        next_adv = torch.where(m, a, next_adv)
    m = B_m
    ret = (adv + B_v)[m]
    return Rollout(B_obs[m], B_u[m], B_logp[m], adv[m], ret, result, kinds_t, strokes,
                   float(np.mean(spread)) if spread else 0.0)


def ppo_update(ac: ActorCritic, opt, ro: Rollout, cfg: dict) -> dict:
    pc = cfg["ppo"]
    adv = (ro.adv - ro.adv.mean()) / (ro.adv.std() + 1e-8)
    n = len(adv)
    stats = []
    for _ in range(pc["epochs"]):
        perm = torch.randperm(n, device=adv.device)
        for i in range(0, n, pc["minibatch"]):
            idx = perm[i:i + pc["minibatch"]]
            d = ac.dist(ro.obs[idx])
            logp = d.log_prob(ro.u[idx]).sum(-1)
            ratio = (logp - ro.logp[idx]).exp()
            a = adv[idx]
            pg = -torch.min(ratio * a, ratio.clamp(1 - pc["clip"], 1 + pc["clip"]) * a).mean()
            vf = 0.5 * (ac.value(ro.obs[idx]) - ro.ret[idx]).pow(2).mean()
            ent = d.entropy().sum(-1).mean()
            loss = pg + pc["vf_coef"] * vf - pc["ent_coef"] * ent
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(ac.parameters(), pc["max_grad_norm"])
            opt.step()
            stats.append((pg.item(), vf.item(), ent.item(), ((ratio.detach() - 1).abs() > pc["clip"]).float().mean().item()))
    s = np.mean(stats, axis=0)
    return dict(pg_loss=s[0], v_loss=s[1], entropy=s[2], clip_frac=s[3], samples=n)


def evaluate_in_model(model, league, cfg, serve_obs, serve_r, opponents, rng, n=4096) -> dict:
    """Deterministic learner ('self', Beta mean) vs each opponent kind (also deterministic where it applies)."""
    out = {}
    for kind in opponents:
        ro = play(model, league, "self", np.array([kind] * n, dtype=object), serve_obs, serve_r, cfg, rng,
                  deterministic=True, record=False)
        res = ro.result.cpu().numpy()
        out[kind] = dict(win=float((res > 0).mean()), draw=float((res == 0).mean()), loss=float((res < 0).mean()),
                         strokes=float(ro.strokes.float().mean()))
    return out
