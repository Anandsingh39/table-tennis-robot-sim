"""Self-play v2: learned model of the rally game at the stroke level (PyTorch, runs on GPU or CPU).

Trained on real transitions from the full MuJoCo + MPC rally (tt.selfplay_data). Input per stroke:
    x = (observation o [7], action [4] scaled to [-1, 1], receiver's rail recovery target [1])
Output per ensemble member:
    logits of the decider's stroke outcome (continue / lose / opp_fault), and a diagonal Gaussian over the
    opponent's next observation o' (used when the outcome is "continue").
An ensemble of independently initialised members, each trained on a bootstrap resample, is used so that the
policy cannot exploit one network's errors (a random member is drawn per stroke, as in model-based RL / MBPO).
"""

from __future__ import annotations

import math

import numpy as np
import torch
from torch import nn

from tt.selfplay_data import N_ACT, N_OBS

N_IN = N_OBS + N_ACT + 1
N_OUT = 3 + 2 * N_OBS


class EnsembleLinear(nn.Module):
    """E independent linear layers applied to E batches at once: [E, B, i] -> [E, B, o]."""

    def __init__(self, n_members: int, n_in: int, n_out: int):
        super().__init__()
        self.weight = nn.Parameter(torch.empty(n_members, n_in, n_out))
        self.bias = nn.Parameter(torch.zeros(n_members, 1, n_out))
        bound = 1.0 / math.sqrt(n_in)
        nn.init.uniform_(self.weight, -bound, bound)
        nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, x):
        return torch.baddbmm(self.bias, x, self.weight)


class GameModel(nn.Module):
    def __init__(self, n_members: int, hidden: int, layers: int, act_lo, act_hi, obs_mean, obs_std, r_scale: float,
                 obs_lo=None, obs_hi=None):
        super().__init__()
        dims = [N_IN] + [hidden] * layers
        self.layers = nn.ModuleList([EnsembleLinear(n_members, a, b) for a, b in zip(dims[:-1], dims[1:])])
        self.head = EnsembleLinear(n_members, hidden, N_OUT)
        self.n_members = n_members
        obs_lo = np.full(N_OBS, -np.inf) if obs_lo is None else obs_lo      # range of the recorded observations:
        obs_hi = np.full(N_OBS, np.inf) if obs_hi is None else obs_hi       # simulated ones are clamped to it
        for name, v in (("act_lo", act_lo), ("act_hi", act_hi), ("obs_mean", obs_mean), ("obs_std", obs_std),
                        ("obs_lo", obs_lo), ("obs_hi", obs_hi)):
            self.register_buffer(name, torch.as_tensor(np.asarray(v), dtype=torch.float32))
        self.register_buffer("r_scale", torch.tensor(float(r_scale)))
        self.config = dict(n_members=n_members, hidden=hidden, layers=layers, r_scale=float(r_scale))

    def inputs(self, obs, act, r_recv):
        """Physical units -> normalised network input [B, N_IN]."""
        a = 2 * (act - self.act_lo) / (self.act_hi - self.act_lo) - 1
        o = (obs - self.obs_mean) / self.obs_std
        return torch.cat([o, a, (r_recv / self.r_scale).unsqueeze(-1)], dim=-1)

    def forward(self, x):
        """x [B, N_IN] (same for all members) or [E, B, N_IN] -> logits [E, B, 3], mean [E, B, 7], log_std [E, B, 7]
        (mean and log_std of the NORMALISED next observation)."""
        h = x if x.dim() == 3 else x.unsqueeze(0).expand(self.n_members, -1, -1)
        for layer in self.layers:
            h = nn.functional.silu(layer(h))
        out = self.head(h)
        logits, mean, log_std = out[..., :3], out[..., 3:3 + N_OBS], out[..., 3 + N_OBS:]
        return logits, mean, log_std.clamp(-6.0, 2.0)

    @torch.no_grad()
    def sample(self, obs, act, r_recv, generator=None):
        """One stroke for a batch: a random member per row. Returns (outcome [B] long, next_obs [B, 7] physical)."""
        logits, mean, log_std = self(self.inputs(obs, act, r_recv))
        b = obs.shape[0]
        k = torch.randint(self.n_members, (b,), device=obs.device, generator=generator)
        rows = torch.arange(b, device=obs.device)
        lg, mu, ls = logits[k, rows], mean[k, rows], log_std[k, rows]
        outcome = torch.multinomial(torch.softmax(lg, -1), 1, generator=generator).squeeze(-1)
        noise = torch.randn(mu.shape, device=obs.device, generator=generator)
        nxt = (mu + ls.exp() * noise) * self.obs_std + self.obs_mean
        return outcome, nxt

    @torch.no_grad()
    def outcome_probs(self, obs, act, r_recv):
        """Ensemble-mean outcome probabilities [B, 3] and their spread across members (std) [B, 3]."""
        logits, _, _ = self(self.inputs(obs, act, r_recv))
        p = torch.softmax(logits, -1)
        return p.mean(0), p.std(0)


def save_model(model: GameModel, path, extra: dict | None = None) -> None:
    torch.save(dict(state=model.state_dict(), config=model.config, extra=extra or {}), path)


def load_model(path, device="cpu") -> GameModel:
    ck = torch.load(path, map_location=device, weights_only=False)
    c = ck["config"]
    s = ck["state"]
    m = GameModel(c["n_members"], c["hidden"], c["layers"], s["act_lo"].cpu().numpy(), s["act_hi"].cpu().numpy(),
                  s["obs_mean"].cpu().numpy(), s["obs_std"].cpu().numpy(), c["r_scale"], s["obs_lo"].cpu().numpy(),
                  s["obs_hi"].cpu().numpy())
    m.load_state_dict(s)
    m.extra = ck.get("extra", {})
    return m.to(device).eval()
