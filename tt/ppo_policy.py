"""Self-play v2 policy: actor-critic network for the shot strategy (PyTorch).

Actor: o (7, normalised) -> MLP -> Beta(alpha, beta) per action dimension (alpha, beta > 1), u in (0, 1)^4,
action = lo + (hi - lo) u with the 4 actions (y_c, z_c, d_b, rail recovery target) in the hitter's frame. A Beta
keeps every sampled action inside its physical range. Critic: o -> V(o), the expected point result (+1 win, -1 loss).
load_policy() wraps a trained network as a strategy callable for the full rally (tt.rally2), on CPU.
"""

from __future__ import annotations

import numpy as np
import torch
from torch import nn

from tt.selfplay_data import N_ACT, N_OBS


def mlp(n_in, hidden, n_out):
    return nn.Sequential(nn.Linear(n_in, hidden), nn.Tanh(), nn.Linear(hidden, hidden), nn.Tanh(), nn.Linear(hidden, n_out))


class ActorCritic(nn.Module):
    def __init__(self, obs_mean, obs_std, act_lo, act_hi, hidden: int = 256):
        super().__init__()
        self.pi = mlp(N_OBS, hidden, 2 * N_ACT)
        self.v = mlp(N_OBS, hidden, 1)
        for name, x in (("obs_mean", obs_mean), ("obs_std", obs_std), ("act_lo", act_lo), ("act_hi", act_hi)):
            self.register_buffer(name, torch.as_tensor(np.asarray(x), dtype=torch.float32))
        self.hidden = hidden

    def _o(self, obs):
        return (obs - self.obs_mean) / self.obs_std

    def dist(self, obs):
        out = self.pi(self._o(obs))
        a = nn.functional.softplus(out[..., :N_ACT]) + 1.0
        b = nn.functional.softplus(out[..., N_ACT:]) + 1.0
        return torch.distributions.Beta(a, b)

    def value(self, obs):
        return self.v(self._o(obs)).squeeze(-1)

    def to_action(self, u):
        return self.act_lo + (self.act_hi - self.act_lo) * u

    @torch.no_grad()
    def act(self, obs, deterministic: bool = False):
        """u in (0, 1)^4 (sampled with torch's global RNG, or the Beta mean when deterministic)."""
        d = self.dist(obs)
        return d.mean if deterministic else d.sample().clamp(1e-4, 1 - 1e-4)


def save_policy(ac: ActorCritic, path, extra: dict | None = None) -> None:
    torch.save(dict(state=ac.state_dict(), hidden=ac.hidden, extra=extra or {}), path)


def load_actor_critic(path, device="cpu") -> ActorCritic:
    ck = torch.load(path, map_location=device, weights_only=False)
    s = ck["state"]
    ac = ActorCritic(s["obs_mean"].cpu().numpy(), s["obs_std"].cpu().numpy(), s["act_lo"].cpu().numpy(),
                     s["act_hi"].cpu().numpy(), ck["hidden"])
    ac.load_state_dict(s)
    return ac.to(device).eval()


class TorchStrategy:
    """Strategy callable for tt.rally2.RallyPlayer: observation (numpy, 7) -> action (numpy, 4) on CPU."""

    def __init__(self, ac: ActorCritic, explore: bool, seed: int):
        self.ac, self.explore = ac, explore
        self.rng = np.random.default_rng(seed)       # exploration samples are seeded (reproducible data rounds)

    def __call__(self, obs: np.ndarray) -> np.ndarray:
        o = torch.as_tensor(np.asarray(obs, np.float32)).unsqueeze(0)
        with torch.no_grad():
            d = self.ac.dist(o)
            if self.explore:
                u = np.clip(self.rng.beta(d.concentration1.numpy()[0], d.concentration0.numpy()[0]), 1e-4, 1 - 1e-4)
            else:
                u = d.mean.numpy()[0]
            lo, hi = self.ac.act_lo.numpy(), self.ac.act_hi.numpy()
        return (lo + (hi - lo) * u).astype(float)


def load_policy(path, cfg: dict | None = None, explore: bool = False, seed: int = 0) -> TorchStrategy:
    torch.set_num_threads(1)
    return TorchStrategy(load_actor_critic(path, "cpu"), explore, seed)
