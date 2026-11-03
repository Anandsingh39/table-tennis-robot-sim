"""GPU self-play bookkeeping (tt.selfplay_ppo) with a fake game model whose outcomes are known; policy wrapper."""

import numpy as np
import pytest
import torch

from tt.config import REPO, load_config
from tt.ppo_policy import ActorCritic, load_policy, save_policy
from tt.selfplay_ppo import League, play, ppo_update


class FakeModel:
    """Every stroke has outcome probabilities p (continue, lose, opp_fault); the next observation = the current one."""

    def __init__(self, p):
        self.p = torch.tensor(p, dtype=torch.float32)
        self.obs_lo, self.obs_hi = torch.full((7,), -10.0), torch.full((7,), 10.0)

    def sample(self, obs, act, r):
        out = torch.multinomial(self.p.expand(obs.shape[0], -1), 1).squeeze(-1)
        return out, obs.clone()

    def outcome_probs(self, obs, act, r):
        return self.p.expand(obs.shape[0], -1), torch.zeros(obs.shape[0], 3)


@pytest.fixture(scope="module")
def setup():
    cfg = load_config(REPO / "config" / "selfplay_ppo.yaml")
    lo = np.array([-0.8, 0.8, 0.35, -0.5])
    hi = np.array([0.8, 1.2, 1.0, 0.5])
    torch.manual_seed(0)
    ac = ActorCritic(np.zeros(7), np.ones(7), lo, hi, hidden=32)
    league = League(ac, cfg, np.zeros(24), 0.1, "cpu")
    serve = torch.zeros(16, 7), torch.full((16,), 0.1)
    return cfg, ac, league, serve


def _play(setup, p, kind="cooperative", n=512, seed=0):
    cfg, ac, league, (so, sr) = setup
    return play(FakeModel(p), league, "self", np.array([kind] * n, dtype=object), so, sr, cfg,
                np.random.default_rng(seed), record=True)


def test_decider_always_loses(setup):
    ro = _play(setup, [0.0, 1.0, 0.0])
    res = ro.result.numpy()
    assert set(np.unique(res)) <= {-1.0, 1.0} and (ro.strokes == 1).all()
    # the learner decides only when it is robot A (serve receiver) -> exactly the points it lost
    assert len(ro.adv) == int((res == -1).sum())
    assert torch.allclose(ro.ret, torch.full_like(ro.ret, -1.0))      # return = final reward


def test_decider_always_wins(setup):
    ro = _play(setup, [0.0, 0.0, 1.0])
    res = ro.result.numpy()
    assert len(ro.adv) == int((res == 1).sum()) and torch.allclose(ro.ret, torch.ones_like(ro.ret))


def test_always_continue_draws_at_the_cap(setup):
    cfg = setup[0]
    ro = _play(setup, [1.0, 0.0, 0.0], n=64)
    cap = cfg["ppo"]["max_hits"]
    assert (ro.result == 0).all() and (ro.strokes == cap).all()
    assert len(ro.adv) == 64 * cap // 2                                 # the learner makes every other decision


def test_ppo_update_runs_and_league_kinds(setup):
    cfg, ac, league, _ = setup
    ro = _play(setup, [0.5, 0.3, 0.2], kind="random")
    st = ppo_update(ac, torch.optim.Adam(ac.parameters(), 1e-3), ro, cfg)
    assert np.isfinite([st["pg_loss"], st["v_loss"], st["entropy"]]).all()
    for kind in ("self", "cooperative", "random", "cem"):
        u = league.act(kind, torch.zeros(5, 7))
        assert u.shape == (5, 4) and (u > 0).all() and (u < 1).all()


def test_torch_strategy_roundtrip(setup, tmp_path):
    _, ac, _, _ = setup
    save_policy(ac, tmp_path / "p.pt")
    det = load_policy(tmp_path / "p.pt")
    a = det(np.zeros(7))
    assert a.shape == (4,) and np.all(a >= ac.act_lo.numpy()) and np.all(a <= ac.act_hi.numpy())
    e1, e2 = load_policy(tmp_path / "p.pt", explore=True, seed=3), load_policy(tmp_path / "p.pt", explore=True, seed=3)
    assert np.allclose(e1(np.zeros(7)), e2(np.zeros(7)))               # seeded exploration is reproducible
