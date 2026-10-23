"""E1 test: the CasADi paddle FK matches MuJoCo's kinematics (plan: error < 1e-6 at random q)."""

import mujoco
import numpy as np

from tt.config import load_config
from tt.fk_casadi import PaddleFK
from tt.kinematics import paddle_fk
from tt.model import build_model


def test_casadi_fk_matches_mujoco():
    cfg = load_config()
    m, ids = build_model(cfg)
    d = mujoco.MjData(m)
    fk = PaddleFK(m, ids.paddle_site, ids.arm_qpos)
    rng = np.random.default_rng(0)
    for _ in range(50):
        q = rng.uniform(ids.q_lo, ids.q_hi)
        qd = rng.normal(size=7)
        p, n, jp, _ = paddle_fk(m, d, ids, q)
        R_mj = d.site_xmat[ids.paddle_site].reshape(3, 3)
        assert np.abs(np.array(fk.pos(q)).ravel() - p).max() < 1e-9
        assert np.abs(np.array(fk.rot(q)) - R_mj).max() < 1e-9
        assert np.abs(np.array(fk.vel(q, qd)).ravel() - jp @ qd).max() < 1e-9
