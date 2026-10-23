"""Phase 4 tests: quintic boundary conditions (M7), impact joint state, computed-torque tracking (M8)."""

import mujoco
import numpy as np
import pytest

from tt.config import load_config
from tt.control import computed_torque
from tt.kinematics import paddle_pose, task_jacobian
from tt.planner import Quintic, impact_joint_state, plan_stroke
from tt.sim import make_sim


@pytest.fixture(scope="module")
def cfg():
    return load_config()


def test_quintic_boundary_conditions():
    rng = np.random.default_rng(0)
    q0, v0, a0, q1, v1, a1 = rng.normal(size=(6, 7))
    qu = Quintic.fit(0.3, 0.4, q0, v0, a0, q1, v1, a1)
    for t, ref in ((0.3, (q0, v0, a0)), (0.7, (q1, v1, a1))):
        for got, want in zip(qu(t), ref):
            assert np.allclose(got, want, atol=1e-9)
    # derivatives are consistent (finite difference)
    h = 1e-6
    q_a, qd_a, _ = qu(0.5)
    q_b, qd_b, qdd_b = qu(0.5 + h)
    assert np.allclose((q_b - q_a) / h, qd_a, atol=1e-4)
    assert np.allclose((qd_b - qd_a) / h, qdd_b, atol=1e-3)


def test_impact_joint_state_realises_task_velocity(cfg):
    sim = make_sim(cfg)
    m, ids = sim.m, sim.ids
    scratch = mujoco.MjData(m)
    p = np.asarray(cfg["hit"]["center"], float)
    n = np.array([0.0, 0.1, 1.0]) / np.linalg.norm([0.0, 0.1, 1.0])
    v = 0.8 * n
    q, qd, info = impact_joint_state(m, scratch, ids, cfg, p, n, v, sim.q_ready)
    assert info["ik_pos_err"] < 1e-6 and info["ik_normal_err_deg"] < 1e-4
    J = task_jacobian(m, scratch, ids, q)
    assert np.allclose(J @ qd, np.concatenate([v, [0, 0]]), atol=1e-4)


def test_computed_torque_tracks_stroke(cfg):
    sim = make_sim(cfg)
    m, d, ids = sim.m, sim.d, sim.ids
    cc, tr = cfg["control"], cfg["tracking"]
    scratch = mujoco.MjData(m)
    p = np.asarray(cfg["hit"]["center"], float)
    n = np.array([0.0, 0.0, 1.0])
    q_hit, qd_hit, _ = impact_joint_state(m, scratch, ids, cfg, p, n, 1.0 * n, sim.q_ready)
    z = np.zeros(7)
    t_hit = cfg["planner"]["approach_time"]
    traj = plan_stroke(0.0, sim.q_ready, z, z, t_hit, q_hit, qd_hit, cfg["planner"]["return_time"], sim.q_ready)
    p_at_hit, sat_any = None, False
    for _ in range(int(round(t_hit / m.opt.timestep)) + 1):
        mujoco.mj_step1(m, d)
        q_d, qd_d, qdd_d = traj(d.time)
        tau, sat = computed_torque(m, d, ids, q_d, qd_d, qdd_d, cc["ct_kp"], cc["ct_kd"], cc["friction_ff"],
                                   cc["friction_smoothing"])
        sat_any |= sat
        d.ctrl[ids.arm_act] = tau
        if abs(d.time - t_hit) < m.opt.timestep / 2:
            p_at_hit = paddle_pose(m, d, ids)[0]
        mujoco.mj_step2(m, d)
    assert not sat_any
    assert np.linalg.norm(p_at_hit - p) < tr["max_pos_err"]
