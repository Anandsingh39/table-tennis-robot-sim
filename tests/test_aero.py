"""Phase 2 tests: aerodynamic force (M2), state conventions, and MuJoCo vs RK4 accuracy."""

import mujoco
import numpy as np
import pytest

from tt.aero import AeroParams, BallAero, aero_accel, rk4_rollout
from tt.config import load_config
from tt.sim import make_sim


@pytest.fixture(scope="module")
def cfg():
    return load_config()


def test_drag_coefficient_and_direction(cfg):
    p = AeroParams.from_config(cfg)
    ac, bc = cfg["aero"], cfg["ball"]
    k_d = ac["air_density"] * ac["drag_coefficient"] * np.pi * bc["radius"] ** 2 / (2 * bc["mass"])
    assert p.k_d == pytest.approx(k_d)
    v = np.array([3.0, -1.0, 2.0])
    a = aero_accel(v, np.zeros(3), p)
    assert np.allclose(np.cross(a, v), 0) and a @ v < 0           # opposes velocity
    assert np.linalg.norm(a) == pytest.approx(p.k_d * (v @ v))    # quadratic in speed


def test_magnus_direction(cfg):
    p = AeroParams.from_config(cfg, magnus_coefficient=0.5)
    # backspin (w along -y for a ball moving +x) lifts the ball
    a = aero_accel(np.array([5.0, 0, 0]), np.array([0, -100.0, 0]), p) + p.k_d * 25 * np.array([1.0, 0, 0])
    assert a[2] > 0 and abs(a[0]) < 1e-12 and abs(a[1]) < 1e-12


def test_freejoint_velocity_frames(cfg):
    """qvel = [v_world, w_local]: set_ball_state with a rotated ball must give world spin w via mj_objectVelocity."""
    sim = make_sim(cfg)
    m, d, ids = sim.m, sim.d, sim.ids
    aero = BallAero(m, ids, AeroParams.from_config(cfg))
    quat = np.array([0.3, -0.5, 0.7, 0.2])
    d.qpos[ids.ball_qpos + 3:ids.ball_qpos + 7] = quat / np.linalg.norm(quat)
    v, w = np.array([1.0, -2.0, 3.0]), np.array([10.0, 20.0, -30.0])
    aero.set_ball_state(m, d, [0.4, 0.0, 1.5], v, w)
    mujoco.mj_forward(m, d)
    res = np.zeros(6)
    mujoco.mj_objectVelocity(m, d, mujoco.mjtObj.mjOBJ_BODY, ids.ball_body, res, 0)   # world frame: [ang; lin]
    assert np.allclose(res[:3], w)
    _, v_back, w_back = aero.ball_state(d)
    assert np.allclose(w_back, w)
    assert np.allclose(v_back, v, atol=1e-6)    # half-step round trip is exact up to O(dt^2)


def _fly(cfg, params, pos, vel, spin, T, half_step=True, quat=None):
    sim = make_sim(cfg)
    m, d, ids = sim.m, sim.d, sim.ids
    if quat is not None:
        d.qpos[ids.ball_qpos + 3:ids.ball_qpos + 7] = np.asarray(quat) / np.linalg.norm(quat)
    aero = BallAero(m, ids, params, half_step=half_step)
    aero.set_ball_state(m, d, pos, vel, spin)
    xs = []
    for _ in range(int(round(T / m.opt.timestep)) + 1):
        mujoco.mj_step1(m, d)
        sim.hold_arm()
        aero.apply(d)
        xs.append(aero.ball_state(d)[0])
        mujoco.mj_step2(m, d)
    return np.array(xs)


def test_vacuum_matches_parabola(cfg):
    p = AeroParams.from_config(cfg, drag_coefficient=0.0, magnus_coefficient=0.0)
    pos, vel, T = np.array([0.0, 0.0, 1.2]), np.array([1.0, 0.5, 4.0]), 0.5
    x = _fly(cfg, p, pos, vel, np.zeros(3), T)
    t = np.arange(len(x)) * cfg["physics"]["timestep"]
    exact = pos + np.outer(t, vel) + 0.5 * np.outer(t ** 2, cfg["physics"]["gravity"])
    assert np.abs(x - exact).max() < 1e-9


def test_mujoco_matches_rk4_with_rotated_spinning_ball(cfg):
    ck = cfg["checks"]
    p = AeroParams.from_config(cfg, magnus_coefficient=ck["ballistic_test_magnus_coefficient"])
    pos, vel, spin, T = [0.0, 0.0, 1.2], [2.0, 0.5, 4.5], [20.0, 40.0, 100.0], ck["ballistic_duration"]
    x = _fly(cfg, p, pos, vel, spin, T, quat=[0.6, 0.1, -0.7, 0.3])
    x_rk, _ = rk4_rollout(pos, vel, spin, p, cfg["physics"]["timestep"], len(x) - 1)
    assert np.linalg.norm(x - x_rk, axis=1).max() < ck["ballistic_max_error"]
