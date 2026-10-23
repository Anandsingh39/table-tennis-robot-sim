"""Phase 6 tests: sensor model and EKF (M4)."""

import copy

import numpy as np
import pytest

from tt.aero import AeroParams, rk4_rollout
from tt.config import load_config
from tt.metrics import juggle_metrics
from tt.sensing import BallEKF, BallSensor, Measurement
from tt.sim import run_juggle


@pytest.fixture(scope="module")
def cfg():
    return load_config()


def test_sensor_rate_noise_latency_dropout(cfg):
    c = copy.deepcopy(cfg)
    c["sensing"].update(pos_noise_std=0.002, latency=0.01, dropout_prob=0.2)
    s = BallSensor(c, np.random.default_rng(0))
    dt, n_steps = 0.001, 20000
    taken = []
    for k in range(n_steps):
        m = s.sample(k * dt, np.zeros(3))
        if m is not None:
            taken.append(m)
            assert m.t_deliver == pytest.approx(m.t + 0.01)
    expected = n_steps * dt * c["sensing"]["rate"]
    assert len(taken) == pytest.approx(0.8 * expected, rel=0.05)            # 20% dropped
    z = np.array([m.z for m in taken])
    assert np.std(z) == pytest.approx(0.002, rel=0.05) and abs(np.mean(z)) < 2e-4
    # delivery: each sample is released exactly `latency` after it was taken, in order
    assert s.delivered(0.0099) == []                                        # nothing can be due before 10 ms
    got = s.delivered(0.0205)
    assert got and [m.t for m in got] == [m.t for m in taken if m.t_deliver <= 0.0205 + 1e-9]


def test_ekf_tracks_ballistic_flight(cfg):
    p = AeroParams.from_config(cfg)
    sigma, rate = cfg["sensing"]["pos_noise_std"], cfg["sensing"]["rate"]
    dt = 1.0 / rate
    x, v = rk4_rollout([0.4, 0.0, 1.0], [0.1, -0.05, 2.9], [0, 0, 0], p, dt, int(0.5 * rate))
    ekf = BallEKF(cfg, p, z_impact=0.0)
    rng = np.random.default_rng(3)
    for k in range(len(x)):
        ekf.update(Measurement(k * dt, x[k] + rng.normal(0, sigma, 3), k * dt))
    p_hat, v_hat = ekf.state_at(0.5)
    assert np.linalg.norm(p_hat - x[-1]) < 3 * sigma
    assert np.linalg.norm(v_hat - v[-1]) < 0.05
    assert ekf.bounces == []                                                # no false bounce in free flight


def test_ekf_detects_bounce_and_uses_commanded_velocity(cfg):
    p = AeroParams.from_config(cfg)
    sigma, dt = cfg["sensing"]["pos_noise_std"], 1.0 / cfg["sensing"]["rate"]
    v_out = np.array([0.0, 0.0, 2.9])
    ekf = BallEKF(cfg, p, z_impact=1.0, expected_v_out=lambda: v_out)
    rng = np.random.default_rng(4)
    # falling onto z = 1.0, bouncing at t_b with v_out, then rising
    x_down, v_down = rk4_rollout([0.4, 0, 1.3], [0, 0, 0], [0, 0, 0], p, dt, 1000)
    k_b = int(np.argmax(x_down[:, 2] < 1.0))
    x_up, v_up = rk4_rollout(x_down[k_b], v_out, [0, 0, 0], p, dt, 30)
    traj = np.vstack([x_down[:k_b], x_up[1:]])
    for k, xk in enumerate(traj):
        ekf.update(Measurement(k * dt, xk + rng.normal(0, sigma, 3), k * dt))
    assert len(ekf.bounces) == 1 and k_b * dt <= ekf.bounces[0] <= (k_b + 2) * dt
    _, v_hat = ekf.state_at((len(traj) - 1) * dt)
    assert np.linalg.norm(v_hat - v_up[-1]) < 0.1                           # tracking the rising ball


def test_short_juggle_true_state(cfg):
    c = copy.deepcopy(cfg)
    c["sensing"]["ball_source"] = "true"
    run = run_juggle(c, seed=1, max_bounces=8)
    mt = juggle_metrics(run, c)
    assert run.failure is None and not mt["saturated"]
    assert np.all(np.abs(mt["apex_err"][c["juggle"]["transient_bounces"]:]) < c["juggle"]["max_apex_err"])


def test_short_juggle_noisy(cfg):
    run = run_juggle(cfg, seed=2, max_bounces=8)
    mt = juggle_metrics(run, cfg)
    assert run.failure is None and mt["bounces"] >= 8
    assert len(run.detected_bounces) >= 8
