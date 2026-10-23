"""E1 tests: predictor with the table bounce, return-shot solver, MPC swing constraints."""

import mujoco
import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from tt.aero import AeroParams
from tt.config import RALLY_CONFIG, load_config
from tt.kinematics import paddle_fk
from tt.mpc import SwingMPC
from tt.predictor import BallPredictor, TableBounce
from tt.shot import ReturnShot, fly_for
from tt.sim import make_sim


@pytest.fixture(scope="module")
def cfg():
    return load_config(RALLY_CONFIG)


def test_rally_config_extends_default(cfg):
    base = load_config()
    assert cfg["paddle"]["flange_quat"] != base["paddle"]["flange_quat"]
    assert cfg["ball"] == base["ball"] and cfg["table"] == base["table"]


def test_plane_crossing_after_table_bounce_vacuum(cfg):
    p = AeroParams.from_config(cfg, drag_coefficient=0.0)
    g, e = 9.81, 0.9
    table = TableBounce((0.0, 3.0), (-1.0, 1.0), 0.78, e)
    pos, vel = np.array([2.5, 0.0, 1.0]), np.array([-5.0, 0.0, 0.0])      # fast enough for exactly one bounce
    pred = BallPredictor(p, 0.005, 3.0).plane_crossing(pos, vel, np.zeros(3), 0, 0.4, -1.0, table, require_bounce=True)
    # analytic: fall 0.22 m to the bounce, then rebound with e * v_z until x = 0.4
    t_b = np.sqrt(2 * 0.22 / g)
    vz_b = e * g * t_b
    t_c = (2.5 - 0.4) / 5.0
    z_c = 0.78 + vz_b * (t_c - t_b) - 0.5 * g * (t_c - t_b) ** 2
    assert pred.bounced and pred.t_rel == pytest.approx(t_c, abs=1e-6)
    assert pred.pos[2] == pytest.approx(z_c, abs=1e-6)
    assert pred.vel[2] == pytest.approx(vz_b - g * (t_c - t_b), abs=1e-5)


def test_return_shot_lands_on_target(cfg):
    p = AeroParams.from_config(cfg)
    shot = ReturnShot(cfg, p)
    p_hit, target = np.array([0.41, 0.01, 0.93]), np.array([3.05, 0.1])
    plan = shot.plan(p_hit, np.array([-3.5, 0.0, -1.0]), target)
    end = fly_for(p, p_hit, plan.v_out, plan.T, 1e-4)        # independent finer integration
    assert np.linalg.norm(end[:2] - target) < 2e-3
    assert end[2] == pytest.approx(shot.z_land, abs=2e-3)
    assert plan.net_clearance >= cfg["shot"]["net_margin"]


def test_mpc_swing_meets_impact_and_limits(cfg):
    sim = make_sim(cfg)
    m, ids = sim.m, sim.ids
    mpc = SwingMPC(m, ids, cfg, sim.q_ready)
    c = np.array([0.40, 0.02, 0.92])
    n = Rotation.from_euler("y", 50, degrees=True).apply([0, 0, 1.0])
    plan = mpc.solve(0.0, sim.q_ready, np.zeros(7), 0.35, c, n, 1.8)
    assert plan.success
    d = mujoco.MjData(m)
    p, nn, jp, _ = paddle_fk(m, d, ids, plan.q[plan.n_pre])
    assert np.linalg.norm(p - c) < 1e-6 and np.degrees(np.arccos(np.clip(nn @ n, -1, 1))) < 1e-3
    assert n @ (jp @ plan.qd[plan.n_pre]) == pytest.approx(1.8, abs=1e-6)
    tol = 1e-6
    assert np.all(plan.q >= mpc.q_lo - tol) and np.all(plan.q <= mpc.q_hi + tol)
    assert np.all(np.abs(plan.qd) <= mpc.qd_max + tol) and np.all(np.abs(plan.a) <= mpc.a_max + tol)
    assert np.allclose(plan.qd[-1], 0, atol=tol)
    q, qd, _ = plan(plan.t_hit)                                   # the trajectory passes the impact node
    assert np.allclose(q, plan.q[plan.n_pre], atol=1e-9)


def test_two_robot_rally_short():
    """E2: two mirrored arms, cooperative lobs; a short rally completes without a fault."""
    from tt.config import REPO
    from tt.rally2 import run_rally

    cfg2 = load_config(REPO / "config" / "rally2.yaml")
    r = run_rally(cfg2, seed=1, max_hits=4)
    assert r.fault == "max hits" and len(r.hits) == 4
    assert [who for who, _ in r.hits] == ["A", "B", "A", "B"]
    assert not (r.logs["sat_A"].any() or r.logs["sat_B"].any())
