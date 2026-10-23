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


def test_rail_kinematics_and_short_rally():
    """E2 + rail: slide joint in the symbolic FK; a short rally with lateral targets completes."""
    from tt.config import REPO
    from tt.fk_casadi import PaddleFK
    from tt.model import build_model, scene_ids
    from tt.rally2 import run_rally

    cfg3 = load_config(REPO / "config" / "rally_rail.yaml")
    m, ids = build_model(cfg3)
    ido = scene_ids(m, cfg3, cfg3["opponent"]["prefix"])
    assert ids.joint_names[0] == "rail_slide" and len(ids.arm_qpos) == 8
    d = mujoco.MjData(m)
    rng = np.random.default_rng(0)
    for idx in (ids, ido):
        fk = PaddleFK(m, idx.paddle_site, idx.arm_qpos)
        q = rng.uniform(idx.q_lo, idx.q_hi)
        p, _, _, _ = paddle_fk(m, d, idx, q)
        assert np.allclose(np.array(fk.pos(q)).ravel(), p, atol=1e-9)
    r = run_rally(cfg3, seed=2, max_hits=3)
    assert r.fault == "max hits" and len(r.hits) == 3


def test_grip_contact_inverse_round_trip():
    """Spin: the grip paddle inverse reproduces the requested outgoing velocity and in-plane spin."""
    from tt.contact import impact_grip
    from tt.juggle_law import grip_paddle_command

    m, r = 0.0027, 0.02
    inertia = 2 / 3 * m * r ** 2
    v_in, w_in = np.array([-3.2, 0.1, -2.8]), np.array([0.0, -40.0, 5.0])
    v_out = np.array([2.9, 0.05, 4.1])
    for spin in (-80.0, 0.0, 60.0):
        w_des = spin * np.array([0.0, 1.0, 0.0])          # topspin axis for a ball going +x
        n, v_c, w_out, res = grip_paddle_command(v_in, w_in, v_out, w_des, 0.8, 0.819, 0.805, m, r, inertia)
        vo, wo = impact_grip(v_in, w_in, n, v_c, 0.8, 0.819, 0.805, m, r, inertia)
        assert res < 1e-8 and np.allclose(vo, v_out, atol=1e-8)
        dw = wo - w_des
        assert np.linalg.norm(dw - (dw @ n) * n) < 1e-5


def test_predictor_spinning_table_bounce_matches_contact_law():
    """Spin: the predictor's table bounce (with friction) is the simulator's impulse law."""
    from tt.contact import impact

    tb = TableBounce((0.0, 3.0), (-1.0, 1.0), 0.78, 0.9, 0.25, 0.0027, 0.02, 2 / 3 * 0.0027 * 0.02 ** 2)
    v, w = np.array([3.0, 0.2, -4.0]), np.array([0.0, 100.0, 10.0])
    v1, w1 = tb.bounce(v, w)
    v2, w2, _ = impact(v, w, np.array([0, 0, 1.0]), np.zeros(3), 0.9, 0.25, 0.0027, 0.02, 2 / 3 * 0.0027 * 0.02 ** 2)
    assert np.allclose(v1, v2) and np.allclose(w1, w2)
    v0, _ = tb.bounce(v, np.zeros(3))
    vb, _ = tb.bounce(v, -w)
    assert v1[0] > v0[0] > vb[0]              # topspin keeps more forward speed off the table than no spin; backspin checks
