"""Phase 3 tests: impact law (M3), table restitution from the ITTF rule, Section 6 inverse, event handling."""

import mujoco
import numpy as np
import pytest

from tt.aero import AeroParams, BallAero
from tt.config import load_config
from tt.contact import (ContactModel, arm_paddle_state, impact, paddle_surface, table_restitution_from_rule,
                        table_surface)
from tt.juggle_law import frictionless_rebound, paddle_command
from tt.sim import make_sim

M, R = 0.0027, 0.02
I_SHELL = 2 / 3 * M * R ** 2


@pytest.fixture(scope="module")
def cfg():
    return load_config()


def test_frictionless_impact_matches_section6_formula():
    rng = np.random.default_rng(1)
    for _ in range(20):
        n = rng.normal(size=3)
        n /= np.linalg.norm(n)
        v, v_c, w = rng.normal(size=3), rng.normal(size=3), rng.normal(size=3) * 50
        if (v - v_c) @ n > 0:
            v = -v
        v_new, w_new, _ = impact(v, w, n, v_c, 0.8, 0.0, M, R, I_SHELL)
        assert np.allclose(v_new, frictionless_rebound(v, n, v_c, 0.8))
        assert np.allclose(w_new, w)                                    # no friction -> spin unchanged


def test_friction_rolling_and_sliding():
    n, e = np.array([0, 0, 1.0]), 0.8
    v = np.array([0.5, 0.0, -3.0])
    # large mu: rolling -> contact-point tangential velocity ends at zero
    v1, w1, rolling = impact(v, np.zeros(3), n, np.zeros(3), e, 10.0, M, R, I_SHELL)
    u_t = v1 + np.cross(w1, -R * n)
    assert rolling and abs(u_t[0]) < 1e-12 and v1[2] == pytest.approx(e * 3.0)
    assert v1[0] == pytest.approx(0.5 * (1 - (2 / 3) / (1 + 2 / 3)))      # v_t' = v_t / (1 + a), a = 2/3
    # small mu: sliding -> tangential impulse = mu * normal impulse
    # (rolling would need |J_t|/m = 0.5 a/(1+a) = 0.2 > mu (1+e) 3.0 = 0.108)
    mu = 0.02
    v2, _, rolling = impact(v, np.zeros(3), n, np.zeros(3), e, mu, M, R, I_SHELL)
    assert not rolling and (0.5 - v2[0]) == pytest.approx(mu * (1 + e) * 3.0)


def test_table_restitution_from_rule(cfg):
    assert table_restitution_from_rule(0.30, 0.23, 0.0) == pytest.approx(np.sqrt(0.23 / 0.30))
    k = AeroParams.from_config(cfg).k_d
    e = table_restitution_from_rule(0.30, 0.23, k)
    assert e > np.sqrt(0.23 / 0.30)          # drag eats energy, so the surface must be bouncier
    # consistency with the closed-form rise height after a drop with drag
    v2 = (1 / k) * -np.expm1(-2 * k * 0.30)  # v^2 / g at impact
    assert np.log1p(k * e ** 2 * v2) / (2 * k) == pytest.approx(0.23)


def test_paddle_command_inverts_impact():
    v_in, v_out, e = np.array([0.1, -0.2, -2.8]), np.array([-0.05, 0.1, 2.8]), 0.8
    n, s = paddle_command(v_in, v_out, e)
    assert np.linalg.norm(n) == pytest.approx(1.0)
    assert np.allclose(frictionless_rebound(v_in, n, s * n, e), v_out)


def test_static_paddle_drop_single_impact_per_bounce(cfg):
    sim = make_sim(cfg)
    params = AeroParams.from_config(cfg, drag_coefficient=0.0)
    aero = BallAero(sim.m, sim.ids, params)
    contact = ContactModel(sim.m, sim.ids, aero, [paddle_surface(cfg, arm_paddle_state(sim.ids)),
                                                  table_surface(cfg, params)])
    h, e, g = 0.3, cfg["contact"]["paddle_restitution"], 9.81
    z_c = sim.p_ready[2] + cfg["paddle"]["blade_thickness"] / 2 + cfg["ball"]["radius"]
    aero.set_ball_state(sim.m, sim.d, [*sim.p_ready[:2], z_c + h], np.zeros(3))
    m, d = sim.m, sim.d
    for _ in range(int(1.0 / m.opt.timestep)):
        mujoco.mj_step1(m, d)
        sim.hold_arm()
        contact.step(m, d)
        aero.apply(d)
        mujoco.mj_step2(m, d)
    t1 = np.sqrt(2 * h / g)
    t_an = [t1, t1 + 2 * e * np.sqrt(2 * g * h) / g]
    t_an.append(t_an[-1] + 2 * e ** 2 * np.sqrt(2 * g * h) / g)
    t_ev = [ev.t for ev in contact.events]
    assert [ev.surface for ev in contact.events] == ["paddle"] * 3
    assert np.allclose(t_ev, t_an, atol=cfg["checks"]["event_time_tol"])
    assert all(ev.u_n_in < 0 for ev in contact.events)
