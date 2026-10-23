"""Phase 5 tests: predictor (M5) and juggling law (M6). End-to-end juggles are in test_sensing.py."""

import numpy as np
import pytest

from tt.aero import AeroParams, rk4_rollout
from tt.config import load_config
from tt.juggle_law import ImpactTargetingLaw, PDTiltLaw, frictionless_rebound, launch_speed_for_apex
from tt.predictor import BallPredictor


@pytest.fixture(scope="module")
def cfg():
    return load_config()


def test_predictor_matches_vacuum_parabola(cfg):
    p = AeroParams.from_config(cfg, drag_coefficient=0.0)
    g = -p.gravity[2]
    pos, vel, z_c = np.array([0.4, 0.0, 1.5]), np.array([0.3, -0.2, 1.0]), 1.0
    pred = BallPredictor(p, 0.005, 2.0).crossing(pos, vel, np.zeros(3), z_c)
    t = (vel[2] + np.sqrt(vel[2] ** 2 + 2 * g * (pos[2] - z_c))) / g      # z(t) = z_c, descending root
    assert pred.t_rel == pytest.approx(t, abs=1e-6)
    assert np.allclose(pred.pos[:2], pos[:2] + vel[:2] * t, atol=1e-6)
    assert pred.vel[2] == pytest.approx(vel[2] - g * t, abs=1e-5)
    assert BallPredictor(p, 0.005, 2.0).crossing([0, 0, 0.9], [0, 0, -1.0], np.zeros(3), z_c) is None


def test_launch_speed_reaches_apex_with_drag(cfg):
    p = AeroParams.from_config(cfg)
    h = cfg["juggle"]["apex_height"]
    v = launch_speed_for_apex(h, p.k_d, -p.gravity[2])
    x, _ = rk4_rollout([0, 0, 0], [0, 0, v], [0, 0, 0], p, 1e-4, 6000)
    assert x[:, 2].max() == pytest.approx(h, abs=1e-5)


def test_law_on_centre_is_vertical_and_tilt_saturates(cfg):
    p = AeroParams.from_config(cfg)
    law = ImpactTargetingLaw.from_config(cfg, p, cfg["juggle"]["predictor_dt"])
    c = np.array([*law.center_xy, 1.0])
    v_in = np.array([0.0, 0.0, -2.8])
    n, s, v_out = law(c, v_in)
    assert np.allclose(n, [0, 0, 1]) and np.allclose(v_out, [0, 0, law.v_z])
    assert np.allclose(frictionless_rebound(v_in, n, s * n, law.e), v_out)
    # a ball arriving fast sideways and far off-centre would need a large tilt: capped at max_tilt
    n, s, v_out = law(c + [0.5, 0, 0], np.array([3.0, 0.0, -2.8]))
    assert np.degrees(np.arccos(n[2])) == pytest.approx(cfg["juggle"]["max_tilt_deg"], abs=1e-6)
    assert np.allclose(frictionless_rebound(np.array([3.0, 0.0, -2.8]), n, s * n, law.e), v_out)


def test_pd_tilt_law_is_linearised_impact_targeting(cfg):
    """With the derived gains the PD-tilt law matches the impact-targeting law for small offsets,
    and its paddle speed always gives the apex-target vertical speed."""
    import copy

    c = copy.deepcopy(cfg)
    c["juggle"]["pd_kp"] = c["juggle"]["pd_kd"] = None
    p = AeroParams.from_config(c)
    it, pd = ImpactTargetingLaw.from_config(c, p, 0.005), PDTiltLaw.from_config(c, p, 0.005)
    centre = np.array([*it.center_xy, 1.0])
    for off, v_in in (([0.01, -0.005, 0], [0.02, 0.01, -2.75]), ([0.0, 0.02, 0], [-0.03, 0.0, -2.7])):
        n_i, _, vo_i = it(centre + off, np.array(v_in))
        n_p, s_p, vo_p = pd(centre + off, np.array(v_in))
        assert vo_p[2] == pytest.approx(it.v_z, abs=1e-9)
        assert np.allclose(vo_p, frictionless_rebound(np.array(v_in), n_p, s_p * n_p, pd.e))
        assert np.linalg.norm(vo_p[:2] - vo_i[:2]) < 2e-3                    # m/s, within ~0.1 % of |v_out|
