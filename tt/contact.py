"""M3 — Analytic ball contact with the paddle and the table (impulse model).

Geometry: a contact surface is a slab of half-thickness h around a centre c with
orientation R (columns x, y, n). Ball centre in the surface frame: l = R^T (p - c).
The face the ball is on is s = sign(l_z) from the previous step (two-sided paddle;
the table only has its top face). Signed gap to that face:
    gap = s l_z - (h + r).
Event (one impulse per approach):
    armed  and  gap <= 0  and  (l_x, l_y) inside the face  and  u_n < 0,
where u_n is the normal relative velocity of the ball's contact point. The surface
re-arms once gap > 0 (the ball has separated). A ball that passes through the slab
within one step still has gap <= 0 on its previous side, so tunneling is caught.

Impact law at the contact point (n_s = s n points from the surface to the ball,
v_c = surface velocity at the contact point, u = v + w x (-r n_s) - v_c):
    normal:      J_n = -m (1 + e) u_n                      (=> u_n+ = -e u_n)
    tangential:  rolling impulse  J_roll = -m u_t a / (1 + a),  a = I / (m r^2)
                 J_t = J_roll if |J_roll| <= mu J_n (rolling), else -mu J_n u_t/|u_t| (sliding)
    v+ = v + (J_n n_s + J_t) / m,   w+ = w + (-r n_s) x J_t / I
With mu = 0 the tangential velocity is unchanged (plan §5 frictionless baseline) and
    v+ = v - (1 + e) ((v - v_c) . n_s) n_s        (the Section 6 formula).
Standard rigid-body impact with Coulomb friction (sliding/rolling cases).

Impact-time rewind: the event is detected up to one step after the actual crossing.
The crossing happened tau = gap / u_n ago (0 <= tau <= dt), so we rewind the ball
along its flight, apply the impulse there, and fly it forward by tau again.
The reaction on the arm is ignored (2.7 g ball vs 0.15 kg paddle on a 6 kg arm).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import mujoco
import numpy as np

from tt.aero import AeroParams, BallAero, total_accel
from tt.model import SceneIds


@dataclass(frozen=True)
class SurfaceState:
    """Pose and velocity of a rigid surface at the current time (world frame)."""

    center: np.ndarray        # slab centre (mid-thickness)
    R: np.ndarray             # columns: x, y, normal n
    v: np.ndarray             # linear velocity of the centre
    w: np.ndarray             # angular velocity

    def point_velocity(self, point: np.ndarray) -> np.ndarray:
        return self.v + np.cross(self.w, point - self.center)


@dataclass(frozen=True)
class Surface:
    name: str
    half_thickness: float
    shape: str                       # "disk" (extent = (radius,)) or "rect" (extent = (hx, hy))
    extent: tuple[float, ...]
    two_sided: bool
    restitution: float
    friction: float
    state: Callable[[mujoco.MjModel, mujoco.MjData], SurfaceState]

    def inside(self, lx: float, ly: float) -> bool:
        if self.shape == "disk":
            return lx * lx + ly * ly <= self.extent[0] ** 2
        return abs(lx) <= self.extent[0] and abs(ly) <= self.extent[1]


@dataclass
class ContactEvent:
    t: float                 # estimated impact time
    surface: str
    pos: np.ndarray          # ball centre at impact
    v_in: np.ndarray
    v_out: np.ndarray
    w_in: np.ndarray
    w_out: np.ndarray
    normal: np.ndarray       # n_s (surface -> ball)
    v_surface: np.ndarray    # surface velocity at the contact point
    u_n_in: float            # normal relative velocity before (< 0)
    rolling: bool


def impact(v, w, n_s, v_c, e, mu, mass, radius, inertia):
    """Post-impact (v, w, rolling) for a ball hitting a surface with outward normal n_s."""
    r_c = -radius * n_s                          # ball centre -> contact point
    u = v + np.cross(w, r_c) - v_c
    u_n = u @ n_s
    u_t = u - u_n * n_s
    j_n = -mass * (1.0 + e) * u_n
    a = inertia / (mass * radius ** 2)
    j_roll = -mass * u_t * a / (1.0 + a)
    rolling = np.linalg.norm(j_roll) <= mu * j_n
    if rolling:
        j_t = j_roll
    else:
        ut = np.linalg.norm(u_t)
        j_t = -mu * j_n * u_t / ut if ut > 0 else np.zeros(3)
    v_new = v + (j_n * n_s + j_t) / mass
    w_new = w + np.cross(r_c, j_t) / inertia
    return v_new, w_new, bool(rolling)


def table_restitution_from_rule(h_drop: float, h_rebound: float, k_d: float) -> float:
    """Restitution that reproduces a vertical drop h_drop -> rebound h_rebound with quadratic drag k_d.

    1-D vertical flight with drag a = -g -/+ k v^2:  falling  v^2 = (g/k)(1 - exp(-2 k h)),
    rising  v^2 = (g/k)(exp(2 k h) - 1). Ratio of the two gives e^2 (g cancels).
    """
    if k_d == 0.0:
        return float(np.sqrt(h_rebound / h_drop))
    return float(np.sqrt(np.expm1(2 * k_d * h_rebound) / -np.expm1(-2 * k_d * h_drop)))


class ContactModel:
    """Detects ball impacts with the given surfaces and applies the impulse law. Call between mj_step1 and mj_step2,
    before BallAero.apply()."""

    def __init__(self, m: mujoco.MjModel, ids: SceneIds, aero: BallAero, surfaces: list[Surface],
                 rewind: bool = True):
        self.m, self.ids, self.aero, self.surfaces, self.rewind = m, ids, aero, surfaces, rewind
        self.mass = float(m.body_mass[ids.ball_body])
        self.radius = float(m.geom_size[ids.ball_geom, 0])
        self.inertia = float(m.body_inertia[ids.ball_body, 0])
        self.armed = {s.name: True for s in surfaces}
        self.side = {s.name: None for s in surfaces}
        self.events: list[ContactEvent] = []

    def step(self, m: mujoco.MjModel, d: mujoco.MjData) -> list[ContactEvent]:
        new = []
        for s in self.surfaces:
            p, v, w = self.aero.ball_state(d)
            st = s.state(m, d)
            n = st.R[:, 2]
            l = st.R.T @ (p - st.center)
            side = self.side[s.name]
            if side is None or not s.two_sided:
                side = 1.0 if (l[2] >= 0 or not s.two_sided) else -1.0
            gap = side * l[2] - (s.half_thickness + self.radius)
            if gap > 0:
                self.armed[s.name] = True
                self.side[s.name] = 1.0 if (l[2] >= 0 or not s.two_sided) else -1.0
                continue
            if not (self.armed[s.name] and s.inside(l[0], l[1])):
                continue
            n_s = side * n
            p_contact = p - self.radius * n_s
            v_c = st.point_velocity(p_contact)
            u_n = (v + np.cross(w, -self.radius * n_s) - v_c) @ n_s
            if u_n >= 0:
                continue
            tau = min(gap / u_n, m.opt.timestep) if self.rewind else 0.0   # time since the crossing
            a = total_accel(v, w, self.aero.p)
            p_hit, v_hit = p - v * tau + 0.5 * a * tau ** 2, v - a * tau
            v_c_hit = v_c          # surface velocity taken as constant over tau (<= 1 step)
            v_out, w_out, rolling = impact(v_hit, w, n_s, v_c_hit, s.restitution, s.friction,
                                           self.mass, self.radius, self.inertia)
            a_out = total_accel(v_out, w_out, self.aero.p)
            p_new = p_hit + v_out * tau + 0.5 * a_out * tau ** 2
            v_new = v_out + a_out * tau
            # never leave the ball inside the slab (surface moved by v_c * tau since the crossing)
            l_new = st.R.T @ (p_new - st.center)
            g_new = side * l_new[2] - (s.half_thickness + self.radius)
            if g_new < 0:
                p_new = p_new - g_new * n_s
            self.aero.set_ball_state(m, d, p_new, v_new, w_out)
            self.armed[s.name] = False
            ev = ContactEvent(t=d.time - tau, surface=s.name, pos=p_hit, v_in=v_hit, v_out=v_out, w_in=w.copy(),
                              w_out=w_out, normal=n_s, v_surface=v_c_hit, u_n_in=float(u_n), rolling=rolling)
            self.events.append(ev)
            new.append(ev)
        return new


# ---- surface state providers ------------------------------------------------------------------

def arm_paddle_state(ids: SceneIds) -> Callable[[mujoco.MjModel, mujoco.MjData], SurfaceState]:
    """Paddle on the arm. Velocity from the site Jacobian with the half-step correction
    qdot(t) ~= qvel + qacc dt / 2 (qvel lags by dt/2 in MuJoCo's semi-implicit Euler; qacc is the last step's)."""
    jp = None
    jr = None

    def state(m: mujoco.MjModel, d: mujoco.MjData) -> SurfaceState:
        nonlocal jp, jr
        if jp is None:
            jp, jr = np.zeros((3, m.nv)), np.zeros((3, m.nv))
        mujoco.mj_jacSite(m, d, jp, jr, ids.paddle_site)
        qd = d.qvel + 0.5 * m.opt.timestep * d.qacc
        return SurfaceState(center=d.site_xpos[ids.paddle_site].copy(),
                            R=d.site_xmat[ids.paddle_site].reshape(3, 3).copy(), v=jp @ qd, w=jr @ qd)

    return state


def static_state(center, R=np.eye(3)) -> Callable[[mujoco.MjModel, mujoco.MjData], SurfaceState]:
    st = SurfaceState(center=np.asarray(center, float), R=np.asarray(R, float), v=np.zeros(3), w=np.zeros(3))
    return lambda m, d: st


def paddle_surface(cfg: dict, state, restitution: float | None = None, name: str = "paddle") -> Surface:
    """restitution overrides contact.paddle_restitution (used for the true-vs-nominal mismatch study)."""
    pc, cc = cfg["paddle"], cfg["contact"]
    e = cc["paddle_restitution"] if restitution is None else restitution
    return Surface(name, pc["blade_thickness"] / 2, "disk", (pc["blade_radius"],), True,
                   e, cc["paddle_friction"], state)


def table_surface(cfg: dict, aero_params: AeroParams) -> Surface:
    tc, cc = cfg["table"], cfg["contact"]
    e_t = table_restitution_from_rule(cc["table_rule_drop_height"], cc["table_rule_rebound_height"], aero_params.k_d)
    center = [tc["near_edge_x"] + tc["length"] / 2, 0.0, tc["top_height"] - tc["thickness"] / 2]
    return Surface("table", tc["thickness"] / 2, "rect", (tc["length"] / 2, tc["width"] / 2), False,
                   e_t, cc["table_friction"], static_state(center))
