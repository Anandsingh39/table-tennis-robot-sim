"""M2 — Ball aerodynamics (gravity + quadratic drag + optional Magnus).

Equation of motion of the ball centre (still air, spin w in world frame):
    m dv/dt = m g  -  1/2 rho C_d A |v| v  +  1/2 rho C_m A r (w x v)
    =>  dv/dt = g - k_d |v| v + k_m (w x v),   k_d = rho C_d A / (2m),  k_m = rho C_m A r / (2m)
with A = pi r^2. MuJoCo applies gravity; we add the aerodynamic force through
data.xfrc_applied (a world-frame force at the body's centre of mass).

Time discretisation (why the half step):
MuJoCo's Euler/implicitfast integrators are semi-implicit:  v+ = v + a dt,  x+ = x + v+ dt.
That is the leapfrog (kick-drift) scheme with the stored velocity living half a step
behind the position: qvel_n = v(t_n - dt/2). Treated that way it is 2nd-order
accurate; read naively (qvel_n = v(t_n)) positions drift by a t dt / 2
(about 5 mm after 1 s of free fall at dt = 1 ms). So:
  * set_ball_state() stores  qvel = v - a(v) dt/2,
  * ball_state() returns     v    = qvel + a(qvel) dt/2,
  * the drag/Magnus force for step n is evaluated at the time-centred velocity
    v(t_n) ~= qvel_n + a(qvel_n) dt/2 (predictor), keeping the scheme 2nd order.
Free-joint qvel layout (verified in tests with mj_objectVelocity): [v_world (3), w_body-local (3)].
"""

from __future__ import annotations

from dataclasses import dataclass

import mujoco
import numpy as np

from tt.model import SceneIds


@dataclass(frozen=True)
class AeroParams:
    mass: float
    radius: float
    gravity: np.ndarray
    k_d: float      # 1/m   drag acceleration coefficient
    k_m: float      # 1     Magnus acceleration coefficient (multiplies w x v)

    @classmethod
    def from_config(cls, cfg: dict, drag_coefficient: float | None = None,
                    magnus_coefficient: float | None = None) -> "AeroParams":
        ac, bc = cfg["aero"], cfg["ball"]
        cd = ac["drag_coefficient"] if drag_coefficient is None else drag_coefficient
        cm = ac["magnus_coefficient"] if magnus_coefficient is None else magnus_coefficient
        m, r, rho = bc["mass"], bc["radius"], ac["air_density"]
        area = np.pi * r ** 2
        return cls(mass=m, radius=r, gravity=np.array(cfg["physics"]["gravity"], float),
                   k_d=rho * cd * area / (2 * m), k_m=rho * cm * area * r / (2 * m))


def aero_accel(v: np.ndarray, w: np.ndarray, p: AeroParams) -> np.ndarray:
    """Aerodynamic acceleration only (no gravity)."""
    return -p.k_d * np.linalg.norm(v) * v + p.k_m * np.cross(w, v)


def total_accel(v: np.ndarray, w: np.ndarray, p: AeroParams) -> np.ndarray:
    return p.gravity + aero_accel(v, w, p)


def rk4_step(y: np.ndarray, w: np.ndarray, p: AeroParams, dt: float) -> np.ndarray:
    """One classic RK4 step of y = [x, v] under gravity + drag + Magnus (constant world spin w)."""

    def f(y):
        return np.concatenate([y[3:], total_accel(y[3:], w, p)])

    k1 = f(y)
    k2 = f(y + 0.5 * dt * k1)
    k3 = f(y + 0.5 * dt * k2)
    k4 = f(y + dt * k3)
    return y + dt / 6 * (k1 + 2 * k2 + 2 * k3 + k4)


def rk4_rollout(pos0, vel0, spin, p: AeroParams, dt: float, n: int) -> tuple[np.ndarray, np.ndarray]:
    """Independent reference / predictor: classic RK4 on y = [x, v], constant world spin.

    Returns (pos, vel) arrays of shape (n+1, 3) at t = 0, dt, ..., n dt.
    """
    w = np.asarray(spin, float)
    y = np.concatenate([pos0, vel0]).astype(float)
    out = np.empty((n + 1, 6))
    out[0] = y
    for i in range(n):
        y = rk4_step(y, w, p, dt)
        out[i + 1] = y
    return out[:, :3], out[:, 3:]


class BallAero:
    """Applies M2 to the MuJoCo ball and converts between MuJoCo's staggered qvel and v(t)."""

    def __init__(self, m: mujoco.MjModel, ids: SceneIds, params: AeroParams, half_step: bool = True):
        self.ids, self.p, self.dt = ids, params, m.opt.timestep
        self.half_step = half_step     # False = naive scheme, kept only to show its error

    def _spin_world(self, d: mujoco.MjData) -> np.ndarray:
        w_local = d.qvel[self.ids.ball_dof + 3:self.ids.ball_dof + 6]
        return d.xmat[self.ids.ball_body].reshape(3, 3) @ w_local

    def ball_state(self, d: mujoco.MjData) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """(position, velocity at d.time, spin in world frame). Needs d.xmat current (after mj_step1/forward)."""
        pos = d.qpos[self.ids.ball_qpos:self.ids.ball_qpos + 3].copy()
        v_stored = d.qvel[self.ids.ball_dof:self.ids.ball_dof + 3].copy()
        w = self._spin_world(d)
        v = v_stored + 0.5 * self.dt * total_accel(v_stored, w, self.p) if self.half_step else v_stored
        return pos, v, w

    def set_ball_state(self, m: mujoco.MjModel, d: mujoco.MjData, pos, vel, spin=(0.0, 0.0, 0.0)) -> None:
        """Write position, velocity at d.time and world-frame spin into qpos/qvel, then refresh kinematics."""
        a, b = self.ids.ball_qpos, self.ids.ball_dof
        d.qpos[a:a + 3] = pos
        mujoco.mj_kinematics(m, d)
        w = np.asarray(spin, float)
        vel = np.asarray(vel, float)
        d.qvel[b:b + 3] = vel - 0.5 * self.dt * total_accel(vel, w, self.p) if self.half_step else vel
        d.qvel[b + 3:b + 6] = d.xmat[self.ids.ball_body].reshape(3, 3).T @ w

    def apply(self, d: mujoco.MjData) -> np.ndarray:
        """Set xfrc_applied for this step (call between mj_step1 and mj_step2). Returns the force."""
        _, v, w = self.ball_state(d)          # time-centred velocity predictor
        force = self.p.mass * aero_accel(v, w, self.p)
        d.xfrc_applied[self.ids.ball_body, :3] = force
        d.xfrc_applied[self.ids.ball_body, 3:] = 0.0
        return force
