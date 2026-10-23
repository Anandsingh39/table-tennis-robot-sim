"""M5 — Ball predictor.

RK4 rollout of the nominal flight model (tt.aero: gravity + drag + Magnus) from the current ball
state until an event. Inside the step where the event happens the motion is taken as constant-
acceleration (a = dv/dt over the step) and the quadratic is solved for the event time (exact in
vacuum; with drag the error is O(dt^3), far below a micrometre). Linear interpolation would be off
by up to a dt^2 / 8 ~ 30 um (~9 us) at dt = 5 ms.

Events:
  crossing()        juggling: ball centre descends through z = z_cross;
  plane_crossing()  rally: ball centre crosses a plane coord[axis] = value moving in `direction`,
                    optionally only after a bounce on the table (TableBounce: the same impulse law as
                    M3, frictionless v_z -> -e_t v_z, or with table friction the spin-dependent bounce).
The predictor uses its own nominal parameters, so model mismatch is possible.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from tt.aero import AeroParams, rk4_step


@dataclass(frozen=True)
class Prediction:
    t_rel: float          # time from now to the event [s]
    pos: np.ndarray       # ball centre at the event
    vel: np.ndarray       # ball velocity at the event (v_in)
    bounced: bool = False
    spin: np.ndarray | None = None   # ball spin (world frame) at the event


@dataclass(frozen=True)
class TableBounce:
    x_range: tuple[float, float]
    y_range: tuple[float, float]
    z_contact: float      # ball centre height at contact = table top + ball radius
    e: float              # table restitution (normal)
    mu: float = 0.0       # table friction (Coulomb, sliding/rolling as M3); 0 = frictionless
    mass: float = 0.0027
    radius: float = 0.02
    inertia: float = 2.0 / 3.0 * 0.0027 * 0.02 ** 2

    @classmethod
    def from_config(cls, cfg: dict, e_t: float) -> "TableBounce":
        tc, bc = cfg["table"], cfg["ball"]
        x0 = tc["near_edge_x"]
        inertia = (2.0 / 3.0 if bc["hollow_shell"] else 0.4) * bc["mass"] * bc["radius"] ** 2
        return cls((x0, x0 + tc["length"]), (-tc["width"] / 2, tc["width"] / 2),
                   tc["top_height"] + bc["radius"], e_t, cfg["contact"]["table_friction"],
                   bc["mass"], bc["radius"], inertia)

    def bounce(self, v: np.ndarray, w: np.ndarray):
        """Post-bounce (v, w): the same impulse law as the simulator's table contact (tt.contact.impact)."""
        if self.mu == 0.0:
            return np.array([v[0], v[1], -self.e * v[2]]), w
        from tt.contact import impact

        v_new, w_new, _ = impact(v, w, np.array([0.0, 0.0, 1.0]), np.zeros(3), self.e, self.mu,
                                 self.mass, self.radius, self.inertia)
        return v_new, w_new

    def on_table(self, x: float, y: float) -> bool:
        return self.x_range[0] <= x <= self.x_range[1] and self.y_range[0] <= y <= self.y_range[1]


def _event_time_in_step(y: np.ndarray, y_next: np.ndarray, dt: float, idx: int, value: float) -> float:
    """Time tau in [0, dt] at which component idx (0..2) of the position reaches `value`."""
    acc = (y_next[3:] - y[3:]) / dt
    a, v, d = acc[idx], y[3 + idx], value - y[idx]
    if abs(a) > 1e-12:
        disc = np.sqrt(max(v * v + 2 * a * d, 0.0))
        roots = [r for r in ((-v + disc) / a, (-v - disc) / a) if r >= 0.0]
        tau = min(roots) if roots else dt
    else:
        tau = d / v if v != 0 else dt
    return min(max(tau, 0.0), dt)


def _state_at(y: np.ndarray, y_next: np.ndarray, dt: float, tau: float) -> np.ndarray:
    acc = (y_next[3:] - y[3:]) / dt
    return np.concatenate([y[:3] + y[3:] * tau + 0.5 * acc * tau ** 2, y[3:] + acc * tau])


class BallPredictor:
    def __init__(self, params: AeroParams, dt: float, horizon: float):
        self.p, self.dt, self.horizon = params, dt, horizon

    def crossing(self, pos, vel, spin, z_cross: float) -> Prediction | None:
        """Next descending crossing of z_cross within the horizon, or None (e.g. already below and falling)."""
        y = np.concatenate([pos, vel]).astype(float)
        w = np.asarray(spin, float)
        if y[2] <= z_cross and y[5] <= 0:
            return None
        t = 0.0
        while t < self.horizon:
            y_next = rk4_step(y, w, self.p, self.dt)
            if y_next[5] < 0 and y[2] > z_cross >= y_next[2]:
                tau = _event_time_in_step(y, y_next, self.dt, 2, z_cross)
                yc = _state_at(y, y_next, self.dt, tau)
                return Prediction(t + tau, yc[:3], yc[3:])
            y, t = y_next, t + self.dt
        return None

    def _step_with_table(self, y, w, table: TableBounce | None):
        """One RK4 step; if the ball reaches the table top inside the step, bounce there (spin-dependent
        with table friction). Returns (y_next, contact, w_next); contact = ball centre at the bounce or None."""
        y_next = rk4_step(y, w, self.p, self.dt)
        if (table is not None and y_next[5] < 0 and y[2] > table.z_contact >= y_next[2]):
            tau = _event_time_in_step(y, y_next, self.dt, 2, table.z_contact)
            yc = _state_at(y, y_next, self.dt, tau)
            if table.on_table(yc[0], yc[1]):
                contact = yc[:3].copy()
                yc[3:], w = table.bounce(yc[3:], w)
                rest = self.dt - tau
                return (rk4_step(yc, w, self.p, rest) if rest > 1e-12 else yc), contact, w
        return y_next, None, w

    def plane_crossing(self, pos, vel, spin, axis: int, value: float, direction: float,
                       table: TableBounce | None = None, require_bounce: bool = False,
                       already_bounced: bool = False) -> Prediction | None:
        """Next crossing of coord[axis] = value while moving with sign(direction), within the horizon.

        With require_bounce, a crossing only counts after a table bounce (already_bounced: a bounce
        has happened before `pos`, e.g. observed by the estimator)."""
        y = np.concatenate([pos, vel]).astype(float)
        w = np.asarray(spin, float)
        bounced, t = already_bounced, 0.0
        while t < self.horizon:
            y_next, contact, w = self._step_with_table(y, w, table)
            bounced |= contact is not None
            crossed = (y[axis] - value) * direction < 0 <= (y_next[axis] - value) * direction
            if crossed and (bounced or not require_bounce):
                tau = _event_time_in_step(y, y_next, self.dt, axis, value)
                yc = _state_at(y, y_next, self.dt, tau)
                return Prediction(t + tau, yc[:3], yc[3:], bounced, w.copy())
            if y_next[2] < (table.z_contact - 0.5 if table else -np.inf):
                return None                       # fell well below the table: no hit possible
            y, t = y_next, t + self.dt
        return None

    def rollout(self, pos, vel, spin, t_end: float, table: TableBounce | None = None):
        """States at t = 0, dt, ... up to t_end (with table bounces). Returns (t, Y[k, 6], bounce flags)."""
        y = np.concatenate([pos, vel]).astype(float)
        w = np.asarray(spin, float)
        ts, ys, bs = [0.0], [y.copy()], [False]
        t = 0.0
        while t < t_end - 1e-12:
            y, contact, w = self._step_with_table(y, w, table)
            t += self.dt
            ts.append(t)
            ys.append(y.copy())
            bs.append(contact is not None)
        return np.array(ts), np.array(ys), np.array(bs)
