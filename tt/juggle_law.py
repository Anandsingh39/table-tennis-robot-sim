"""M6 — Juggling law (Section 6 of the plan).

Phase 3 provides the inverse of the frictionless impact law; Phase 5 adds the choice of v_out.

For the frictionless impulse (M3), the ball's velocity changes only along the paddle normal n:
    v_out = v_in - (1 + e) ((v_in - v_p) . n) n
Hence (v_out - v_in) is parallel to n, which fixes the paddle normal, and projecting on n:
    v_out . n = -e v_in . n + (1 + e) v_p . n
    =>  n       = (v_out - v_in) / |v_out - v_in|
        v_p . n = (v_out . n + e v_in . n) / (1 + e)
Only the paddle's normal velocity matters; its tangential velocity is free.
"""

from __future__ import annotations

import numpy as np


def paddle_command(v_in: np.ndarray, v_out: np.ndarray, e: float) -> tuple[np.ndarray, float]:
    """Paddle normal n and normal speed v_p . n that turn v_in into v_out."""
    dv = np.asarray(v_out, float) - np.asarray(v_in, float)
    n = dv / np.linalg.norm(dv)
    return n, float((v_out @ n + e * (v_in @ n)) / (1.0 + e))


def frictionless_rebound(v_in: np.ndarray, n: np.ndarray, v_p: np.ndarray, e: float) -> np.ndarray:
    """Forward formula (M3 with mu = 0) for checks."""
    return v_in - (1.0 + e) * ((v_in - v_p) @ n) * n
