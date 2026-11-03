"""Stroke-level transitions of real rally points (full MuJoCo + MPC) for the GPU game model (scripts/train_game_model.py).

One transition per strategy decision (a player choosing its shot for an incoming ball, tt.rally2.RallyPlayer):
    x = (observation o [7], action a = (y_c, z_c, d_b, rail recovery target) [4], receiver's rail recovery target [1])
    outcome of the decider's stroke:
        "continue"   the opponent made the next decision (o' = its observation),
        "lose"       the point ended with a fault by the decider (missed / net / out / short / double bounce ...),
        "opp_fault"  the opponent faulted before it could decide,
        None         truncated (hit cap, time limit) -> not used for the outcome model.
The receiver's recovery target is in the state because it moves the receiver's rail before its next decision, but it
is not in the decider's observation (it was chosen at the receiver's previous decision).
"""

from __future__ import annotations

import numpy as np

OUTCOMES = ("continue", "lose", "opp_fault")
N_OBS, N_ACT = 7, 4
NO_FAULT = (None, "max hits", "time limit")


def point_transitions(res: dict) -> list[dict]:
    """Transitions of one point, from a tt.strategy.play_point result."""
    if str(res.get("fault", "")).startswith("error"):
        return []
    rail0 = res.get("rail_ready", 0.0)
    dec = sorted((d["t"], who, d) for who in ("A", "B") for d in res["decisions"].get(who, []))
    recover = {"A": rail0, "B": rail0}            # each player's current rail recovery target
    fault, by = res["fault"], res["by"]
    out = []
    for i, (t, who, d) in enumerate(dec):
        other = "B" if who == "A" else "A"
        nxt = next((x for x in dec[i + 1:] if x[1] == other), None)
        later_own = any(x[1] == who for x in dec[i + 1:])
        act = np.asarray(d["act"], float)
        if len(act) == 3:
            act = np.append(act, rail0)
        if fault not in NO_FAULT and by == who and not later_own:
            outcome = "lose"
        elif nxt is not None:
            outcome = "continue"
        elif fault not in NO_FAULT and by == other:
            outcome = "opp_fault"
        else:
            outcome = None
        out.append(dict(obs=np.asarray(d["obs"], float), act=act, r_recv=float(recover[other]), outcome=outcome,
                        next_obs=np.asarray(nxt[2]["obs"], float) if (outcome == "continue") else np.full(N_OBS, np.nan),
                        who=who, t=float(t), fallback=bool(d["fallback"]),
                        hits_before=sum(1 for w, th in res.get("hit_log", []) if th < t)))
        recover[who] = float(act[3])
    return out


def stack(transitions: list[dict], **meta) -> dict:
    """Arrays for an .npz shard."""
    n = len(transitions)
    code = {o: k for k, o in enumerate(OUTCOMES)}
    arr = dict(obs=np.array([x["obs"] for x in transitions]).reshape(n, N_OBS),
               act=np.array([x["act"] for x in transitions]).reshape(n, N_ACT),
               r_recv=np.array([x["r_recv"] for x in transitions], float),
               outcome=np.array([code.get(x["outcome"], -1) for x in transitions], int),
               next_obs=np.array([x["next_obs"] for x in transitions]).reshape(n, N_OBS),
               fallback=np.array([x["fallback"] for x in transitions], bool),
               hits_before=np.array([x["hits_before"] for x in transitions], int))
    for k, v in meta.items():
        arr[k] = np.asarray(v)
    return arr
