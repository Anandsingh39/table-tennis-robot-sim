"""Transition extraction for the GPU game model (tt.selfplay_data)."""

import numpy as np

from tt.selfplay_data import point_transitions, stack


def _dec(t, obs0, act):
    return dict(t=t, obs=[obs0] * 7, act=act, fallback=False)


def _res(fault, by, dec_a, dec_b, hits):
    return dict(fault=fault, by=by, decisions={"A": dec_a, "B": dec_b}, hit_log=hits, rail_ready=0.1)


def test_receiver_misses():
    """A serves back, B decides and misses: A continue (o' = B's obs), B lose; recovery targets carried over."""
    res = _res("missed", "B", [_dec(0.0, 1.0, [0, 0.9, 0.4, 0.3])], [_dec(0.6, 2.0, [0, 0.9, 0.4, -0.2])], [("A", 0.5)])
    tr = point_transitions(res)
    assert [x["outcome"] for x in tr] == ["continue", "lose"]
    assert np.allclose(tr[0]["next_obs"], 2.0) and np.isnan(tr[1]["next_obs"]).all()
    assert tr[0]["r_recv"] == 0.1 and tr[1]["r_recv"] == 0.3          # B's default, then A's chosen target


def test_hitter_fault_and_opponent_fault_without_decision():
    net = _res("net", "A", [_dec(0.0, 1.0, [0, 0.9, 0.4, 0.0])], [], [("A", 0.5)])
    assert [x["outcome"] for x in point_transitions(net)] == ["lose"]
    early = _res("double bounce", "B", [_dec(0.0, 1.0, [0, 0.9, 0.4, 0.0])], [], [("A", 0.5)])
    assert [x["outcome"] for x in point_transitions(early)] == ["opp_fault"]


def test_three_dim_actions_and_truncation():
    res = _res("max hits", None, [_dec(0.0, 1.0, [0, 0.9, 0.4])], [_dec(0.6, 2.0, [0, 0.9, 0.4])],
               [("A", 0.5), ("B", 1.9)])
    tr = point_transitions(res)
    assert [x["outcome"] for x in tr] == ["continue", None]
    assert tr[0]["act"][3] == 0.1 and tr[1]["hits_before"] == 1       # 3-D action: recovery = ready rail
    arr = stack(tr, seed=[7, 7])
    assert arr["obs"].shape == (2, 7) and arr["act"].shape == (2, 4) and list(arr["outcome"]) == [0, -1]


def test_error_point_gives_nothing():
    assert point_transitions(dict(fault="error: x", by=None, decisions={}, hit_log=[])) == []
