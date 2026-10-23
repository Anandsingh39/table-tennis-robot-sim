"""M9 — Juggling metrics from a run's logs and impact events (never from the live sim)."""

from __future__ import annotations

import numpy as np

from tt.sim import JuggleRun


def juggle_metrics(run: JuggleRun, cfg: dict) -> dict:
    jc = cfg["juggle"]
    center = np.asarray(cfg["hit"]["center"][:2], float)
    t = run.logs["t"]
    z = run.logs["ball_p"][:, 2]
    ev_t = np.array([e.t for e in run.events])
    n_b = len(run.events)
    # a "bounce" = an impact followed by a full flight to the next impact (the last impact only starts one)
    n_complete = n_b - 1 if run.failure is None else max(n_b - 1, 0)
    apex = np.array([z[(t > ev_t[k]) & (t < ev_t[k + 1])].max() - run.z_impact for k in range(n_b - 1)])
    lateral = np.array([np.linalg.norm(e.pos[:2] - center) for e in run.events])
    impact_xy = np.array([e.pos[:2] - center for e in run.events]).reshape(-1, 2)
    # timing: actual impact vs the impact time of the last plan made before it
    plan_t = np.array([p.t for p in run.plans])
    plan_hit = np.array([p.t_hit for p in run.plans])
    timing = np.array([e.t - plan_hit[np.searchsorted(plan_t, e.t) - 1] for e in run.events])
    s = jc["transient_bounces"]
    apex_err = apex - jc["apex_height"]
    ss = slice(s, None)
    return dict(
        bounces=n_complete, failure=run.failure, seed=run.seed,
        apex=apex, apex_err=apex_err, lateral=lateral, impact_xy=impact_xy, timing=timing,
        apex_err_mean=float(np.mean(apex_err[ss])) if len(apex_err) > s else np.nan,
        apex_err_std=float(np.std(apex_err[ss])) if len(apex_err) > s else np.nan,
        apex_err_max=float(np.max(np.abs(apex_err[ss]))) if len(apex_err) > s else np.nan,
        lateral_rms=float(np.sqrt(np.mean(lateral[ss] ** 2))) if len(lateral) > s else np.nan,
        lateral_max=float(np.max(lateral[ss])) if len(lateral) > s else np.nan,
        timing_rms=float(np.sqrt(np.mean(timing[ss] ** 2))) if len(timing) > s else np.nan,
        peak_tau_frac=np.abs(run.logs["tau"]).max(axis=0) / run.tau_max,
        saturated=bool(run.logs["sat"].any()),
    )
