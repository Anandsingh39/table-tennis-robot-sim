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


def estimator_metrics(run: JuggleRun, cfg: dict, tti_bins: np.ndarray) -> dict:
    """Estimation quality (Phase 6). Uses the TRUE impact events only for evaluation, never in the loop.

    * steady-state position/velocity RMSE: samples more than settle_after_bounce after a true impact;
    * impact-point prediction error vs time-to-impact: every juggler prediction is compared with the
      actual next paddle impact (horizontal position error, impact-time error);
    * bounce detection: delay from each true impact to the next detection; false / missed detections.
    """
    L, t = run.logs, run.logs["t"]
    ev_t = np.array([e.t for e in run.events])
    settle = cfg["estimator"]["settle_after_bounce"]
    last = np.full_like(t, -np.inf)
    idx = np.searchsorted(ev_t, t, side="right") - 1
    last[idx >= 0] = ev_t[idx[idx >= 0]]
    ok = ~np.isnan(L["est_v"][:, 0])
    steady = ok & (t - last > settle) & (t > settle)
    dv = L["est_v"][steady] - L["ball_v"][steady]
    dp = L["est_p"][steady] - L["ball_p"][steady]

    rows = []
    for tp, t_hit, p_star, _ in run.predictions:
        j = np.searchsorted(ev_t, tp)
        if j < len(ev_t):
            e = run.events[j]
            rows.append((e.t - tp, np.linalg.norm(p_star[:2] - e.pos[:2]), t_hit - e.t))
    rows = np.array(rows).reshape(-1, 3)
    centers = 0.5 * (tti_bins[1:] + tti_bins[:-1])
    med, p90, tmed = (np.full(len(centers), np.nan) for _ in range(3))
    for i in range(len(centers)):
        s = (rows[:, 0] > tti_bins[i]) & (rows[:, 0] <= tti_bins[i + 1])
        if s.any():
            med[i], p90[i] = np.median(rows[s, 1]), np.percentile(rows[s, 1], 90)
            tmed[i] = np.median(np.abs(rows[s, 2]))

    det = np.array(run.detected_bounces or [])
    window = cfg["estimator"]["settle_after_bounce"]
    ev_obs = ev_t[ev_t + window <= t[-1]]          # impacts followed by a full detection window before the run ended
    delays = np.array([det[(det >= te) & (det < te + window)].min() - te for te in ev_obs
                       if np.any((det >= te) & (det < te + window))])
    matched = sum(np.any((det >= te) & (det < te + window)) for te in ev_obs)
    false = sum(not np.any((ev_t <= td) & (ev_t > td - window)) for td in det)
    return dict(vel_rmse=np.sqrt(np.mean(dv ** 2, axis=0)), pos_rmse=np.sqrt(np.mean(dp ** 2, axis=0)),
                vel_rmse_norm=float(np.sqrt(np.mean(np.sum(dv ** 2, axis=1)))),
                tti_centers=centers, pred_err_median=med, pred_err_p90=p90, pred_time_err_median=tmed,
                pred_rows=rows, detect_delays=delays, missed=len(ev_obs) - matched, false_detections=false)
