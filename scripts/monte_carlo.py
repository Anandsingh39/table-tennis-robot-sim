"""Phase 7: Monte Carlo evaluation of the juggler.

Runs every condition in monte_carlo.conditions for seeds 0..n_seeds-1 in parallel and stores one
compact result file per run: results/mc/<condition>/seed_XXX.npz (per-bounce metrics + estimator
metrics, not the per-step logs). Existing files are skipped, so an interrupted sweep resumes.
Figures and tables: scripts/make_figures.py.

Usage (long: run in the background):
    nohup python scripts/monte_carlo.py > results/mc/monte_carlo.log 2>&1 &
    python scripts/monte_carlo.py --conditions "sigma 2 mm" --seeds 5        # quick subset
"""

from __future__ import annotations

import os

os.environ.setdefault("OMP_NUM_THREADS", "1")       # one BLAS thread per worker process
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

import argparse  # noqa: E402
import copy  # noqa: E402
import re  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402
import traceback  # noqa: E402
from concurrent.futures import ProcessPoolExecutor, as_completed  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tt.config import RESULTS, load_config  # noqa: E402

MC_DIR = RESULTS / "mc"


def slug(name: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", name.lower()).strip("_")


def apply_overrides(cfg: dict, overrides: dict) -> dict:
    cfg = copy.deepcopy(cfg)
    for key, val in overrides.items():
        sec, k = key.split(".")
        if val is True and k == "ball_source":
            val = "true"
        cfg[sec][k] = val
    return cfg


def run_one(cfg_path, cond: dict, seed: int, max_bounces: int, out: str) -> tuple[str, int, str]:
    from tt.metrics import estimator_metrics, juggle_metrics
    from tt.sim import run_juggle

    try:
        cfg = apply_overrides(load_config(cfg_path), cond["overrides"])
        t0 = time.time()
        run = run_juggle(cfg, seed, max_bounces)
        mt = juggle_metrics(run, cfg)
        res = dict(bounces=mt["bounces"], failure=run.failure or "", seed=seed, wall=time.time() - t0,
                   apex=mt["apex"], apex_err=mt["apex_err"], lateral=mt["lateral"], impact_xy=mt["impact_xy"],
                   timing=mt["timing"], peak_tau_frac=mt["peak_tau_frac"], saturated=mt["saturated"],
                   sim_time=run.logs["t"][-1])
        if cfg["sensing"]["ball_source"] == "estimate":
            ec = cfg["estimator"]
            em = estimator_metrics(run, cfg, np.arange(ec["tti_bin"] / 2, 0.6 + 1e-9, ec["tti_bin"]))
            i_rep = int(np.argmin(np.abs(em["tti_centers"] - ec["report_time_to_impact"])))
            res.update(vel_rmse=em["vel_rmse"], vel_rmse_norm=em["vel_rmse_norm"], pos_rmse=em["pos_rmse"],
                       tti=em["tti_centers"], pred_err_median=em["pred_err_median"],
                       pred_err_at_report=em["pred_err_median"][i_rep], detect_delays=em["detect_delays"],
                       missed=em["missed"], false_detections=em["false_detections"])
        Path(out).parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(out, **res)
        return cond["name"], seed, f"{mt['bounces']} bounces ({run.failure or 'max'}) in {res['wall']:.0f} s"
    except Exception:                                   # keep the sweep going; record the error
        return cond["name"], seed, "ERROR\n" + traceback.format_exc()


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", type=Path, default=None)
    ap.add_argument("--conditions", nargs="*", default=None, help="subset of condition names")
    ap.add_argument("--seeds", type=int, default=None, help="override monte_carlo.n_seeds")
    ap.add_argument("--workers", type=int, default=None)
    args = ap.parse_args()
    cfg = load_config(args.config)
    mc = cfg["monte_carlo"]
    conds = [c for c in mc["conditions"] if args.conditions is None or c["name"] in args.conditions]
    n_seeds = args.seeds or mc["n_seeds"]
    jobs = []
    for c in conds:
        for s in range(n_seeds):
            out = MC_DIR / slug(c["name"]) / f"seed_{s:03d}.npz"
            if not out.exists():
                jobs.append((c, s, str(out)))
    print(f"{len(conds)} conditions x {n_seeds} seeds: {len(jobs)} runs to do "
          f"({len(conds) * n_seeds - len(jobs)} already done), {args.workers or mc['workers']} workers", flush=True)
    t0 = time.time()
    with ProcessPoolExecutor(args.workers or mc["workers"]) as ex:
        futs = [ex.submit(run_one, args.config, c, s, mc["max_bounces"], out) for c, s, out in jobs]
        for i, f in enumerate(as_completed(futs), 1):
            name, seed, msg = f.result()
            print(f"[{i}/{len(jobs)} {time.time() - t0:6.0f} s] {name:24s} seed {seed:3d}: {msg}", flush=True)
    print(f"done in {time.time() - t0:.0f} s", flush=True)


if __name__ == "__main__":
    main()
