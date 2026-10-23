"""Load config/default.yaml (all physical parameters and gains live there)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import yaml

REPO = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = REPO / "config" / "default.yaml"
RESULTS = REPO / "results"


def _deep_merge(base: dict, over: dict) -> dict:
    out = dict(base)
    for k, v in over.items():
        out[k] = _deep_merge(out[k], v) if isinstance(v, dict) and isinstance(out.get(k), dict) else v
    return out


def load_config(path: str | Path | None = None) -> dict:
    """Load a config. A file with `extends: <other.yaml>` (relative to its folder) is deep-merged on top of it."""
    path = Path(path or DEFAULT_CONFIG)
    with open(path) as f:
        cfg = yaml.safe_load(f)
    parent = cfg.pop("extends", None)
    return _deep_merge(load_config(path.parent / parent), cfg) if parent else cfg


RALLY_CONFIG = REPO / "config" / "rally.yaml"


def ready_paddle_pos(cfg: dict) -> np.ndarray:
    """Ready-pose paddle centre = hit-plane centre + ready offset (both in the config)."""
    return np.asarray(cfg["hit"]["center"], float) + np.asarray(cfg["ready_pose"]["offset_from_hit_center"], float)
