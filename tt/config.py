"""Load config/default.yaml (all physical parameters and gains live there)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import yaml

REPO = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = REPO / "config" / "default.yaml"
RESULTS = REPO / "results"


def load_config(path: str | Path | None = None) -> dict:
    with open(path or DEFAULT_CONFIG) as f:
        return yaml.safe_load(f)


def ready_paddle_pos(cfg: dict) -> np.ndarray:
    """Ready-pose paddle centre = hit-plane centre + ready offset (both in the config)."""
    return np.asarray(cfg["hit"]["center"], float) + np.asarray(cfg["ready_pose"]["offset_from_hit_center"], float)
