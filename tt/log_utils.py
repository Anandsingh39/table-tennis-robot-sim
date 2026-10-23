"""M9 — Logging: accumulate per-step arrays and save them as NPZ (+ CSV for flat columns)."""

from __future__ import annotations

from collections import defaultdict
from pathlib import Path

import numpy as np


class Logger:
    def __init__(self) -> None:
        self._data: dict[str, list] = defaultdict(list)

    def log(self, **values) -> None:
        for k, v in values.items():
            self._data[k].append(np.array(v, copy=True))

    def arrays(self) -> dict[str, np.ndarray]:
        return {k: np.stack(v) for k, v in self._data.items()}

    def save(self, path: str | Path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(path, **self.arrays())
        return path
