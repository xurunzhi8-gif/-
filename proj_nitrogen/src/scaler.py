from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd


class TargetScaler:
    def __init__(self, targets: list[str]) -> None:
        self.targets = targets
        self.mean: np.ndarray | None = None
        self.std: np.ndarray | None = None

    def fit(self, df: pd.DataFrame) -> None:
        values = df[self.targets].to_numpy(dtype=float)
        self.mean = np.nanmean(values, axis=0)
        self.std = np.nanstd(values, axis=0)
        self.std = np.where(self.std == 0, 1.0, self.std)

    def transform_with_mask(self, values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        if self.mean is None or self.std is None:
            raise ValueError("Scaler is not fitted.")
        mask = ~np.isnan(values)
        scaled = (values - self.mean) / self.std
        scaled[~mask] = 0.0
        return scaled.astype(np.float32), mask.astype(np.float32)

    def inverse(self, values: np.ndarray) -> np.ndarray:
        if self.mean is None or self.std is None:
            raise ValueError("Scaler is not fitted.")
        return values * self.std + self.mean

    def save(self, path: Path) -> None:
        if self.mean is None or self.std is None:
            raise ValueError("Scaler is not fitted.")
        payload = {"targets": self.targets, "mean": self.mean.tolist(), "std": self.std.tolist()}
        path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    def load(self, path: Path) -> None:
        payload = json.loads(path.read_text(encoding="utf-8"))
        self.targets = payload["targets"]
        self.mean = np.array(payload["mean"], dtype=float)
        self.std = np.array(payload["std"], dtype=float)
