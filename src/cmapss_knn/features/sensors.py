"""Sensor selection, fitted on training data only.

Two experimental configurations (Phase 2 decision):

* ``A``: drop sensors that are constant in the training data (no information).
* ``B``: additionally drop ``s6``, a two-valued channel that is constant within
  many engines but not globally constant.

``keep_only`` exists for diagnostics, e.g. asking whether ``s6`` alone predicts
RUL better than a constant guess.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd

from ..data.schema import SENSOR_COLUMNS

SENSOR_CONFIGS: dict[str, tuple[str, ...]] = {"A": (), "B": ("s6",)}


@dataclass
class SensorSelector:
    """Chooses the sensors that become model features.

    Fit on the *training* rows only; ``selected_sensors_`` is then reused
    unchanged on validation or test data.
    """

    config: str = "A"
    keep_only: tuple[str, ...] | None = None
    selected_sensors_: tuple[str, ...] = field(default=(), init=False)
    constant_sensors_: tuple[str, ...] = field(default=(), init=False)

    def __post_init__(self) -> None:
        if self.config not in SENSOR_CONFIGS:
            raise ValueError(f"config must be one of {tuple(SENSOR_CONFIGS)}, got {self.config!r}.")
        if self.keep_only is not None:
            unknown = set(self.keep_only) - set(SENSOR_COLUMNS)
            if unknown:
                raise ValueError(f"Unknown sensors in keep_only: {sorted(unknown)}.")

    def fit(self, train: pd.DataFrame) -> "SensorSelector":
        nunique = train[list(SENSOR_COLUMNS)].nunique()
        self.constant_sensors_ = tuple(s for s in SENSOR_COLUMNS if nunique[s] <= 1)
        dropped = set(self.constant_sensors_) | set(SENSOR_CONFIGS[self.config])
        keep = [s for s in SENSOR_COLUMNS if s not in dropped]
        if self.keep_only is not None:
            keep = [s for s in keep if s in self.keep_only]
        if not keep:
            raise ValueError("No sensors left after selection.")
        self.selected_sensors_ = tuple(keep)
        return self


def feature_column_indices(columns: list[str], sensors: tuple[str, ...] | list[str]) -> list[int]:
    """Positions of feature columns (named ``<sensor>_<stat>``) that belong to ``sensors``."""
    wanted = set(sensors)
    return [i for i, c in enumerate(columns) if c.rsplit("_", 1)[0] in wanted]
