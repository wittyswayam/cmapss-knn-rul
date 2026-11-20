"""Causal sliding-window features and per-engine baselining.

Every sample is one (engine, cycle t) pair, and its features may only depend on
the observations of that engine at cycles 1..t. Nothing from the future, from the
rest of the engine's life, or from other engines is used. ``tests/`` checks this
directly: features computed on a history truncated at t equal the features of
the full history at t.

Short histories (fewer than ``window`` cycles seen so far) are never dropped.
Two strategies are available:

``adaptive``   the window shrinks to the cycles available, so a statistic at
               cycle 7 describes cycles 1..7. No invented data, but statistics
               are computed from fewer points (noisier, slope undefined for
               a single point and set to 0).
``pad_mean``   missing cycles are filled with the mean of the observations that
               do exist in that window: the window mean equals the adaptive one,
               but the window always has full length, so the slope is damped
               less than with ``pad_first`` when the true trend is flat and
               more when it is not. Added in Phase 3 as an alternative to test.
``pad_first``  the history is extended backwards by repeating the first
               observation until the window is full. Every window has the same
               length, but the padding fabricates a flat segment, which biases
               the mean towards the first reading and flattens the slope.

Per-engine baselining (``baseline="causal"``) subtracts, at cycle t, the mean of
the first ``min(t, baseline_cycles)`` observations. After ``baseline_cycles``
cycles the baseline is frozen; before that it is the mean of everything seen so far.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from numpy.lib.stride_tricks import sliding_window_view

from ..data.schema import CYCLE_COLUMN, ID_COLUMN, SENSOR_COLUMNS

STATS = ("last", "mean", "std", "slope")
HISTORY_STRATEGIES = ("adaptive", "pad_first", "pad_mean")
BASELINES = ("none", "causal")


@dataclass(frozen=True)
class WindowConfig:
    window: int = 30
    stats: tuple[str, ...] = ("mean", "slope")
    history: str = "adaptive"
    baseline: str = "none"
    baseline_cycles: int = 20

    def __post_init__(self) -> None:
        if self.window < 1:
            raise ValueError("window must be >= 1.")
        if not self.stats or any(s not in STATS for s in self.stats):
            raise ValueError(f"stats must be a non-empty subset of {STATS}, got {self.stats}.")
        if self.history not in HISTORY_STRATEGIES:
            raise ValueError(f"history must be one of {HISTORY_STRATEGIES}, got {self.history!r}.")
        if self.baseline not in BASELINES:
            raise ValueError(f"baseline must be one of {BASELINES}, got {self.baseline!r}.")
        if self.baseline_cycles < 1:
            raise ValueError("baseline_cycles must be >= 1.")


@dataclass(frozen=True, eq=False)
class FeatureTable:
    """Feature matrix with one row per input row (same order)."""

    X: np.ndarray
    columns: list[str]
    index: pd.DataFrame  # columns: unit, cycle


def causal_baseline(values: np.ndarray, n_cycles: int) -> np.ndarray:
    """Per-row baseline using only observations up to that row.

    Row t (1-based) gets the mean of rows 1..min(t, n_cycles).
    """
    t = np.arange(1, values.shape[0] + 1)
    m = np.minimum(t, n_cycles)
    return np.cumsum(values, axis=0)[m - 1] / m[:, None]


def _window_stats(values: np.ndarray, cfg: WindowConfig) -> dict[str, np.ndarray]:
    """Statistics of the trailing window ending at each row. ``values`` is (T, S)."""
    T, S = values.shape
    W = cfg.window
    if cfg.history in ("adaptive", "pad_mean"):
        front = np.full((W - 1, S), np.nan)
    else:
        front = np.repeat(values[:1], W - 1, axis=0)
    padded = np.vstack([front, values])
    win = sliding_window_view(padded, W, axis=0)  # (T, S, W); window t ends at row t
    if cfg.history == "pad_mean":
        win = np.where(np.isnan(win), np.nanmean(win, axis=-1, keepdims=True), win)
    valid = ~np.isnan(win)
    n = valid.sum(axis=-1)  # >= 1: the last element is always a real observation

    mean = np.where(valid, win, 0.0).sum(axis=-1) / n
    out: dict[str, np.ndarray] = {"last": values.copy(), "mean": mean}
    if "std" in cfg.stats:
        dev = np.where(valid, win - mean[..., None], 0.0)
        out["std"] = np.sqrt((dev**2).sum(axis=-1) / n)
    if "slope" in cfg.stats:
        pos = np.arange(W, dtype=float)
        pos_mean = (np.where(valid, pos, 0.0)).sum(axis=-1) / n
        pc = pos - pos_mean[..., None]
        num = np.where(valid, pc * (win - mean[..., None]), 0.0).sum(axis=-1)
        den = np.where(valid, pc**2, 0.0).sum(axis=-1)
        out["slope"] = np.divide(num, den, out=np.zeros_like(num), where=den > 0)  # ordinary least squares slope per cycle
    return out


def _engine_features(values: np.ndarray, cfg: WindowConfig) -> np.ndarray:
    if cfg.baseline == "causal":
        values = values - causal_baseline(values, cfg.baseline_cycles)
    stats = _window_stats(values, cfg)
    # column order: sensor-major, e.g. s2_mean, s2_slope, s3_mean, ...
    cols = [stats[st][:, j] for j in range(values.shape[1]) for st in cfg.stats]
    return np.column_stack(cols)


def build_features(df: pd.DataFrame, cfg: WindowConfig, sensors: tuple[str, ...] = SENSOR_COLUMNS) -> FeatureTable:
    """Causal features for every row of ``df``.

    ``df`` must contain whole engine histories sorted by (unit, cycle) with
    cycles 1, 2, 3, ...; this is checked, not assumed.
    """
    columns = [f"{s}_{st}" for s in sensors for st in cfg.stats]
    blocks: list[np.ndarray] = []
    for unit, g in df.groupby(ID_COLUMN, sort=False):
        cycles = g[CYCLE_COLUMN].to_numpy()
        if not np.array_equal(cycles, np.arange(1, len(cycles) + 1)):
            raise ValueError(f"Engine {unit}: cycles must be 1..T in order, histories must start at cycle 1.")
        blocks.append(_engine_features(g[list(sensors)].to_numpy(dtype=float), cfg))
    X = np.vstack(blocks)
    return FeatureTable(X=X, columns=columns, index=df[[ID_COLUMN, CYCLE_COLUMN]].reset_index(drop=True))
