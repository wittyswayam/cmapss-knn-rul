"""Numerical EDA helpers. All functions are pure and operate on training data
unless stated otherwise; nothing here is used to fit a model."""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd

from ..data.schema import CYCLE_COLUMN, ID_COLUMN, RUL_CLIPPED_COLUMN


def engine_lifetimes(train: pd.DataFrame) -> pd.Series:
    """Number of cycles each training engine ran before failure."""
    return train.groupby(ID_COLUMN)[CYCLE_COLUMN].max().rename("lifetime")


def describe_series(values: pd.Series) -> dict[str, float]:
    q = values.quantile([0.25, 0.5, 0.75])
    return {
        "count": int(values.count()),
        "mean": float(values.mean()),
        "std": float(values.std()),
        "min": float(values.min()),
        "q25": float(q[0.25]),
        "median": float(q[0.5]),
        "q75": float(q[0.75]),
        "max": float(values.max()),
    }


def sensor_summary(df: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    """Per-column level statistics, including a constant-column flag."""
    out = df[columns].agg(["mean", "std", "min", "max", "nunique"]).T
    out["nunique"] = out["nunique"].astype(int)
    denom = out["mean"].abs().replace(0, np.nan)
    out["cv"] = out["std"] / denom
    out["is_constant"] = out["nunique"] <= 1
    return out


def within_engine_trend(train: pd.DataFrame, sensors: list[str]) -> pd.DataFrame:
    """Spearman correlation of each sensor with the cycle index, computed within each engine.

    ``mean_rho`` near +/-1 with a small ``std_rho`` means the sensor drifts
    monotonically as the engine ages, consistently across engines.
    Engines in which a sensor is constant have an undefined correlation; they
    are excluded from that sensor's statistics, and ``n_engines`` reports how
    many engines each statistic is based on. ``sign_consistency`` is the share
    of those engines whose correlation has the same sign as the mean.
    """
    rows = {}
    for unit, g in train.groupby(ID_COLUMN):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")  # constant input -> NaN, handled below
            rows[unit] = g[sensors].corrwith(g[CYCLE_COLUMN], method="spearman")
    rho = pd.DataFrame(rows).T
    agrees = (np.sign(rho) == np.sign(rho.mean())).where(rho.notna())
    return pd.DataFrame(
        {
            "n_engines": rho.notna().sum(),
            "mean_rho": rho.mean(),
            "std_rho": rho.std(),
            "mean_abs_rho": rho.abs().mean(),
            "sign_consistency": agrees.mean(),
        }
    )


def rul_correlation(train_with_rul: pd.DataFrame, sensors: list[str], target: str = RUL_CLIPPED_COLUMN) -> pd.Series:
    """Pooled Spearman correlation between each sensor and the (clipped) RUL.

    Pooled across engines, so it mixes within-engine drift with unit-to-unit offsets.
    """
    return train_with_rul[sensors].corrwith(train_with_rul[target], method="spearman").rename("spearman_vs_rul")


def spike_fraction(train: pd.DataFrame, sensors: list[str], window: int, z_threshold: float) -> pd.DataFrame:
    """Share of readings that are isolated spikes relative to the local trend.

    Each sensor is detrended within each engine by subtracting a centred
    rolling median; the residuals are standardised with a robust scale
    (median absolute deviation x 1.4826) and counted when |z| > threshold.
    Detrending first prevents genuine degradation from being mistaken for outliers.
    ``mad_zero`` flags sensors whose residual MAD is 0 (heavily quantised
    channels), for which z-scores are undefined.
    """

    def _detrend(s: pd.Series) -> pd.Series:
        return s - s.rolling(window, center=True, min_periods=1).median()

    resid = train.groupby(ID_COLUMN)[sensors].transform(_detrend)
    med = resid.median()
    mad = (resid - med).abs().median() * 1.4826
    z = (resid - med) / mad.replace(0, np.nan)
    return pd.DataFrame(
        {
            "spike_fraction": (z.abs() > z_threshold).mean().where(mad > 0),
            "spike_count": (z.abs() > z_threshold).sum().where(mad > 0),
            "residual_scale": mad,
            "mad_zero": mad == 0,
        }
    )


def initial_unit_spread(train: pd.DataFrame, sensors: list[str], n_cycles: int) -> pd.DataFrame:
    """How much engines differ from each other during early life.

    Compares the spread (std) of per-engine early-life means with what pure
    cycle-to-cycle noise would produce (within-engine std / sqrt(n_cycles)).
    ``offset_ratio`` ~ 1 means early-life differences are explained by noise;
    values well above 1 indicate persistent unit-to-unit offsets (initial wear,
    manufacturing variation).
    """
    early = train[train[CYCLE_COLUMN] <= n_cycles]
    per_engine = early.groupby(ID_COLUMN)[sensors]
    between = per_engine.mean().std()
    within = per_engine.std().mean()
    expected = within / np.sqrt(n_cycles)
    return pd.DataFrame(
        {
            "between_engine_std": between,
            "within_engine_std": within,
            "noise_only_expected_std": expected,
            "offset_ratio": between / expected.replace(0, np.nan),
        }
    )


def share_outside_train_range(train: pd.DataFrame, test: pd.DataFrame, columns: list[str]) -> pd.Series:
    """Fraction of test readings outside the [min, max] seen in training, per column."""
    lo, hi = train[columns].min(), train[columns].max()
    return ((test[columns] < lo) | (test[columns] > hi)).mean().rename("share_outside_train_range")


def high_correlation_pairs(corr: pd.DataFrame, threshold: float) -> list[tuple[str, str, float]]:
    """Distinct column pairs whose absolute correlation exceeds ``threshold``."""
    cols = list(corr.columns)
    pairs = [
        (cols[i], cols[j], float(corr.iloc[i, j]))
        for i in range(len(cols))
        for j in range(i + 1, len(cols))
        if abs(corr.iloc[i, j]) > threshold
    ]
    return sorted(pairs, key=lambda p: -abs(p[2]))
