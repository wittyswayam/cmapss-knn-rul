"""Operating-regime identification and regime-wise sensor normalisation.

FD002 and FD004 mix several operating conditions: every flight cycle runs at one
of a few (altitude, Mach, throttle) settings, and sensor readings shift
strongly with the setting. Without correcting for it, a distance-based model
matches engines by the regime they happened to be in, not by their health.

How regimes are found (``RegimeDetector``, fitted on training rows only)
-----------------------------------------------------------------------
1. Standardise the three operational settings (they have very different scales).
2. For k = 2..``max_regimes``, run k-means and compute the silhouette score
   (on a seeded random sample of rows, because silhouette is quadratic in n).
3. If the best silhouette is at least ``min_silhouette`` (default 0.9, i.e. very
   clearly separated groups) accept that k as the number of regimes; otherwise
   conclude there is a single regime. Single-condition subsets (FD001, FD003)
   only have noise in their settings, which never produces such a clean split.
4. Refit k-means with the chosen k on all training rows. A row's regime is the
   nearest centre, which depends only on that row's own settings, so it can be
   assigned at prediction time without looking at any other cycle.

How sensors are normalised (``RegimeNormalizer``)
-------------------------------------------------
For every (regime, sensor) pair, the mean and standard deviation of the
training rows in that regime are stored; a reading is replaced by
``(x - mean_regime) / std_regime``. Everything is learned from the training rows
passed to ``fit``; ``transform`` only applies stored numbers.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

from ..data.schema import OP_SETTING_COLUMNS, SENSOR_COLUMNS
from ..exceptions import ModelNotFittedError


@dataclass
class RegimeDetector:
    max_regimes: int = 10
    min_silhouette: float = 0.9
    sample_size: int = 5000
    seed: int = 0
    n_regimes_: int = field(default=0, init=False)
    silhouette_by_k_: dict[int, float] = field(default_factory=dict, init=False)
    _mean: np.ndarray | None = field(default=None, init=False, repr=False)
    _scale: np.ndarray | None = field(default=None, init=False, repr=False)
    _centers: np.ndarray | None = field(default=None, init=False, repr=False)

    def fit(self, train: pd.DataFrame) -> "RegimeDetector":
        ops = train[list(OP_SETTING_COLUMNS)].to_numpy(dtype=float)
        self._mean = ops.mean(axis=0)
        scale = ops.std(axis=0)
        self._scale = np.where(scale > 0, scale, 1.0)
        Z = (ops - self._mean) / self._scale
        rng = np.random.default_rng(self.seed)
        sample = Z[rng.choice(len(Z), size=min(self.sample_size, len(Z)), replace=False)]
        self.silhouette_by_k_ = {}
        for k in range(2, min(self.max_regimes, len(np.unique(sample, axis=0))) + 1):
            labels = KMeans(k, n_init=5, random_state=self.seed).fit_predict(sample)
            if len(np.unique(labels)) < 2:
                continue
            self.silhouette_by_k_[k] = float(silhouette_score(sample, labels))
        best_k = max(self.silhouette_by_k_, key=self.silhouette_by_k_.get) if self.silhouette_by_k_ else 1
        if best_k > 1 and self.silhouette_by_k_[best_k] >= self.min_silhouette:
            self.n_regimes_ = best_k
            self._centers = KMeans(best_k, n_init=10, random_state=self.seed).fit(Z).cluster_centers_
        else:
            self.n_regimes_ = 1
            self._centers = Z.mean(axis=0, keepdims=True)
        return self

    def predict(self, df: pd.DataFrame) -> np.ndarray:
        """Regime index (0..n_regimes-1) of every row, from that row's own settings."""
        if self._centers is None or self._mean is None or self._scale is None:
            raise ModelNotFittedError("RegimeDetector is not fitted yet.")
        if self.n_regimes_ == 1:
            return np.zeros(len(df), dtype=int)
        Z = (df[list(OP_SETTING_COLUMNS)].to_numpy(dtype=float) - self._mean) / self._scale
        d2 = ((Z[:, None, :] - self._centers[None, :, :]) ** 2).sum(axis=2)
        return d2.argmin(axis=1)


@dataclass
class RegimeNormalizer:
    """Per-regime z-scoring of the sensor channels (statistics from training rows only)."""

    max_regimes: int = 10
    min_silhouette: float = 0.9
    seed: int = 0
    min_rows_per_regime: int = 30
    detector_: RegimeDetector | None = field(default=None, init=False)
    mean_: np.ndarray | None = field(default=None, init=False, repr=False)  # (n_regimes, n_sensors)
    std_: np.ndarray | None = field(default=None, init=False, repr=False)

    @property
    def n_regimes_(self) -> int:
        if self.detector_ is None:
            raise ModelNotFittedError("RegimeNormalizer is not fitted yet.")
        return self.detector_.n_regimes_

    def fit(self, train: pd.DataFrame) -> "RegimeNormalizer":
        det = RegimeDetector(self.max_regimes, self.min_silhouette, seed=self.seed).fit(train)
        regimes = det.predict(train)
        X = train[list(SENSOR_COLUMNS)].to_numpy(dtype=float)
        global_mean, global_std = X.mean(axis=0), X.std(axis=0)
        mean = np.empty((det.n_regimes_, X.shape[1]))
        std = np.empty_like(mean)
        for r in range(det.n_regimes_):
            rows = X[regimes == r]
            if len(rows) < self.min_rows_per_regime:  # too few rows to estimate: fall back to global statistics
                mean[r], std[r] = global_mean, global_std
            else:
                mean[r], std[r] = rows.mean(axis=0), rows.std(axis=0)
        std = np.where(std > 1e-12, std, 1.0)  # constant within a regime: leave unscaled (becomes exactly 0 after centring)
        self.detector_, self.mean_, self.std_ = det, mean, std
        return self

    def regime_table(self, train: pd.DataFrame) -> pd.DataFrame:
        """Row counts and mean operational settings per regime (for reporting)."""
        if self.detector_ is None:
            raise ModelNotFittedError("RegimeNormalizer is not fitted yet.")
        reg = self.detector_.predict(train)
        out = train[list(OP_SETTING_COLUMNS)].groupby(reg).mean()
        out.insert(0, "rows", pd.Series(reg).value_counts().sort_index().to_numpy())
        out.index.name = "regime"
        return out

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Copy of ``df`` with the sensor columns replaced by regime-normalised values."""
        if self.detector_ is None or self.mean_ is None or self.std_ is None:
            raise ModelNotFittedError("RegimeNormalizer is not fitted yet.")
        regimes = self.detector_.predict(df)
        X = df[list(SENSOR_COLUMNS)].to_numpy(dtype=float)
        out = df.copy()
        out[list(SENSOR_COLUMNS)] = (X - self.mean_[regimes]) / self.std_[regimes]
        return out
