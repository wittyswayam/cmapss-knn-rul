"""End-to-end RUL model: causal features -> sensor selection -> scaling -> KNN.

Everything that learns something from data (which sensors are constant, the
feature means and standard deviations used for scaling, the stored neighbours)
is fitted in ``fit`` on the rows it is given and reused unchanged afterwards.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

from ..data.schema import CYCLE_COLUMN, ID_COLUMN, RUL_CLIPPED_COLUMN
from ..exceptions import ModelNotFittedError
from ..features.regimes import RegimeNormalizer
from ..features.sensors import SensorSelector, feature_column_indices
from ..features.windows import WindowConfig, build_features
from .knn_common import KNNRegressorBase
from .knn_scratch import ScratchKNNRegressor
from .knn_sklearn import SklearnKNNRegressor


@dataclass(frozen=True)
class PipelineConfig:
    window: WindowConfig = field(default_factory=WindowConfig)
    sensor_config: str = "A"
    n_neighbors: int = 10
    metric: str = "euclidean"
    weights: str = "uniform"
    backend: str = "sklearn"  # "sklearn" or "scratch"
    normalization: str = "global"  # "global": only the final feature scaling; "regime": per-regime sensor z-scores first
    seed: int = 0  # seeds the regime detector


def make_knn(backend: str, n_neighbors: int, metric: str, weights: str) -> KNNRegressorBase:
    if backend == "sklearn":
        return SklearnKNNRegressor(n_neighbors, metric, weights, algorithm="brute")
    if backend == "scratch":
        return ScratchKNNRegressor(n_neighbors, metric, weights)
    raise ValueError(f"backend must be 'sklearn' or 'scratch', got {backend!r}.")


class RULKNNPipeline:
    """Predicts (clipped) RUL for every cycle of an engine history."""

    def __init__(self, config: PipelineConfig) -> None:
        self.config = config
        self.normalizer_: RegimeNormalizer | None = None
        self._selector: SensorSelector | None = None
        self._scaler: StandardScaler | None = None
        self._knn: KNNRegressorBase | None = None
        self._cols: list[int] = []
        self.feature_names_: list[str] = []

    def fit(self, train: pd.DataFrame, target: str = RUL_CLIPPED_COLUMN, normalizer: RegimeNormalizer | None = None) -> "RULKNNPipeline":
        """Fit on ``train``. ``normalizer`` may be a RegimeNormalizer already fitted on exactly these rows (saves refitting)."""
        cfg = self.config
        if cfg.normalization not in ("global", "regime"):
            raise ValueError(f"normalization must be 'global' or 'regime', got {cfg.normalization!r}.")
        raw_train = train
        if cfg.normalization == "regime":
            self.normalizer_ = normalizer if normalizer is not None else RegimeNormalizer(seed=cfg.seed).fit(train)
            train = self.normalizer_.transform(train)
        self._selector = SensorSelector(cfg.sensor_config).fit(train)
        table = build_features(train, cfg.window)
        self._cols = feature_column_indices(table.columns, self._selector.selected_sensors_)
        self.feature_names_ = [table.columns[i] for i in self._cols]
        X = table.X[:, self._cols]
        self._scaler = StandardScaler().fit(X)
        self._knn = make_knn(cfg.backend, cfg.n_neighbors, cfg.metric, cfg.weights)
        self._knn.fit(self._scaler.transform(X), raw_train[target].to_numpy())
        return self

    def _scaled_features(self, df: pd.DataFrame) -> np.ndarray:
        if self._scaler is None or self._knn is None:
            raise ModelNotFittedError("RULKNNPipeline is not fitted yet; call fit() first.")
        if self.normalizer_ is not None:
            df = self.normalizer_.transform(df)
        table = build_features(df, self.config.window)
        return self._scaler.transform(table.X[:, self._cols])

    def predict_rows(self, df: pd.DataFrame) -> np.ndarray:
        """Prediction for every row, each using only that engine's history up to that row."""
        X = self._scaled_features(df)
        assert self._knn is not None
        return self._knn.predict(X)

    def predict_last_cycle(self, df: pd.DataFrame) -> pd.Series:
        """One prediction per engine, at its last observed cycle (the official test protocol).

        Features are built for every row (they need the whole history), but only the last row of each engine is
        sent to the KNN model; the values equal those of ``predict_rows`` at those rows.
        """
        X = self._scaled_features(df)
        assert self._knn is not None
        last = df.groupby(ID_COLUMN)[CYCLE_COLUMN].idxmax()
        pos = df.index.get_indexer(last.to_numpy())
        return pd.Series(self._knn.predict(X[pos]), index=last.index, name="predicted_rul")

    def kneighbors_rows(self, df: pd.DataFrame, n_neighbors: int | None = None) -> tuple[np.ndarray, np.ndarray]:
        X = self._scaled_features(df)
        assert self._knn is not None
        return self._knn.kneighbors(X, n_neighbors)
