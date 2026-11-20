"""Production KNN regressor: a thin, validated wrapper around scikit-learn.

It exposes the same constructor and methods as :class:`ScratchKNNRegressor`, so
the two can be swapped one-for-one. All input checks are shared (see
``knn_common``); the neighbour search and the prediction itself are done by
``sklearn.neighbors.KNeighborsRegressor``.
"""

from __future__ import annotations

import numpy as np
from sklearn.neighbors import KNeighborsRegressor

from .knn_common import KNNRegressorBase

_ALGORITHMS = ("auto", "brute", "kd_tree", "ball_tree")


class SklearnKNNRegressor(KNNRegressorBase):
    """KNN regressor backed by scikit-learn.

    Args:
        n_neighbors, metric, weights: as in :class:`ScratchKNNRegressor`.
        algorithm: neighbour-search structure passed to scikit-learn
            (``"auto"``, ``"brute"``, ``"kd_tree"`` or ``"ball_tree"``). It affects
            speed, and in rare tie situations which of several equally distant
            points is returned, but not the model definition.
    """

    def __init__(self, n_neighbors: int = 5, metric: str = "euclidean", weights: str = "uniform", algorithm: str = "auto") -> None:
        super().__init__(n_neighbors, metric, weights)
        if algorithm not in _ALGORITHMS:
            raise ValueError(f"algorithm must be one of {_ALGORITHMS}, got {algorithm!r}.")
        self.algorithm = algorithm
        self._model: KNeighborsRegressor | None = None

    def _fit(self, X: np.ndarray, y: np.ndarray) -> None:
        self._model = KNeighborsRegressor(
            n_neighbors=self.n_neighbors, metric=self.metric, weights=self.weights, algorithm=self.algorithm
        ).fit(X, y)

    def _kneighbors(self, X: np.ndarray, k: int) -> tuple[np.ndarray, np.ndarray]:
        assert self._model is not None
        dist, idx = self._model.kneighbors(X, n_neighbors=k)
        return dist, idx

    def _predict(self, X: np.ndarray) -> np.ndarray:
        assert self._model is not None
        return self._model.predict(X)
