"""Shared pieces of the two KNN regressors: parameter checks, input checks and
the neighbour-averaging rule.

Both implementations inherit from :class:`KNNRegressorBase`, so they accept
exactly the same parameters, reject exactly the same invalid inputs, and differ
only in *how neighbours are found* (``_kneighbors``) and, for the production
backend, how the prediction is produced (``_predict``).
"""

from __future__ import annotations

import numbers
from abc import ABC, abstractmethod

import numpy as np

from ..exceptions import ModelNotFittedError

SUPPORTED_METRICS: tuple[str, ...] = ("euclidean", "manhattan")
SUPPORTED_WEIGHTS: tuple[str, ...] = ("uniform", "distance")


# --------------------------------------------------------------------------- #
# The KNN prediction rule
# --------------------------------------------------------------------------- #
def aggregate_neighbours(distances: np.ndarray, targets: np.ndarray, weights: str) -> np.ndarray:
    """Turn the k nearest neighbours of each query into one prediction.

    Args:
        distances: ``(n_queries, k)`` distances to the k nearest training points.
        targets: ``(n_queries, k)`` targets of those training points.
        weights: ``"uniform"`` or ``"distance"``.

    * uniform: the plain mean of the k targets.
    * distance: a weighted mean with weight ``1 / distance``, so closer neighbours
      count more. If a query coincides exactly with one or more training points
      (distance 0) the weight would be infinite; the convention, shared with
      scikit-learn, is that those coinciding points get weight 1 and all other
      neighbours weight 0.
    """
    if weights == "uniform":
        return targets.mean(axis=1)
    if weights != "distance":
        raise ValueError(f"weights must be one of {SUPPORTED_WEIGHTS}, got {weights!r}.")
    with np.errstate(divide="ignore"):
        w = 1.0 / distances
    exact = np.isinf(w).any(axis=1)
    if exact.any():
        w[exact] = np.isinf(w[exact]).astype(float)
    return (w * targets).sum(axis=1) / w.sum(axis=1)


# --------------------------------------------------------------------------- #
# Input validation
# --------------------------------------------------------------------------- #
def check_n_neighbors(n_neighbors: object) -> int:
    if isinstance(n_neighbors, bool) or not isinstance(n_neighbors, numbers.Integral):
        raise ValueError(f"n_neighbors must be a positive integer, got {n_neighbors!r}.")
    if n_neighbors < 1:
        raise ValueError(f"n_neighbors must be >= 1, got {n_neighbors}.")
    return int(n_neighbors)


def check_metric(metric: object) -> str:
    if metric not in SUPPORTED_METRICS:
        raise ValueError(f"metric must be one of {SUPPORTED_METRICS}, got {metric!r}.")
    return str(metric)


def check_weights(weights: object) -> str:
    if weights not in SUPPORTED_WEIGHTS:
        raise ValueError(f"weights must be one of {SUPPORTED_WEIGHTS}, got {weights!r}.")
    return str(weights)


def check_X(X: object, *, n_features: int | None = None, name: str = "X") -> np.ndarray:
    """Return ``X`` as a finite 2-D float64 array, or raise ``ValueError``."""
    try:
        arr = np.asarray(X, dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be numeric: {exc}") from exc
    if arr.ndim != 2:
        raise ValueError(f"{name} must be 2-D (n_samples, n_features), got shape {arr.shape}.")
    if arr.shape[0] == 0 or arr.shape[1] == 0:
        raise ValueError(f"{name} must not be empty, got shape {arr.shape}.")
    if not np.isfinite(arr).all():
        raise ValueError(f"{name} contains NaN or infinite values.")
    if n_features is not None and arr.shape[1] != n_features:
        raise ValueError(f"{name} has {arr.shape[1]} features, but the model was fitted with {n_features}.")
    return arr


def check_y(y: object, n_samples: int) -> np.ndarray:
    try:
        arr = np.asarray(y, dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"y must be numeric: {exc}") from exc
    if arr.ndim != 1:
        raise ValueError(f"y must be 1-D, got shape {arr.shape}.")
    if arr.shape[0] != n_samples:
        raise ValueError(f"X has {n_samples} samples but y has {arr.shape[0]}.")
    if not np.isfinite(arr).all():
        raise ValueError("y contains NaN or infinite values.")
    return arr


# --------------------------------------------------------------------------- #
# Base class
# --------------------------------------------------------------------------- #
class KNNRegressorBase(ABC):
    """Common interface: ``fit``, ``kneighbors``, ``predict``."""

    def __init__(self, n_neighbors: int = 5, metric: str = "euclidean", weights: str = "uniform") -> None:
        self.n_neighbors = check_n_neighbors(n_neighbors)
        self.metric = check_metric(metric)
        self.weights = check_weights(weights)
        self._X: np.ndarray | None = None
        self._y: np.ndarray | None = None

    # -- public API ---------------------------------------------------------
    def fit(self, X: object, y: object) -> "KNNRegressorBase":
        X_arr = check_X(X)
        y_arr = check_y(y, X_arr.shape[0])
        if self.n_neighbors > X_arr.shape[0]:
            raise ValueError(f"n_neighbors={self.n_neighbors} exceeds the number of training samples ({X_arr.shape[0]}).")
        self._X, self._y = X_arr.copy(), y_arr.copy()
        self._fit(self._X, self._y)
        return self

    def kneighbors(self, X: object, n_neighbors: int | None = None) -> tuple[np.ndarray, np.ndarray]:
        """Distances and training-row indices of the nearest neighbours, nearest first."""
        X_arr, k = self._prepare_query(X, n_neighbors)
        return self._kneighbors(X_arr, k)

    def predict(self, X: object) -> np.ndarray:
        X_arr, _ = self._prepare_query(X, None)
        return self._predict(X_arr)

    # -- helpers ------------------------------------------------------------
    @property
    def n_features_(self) -> int:
        self._require_fitted()
        assert self._X is not None
        return self._X.shape[1]

    @property
    def n_samples_fit_(self) -> int:
        self._require_fitted()
        assert self._X is not None
        return self._X.shape[0]

    def _require_fitted(self) -> None:
        if self._X is None or self._y is None:
            raise ModelNotFittedError(f"This {type(self).__name__} is not fitted yet; call fit() first.")

    def _prepare_query(self, X: object, n_neighbors: int | None) -> tuple[np.ndarray, int]:
        self._require_fitted()
        assert self._X is not None
        X_arr = check_X(X, n_features=self._X.shape[1])
        k = self.n_neighbors if n_neighbors is None else check_n_neighbors(n_neighbors)
        if k > self._X.shape[0]:
            raise ValueError(f"n_neighbors={k} exceeds the number of training samples ({self._X.shape[0]}).")
        return X_arr, k

    # -- backend hooks ------------------------------------------------------
    @abstractmethod
    def _fit(self, X: np.ndarray, y: np.ndarray) -> None: ...

    @abstractmethod
    def _kneighbors(self, X: np.ndarray, k: int) -> tuple[np.ndarray, np.ndarray]: ...

    def _predict(self, X: np.ndarray) -> np.ndarray:
        """Default: find neighbours, then apply the shared averaging rule."""
        assert self._y is not None
        dist, idx = self._kneighbors(X, self.n_neighbors)
        return aggregate_neighbours(dist, self._y[idx], self.weights)
