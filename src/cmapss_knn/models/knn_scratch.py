"""K-Nearest-Neighbours regression written from scratch with NumPy.

The algorithm in four steps (this is the whole model):

1. ``fit``       : store the training points and their targets. Nothing is learned.
2. distances     : for a query point, compute its distance to every training point.
3. neighbours    : keep the k training points with the smallest distances.
4. predict       : average their targets (uniformly, or weighted by 1/distance).

Cost: ``fit`` is O(1). ``predict`` is O(n_train * d) per query for the distances
plus the sort. Memory is O(n_train * d) for the stored data and, transiently,
O(chunk_size * n_train) for a block of distances.

Implementation choices worth knowing (and defending in an interview):

* Distances are computed from coordinate differences, one feature at a time.
  The faster ``|a|^2 + |b|^2 - 2ab`` expansion is avoided on purpose: it suffers
  cancellation, so two identical points can get a tiny non-zero distance, which
  would make distance weighting unstable.
* Neighbours are chosen with a *stable* sort, so equal distances are resolved in
  favour of the training row that appears first. The result is deterministic.
* Queries are processed in chunks so the full ``n_queries x n_train`` matrix is
  never held in memory.
"""

from __future__ import annotations

import numpy as np

from .knn_common import KNNRegressorBase


def pairwise_distances(queries: np.ndarray, points: np.ndarray, metric: str) -> np.ndarray:
    """Distance matrix of shape ``(len(queries), len(points))``.

    * euclidean: ``sqrt(sum_j (q_j - p_j)^2)``  (Minkowski p = 2)
    * manhattan: ``sum_j |q_j - p_j|``          (Minkowski p = 1)
    """
    acc = np.zeros((queries.shape[0], points.shape[0]))
    for j in range(queries.shape[1]):
        diff = queries[:, j, None] - points[None, :, j]
        acc += diff * diff if metric == "euclidean" else np.abs(diff)
    return np.sqrt(acc) if metric == "euclidean" else acc


class ScratchKNNRegressor(KNNRegressorBase):
    """KNN regressor implemented directly in NumPy.

    Args:
        n_neighbors: number of neighbours k.
        metric: ``"euclidean"`` or ``"manhattan"``.
        weights: ``"uniform"`` or ``"distance"``.
        chunk_size: queries processed per block (memory/speed trade-off only;
            it never changes the result).
    """

    def __init__(self, n_neighbors: int = 5, metric: str = "euclidean", weights: str = "uniform", chunk_size: int = 512) -> None:
        super().__init__(n_neighbors, metric, weights)
        if int(chunk_size) < 1:
            raise ValueError("chunk_size must be >= 1.")
        self.chunk_size = int(chunk_size)

    def _fit(self, X: np.ndarray, y: np.ndarray) -> None:
        pass  # lazy learner: the stored copies are the model

    def _kneighbors(self, X: np.ndarray, k: int) -> tuple[np.ndarray, np.ndarray]:
        assert self._X is not None
        n = X.shape[0]
        out_dist = np.empty((n, k))
        out_idx = np.empty((n, k), dtype=np.int64)
        for start in range(0, n, self.chunk_size):
            block = X[start : start + self.chunk_size]
            dist = pairwise_distances(block, self._X, self.metric)
            order = np.argsort(dist, axis=1, kind="stable")[:, :k]  # ties -> lowest training index first
            out_idx[start : start + len(block)] = order
            out_dist[start : start + len(block)] = np.take_along_axis(dist, order, axis=1)
        return out_dist, out_idx
