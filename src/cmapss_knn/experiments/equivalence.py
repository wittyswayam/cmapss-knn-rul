"""Measured comparison of the from-scratch and scikit-learn KNN regressors."""

from __future__ import annotations

import itertools
from typing import Callable

import numpy as np
import pandas as pd

from ..models import ScratchKNNRegressor, SklearnKNNRegressor

METRICS = ("euclidean", "manhattan")
WEIGHTS = ("uniform", "distance")


def _row(case: str, k: int, metric: str, weights: str, algo: str, X, y, Q) -> dict:
    a = ScratchKNNRegressor(k, metric, weights).fit(X, y)
    b = SklearnKNNRegressor(k, metric, weights, algo).fit(X, y)
    pa, pb = a.predict(Q), b.predict(Q)
    da, ia = a.kneighbors(Q)
    db, ib = b.kneighbors(Q)
    same_set = np.array([set(r1) == set(r2) for r1, r2 in zip(ia, ib)])
    return {
        "case": case, "metric": metric, "weights": weights, "sklearn_algorithm": algo, "k": k, "n_queries": len(Q),
        "max_abs_pred_diff": float(np.abs(pa - pb).max()), "n_pred_diff_gt_1e-9": int((np.abs(pa - pb) > 1e-9).sum()),
        "share_same_neighbour_set": float(same_set.mean()), "max_abs_dist_diff": float(np.abs(da - db).max()),
    }


def _continuous_case() -> Callable[[], tuple]:
    rng = np.random.default_rng(42)
    return lambda: (rng.normal(size=(500, 12)), rng.normal(size=500) * 30 + 60, rng.normal(size=(150, 12)))


def run_equivalence(real_data: tuple[np.ndarray, np.ndarray, np.ndarray] | None = None) -> pd.DataFrame:
    """Run all cases. ``real_data`` is an optional (X_train, y_train, X_query) of real scaled features."""
    rows: list[dict] = []

    X, y, Q = _continuous_case()()
    for k, m, w, algo in itertools.product((1, 3, 10, 50), METRICS, WEIGHTS, ("brute", "kd_tree")):
        rows.append(_row("continuous random data (no ties)", k, m, w, algo, X, y, Q))

    rng = np.random.default_rng(0)
    Xd = np.vstack([np.tile([1.0, 2.0, 3.0], (5, 1)), rng.normal(size=(60, 3)) * 5 + 10])
    yd = np.concatenate([[10.0, 20.0, 30.0, 40.0, 50.0], rng.uniform(0, 100, 60)])
    Qd = np.vstack([[1.0, 2.0, 3.0], Xd[10:30] + 1e-3])  # one query on the duplicates, others near distinct points
    for k, m, w, algo in itertools.product((3, 5, 8), METRICS, WEIGHTS, ("brute", "kd_tree")):
        rows.append(_row("5 duplicate training points + coinciding query", k, m, w, algo, Xd, yd, Qd))

    gx, gy = np.meshgrid(np.arange(10.0), np.arange(10.0))
    Xl = np.column_stack([gx.ravel(), gy.ravel()])
    yl = np.random.default_rng(5).uniform(0, 100, len(Xl))
    Ql = np.array([[x + dx, yv + dy] for x in (2.0, 5.0) for yv in (3.0, 6.0) for dx, dy in ((0, 0), (0.5, 0.5), (0.5, 0))])
    for k, m, w, algo in itertools.product((1, 2, 3, 4, 8, 12), METRICS, WEIGHTS, ("brute", "kd_tree")):
        rows.append(_row("integer lattice (many tied distances)", k, m, w, algo, Xl, yl, Ql))

    if real_data is not None:
        Xr, yr, Qr = real_data
        for k, m, w in itertools.product((5, 20), METRICS, WEIGHTS):
            rows.append(_row("real FD001 scaled features (one CV fold)", k, m, w, "brute", Xr, yr, Qr))
    return pd.DataFrame(rows)


def summarise_equivalence(df: pd.DataFrame) -> pd.DataFrame:
    g = df.groupby("case", sort=False)
    return pd.DataFrame(
        {
            "configurations": g.size(),
            "max |pred diff|": g["max_abs_pred_diff"].max(),
            "configs with any pred diff > 1e-9": g["n_pred_diff_gt_1e-9"].apply(lambda s: int((s > 0).sum())),
            "min share of identical neighbour sets": g["share_same_neighbour_set"].min(),
            "max |distance diff|": g["max_abs_dist_diff"].max(),
        }
    )
