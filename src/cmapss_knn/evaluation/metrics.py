"""Evaluation metrics, including the NASA asymmetric score.

Sign convention used everywhere in this project:

    error d = predicted RUL - true RUL

so ``d > 0`` means the model is *optimistic* (it predicts more remaining life
than there really is, i.e. a late maintenance warning) and ``d < 0`` means it is
pessimistic (an early warning).

NASA / PHM08 score (Saxena et al., 2008), per engine::

    s = exp(-d / 13) - 1    if d < 0   (early prediction)
    s = exp( d / 10) - 1    if d >= 0  (late prediction)

Late predictions use the smaller constant (10), so they grow faster and are
penalised more. The official score is the *sum* over engines; ``mean_nasa_score``
divides by the number of predictions so that evaluation sets of different size
can be compared. Lower is better; a perfect model scores 0.
"""

from __future__ import annotations

import numpy as np

EARLY_SCALE = 13.0  # d < 0
LATE_SCALE = 10.0  # d >= 0


def _as_arrays(y_true: object, y_pred: object) -> tuple[np.ndarray, np.ndarray]:
    t = np.asarray(y_true, dtype=float)
    p = np.asarray(y_pred, dtype=float)
    if t.shape != p.shape or t.ndim != 1:
        raise ValueError(f"y_true and y_pred must be 1-D arrays of equal length, got {t.shape} and {p.shape}.")
    if t.size == 0:
        raise ValueError("Cannot compute a metric on an empty array.")
    return t, p


def rmse(y_true: object, y_pred: object) -> float:
    t, p = _as_arrays(y_true, y_pred)
    return float(np.sqrt(np.mean((p - t) ** 2)))


def mae(y_true: object, y_pred: object) -> float:
    t, p = _as_arrays(y_true, y_pred)
    return float(np.mean(np.abs(p - t)))


def nasa_score(y_true: object, y_pred: object) -> float:
    """Sum of the asymmetric NASA penalties (the official definition)."""
    t, p = _as_arrays(y_true, y_pred)
    d = p - t
    return float(np.sum(np.where(d < 0, np.exp(-d / EARLY_SCALE) - 1.0, np.exp(d / LATE_SCALE) - 1.0)))


def nasa_terms(y_true: object, y_pred: object) -> np.ndarray:
    """The individual NASA penalties (one per prediction); ``nasa_score`` is their sum."""
    t, p = _as_arrays(y_true, y_pred)
    d = p - t
    return np.where(d < 0, np.exp(-d / EARLY_SCALE) - 1.0, np.exp(d / LATE_SCALE) - 1.0)


def mean_nasa_score(y_true: object, y_pred: object) -> float:
    t, _ = _as_arrays(y_true, y_pred)
    return nasa_score(y_true, y_pred) / t.size


def regression_metrics(y_true: object, y_pred: object) -> dict[str, float]:
    t, p = _as_arrays(y_true, y_pred)
    return {
        "rmse": rmse(t, p),
        "mae": mae(t, p),
        "bias": float(np.mean(p - t)),  # > 0: optimistic on average
        "late_share": float(np.mean(p - t > 0)),  # share of optimistic (late) predictions
        "nasa_mean": mean_nasa_score(t, p),
    }
