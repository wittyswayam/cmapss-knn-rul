"""Deliberately naive reference implementations of the metrics, used only to verify the vectorised ones.

They follow the published definitions literally, with plain Python loops and ``math.exp``: no NumPy, no
vectorisation, no shared helper code with ``metrics.py``.

NASA / PHM08 score (Saxena et al., 2008): with d = estimated RUL - true RUL, each engine contributes
``exp(-d/13) - 1`` if d < 0 and ``exp(d/10) - 1`` if d >= 0; the score is the sum over engines.
"""

from __future__ import annotations

import math
from typing import Sequence


def nasa_score_reference(y_true: Sequence[float], y_pred: Sequence[float]) -> float:
    if len(y_true) != len(y_pred):
        raise ValueError("length mismatch")
    total = 0.0
    for true, pred in zip(y_true, y_pred):
        d = pred - true  # positive: the model predicts more life than there is (late warning)
        total += math.exp(-d / 13.0) - 1.0 if d < 0 else math.exp(d / 10.0) - 1.0
    return total


def mean_nasa_reference(y_true: Sequence[float], y_pred: Sequence[float]) -> float:
    return nasa_score_reference(y_true, y_pred) / len(y_true)


def rmse_reference(y_true: Sequence[float], y_pred: Sequence[float]) -> float:
    return math.sqrt(sum((p - t) ** 2 for t, p in zip(y_true, y_pred)) / len(y_true))


def mae_reference(y_true: Sequence[float], y_pred: Sequence[float]) -> float:
    return sum(abs(p - t) for t, p in zip(y_true, y_pred)) / len(y_true)
