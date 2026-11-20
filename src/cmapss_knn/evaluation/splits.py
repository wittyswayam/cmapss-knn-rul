"""Engine-level splitting and evaluation-point sampling.

All splits are by *engine*, never by row: consecutive cycles of one engine are
near-duplicates, so putting some in training and some in validation would leak.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def holdout_engines(units: np.ndarray, n_holdout: int, seed: int) -> tuple[np.ndarray, np.ndarray]:
    """Split engine ids into (tuning, holdout), reproducibly."""
    units = np.unique(np.asarray(units))
    if not 0 < n_holdout < units.size:
        raise ValueError(f"n_holdout must be between 1 and {units.size - 1}, got {n_holdout}.")
    rng = np.random.default_rng(seed)
    shuffled = rng.permutation(units)
    return np.sort(shuffled[n_holdout:]), np.sort(shuffled[:n_holdout])


def engine_folds(units: np.ndarray, n_splits: int, seed: int) -> list[tuple[np.ndarray, np.ndarray]]:
    """K-fold over engines: a list of (train_units, validation_units) pairs.

    Every engine appears in exactly one validation fold; fold sizes differ by at most one.
    """
    units = np.unique(np.asarray(units))
    if not 2 <= n_splits <= units.size:
        raise ValueError(f"n_splits must be between 2 and {units.size}, got {n_splits}.")
    shuffled = np.random.default_rng(seed).permutation(units)
    parts = np.array_split(shuffled, n_splits)
    return [(np.sort(np.concatenate([p for j, p in enumerate(parts) if j != i])), np.sort(parts[i])) for i in range(n_splits)]


def sample_cutoffs(lifetimes: pd.Series, n_per_engine: int, min_cutoff: int, seed: int) -> pd.DataFrame:
    """Random truncation points that imitate the official test protocol.

    The official test files stop each engine at some cycle before failure and
    ask for the RUL there. For validation we do the same: for each engine we
    draw ``n_per_engine`` distinct cutoffs uniformly from
    ``min_cutoff .. lifetime - 1`` (capped to what the engine allows). The
    prediction at a cutoff uses only the first ``cutoff`` cycles, and the true
    RUL is ``lifetime - cutoff``.

    Returns a DataFrame with columns ``unit`` and ``cutoff``.
    """
    rng = np.random.default_rng(seed)
    rows: list[tuple[int, int]] = []
    for unit, life in lifetimes.sort_index().items():
        lo = min(min_cutoff, int(life) - 1)
        pool = np.arange(lo, int(life))  # lifetime - 1 is the last cycle with RUL >= 1
        chosen = rng.choice(pool, size=min(n_per_engine, pool.size), replace=False)
        rows.extend((int(unit), int(c)) for c in np.sort(chosen))
    return pd.DataFrame(rows, columns=["unit", "cutoff"])
