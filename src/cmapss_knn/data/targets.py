"""Target (remaining useful life) construction.

Conventions, which differ between train and test:

* Training engines run to failure. The last recorded cycle is the failure
  cycle, so RUL = (last cycle) - (current cycle) and is 0 on the last row.
* Test engines stop before failure. The NASA RUL file gives the number of
  cycles that remain *after* the last observed cycle, so the implied failure
  cycle is (last observed cycle) + RUL.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .schema import CYCLE_COLUMN, ID_COLUMN, RUL_CLIPPED_COLUMN, RUL_COLUMN


def clip_rul(rul: pd.Series | np.ndarray, cap: int) -> pd.Series | np.ndarray:
    """Piecewise-linear target: ``min(rul, cap)``."""
    if cap <= 0:
        raise ValueError("RUL cap must be positive.")
    return np.minimum(rul, cap)


def add_train_rul(train: pd.DataFrame, cap: int | None = None) -> pd.DataFrame:
    """Return a copy of ``train`` with ``rul`` (and ``rul_clipped`` if ``cap`` is given)."""
    out = train.copy()
    last_cycle = out.groupby(ID_COLUMN)[CYCLE_COLUMN].transform("max")
    out[RUL_COLUMN] = (last_cycle - out[CYCLE_COLUMN]).astype("int64")
    if cap is not None:
        out[RUL_CLIPPED_COLUMN] = clip_rul(out[RUL_COLUMN], cap)
    return out


def truncated_engine_table(test: pd.DataFrame, rul: pd.Series) -> pd.DataFrame:
    """One row per test engine: observed cycles, true RUL and implied lifetime."""
    observed = test.groupby(ID_COLUMN)[CYCLE_COLUMN].max().rename("observed_cycles")
    table = observed.to_frame()
    table["true_rul"] = rul.reindex(table.index).to_numpy()
    table["implied_lifetime"] = table["observed_cycles"] + table["true_rul"]
    table["fraction_observed"] = table["observed_cycles"] / table["implied_lifetime"]
    return table
