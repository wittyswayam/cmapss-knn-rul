"""Short-history robustness probe.

Phase 2 found that padding a short history with its first observation (``pad_first``) beats a
shrinking window (``adaptive``). But every test history in C-MAPSS starts at the engine's first
cycle, so a short history always means "early, healthy life". Does the finding survive when the
engine is *already degraded* at the first observed cycle (monitoring started mid-life)?

For each validation engine, an artificial history is cut out of its trajectory:

* ``stage`` is where in the engine's life observation starts: 0.0 = its first cycle (the normal case),
  0.5 = half-way through its life, 0.75 = three quarters through (clearly degraded).
* ``n_obs`` is how many consecutive cycles are observed (3, 5, 10, 20, 30).
* The cycles are re-numbered 1..n, so the model sees exactly what a system that started monitoring
  at that moment would see. The prediction is made at the last observed cycle; the truth is the
  engine's real remaining life there.

The models are trained as usual on complete histories from cycle 1; the probe only changes what they are
asked about. Hyper-parameters other than window, history strategy and baseline are held fixed.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from ..data.schema import CYCLE_COLUMN, ID_COLUMN, SENSOR_COLUMNS


def make_probes(df: pd.DataFrame, stages: dict[str, float], n_obs: tuple[int, ...], cap: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Cut artificial histories out of complete engine trajectories.

    Returns ``(segments, truth)``. ``segments`` has one unique ``unit`` id per probe and cycles 1..n;
    ``truth`` has one row per probe with the true RUL at the last observed cycle.
    """
    segs, truths = [], []
    pid = 0
    for stage, frac in stages.items():
        for n in n_obs:
            for unit, g in df.groupby(ID_COLUMN, sort=True):
                life = len(g)
                if life < n + 1:
                    continue
                start = 1 if frac == 0 else int(round(frac * life))
                start = max(1, min(start, life - n))  # the last observed cycle must precede failure
                end = start + n - 1
                seg = g.iloc[start - 1 : end].copy()
                pid += 1
                seg[ID_COLUMN] = pid
                seg[CYCLE_COLUMN] = np.arange(1, n + 1)
                segs.append(seg)
                true_rul = life - end
                truths.append({"probe_id": pid, "engine": int(unit), "stage": stage, "n_obs": n, "start_cycle": start, "end_cycle": end,
                               "life": life, "true_rul": true_rul, "true_rul_clip": min(true_rul, cap)})
    if not segs:
        raise ValueError(f"No engine has more than {max(n_obs)} cycles, so no probe history can be built.")
    return pd.concat(segs, ignore_index=True), pd.DataFrame(truths)


def degradation_index(segments: pd.DataFrame, reference: pd.DataFrame, sensors: tuple[str, ...], n_first: int = 3) -> pd.Series:
    """How far the first observations of each history are from the healthy reference (data-based evidence).

    ``reference`` is training rows from the engines' first 20 cycles (already in the representation the model
    uses). The index is the mean absolute z-score of the first ``n_first`` readings, averaged over sensors.
    Around 0.8 is what a healthy reading looks like (the mean of |N(0,1)|); larger means visibly degraded.
    """
    ref = reference[list(sensors)]
    mu, sd = ref.mean(), ref.std().replace(0, np.nan)
    first = segments[segments[CYCLE_COLUMN] <= n_first]
    z = ((first[list(sensors)] - mu) / sd).abs().mean(axis=1)
    return z.groupby(first[ID_COLUMN]).mean().rename("degradation_index")
