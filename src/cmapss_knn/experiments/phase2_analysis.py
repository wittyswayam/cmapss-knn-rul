"""Comparisons derived from the CV sweep: best configurations and paired factor effects."""

from __future__ import annotations

import numpy as np
import pandas as pd

from .phase2 import CONFIG_KEYS

SELECTION = "eng_rmse_clip_mean"  # engine-level RMSE against the clipped RUL, averaged over folds


def best_rows(summary: pd.DataFrame, by: list[str] | None = None) -> pd.DataFrame:
    """Best configuration (lowest engine-level RMSE) overall or per group."""
    if not by:
        return summary.loc[[summary[SELECTION].idxmin()]]
    return summary.loc[summary.groupby(by)[SELECTION].idxmin()].reset_index(drop=True)


def paired_effect(summary: pd.DataFrame, factor: str, base: str, other: str, metrics: list[str]) -> pd.DataFrame:
    """Effect of switching ``factor`` from ``base`` to ``other`` with everything else identical.

    Returns one row per matched configuration with ``<metric>_base``, ``<metric>_other`` and
    ``<metric>_diff`` = other - base (negative means ``other`` has lower error).
    """
    keys = [k for k in CONFIG_KEYS if k != factor]
    a = summary[summary[factor].astype(str) == str(base)].set_index(keys)[metrics]
    b = summary[summary[factor].astype(str) == str(other)].set_index(keys)[metrics]
    joined = a.join(b, lsuffix="_base", rsuffix="_other", how="inner")
    for m in metrics:
        joined[f"{m}_diff"] = joined[f"{m}_other"] - joined[f"{m}_base"]
    return joined.reset_index()


def describe_effect(effect: pd.DataFrame, metric: str) -> dict[str, float]:
    d = effect[f"{metric}_diff"]
    return {"matched_configs": int(len(d)), "mean_diff": float(d.mean()), "median_diff": float(d.median()),
            "share_other_better": float((d < 0).mean()), "min_diff": float(d.min()), "max_diff": float(d.max())}


def fold_pairs(cv: pd.DataFrame, config: pd.Series, factor: str, other: str, metric: str) -> pd.DataFrame:
    """Per-fold metric for ``config`` and for the same config with ``factor`` set to ``other``."""
    keys = [k for k in CONFIG_KEYS if k != factor]
    mask = np.ones(len(cv), dtype=bool)
    for k in keys:
        mask &= (cv[k].astype(str) == str(config[k])).to_numpy()
    sel = cv[mask]
    base = sel[sel[factor].astype(str) == str(config[factor])].set_index("fold")[metric]
    oth = sel[sel[factor].astype(str) == str(other)].set_index("fold")[metric]
    out = pd.DataFrame({"base": base, "other": oth})
    out["diff"] = out["other"] - out["base"]
    return out


def s6_dip_rate_by_rul(train_with_rul: pd.DataFrame, bins: list[int]) -> pd.DataFrame:
    """Share of s6 readings equal to the lower of its two values (21.60), by remaining life.

    Computed on the rows it is given (the tuning engines). Used to explain why s6, a
    two-valued channel, carries an age signal.
    """
    low = train_with_rul["s6"] < train_with_rul["s6"].mean()  # the two values are 21.60 and 21.61
    # NOTE: with two values the mean always separates them, so this is exact, not a threshold choice.
    bin_col = pd.cut(train_with_rul["rul"], bins)
    out = pd.DataFrame({"rows": low.groupby(bin_col, observed=True).size(), "low_value_rate": low.groupby(bin_col, observed=True).mean()})
    out.index = out.index.astype(str)
    out.index.name = "rul_bin"
    return out
