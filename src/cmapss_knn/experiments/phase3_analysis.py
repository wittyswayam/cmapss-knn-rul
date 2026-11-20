"""Analyses over the Phase 3 CV results (selection criteria, factor effects, regime evidence)."""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from ..data.schema import SENSOR_COLUMNS
from ..eda.statistics import within_engine_trend
from ..features.regimes import RegimeNormalizer
from .phase3 import CONFIG_KEYS

RMSE_COL, NASA_COL = "eng_rmse_clip_mean", "eng_nasa_clip_mean"
REPORT_METRICS = ["eng_rmse_clip", "eng_mae_clip", "eng_nasa_clip", "eng_bias_clip", "eng_late_clip", "eng_rmse_raw", "eng_nasa_raw", "win_rmse_clip", "stress_rmse_clip"]


def _key(row: pd.Series) -> tuple:
    return tuple(str(row[k]) for k in CONFIG_KEYS)


def fold_values(cv: pd.DataFrame, config: pd.Series, metrics: list[str]) -> pd.DataFrame:
    """Per-fold metrics of one configuration."""
    mask = np.ones(len(cv), dtype=bool)
    for k in CONFIG_KEYS:
        mask &= (cv[k].astype(str) == str(config[k])).to_numpy()
    return cv[mask].set_index("fold")[metrics].sort_index()


def compare_selection(summary: pd.DataFrame, cv: pd.DataFrame) -> dict:
    """Select by engine-level RMSE and by engine-level NASA score (both vs clipped RUL) and compare."""
    by_rmse = summary.loc[summary[RMSE_COL].idxmin()]
    by_nasa = summary.loc[summary[NASA_COL].idxmin()]
    top = lambda col: {_key(r) for _, r in summary.nsmallest(10, col).iterrows()}
    rho = spearmanr(summary[RMSE_COL], summary[NASA_COL]).statistic
    fr, fn = fold_values(cv, by_rmse, ["eng_rmse_clip", "eng_nasa_clip"]), fold_values(cv, by_nasa, ["eng_rmse_clip", "eng_nasa_clip"])
    return {
        "same_configuration": _key(by_rmse) == _key(by_nasa),
        "spearman_rmse_vs_nasa_over_configs": float(rho),
        "top10_overlap": len(top(RMSE_COL) & top(NASA_COL)),
        "rmse_selected": by_rmse, "nasa_selected": by_nasa,
        "rmse_of_nasa_selected_minus_rmse_selected": float(by_nasa[RMSE_COL] - by_rmse[RMSE_COL]),
        "nasa_of_rmse_selected_minus_nasa_selected": float(by_rmse[NASA_COL] - by_nasa[NASA_COL]),
        "folds_where_rmse_selected_has_lower_rmse": int((fr["eng_rmse_clip"] < fn["eng_rmse_clip"]).sum()),
        "folds_where_nasa_selected_has_lower_nasa": int((fn["eng_nasa_clip"] < fr["eng_nasa_clip"]).sum()),
    }


def paired_effect3(summary: pd.DataFrame, factor: str, base: str, other: str, metrics: list[str]) -> pd.DataFrame:
    """Switch ``factor`` from ``base`` to ``other`` with all other settings identical (diff = other - base)."""
    keys = [k for k in CONFIG_KEYS if k != factor]
    a = summary[summary[factor].astype(str) == base].set_index(keys)[metrics]
    b = summary[summary[factor].astype(str) == other].set_index(keys)[metrics]
    j = a.join(b, lsuffix="_base", rsuffix="_other", how="inner")
    for m in metrics:
        j[f"{m}_diff"] = j[f"{m}_other"] - j[f"{m}_base"]
    return j.reset_index()


def fold_pairs3(cv: pd.DataFrame, config: pd.Series, factor: str, other: str, metric: str) -> pd.DataFrame:
    alt = config.copy()
    alt[factor] = other
    a, b = fold_values(cv, config, [metric])[metric], fold_values(cv, alt, [metric])[metric]
    return pd.DataFrame({"base": a, "other": b, "diff": b - a})


def regime_variance_share(train: pd.DataFrame, norm: RegimeNormalizer) -> pd.Series:
    """Share of each raw sensor's variance that is explained by the operating regime (between-regime / total)."""
    reg = norm.detector_.predict(train)
    out = {}
    for s in SENSOR_COLUMNS:
        x = train[s].to_numpy()
        if x.var() == 0:
            continue
        means = pd.Series(x).groupby(reg).transform("mean").to_numpy()
        out[s] = float(((means - x.mean()) ** 2).mean() / x.var())
    return pd.Series(out, name="regime_variance_share")


def trend_strength(train: pd.DataFrame, sensors: list[str]) -> pd.Series:
    """Mean |within-engine Spearman correlation with cycle| per sensor (how clearly it drifts with age)."""
    return within_engine_trend(train, sensors)["mean_abs_rho"]


def load_probe(probe_dir, subset: str) -> pd.DataFrame:
    import glob

    files = sorted(glob.glob(str(probe_dir / f"{subset.lower()}_fold*.csv")))
    if not files:
        raise FileNotFoundError(f"No probe results for {subset} in {probe_dir}")
    return pd.concat([pd.read_csv(f) for f in files], ignore_index=True)


def probe_metrics(d: pd.DataFrame, by: list[str]) -> pd.DataFrame:
    """RMSE, MAE, bias, mean NASA score and late share of probe predictions, grouped by ``by``."""
    from ..evaluation.metrics import regression_metrics

    rows = []
    for key, g in d.groupby(by, sort=False):
        key = key if isinstance(key, tuple) else (key,)
        m = regression_metrics(g["true_rul_clip"], g["pred"])
        rows.append({**dict(zip(by, key)), "n": len(g), "rmse": m["rmse"], "mae": m["mae"], "bias": m["bias"], "nasa": m["nasa_mean"], "late": m["late_share"]})
    return pd.DataFrame(rows)
