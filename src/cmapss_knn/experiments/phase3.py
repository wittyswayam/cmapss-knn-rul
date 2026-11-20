"""Phase 3: the Phase 2 methodology applied to all four subsets, with regime normalisation.

Differences from ``phase2.py`` (which is left untouched):

* The sensor configuration is fixed to A (drop constant sensors); the s6 question was settled on FD001.
* The grid has a ``normalization`` axis: ``global`` (raw sensors, scaled only at the end) versus
  ``regime`` (per-regime z-scores, fitted on the training part of each fold). It is only run on
  subsets where the regime detector finds more than one operating regime.
* Queries are restricted to the rows that are actually scored (engine-level cutoffs, short-history
  stress points and every ``window_eval_stride``-th cycle), which keeps the large subsets tractable.
  Engine-level metrics, used for selection, are unaffected; window-level metrics use that stride.
* Metrics reported everywhere: RMSE, MAE, NASA score (mean per prediction), plus bias and the
  share of late (optimistic) predictions.
"""

from __future__ import annotations

import itertools
import json
import logging
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler

from ..config import Config
from ..data.schema import CYCLE_COLUMN, ID_COLUMN, RUL_CLIPPED_COLUMN, RUL_COLUMN
from ..data.targets import add_train_rul
from ..evaluation.metrics import regression_metrics
from ..evaluation.splits import engine_folds, holdout_engines, sample_cutoffs
from ..features.regimes import RegimeNormalizer
from ..features.sensors import SensorSelector, feature_column_indices
from ..features.windows import WindowConfig, build_features
from ..models.knn_common import aggregate_neighbours
from .phase2 import _eval_flags

logger = logging.getLogger(__name__)

CONFIG_KEYS = ["normalization", "baseline", "window", "history", "metric", "weights", "k"]
_NAMES = {"rmse": "rmse", "mae": "mae", "bias": "bias", "late_share": "late", "nasa_mean": "nasa"}


@dataclass(frozen=True, eq=False)
class SubsetSetup:
    name: str
    tuning: pd.DataFrame
    holdout: pd.DataFrame
    folds: list[tuple[np.ndarray, np.ndarray]]
    cutoffs: pd.DataFrame
    holdout_cutoffs: pd.DataFrame


def make_subset_setup(name: str, train_raw: pd.DataFrame, cfg: Config) -> SubsetSetup:
    """Engine-level split: tuning engines (for CV) and a development holdout (20% of engines)."""
    v = cfg.validation
    train = add_train_rul(train_raw, cap=cfg.rul_cap)
    units = train[ID_COLUMN].unique()
    n_holdout = int(round(cfg.phase3.holdout_fraction * len(units)))
    tuning_units, holdout_units = holdout_engines(units, n_holdout, cfg.seed)
    lifetimes = train.groupby(ID_COLUMN)[CYCLE_COLUMN].max()
    cuts = sample_cutoffs(lifetimes, v.eval_cutoffs_per_engine, v.eval_min_cutoff, cfg.seed)
    return SubsetSetup(
        name=name,
        tuning=train[train[ID_COLUMN].isin(tuning_units)].reset_index(drop=True),
        holdout=train[train[ID_COLUMN].isin(holdout_units)].reset_index(drop=True),
        folds=engine_folds(tuning_units, v.n_folds, cfg.seed),
        cutoffs=cuts[cuts[ID_COLUMN].isin(tuning_units)].reset_index(drop=True),
        holdout_cutoffs=cuts[cuts[ID_COLUMN].isin(holdout_units)].reset_index(drop=True),
    )


def query_meta(df: pd.DataFrame, cutoffs: pd.DataFrame, cfg: Config) -> pd.DataFrame:
    """Row-aligned truth and evaluation flags. ``is_query`` marks every row that gets scored."""
    meta = pd.concat([df[[RUL_COLUMN, RUL_CLIPPED_COLUMN]], _eval_flags(df, cutoffs, cfg.validation.stress_cutoffs)], axis=1)
    meta["is_win"] = (df[CYCLE_COLUMN] % cfg.phase3.window_eval_stride == 0).to_numpy()
    meta["is_query"] = meta["is_cutoff"] | meta["is_stress"] | meta["is_win"]
    return meta


def _group(prefix: str, truth: np.ndarray, pred: np.ndarray, suffix: str) -> dict[str, float]:
    m = regression_metrics(truth, pred)
    return {f"{prefix}_{_NAMES[k]}_{suffix}": v for k, v in m.items()}


def score3(meta: pd.DataFrame, pred: np.ndarray) -> dict[str, float]:
    """RMSE, MAE, NASA, bias and late share on the three evaluation subsets.

    ``eng``: engine-level cutoffs (clipped and raw truth); ``win``: strided window-level points;
    ``stress``: very short histories (clipped truth). Rows are expected to be query rows only.
    """
    clip, raw = meta[RUL_CLIPPED_COLUMN].to_numpy(), meta[RUL_COLUMN].to_numpy()
    cut, win, stress = meta["is_cutoff"].to_numpy(), meta["is_win"].to_numpy(), meta["is_stress"].to_numpy()
    out = _group("eng", clip[cut], pred[cut], "clip")
    out.update(_group("eng", raw[cut], pred[cut], "raw"))
    out.update(_group("win", clip[win], pred[win], "clip"))
    out.update(_group("stress", clip[stress], pred[stress], "clip"))
    out.update({"n_eng": int(cut.sum()), "n_win": int(win.sum()), "n_stress": int(stress.sum())})
    return out


def regimes_for(setup: SubsetSetup, seed: int) -> RegimeNormalizer:
    """Regime detector/normaliser fitted on *all* tuning rows (descriptive; CV refits per fold)."""
    return RegimeNormalizer(seed=seed).fit(setup.tuning)


def run_fold(setup: SubsetSetup, cfg: Config, fold: int, normalizations: tuple[str, ...], grid=None, include_reference: bool = True) -> pd.DataFrame:
    """All grid configurations for one CV fold. Everything is fitted on the fold's training engines.

    ``grid`` overrides the configured grid (used by the extension runner); ``include_reference`` adds the constant-predictor row."""
    g = grid if grid is not None else cfg.phase3.grid
    df = setup.tuning
    meta = query_meta(df, setup.cutoffs, cfg)
    train_units, val_units = setup.folds[fold]
    tr = df[ID_COLUMN].isin(train_units).to_numpy()
    va = df[ID_COLUMN].isin(val_units).to_numpy()
    q = (meta["is_query"].to_numpy()) & va
    meta_q = meta[q]
    ytr = df[RUL_CLIPPED_COLUMN].to_numpy()[tr]
    kmax = max(g.k)
    rows: list[dict[str, Any]] = []
    for norm in normalizations:
        dfn = RegimeNormalizer(seed=cfg.seed).fit(df[tr]).transform(df) if norm == "regime" else df
        selector = SensorSelector("A").fit(dfn[tr])
        for baseline, window, history in itertools.product(g.baseline, g.window, g.history):
            wc = WindowConfig(window, cfg.features.stats, history, baseline, cfg.features.baseline_cycles)
            table = build_features(dfn, wc)
            cols = feature_column_indices(table.columns, selector.selected_sensors_)
            scaler = StandardScaler().fit(table.X[tr][:, cols])
            Xtr, Xq = scaler.transform(table.X[tr][:, cols]), scaler.transform(table.X[q][:, cols])
            for metric in g.metric:
                dist, idx = NearestNeighbors(n_neighbors=kmax, metric=metric, algorithm="brute").fit(Xtr).kneighbors(Xq)
                y_nb = ytr[idx]
                for weights in g.weights:
                    for k in g.k:
                        pred = aggregate_neighbours(dist[:, :k], y_nb[:, :k], weights)
                        rows.append({"subset": setup.name, "fold": fold, "normalization": norm, "baseline": baseline, "window": window,
                                     "history": history, "metric": metric, "weights": weights, "k": k, "n_features": len(cols), **score3(meta_q, pred)})
    if not include_reference:
        return pd.DataFrame(rows)
    # reference: constant prediction at the training mean
    const = score3(meta_q, np.full(len(meta_q), ytr.mean()))
    rows.append({"subset": setup.name, "fold": fold, "normalization": "constant", "baseline": "-", "window": 0, "history": "-", "metric": "-", "weights": "-", "k": 0, "n_features": 0, **const})
    return pd.DataFrame(rows)


def run_sweep(setup: SubsetSetup, cfg: Config, out_dir: Path, normalizations: tuple[str, ...], only_folds: list[int] | None = None) -> None:
    """Run (or resume) the sweep; each fold is saved to its own CSV and skipped if it already exists."""
    out_dir.mkdir(parents=True, exist_ok=True)
    for fold in range(len(setup.folds)):
        path = out_dir / f"{setup.name.lower()}_fold{fold}.csv"
        if (only_folds is not None and fold not in only_folds) or path.is_file():
            continue
        t0 = time.time()
        run_fold(setup, cfg, fold, normalizations).to_csv(path, index=False)
        logger.info("%s fold %d done in %.0fs -> %s", setup.name, fold, time.time() - t0, path.name)


def load_cv(out_dir: Path, name: str, n_folds: int) -> pd.DataFrame:
    parts = [out_dir / f"{name.lower()}_fold{f}.csv" for f in range(n_folds)]
    missing = [p.name for p in parts if not p.is_file()]
    if missing:
        raise FileNotFoundError(f"Sweep incomplete for {name}; missing {missing}")
    return pd.concat([pd.read_csv(p) for p in parts], ignore_index=True)


def summarise3(cv: pd.DataFrame) -> pd.DataFrame:
    """Mean and std over folds per configuration (the constant reference is returned separately)."""
    cfgs = cv[cv["normalization"] != "constant"]
    metric_cols = [c for c in cfgs.columns if c not in CONFIG_KEYS + ["subset", "fold", "n_features", "n_eng", "n_win", "n_stress"]]
    agg = cfgs.groupby(CONFIG_KEYS)[metric_cols].agg(["mean", "std"])
    agg.columns = [f"{a}_{b}" for a, b in agg.columns]
    out = agg.reset_index()
    out["n_features"] = cfgs.groupby(CONFIG_KEYS)["n_features"].first().to_numpy()
    return out


def constant_reference(cv: pd.DataFrame) -> pd.Series:
    c = cv[cv["normalization"] == "constant"]
    return c.drop(columns=["subset", "fold", "normalization", "baseline", "history", "metric", "weights"]).mean()


# ------------------------------------------------------------------ dev holdout
def holdout_audit_path(results_dir: Path) -> Path:
    return results_dir / "phase3" / "holdout_audit.json"


def record_holdout_evaluation(results_dir: Path, subset: str, note: str) -> int:
    """Count how many times a subset's development holdout has been evaluated (auditable)."""
    path = holdout_audit_path(results_dir)
    audit = json.loads(path.read_text(encoding="utf-8")) if path.is_file() else {}
    entry = audit.setdefault(subset, {"evaluations": 0, "log": []})
    entry["evaluations"] += 1
    entry["log"].append({"time_utc": pd.Timestamp.now("UTC").isoformat(timespec="seconds"), "note": note})
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(audit, indent=2), encoding="utf-8")
    return entry["evaluations"]
