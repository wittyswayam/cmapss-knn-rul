"""Phase 2 experiments on FD001: sensor configuration, baselining, short-history
strategy and KNN hyper-parameters, evaluated with grouped-by-engine CV.

Protocol
--------
* The 100 training engines are split once into 80 *tuning* engines and 20
  *holdout* engines. The holdout is touched only once, at the end. The
  official test files are not used at all in this phase.
* Hyper-parameters are chosen by 5-fold CV over the 80 tuning engines; folds
  split engines, never rows. Scaling, sensor selection and the KNN model are
  all fitted on the training part of each fold.
* Two kinds of evaluation, kept strictly separate:
    - *window-level*: every cycle of every validation engine is one sample;
      long-lived engines and early (healthy) cycles dominate.
    - *engine-level*: the official test protocol imitated on validation
      engines: each engine is truncated at several random cutoffs and the RUL
      at the cutoff is predicted from the history up to it. This is the
      selection criterion.
* Short-history stress: cutoffs at 5, 10 and 20 cycles, fewer than most windows.
* Speed trick: for each (features, metric, fold) the 100 nearest neighbours are
  searched once, and the predictions for every k and weighting are derived from
  that list with the same ``aggregate_neighbours`` rule the models use.
"""

from __future__ import annotations

import itertools
import logging
import time
from dataclasses import dataclass
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
from ..features.sensors import SensorSelector, feature_column_indices
from ..features.windows import FeatureTable, WindowConfig, build_features
from ..models.knn_common import aggregate_neighbours
from ..models.pipeline import PipelineConfig, RULKNNPipeline, make_knn

logger = logging.getLogger(__name__)


@dataclass(frozen=True, eq=False)
class Setup:
    """Data prepared once: engine split, evaluation points, folds."""

    tuning: pd.DataFrame  # 80 engines, with rul / rul_clipped
    holdout: pd.DataFrame  # 20 engines
    folds: list[tuple[np.ndarray, np.ndarray]]
    cutoffs: pd.DataFrame  # unit, cutoff for the tuning engines
    holdout_cutoffs: pd.DataFrame


def _eval_flags(df: pd.DataFrame, cutoffs: pd.DataFrame, stress: tuple[int, ...]) -> pd.DataFrame:
    """Row-aligned flags: which rows are engine-level cutoffs / stress points."""
    key = pd.MultiIndex.from_frame(df[[ID_COLUMN, CYCLE_COLUMN]])
    cut_key = pd.MultiIndex.from_frame(cutoffs.rename(columns={"cutoff": CYCLE_COLUMN}))
    return pd.DataFrame(
        {
            "is_cutoff": key.isin(cut_key),
            "is_stress": df[CYCLE_COLUMN].isin(stress).to_numpy(),
        },
        index=df.index,
    )


def make_setup(train_raw: pd.DataFrame, cfg: Config) -> Setup:
    v = cfg.validation
    train = add_train_rul(train_raw, cap=cfg.rul_cap)
    units = train[ID_COLUMN].unique()
    tuning_units, holdout_units = holdout_engines(units, v.n_holdout_engines, cfg.seed)
    tuning = train[train[ID_COLUMN].isin(tuning_units)].reset_index(drop=True)
    holdout = train[train[ID_COLUMN].isin(holdout_units)].reset_index(drop=True)
    folds = engine_folds(tuning_units, v.n_folds, cfg.seed)
    lifetimes = train.groupby(ID_COLUMN)[CYCLE_COLUMN].max()
    all_cut = sample_cutoffs(lifetimes, v.eval_cutoffs_per_engine, v.eval_min_cutoff, cfg.seed)
    return Setup(
        tuning=tuning,
        holdout=holdout,
        folds=folds,
        cutoffs=all_cut[all_cut[ID_COLUMN].isin(tuning_units)].reset_index(drop=True),
        holdout_cutoffs=all_cut[all_cut[ID_COLUMN].isin(holdout_units)].reset_index(drop=True),
    )


def _subset_metrics(prefix: str, truth_clip: np.ndarray, truth_raw: np.ndarray | None, pred: np.ndarray) -> dict[str, float]:
    m = regression_metrics(truth_clip, pred)
    out = {f"{prefix}_rmse_clip": m["rmse"], f"{prefix}_mae_clip": m["mae"], f"{prefix}_bias_clip": m["bias"], f"{prefix}_nasa_clip": m["nasa_mean"]}
    if truth_raw is not None:
        r = regression_metrics(truth_raw, pred)
        out.update({f"{prefix}_rmse_raw": r["rmse"], f"{prefix}_mae_raw": r["mae"], f"{prefix}_bias_raw": r["bias"], f"{prefix}_nasa_raw": r["nasa_mean"]})
    return out


def score_predictions(meta: pd.DataFrame, pred: np.ndarray) -> dict[str, float]:
    """All reported metrics for predictions on rows described by ``meta``.

    ``meta`` needs columns rul, rul_clipped, is_cutoff, is_stress. Targets of
    the model are clipped, so the clipped truth is the selection reference;
    the raw truth is reported alongside because that is what the official test uses.
    """
    clip, raw = meta[RUL_CLIPPED_COLUMN].to_numpy(), meta[RUL_COLUMN].to_numpy()
    cut, stress = meta["is_cutoff"].to_numpy(), meta["is_stress"].to_numpy()
    out = _subset_metrics("win", clip, None, pred)
    out.update(_subset_metrics("eng", clip[cut], raw[cut], pred[cut]))
    out.update(_subset_metrics("stress", clip[stress], None, pred[stress]))
    out["n_win"], out["n_eng"], out["n_stress"] = len(pred), int(cut.sum()), int(stress.sum())
    return out


def build_feature_cache(df: pd.DataFrame, cfg: Config) -> dict[tuple[str, int, str], FeatureTable]:
    g = cfg.grid
    cache = {}
    for baseline, window, history in itertools.product(g.baseline, g.window, g.history):
        wc = WindowConfig(window, cfg.features.stats, history, baseline, cfg.features.baseline_cycles)
        cache[(baseline, window, history)] = build_features(df, wc)
    return cache


def _fold_arrays(table: FeatureTable, df: pd.DataFrame, train_units: np.ndarray, val_units: np.ndarray, sensor_config: str, keep_only: tuple[str, ...] | None = None):
    """Scaled train/validation matrices for one fold, everything fitted on the training rows."""
    tr = df[ID_COLUMN].isin(train_units).to_numpy()
    va = df[ID_COLUMN].isin(val_units).to_numpy()
    selector = SensorSelector(sensor_config, keep_only=keep_only).fit(df[tr])
    cols = feature_column_indices(table.columns, selector.selected_sensors_)
    Xtr, Xva = table.X[tr][:, cols], table.X[va][:, cols]
    scaler = StandardScaler().fit(Xtr)
    return scaler.transform(Xtr), scaler.transform(Xva), tr, va, len(cols)


def _run_task(args: tuple[Any, ...]) -> list[dict[str, Any]]:
    table, df, meta, fold_id, train_units, val_units, key, sensor_config, metric, ks, weights_list = args
    Xtr, Xva, tr, va, n_feat = _fold_arrays(table, df, train_units, val_units, sensor_config)
    ytr = df[RUL_CLIPPED_COLUMN].to_numpy()[tr]
    kmax = max(ks)
    nn = NearestNeighbors(n_neighbors=kmax, metric=metric, algorithm="brute").fit(Xtr)
    dist, idx = nn.kneighbors(Xva)
    y_nb = ytr[idx]
    meta_va = meta[va]
    baseline, window, history = key
    rows = []
    for w in weights_list:
        for k in ks:
            pred = aggregate_neighbours(dist[:, :k], y_nb[:, :k], w)
            rows.append(
                {"sensor_config": sensor_config, "baseline": baseline, "window": window, "history": history, "metric": metric,
                 "weights": w, "k": k, "fold": fold_id, "n_features": n_feat, **score_predictions(meta_va, pred)}
            )
    return rows


def run_cv_sweep(setup: Setup, cfg: Config) -> pd.DataFrame:
    """Full grid x folds. Returns one row per (configuration, fold)."""
    df, g = setup.tuning, cfg.grid
    meta = pd.concat([df[[RUL_COLUMN, RUL_CLIPPED_COLUMN]], _eval_flags(df, setup.cutoffs, cfg.validation.stress_cutoffs)], axis=1)
    cache = build_feature_cache(df, cfg)
    tasks = [
        (cache[key], df, meta, f, tru, vau, key, sc, metric, g.k, g.weights)
        for key, sc, metric, (f, (tru, vau)) in itertools.product(cache, g.sensor_configs, g.metric, enumerate(setup.folds))
    ]
    logger.info("Running %d neighbour searches (%d configurations x %d folds)", len(tasks), len(tasks) // len(setup.folds) * len(g.weights) * len(g.k), len(setup.folds))
    rows: list[dict[str, Any]] = []
    t0 = time.time()
    for i, task in enumerate(tasks, 1):
        rows.extend(_run_task(task))
        if i % 20 == 0 or i == len(tasks):
            logger.info("  %d/%d searches done (%.0fs)", i, len(tasks), time.time() - t0)
    return pd.DataFrame(rows)


def constant_baseline(setup: Setup, cfg: Config) -> pd.DataFrame:
    """Reference: always predict the training-fold mean of the clipped RUL."""
    df = setup.tuning
    meta = pd.concat([df[[RUL_COLUMN, RUL_CLIPPED_COLUMN]], _eval_flags(df, setup.cutoffs, cfg.validation.stress_cutoffs)], axis=1)
    rows = []
    for f, (tru, vau) in enumerate(setup.folds):
        tr, va = df[ID_COLUMN].isin(tru).to_numpy(), df[ID_COLUMN].isin(vau).to_numpy()
        pred = np.full(int(va.sum()), df.loc[tr, RUL_CLIPPED_COLUMN].mean())
        rows.append({"fold": f, **score_predictions(meta[va], pred)})
    return pd.DataFrame(rows)


CONFIG_KEYS = ["sensor_config", "baseline", "window", "history", "metric", "weights", "k"]


def summarise(cv: pd.DataFrame) -> pd.DataFrame:
    """Mean and std over folds for every configuration."""
    metric_cols = [c for c in cv.columns if c not in CONFIG_KEYS + ["fold", "n_features", "n_win", "n_eng", "n_stress"]]
    agg = cv.groupby(CONFIG_KEYS)[metric_cols].agg(["mean", "std"])
    agg.columns = [f"{a}_{b}" for a, b in agg.columns]
    return agg.reset_index().assign(n_features=cv.groupby(CONFIG_KEYS)["n_features"].first().to_numpy())


def holdout_predictions(setup: Setup, cfg: Config, pcfg: PipelineConfig) -> tuple[pd.DataFrame, np.ndarray]:
    """Fit the full pipeline on all tuning engines; predict every holdout row."""
    model = RULKNNPipeline(pcfg).fit(setup.tuning)
    return setup.holdout, model.predict_rows(setup.holdout)


def holdout_meta(setup: Setup, cfg: Config) -> pd.DataFrame:
    h = setup.holdout
    return pd.concat([h[[RUL_COLUMN, RUL_CLIPPED_COLUMN]], _eval_flags(h, setup.holdout_cutoffs, cfg.validation.stress_cutoffs)], axis=1)


def pipeline_config_from_row(row: pd.Series, cfg: Config, backend: str) -> PipelineConfig:
    return PipelineConfig(
        window=WindowConfig(int(row["window"]), cfg.features.stats, str(row["history"]), str(row["baseline"]), cfg.features.baseline_cycles),
        sensor_config=str(row["sensor_config"]),
        n_neighbors=int(row["k"]),
        metric=str(row["metric"]),
        weights=str(row["weights"]),
        backend=backend,
    )


def compare_backends_on_folds(setup: Setup, cfg: Config, row: pd.Series, folds: list[int]) -> pd.DataFrame:
    """Run both KNN implementations on identical scaled fold data and compare predictions."""
    df = setup.tuning
    wc = WindowConfig(int(row["window"]), cfg.features.stats, str(row["history"]), str(row["baseline"]), cfg.features.baseline_cycles)
    table = build_features(df, wc)
    out = []
    for f in folds:
        tru, vau = setup.folds[f]
        Xtr, Xva, tr, va, _ = _fold_arrays(table, df, tru, vau, str(row["sensor_config"]))
        ytr = df[RUL_CLIPPED_COLUMN].to_numpy()[tr]
        k, metric, weights = int(row["k"]), str(row["metric"]), str(row["weights"])
        t0 = time.time(); a = make_knn("scratch", k, metric, weights).fit(Xtr, ytr).predict(Xva); t_scratch = time.time() - t0
        t0 = time.time(); b = make_knn("sklearn", k, metric, weights).fit(Xtr, ytr).predict(Xva); t_sklearn = time.time() - t0
        diff = np.abs(a - b)
        out.append({"fold": f, "k": k, "metric": metric, "weights": weights, "n_predictions": len(a), "max_abs_diff": float(diff.max()),
                    "n_diff_gt_1e-9": int((diff > 1e-9).sum()), "seconds_scratch": t_scratch, "seconds_sklearn": t_sklearn})
    return pd.DataFrame(out)


def eval_meta(df: pd.DataFrame, cutoffs: pd.DataFrame, cfg: Config) -> pd.DataFrame:
    return pd.concat([df[[RUL_COLUMN, RUL_CLIPPED_COLUMN]], _eval_flags(df, cutoffs, cfg.validation.stress_cutoffs)], axis=1)


def cv_single_config(setup: Setup, cfg: Config, row: pd.Series, keep_only: tuple[str, ...] | None = None,
                     permute_sensor: str | None = None, perm_seed: int = 0) -> pd.DataFrame:
    """Grouped CV of one configuration; one row per fold.

    ``keep_only`` restricts the model to some sensors. ``permute_sensor`` is a control experiment:
    that sensor's feature columns are shuffled across rows (same shuffle for all its columns), which
    keeps their distribution but destroys any relation to the engine's state. If the real sensor
    beats its shuffled copy, its content, not just its presence in the distance, helps.
    """
    df = setup.tuning
    meta = eval_meta(df, setup.cutoffs, cfg)
    wc = WindowConfig(int(row["window"]), cfg.features.stats, str(row["history"]), str(row["baseline"]), cfg.features.baseline_cycles)
    table = build_features(df, wc)
    if permute_sensor is not None:
        cols = feature_column_indices(table.columns, (permute_sensor,))
        X = table.X.copy()
        X[:, cols] = X[np.random.default_rng(perm_seed).permutation(len(X))][:, cols]
        table = FeatureTable(X=X, columns=table.columns, index=table.index)
    rows = []
    for f, (tru, vau) in enumerate(setup.folds):
        Xtr, Xva, tr, va, n_feat = _fold_arrays(table, df, tru, vau, str(row["sensor_config"]), keep_only)
        ytr = df[RUL_CLIPPED_COLUMN].to_numpy()[tr]
        pred = make_knn("sklearn", int(row["k"]), str(row["metric"]), str(row["weights"])).fit(Xtr, ytr).predict(Xva)
        rows.append({"fold": f, "n_features": n_feat, **score_predictions(meta[va], pred)})
    return pd.DataFrame(rows)


def duplicate_feature_rows(setup: Setup, cfg: Config, row: pd.Series) -> dict[str, float]:
    """How many training rows of fold 0 have a feature vector identical to another row (exact duplicates)."""
    df = setup.tuning
    wc = WindowConfig(int(row["window"]), cfg.features.stats, str(row["history"]), str(row["baseline"]), cfg.features.baseline_cycles)
    table = build_features(df, wc)
    tru, vau = setup.folds[0]
    Xtr, _, tr, _, _ = _fold_arrays(table, df, tru, vau, str(row["sensor_config"]))
    uniq, inv, counts = np.unique(np.round(Xtr, 12), axis=0, return_inverse=True, return_counts=True)
    dup_rows = int((counts[inv.ravel()] > 1).sum())
    cycles = df.loc[tr, CYCLE_COLUMN].to_numpy()
    in_dup = counts[inv.ravel()] > 1
    return {"train_rows": int(tr.sum()), "rows_with_exact_duplicate": dup_rows, "largest_duplicate_group": int(counts.max()),
            "duplicate_rows_at_cycle_1": int((in_dup & (cycles == 1)).sum()), "duplicate_rows_after_cycle_1": int((in_dup & (cycles > 1)).sum())}
