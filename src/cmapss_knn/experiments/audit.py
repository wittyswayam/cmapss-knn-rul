"""Independent checks of Phase 3 results.

* ``recompute_cv``: re-derives saved CV metrics through a different code path (the end-to-end
  ``RULKNNPipeline`` fitted per fold on that fold's training engines only, plus the naive reference metrics)
  and compares them with what the sweep saved.
* ``probe_checks``: structural and cross-path checks of the short-history probe.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from ..config import Config
from ..evaluation.reference import mae_reference, mean_nasa_reference, rmse_reference
from ..features.windows import WindowConfig
from ..models.pipeline import PipelineConfig, RULKNNPipeline
from . import phase3 as p3
from . import phase3_analysis as an
from .phase3_commands import pipeline_config_from_selection


def recompute_cv(setup: p3.SubsetSetup, cfg: Config, cv: pd.DataFrame, config: dict) -> pd.DataFrame:
    """Per fold: saved vs independently recomputed engine-level RMSE, MAE and mean NASA (vs clipped RUL)."""
    df = setup.tuning
    meta = p3.query_meta(df, setup.cutoffs, cfg)
    saved = an.fold_values(cv, pd.Series(config), ["eng_rmse_clip", "eng_mae_clip", "eng_nasa_clip", "n_eng"])
    rows = []
    for f, (train_units, val_units) in enumerate(setup.folds):
        in_val = df["unit"].isin(val_units).to_numpy()
        train, val = df[~in_val & df["unit"].isin(train_units).to_numpy()].reset_index(drop=True), df[in_val].reset_index(drop=True)
        model = RULKNNPipeline(pipeline_config_from_selection(config, cfg)).fit(train)
        pred = model.predict_rows(val)
        mv = meta[in_val].reset_index(drop=True)
        cut = mv["is_cutoff"].to_numpy()
        truth, p = mv["rul_clipped"].to_numpy()[cut].tolist(), pred[cut].tolist()
        rec = {"fold": f, "n_eng": len(truth), "rmse": rmse_reference(truth, p), "mae": mae_reference(truth, p), "nasa": mean_nasa_reference(truth, p)}
        rows.append({**rec, "saved_rmse": saved.loc[f, "eng_rmse_clip"], "saved_mae": saved.loc[f, "eng_mae_clip"], "saved_nasa": saved.loc[f, "eng_nasa_clip"], "saved_n_eng": int(saved.loc[f, "n_eng"])})
    out = pd.DataFrame(rows)
    out["abs_diff_rmse"], out["abs_diff_mae"], out["abs_diff_nasa"] = [(out[m] - out[f"saved_{m}"]).abs() for m in ("rmse", "mae", "nasa")]
    return out


def probe_structure(setup: p3.SubsetSetup, probe: pd.DataFrame) -> dict:
    """Probe histories must come from validation engines only, never from the engines the model was fitted on."""
    bad, n_checked = 0, 0
    for f, (train_units, val_units) in enumerate(setup.folds):
        eng = set(probe.loc[probe["fold"] == f, "engine"].unique())
        n_checked += len(eng)
        bad += len(eng & set(train_units)) + len(eng - set(val_units))
    return {"engines_checked": n_checked, "engines_in_training_or_wrong_fold": bad}


def probe_vs_sweep_stress(probe: pd.DataFrame, cv: pd.DataFrame, norm_mode: str, pc) -> pd.DataFrame:
    """Healthy-start probes of 5, 10 and 20 cycles are the same prediction problem as the sweep's stress points
    (cycles 5, 10, 20 of validation engines). Compare per-fold RMSE of both code paths."""
    from ..evaluation.metrics import rmse

    rows = []
    healthy = probe[(probe["stage"] == "healthy_start") & probe["n_obs"].isin([5, 10, 20]) & probe["history"].isin(["adaptive", "pad_first"])]
    for (f, w, h, b), g in healthy.groupby(["fold", "window", "history", "baseline"]):
        m = cv[(cv["fold"] == f) & (cv["normalization"] == norm_mode) & (cv["baseline"] == b) & (cv["window"] == w) & (cv["history"] == h)
               & (cv["metric"] == pc.metric) & (cv["weights"] == pc.weights) & (cv["k"] == pc.k)]
        rows.append({"fold": f, "window": w, "history": h, "baseline": b, "probe_rmse": rmse(g["true_rul_clip"], g["pred"]), "sweep_stress_rmse": float(m["stress_rmse_clip"].iloc[0])})
    out = pd.DataFrame(rows)
    out["abs_diff"] = (out["probe_rmse"] - out["sweep_stress_rmse"]).abs()
    return out


def full_history_reference(setup: p3.SubsetSetup, cfg: Config, probe: pd.DataFrame, windows=(30, 75), baselines=("none", "causal"), history="pad_first") -> pd.DataFrame:
    """For each probe, what the *same* model predicts at the same engine and cycle when it is given the engine's full
    history from cycle 1. The gap to the probe prediction is the cost of the missing history; for healthy starts it must be zero."""
    pc = cfg.phase3.probe
    df = setup.tuning
    rows = []
    for f, (train_units, val_units) in enumerate(setup.folds):
        train, val = df[df["unit"].isin(train_units)].reset_index(drop=True), df[df["unit"].isin(val_units)].reset_index(drop=True)
        for w in windows:
            for b in baselines:
                wc = WindowConfig(w, cfg.features.stats, history, b, cfg.features.baseline_cycles)
                model = RULKNNPipeline(PipelineConfig(wc, "A", pc.k, pc.metric, pc.weights, "sklearn", "global", cfg.seed)).fit(train)
                full = pd.Series(model.predict_rows(val), index=pd.MultiIndex.from_arrays([val["unit"], val["cycle"]]))
                pr = probe[(probe["fold"] == f) & (probe["window"] == w) & (probe["baseline"] == b) & (probe["history"] == history)]
                key = pd.MultiIndex.from_arrays([pr["engine"], pr["end_cycle"]])
                rows.append(pr.assign(full_history_pred=full.reindex(key).to_numpy()))
    return pd.concat(rows, ignore_index=True)
