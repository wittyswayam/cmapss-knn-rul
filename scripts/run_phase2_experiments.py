"""Phase 2 experiments on FD001 (grouped-by-engine CV, holdout engines, backend comparison).

Usage (from the repository root):
    python scripts/run_phase2_experiments.py                 # everything
    python scripts/run_phase2_experiments.py --sweep-only    # just the CV sweep (saved to disk)
    python scripts/run_phase2_experiments.py --reuse-sweep   # skip the sweep, reuse the saved CSV
Writes tables to results/phase2/ and figures to figures/fd001/. Takes a few minutes.
The official test files are not used.
"""

from __future__ import annotations

import _paths  # noqa: F401

import argparse
import json
import logging
import sys

import numpy as np
import pandas as pd

from cmapss_knn.config import load_config
from cmapss_knn.data.ingestion import read_subset
from cmapss_knn.data.validation import validate_subset
from cmapss_knn.evaluation.metrics import regression_metrics
from cmapss_knn.exceptions import CMAPSSError
from cmapss_knn.features import WindowConfig, build_features
from cmapss_knn.experiments import phase2 as p2
from cmapss_knn.experiments import phase2_analysis as an
from cmapss_knn.experiments import phase2_plots as pp
from cmapss_knn.logging_utils import setup_logging
from cmapss_knn.models.pipeline import RULKNNPipeline

logger = logging.getLogger("phase2")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--sweep-only", action="store_true")
    parser.add_argument("--reuse-sweep", action="store_true")
    args = parser.parse_args()
    cfg = load_config()
    setup_logging(cfg.log_level, cfg.paths.logs_dir / "phase2.log")
    out = cfg.paths.results_dir / "phase2"
    figs = cfg.paths.figures_dir / "fd001"
    out.mkdir(parents=True, exist_ok=True)
    try:
        subset = read_subset("FD001", cfg.paths.raw_data_dir, cfg.archive_name)
        validate_subset(subset, cfg.subsets["FD001"])
    except CMAPSSError as exc:
        logger.error("%s", exc)
        return 1

    setup = p2.make_setup(subset.train, cfg)
    info = {
        "tuning_engines": int(setup.tuning["unit"].nunique()), "holdout_engines": int(setup.holdout["unit"].nunique()),
        "tuning_rows": len(setup.tuning), "holdout_rows": len(setup.holdout),
        "engine_level_points_tuning": len(setup.cutoffs), "engine_level_points_holdout": len(setup.holdout_cutoffs),
        "holdout_units": sorted(int(u) for u in setup.holdout["unit"].unique()),
        "fold_validation_engines": [len(v) for _, v in setup.folds],
    }

    # ---------------- sweep ----------------
    cv_path = out / "cv_results_fd001.csv"
    if args.reuse_sweep and cv_path.is_file():
        logger.info("Reusing saved sweep %s", cv_path)
        cv = pd.read_csv(cv_path)
    else:
        cv = p2.run_cv_sweep(setup, cfg)
        cv.to_csv(cv_path, index=False)
    if args.sweep_only:
        return 0
    summary = p2.summarise(cv)
    summary.to_csv(out / "cv_summary_fd001.csv", index=False)
    const = p2.constant_baseline(setup, cfg)
    const.to_csv(out / "cv_constant_baseline_fd001.csv", index=False)
    info["n_configurations"] = len(summary)

    # ---------------- selection ----------------
    best = an.best_rows(summary).iloc[0]
    best_by_sensor = an.best_rows(summary, ["sensor_config"]).set_index("sensor_config")
    best_by_sensor.to_csv(out / "best_by_sensor_config.csv")
    summary.sort_values(an.SELECTION).head(15).to_csv(out / "top15_configurations.csv", index=False)
    best_a, best_b = best_by_sensor.loc["A"].copy(), best_by_sensor.loc["B"].copy()
    best_a["sensor_config"], best_b["sensor_config"] = "A", "B"  # the index column was dropped by .loc

    # ---------------- s6 (A vs B) ----------------
    metrics = ["eng_rmse_clip_mean", "win_rmse_clip_mean", "stress_rmse_clip_mean", "eng_rmse_raw_mean", "eng_nasa_raw_mean"]
    s6 = an.paired_effect(summary, "sensor_config", "A", "B", metrics)
    s6.to_csv(out / "s6_paired_configs.csv", index=False)
    s6_stats = {m: an.describe_effect(s6, m) for m in metrics}
    s6_folds = an.fold_pairs(cv, best_a, "sensor_config", "B", "eng_rmse_clip")  # best-A hyper-parameters, with and without s6
    s6_folds_bestb = an.fold_pairs(cv, best_b, "sensor_config", "A", "eng_rmse_clip")
    s6_only = p2.cv_single_config(setup, cfg, best_a, keep_only=("s6",))
    s6_only.to_csv(out / "s6_only_diagnostic.csv", index=False)
    s6_by_baseline = {b: {m: an.describe_effect(s6[s6["baseline"] == b], m) for m in ("eng_rmse_clip_mean", "win_rmse_clip_mean", "stress_rmse_clip_mean")} for b in cfg.grid.baseline}
    # permutation control: real s6 vs shuffled s6 vs no s6, at the best config for each baseline option
    control = []
    for b in cfg.grid.baseline:
        row = an.best_rows(summary[(summary["sensor_config"] == "A") & (summary["baseline"] == b)]).iloc[0].copy()
        real = p2.cv_single_config(setup, cfg, row)["eng_rmse_clip"].mean()
        row_b = row.copy(); row_b["sensor_config"] = "B"
        without = p2.cv_single_config(setup, cfg, row_b)["eng_rmse_clip"].mean()
        shuffled = [p2.cv_single_config(setup, cfg, row, permute_sensor="s6", perm_seed=s)["eng_rmse_clip"].mean() for s in range(10)]
        control.append({"baseline": b, "window": int(row["window"]), "history": row["history"], "metric": row["metric"], "weights": row["weights"], "k": int(row["k"]),
                        "rmse_real_s6": real, "rmse_no_s6": without, "rmse_shuffled_s6_mean": float(np.mean(shuffled)), "rmse_shuffled_s6_std": float(np.std(shuffled)),
                        "rmse_shuffled_s6_min": float(np.min(shuffled)), "rmse_shuffled_s6_max": float(np.max(shuffled))})
    control_df = pd.DataFrame(control)
    control_df.to_csv(out / "s6_permutation_control.csv", index=False)
    Xs6, _, _, _, _ = p2._fold_arrays(build_features(setup.tuning, WindowConfig(int(best_a["window"]), cfg.features.stats, str(best_a["history"]), str(best_a["baseline"]), cfg.features.baseline_cycles)),
                                      setup.tuning, setup.folds[0][0], setup.folds[0][1], "A", keep_only=("s6",))
    _, s6_counts = np.unique(np.round(Xs6, 10), axis=0, return_counts=True)
    dip = an.s6_dip_rate_by_rul(setup.tuning, [-1, 10, 25, 50, 75, 100, 125, 150, 400])
    dip.to_csv(out / "s6_low_value_rate_by_rul.csv")
    last_low = setup.tuning[setup.tuning["s6"] < setup.tuning["s6"].mean()].groupby("unit")["rul"].min()
    s6_dips = {"engines_with_any_low_reading": int(len(last_low)), "tuning_engines": int(setup.tuning["unit"].nunique()),
               "min_rul_of_last_low_reading": int(last_low.min()), "median_rul_of_last_low_reading": float(last_low.median()),
               "low_rate_first_20_cycles": float((setup.tuning.loc[setup.tuning["cycle"] <= 20, "s6"] < setup.tuning["s6"].mean()).mean())}
    s6_ties = {"train_rows_fold0": int(len(Xs6)), "unique_feature_vectors": int(len(s6_counts)), "largest_tie_group": int(s6_counts.max())}

    # ---------------- baselining and short-history strategy ----------------
    base_eff = an.paired_effect(summary, "baseline", "none", "causal", metrics)
    base_eff.to_csv(out / "baseline_paired_configs.csv", index=False)
    hist_eff = an.paired_effect(summary, "history", "adaptive", "pad_first", metrics)
    hist_eff.to_csv(out / "history_paired_configs.csv", index=False)
    base_stats = {m: an.describe_effect(base_eff, m) for m in metrics}
    hist_stats = {m: an.describe_effect(hist_eff, m) for m in metrics}
    base_by_window = {int(w): {b: float(summary[(summary.window == w) & (summary.baseline == b)][an.SELECTION].min()) for b in cfg.grid.baseline} for w in cfg.grid.window}
    base_folds = an.fold_pairs(cv, best, "baseline", "none" if best["baseline"] == "causal" else "causal", "eng_rmse_clip")
    hist_by_window = {int(w): {h: {"stress_rmse_mean": float(summary[(summary.window == w) & (summary.history == h)]["stress_rmse_clip_mean"].mean()),
                                   "eng_rmse_best": float(summary[(summary.window == w) & (summary.history == h)][an.SELECTION].min())} for h in cfg.grid.history} for w in cfg.grid.window}
    dup_info = p2.duplicate_feature_rows(setup, cfg, best)

    # ---------------- backend comparison on CV folds ----------------
    comp = []
    for label, row in (("best A", best_a), ("best B", best_b)):
        c = p2.compare_backends_on_folds(setup, cfg, row, list(range(len(setup.folds))))
        comp.append(c.assign(config=label))
    rng = np.random.default_rng(cfg.seed)
    random_rows = summary.iloc[rng.choice(len(summary), size=6, replace=False)]
    for i, (_, row) in enumerate(random_rows.iterrows()):
        comp.append(p2.compare_backends_on_folds(setup, cfg, row, [0]).assign(config=f"random grid config {i + 1}"))
    comp_df = pd.concat(comp, ignore_index=True)
    comp_df.to_csv(out / "backend_comparison_cv.csv", index=False)

    # ---------------- holdout (touched once) ----------------
    meta_h = p2.eval_meta(setup.holdout, setup.holdout_cutoffs, cfg)
    holdout_rows, holdout_preds = [], {}
    for label, row in (("best A", best_a), ("best B", best_b)):
        preds = {}
        for backend in ("sklearn", "scratch"):
            pcfg = p2.pipeline_config_from_row(row, cfg, backend)
            preds[backend] = RULKNNPipeline(pcfg).fit(setup.tuning).predict_rows(setup.holdout)
        holdout_preds[label] = preds["sklearn"]
        sc = p2.score_predictions(meta_h, preds["sklearn"])
        holdout_rows.append({"model": f"KNN {label}", "max_abs_diff_scratch_vs_sklearn": float(np.abs(preds["sklearn"] - preds["scratch"]).max()), **sc})
    tr_mean = setup.tuning["rul_clipped"].mean()
    holdout_rows.append({"model": "constant (training mean)", "max_abs_diff_scratch_vs_sklearn": np.nan, **p2.score_predictions(meta_h, np.full(len(setup.holdout), tr_mean))})
    holdout = pd.DataFrame(holdout_rows)
    holdout.to_csv(out / "holdout_results.csv", index=False)

    # ---------------- figures ----------------
    f = {
        "k_curves": pp.plot_k_curves(summary, best, figs / "10_k_metric_weights.png"),
        "s6": pp.plot_paired_scatter(s6, "eng_rmse_clip_mean", figs / "11_s6_configuration_A_vs_B.png", "Does removing s6 help? (each dot = one hyper-parameter setting)",
                                    "Engine-level RMSE, config A (s6 kept)", "Engine-level RMSE, config B (s6 removed)"),
        "baseline_history": pp.plot_baseline_and_history(summary, figs / "12_baseline_and_short_history.png"),
        "baseline_scatter": pp.plot_paired_scatter(base_eff, "eng_rmse_clip_mean", figs / "13_baselining_none_vs_causal.png", "Per-engine causal baselining vs raw sensors",
                                                  "Engine-level RMSE, raw sensors", "Engine-level RMSE, causal baseline", color_by="sensor_config"),
    }
    mask = meta_h["is_cutoff"].to_numpy()
    f["holdout"] = pp.plot_holdout_scatter(meta_h["rul_clipped"].to_numpy()[mask], meta_h["rul"].to_numpy()[mask], holdout_preds["best A"][mask], cfg.rul_cap,
                                           "Holdout engines, engine-level cutoffs (best config A)", figs / "14_holdout_predicted_vs_true.png")

    # ---------------- machine-readable summary ----------------
    result = {
        "setup": info, "best_overall": {k: (v.item() if hasattr(v, "item") else v) for k, v in best.items()},
        "best_A": {k: (v.item() if hasattr(v, "item") else v) for k, v in best_a.items()},
        "best_B": {k: (v.item() if hasattr(v, "item") else v) for k, v in best_b.items()},
        "constant_baseline_cv_mean": {c: float(const[c].mean()) for c in const.columns if c != "fold"},
        "s6_paired_stats": s6_stats, "s6_fold_pairs_bestA_hparams": s6_folds.reset_index().to_dict("records"),
        "s6_fold_pairs_bestB_hparams": s6_folds_bestb.reset_index().to_dict("records"),
        "s6_only_cv_mean": {c: float(s6_only[c].mean()) for c in s6_only.columns if c != "fold"}, "s6_only_ties": s6_ties, "s6_low_value_summary": s6_dips,
        "s6_paired_by_baseline": s6_by_baseline, "s6_permutation_control": control_df.to_dict("records"),
        "baseline_paired_stats": base_stats, "baseline_best_by_window": base_by_window, "baseline_fold_pairs_best": base_folds.reset_index().to_dict("records"),
        "history_paired_stats": hist_stats, "history_by_window": hist_by_window, "duplicate_feature_rows_best_config_fold0": dup_info,
        "backend_comparison_max_abs_diff": float(comp_df["max_abs_diff"].max()), "backend_comparison_n_diff_gt_1e-9": int(comp_df["n_diff_gt_1e-9"].sum()),
        "backend_comparison_total_predictions": int(comp_df["n_predictions"].sum()),
        "backend_seconds": {"scratch_mean": float(comp_df["seconds_scratch"].mean()), "sklearn_mean": float(comp_df["seconds_sklearn"].mean())},
        "figures": {k: str(v.relative_to(cfg.paths.root)) for k, v in f.items()},
    }
    (out / "phase2_summary.json").write_text(json.dumps(result, indent=2, default=lambda o: o.item() if hasattr(o, "item") else str(o)), encoding="utf-8")
    logger.info("Phase 2 experiments complete.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
