"""Implementations of the ``scripts/run_phase3.py`` sub-commands (kept out of the script for testability)."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from ..config import Config
from ..data.ingestion import read_subset
from ..data.schema import SENSOR_COLUMNS, SUBSET_NAMES
from ..data.validation import validate_subset
from ..features.regimes import RegimeNormalizer
from . import phase2_analysis as an2
from . import phase3 as p3
from . import phase3_analysis as an
from . import phase3_plots as pl

logger = logging.getLogger(__name__)
EFFECT_METRICS = ["eng_rmse_clip_mean", "eng_nasa_clip_mean", "win_rmse_clip_mean", "stress_rmse_clip_mean"]


def load_setup(cfg: Config, name: str) -> p3.SubsetSetup:
    subset = read_subset(name, cfg.paths.raw_data_dir, cfg.archive_name)
    validate_subset(subset, cfg.subsets[name])
    return p3.make_subset_setup(name, subset.train, cfg)  # only the training file is used


def plain(obj: Any) -> Any:
    if isinstance(obj, pd.Series):
        return {k: plain(v) for k, v in obj.items()}
    if isinstance(obj, pd.DataFrame):
        return [plain(r) for r in obj.to_dict("records")]
    if isinstance(obj, dict):
        return {str(k): plain(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [plain(v) for v in obj]
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating, float)):
        return None if np.isnan(obj) else float(obj)
    if isinstance(obj, (np.bool_,)):
        return bool(obj)
    return obj


def _config_and_metrics(row: pd.Series) -> dict:
    return {k: row[k] for k in p3.CONFIG_KEYS} | {c: float(row[c]) for c in row.index if c.startswith(("eng_", "win_", "stress_")) or c == "n_features"}


def cmd_analyze(cfg: Config, args) -> int:
    out, figs = cfg.paths.results_dir / "phase3", cfg.paths.figures_dir / "phase3"
    cv_dir = out / "cv"
    result: dict[str, Any] = {}
    scatter_data, effects_scatter, cross_rows = {}, {}, []
    for name in SUBSET_NAMES:
        try:
            cv = p3.load_cv(cv_dir, name, cfg.validation.n_folds)
        except FileNotFoundError as exc:
            logger.warning("%s", exc)
            continue
        summary, const = p3.summarise3(cv), p3.constant_reference(cv)
        summary.to_csv(out / f"cv_summary_{name.lower()}.csv", index=False)
        comp = an.compare_selection(summary, cv)
        by_r, by_n = comp["rmse_selected"], comp["nasa_selected"]
        (out / f"selection_{name.lower()}.json").write_text(json.dumps(plain({"rmse_selected": _config_and_metrics(by_r), "nasa_selected": _config_and_metrics(by_n)}), indent=2), encoding="utf-8")
        scatter_data[name] = (summary, by_r, by_n)
        cross_rows.append({"subset": name, **{f"{c}_rmse_sel": by_r[f"eng_{c}_clip_mean"] for c in ("rmse", "mae", "nasa")}, **{f"{c}_nasa_sel": by_n[f"eng_{c}_clip_mean"] for c in ("rmse", "mae", "nasa")},
                           **{f"{c}_const": const[f"eng_{c}_clip"] for c in ("rmse", "mae", "nasa")}})

        entry: dict[str, Any] = {"n_configurations": len(summary), "constant_reference": const,
                                 "selection": {k: v for k, v in comp.items() if k not in ("rmse_selected", "nasa_selected")},
                                 "rmse_selected": _config_and_metrics(by_r), "nasa_selected": _config_and_metrics(by_n)}
        # regime evidence (descriptive, on tuning rows; CV refits per fold)
        setup = load_setup(cfg, name)
        norm = p3.regimes_for(setup, cfg.seed)
        entry["regimes"] = {"n_regimes": norm.n_regimes_, "silhouette_by_k": norm.detector_.silhouette_by_k_}
        if norm.n_regimes_ > 1:
            normed = norm.transform(setup.tuning)
            share = an.regime_variance_share(setup.tuning, norm)
            active = list(share.index)
            raw_t, norm_t = an.trend_strength(setup.tuning, active), an.trend_strength(normed, active)
            tbl = pd.DataFrame({"regime_variance_share": share, "trend_raw": raw_t, "trend_normalised": norm_t})
            tbl.to_csv(out / f"regime_effect_by_sensor_{name.lower()}.csv")
            norm.regime_table(setup.tuning).to_csv(out / f"regimes_{name.lower()}.csv")
            entry["regimes"].update({"regime_variance_share_median": float(share.median()), "regime_variance_share_min": float(share.min()), "regime_variance_share_max": float(share.max()),
                                     "trend_raw_mean": float(raw_t.mean()), "trend_normalised_mean": float(norm_t.mean())})
            sensor = (norm_t - raw_t).idxmax()
            units = sorted(setup.tuning["unit"].unique())[:4]
            pl.plot_regime_effect(setup.tuning, normed, sensor, [int(u) for u in units], figs / f"regime_effect_{name.lower()}_{sensor}.png")
            # normalisation ablation
            eff = an.paired_effect3(summary, "normalization", "global", "regime", EFFECT_METRICS)
            effects_scatter[name] = eff
            ab = {"all": {m: an2.describe_effect(eff, m) for m in EFFECT_METRICS}}
            ab["by_baseline"] = {b: {m: an2.describe_effect(eff[eff["baseline"] == b], m) for m in EFFECT_METRICS} for b in cfg.phase3.grid.baseline}
            ab["best_by_normalization"] = {n: _config_and_metrics(summary[summary["normalization"] == n].loc[lambda d: d[an.RMSE_COL].idxmin()]) for n in ("global", "regime")}
            ab["best_by_normalization_and_baseline"] = {f"{n}/{b}": float(summary[(summary["normalization"] == n) & (summary["baseline"] == b)][an.RMSE_COL].min()) for n in ("global", "regime") for b in cfg.phase3.grid.baseline}
            ab["best_nasa_by_normalization"] = {n: float(summary[summary["normalization"] == n][an.NASA_COL].min()) for n in ("global", "regime")}
            best_regime = summary[summary["normalization"] == "regime"].loc[lambda d: d[an.RMSE_COL].idxmin()]
            ab["fold_pairs_at_best_regime_config"] = an.fold_pairs3(cv, best_regime, "normalization", "global", "eng_rmse_clip").reset_index()
            entry["normalisation_ablation"] = ab
        # factor effects (baselining, history strategy), within each normalisation variant
        entry["factor_effects"] = {}
        for nz in summary["normalization"].unique():
            sub = summary[summary["normalization"] == nz]
            b = an.paired_effect3(sub, "baseline", "none", "causal", EFFECT_METRICS)
            h = an.paired_effect3(sub, "history", "adaptive", "pad_first", EFFECT_METRICS)
            entry["factor_effects"][nz] = {"causal_minus_none": {m: an2.describe_effect(b, m) for m in EFFECT_METRICS}, "pad_first_minus_adaptive": {m: an2.describe_effect(h, m) for m in EFFECT_METRICS}}
        result[name] = entry
        logger.info("%s analysed (%d configurations)", name, len(summary))

    pl.plot_selection_scatter(scatter_data, figs / "rmse_vs_nasa_selection.png")
    if effects_scatter:
        pl.plot_normalisation_scatter(effects_scatter, "eng_rmse_clip_mean", figs / "regime_normalisation_ablation.png")
    cross = pd.DataFrame(cross_rows)
    cross.to_csv(out / "cross_subset_cv_summary.csv", index=False)
    pl.plot_cross_subset(cross, figs / "cross_subset_cv.png")
    (out / "phase3_analysis.json").write_text(json.dumps(plain(result), indent=2), encoding="utf-8")
    return 0


def pipeline_config_from_selection(sel: dict, cfg: Config, normalization: str | None = None):
    from ..features.windows import WindowConfig
    from ..models.pipeline import PipelineConfig

    return PipelineConfig(
        window=WindowConfig(int(sel["window"]), cfg.features.stats, str(sel["history"]), str(sel["baseline"]), cfg.features.baseline_cycles),
        sensor_config="A", n_neighbors=int(sel["k"]), metric=str(sel["metric"]), weights=str(sel["weights"]), backend="sklearn",
        normalization=normalization or str(sel["normalization"]), seed=cfg.seed,
    )


def cmd_holdout(cfg: Config, args) -> int:
    """Evaluate the frozen CV selections on the subset's development holdout, once (counted in an audit file)."""
    from ..models.pipeline import RULKNNPipeline

    name, out = args.subset, cfg.paths.results_dir / "phase3"
    if name == "FD001":
        logger.error("FD001's development holdout was already evaluated repeatedly in Phase 2 and is not re-evaluated in Phase 3.")
        return 1
    audit_path = p3.holdout_audit_path(cfg.paths.results_dir)
    audit = json.loads(audit_path.read_text(encoding="utf-8")) if audit_path.is_file() else {}
    if audit.get(name, {}).get("evaluations", 0) >= 1 and not args.force:
        logger.error("%s's development holdout has already been evaluated %d time(s). Use --force to repeat (it will be counted).", name, audit[name]["evaluations"])
        return 1
    sel = json.loads((out / f"selection_{name.lower()}.json").read_text(encoding="utf-8"))
    setup = load_setup(cfg, name)
    meta = p3.query_meta(setup.holdout, setup.holdout_cutoffs, cfg)
    q = meta["is_query"].to_numpy()
    models = {"KNN selected by RMSE": sel["rmse_selected"], "KNN selected by NASA": sel["nasa_selected"]}
    if sel["rmse_selected"]["normalization"] == "regime":
        models["KNN selected by RMSE, but raw-sensor scaling"] = dict(sel["rmse_selected"], normalization="global")
    rows, cache = [], {}
    for label, s in models.items():
        key = tuple(s[k] for k in p3.CONFIG_KEYS)
        if key not in cache:
            cache[key] = RULKNNPipeline(pipeline_config_from_selection(s, cfg)).fit(setup.tuning).predict_rows(setup.holdout)
        rows.append({"subset": name, "model": label, **{k: s[k] for k in p3.CONFIG_KEYS}, **p3.score3(meta[q], cache[key][q])})
    rows.append({"subset": name, "model": "constant (tuning-set mean)", **{k: "-" for k in p3.CONFIG_KEYS}, **p3.score3(meta[q], np.full(int(q.sum()), setup.tuning["rul_clipped"].mean()))})
    pd.DataFrame(rows).to_csv(out / f"holdout_{name.lower()}.csv", index=False)
    n = p3.record_holdout_evaluation(cfg.paths.results_dir, name, "frozen CV selections evaluated once")
    logger.info("%s development holdout evaluated (evaluation #%d); %d engines", name, n, setup.holdout["unit"].nunique())
    return 0


def cmd_probe(cfg: Config, args) -> int:
    """Short-history probe (resumable per fold)."""
    import itertools
    import time

    from ..features.regimes import RegimeNormalizer
    from ..features.sensors import SensorSelector
    from ..features.windows import WindowConfig
    from ..models.pipeline import PipelineConfig, RULKNNPipeline
    from . import history_probe as hp

    name, pc = args.subset, cfg.phase3.probe
    out = cfg.paths.results_dir / "phase3" / "probe"
    out.mkdir(parents=True, exist_ok=True)
    setup = load_setup(cfg, name)
    norm_mode = "regime" if p3.regimes_for(setup, cfg.seed).n_regimes_ > 1 else "global"
    df = setup.tuning
    for fold, (train_units, val_units) in enumerate(setup.folds):
        path = out / f"{name.lower()}_fold{fold}.csv"
        if path.is_file() or (args.folds is not None and fold not in args.folds):
            continue
        t0 = time.time()
        train, val = df[df["unit"].isin(train_units)].reset_index(drop=True), df[df["unit"].isin(val_units)].reset_index(drop=True)
        segments, truth = hp.make_probes(val, pc.stages, pc.n_obs, cfg.rul_cap)
        normalizer = RegimeNormalizer(seed=cfg.seed).fit(train) if norm_mode == "regime" else None
        rep_train = normalizer.transform(train) if normalizer else train
        rep_seg = normalizer.transform(segments) if normalizer else segments
        sensors = SensorSelector("A").fit(rep_train).selected_sensors_
        truth = truth.join(hp.degradation_index(rep_seg, rep_train[rep_train["cycle"] <= 20], sensors), on="probe_id")
        rows = []
        for window, history, baseline in itertools.product(pc.window, pc.history, pc.baseline):
            wc = WindowConfig(window, cfg.features.stats, history, baseline, cfg.features.baseline_cycles)
            model = RULKNNPipeline(PipelineConfig(wc, "A", pc.k, pc.metric, pc.weights, "sklearn", norm_mode, cfg.seed)).fit(train, normalizer=normalizer)
            pred = model.predict_last_cycle(segments)
            r = truth.assign(pred=pred.reindex(truth["probe_id"]).to_numpy(), window=window, history=history, baseline=baseline, fold=fold)
            rows.append(r)
        pd.concat(rows, ignore_index=True).to_csv(path, index=False)
        logger.info("%s probe fold %d done in %.0fs (normalisation: %s)", name, fold, time.time() - t0, norm_mode)
    return 0


def cmd_faultmodes(cfg: Config, args) -> int:
    """Two-group structure in end-of-life sensor signatures, and CV error per group."""
    from ..features.sensors import SensorSelector
    from ..features.windows import WindowConfig  # noqa: F401
    from ..models.pipeline import RULKNNPipeline
    from . import fault_modes as fm

    out, figs = cfg.paths.results_dir / "phase3", cfg.paths.figures_dir / "phase3"
    rows, pcas = [], {}
    for name in SUBSET_NAMES:
        setup = load_setup(cfg, name)
        df = setup.tuning
        norm = p3.regimes_for(setup, cfg.seed)
        rep = norm.transform(df) if norm.n_regimes_ > 1 else df
        sensors = list(SensorSelector("A").fit(rep).selected_sensors_)
        res = fm.split_in_two(fm.end_of_life_signature(rep, sensors), cfg.seed)
        lifetimes = df.groupby("unit")["cycle"].max().sort_index()
        cluster = pd.Series(res["labels"], index=lifetimes.index)
        row = {"subset": name, "documented_fault_modes": "+".join(cfg.subsets[name].fault_modes), "engines": len(lifetimes), "silhouette_k2": res["silhouette"],
               "null_silhouette_mean": res["null_silhouette_mean"], "null_silhouette_max": res["null_silhouette_max"], "share_of_null_below": res["share_of_null_below"],
               "group_sizes": str(res["sizes"]), "top_sensors": ",".join(res["top_sensors"]),
               "median_lifetime_group0": float(lifetimes[cluster == 0].median()), "median_lifetime_group1": float(lifetimes[cluster == 1].median())}
        # CV error per group with the RMSE-selected configuration (out-of-fold predictions at engine-level cutoffs)
        sel = json.loads((out / f"selection_{name.lower()}.json").read_text(encoding="utf-8"))["rmse_selected"]
        meta = p3.query_meta(df, setup.cutoffs, cfg)
        pred = np.full(len(df), np.nan)
        for tru, vau in setup.folds:
            tr, va = df["unit"].isin(tru).to_numpy(), df["unit"].isin(vau).to_numpy()
            model = RULKNNPipeline(pipeline_config_from_selection(sel, cfg)).fit(df[tr].reset_index(drop=True))
            pred[va] = model.predict_rows(df[va].reset_index(drop=True))
        cut = meta["is_cutoff"].to_numpy()
        err = pd.DataFrame({"unit": df["unit"].to_numpy()[cut], "err": pred[cut] - meta["rul_clipped"].to_numpy()[cut]})
        err["group"] = err["unit"].map(cluster)
        for gname, g in err.groupby("group"):
            row[f"cv_rmse_group{int(gname)}"] = float(np.sqrt((g["err"] ** 2).mean()))
        rows.append(row)
        pcas[name] = (res["pca"], res["labels"], res["silhouette"])
        logger.info("%s: silhouette(k=2) %.3f vs null mean %.3f", name, res["silhouette"], res["null_silhouette_mean"])
    pd.DataFrame(rows).to_csv(out / "fault_mode_structure.csv", index=False)
    pts = [pd.DataFrame({"subset": n, "pc1": xy[:, 0], "pc2": xy[:, 1], "group": lab}) for n, (xy, lab, _) in pcas.items()]
    pd.concat(pts, ignore_index=True).to_csv(out / "fault_mode_points.csv", index=False)  # the figure is drawn from this file
    return 0


def plot_fault_modes_from_saved(cfg: Config) -> None:
    from matplotlib.figure import Figure

    from ..eda.plots import _save, _styled

    out, figs = cfg.paths.results_dir / "phase3", cfg.paths.figures_dir / "phase3"
    pts, tab = pd.read_csv(out / "fault_mode_points.csv"), pd.read_csv(out / "fault_mode_structure.csv").set_index("subset")

    @_styled
    def _plot(path):
        fig = Figure(figsize=(11, 8.5), layout="constrained")
        axes = fig.subplots(2, 2)
        for ax, (name, g) in zip(axes.flat, pts.groupby("subset", sort=False)):
            ax.scatter(g["pc1"], g["pc2"], c=np.where(g["group"] == 0, "#1f77b4", "#d62728"), s=16, alpha=0.8)
            ax.set_title(f"{name} ({tab.loc[name, 'documented_fault_modes']}), silhouette {tab.loc[name, 'silhouette_k2']:.2f}")
            ax.set_xlabel("PC 1"); ax.set_ylabel("PC 2")
        fig.suptitle("End-of-life sensor signatures of training engines (two k-means groups, unlabelled data)")
        return _save(fig, path)

    _plot(figs / "fault_mode_structure.png")


def _md(df: pd.DataFrame, fmt: str = "{:.3f}") -> str:
    head = "| " + " | ".join(str(c) for c in df.columns) + " |"
    sep = "|" + "|".join(["---"] * len(df.columns)) + "|"
    rows = ["| " + " | ".join(fmt.format(v) if isinstance(v, (float, np.floating)) else str(v) for v in r) + " |" for r in df.itertuples(index=False)]
    return "\n".join([head, sep, *rows])


def cmd_report(cfg: Config, args) -> int:
    """Write results/phase3/phase3_report.md; every number is read from files written by the other sub-commands."""
    out, figs = cfg.paths.results_dir / "phase3", cfg.paths.figures_dir / "phase3"
    A = json.loads((out / "phase3_analysis.json").read_text(encoding="utf-8"))
    audit = json.loads(p3.holdout_audit_path(cfg.paths.results_dir).read_text(encoding="utf-8"))
    g = cfg.phase3.grid
    L = ["# Phase 3 results (generated)", "", "Generated by `scripts/run_phase3.py report`. The official test files were not read by any Phase 3 step.", "",
         "## Protocol", "",
         f"- Same grid for every subset (frozen before inspecting results): baseline {list(g.baseline)}, normalisation {list(g.normalization)} (regime only where more than one regime is detected), window {list(g.window)}, history {list(g.history)}, metric {list(g.metric)}, weights {list(g.weights)}, k {list(g.k)}.",
         f"- Engine-level CV ({cfg.validation.n_folds} folds over tuning engines); {cfg.validation.eval_cutoffs_per_engine} random cutoffs per engine from cycle {cfg.validation.eval_min_cutoff}; stress cutoffs {list(cfg.validation.stress_cutoffs)}; window-level metrics use every {cfg.phase3.window_eval_stride}th cycle.",
         "- All `*_clip` metrics compare with the RUL clipped at 125; `*_raw` with the true RUL. NASA scores are the mean per prediction (see docs/EVALUATION_PROTOCOL.md).",
         f"- Development holdouts (exploratory): {int(cfg.phase3.holdout_fraction * 100)}% of training engines per subset. Evaluations so far: { {k: v['evaluations'] for k, v in audit.items()} } (FD001's holdout was evaluated repeatedly in Phase 2 and is not used here).", ""]

    L += ["## Operating regimes", "",
          "Rule (see `features/regimes.py`): for k = 2..10 run k-means on the standardised operational settings (training rows only), compute the silhouette score, "
          "pick the k with the highest silhouette, and accept it only if that silhouette is at least 0.9; otherwise the subset has one regime. "
          "Detector outputs are saved in `results/phase3/regimes/<subset>_detector.json`.", ""]
    det = {n: json.loads((out / "regimes" / f"{n.lower()}_detector.json").read_text(encoding="utf-8")) for n in SUBSET_NAMES if (out / "regimes" / f"{n.lower()}_detector.json").is_file()}
    rows = []
    for n, d in det.items():
        e = A.get(n, {}).get("regimes", {})
        rows.append({"subset": n, "selected number of regimes": d["selected_n_regimes"], "k with highest silhouette": d["k_with_highest_silhouette"], "highest silhouette": d["highest_silhouette"],
                     "distinct rounded settings (independent)": d["independent_check"]["distinct_rounded_operating_settings"], "agreement with detector (ARI)": d["independent_check"]["adjusted_rand_index_vs_detector"],
                     "regimes found in each CV fold": str([f["n_regimes"] for f in d["per_fold"]]), "min fold ARI": min(f["ari_vs_full_tuning_fit"] for f in d["per_fold"]),
                     "median share of raw sensor variance explained by regime": e.get("regime_variance_share_median", ""), "mean |trend| raw": e.get("trend_raw_mean", ""), "mean |trend| normalised": e.get("trend_normalised_mean", "")})
    L += [_md(pd.DataFrame(rows)), "", "Silhouette score for every k tried (the selected k is the highest value, and only if it reaches 0.9):", ""]
    ks = sorted({int(k) for d in det.values() for k in d["silhouette_by_k"]})
    L += [_md(pd.DataFrame([{"k": k, **{n: d["silhouette_by_k"].get(str(k), float("nan")) for n, d in det.items()}} for k in ks])), "", "Regimes (tuning rows):", ""]
    for n in ("FD002", "FD004"):
        if (out / f"regimes_{n.lower()}.csv").is_file():
            L += [n, "", _md(pd.read_csv(out / f"regimes_{n.lower()}.csv")), ""]

    L += ["## Cross-validation results per subset (engine level, vs clipped RUL)", ""]
    rows = []
    for n, e in A.items():
        for lab, key in (("selected by RMSE", "rmse_selected"), ("selected by NASA", "nasa_selected")):
            s = e[key]
            rows.append({"subset": n, "criterion": lab, "normalisation": s["normalization"], "baseline": s["baseline"], "window": s["window"], "history": s["history"], "metric": s["metric"], "weights": s["weights"], "k": s["k"],
                         "RMSE": s["eng_rmse_clip_mean"], "RMSE std": s["eng_rmse_clip_std"], "MAE": s["eng_mae_clip_mean"], "NASA (mean)": s["eng_nasa_clip_mean"], "bias": s["eng_bias_clip_mean"], "late share": s["eng_late_clip_mean"]})
        c = e["constant_reference"]
        rows.append({"subset": n, "criterion": "constant predictor", **{k: "" for k in ("normalisation", "baseline", "window", "history", "metric", "weights", "k", "RMSE std")}, "RMSE": c["eng_rmse_clip"], "MAE": c["eng_mae_clip"], "NASA (mean)": c["eng_nasa_clip"], "bias": c["eng_bias_clip"], "late share": c["eng_late_clip"]})
    L += [_md(pd.DataFrame(rows)), "", f"Best configuration sits at the grid edge for: window {g.window[-1]} (all subsets where it was selected), k {g.k[-1]} where selected.", ""]

    L += ["## Grid extension", "", "Window 100 and k = 200 were added for FD002 and FD004 after this analysis (the selected values sat at the grid boundary). They are analysed in `results/phase3_extended/phase3_extended_report.md`; the tables in this report use the original Phase 3 grid and are unchanged.", ""]
    L += ["## RMSE-selected vs NASA-selected configurations", ""]
    rows = [{"subset": n, "same configuration": e["selection"]["same_configuration"], "Spearman (RMSE vs NASA over configs)": e["selection"]["spearman_rmse_vs_nasa_over_configs"], "top-10 overlap": e["selection"]["top10_overlap"],
             "RMSE(NASA-sel) - RMSE(RMSE-sel)": e["selection"]["rmse_of_nasa_selected_minus_rmse_selected"], "NASA(RMSE-sel) - NASA(NASA-sel)": e["selection"]["nasa_of_rmse_selected_minus_nasa_selected"],
             "folds: RMSE-sel has lower RMSE": e["selection"]["folds_where_rmse_selected_has_lower_rmse"], "folds: NASA-sel has lower NASA": e["selection"]["folds_where_nasa_selected_has_lower_nasa"]} for n, e in A.items()]
    L += [_md(pd.DataFrame(rows)), ""]

    L += ["## Operating-condition normalisation ablation (global scaling -> regime-wise normalisation)", "", "Difference = regime - global over matched configurations; negative means regime normalisation is better.", ""]
    for n, e in A.items():
        if "normalisation_ablation" not in e:
            continue
        ab = e["normalisation_ablation"]
        rows = [{"metric": m.replace("_mean", ""), "matched configs": v["matched_configs"], "mean diff": v["mean_diff"], "share where regime is better": v["share_other_better"], "min diff": v["min_diff"], "max diff": v["max_diff"]} for m, v in ab["all"].items()]
        L += [f"### {n}", "", _md(pd.DataFrame(rows)), "", "Best CV RMSE by normalisation and baseline: " + ", ".join(f"{k}: {v:.2f}" for k, v in ab["best_by_normalization_and_baseline"].items()),
              "", "Best CV NASA by normalisation: " + ", ".join(f"{k}: {v:.3f}" for k, v in ab["best_nasa_by_normalization"].items()),
              "", "Per fold at the best regime configuration, global RMSE minus regime RMSE (positive = regime normalisation is better): " + ", ".join(f"{x['diff']:+.2f}" for x in ab["fold_pairs_at_best_regime_config"]), ""]

    L += ["## Baselining and history strategy across subsets", "", "Difference = second - first over matched configurations within each normalisation variant (negative = second is better).", ""]
    rows = []
    for n, e in A.items():
        for nz, fe in e["factor_effects"].items():
            for lab, key in (("causal - none", "causal_minus_none"), ("pad_first - adaptive", "pad_first_minus_adaptive")):
                d = fe[key]
                rows.append({"subset": n, "normalisation": nz, "switch": lab, "RMSE mean diff": d["eng_rmse_clip_mean"]["mean_diff"], "share better (RMSE)": d["eng_rmse_clip_mean"]["share_other_better"], "NASA mean diff": d["eng_nasa_clip_mean"]["mean_diff"],
                             "share better (NASA)": d["eng_nasa_clip_mean"]["share_other_better"], "stress RMSE mean diff": d["stress_rmse_clip_mean"]["mean_diff"], "share better (stress)": d["stress_rmse_clip_mean"]["share_other_better"]})
    L += [_md(pd.DataFrame(rows)), ""]

    L += ["## Development holdouts (exploratory; each evaluated once, frozen CV selections)", ""]
    for n in ("FD002", "FD003", "FD004"):
        p = out / f"holdout_{n.lower()}.csv"
        if p.is_file():
            h = pd.read_csv(p)
            L += [f"### {n}", "", _md(h[["model", "normalization", "baseline", "window", "history", "metric", "weights", "k", "eng_rmse_clip", "eng_mae_clip", "eng_nasa_clip", "eng_bias_clip", "eng_late_clip", "eng_rmse_raw", "win_rmse_clip", "stress_rmse_clip", "n_eng"]]), ""]
    p2h = pd.read_csv(cfg.paths.results_dir / "phase2" / "holdout_results.csv")
    L += ["### FD001 (Phase 2 exploratory holdout, evaluated more than once after the grid changed)", "", _md(p2h[["model", "eng_rmse_clip", "eng_mae_clip", "eng_nasa_clip", "eng_bias_clip", "eng_rmse_raw", "win_rmse_clip", "stress_rmse_clip", "n_eng"]]), ""]

    L += ["## Short-history probe", "", "Histories are cut from validation engines (re-numbered from cycle 1) and predicted at their last cycle. Fixed model: Manhattan, distance weights, k=50; truth is the clipped RUL. "
          "`degradation_index` is the mean |z| of the first 3 readings against the healthy reference (about 0.8 = healthy).", ""]
    probe_metrics = {}
    for n in ("FD001", "FD002"):
        try:
            d = an.load_probe(out / "probe", n)
        except FileNotFoundError:
            continue
        t = d.drop_duplicates(["fold", "probe_id"])
        L += [f"### {n}", "", "Degradation index quartiles by stage:", "", _md(t.groupby("stage")["degradation_index"].quantile([0.25, 0.5, 0.75]).unstack().reset_index().rename(columns={0.25: "q25", 0.5: "median", 0.75: "q75"})), ""]
        m = an.probe_metrics(d, ["stage", "n_obs", "history", "baseline"])
        m.to_csv(out / f"probe_summary_{n.lower()}.csv", index=False)
        probe_metrics[n] = m
        for base in ("none", "causal"):
            L += [f"RMSE by stage and history length, baseline = {base} (pooled over window lengths):", "", _md(m[m.baseline == base].pivot_table(index=["stage", "n_obs"], columns="history", values="rmse").reset_index(), "{:.1f}"), ""]
        k = d[(d.baseline == "none") & (d.n_obs == 10)]
        L += ["By window length (baseline none, 10 observed cycles):", "", _md(an.probe_metrics(k, ["stage", "window", "history"]).pivot_table(index=["stage", "window"], columns="history", values="rmse").reset_index(), "{:.1f}"), "",
              "Bias, late share and mean NASA score (baseline none, 10 observed cycles, pooled windows):", "", _md(an.probe_metrics(k, ["stage", "history"])[["stage", "history", "n", "rmse", "mae", "bias", "late", "nasa"]], "{:.2f}"), ""]
    if probe_metrics:
        pl.plot_probe(probe_metrics, list(cfg.phase3.probe.stages), figs / "short_history_probe.png")
    fmodes = out / "fault_mode_structure.csv"
    if fmodes.is_file():
        plot_fault_modes_from_saved(cfg)
        L += ["## Fault-mode structure (unsupervised, descriptive)", "", _md(pd.read_csv(fmodes)), ""]
    (out / "phase3_report.md").write_text("\n".join(L), encoding="utf-8")
    return 0


def cmd_regimes(cfg: Config, args) -> int:
    """Save the regime-detector output of every subset and check it independently.

    Independent check: group the rows by their operating settings rounded to whole numbers (no clustering involved)
    and compare that grouping with the detector's regimes (adjusted Rand index; 1.0 = identical partitions).
    Also refits the detector inside every CV fold and compares each fold's regimes with the full-tuning fit.
    Only training rows are used.
    """
    from sklearn.metrics import adjusted_rand_score

    out = cfg.paths.results_dir / "phase3" / "regimes"
    out.mkdir(parents=True, exist_ok=True)
    for name in SUBSET_NAMES:
        setup = load_setup(cfg, name)
        df = setup.tuning
        norm = RegimeNormalizer(seed=cfg.seed).fit(df)
        det = norm.detector_
        labels = det.predict(df)
        centres = det._centers * det._scale + det._mean  # back to the original units of op1, op2, op3
        rounded = df[["op1", "op2", "op3"]].round(0)
        codes = rounded.apply(tuple, axis=1).astype("category").cat.codes.to_numpy()
        folds = []
        for f, (tru, _) in enumerate(setup.folds):
            fd = RegimeNormalizer(seed=cfg.seed).fit(df[df["unit"].isin(tru)]).detector_
            folds.append({"fold": f, "n_regimes": fd.n_regimes_, "ari_vs_full_tuning_fit": float(adjusted_rand_score(labels, fd.predict(df)))})
        sil = {int(k): float(v) for k, v in det.silhouette_by_k_.items()}
        best_k = max(sil, key=sil.get) if sil else 1
        rec = {"subset": name, "seed": cfg.seed, "rows_used": int(len(df)), "uses_test_data": False,
               "rule": {"k_range": [2, int(det.max_regimes)], "accept_best_k_if_silhouette_at_least": det.min_silhouette, "silhouette_sample_size": det.sample_size},
               "selected_n_regimes": int(det.n_regimes_), "k_with_highest_silhouette": int(best_k), "highest_silhouette": sil.get(best_k),
               "silhouette_by_k": sil, "rows_per_regime": np.bincount(labels, minlength=det.n_regimes_).tolist(),
               "centres_original_units": pd.DataFrame(centres, columns=["op1", "op2", "op3"]).round(4).to_dict("records"),
               "independent_check": {"distinct_rounded_operating_settings": int(len(np.unique(codes))), "adjusted_rand_index_vs_detector": float(adjusted_rand_score(codes, labels))},
               "per_fold": folds}
        (out / f"{name.lower()}_detector.json").write_text(json.dumps(plain(rec), indent=2), encoding="utf-8")
        logger.info("%s: selected %d regime(s); independent check: %d distinct rounded settings, ARI %.3f; fold regimes %s",
                    name, det.n_regimes_, rec["independent_check"]["distinct_rounded_operating_settings"], rec["independent_check"]["adjusted_rand_index_vs_detector"], [x["n_regimes"] for x in folds])
    return 0


def cmd_extend(cfg: Config, args) -> int:
    """Run (or resume) the FD002/FD004 grid extension; ``--status`` only reports progress."""
    from . import phase3_extension as ext

    if args.subset not in cfg.phase3_extension.subsets:
        logger.error("The extension applies to %s only.", list(cfg.phase3_extension.subsets))
        return 1
    out_dir = cfg.paths.results_dir / "phase3_extended" / "cv"
    norms = ("global", "regime")  # both extended subsets have six regimes (verified by the detector, see results/phase3/regimes)
    folds = args.folds if args.folds is not None else list(range(cfg.validation.n_folds))
    parts = args.parts or list(ext.PARTS)
    if args.status:
        for f in range(cfg.validation.n_folds):
            print(args.subset, f, {p: ext.chunk_status(cfg, out_dir, args.subset, f, p, norms) for p in ext.PARTS})
        return 0
    setup = load_setup(cfg, args.subset)
    if p3.regimes_for(setup, cfg.seed).n_regimes_ <= 1:
        logger.error("%s has a single regime; normalisation variants would be wrong.", args.subset)
        return 1
    for f in folds:
        for part in parts:
            ext.run_chunk(setup, cfg, out_dir, f, part, norms)
    return 0


def cmd_analyze_extended(cfg: Config, args) -> int:
    """Selection over the extended grid (FD002, FD004). Reads saved chunks only; writes to results/phase3_extended/."""
    from . import phase3_extension as ext

    out, figs = cfg.paths.results_dir / "phase3_extended", cfg.paths.figures_dir / "phase3_extended"
    out.mkdir(parents=True, exist_ok=True)
    norms, k_old = ("global", "regime"), max(cfg.phase3.grid.k)
    L = ["# Phase 3 grid extension: FD002 and FD004 (generated)", "",
         "Generated by `scripts/run_phase3.py analyze-extended` from saved chunks in `results/phase3_extended/cv/` and the original Phase 3 folds. "
         "The original Phase 3 results are not modified. The official test files were not read, and the development holdouts were not re-evaluated.", "",
         f"Added to the Phase 3 grid: window {list(cfg.phase3_extension.add_window)} and k {list(cfg.phase3_extension.add_k)}. Primary selection criterion: engine-level CV RMSE (vs clipped RUL). "
         "MAE and the mean NASA score are reported with it; the NASA-selected model is shown only as a secondary comparison.", ""]
    result, curves, rows_tab = {}, {}, []
    for name in cfg.phase3_extension.subsets:
        orig = p3.load_cv(cfg.paths.results_dir / "phase3" / "cv", name, cfg.validation.n_folds)
        chunks = []
        for f in range(cfg.validation.n_folds):
            for part in ext.PARTS:
                if ext.chunk_status(cfg, out / "cv", name, f, part, norms) != "complete":
                    raise FileNotFoundError(f"Extension incomplete for {name}: fold {f} part {part} (run `extend` first)")
                d = pd.read_csv(ext.chunk_files(out / "cv", name, f, part)[0])
                chunks.append(d.assign(part=part))
        extra = pd.concat(chunks, ignore_index=True)
        overlap = extra[(extra["part"] == "k200") & (extra["k"] == k_old)]
        orig_cfg = orig[orig["normalization"] != "constant"]
        merged = overlap.merge(orig_cfg, on=["fold"] + p3.CONFIG_KEYS, suffixes=("_new", "_old"))
        cols = [c for c in orig_cfg.columns if c.startswith(("eng_", "win_", "stress_"))]
        max_diff = float(max((merged[c + "_new"] - merged[c + "_old"]).abs().max() for c in cols))
        new_rows = extra[~((extra["part"] == "k200") & (extra["k"] == k_old))].drop(columns="part")
        full = pd.concat([orig, new_rows], ignore_index=True)
        n_dupes = int(full[full["normalization"] != "constant"].duplicated(["fold"] + p3.CONFIG_KEYS).sum())
        summary, const = p3.summarise3(full), p3.constant_reference(full)
        n_original = len(p3.summarise3(orig))
        summary.to_csv(out / f"cv_summary_extended_{name.lower()}.csv", index=False)
        comp = an.compare_selection(summary, full)
        new_r, new_n = comp["rmse_selected"], comp["nasa_selected"]
        old = json.loads((cfg.paths.results_dir / "phase3" / f"selection_{name.lower()}.json").read_text(encoding="utf-8"))["rmse_selected"]
        old_row = summary[(summary[p3.CONFIG_KEYS].astype(str) == pd.Series({k: str(old[k]) for k in p3.CONFIG_KEYS})).all(axis=1)].iloc[0]
        changed = [k for k in p3.CONFIG_KEYS if str(old[k]) != str(new_r[k])]
        (out / f"selection_extended_{name.lower()}.json").write_text(json.dumps(plain({"rmse_selected_primary": _config_and_metrics(new_r), "nasa_selected_secondary": _config_and_metrics(new_n)}), indent=2), encoding="utf-8")
        fp = an.fold_values(full, new_r, ["eng_rmse_clip", "eng_mae_clip", "eng_nasa_clip"]).join(an.fold_values(full, old_row, ["eng_rmse_clip", "eng_mae_clip", "eng_nasa_clip"]), lsuffix="_new", rsuffix="_old")
        wins = sorted(summary["window"].unique())
        curves[name] = summary
        result[name] = {"configurations": len(summary), "duplicate_config_rows": n_dupes, "overlap_rows_compared": len(merged), "overlap_max_abs_diff_vs_original": max_diff,
                        "changed_fields_vs_original_selection": changed, "new_best_window": int(new_r["window"]), "new_best_k": int(new_r["k"]), "largest_window_in_grid": int(wins[-1]), "largest_k_in_grid": int(summary["k"].max()),
                        "selection": {k: v for k, v in comp.items() if k not in ("rmse_selected", "nasa_selected")}}
        L += [f"## {name}", "", f"- Configurations after extension: {len(summary)} (original {n_original} + {len(summary) - n_original} new). Duplicate (fold, configuration) rows: {n_dupes}.",
              f"- Reproducibility check: the {len(merged)} k={k_old} rows recomputed in the extension match the original Phase 3 results (max absolute difference over all metrics: {max_diff:.2e}).", ""]
        tab = []
        for lab, r in (("original grid, selected by RMSE", old_row), ("extended grid, selected by RMSE (primary)", new_r), ("extended grid, selected by NASA (secondary)", new_n)):
            tab.append({"model": lab, **{k: r[k] for k in p3.CONFIG_KEYS}, "RMSE": r["eng_rmse_clip_mean"], "RMSE std (folds)": r["eng_rmse_clip_std"], "MAE": r["eng_mae_clip_mean"], "NASA (mean)": r["eng_nasa_clip_mean"], "bias": r["eng_bias_clip_mean"], "late share": r["eng_late_clip_mean"]})
            rows_tab.append({"subset": name, **tab[-1]})
        tab.append({"model": "constant predictor", **{k: "" for k in p3.CONFIG_KEYS}, "RMSE": const["eng_rmse_clip"], "RMSE std (folds)": "", "MAE": const["eng_mae_clip"], "NASA (mean)": const["eng_nasa_clip"], "bias": const["eng_bias_clip"], "late share": const["eng_late_clip"]})
        L += [_md(pd.DataFrame(tab)), "",
              f"- Selection changed relative to the original grid: {'no' if not changed else 'yes, in ' + ', '.join(changed)}. Largest window in the extended grid: {wins[-1]}; largest k: {int(summary['k'].max())}. "
              f"The new best has window {int(new_r['window'])} and k {int(new_r['k'])}.",
              f"- RMSE-selected and NASA-selected models are the {'same' if comp['same_configuration'] else 'different'} configuration; Spearman rank correlation of RMSE and NASA over all configurations: {comp['spearman_rmse_vs_nasa_over_configs']:.3f}; top-10 overlap {comp['top10_overlap']}.",
              "", "Per fold, extended-grid best vs original-grid best (RMSE, MAE, NASA):", "", _md(fp.reset_index())[:0] + _md(fp.reset_index().rename(columns={"index": "fold"})), "",
              "Best CV RMSE by window length (minimum over all other settings):", "", _md(summary.groupby("window")["eng_rmse_clip_mean"].min().reset_index().rename(columns={"eng_rmse_clip_mean": "best RMSE"})), "",
              "Best CV RMSE by k:", "", _md(summary.groupby("k")["eng_rmse_clip_mean"].min().reset_index().rename(columns={"eng_rmse_clip_mean": "best RMSE"})), ""]
    pl.plot_extension_curves(curves, tuple(cfg.phase3.grid.window), k_old, figs / "extension_curves.png")
    pd.DataFrame(rows_tab).to_csv(out / "extended_selection_table.csv", index=False)
    (out / "extended_analysis.json").write_text(json.dumps(plain(result), indent=2), encoding="utf-8")
    (out / "phase3_extended_report.md").write_text("\n".join(L), encoding="utf-8")
    return 0
