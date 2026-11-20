"""Orchestrates the EDA for one subset: statistics, figures, tables and a report."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from ..config import Config
from ..data.ingestion import CMAPSSSubset
from ..data.schema import OP_SETTING_COLUMNS, SENSOR_COLUMNS
from ..data.targets import add_train_rul, truncated_engine_table
from . import plots, statistics as st

logger = logging.getLogger(__name__)


def _clean(obj: Any) -> Any:
    """Make nested structures JSON-safe (numpy scalars -> Python, NaN -> None)."""
    if isinstance(obj, dict):
        return {str(k): _clean(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_clean(v) for v in obj]
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating, float)):
        return None if np.isnan(obj) else float(obj)
    if isinstance(obj, (np.bool_,)):
        return bool(obj)
    return obj


def _md_table(df: pd.DataFrame, floatfmt: str = "{:.3f}") -> str:
    header = "| " + " | ".join([df.index.name or ""] + [str(c) for c in df.columns]) + " |"
    sep = "|" + "|".join(["---"] * (len(df.columns) + 1)) + "|"
    rows = []
    for idx, row in df.iterrows():
        cells = [str(idx)]
        for v in row:
            if isinstance(v, (float, np.floating)):
                cells.append("" if np.isnan(v) else floatfmt.format(v))
            else:
                cells.append(str(v))
        rows.append("| " + " | ".join(cells) + " |")
    return "\n".join([header, sep, *rows])


def run_eda(subset: CMAPSSSubset, cfg: Config, figures_dir: Path | None = None, results_dir: Path | None = None) -> dict[str, Any]:
    """Run the full EDA on ``subset`` and write figures, CSV/JSON tables and a markdown report.

    Only the training data is used for anything that resembles fitting
    (constant-sensor detection, correlations, trends). Test data appears only
    in descriptive comparisons.
    """
    name = subset.name
    tag = name.lower()
    fig_dir = (figures_dir or cfg.paths.figures_dir) / tag
    res_dir = (results_dir or cfg.paths.results_dir) / "eda"
    res_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(cfg.seed)
    ecfg = cfg.eda

    train, test = subset.train, subset.test
    train_r = add_train_rul(train, cap=cfg.rul_cap)

    # ---- sensor classification (training data only) ------------------------------
    sensor_cols = list(SENSOR_COLUMNS)
    op_cols = list(OP_SETTING_COLUMNS)
    summary_all = st.sensor_summary(train, op_cols + sensor_cols)
    constant = list(summary_all.index[summary_all["is_constant"]])
    active_sensors = [s for s in sensor_cols if s not in constant]
    logger.info("%s: %d constant columns %s; %d active sensors", name, len(constant), constant, len(active_sensors))

    # ---- engine-level facts ---------------------------------------------------------
    lifetimes = st.engine_lifetimes(train)
    engines = truncated_engine_table(test, subset.rul)
    frac_at_cap = float((train_r["rul"] >= cfg.rul_cap).mean())

    trend = st.within_engine_trend(train, active_sensors)
    rul_corr = st.rul_correlation(train_r, active_sensors)
    spikes = st.spike_fraction(train, active_sensors, ecfg.spike_window, ecfg.spike_z_threshold)
    spread = st.initial_unit_spread(train, active_sensors, ecfg.initial_window)
    outside = st.share_outside_train_range(train, test, active_sensors)
    corr = train[active_sensors].corr(method="spearman")
    pairs = st.high_correlation_pairs(corr, ecfg.high_correlation_threshold)

    sensor_table = (
        summary_all.loc[active_sensors, ["mean", "std", "min", "max", "nunique", "cv"]]
        .join(trend[["n_engines", "mean_rho", "std_rho", "sign_consistency"]])
        .join(rul_corr)
        .join(spread[["offset_ratio"]])
        .join(spikes[["spike_fraction", "mad_zero"]])
        .join(outside)
    )
    sensor_table.index.name = "sensor"

    # ---- tables ---------------------------------------------------------------------
    sensor_table.to_csv(res_dir / f"{tag}_sensor_table.csv")
    engines.to_csv(res_dir / f"{tag}_test_engines.csv")
    summary_all.to_csv(res_dir / f"{tag}_column_summary.csv")

    # ---- figures --------------------------------------------------------------------
    units_sorted = lifetimes.sort_values()
    cap_units = [int(units_sorted.index[0]), int(units_sorted.index[len(units_sorted) // 2]), int(units_sorted.index[-1])]
    example_units = sorted(int(u) for u in rng.choice(lifetimes.index.to_numpy(), size=min(ecfg.n_example_engines, len(lifetimes)), replace=False))
    figs = {
        "lifetime": plots.plot_lifetime_distribution(lifetimes, engines["implied_lifetime"], fig_dir / "01_lifetime_distribution.png"),
        "rul_cap": plots.plot_rul_cap(train_r, cap_units, cfg.rul_cap, frac_at_cap, fig_dir / "02_rul_target_and_cap.png"),
        "test_truncation": plots.plot_test_truncation(engines, fig_dir / "03_test_truncation.png"),
        "trajectories": plots.plot_sensor_trajectories(train_r, active_sensors, example_units, fig_dir / "04_sensor_trajectories.png"),
        "relevance": plots.plot_sensor_relevance(trend, rul_corr, fig_dir / "05_sensor_trend_and_rul_correlation.png"),
        "initial_spread": plots.plot_initial_spread(spread, fig_dir / "06_initial_unit_spread.png"),
        "distributions": plots.plot_sensor_distributions(train, test, active_sensors, fig_dir / "07_sensor_distributions.png"),
        "operating": plots.plot_operating_settings(train, fig_dir / "08_operating_settings.png"),
        "correlation": plots.plot_correlation_matrix(corr, fig_dir / "09_sensor_correlation.png"),
    }

    # ---- summary --------------------------------------------------------------------
    obs = engines["observed_cycles"]
    summary: dict[str, Any] = {
        "subset": name,
        "rul_cap": cfg.rul_cap,
        "shapes": {"train": list(train.shape), "test": list(test.shape), "n_train_engines": int(lifetimes.size), "n_test_engines": int(len(engines))},
        "train_lifetime": st.describe_series(lifetimes),
        "test_observed_cycles": st.describe_series(obs),
        "test_true_rul": st.describe_series(engines["true_rul"]),
        "test_implied_lifetime": st.describe_series(engines["implied_lifetime"]),
        "test_engines_with_fewer_cycles_than": {str(w): int((obs < w).sum()) for w in (10, 20, 30, 40, 50)},
        "fraction_train_rows_at_cap": frac_at_cap,
        "constant_columns": constant,
        "active_sensors": active_sensors,
        "spike_threshold": {"window": ecfg.spike_window, "z": ecfg.spike_z_threshold},
        "spike_total_fraction": float(spikes["spike_count"].sum() / (len(train) * len(active_sensors))),
        "sensors_with_zero_mad": list(spikes.index[spikes["mad_zero"]]),
        "high_correlation_threshold": ecfg.high_correlation_threshold,
        "high_correlation_pairs": [[a, b, r] for a, b, r in pairs],
        "sensors_test_outside_train_range": {s: float(v) for s, v in outside.items() if v > 0},
        "op_settings": summary_all.loc[op_cols, ["mean", "std", "min", "max", "nunique"]].to_dict("index"),
        "figures": {k: str(Path(p).relative_to(cfg.paths.root)) if Path(p).is_relative_to(cfg.paths.root) else str(p) for k, p in figs.items()},
    }
    (res_dir / f"{tag}_eda_summary.json").write_text(json.dumps(_clean(summary), indent=2), encoding="utf-8")

    _write_report(res_dir / f"{name}_eda_report.md", summary, sensor_table, engines)
    logger.info("%s EDA complete: %d figures, tables and report in %s", name, len(figs), res_dir)
    return summary


def _write_report(path: Path, s: dict[str, Any], sensor_table: pd.DataFrame, engines: pd.DataFrame) -> None:
    """Write a markdown report containing only numbers computed in this run."""
    lt, ob, tr, il = s["train_lifetime"], s["test_observed_cycles"], s["test_true_rul"], s["test_implied_lifetime"]

    def row(label: str, d: dict[str, float]) -> str:
        return f"| {label} | {d['count']} | {d['mean']:.1f} | {d['std']:.1f} | {d['min']:.0f} | {d['median']:.1f} | {d['max']:.0f} |"

    lines = [
        f"# {s['subset']} exploratory data analysis",
        "",
        "Generated by `scripts/run_eda.py`. Every number below was computed from the data in that run.",
        "",
        "## Size",
        f"- Training: {s['shapes']['train'][0]} rows, {s['shapes']['n_train_engines']} engines",
        f"- Test: {s['shapes']['test'][0]} rows, {s['shapes']['n_test_engines']} engines",
        "",
        "## Engine lifetimes and truncation",
        "| quantity | n | mean | std | min | median | max |",
        "|---|---|---|---|---|---|---|",
        row("Training lifetime (cycles)", lt),
        row("Test observed cycles", ob),
        row("Test true RUL at last observed cycle", tr),
        row("Test implied lifetime (observed + RUL)", il),
        "",
        "Test engines with fewer observed cycles than a given window length:",
        "",
        "| window | engines shorter | share |",
        "|---|---|---|",
        *[f"| {w} | {n} | {n / s['shapes']['n_test_engines']:.1%} |" for w, n in s["test_engines_with_fewer_cycles_than"].items()],
        "",
        f"## RUL target (cap = {s['rul_cap']})",
        f"- Share of training rows whose raw RUL is at or above the cap: {s['fraction_train_rows_at_cap']:.1%}",
        "",
        "## Constant and active columns (training data)",
        f"- Constant columns: {', '.join(s['constant_columns']) or 'none'}",
        f"- Active sensors ({len(s['active_sensors'])}): {', '.join(s['active_sensors'])}",
        "",
        "## Per-sensor table (active sensors)",
        "`mean_rho`/`std_rho`: within-engine Spearman correlation with the cycle index (mean / std across the `n_engines` engines in which the sensor is not constant). "
        "`spearman_vs_rul`: pooled Spearman with the clipped RUL. `offset_ratio`: early-life between-engine spread divided by the "
        "noise-only expectation. `spike_fraction`: share of readings with a detrended robust |z| above the threshold "
        f"({s['spike_threshold']['z']}, rolling-median window {s['spike_threshold']['window']}). "
        "`share_outside_train_range`: share of test readings outside the training min..max.",
        "",
        _md_table(floatfmt="{:.4f}", df=sensor_table[["n_engines", "mean_rho", "std_rho", "sign_consistency", "spearman_vs_rul", "offset_ratio", "spike_fraction", "cv", "share_outside_train_range"]]),
        "",
        f"Sensors with zero residual MAD (spike rate undefined): {', '.join(s['sensors_with_zero_mad']) or 'none'}",
        f"Overall share of readings flagged as spikes: {s['spike_total_fraction']:.4%}",
        "",
        f"## Redundancy: sensor pairs with |Spearman| > {s['high_correlation_threshold']}",
        f"{len(s['high_correlation_pairs'])} pairs.",
        "",
        *[f"- {a} / {b}: {r:+.3f}" for a, b, r in s["high_correlation_pairs"][:15]],
        "",
        "## Figures",
        *[f"- `{p}`" for p in s["figures"].values()],
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")
