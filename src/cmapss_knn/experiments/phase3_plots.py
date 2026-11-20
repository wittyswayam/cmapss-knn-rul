"""Figures for Phase 3."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from matplotlib.figure import Figure

from ..eda.plots import _save, _styled

C = {"none": "#1f77b4", "causal": "#ff7f0e"}


@_styled
def plot_normalisation_scatter(effects: dict[str, pd.DataFrame], metric: str, path: Path) -> Path:
    fig = Figure(figsize=(5.8 * len(effects), 5.2), layout="constrained")
    axes = fig.subplots(1, len(effects), squeeze=False)[0]
    for ax, (name, e) in zip(axes, effects.items()):
        x, y = e[f"{metric}_base"], e[f"{metric}_other"]
        lo, hi = min(x.min(), y.min()) * 0.97, max(x.max(), y.max()) * 1.03
        ax.plot([lo, hi], [lo, hi], color="gray", ls="--", lw=1)
        for b, g in e.groupby("baseline"):
            ax.scatter(g[f"{metric}_base"], g[f"{metric}_other"], s=12, alpha=0.6, color=C[b], label=f"baseline = {b}")
        ax.set_xlim(lo, hi); ax.set_ylim(lo, hi)
        ax.set_xlabel("Engine-level RMSE, raw sensors (global scaling)")
        ax.set_ylabel("Engine-level RMSE, regime-wise normalisation")
        ax.set_title(f"{name}: each dot is one hyper-parameter setting")
        ax.legend()
    return _save(fig, path)


@_styled
def plot_regime_effect(raw: pd.DataFrame, normed: pd.DataFrame, sensor: str, units: list[int], path: Path) -> Path:
    fig = Figure(figsize=(11.5, 4.2), layout="constrained")
    a1, a2 = fig.subplots(1, 2, sharex=True)
    for u in units:
        g, h = raw[raw.unit == u], normed[normed.unit == u]
        a1.plot(g.cycle, g[sensor], lw=0.6, alpha=0.8)
        a2.plot(h.cycle, h[sensor], lw=0.6, alpha=0.8)
    a1.set_title(f"{sensor}, raw values"); a2.set_title(f"{sensor}, regime-normalised")
    for ax in (a1, a2):
        ax.set_xlabel("Cycle")
    a1.set_ylabel("Sensor value"); a2.set_ylabel("z-score within regime")
    return _save(fig, path)


@_styled
def plot_selection_scatter(data: dict[str, tuple[pd.DataFrame, pd.Series, pd.Series]], path: Path) -> Path:
    fig = Figure(figsize=(5 * len(data), 4.6), layout="constrained")
    axes = fig.subplots(1, len(data), squeeze=False)[0]
    for ax, (name, (s, by_r, by_n)) in zip(axes, data.items()):
        ax.scatter(s["eng_rmse_clip_mean"], s["eng_nasa_clip_mean"], s=8, alpha=0.35, color="#7f7f7f")
        ax.scatter([by_r["eng_rmse_clip_mean"]], [by_r["eng_nasa_clip_mean"]], s=70, color="#1f77b4", label="selected by RMSE", zorder=3)
        ax.scatter([by_n["eng_rmse_clip_mean"]], [by_n["eng_nasa_clip_mean"]], s=70, marker="s", color="#d62728", label="selected by NASA", zorder=3)
        ax.set_yscale("log")
        ax.set_xlabel("CV engine-level RMSE"); ax.set_ylabel("CV mean NASA score (log)")
        ax.set_title(name); ax.legend()
    return _save(fig, path)


@_styled
def plot_cross_subset(table: pd.DataFrame, path: Path) -> Path:
    fig = Figure(figsize=(12, 4.2), layout="constrained")
    axes = fig.subplots(1, 3)
    x = np.arange(len(table))
    for ax, (col, title) in zip(axes, (("rmse", "RMSE"), ("mae", "MAE"), ("nasa", "mean NASA score"))):
        ax.bar(x - 0.2, table[f"{col}_rmse_sel"], 0.4, label="selected by RMSE", color="#1f77b4")
        ax.bar(x + 0.2, table[f"{col}_nasa_sel"], 0.4, label="selected by NASA", color="#d62728")
        ax.scatter(x, table[f"{col}_const"], marker="_", s=400, color="black", label="constant predictor", zorder=3)
        ax.set_xticks(x, table["subset"]); ax.set_title(f"CV {title} (engine level, vs clipped RUL)")
        ax.set_yscale("log") if col == "nasa" else None
    axes[0].legend()
    return _save(fig, path)


@_styled
def plot_probe(metrics: dict[str, pd.DataFrame], stages: list[str], path: Path) -> Path:
    fig = Figure(figsize=(5.2 * len(stages), 4.2 * len(metrics)), layout="constrained")
    axes = fig.subplots(len(metrics), len(stages), squeeze=False)
    colors = {"adaptive": "#2ca02c", "pad_first": "#d62728", "pad_mean": "#9467bd"}
    for r, (name, m) in enumerate(metrics.items()):
        for c, stage in enumerate(stages):
            ax = axes[r][c]
            for (hist, base), g in m[m["stage"] == stage].groupby(["history", "baseline"]):
                g = g.sort_values("n_obs")
                ax.plot(g["n_obs"], g["rmse"], marker="o", ms=3, color=colors[hist], ls="-" if base == "none" else "--", label=f"{hist}, baseline {base}")
            ax.set_title(f"{name}: {stage.replace('_', ' ')}")
            ax.set_xlabel("Observed cycles"); ax.set_ylabel("RMSE vs clipped RUL")
    axes[0][0].legend(fontsize=7)
    return _save(fig, path)


@_styled
def plot_extension_curves(data: dict[str, pd.DataFrame], old_windows: tuple[int, ...], old_max_k: int, path: Path) -> Path:
    fig = Figure(figsize=(11.5, 4.4 * len(data)), layout="constrained")
    axes = fig.subplots(len(data), 2, squeeze=False)
    for r, (name, s) in enumerate(data.items()):
        w = s.groupby("window")["eng_rmse_clip_mean"].min()
        k = s.groupby("k")["eng_rmse_clip_mean"].min()
        a1, a2 = axes[r]
        a1.plot(w.index, w.values, color="#555555", lw=1)
        a1.scatter(w.index, w.values, c=["#1f77b4" if i in old_windows else "#d62728" for i in w.index], s=45, zorder=3)
        a2.plot(k.index, k.values, color="#555555", lw=1)
        a2.scatter(k.index, k.values, c=["#1f77b4" if i <= old_max_k else "#d62728" for i in k.index], s=45, zorder=3)
        a2.set_xscale("log")
        a1.set_title(f"{name}: best CV RMSE by window length"); a2.set_title(f"{name}: best CV RMSE by number of neighbours")
        a1.set_xlabel("Window length (cycles)"); a2.set_xlabel("k (log scale)")
        a1.set_ylabel("Engine-level CV RMSE (min over other settings)")
    axes[0][0].scatter([], [], c="#1f77b4", label="original grid"); axes[0][0].scatter([], [], c="#d62728", label="added values"); axes[0][0].legend()
    return _save(fig, path)
