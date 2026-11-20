"""Figures for the Phase 2 experiments (matplotlib object-oriented API)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from matplotlib.figure import Figure

from ..eda.plots import _save, _styled

COLORS = {"euclidean/uniform": "#1f77b4", "euclidean/distance": "#2ca02c", "manhattan/uniform": "#ff7f0e", "manhattan/distance": "#d62728"}


@_styled
def plot_k_curves(summary: pd.DataFrame, best: pd.Series, path: Path) -> Path:
    keys = ["sensor_config", "baseline", "window", "history"]
    sel = summary.copy()
    for k in keys:
        sel = sel[sel[k].astype(str) == str(best[k])]
    fig = Figure(figsize=(8.5, 4.8), layout="constrained")
    ax = fig.subplots()
    for (metric, weights), g in sel.groupby(["metric", "weights"]):
        g = g.sort_values("k")
        label = f"{metric}/{weights}"
        ax.errorbar(g["k"], g["eng_rmse_clip_mean"], yerr=g["eng_rmse_clip_std"], marker="o", ms=4, capsize=2, label=label, color=COLORS[label], alpha=0.9)
    ax.set_xscale("log")
    ax.set_xlabel("Number of neighbours k (log scale)")
    ax.set_ylabel("Engine-level RMSE vs clipped RUL (cycles)")
    ax.set_title(f"Effect of k, metric and weighting (sensors {best['sensor_config']}, baseline {best['baseline']}, window {int(best['window'])}, {best['history']})")
    ax.legend(title="metric / weights")
    return _save(fig, path)


@_styled
def plot_paired_scatter(effect: pd.DataFrame, metric: str, path: Path, title: str, xlabel: str, ylabel: str, color_by: str = "baseline") -> Path:
    fig = Figure(figsize=(5.6, 5.4), layout="constrained")
    ax = fig.subplots()
    x, y = effect[f"{metric}_base"], effect[f"{metric}_other"]
    lo, hi = min(x.min(), y.min()) * 0.98, max(x.max(), y.max()) * 1.02
    ax.plot([lo, hi], [lo, hi], color="gray", lw=1, ls="--")
    for i, (val, g) in enumerate(effect.groupby(color_by)):
        ax.scatter(g[f"{metric}_base"], g[f"{metric}_other"], s=14, alpha=0.7, label=f"{color_by} = {val}", color=["#1f77b4", "#ff7f0e", "#2ca02c"][i % 3])
    ax.set_xlim(lo, hi)
    ax.set_ylim(lo, hi)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend()
    return _save(fig, path)


@_styled
def plot_baseline_and_history(summary: pd.DataFrame, path: Path) -> Path:
    fig = Figure(figsize=(11.5, 4.4), layout="constrained")
    a1, a2 = fig.subplots(1, 2)
    windows = sorted(summary["window"].unique())
    x = np.arange(len(windows))
    for j, (b, c) in enumerate((("none", "#7f7f7f"), ("causal", "#1f77b4"))):
        vals = [summary[(summary["window"] == w) & (summary["baseline"] == b)]["eng_rmse_clip_mean"].min() for w in windows]
        a1.bar(x + (j - 0.5) * 0.38, vals, 0.38, label=f"baseline = {b}", color=c)
    a1.set_xticks(x, [f"W = {w}" for w in windows])
    a1.set_ylabel("Best engine-level RMSE over the rest of the grid")
    a1.set_title("Raw sensors vs causal per-engine baseline")
    a1.legend()
    for j, (h, c) in enumerate((("adaptive", "#2ca02c"), ("pad_first", "#d62728"))):
        vals = [summary[(summary["window"] == w) & (summary["history"] == h)]["stress_rmse_clip_mean"].mean() for w in windows]
        a2.bar(x + (j - 0.5) * 0.38, vals, 0.38, label=h, color=c)
    a2.set_xticks(x, [f"W = {w}" for w in windows])
    a2.set_ylabel("Mean short-history RMSE (cutoffs 5, 10, 20 cycles)")
    a2.set_title("Short-history strategy, averaged over the grid")
    a2.legend()
    return _save(fig, path)


@_styled
def plot_holdout_scatter(true_clip: np.ndarray, true_raw: np.ndarray, pred: np.ndarray, cap: int, title: str, path: Path) -> Path:
    fig = Figure(figsize=(6.2, 5.6), layout="constrained")
    ax = fig.subplots()
    ax.scatter(true_raw, pred, s=14, alpha=0.6, color="#1f77b4")
    ax.plot([0, true_raw.max()], [0, true_raw.max()], color="gray", ls="--", lw=1, label="perfect prediction")
    ax.axhline(cap, color="#d62728", ls=":", lw=1, label=f"model ceiling (clip = {cap})")
    ax.set_xlabel("True RUL at the cutoff (cycles)")
    ax.set_ylabel("Predicted RUL (cycles)")
    ax.set_title(title)
    ax.legend()
    return _save(fig, path)
