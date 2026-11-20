"""Figures for the official test evaluation. Every figure is drawn from saved CSVs (per-engine results and the CV reference table)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from matplotlib.figure import Figure

from ..eda.plots import _save, _styled

BLUE, RED, GREY = "#1f77b4", "#d62728", "#7f7f7f"


def _panels(df: pd.DataFrame):
    return list(df.groupby("subset", sort=True))


@_styled
def plot_pred_vs_true(df: pd.DataFrame, cap: int, path: Path) -> Path:
    fig = Figure(figsize=(11, 9), layout="constrained")
    axes = fig.subplots(2, 2)
    for ax, (name, g) in zip(axes.flat, _panels(df)):
        late = g["error"] > 0
        ax.scatter(g.loc[~late, "true_rul"], g.loc[~late, "predicted_rul"], s=16, alpha=0.7, color=BLUE, label="early / exact (pred <= true)")
        ax.scatter(g.loc[late, "true_rul"], g.loc[late, "predicted_rul"], s=16, alpha=0.7, color=RED, label="late (pred > true)")
        top = max(g["true_rul"].max(), g["predicted_rul"].max())
        ax.plot([0, top], [0, top], color=GREY, ls="--", lw=1)
        ax.axhline(cap, color=GREY, ls=":", lw=1)
        ax.set_title(f"{name}: official test, {len(g)} engines")
        ax.set_xlabel("True RUL after the last observed cycle"); ax.set_ylabel("Predicted RUL")
    axes.flat[0].legend(fontsize=8)
    return _save(fig, path)


@_styled
def plot_error_hist(df: pd.DataFrame, path: Path) -> Path:
    fig = Figure(figsize=(11, 8), layout="constrained")
    axes = fig.subplots(2, 2)
    for ax, (name, g) in zip(axes.flat, _panels(df)):
        ax.hist(g["error"], bins=30, color=BLUE, alpha=0.8)
        ax.axvline(0, color="black", lw=1)
        ax.axvline(g["error"].mean(), color=RED, ls="--", lw=1.5, label=f"mean {g['error'].mean():+.1f}")
        ax.set_title(f"{name}: prediction error (predicted - true)")
        ax.set_xlabel("Error in cycles (positive = late warning)"); ax.legend()
    return _save(fig, path)


@_styled
def plot_cv_vs_test(ref: pd.DataFrame, path: Path) -> Path:
    fig = Figure(figsize=(9.5, 4.8), layout="constrained")
    ax = fig.subplots()
    x = np.arange(len(ref))
    ax.bar(x - 0.27, ref["cv_rmse_clipped_truth"], 0.27, color=GREY, label="CV, vs clipped RUL (selection metric)")
    ax.bar(x, ref["cv_rmse_raw_truth"], 0.27, color="#aec7e8", label="CV, vs true RUL")
    ax.bar(x + 0.27, ref["official_test_rmse"], 0.27, color=BLUE, label="OFFICIAL TEST, vs true RUL")
    ax.set_xticks(x, ref["subset"]); ax.set_ylabel("RMSE (cycles)")
    ax.set_title("Cross-validation (training engines, random cutoffs) vs official test (last observed cycle)")
    ax.legend(fontsize=8)
    return _save(fig, path)


@_styled
def plot_error_vs_history(df: pd.DataFrame, path: Path) -> Path:
    fig = Figure(figsize=(11, 8), layout="constrained")
    axes = fig.subplots(2, 2)
    for ax, (name, g) in zip(axes.flat, _panels(df)):
        ax.scatter(g["observed_cycles"], g["error"], s=16, alpha=0.7, color=np.where(g["short_history"], RED, BLUE))
        ax.axhline(0, color="black", lw=1)
        ax.axvline(g["window"].iloc[0], color=GREY, ls="--", lw=1)
        ax.set_title(f"{name}: error vs observed history (red = shorter than the {g['window'].iloc[0]}-cycle window)")
        ax.set_xlabel("Observed cycles"); ax.set_ylabel("Error (predicted - true)")
    return _save(fig, path)


@_styled
def plot_nasa_terms(df: pd.DataFrame, path: Path) -> Path:
    fig = Figure(figsize=(11, 8), layout="constrained")
    axes = fig.subplots(2, 2)
    for ax, (name, g) in zip(axes.flat, _panels(df)):
        t = np.sort(g["nasa_term"].to_numpy())[::-1]
        ax.bar(np.arange(len(t)), t, color=RED, width=1.0)
        ax.set_yscale("log")
        ax.set_title(f"{name}: per-engine NASA penalty, sorted (top 5 hold {t[:5].sum() / t.sum():.0%} of the sum)")
        ax.set_xlabel("Engine rank"); ax.set_ylabel("Penalty (log scale)")
    return _save(fig, path)
