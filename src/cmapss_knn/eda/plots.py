"""Figures for the EDA. Uses matplotlib's object-oriented ``Figure`` API so that
importing this module never changes the user's pyplot backend."""

from __future__ import annotations

import functools
import math
from pathlib import Path
from typing import Callable, TypeVar

import matplotlib as mpl
import numpy as np
import pandas as pd
from matplotlib.figure import Figure

F = TypeVar("F", bound=Callable[..., Path])

_STYLE = {
    "axes.grid": True,
    "grid.alpha": 0.25,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "font.size": 10,
    "axes.titlesize": 11,
    "axes.titleweight": "bold",
}
TRAIN_COLOR = "#1f77b4"
TEST_COLOR = "#ff7f0e"


def _styled(func: F) -> F:
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        with mpl.rc_context(_STYLE):
            return func(*args, **kwargs)

    return wrapper  # type: ignore[return-value]


def _save(fig: Figure, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    return path


@_styled
def plot_lifetime_distribution(train_lifetimes: pd.Series, test_implied: pd.Series, path: Path) -> Path:
    fig = Figure(figsize=(8, 4.5), layout="constrained")
    ax = fig.subplots()
    lo = min(train_lifetimes.min(), test_implied.min())
    hi = max(train_lifetimes.max(), test_implied.max())
    bins = np.linspace(lo, hi, 25)
    ax.hist(train_lifetimes, bins=bins, color=TRAIN_COLOR, alpha=0.6, label=f"Training engines (n={len(train_lifetimes)})")
    ax.hist(test_implied, bins=bins, histtype="step", color=TEST_COLOR, lw=2, label=f"Test engines, implied lifetime (n={len(test_implied)})")
    ax.axvline(train_lifetimes.median(), color=TRAIN_COLOR, ls="--", lw=1)
    ax.axvline(test_implied.median(), color=TEST_COLOR, ls="--", lw=1)
    ax.set_xlabel("Engine lifetime (cycles)")
    ax.set_ylabel("Number of engines")
    ax.set_title("Engine lifetimes: training vs. test (observed cycles + true RUL)")
    ax.legend()
    return _save(fig, path)


@_styled
def plot_rul_cap(train_with_rul: pd.DataFrame, units: list[int], cap: int, fraction_at_cap: float, path: Path) -> Path:
    fig = Figure(figsize=(8, 4.5), layout="constrained")
    ax = fig.subplots()
    colors = ["#1f77b4", "#2ca02c", "#d62728", "#9467bd"]
    for color, unit in zip(colors, units):
        g = train_with_rul[train_with_rul["unit"] == unit]
        ax.plot(g["cycle"], g["rul"], ls="--", color=color, alpha=0.5)
        ax.plot(g["cycle"], g["rul_clipped"], color=color, lw=2, label=f"Engine {unit} ({len(g)} cycles)")
    ax.axhline(cap, color="gray", ls=":", lw=1.5)
    ax.text(ax.get_xlim()[1], cap, f" cap = {cap}", va="bottom", ha="right", color="gray")
    ax.set_xlabel("Cycle")
    ax.set_ylabel("Remaining useful life (cycles)")
    ax.set_title(f"Raw (dashed) vs. clipped (solid) RUL target; {fraction_at_cap:.1%} of training rows sit at the cap")
    ax.legend()
    return _save(fig, path)


@_styled
def plot_test_truncation(table: pd.DataFrame, path: Path) -> Path:
    fig = Figure(figsize=(10, 4), layout="constrained")
    a1, a2 = fig.subplots(1, 2)
    a1.hist(table["observed_cycles"], bins=20, color=TEST_COLOR, alpha=0.8)
    a1.set_xlabel("Observed cycles before truncation")
    a1.set_ylabel("Number of test engines")
    a1.set_title("How much history each test engine has")
    a2.hist(table["true_rul"], bins=20, color=TEST_COLOR, alpha=0.8)
    a2.set_xlabel("True RUL at last observed cycle")
    a2.set_title("What the model must predict (test)")
    return _save(fig, path)


@_styled
def plot_sensor_trajectories(train_with_rul: pd.DataFrame, sensors: list[str], units: list[int], path: Path, ncols: int = 3) -> Path:
    nrows = math.ceil(len(sensors) / ncols)
    fig = Figure(figsize=(4.2 * ncols, 2.3 * nrows), layout="constrained")
    axes = fig.subplots(nrows, ncols, sharex=True, squeeze=False)
    groups = {u: g for u, g in train_with_rul.groupby("unit") if u in set(units)}
    for ax, sensor in zip(axes.flat, sensors):
        for g in groups.values():
            ax.plot(g["rul"], g[sensor], lw=0.7, alpha=0.8)
        ax.set_title(sensor)
    for ax in list(axes.flat)[len(sensors):]:
        ax.set_visible(False)
    axes.flat[0].invert_xaxis()  # time flows left to right: failure on the right
    for ax in axes[-1]:
        ax.set_xlabel("True RUL (cycles)")
    fig.suptitle(f"Sensor trajectories of {len(groups)} random training engines (x-axis reversed: failure at right)")
    return _save(fig, path)


@_styled
def plot_sensor_relevance(trend: pd.DataFrame, rul_corr: pd.Series, path: Path) -> Path:
    fig = Figure(figsize=(12, 4.2), layout="constrained")
    a1, a2 = fig.subplots(1, 2)
    x = np.arange(len(trend))
    colors = np.where(trend["mean_rho"] > 0, "#d62728", "#1f77b4")
    a1.bar(x, trend["mean_rho"], yerr=trend["std_rho"], color=colors, capsize=2, alpha=0.85)
    a1.set_xticks(x, trend.index, rotation=90)
    a1.set_ylim(-1.05, 1.05)
    a1.set_ylabel("Spearman rho with cycle (within engine)")
    a1.set_title("Monotone drift with age (mean +/- std over engines)")
    xr = np.arange(len(rul_corr))
    a2.bar(xr, rul_corr.to_numpy(), color=np.where(rul_corr > 0, "#d62728", "#1f77b4"), alpha=0.85)
    a2.set_xticks(xr, rul_corr.index, rotation=90)
    a2.set_ylim(-1.05, 1.05)
    a2.set_ylabel("Spearman rho with clipped RUL (pooled)")
    a2.set_title("Association with the clipped RUL target")
    return _save(fig, path)


@_styled
def plot_initial_spread(spread: pd.DataFrame, path: Path) -> Path:
    fig = Figure(figsize=(8, 4), layout="constrained")
    ax = fig.subplots()
    x = np.arange(len(spread))
    ax.bar(x, spread["offset_ratio"], color="#7f7f7f", alpha=0.85)
    ax.axhline(1.0, color="black", ls="--", lw=1)
    ax.set_xticks(x, spread.index, rotation=90)
    ax.set_ylabel("Between-engine std / noise-only expectation")
    ax.set_title("Early-life unit-to-unit offsets (1 = explained by sensor noise alone)")
    return _save(fig, path)


@_styled
def plot_sensor_distributions(train: pd.DataFrame, test: pd.DataFrame, sensors: list[str], path: Path, ncols: int = 3) -> Path:
    nrows = math.ceil(len(sensors) / ncols)
    fig = Figure(figsize=(4.2 * ncols, 2.3 * nrows), layout="constrained")
    axes = fig.subplots(nrows, ncols, squeeze=False)
    for ax, s in zip(axes.flat, sensors):
        lo = min(train[s].min(), test[s].min())
        hi = max(train[s].max(), test[s].max())
        bins = np.linspace(lo, hi, 40) if hi > lo else 10
        ax.hist(train[s], bins=bins, density=True, color=TRAIN_COLOR, alpha=0.5, label="train")
        ax.hist(test[s], bins=bins, density=True, histtype="step", color=TEST_COLOR, lw=1.5, label="test")
        ax.set_title(s)
        ax.set_yticks([])
    for ax in list(axes.flat)[len(sensors):]:
        ax.set_visible(False)
    axes.flat[0].legend()
    fig.suptitle("Sensor value distributions (all cycles), training vs. test")
    return _save(fig, path)


@_styled
def plot_operating_settings(train: pd.DataFrame, path: Path) -> Path:
    fig = Figure(figsize=(12, 3.8), layout="constrained")
    a1, a2, a3 = fig.subplots(1, 3)
    a1.hist(train["op1"], bins=40, color=TRAIN_COLOR, alpha=0.8)
    a1.set_xlabel("op1")
    a1.set_title("Operational setting 1")
    a2.hist(train["op2"], bins=40, color=TRAIN_COLOR, alpha=0.8)
    a2.set_xlabel("op2")
    a2.set_title("Operational setting 2")
    a3.scatter(train["op1"], train["op2"], s=2, alpha=0.2, color=TRAIN_COLOR)
    a3.set_xlabel("op1")
    a3.set_ylabel("op2")
    a3.set_title("op1 vs. op2")
    op3_values = sorted(train["op3"].unique())
    fig.suptitle(f"Operating settings in training data (op3 takes {len(op3_values)} distinct value(s): {op3_values[:3]})")
    return _save(fig, path)


@_styled
def plot_correlation_matrix(corr: pd.DataFrame, path: Path) -> Path:
    fig = Figure(figsize=(8.5, 7.5), layout="constrained")
    ax = fig.subplots()
    ax.grid(False)
    im = ax.imshow(corr.to_numpy(), cmap="RdBu_r", vmin=-1, vmax=1)
    n = len(corr)
    ax.set_xticks(range(n), corr.columns, rotation=90)
    ax.set_yticks(range(n), corr.index)
    for i in range(n):
        for j in range(n):
            ax.text(j, i, f"{corr.iloc[i, j]:.2f}", ha="center", va="center", fontsize=6)
    fig.colorbar(im, ax=ax, shrink=0.8, label="Spearman correlation")
    ax.set_title("Sensor-to-sensor correlation (training data, all cycles)")
    return _save(fig, path)
