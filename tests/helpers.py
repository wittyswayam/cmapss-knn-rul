"""Synthetic C-MAPSS-like data for tests (no real data needed)."""

from __future__ import annotations

import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

from cmapss_knn.config import SubsetSpec
from cmapss_knn.data.ingestion import CMAPSSSubset
from cmapss_knn.data.schema import ALL_COLUMNS, SENSOR_COLUMNS, rul_filename, test_filename, train_filename


def _engine(unit: int, n_cycles: int, rng: np.random.Generator) -> pd.DataFrame:
    cycles = np.arange(1, n_cycles + 1)
    data = {"unit": unit, "cycle": cycles, "op1": rng.normal(0, 0.002, n_cycles), "op2": rng.normal(0, 0.0003, n_cycles), "op3": 100.0}
    for i, s in enumerate(SENSOR_COLUMNS):
        if i % 3 == 0:
            data[s] = 5.0  # constant sensor
        else:
            data[s] = 100 + (i % 2 * 2 - 1) * cycles / n_cycles * 5 + rng.normal(0, 0.1, n_cycles)
    return pd.DataFrame(data)[list(ALL_COLUMNS)]


def make_subset(n_train: int = 6, n_test: int = 4, seed: int = 0) -> CMAPSSSubset:
    rng = np.random.default_rng(seed)
    train = pd.concat([_engine(u, int(rng.integers(40, 60)), rng) for u in range(1, n_train + 1)], ignore_index=True)
    full = [_engine(u, int(rng.integers(40, 60)), rng) for u in range(1, n_test + 1)]
    rul_vals, test_parts = [], []
    for df in full:
        keep = int(rng.integers(15, len(df) - 5))
        rul_vals.append(len(df) - keep)
        test_parts.append(df.iloc[:keep])
    test = pd.concat(test_parts, ignore_index=True)
    rul = pd.Series(rul_vals, index=pd.RangeIndex(1, n_test + 1, name="unit"), name="rul")
    return CMAPSSSubset("FD001", train, test, rul)


def spec_for(subset: CMAPSSSubset) -> SubsetSpec:
    return SubsetSpec(
        name=subset.name,
        n_train_units=subset.train["unit"].nunique(),
        n_test_units=subset.test["unit"].nunique(),
        n_train_rows=len(subset.train),
        n_test_rows=len(subset.test),
        n_operating_conditions=1,
        fault_modes=("HPC",),
    )


def _to_text(df: pd.DataFrame) -> str:
    lines = []
    for row in df.itertuples(index=False):
        cells = [str(int(row[0])), str(int(row[1]))] + [f"{v:.4f}" for v in row[2:]]
        lines.append(" ".join(cells) + " ")  # NASA files end each line with a space
    return "\n".join(lines) + "\n"


def write_raw_files(directory: Path, subset: CMAPSSSubset, as_zip: bool = False, archive_name: str = "CMAPSSData.zip") -> None:
    texts = {
        train_filename(subset.name): _to_text(subset.train),
        test_filename(subset.name): _to_text(subset.test),
        rul_filename(subset.name): "".join(f"{int(v)} \n" for v in subset.rul),
    }
    directory.mkdir(parents=True, exist_ok=True)
    if as_zip:
        with zipfile.ZipFile(directory / archive_name, "w") as zf:
            for name, text in texts.items():
                zf.writestr(name, text)
    else:
        for name, text in texts.items():
            (directory / name).write_text(text, encoding="utf-8")
