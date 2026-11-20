"""Resumable grid extension for FD002 and FD004 (window 100 and k = 200 added to the Phase 3 grid).

Everything computed in Phase 3 is kept and not recomputed. Two *parts* per CV fold are run:

* ``w100``: the new window length with every k (including 200), all other axes as in Phase 3.
* ``k200``: every existing window with k in {150, 200}. k = 200 is new. k = 150 repeats configurations already saved by Phase 3,
  and is compared with the saved values as a reproducibility check; those overlap rows are not used for selection.

Each completed chunk is written as ``<subset>_fold<f>_<part>.csv`` plus ``<...>.json``, a manifest holding the exact grid, seed, a hash of the
relevant configuration and the row count. The manifest is written last, so a chunk without one is treated as unfinished and recomputed.
A chunk is skipped on resume only if its manifest matches the current configuration; a mismatch is an error (never silently mixed).
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import time
from dataclasses import asdict, dataclass, replace
from pathlib import Path

import pandas as pd

from ..config import Config, Phase3Grid
from . import phase3 as p3

logger = logging.getLogger(__name__)
PARTS = ("w100", "k200")


@dataclass(frozen=True)
class ExtensionMismatch(Exception):
    message: str

    def __str__(self) -> str:
        return self.message


def part_grid(cfg: Config, part: str) -> Phase3Grid:
    g, e = cfg.phase3.grid, cfg.phase3_extension
    if part == "w100":
        return replace(g, window=tuple(e.add_window), k=tuple(sorted(set(g.k) | set(e.add_k))))
    if part == "k200":
        return replace(g, k=tuple(sorted({max(g.k)} | set(e.add_k))))
    raise ValueError(f"unknown part {part!r}; expected one of {PARTS}")


def config_hash(cfg: Config, part: str, normalizations: tuple[str, ...]) -> str:
    payload = {"seed": cfg.seed, "rul_cap": cfg.rul_cap, "features": asdict(cfg.features), "validation": asdict(cfg.validation), "stride": cfg.phase3.window_eval_stride,
               "holdout_fraction": cfg.phase3.holdout_fraction, "grid": asdict(part_grid(cfg, part)), "normalizations": list(normalizations), "part": part}
    return hashlib.sha256(json.dumps(payload, sort_keys=True, default=list).encode()).hexdigest()


def expected_rows(cfg: Config, part: str, normalizations: tuple[str, ...]) -> int:
    g = part_grid(cfg, part)
    return len(normalizations) * len(g.baseline) * len(g.window) * len(g.history) * len(g.metric) * len(g.weights) * len(g.k)


def chunk_files(out_dir: Path, subset: str, fold: int, part: str) -> tuple[Path, Path]:
    base = out_dir / f"{subset.lower()}_fold{fold}_{part}"
    return base.with_suffix(".csv"), base.with_suffix(".json")


def chunk_status(cfg: Config, out_dir: Path, subset: str, fold: int, part: str, normalizations: tuple[str, ...]) -> str:
    """``complete``, ``missing`` (or unfinished), or raises ExtensionMismatch if a manifest disagrees with the current configuration."""
    csv, man = chunk_files(out_dir, subset, fold, part)
    if not man.is_file() or not csv.is_file():
        return "missing"
    m = json.loads(man.read_text(encoding="utf-8"))
    if m.get("config_hash") != config_hash(cfg, part, normalizations) or m.get("rows") != expected_rows(cfg, part, normalizations):
        raise ExtensionMismatch(f"{man.name}: saved with a different configuration; refusing to mix results. Move the old chunk away to recompute it.")
    return "complete"


def run_chunk(setup: p3.SubsetSetup, cfg: Config, out_dir: Path, fold: int, part: str, normalizations: tuple[str, ...]) -> str:
    status = chunk_status(cfg, out_dir, setup.name, fold, part, normalizations)
    if status == "complete":
        return "skipped"
    out_dir.mkdir(parents=True, exist_ok=True)
    csv, man = chunk_files(out_dir, setup.name, fold, part)
    t0 = time.time()
    df = p3.run_fold(setup, cfg, fold, normalizations, grid=part_grid(cfg, part), include_reference=False)
    if len(df) != expected_rows(cfg, part, normalizations):
        raise RuntimeError(f"{csv.name}: produced {len(df)} rows, expected {expected_rows(cfg, part, normalizations)}")
    tmp = csv.with_suffix(".csv.tmp")
    df.to_csv(tmp, index=False)
    os.replace(tmp, csv)  # the CSV is complete before it gets its final name
    man.write_text(json.dumps({"subset": setup.name, "fold": fold, "part": part, "rows": len(df), "seconds": round(time.time() - t0, 1), "seed": cfg.seed,
                               "normalizations": list(normalizations), "grid": asdict(part_grid(cfg, part)), "config_hash": config_hash(cfg, part, normalizations),
                               "completed_utc": pd.Timestamp.now("UTC").isoformat(timespec="seconds")}, indent=2), encoding="utf-8")
    logger.info("%s fold %d part %s done in %.0fs (%d rows)", setup.name, fold, part, time.time() - t0, len(df))
    return "ran"
