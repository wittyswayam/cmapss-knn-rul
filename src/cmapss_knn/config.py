"""Typed access to ``configs/default.yaml``."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import yaml

from .exceptions import ConfigError

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG_PATH = REPO_ROOT / "configs" / "default.yaml"


@dataclass(frozen=True)
class SubsetSpec:
    """Expected structure of one C-MAPSS subset (FD001 ... FD004)."""

    name: str
    n_train_units: int
    n_test_units: int
    n_train_rows: int
    n_test_rows: int
    n_operating_conditions: int
    fault_modes: tuple[str, ...]


@dataclass(frozen=True)
class PathsConfig:
    root: Path
    raw_data_dir: Path
    figures_dir: Path
    results_dir: Path
    models_dir: Path
    logs_dir: Path


@dataclass(frozen=True)
class EDAConfig:
    n_example_engines: int
    spike_window: int
    spike_z_threshold: float
    initial_window: int
    high_correlation_threshold: float


@dataclass(frozen=True)
class FeaturesConfig:
    stats: tuple[str, ...]
    baseline_cycles: int


@dataclass(frozen=True)
class ValidationConfig:
    n_folds: int
    n_holdout_engines: int
    eval_cutoffs_per_engine: int
    eval_min_cutoff: int
    stress_cutoffs: tuple[int, ...]


@dataclass(frozen=True)
class GridConfig:
    sensor_configs: tuple[str, ...]
    baseline: tuple[str, ...]
    window: tuple[int, ...]
    history: tuple[str, ...]
    metric: tuple[str, ...]
    weights: tuple[str, ...]
    k: tuple[int, ...]


@dataclass(frozen=True)
class Phase3Grid:
    baseline: tuple[str, ...]
    normalization: tuple[str, ...]
    window: tuple[int, ...]
    history: tuple[str, ...]
    metric: tuple[str, ...]
    weights: tuple[str, ...]
    k: tuple[int, ...]


@dataclass(frozen=True)
class ProbeConfig:
    stages: Mapping[str, float]
    n_obs: tuple[int, ...]
    window: tuple[int, ...]
    history: tuple[str, ...]
    baseline: tuple[str, ...]
    metric: str
    weights: str
    k: int


@dataclass(frozen=True)
class Phase3Config:
    holdout_fraction: float
    window_eval_stride: int
    grid: Phase3Grid
    probe: ProbeConfig


@dataclass(frozen=True)
class ExtensionConfig:
    subsets: tuple[str, ...]
    add_window: tuple[int, ...]
    add_k: tuple[int, ...]


@dataclass(frozen=True)
class Config:
    seed: int
    log_level: str
    paths: PathsConfig
    archive_name: str
    archive_sha256: str
    subsets: Mapping[str, SubsetSpec]
    rul_cap: int
    eda: EDAConfig
    features: FeaturesConfig
    validation: ValidationConfig
    grid: GridConfig
    phase3: Phase3Config
    phase3_extension: ExtensionConfig


def _require(mapping: Mapping[str, Any], key: str, where: str) -> Any:
    if not isinstance(mapping, Mapping) or key not in mapping:
        raise ConfigError(f"Missing required config key '{key}' in section '{where}'.")
    return mapping[key]


def load_config(path: str | Path | None = None) -> Config:
    """Load and validate the YAML configuration.

    Relative paths in the file are resolved against the repository root, which
    is taken to be the parent of the directory containing the config file.
    """
    config_path = Path(path) if path is not None else DEFAULT_CONFIG_PATH
    try:
        with config_path.open("r", encoding="utf-8") as fh:
            raw = yaml.safe_load(fh)
    except FileNotFoundError as exc:
        raise ConfigError(f"Config file not found: {config_path}") from exc
    except yaml.YAMLError as exc:
        raise ConfigError(f"Config file is not valid YAML: {config_path}") from exc
    if not isinstance(raw, Mapping):
        raise ConfigError(f"Config file must contain a mapping at top level: {config_path}")

    root = config_path.resolve().parent.parent

    project = _require(raw, "project", "<root>")
    paths_raw = _require(raw, "paths", "<root>")
    data_raw = _require(raw, "data", "<root>")
    target_raw = _require(raw, "target", "<root>")
    eda_raw = _require(raw, "eda", "<root>")
    feat_raw = _require(raw, "features", "<root>")
    val_raw = _require(raw, "validation", "<root>")
    grid_raw = _require(raw, "grid", "<root>")
    p3_raw = _require(raw, "phase3", "<root>")
    p3g, p3p = _require(p3_raw, "grid", "phase3"), _require(p3_raw, "probe", "phase3")
    ext_raw = _require(raw, "phase3_extension", "<root>")

    paths = PathsConfig(
        root=root,
        **{
            key: root / str(_require(paths_raw, key, "paths"))
            for key in ("raw_data_dir", "figures_dir", "results_dir", "models_dir", "logs_dir")
        },
    )

    subsets: dict[str, SubsetSpec] = {}
    for name, spec in _require(data_raw, "subsets", "data").items():
        try:
            subsets[name] = SubsetSpec(
                name=name,
                n_train_units=int(_require(spec, "n_train_units", name)),
                n_test_units=int(_require(spec, "n_test_units", name)),
                n_train_rows=int(_require(spec, "n_train_rows", name)),
                n_test_rows=int(_require(spec, "n_test_rows", name)),
                n_operating_conditions=int(_require(spec, "n_operating_conditions", name)),
                fault_modes=tuple(str(m) for m in _require(spec, "fault_modes", name)),
            )
        except (TypeError, ValueError) as exc:
            raise ConfigError(f"Invalid values for subset '{name}': {exc}") from exc

    rul_cap = int(_require(target_raw, "rul_cap", "target"))
    if rul_cap <= 0:
        raise ConfigError("target.rul_cap must be a positive integer.")

    return Config(
        seed=int(_require(project, "seed", "project")),
        log_level=str(_require(project, "log_level", "project")).upper(),
        paths=paths,
        archive_name=str(_require(data_raw, "archive_name", "data")),
        archive_sha256=str(_require(data_raw, "archive_sha256", "data")).lower(),
        subsets=subsets,
        rul_cap=rul_cap,
        eda=EDAConfig(
            n_example_engines=int(_require(eda_raw, "n_example_engines", "eda")),
            spike_window=int(_require(eda_raw, "spike_window", "eda")),
            spike_z_threshold=float(_require(eda_raw, "spike_z_threshold", "eda")),
            initial_window=int(_require(eda_raw, "initial_window", "eda")),
            high_correlation_threshold=float(_require(eda_raw, "high_correlation_threshold", "eda")),
        ),
        features=FeaturesConfig(
            stats=tuple(str(s) for s in _require(feat_raw, "stats", "features")),
            baseline_cycles=int(_require(feat_raw, "baseline_cycles", "features")),
        ),
        validation=ValidationConfig(
            n_folds=int(_require(val_raw, "n_folds", "validation")),
            n_holdout_engines=int(_require(val_raw, "n_holdout_engines", "validation")),
            eval_cutoffs_per_engine=int(_require(val_raw, "eval_cutoffs_per_engine", "validation")),
            eval_min_cutoff=int(_require(val_raw, "eval_min_cutoff", "validation")),
            stress_cutoffs=tuple(int(c) for c in _require(val_raw, "stress_cutoffs", "validation")),
        ),
        grid=GridConfig(
            sensor_configs=tuple(str(v) for v in _require(grid_raw, "sensor_configs", "grid")),
            baseline=tuple(str(v) for v in _require(grid_raw, "baseline", "grid")),
            window=tuple(int(v) for v in _require(grid_raw, "window", "grid")),
            history=tuple(str(v) for v in _require(grid_raw, "history", "grid")),
            metric=tuple(str(v) for v in _require(grid_raw, "metric", "grid")),
            weights=tuple(str(v) for v in _require(grid_raw, "weights", "grid")),
            k=tuple(int(v) for v in _require(grid_raw, "k", "grid")),
        ),
        phase3=Phase3Config(
            holdout_fraction=float(_require(p3_raw, "holdout_fraction", "phase3")),
            window_eval_stride=int(_require(p3_raw, "window_eval_stride", "phase3")),
            grid=Phase3Grid(
                baseline=tuple(str(v) for v in _require(p3g, "baseline", "phase3.grid")),
                normalization=tuple(str(v) for v in _require(p3g, "normalization", "phase3.grid")),
                window=tuple(int(v) for v in _require(p3g, "window", "phase3.grid")),
                history=tuple(str(v) for v in _require(p3g, "history", "phase3.grid")),
                metric=tuple(str(v) for v in _require(p3g, "metric", "phase3.grid")),
                weights=tuple(str(v) for v in _require(p3g, "weights", "phase3.grid")),
                k=tuple(int(v) for v in _require(p3g, "k", "phase3.grid")),
            ),
            probe=ProbeConfig(
                stages={str(k): float(v) for k, v in _require(p3p, "stages", "phase3.probe").items()},
                n_obs=tuple(int(v) for v in _require(p3p, "n_obs", "phase3.probe")),
                window=tuple(int(v) for v in _require(p3p, "window", "phase3.probe")),
                history=tuple(str(v) for v in _require(p3p, "history", "phase3.probe")),
                baseline=tuple(str(v) for v in _require(p3p, "baseline", "phase3.probe")),
                metric=str(_require(p3p, "metric", "phase3.probe")),
                weights=str(_require(p3p, "weights", "phase3.probe")),
                k=int(_require(p3p, "k", "phase3.probe")),
            ),
        ),
        phase3_extension=ExtensionConfig(
            subsets=tuple(str(v) for v in _require(ext_raw, "subsets", "phase3_extension")),
            add_window=tuple(int(v) for v in _require(ext_raw, "add_window", "phase3_extension")),
            add_k=tuple(int(v) for v in _require(ext_raw, "add_k", "phase3_extension")),
        ),
    )
