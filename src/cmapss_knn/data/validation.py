"""Integrity checks for a loaded C-MAPSS subset.

Checks are either ``error`` (the data is unusable if they fail) or ``info``
(facts worth knowing, never a failure). Nothing here modifies the data.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
import pandas as pd

from ..config import SubsetSpec
from ..exceptions import DataValidationError
from .ingestion import CMAPSSSubset
from .schema import ALL_COLUMNS, CYCLE_COLUMN, ID_COLUMN, SENSOR_COLUMNS, OP_SETTING_COLUMNS

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class CheckResult:
    name: str
    passed: bool
    detail: str
    severity: str = "error"  # "error" or "info"


@dataclass(frozen=True)
class ValidationReport:
    subset: str
    checks: tuple[CheckResult, ...]

    @property
    def passed(self) -> bool:
        return all(c.passed for c in self.checks if c.severity == "error")

    @property
    def failures(self) -> tuple[CheckResult, ...]:
        return tuple(c for c in self.checks if c.severity == "error" and not c.passed)

    def to_frame(self) -> pd.DataFrame:
        return pd.DataFrame(
            [(c.name, c.severity, c.passed, c.detail) for c in self.checks],
            columns=["check", "severity", "passed", "detail"],
        )

    def format_text(self) -> str:
        lines = [f"Validation report for {self.subset}: {'PASSED' if self.passed else 'FAILED'}"]
        for c in self.checks:
            tag = "INFO" if c.severity == "info" else ("ok  " if c.passed else "FAIL")
            lines.append(f"  [{tag}] {c.name}: {c.detail}")
        return "\n".join(lines)


def _fmt_list(items: list[str], limit: int = 8) -> str:
    shown = ", ".join(items[:limit])
    return shown + (f", ... (+{len(items) - limit} more)" if len(items) > limit else "")


def _frame_checks(label: str, df: pd.DataFrame) -> list[CheckResult]:
    checks: list[CheckResult] = []

    schema_ok = list(df.columns) == list(ALL_COLUMNS)
    checks.append(CheckResult(f"{label}: column schema", schema_ok, f"{df.shape[1]} columns"))
    if not schema_ok:
        return checks  # later checks assume the schema

    n_missing = int(df.isna().sum().sum())
    checks.append(CheckResult(f"{label}: no missing values", n_missing == 0, f"{n_missing} missing cells"))

    n_inf = int(np.isinf(df.to_numpy(dtype="float64")).sum())
    checks.append(CheckResult(f"{label}: no infinite values", n_inf == 0, f"{n_inf} infinite cells"))

    units = np.sort(df[ID_COLUMN].unique())
    contiguous = len(units) > 0 and bool(np.array_equal(units, np.arange(1, len(units) + 1)))
    checks.append(
        CheckResult(
            f"{label}: unit ids are 1..N",
            contiguous,
            f"{len(units)} units, ids {units.min() if len(units) else 'n/a'}..{units.max() if len(units) else 'n/a'}",
        )
    )

    n_dup = int(df.duplicated([ID_COLUMN, CYCLE_COLUMN]).sum())
    checks.append(CheckResult(f"{label}: no duplicate (unit, cycle)", n_dup == 0, f"{n_dup} duplicates"))

    grouped = df.groupby(ID_COLUMN)[CYCLE_COLUMN]
    bad_start = int((grouped.min() != 1).sum())
    bad_steps = int((grouped.diff().dropna() != 1).sum())
    checks.append(
        CheckResult(
            f"{label}: cycles start at 1 and increase by 1",
            bad_start == 0 and bad_steps == 0,
            f"{bad_start} engines not starting at 1, {bad_steps} irregular steps",
        )
    )

    keys = df[ID_COLUMN].to_numpy(dtype="int64") * 1_000_000 + df[CYCLE_COLUMN].to_numpy(dtype="int64")
    checks.append(
        CheckResult(
            f"{label}: rows sorted by (unit, cycle)",
            bool(np.all(np.diff(keys) > 0)),
            "strictly increasing key" if np.all(np.diff(keys) > 0) else "rows are not sorted",
        )
    )
    return checks


def validate_subset(subset: CMAPSSSubset, spec: SubsetSpec, *, raise_on_error: bool = True) -> ValidationReport:
    """Run all integrity checks on ``subset`` against the expected ``spec``."""
    checks: list[CheckResult] = []
    checks += _frame_checks("train", subset.train)
    checks += _frame_checks("test", subset.test)

    n_train_units = subset.train[ID_COLUMN].nunique() if ID_COLUMN in subset.train else -1
    n_test_units = subset.test[ID_COLUMN].nunique() if ID_COLUMN in subset.test else -1
    for name, got, want in (
        ("train engines match spec", n_train_units, spec.n_train_units),
        ("test engines match spec", n_test_units, spec.n_test_units),
        ("train rows match spec", len(subset.train), spec.n_train_rows),
        ("test rows match spec", len(subset.test), spec.n_test_rows),
    ):
        checks.append(CheckResult(name, got == want, f"got {got}, expected {want}"))

    rul = subset.rul
    checks.append(
        CheckResult("RUL file: one value per test engine", len(rul) == n_test_units, f"{len(rul)} values, {n_test_units} engines")
    )
    rul_valid = bool(rul.notna().all()) and bool((rul >= 0).all()) and bool(np.all(np.equal(np.mod(rul.to_numpy(), 1), 0)))
    checks.append(CheckResult("RUL file: non-negative whole numbers", rul_valid, f"min {rul.min()}, max {rul.max()}"))

    # Informational facts (never failures).
    if list(subset.train.columns) == list(ALL_COLUMNS):
        feature_cols = list(OP_SETTING_COLUMNS) + list(SENSOR_COLUMNS)
        const = [c for c in feature_cols if subset.train[c].nunique() <= 1]
        checks.append(CheckResult("train: constant columns", True, _fmt_list(const) or "none", "info"))

        if list(subset.test.columns) == list(ALL_COLUMNS):
            lo, hi = subset.train[feature_cols].min(), subset.train[feature_cols].max()
            outside = {
                c: float(((subset.test[c] < lo[c]) | (subset.test[c] > hi[c])).mean())
                for c in feature_cols
                if c not in const
            }
            flagged = [f"{c} ({p:.2%})" for c, p in outside.items() if p > 0]
            checks.append(
                CheckResult(
                    "test: values outside the training range",
                    True,
                    _fmt_list(flagged) or "none",
                    "info",
                )
            )

    report = ValidationReport(subset.name, tuple(checks))
    logger.info("%s validation %s", subset.name, "passed" if report.passed else "FAILED")
    if raise_on_error and not report.passed:
        failed = "; ".join(f"{c.name} ({c.detail})" for c in report.failures)
        raise DataValidationError(f"{subset.name} failed validation: {failed}")
    return report
