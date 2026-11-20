"""Reading the raw C-MAPSS files, from extracted ``.txt`` files or from the zip."""

from __future__ import annotations

import hashlib
import logging
import zipfile
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import IO, Iterator

import pandas as pd

from ..exceptions import DataFormatError, DataNotFoundError
from .schema import (
    ALL_COLUMNS,
    CYCLE_COLUMN,
    ID_COLUMN,
    N_COLUMNS,
    SUBSET_NAMES,
    rul_filename,
    test_filename,
    train_filename,
)

logger = logging.getLogger(__name__)


@dataclass(frozen=True, eq=False)
class CMAPSSSubset:
    """One C-MAPSS subset.

    Attributes:
        name: Subset name, e.g. ``"FD001"``.
        train: Run-to-failure trajectories; every engine is observed until failure.
        test: Truncated trajectories; each engine stops some cycles before failure.
        rul: True remaining useful life after the last observed test cycle,
            indexed by test unit id (the NASA RUL file lists engines in order).
    """

    name: str
    train: pd.DataFrame
    test: pd.DataFrame
    rul: pd.Series


@dataclass(frozen=True)
class ArchiveCheck:
    path: Path
    sha256: str
    matches_expected: bool


def sha256_of(path: Path, chunk_size: int = 1 << 20) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify_archive(raw_dir: Path, archive_name: str, expected_sha256: str) -> ArchiveCheck | None:
    """Hash the original archive and compare it with the expected digest.

    Returns ``None`` when the archive is absent (e.g. only extracted files exist),
    in which case integrity can only be judged by the structural validation.
    """
    archive = raw_dir / archive_name
    if not archive.is_file():
        return None
    digest = sha256_of(archive)
    return ArchiveCheck(archive, digest, digest == expected_sha256.lower())


@contextmanager
def _open_member(raw_dir: Path, archive_name: str, filename: str) -> Iterator[IO[bytes]]:
    """Open ``filename`` from ``raw_dir`` directly, or from the archive inside it."""
    direct = raw_dir / filename
    archive = raw_dir / archive_name

    if direct.is_file():
        with direct.open("rb") as fh:
            yield fh
        return

    if archive.is_file():
        try:
            zf = zipfile.ZipFile(archive)
        except zipfile.BadZipFile as exc:
            raise DataFormatError(f"'{archive}' is not a valid zip archive.") from exc
        with zf:
            members = {PurePosixPath(n).name: n for n in zf.namelist() if not n.endswith("/")}
            if filename not in members:
                raise DataNotFoundError(f"'{filename}' not found inside '{archive}'.")
            with zf.open(members[filename]) as fh:
                yield fh
        return

    raise DataNotFoundError(
        f"Could not find '{filename}' in '{raw_dir}', and no '{archive_name}' there either. "
        "Place the original CMAPSSData.zip (or the extracted .txt files) in that folder; "
        "see data/README.md."
    )


def _read_table(fh: IO[bytes], filename: str) -> pd.DataFrame:
    try:
        return pd.read_csv(fh, sep=r"\s+", header=None)
    except (pd.errors.ParserError, pd.errors.EmptyDataError, ValueError) as exc:
        raise DataFormatError(f"Could not parse '{filename}' as whitespace-separated numbers: {exc}") from exc


def _read_observations(raw_dir: Path, archive_name: str, filename: str) -> pd.DataFrame:
    with _open_member(raw_dir, archive_name, filename) as fh:
        df = _read_table(fh, filename)
    if df.shape[1] != N_COLUMNS:
        raise DataFormatError(f"'{filename}' has {df.shape[1]} columns, expected {N_COLUMNS}.")
    df.columns = list(ALL_COLUMNS)
    try:
        return df.astype({ID_COLUMN: "int64", CYCLE_COLUMN: "int64"})
    except (ValueError, TypeError) as exc:
        raise DataFormatError(f"'{filename}': unit/cycle columns are not integers: {exc}") from exc


def _read_rul(raw_dir: Path, archive_name: str, filename: str) -> pd.Series:
    with _open_member(raw_dir, archive_name, filename) as fh:
        df = _read_table(fh, filename)
    if df.shape[1] != 1:
        raise DataFormatError(f"'{filename}' has {df.shape[1]} columns, expected 1.")
    values = df.iloc[:, 0]
    index = pd.RangeIndex(1, len(values) + 1, name=ID_COLUMN)
    return pd.Series(values.to_numpy(), index=index, name="rul")


def read_subset(name: str, raw_dir: Path, archive_name: str = "CMAPSSData.zip") -> CMAPSSSubset:
    """Load the train, test and RUL files of one subset.

    Raises:
        ValueError: ``name`` is not one of FD001..FD004.
        DataNotFoundError / DataFormatError: files are missing or unparsable.
    """
    if name not in SUBSET_NAMES:
        raise ValueError(f"Unknown subset '{name}'; expected one of {SUBSET_NAMES}.")
    raw_dir = Path(raw_dir)
    logger.info("Reading %s from %s", name, raw_dir)
    train = _read_observations(raw_dir, archive_name, train_filename(name))
    test = _read_observations(raw_dir, archive_name, test_filename(name))
    rul = _read_rul(raw_dir, archive_name, rul_filename(name))
    logger.info("%s: train %s, test %s, rul %s", name, train.shape, test.shape, rul.shape)
    return CMAPSSSubset(name=name, train=train, test=test, rul=rul)


def read_inputs(name: str, raw_dir: Path, archive_name: str = "CMAPSSData.zip") -> tuple[pd.DataFrame, pd.DataFrame]:
    """Training and test *observations* of one subset. The RUL label file is never opened here."""
    if name not in SUBSET_NAMES:
        raise ValueError(f"Unknown subset '{name}'; expected one of {SUBSET_NAMES}.")
    raw_dir = Path(raw_dir)
    return (_read_observations(raw_dir, archive_name, train_filename(name)), _read_observations(raw_dir, archive_name, test_filename(name)))


def read_test_labels(name: str, raw_dir: Path, archive_name: str = "CMAPSSData.zip") -> pd.Series:
    """True remaining useful life after the last observed test cycle, indexed by test unit id."""
    if name not in SUBSET_NAMES:
        raise ValueError(f"Unknown subset '{name}'; expected one of {SUBSET_NAMES}.")
    return _read_rul(Path(raw_dir), archive_name, rul_filename(name))
