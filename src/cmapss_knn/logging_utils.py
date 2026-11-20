"""Logging setup shared by all scripts."""

from __future__ import annotations

import logging
from pathlib import Path

_FORMAT = "%(asctime)s | %(levelname)-7s | %(name)s | %(message)s"


def setup_logging(level: str = "INFO", log_file: Path | None = None) -> None:
    """Configure the root logger for console output and, optionally, a file.

    Calling this more than once replaces the handlers instead of stacking them.
    """
    root = logging.getLogger()
    for handler in list(root.handlers):
        root.removeHandler(handler)
    root.setLevel(getattr(logging, level.upper(), logging.INFO))

    formatter = logging.Formatter(_FORMAT)
    console = logging.StreamHandler()
    console.setFormatter(formatter)
    root.addHandler(console)

    if log_file is not None:
        log_file.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_file, mode="w", encoding="utf-8")
        file_handler.setFormatter(formatter)
        root.addHandler(file_handler)
