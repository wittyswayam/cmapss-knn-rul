"""Run the exploratory data analysis for one C-MAPSS subset.

Usage (from the repository root):
    python scripts/run_eda.py --subset FD001
Writes figures to figures/<subset>/ and tables plus a report to results/eda/.
"""

from __future__ import annotations

import _paths  # noqa: F401  (adds src/ to sys.path)

import argparse
import logging
import sys

from cmapss_knn.config import load_config
from cmapss_knn.data.ingestion import read_subset
from cmapss_knn.data.schema import SUBSET_NAMES
from cmapss_knn.data.validation import validate_subset
from cmapss_knn.eda.pipeline import run_eda
from cmapss_knn.exceptions import CMAPSSError
from cmapss_knn.logging_utils import setup_logging

logger = logging.getLogger("run_eda")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--subset", default="FD001", choices=SUBSET_NAMES)
    parser.add_argument("--config", default=None, help="Path to a YAML config (default: configs/default.yaml)")
    args = parser.parse_args()

    cfg = load_config(args.config)
    setup_logging(cfg.log_level, cfg.paths.logs_dir / "run_eda.log")
    try:
        subset = read_subset(args.subset, cfg.paths.raw_data_dir, cfg.archive_name)
        validate_subset(subset, cfg.subsets[args.subset])  # raises on failure
        run_eda(subset, cfg)
    except CMAPSSError as exc:
        logger.error("%s", exc)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
