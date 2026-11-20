"""Check that the raw C-MAPSS data is present, intact and structurally valid.

Usage (from the repository root):
    python scripts/verify_data.py                     # FD001 only
    python scripts/verify_data.py --subsets FD001 FD002 FD003 FD004
Exit code is 0 when everything passes, 1 otherwise.
"""

from __future__ import annotations

import _paths  # noqa: F401  (adds src/ to sys.path)

import argparse
import logging
import sys

from cmapss_knn.config import load_config
from cmapss_knn.data.ingestion import read_subset, verify_archive
from cmapss_knn.data.schema import SUBSET_NAMES
from cmapss_knn.data.validation import validate_subset
from cmapss_knn.exceptions import CMAPSSError
from cmapss_knn.logging_utils import setup_logging

logger = logging.getLogger("verify_data")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--subsets", nargs="+", default=["FD001"], choices=SUBSET_NAMES)
    parser.add_argument("--config", default=None, help="Path to a YAML config (default: configs/default.yaml)")
    args = parser.parse_args()

    cfg = load_config(args.config)
    setup_logging(cfg.log_level, cfg.paths.logs_dir / "verify_data.log")

    try:
        check = verify_archive(cfg.paths.raw_data_dir, cfg.archive_name, cfg.archive_sha256)
        if check is None:
            logger.warning("%s not found in %s; relying on structural validation of extracted files.", cfg.archive_name, cfg.paths.raw_data_dir)
        elif check.matches_expected:
            logger.info("Archive SHA-256 matches the expected digest (%s).", check.sha256)
        else:
            logger.warning("Archive SHA-256 is %s, which differs from the expected digest. The data may be a different copy.", check.sha256)

        all_ok = True
        for name in args.subsets:
            subset = read_subset(name, cfg.paths.raw_data_dir, cfg.archive_name)
            report = validate_subset(subset, cfg.subsets[name], raise_on_error=False)
            print(report.format_text())
            all_ok &= report.passed
    except CMAPSSError as exc:
        logger.error("%s", exc)
        return 1
    return 0 if all_ok else 1


if __name__ == "__main__":
    sys.exit(main())
