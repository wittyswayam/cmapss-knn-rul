"""Phase 3 driver. Sub-commands (run from the repository root):

    python scripts/run_phase3.py sweep --subset FD002 [--folds 0 1]   # CV sweep, resumable per fold
    python scripts/run_phase3.py analyze                              # selection analysis over all finished subsets
    python scripts/run_phase3.py holdout --subset FD002               # development holdout, evaluated at most once
    python scripts/run_phase3.py probe --subset FD001                 # short-history robustness experiment
    python scripts/run_phase3.py faultmodes                           # unsupervised look at fault-mode structure
    python scripts/run_phase3.py extend --subset FD002 [--folds 0] [--status]   # FD002/FD004 grid extension, resumable per chunk
    python scripts/run_phase3.py analyze-extended                    # selection over the extended grid
    python scripts/run_phase3.py regimes                              # save and independently check the regime detector output
    python scripts/run_phase3.py report                               # write results/phase3/phase3_report.md
The official test files are never read by any sub-command.
"""

from __future__ import annotations

import _paths  # noqa: F401

import argparse
import logging
import sys

from cmapss_knn.config import load_config
from cmapss_knn.data.ingestion import read_subset
from cmapss_knn.data.schema import SUBSET_NAMES
from cmapss_knn.data.validation import validate_subset
from cmapss_knn.exceptions import CMAPSSError
from cmapss_knn.experiments import phase3 as p3
from cmapss_knn.logging_utils import setup_logging

logger = logging.getLogger("phase3")


def _setup(cfg, name):
    subset = read_subset(name, cfg.paths.raw_data_dir, cfg.archive_name)
    validate_subset(subset, cfg.subsets[name])
    return p3.make_subset_setup(name, subset.train, cfg)  # only subset.train is used; test data is never touched


def cmd_sweep(cfg, args) -> int:
    setup = _setup(cfg, args.subset)
    norm = p3.regimes_for(setup, cfg.seed)
    variants = tuple(n for n in cfg.phase3.grid.normalization if n == "global" or norm.n_regimes_ > 1)
    logger.info("%s: %d regime(s) detected on tuning rows; normalisation variants: %s", args.subset, norm.n_regimes_, variants)
    p3.run_sweep(setup, cfg, cfg.paths.results_dir / "phase3" / "cv", variants, args.folds)
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("sweep"); s.add_argument("--subset", required=True, choices=SUBSET_NAMES); s.add_argument("--folds", nargs="+", type=int, default=None)
    for name in ("holdout", "probe"):
        q = sub.add_parser(name); q.add_argument("--subset", required=True, choices=SUBSET_NAMES)
        if name == "probe":
            q.add_argument("--folds", nargs="+", type=int, default=None)
        if name == "holdout":
            q.add_argument("--force", action="store_true", help="allow a repeat evaluation (it will be counted in the audit file)")
    e = sub.add_parser("extend"); e.add_argument("--subset", required=True, choices=SUBSET_NAMES)
    e.add_argument("--folds", nargs="+", type=int, default=None); e.add_argument("--parts", nargs="+", choices=("w100", "k200"), default=None)
    e.add_argument("--status", action="store_true", help="only print which chunks are complete")
    for name in ("analyze", "faultmodes", "report", "regimes", "analyze-extended"):
        sub.add_parser(name)
    args = parser.parse_args()
    cfg = load_config()
    setup_logging(cfg.log_level, cfg.paths.logs_dir / f"phase3_{args.cmd}.log")
    try:
        handler = {"sweep": cmd_sweep}.get(args.cmd)
        if handler is None:
            from cmapss_knn.experiments import phase3_commands as cmds  # later sub-commands live in their own module
            handler = getattr(cmds, f"cmd_{args.cmd.replace('-', '_')}")
        return handler(cfg, args)
    except CMAPSSError as exc:
        logger.error("%s", exc)
        return 1


if __name__ == "__main__":
    sys.exit(main())
