"""Phase 4: official test-set evaluation (frozen models). Run from the repository root.

    python scripts/run_phase4.py freeze                       # once, before any test prediction
    python scripts/run_phase4.py status                       # which chunks are complete
    python scripts/run_phase4.py evaluate [--chunk FD002:primary ...] [--force]   # resumable; completed chunks are skipped
    python scripts/run_phase4.py report                       # join labels to the locked predictions; tables, figures, report
    python scripts/run_phase4.py verify                       # independent checks of the saved results
"""

from __future__ import annotations

import _paths  # noqa: F401

import argparse
import logging
import sys

from cmapss_knn.config import load_config
from cmapss_knn.exceptions import CMAPSSError
from cmapss_knn.experiments import phase4_eval as ev
from cmapss_knn.logging_utils import setup_logging

logger = logging.getLogger("phase4")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    e = sub.add_parser("evaluate"); e.add_argument("--chunk", nargs="+", default=None); e.add_argument("--force", action="store_true")
    for n in ("freeze", "status", "report", "verify"):
        sub.add_parser(n)
    a = ap.parse_args()
    cfg = load_config()
    setup_logging(cfg.log_level, cfg.paths.logs_dir / f"phase4_{a.cmd}.log")
    try:
        if a.cmd == "freeze":
            print(ev.freeze(cfg))
        elif a.cmd == "status":
            frozen = ev.load_frozen(cfg)
            for c in frozen["chunks"]:
                print(c["id"].ljust(20), ev.chunk_status(cfg, frozen, c))
        elif a.cmd == "evaluate":
            frozen = ev.load_frozen(cfg)
            for cid in a.chunk or [c["id"] for c in frozen["chunks"]]:
                print(cid, ev.evaluate_chunk(cfg, cid, a.force))
        else:
            from cmapss_knn.experiments import phase4_report as rp
            return {"report": rp.run_report, "verify": rp.run_verify}[a.cmd](cfg)
    except CMAPSSError as exc:
        logger.error("%s", exc)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
