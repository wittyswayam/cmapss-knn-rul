"""Compare result files of two project directories (reference vs. a fresh re-run).

    python scripts/compare_results.py --reference DIR_A --candidate DIR_B [--log LOG.md] PATH [PATH ...]

PATH is relative to both roots and may contain wildcards. CSV and JSON are compared numerically (relative and absolute
tolerance 1e-9; text fields exactly), Markdown byte for byte, PNG pixel for pixel. Timestamps and durations in JSON are ignored.
Exit code 0 only if every compared file matches.
"""

from __future__ import annotations

import argparse
import glob
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

IGNORED_KEYS = {"seconds", "completed_utc", "time_utc", "backend_seconds"}  # timings and timestamps legitimately differ between runs


def _close(a, b) -> tuple[bool, float]:
    if isinstance(a, dict) and isinstance(b, dict):
        if set(a) - IGNORED_KEYS != set(b) - IGNORED_KEYS:
            return False, float("inf")
        res = [_close(a[k], b[k]) for k in a if k not in IGNORED_KEYS]
        return all(r[0] for r in res), max([r[1] for r in res], default=0.0)
    if isinstance(a, list) and isinstance(b, list):
        if len(a) != len(b):
            return False, float("inf")
        res = [_close(x, y) for x, y in zip(a, b)]
        return all(r[0] for r in res), max([r[1] for r in res], default=0.0)
    if isinstance(a, (int, float)) and isinstance(b, (int, float)) and not isinstance(a, bool):
        d = abs(float(a) - float(b))
        return bool(np.isclose(a, b, rtol=1e-9, atol=1e-9)), d
    return a == b, 0.0 if a == b else float("inf")


def compare(ref: Path, cand: Path) -> tuple[bool, str]:
    if not cand.is_file():
        return False, "missing in candidate"
    suffix = ref.suffix.lower()
    if suffix == ".csv":
        a, b = pd.read_csv(ref), pd.read_csv(cand)
        if a.shape != b.shape or list(a.columns) != list(b.columns):
            return False, f"shape/columns differ {a.shape} vs {b.shape}"
        worst = 0.0
        for c in a.columns:
            if c.startswith("seconds_"):
                continue  # timing columns
            if pd.api.types.is_numeric_dtype(a[c]) and pd.api.types.is_numeric_dtype(b[c]):
                x, y = a[c].to_numpy(float), b[c].to_numpy(float)
                if not np.allclose(x, y, rtol=1e-9, atol=1e-9, equal_nan=True):
                    return False, f"column {c}: max abs diff {np.nanmax(np.abs(x - y)):.3e}"
                worst = max(worst, float(np.nanmax(np.abs(x - y))) if len(x) else 0.0)
            elif not a[c].astype(str).equals(b[c].astype(str)):
                return False, f"text column {c} differs"
        return True, f"max abs numeric diff {worst:.2e}"
    if suffix == ".json":
        ok, d = _close(json.loads(ref.read_text()), json.loads(cand.read_text()))
        return ok, f"max abs numeric diff {d:.2e}"
    if suffix == ".png":
        import matplotlib.image as mpimg

        x, y = mpimg.imread(ref), mpimg.imread(cand)
        return (x.shape == y.shape and bool(np.array_equal(x, y))), "pixel-identical" if x.shape == y.shape and np.array_equal(x, y) else "pixels differ"
    return ref.read_bytes() == cand.read_bytes(), "byte-identical" if ref.read_bytes() == cand.read_bytes() else "bytes differ"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--reference", required=True)
    ap.add_argument("--candidate", required=True)
    ap.add_argument("--log", default=None)
    ap.add_argument("--exclude", nargs="*", default=[], help="skip files whose path contains one of these strings")
    ap.add_argument("paths", nargs="+")
    a = ap.parse_args()
    ref_root, cand_root = Path(a.reference), Path(a.candidate)
    files = sorted({Path(p).relative_to(ref_root) for pat in a.paths for p in glob.glob(str(ref_root / pat), recursive=True) if Path(p).is_file() and not any(x in p for x in a.exclude)})
    if not files:
        print("no files matched")
        return 1
    ok_all, lines = True, []
    for f in files:
        ok, note = compare(ref_root / f, cand_root / f)
        ok_all &= ok
        lines.append(f"| {f.as_posix()} | {'MATCH' if ok else 'DIFFERENT'} | {note} |")
    print(f"{sum('MATCH' in l and 'DIFFERENT' not in l for l in lines)}/{len(lines)} files match" + ("" if ok_all else "  <-- DIFFERENCES: " + "; ".join(l for l in lines if 'DIFFERENT' in l)))
    if a.log:
        Path(a.log).parent.mkdir(parents=True, exist_ok=True)
        with open(a.log, "a", encoding="utf-8") as fh:
            fh.write("\n".join(lines) + "\n")
    return 0 if ok_all else 1


if __name__ == "__main__":
    sys.exit(main())
