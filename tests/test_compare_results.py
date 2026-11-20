import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
TOOL = ROOT / "scripts" / "compare_results.py"


def _run(ref: Path, cand: Path, *paths: str) -> subprocess.CompletedProcess:
    return subprocess.run([sys.executable, str(TOOL), "--reference", str(ref), "--candidate", str(cand), *paths], capture_output=True, text=True)


class CompareResultsTests(unittest.TestCase):
    def test_matches_within_tolerance_and_flags_real_differences(self):
        with tempfile.TemporaryDirectory() as a, tempfile.TemporaryDirectory() as b:
            a, b = Path(a), Path(b)
            pd.DataFrame({"k": [1, 2], "rmse": [1.0, 2.0], "seconds_scratch": [1.0, 2.0], "name": ["x", "y"]}).to_csv(a / "t.csv", index=False)
            pd.DataFrame({"k": [1, 2], "rmse": [1.0 + 1e-12, 2.0], "seconds_scratch": [9.0, 9.0], "name": ["x", "y"]}).to_csv(b / "t.csv", index=False)
            self.assertEqual(_run(a, b, "t.csv").returncode, 0)  # tiny float noise and timing columns are ignored
            pd.DataFrame({"k": [1, 2], "rmse": [1.0, 2.5], "seconds_scratch": [1.0, 2.0], "name": ["x", "y"]}).to_csv(b / "t.csv", index=False)
            self.assertEqual(_run(a, b, "t.csv").returncode, 1)  # a real difference in a metric
            pd.DataFrame({"k": [1, 2], "rmse": [1.0, 2.0], "seconds_scratch": [1.0, 2.0], "name": ["x", "z"]}).to_csv(b / "t.csv", index=False)
            self.assertEqual(_run(a, b, "t.csv").returncode, 1)  # a text field changed

    def test_json_ignores_timestamps_but_not_values(self):
        with tempfile.TemporaryDirectory() as a, tempfile.TemporaryDirectory() as b:
            a, b = Path(a), Path(b)
            (a / "m.json").write_text(json.dumps({"rows": 4, "seconds": 3.0, "completed_utc": "x", "grid": {"k": [5, 8]}}))
            (b / "m.json").write_text(json.dumps({"rows": 4, "seconds": 99.0, "completed_utc": "y", "grid": {"k": [5, 8]}}))
            self.assertEqual(_run(a, b, "m.json").returncode, 0)
            (b / "m.json").write_text(json.dumps({"rows": 5, "seconds": 3.0, "completed_utc": "x", "grid": {"k": [5, 8]}}))
            self.assertEqual(_run(a, b, "m.json").returncode, 1)

    def test_missing_candidate_and_exclude(self):
        with tempfile.TemporaryDirectory() as a, tempfile.TemporaryDirectory() as b:
            a, b = Path(a), Path(b)
            (a / "r.md").write_text("hello")
            self.assertEqual(_run(a, b, "r.md").returncode, 1)  # missing in candidate
            (b / "r.md").write_text("hello")
            self.assertEqual(_run(a, b, "r.md").returncode, 0)
            (b / "r.md").write_text("bye")
            self.assertEqual(_run(a, b, "r.md").returncode, 1)  # byte difference in text


if __name__ == "__main__":
    unittest.main()
