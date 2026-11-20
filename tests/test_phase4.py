import json
import shutil
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd

from cmapss_knn.config import PathsConfig, load_config
from cmapss_knn.data.targets import add_train_rul
from cmapss_knn.experiments import phase4_eval as ev
from cmapss_knn.features.windows import WindowConfig
from cmapss_knn.models.pipeline import PipelineConfig, RULKNNPipeline

from .helpers import make_subset, spec_for, write_raw_files

ROOT = Path(__file__).resolve().parents[1]
CFG_KEYS = {"normalization": "global", "baseline": "none", "window": 10, "history": "pad_first", "metric": "euclidean", "weights": "distance", "k": 5}


def _selection(extra=None):
    return {**CFG_KEYS, "eng_rmse_clip_mean": 1.0, "eng_mae_clip_mean": 1.0, "eng_nasa_clip_mean": 1.0, **(extra or {})}


class Phase4FlowTests(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())
        self.subset = make_subset(n_train=8, n_test=5)
        raw = self.tmp / "raw"
        write_raw_files(raw, self.subset)
        (self.tmp / "docs").mkdir()
        shutil.copy(ROOT / "docs" / "EVALUATION_PROTOCOL.md", self.tmp / "docs")
        res = self.tmp / "results"
        for name, d, key in (("selection_fd001.json", "phase3", "rmse_selected"), ("selection_fd003.json", "phase3", "rmse_selected")):
            (res / d).mkdir(parents=True, exist_ok=True)
            (res / d / name).write_text(json.dumps({key: _selection()}))
        (res / "phase3_extended").mkdir(parents=True, exist_ok=True)
        for fd in ("fd002", "fd004"):
            (res / "phase3_extended" / f"selection_extended_{fd}.json").write_text(json.dumps({"rmse_selected_primary": _selection(), "nasa_selected_secondary": _selection({"k": 8})}))
        base = load_config()
        paths = PathsConfig(root=self.tmp, raw_data_dir=raw, figures_dir=self.tmp / "figures", results_dir=res, models_dir=self.tmp / "models", logs_dir=self.tmp / "logs")
        self.cfg = replace(base, paths=paths, subsets={**base.subsets, "FD001": spec_for(self.subset)})

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def test_freeze_is_written_once_and_detects_tampering(self):
        path = ev.freeze(self.cfg)
        frozen = ev.load_frozen(self.cfg)
        self.assertEqual([c["id"] for c in frozen["chunks"]], ["FD001:primary", "FD002:primary", "FD002:nasa_secondary", "FD003:primary", "FD004:primary"])
        self.assertIn("data_integrity_disclosure", frozen["protocol"])
        with self.assertRaises(ev.FrozenMismatch):
            ev.freeze(self.cfg)  # never overwritten
        data = json.loads(path.read_text())
        data["chunks"][0]["config"]["k"] = 99  # someone edits a frozen configuration
        path.write_text(json.dumps(data))
        with self.assertRaises(ev.FrozenMismatch):
            ev.load_frozen(self.cfg)

    def test_selection_file_change_after_freeze_is_detected(self):
        ev.freeze(self.cfg)
        p = self.cfg.paths.results_dir / "phase3" / "selection_fd001.json"
        p.write_text(json.dumps({"rmse_selected": _selection({"k": 7})}))
        with self.assertRaises(ev.FrozenMismatch):
            ev.load_frozen(self.cfg)

    def test_chunk_predictions_resume_audit_and_label_independence(self):
        ev.freeze(self.cfg)
        (self.cfg.paths.raw_data_dir / "RUL_FD001.txt").unlink()  # the label file is not needed (or opened) to predict
        self.assertEqual(ev.evaluate_chunk(self.cfg, "FD001:primary"), "ran")
        frozen = ev.load_frozen(self.cfg)
        chunk = ev.find_chunk(frozen, "FD001:primary")
        csv, man = ev.chunk_files(self.cfg, chunk)
        out = pd.read_csv(csv)
        self.assertEqual(len(out), 5)
        self.assertFalse({"true_rul", "rul", "error"} & set(out.columns))  # no labels in the locked predictions
        m = json.loads(man.read_text())
        self.assertFalse(m["labels_read"])
        self.assertLess(m["scratch_vs_sklearn_max_abs_diff"], 1e-9)
        self.assertEqual(m["frozen_sha256"], frozen["frozen_sha256"])
        # matches a direct fit/predict with the same configuration on all training engines
        from cmapss_knn.data.ingestion import read_inputs

        train, test = read_inputs("FD001", self.cfg.paths.raw_data_dir)  # the same (4-decimal) files the chunk read
        pc = PipelineConfig(WindowConfig(10, self.cfg.features.stats, "pad_first", "none", self.cfg.features.baseline_cycles), "A", 5, "euclidean", "distance", "sklearn", "global", self.cfg.seed)
        direct = RULKNNPipeline(pc).fit(add_train_rul(train, cap=self.cfg.rul_cap)).predict_last_cycle(test)
        np.testing.assert_allclose(out["predicted_rul"].to_numpy(), direct.to_numpy(), atol=1e-9)
        np.testing.assert_array_equal(out["observed_cycles"].to_numpy(), test.groupby("unit")["cycle"].max().to_numpy())
        # resume: a completed chunk is skipped and not counted again
        stamp = csv.stat().st_mtime_ns
        self.assertEqual(ev.evaluate_chunk(self.cfg, "FD001:primary"), "skipped")
        self.assertEqual(csv.stat().st_mtime_ns, stamp)
        audit = json.loads(ev._audit_path(self.cfg).read_text())
        self.assertEqual(audit["FD001:primary"]["evaluations"], 1)
        self.assertEqual(ev.chunk_status(self.cfg, frozen, chunk), "complete")
        # a forced repeat is allowed but counted
        self.assertEqual(ev.evaluate_chunk(self.cfg, "FD001:primary", force=True), "ran")
        self.assertEqual(json.loads(ev._audit_path(self.cfg).read_text())["FD001:primary"]["evaluations"], 2)

    def test_altered_predictions_are_detected(self):
        ev.freeze(self.cfg)
        ev.evaluate_chunk(self.cfg, "FD001:primary")
        frozen = ev.load_frozen(self.cfg)
        chunk = ev.find_chunk(frozen, "FD001:primary")
        csv, _ = ev.chunk_files(self.cfg, chunk)
        df = pd.read_csv(csv)
        df.loc[0, "predicted_rul"] += 1.0
        df.to_csv(csv, index=False)
        with self.assertRaises(ev.FrozenMismatch):
            ev.chunk_status(self.cfg, frozen, chunk)

    def test_unknown_chunk_and_missing_freeze(self):
        with self.assertRaises(ev.FrozenMismatch):
            ev.load_frozen(self.cfg)
        ev.freeze(self.cfg)
        with self.assertRaises(ValueError):
            ev.find_chunk(ev.load_frozen(self.cfg), "FD009:primary")

    def test_labels_have_their_own_loader(self):
        from cmapss_knn.data.ingestion import read_inputs, read_test_labels

        train, test = read_inputs("FD001", self.cfg.paths.raw_data_dir)
        labels = read_test_labels("FD001", self.cfg.paths.raw_data_dir)
        self.assertEqual(train["unit"].nunique(), 8)
        self.assertEqual(list(labels.index), [1, 2, 3, 4, 5])
        self.assertEqual(list(labels), list(self.subset.rul))


if __name__ == "__main__":
    unittest.main()
