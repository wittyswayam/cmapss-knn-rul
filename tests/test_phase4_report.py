import math
import shutil
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd

from cmapss_knn.config import PathsConfig, load_config
from cmapss_knn.evaluation.reference import mae_reference, mean_nasa_reference, nasa_score_reference, rmse_reference
from cmapss_knn.experiments import phase4_report as rp
from cmapss_knn.experiments.provenance import md_section

from .helpers import make_subset, spec_for, write_raw_files


class BandTests(unittest.TestCase):
    def test_band_boundaries(self):
        s = pd.Series([0, 30, 31, 60, 61, 100, 101, 195])
        self.assertEqual(list(rp.assign_band(s).astype(str)), ["RUL 0-30", "RUL 0-30", "RUL 31-60", "RUL 31-60", "RUL 61-100", "RUL 61-100", "RUL > 100", "RUL > 100"])


class LabelJoinAndMetricsTests(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())
        self.subset = make_subset(n_train=4, n_test=6)
        write_raw_files(self.tmp / "raw", self.subset)
        base = load_config()
        paths = PathsConfig(root=self.tmp, raw_data_dir=self.tmp / "raw", figures_dir=self.tmp / "f", results_dir=self.tmp / "r", models_dir=self.tmp / "m", logs_dir=self.tmp / "l")
        self.cfg = replace(base, paths=paths, subsets={**base.subsets, "FD001": spec_for(self.subset)})
        rng = np.random.default_rng(0)
        self.preds = pd.DataFrame({"subset": "FD001", "model": "primary", "unit": range(1, 7), "observed_cycles": [20, 40, 80, 90, 25, 70], "window": 75,
                                   "predicted_rul": rng.uniform(5, 125, 6)})
        self.preds["short_history"] = self.preds["observed_cycles"] < self.preds["window"]

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def test_labels_errors_and_nasa_terms(self):
        df = rp.attach_labels(self.cfg, self.preds)
        np.testing.assert_array_equal(df["true_rul"].to_numpy(), self.subset.rul.to_numpy().astype(float))
        np.testing.assert_allclose(df["error"], df["predicted_rul"] - df["true_rul"])  # positive = late
        for r in df.itertuples():
            self.assertTrue(math.isclose(r.nasa_term, nasa_score_reference([r.true_rul], [r.predicted_rul]), rel_tol=1e-12, abs_tol=1e-12))
        self.assertEqual(list(df["true_above_cap"]), [t > 125 for t in df["true_rul"]])

    def test_official_metrics_match_naive_references(self):
        df = rp.attach_labels(self.cfg, self.preds)
        m = rp.official_metrics(df).iloc[0]
        t, p = df["true_rul"].tolist(), df["predicted_rul"].tolist()
        self.assertEqual(m["engines"], 6)
        self.assertTrue(math.isclose(m["RMSE"], rmse_reference(t, p), rel_tol=1e-12))
        self.assertTrue(math.isclose(m["MAE"], mae_reference(t, p), rel_tol=1e-12))
        self.assertTrue(math.isclose(m["NASA score (sum)"], nasa_score_reference(t, p), rel_tol=1e-12))
        self.assertTrue(math.isclose(m["NASA (mean per engine)"], mean_nasa_reference(t, p), rel_tol=1e-12))

    def test_breakdowns_partition_the_engines(self):
        df = rp.attach_labels(self.cfg, self.preds)
        bd = rp.breakdowns(df, cap=125)
        self.assertEqual(int(bd["by_rul_band"]["engines"].sum()), 6)
        self.assertEqual(int(bd["by_history"]["engines"].sum()), 6)
        self.assertEqual(int(bd["above_cap"]["engines"].sum()), 6)
        self.assertEqual(int(bd["posthoc_history_by_cap"]["engines"].sum()), 6)
        w = bd["worst_engines"]
        self.assertLessEqual(len(w), rp.N_WORST)
        self.assertEqual(w.iloc[0]["nasa_term"], df["nasa_term"].max())
        self.assertLessEqual(w["share of subset NASA sum"].sum(), 1.0 + 1e-12)
        clipped = bd["diagnostic_clipped_truth"].iloc[0]
        self.assertTrue(math.isclose(clipped["RMSE"], rmse_reference(np.minimum(df["true_rul"], 125).tolist(), df["predicted_rul"].tolist()), rel_tol=1e-12))

    def test_markdown_section_helper_used_by_the_verification(self):
        text = "## Official test results\n\n| a | b |\n|---|---|\n| 1 | 2 |\n\n## Next\n"
        self.assertIn("| 1 | 2 |", md_section(text, "## Official test results"))


if __name__ == "__main__":
    unittest.main()
