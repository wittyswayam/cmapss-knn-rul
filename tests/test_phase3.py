import json
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd

from cmapss_knn.config import load_config
from cmapss_knn.data.schema import SENSOR_COLUMNS
from cmapss_knn.data.targets import add_train_rul
from cmapss_knn.evaluation.metrics import regression_metrics
from cmapss_knn.experiments import fault_modes as fm
from cmapss_knn.experiments import history_probe as hp
from cmapss_knn.experiments import phase3 as p3
from cmapss_knn.experiments import phase3_analysis as an

from .test_regimes import _multi_regime


def _small_cfg():
    cfg = load_config()
    g = replace(cfg.phase3.grid, baseline=("none",), window=(10,), history=("adaptive",), metric=("euclidean",), weights=("uniform",), k=(5,))
    return replace(cfg, validation=replace(cfg.validation, n_folds=2), phase3=replace(cfg.phase3, grid=g))


def _engines(n_units=10, T=90, seed=0):
    df, _ = _multi_regime(n_units=n_units, T=T, seed=seed)
    return df


class ConfigTests(unittest.TestCase):
    def test_phase3_section(self):
        cfg = load_config()
        self.assertEqual(cfg.phase3.holdout_fraction, 0.2)
        self.assertEqual(cfg.phase3.grid.window, (10, 30, 50, 75))
        self.assertEqual(cfg.phase3.probe.stages["healthy_start"], 0.0)
        self.assertEqual(cfg.phase3.probe.history, ("adaptive", "pad_first", "pad_mean"))


class SweepHelperTests(unittest.TestCase):
    def setUp(self):
        self.cfg = _small_cfg()
        self.setup = p3.make_subset_setup("FD001", _engines(), self.cfg)

    def test_split_is_by_engine_and_sized_by_fraction(self):
        s = self.setup
        self.assertEqual(s.holdout["unit"].nunique(), 2)
        self.assertEqual(len(set(s.tuning["unit"]) & set(s.holdout["unit"])), 0)
        for tr, va in s.folds:
            self.assertEqual(len(set(tr) & set(va)), 0)
            self.assertTrue(set(tr) | set(va) == set(s.tuning["unit"]))

    def test_query_meta_flags(self):
        meta = p3.query_meta(self.setup.tuning, self.setup.cutoffs, self.cfg)
        stride = self.cfg.phase3.window_eval_stride
        self.assertTrue((meta.loc[meta["is_win"], :].index.isin(self.setup.tuning.index[self.setup.tuning["cycle"] % stride == 0])).all())
        self.assertTrue((meta["is_query"] == (meta["is_cutoff"] | meta["is_stress"] | meta["is_win"])).all())
        self.assertEqual(int(meta["is_cutoff"].sum()), len(self.setup.cutoffs))

    def test_score3_matches_regression_metrics(self):
        meta = p3.query_meta(self.setup.tuning, self.setup.cutoffs, self.cfg)
        mq = meta[meta["is_query"]]
        pred = np.linspace(0, 40, len(mq))
        s = p3.score3(mq, pred)
        cut = mq["is_cutoff"].to_numpy()
        m = regression_metrics(mq["rul_clipped"].to_numpy()[cut], pred[cut])
        self.assertAlmostEqual(s["eng_rmse_clip"], m["rmse"])
        self.assertAlmostEqual(s["eng_mae_clip"], m["mae"])
        self.assertAlmostEqual(s["eng_nasa_clip"], m["nasa_mean"])
        self.assertAlmostEqual(s["eng_late_clip"], m["late_share"])
        self.assertEqual(s["n_eng"], int(cut.sum()))

    def test_run_fold_smoke_and_reference_row(self):
        rows = p3.run_fold(self.setup, self.cfg, 0, ("global", "regime"))
        self.assertEqual(set(rows["normalization"]), {"global", "regime", "constant"})
        self.assertEqual(len(rows), 3)  # 1 config x 2 normalisations + constant reference
        self.assertFalse(rows.drop(columns=["baseline", "history", "metric", "weights"]).isna().any().any())
        reg = rows[rows.normalization == "regime"].iloc[0]
        const = rows[rows.normalization == "constant"].iloc[0]
        self.assertTrue(np.isfinite(reg["eng_rmse_clip"]) and np.isfinite(const["eng_rmse_clip"]))

    def test_sweep_resumes_and_loads(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp)
            p3.run_sweep(self.setup, self.cfg, out, ("global",), only_folds=[0])
            self.assertEqual(sorted(p.name for p in out.glob("*.csv")), ["fd001_fold0.csv"])
            first = (out / "fd001_fold0.csv").stat().st_mtime_ns
            p3.run_sweep(self.setup, self.cfg, out, ("global",))
            self.assertEqual((out / "fd001_fold0.csv").stat().st_mtime_ns, first)  # existing fold not recomputed
            cv = p3.load_cv(out, "FD001", 2)
            summary = p3.summarise3(cv)
            self.assertEqual(len(summary), 1)
            self.assertIn("eng_rmse_clip_mean", summary.columns)
            self.assertIn("eng_rmse_clip", p3.constant_reference(cv).index)
            with self.assertRaises(FileNotFoundError):
                p3.load_cv(out, "FD001", 3)

    def test_holdout_audit_counts_every_evaluation(self):
        with tempfile.TemporaryDirectory() as tmp:
            self.assertEqual(p3.record_holdout_evaluation(Path(tmp), "FD002", "first"), 1)
            self.assertEqual(p3.record_holdout_evaluation(Path(tmp), "FD002", "second"), 2)
            self.assertEqual(p3.record_holdout_evaluation(Path(tmp), "FD003", "first"), 1)
            audit = json.loads(p3.holdout_audit_path(Path(tmp)).read_text())
            self.assertEqual(audit["FD002"]["evaluations"], 2)
            self.assertEqual(len(audit["FD002"]["log"]), 2)


def _summary(rows):
    cols = ["normalization", "baseline", "window", "history", "metric", "weights", "k", "eng_rmse_clip_mean", "eng_nasa_clip_mean"]
    return pd.DataFrame(rows, columns=cols)


class SelectionAnalysisTests(unittest.TestCase):
    def _cv(self, summary):
        recs = []
        for _, r in summary.iterrows():
            for f in range(3):
                recs.append({**r.to_dict(), "fold": f, "eng_rmse_clip": r["eng_rmse_clip_mean"] + f * 0.1, "eng_nasa_clip": r["eng_nasa_clip_mean"] + f})
        return pd.DataFrame(recs)

    def test_rmse_and_nasa_can_pick_different_models(self):
        s = _summary([("global", "none", 10, "adaptive", "euclidean", "uniform", 5, 10.0, 9.0), ("global", "none", 10, "adaptive", "euclidean", "uniform", 10, 10.5, 4.0), ("global", "none", 10, "adaptive", "euclidean", "uniform", 20, 12.0, 6.0)])
        c = an.compare_selection(s, self._cv(s))
        self.assertFalse(c["same_configuration"])
        self.assertEqual(int(c["rmse_selected"]["k"]), 5)
        self.assertEqual(int(c["nasa_selected"]["k"]), 10)
        self.assertAlmostEqual(c["rmse_of_nasa_selected_minus_rmse_selected"], 0.5)
        self.assertAlmostEqual(c["nasa_of_rmse_selected_minus_nasa_selected"], 5.0)
        self.assertEqual(c["folds_where_rmse_selected_has_lower_rmse"], 3)

    def test_same_model_when_criteria_agree(self):
        s = _summary([("global", "none", 10, "adaptive", "euclidean", "uniform", 5, 10.0, 3.0), ("global", "none", 10, "adaptive", "euclidean", "uniform", 10, 11.0, 4.0)])
        self.assertTrue(an.compare_selection(s, self._cv(s))["same_configuration"])

    def test_paired_effect(self):
        s = _summary([("global", "none", 10, "adaptive", "euclidean", "uniform", 5, 20.0, 9.0), ("regime", "none", 10, "adaptive", "euclidean", "uniform", 5, 12.0, 3.0),
                      ("global", "none", 10, "adaptive", "euclidean", "uniform", 10, 21.0, 8.0), ("regime", "none", 10, "adaptive", "euclidean", "uniform", 10, 25.0, 2.0)])
        e = an.paired_effect3(s, "normalization", "global", "regime", ["eng_rmse_clip_mean"])
        self.assertEqual(len(e), 2)
        self.assertEqual(sorted(e["eng_rmse_clip_mean_diff"]), [-8.0, 4.0])  # other - base

    def test_regime_variance_share(self):
        from cmapss_knn.features.regimes import RegimeNormalizer

        df = _engines()
        share = an.regime_variance_share(df, RegimeNormalizer(seed=0).fit(df))
        self.assertGreater(share["s2"], 0.9)  # regime shifts dominate the synthetic raw values
        self.assertTrue(((share >= 0) & (share <= 1)).all())


class ProbeTests(unittest.TestCase):
    def setUp(self):
        self.df = add_train_rul(_engines(n_units=4, T=100), cap=125)

    def test_segments_are_rebased_slices_of_the_real_trajectory(self):
        seg, truth = hp.make_probes(self.df, {"healthy_start": 0.0, "mid": 0.5, "late": 0.75}, (5, 10), cap=125)
        self.assertEqual(len(truth), 3 * 2 * 4)
        self.assertEqual(seg["unit"].nunique(), len(truth))  # one unique id per probe
        for _, t in truth.iterrows():
            s = seg[seg["unit"] == t["probe_id"]]
            self.assertEqual(list(s["cycle"]), list(range(1, t["n_obs"] + 1)))
            orig = self.df[(self.df["unit"] == t["engine"]) & self.df["cycle"].between(t["start_cycle"], t["end_cycle"])]
            np.testing.assert_array_equal(s["s2"].to_numpy(), orig["s2"].to_numpy())  # no future cycles leak in
            self.assertEqual(t["true_rul"], t["life"] - t["end_cycle"])
            self.assertGreaterEqual(t["true_rul"], 1)  # always observed before failure
            self.assertEqual(t["true_rul_clip"], min(t["true_rul"], 125))
        starts = truth.groupby("stage")["start_cycle"].median()
        self.assertEqual(starts["healthy_start"], 1)
        self.assertAlmostEqual(starts["mid"], 50, delta=1)
        self.assertAlmostEqual(starts["late"], 75, delta=1)

    def test_start_is_shifted_back_when_history_would_run_past_failure(self):
        _, truth = hp.make_probes(self.df, {"late": 0.99}, (30,), cap=125)
        self.assertTrue((truth["end_cycle"] <= truth["life"] - 1).all())

    def test_short_engines_skipped(self):
        short = self.df[self.df["cycle"] <= 8]
        short = add_train_rul(short.drop(columns=["rul", "rul_clipped"]), cap=125)
        with self.assertRaises(ValueError):  # engines with fewer than n+1 cycles produce no probe at all
            hp.make_probes(short, {"healthy_start": 0.0}, (10,), cap=125)
        _, truth = hp.make_probes(short, {"healthy_start": 0.0}, (5,), cap=125)  # but 8-cycle engines are fine for n=5
        self.assertEqual(len(truth), 4)

    def test_degradation_index_separates_shifted_histories(self):
        ref = self.df[self.df["cycle"] <= 20]
        healthy = self.df[self.df["cycle"] <= 5].copy()
        shifted = healthy.copy()
        shifted[list(SENSOR_COLUMNS)] += 10 * ref[list(SENSOR_COLUMNS)].std()
        sensors = tuple(s for s in SENSOR_COLUMNS if ref[s].std() > 0)
        a = hp.degradation_index(healthy, ref, sensors).median()
        b = hp.degradation_index(shifted, ref, sensors).median()
        self.assertGreater(b, 3 * a)


class FaultModeTests(unittest.TestCase):
    def test_signature_and_split(self):
        rng = np.random.default_rng(0)
        rows = []
        for u in range(1, 61):
            mode = u % 2
            T = 80
            c = np.arange(1, T + 1)
            shift = (np.array([3.0, 0.0, 0.0]) if mode else np.array([0.0, 3.0, 0.0])) * (c / T)[:, None]
            x = shift + rng.normal(0, 0.1, (T, 3))
            rows.append(pd.DataFrame({"unit": u, "cycle": c, "a": x[:, 0], "b": x[:, 1], "c": x[:, 2]}))
        df = pd.concat(rows, ignore_index=True)
        sig = fm.end_of_life_signature(df, ["a", "b", "c"])
        self.assertEqual(sig.shape, (60, 3))
        self.assertGreater(sig.loc[2, "b"], 2.0)  # even engines drift in b
        res = fm.split_in_two(sig, seed=0, n_null=20)
        self.assertGreater(res["silhouette"], res["null_silhouette_max"])
        self.assertEqual(sorted(res["sizes"]), [30, 30])
        pred = res["labels"][np.arange(60) % 2 == 0]
        self.assertEqual(len(set(pred)), 1)  # the split recovers the two modes exactly


if __name__ == "__main__":
    unittest.main()


class ExtensionRunnerTests(unittest.TestCase):
    def setUp(self):
        from cmapss_knn.config import ExtensionConfig
        from cmapss_knn.experiments import phase3_extension as ext

        self.ext = ext
        cfg = _small_cfg()
        self.cfg = replace(cfg, phase3_extension=ExtensionConfig(("FD002",), (20,), (8,)))
        self.setup = p3.make_subset_setup("FD002", _engines(), self.cfg)
        self.norms = ("global", "regime")

    def test_part_grids_keep_existing_values_and_add_only_the_new_ones(self):
        w = self.ext.part_grid(self.cfg, "w100")
        self.assertEqual((w.window, w.k), ((20,), (5, 8)))
        k = self.ext.part_grid(self.cfg, "k200")
        self.assertEqual((k.window, k.k), ((10,), (5, 8)))  # existing window, overlap value 5 (max of the old grid) plus the new 8
        self.assertEqual(self.ext.expected_rows(self.cfg, "w100", self.norms), 4)
        with self.assertRaises(ValueError):
            self.ext.part_grid(self.cfg, "nope")

    def test_chunk_is_saved_with_manifest_and_skipped_on_resume(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp)
            self.assertEqual(self.ext.chunk_status(self.cfg, out, "FD002", 0, "w100", self.norms), "missing")
            self.assertEqual(self.ext.run_chunk(self.setup, self.cfg, out, 0, "w100", self.norms), "ran")
            csv, man = self.ext.chunk_files(out, "FD002", 0, "w100")
            manifest = json.loads(man.read_text())
            self.assertEqual(manifest["rows"], 4)
            self.assertEqual(manifest["grid"]["window"], [20])
            self.assertEqual(len(manifest["config_hash"]), 64)
            self.assertEqual(len(pd.read_csv(csv)), 4)
            stamp = csv.stat().st_mtime_ns
            self.assertEqual(self.ext.run_chunk(self.setup, self.cfg, out, 0, "w100", self.norms), "skipped")
            self.assertEqual(csv.stat().st_mtime_ns, stamp)  # not recomputed

    def test_unfinished_chunk_without_manifest_is_recomputed(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp)
            csv, man = self.ext.chunk_files(out, "FD002", 1, "k200")
            csv.write_text("partial,garbage\n1,2\n")  # a run killed after writing the CSV but before the manifest
            self.assertEqual(self.ext.chunk_status(self.cfg, out, "FD002", 1, "k200", self.norms), "missing")
            self.assertEqual(self.ext.run_chunk(self.setup, self.cfg, out, 1, "k200", self.norms), "ran")
            self.assertEqual(len(pd.read_csv(csv)), 4)

    def test_chunk_from_a_different_configuration_is_refused(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp)
            self.ext.run_chunk(self.setup, self.cfg, out, 0, "w100", self.norms)
            changed = replace(self.cfg, seed=self.cfg.seed + 1)
            with self.assertRaises(self.ext.ExtensionMismatch):
                self.ext.chunk_status(changed, out, "FD002", 0, "w100", self.norms)
            changed_grid = replace(self.cfg, phase3_extension=replace(self.cfg.phase3_extension, add_k=(9,)))
            with self.assertRaises(self.ext.ExtensionMismatch):
                self.ext.chunk_status(changed_grid, out, "FD002", 0, "w100", self.norms)
