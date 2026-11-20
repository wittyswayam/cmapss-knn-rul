import unittest

import numpy as np
import pandas as pd

from cmapss_knn.data.schema import SENSOR_COLUMNS
from cmapss_knn.features import SensorSelector, WindowConfig, build_features, causal_baseline, feature_column_indices

from .helpers import make_subset

SENSORS = ("s2", "s3")


def _df(n_units=3, seed=0, lengths=(60, 45, 12)):
    rng = np.random.default_rng(seed)
    rows = []
    for u, T in enumerate(lengths[:n_units], start=1):
        c = np.arange(1, T + 1)
        rows.append(pd.DataFrame({"unit": u, "cycle": c, **{s: rng.normal(100, 3, T) + 0.1 * c * (i + 1) for i, s in enumerate(SENSORS)}}))
    return pd.concat(rows, ignore_index=True)


def _feat(df, **kw):
    return build_features(df, WindowConfig(**kw), sensors=SENSORS)


class WindowFeatureTests(unittest.TestCase):
    def test_full_window_matches_direct_computation(self):
        df = _df()
        W = 10
        ft = _feat(df, window=W, stats=("last", "mean", "std", "slope"))
        g = df[df.unit == 1]
        x = g["s2"].to_numpy()
        t = 40  # 1-based cycle, window = cycles 31..40
        w = x[t - W : t]
        row = ft.X[(df.unit == 1).to_numpy()][t - 1]
        c = ft.columns.index
        self.assertAlmostEqual(row[c("s2_mean")], w.mean(), places=10)
        self.assertAlmostEqual(row[c("s2_std")], w.std(), places=10)
        self.assertAlmostEqual(row[c("s2_slope")], np.polyfit(np.arange(W), w, 1)[0], places=10)
        self.assertAlmostEqual(row[c("s2_last")], x[t - 1], places=12)

    def test_adaptive_uses_only_available_cycles(self):
        df = _df()
        ft = _feat(df, window=30, history="adaptive")
        x = df[df.unit == 1]["s2"].to_numpy()
        rows = ft.X[(df.unit == 1).to_numpy()]
        c = ft.columns.index
        for t in (1, 2, 7, 29):
            self.assertAlmostEqual(rows[t - 1, c("s2_mean")], x[:t].mean(), places=10)
        self.assertAlmostEqual(rows[6, c("s2_slope")], np.polyfit(np.arange(7), x[:7], 1)[0], places=10)
        self.assertEqual(rows[0, c("s2_slope")], 0.0)  # a single point has no slope

    def test_pad_first_repeats_first_observation(self):
        df = _df()
        W = 8
        ft = _feat(df, window=W, history="pad_first")
        x = df[df.unit == 1]["s2"].to_numpy()
        rows = ft.X[(df.unit == 1).to_numpy()]
        c = ft.columns.index
        t = 3
        padded = np.concatenate([np.full(W - t, x[0]), x[:t]])
        self.assertAlmostEqual(rows[t - 1, c("s2_mean")], padded.mean(), places=10)
        self.assertAlmostEqual(rows[t - 1, c("s2_slope")], np.polyfit(np.arange(W), padded, 1)[0], places=10)

    def test_pad_mean_fills_with_available_mean(self):
        df = _df()
        W = 8
        ft = _feat(df, window=W, history="pad_mean")
        ad = _feat(df, window=W, history="adaptive")
        x = df[df.unit == 1]["s2"].to_numpy()
        rows = ft.X[(df.unit == 1).to_numpy()]
        c = ft.columns.index
        t = 3
        padded = np.concatenate([np.full(W - t, x[:t].mean()), x[:t]])
        self.assertAlmostEqual(rows[t - 1, c("s2_mean")], x[:t].mean(), places=10)  # same mean as adaptive
        self.assertAlmostEqual(rows[t - 1, c("s2_slope")], np.polyfit(np.arange(W), padded, 1)[0], places=10)
        np.testing.assert_allclose(ft.X[(df.cycle >= W).to_numpy()], ad.X[(df.cycle >= W).to_numpy()], atol=1e-12)  # full windows identical

    def test_short_engines_are_kept(self):
        df = _df()  # engine 3 has 12 cycles, shorter than the window
        for history in ("adaptive", "pad_first", "pad_mean"):
            ft = _feat(df, window=30, history=history)
            self.assertEqual(ft.X.shape[0], len(df))
            self.assertEqual(int((ft.index["unit"] == 3).sum()), 12)
            self.assertFalse(np.isnan(ft.X).any())

    def test_engines_are_independent(self):
        df = _df()
        together = _feat(df, window=10, baseline="causal", baseline_cycles=5).X
        alone = _feat(df[df.unit == 2].reset_index(drop=True), window=10, baseline="causal", baseline_cycles=5).X
        np.testing.assert_array_equal(together[(df.unit == 2).to_numpy()], alone)


class CausalityTests(unittest.TestCase):
    """Features at cycle t must not depend on anything after cycle t."""

    CASES = [(h, b) for h in ("adaptive", "pad_first", "pad_mean") for b in ("none", "causal")]

    def test_truncation_invariance(self):
        df = _df()
        for history, baseline in self.CASES:
            with self.subTest(history=history, baseline=baseline):
                kw = dict(window=15, history=history, baseline=baseline, baseline_cycles=10, stats=("last", "mean", "std", "slope"))
                full = _feat(df, **kw)
                for cut in (1, 5, 11, 30):
                    trunc_df = df[df.cycle <= cut].reset_index(drop=True)
                    part = _feat(trunc_df, **kw)
                    mask = (df.cycle <= cut).to_numpy()
                    np.testing.assert_array_equal(full.X[mask], part.X)

    def test_future_perturbation_does_not_change_past_features(self):
        df = _df()
        altered = df.copy()
        altered.loc[altered.cycle > 25, list(SENSORS)] += 1e6  # wreck the future
        for history, baseline in self.CASES:
            with self.subTest(history=history, baseline=baseline):
                kw = dict(window=15, history=history, baseline=baseline, baseline_cycles=10)
                a, b = _feat(df, **kw).X, _feat(altered, **kw).X
                past = (df.cycle <= 25).to_numpy()
                np.testing.assert_array_equal(a[past], b[past])
                self.assertFalse(np.array_equal(a[~past], b[~past]))  # sanity: the perturbation is visible later

    def test_causal_baseline_definition(self):
        x = np.arange(1.0, 31.0)[:, None] ** 2
        B = 10
        base = causal_baseline(x, B)
        for t in (1, 4, 10, 11, 30):
            self.assertAlmostEqual(base[t - 1, 0], x[: min(t, B), 0].mean(), places=10)
        self.assertEqual(base[10, 0], base[29, 0])  # frozen after B cycles

    def test_baselined_window_mean_is_deviation_from_baseline(self):
        df = _df()
        raw = _feat(df, window=10, baseline="none").X
        cb = _feat(df, window=10, baseline="causal", baseline_cycles=5).X
        # slopes are unchanged once the baseline is frozen and the window lies beyond it
        c = _feat(df, window=10).columns.index("s2_slope")
        mask = ((df.unit == 1) & (df.cycle > 15)).to_numpy()
        np.testing.assert_allclose(raw[mask, c], cb[mask, c], atol=1e-9)

    def test_invalid_inputs(self):
        with self.assertRaises(ValueError):
            WindowConfig(window=0)
        with self.assertRaises(ValueError):
            WindowConfig(history="zero_pad")
        with self.assertRaises(ValueError):
            WindowConfig(baseline="global")
        with self.assertRaises(ValueError):
            WindowConfig(stats=("median",))
        bad = _df()
        bad = bad[bad.cycle != 3]  # cycle gap
        with self.assertRaises(ValueError):
            _feat(bad)
        late_start = _df()
        late_start = late_start[~((late_start.unit == 1) & (late_start.cycle == 1))]
        with self.assertRaises(ValueError):
            _feat(late_start)


class SensorSelectorTests(unittest.TestCase):
    def test_config_a_drops_constants_only(self):
        subset = make_subset()
        sel = SensorSelector("A").fit(subset.train)
        const = {s for s in SENSOR_COLUMNS if subset.train[s].nunique() <= 1}
        self.assertTrue(const)
        self.assertEqual(set(sel.selected_sensors_), set(SENSOR_COLUMNS) - const)

    def test_config_b_also_drops_s6(self):
        train = make_subset().train.copy()
        train["s6"] = np.random.default_rng(0).choice([21.60, 21.61], size=len(train))  # two-valued, not constant
        a, b = SensorSelector("A").fit(train), SensorSelector("B").fit(train)
        self.assertIn("s6", a.selected_sensors_)
        self.assertNotIn("s6", b.selected_sensors_)
        self.assertEqual(set(a.selected_sensors_) - {"s6"}, set(b.selected_sensors_))

    def test_constants_are_decided_from_the_fitted_rows_only(self):
        train = make_subset().train.copy()
        train["s5"] = 7.0  # constant in training...
        sel = SensorSelector("A").fit(train)
        later = train.copy()
        later["s5"] = np.arange(len(later), dtype=float)  # ...but varying elsewhere
        self.assertNotIn("s5", sel.selected_sensors_)
        self.assertNotIn("s5", SensorSelector("A").fit(train).selected_sensors_)
        self.assertIn("s5", SensorSelector("A").fit(later).selected_sensors_)

    def test_keep_only_and_errors(self):
        train = make_subset().train
        sel = SensorSelector("A", keep_only=("s2",)).fit(train)
        self.assertEqual(sel.selected_sensors_, ("s2",))
        with self.assertRaises(ValueError):
            SensorSelector("C")
        with self.assertRaises(ValueError):
            SensorSelector("A", keep_only=("s99",))
        with self.assertRaises(ValueError):
            SensorSelector("A", keep_only=("s1",)).fit(train)  # s1 is constant in the synthetic data -> nothing left

    def test_feature_column_indices(self):
        cols = ["s2_mean", "s2_slope", "s3_mean", "s3_slope", "s21_mean"]
        self.assertEqual(feature_column_indices(cols, ("s3", "s21")), [2, 3, 4])


if __name__ == "__main__":
    unittest.main()
