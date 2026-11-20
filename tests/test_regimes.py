import unittest

import numpy as np
import pandas as pd

from cmapss_knn.data.schema import ALL_COLUMNS, SENSOR_COLUMNS
from cmapss_knn.exceptions import ModelNotFittedError
from cmapss_knn.features.regimes import RegimeDetector, RegimeNormalizer

REGIMES = np.array([[0.0, 0.0, 100.0], [20.0, 0.7, 100.0], [42.0, 0.84, 60.0]])
OFFSETS = np.array([0.0, 50.0, -30.0])  # regime-dependent sensor shift


def _multi_regime(n_units=6, T=80, seed=0, n_regimes=3):
    rng = np.random.default_rng(seed)
    rows = []
    for u in range(1, n_units + 1):
        reg = rng.integers(0, n_regimes, T)
        data = {"unit": u, "cycle": np.arange(1, T + 1)}
        for j, c in enumerate(("op1", "op2", "op3")):
            data[c] = REGIMES[reg, j] + rng.normal(0, 0.001, T)
        for i, s in enumerate(SENSOR_COLUMNS):
            data[s] = 100 + OFFSETS[reg] * (1 + i % 3) + np.arange(T) * 0.05 * (i % 4) + rng.normal(0, 1, T)
        data["s5"] = 7.0 + OFFSETS[reg]  # constant *within* each regime
        rows.append(pd.DataFrame(data)[list(ALL_COLUMNS)])
    return pd.concat(rows, ignore_index=True), None


def _single_regime(n=600, seed=1):
    rng = np.random.default_rng(seed)
    df, _ = _multi_regime(n_units=1, T=n, seed=seed)
    df["op1"], df["op2"], df["op3"] = rng.normal(0, 0.002, n), rng.normal(0, 0.0003, n), 100.0
    return df


class RegimeDetectorTests(unittest.TestCase):
    def test_finds_the_true_number_of_regimes(self):
        df, _ = _multi_regime()
        det = RegimeDetector(seed=0).fit(df)
        self.assertEqual(det.n_regimes_, 3)
        labels = det.predict(df)
        for r in range(3):  # each true setting maps to exactly one detected regime
            true = (np.abs(df["op1"] - REGIMES[r, 0]) < 1).to_numpy()
            self.assertEqual(len(np.unique(labels[true])), 1)

    def test_single_condition_noise_is_one_regime(self):
        det = RegimeDetector(seed=0).fit(_single_regime())
        self.assertEqual(det.n_regimes_, 1)
        np.testing.assert_array_equal(det.predict(_single_regime()), 0)

    def test_regime_depends_only_on_the_rows_own_settings(self):
        df, _ = _multi_regime()
        det = RegimeDetector(seed=0).fit(df)
        full = det.predict(df)
        for i in (0, 17, 300):
            self.assertEqual(det.predict(df.iloc[[i]])[0], full[i])

    def test_not_fitted(self):
        with self.assertRaises(ModelNotFittedError):
            RegimeDetector().predict(_multi_regime()[0])


class RegimeNormalizerTests(unittest.TestCase):
    def setUp(self):
        self.df, _ = _multi_regime()
        self.norm = RegimeNormalizer(seed=0).fit(self.df)

    def test_removes_regime_offsets(self):
        out = self.norm.transform(self.df)
        reg = self.norm.detector_.predict(self.df)
        for s in ("s2", "s7"):
            means = [out.loc[reg == r, s].mean() for r in range(3)]
            np.testing.assert_allclose(means, 0.0, atol=1e-9)
            stds = [out.loc[reg == r, s].std(ddof=0) for r in range(3)]
            np.testing.assert_allclose(stds, 1.0, atol=1e-9)
        self.assertGreater(self.df["s2"].std(), 20 * out["s2"].std() / 3)  # raw spread was dominated by the regime shift

    def test_constant_within_regime_becomes_exactly_zero(self):
        out = self.norm.transform(self.df)
        self.assertTrue((out["s5"] == 0.0).all())
        self.assertTrue(np.isfinite(out[list(SENSOR_COLUMNS)].to_numpy()).all())

    def test_fitted_on_training_rows_only(self):
        train, val = self.df[self.df.unit <= 4], self.df[self.df.unit > 4].copy()
        a = RegimeNormalizer(seed=0).fit(train)
        val_shifted = val.copy()
        val_shifted[list(SENSOR_COLUMNS)] += 1e4  # wild validation data must not influence the fit
        b = RegimeNormalizer(seed=0).fit(train)
        np.testing.assert_array_equal(a.mean_, b.mean_)
        np.testing.assert_array_equal(a.std_, b.std_)
        expected = train.loc[a.detector_.predict(train) == 0, "s2"].mean()
        self.assertAlmostEqual(a.mean_[0, SENSOR_COLUMNS.index("s2")], expected, places=9)
        self.assertGreater(a.transform(val_shifted)["s2"].mean(), 1e3)  # transform just applies stored numbers

    def test_row_wise_and_non_mutating(self):
        before = self.df.copy()
        full = self.norm.transform(self.df)
        pd.testing.assert_frame_equal(self.df, before)  # input untouched
        pd.testing.assert_frame_equal(self.norm.transform(self.df.iloc[[5]]), full.iloc[[5]])  # no dependence on other rows
        self.assertEqual(list(full.columns), list(self.df.columns))
        pd.testing.assert_series_equal(full["cycle"], self.df["cycle"])

    def test_single_regime_is_plain_standardisation(self):
        df = _single_regime()
        norm = RegimeNormalizer(seed=0).fit(df)
        self.assertEqual(norm.n_regimes_, 1)
        out = norm.transform(df)
        np.testing.assert_allclose(out["s2"].mean(), 0.0, atol=1e-9)
        np.testing.assert_allclose(out["s2"].std(ddof=0), 1.0, atol=1e-9)

    def test_rare_regime_falls_back_to_global_statistics(self):
        df = self.df.copy()
        rare = df.index[:5]
        df.loc[rare, ["op1", "op2", "op3"]] = [100.0, 3.0, 20.0]  # an isolated operating point with 5 rows
        norm = RegimeNormalizer(seed=0, max_regimes=4, min_silhouette=0.5, min_rows_per_regime=30).fit(df)
        self.assertTrue(np.isfinite(norm.std_).all() and np.isfinite(norm.mean_).all())

    def test_regime_table_and_not_fitted(self):
        table = self.norm.regime_table(self.df)
        self.assertEqual(len(table), 3)
        self.assertEqual(int(table["rows"].sum()), len(self.df))
        with self.assertRaises(ModelNotFittedError):
            RegimeNormalizer().transform(self.df)


if __name__ == "__main__":
    unittest.main()
