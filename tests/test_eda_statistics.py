import unittest

import numpy as np
import pandas as pd

from cmapss_knn.eda import statistics as st


def _frame(n_units=5, n_cycles=60, seed=0):
    rng = np.random.default_rng(seed)
    rows = []
    for u in range(1, n_units + 1):
        c = np.arange(1, n_cycles + 1)
        rows.append(
            pd.DataFrame(
                {
                    "unit": u,
                    "cycle": c,
                    "up": c * 0.5 + rng.normal(0, 0.01, n_cycles),
                    "down": -c * 0.5 + rng.normal(0, 0.01, n_cycles),
                    "flat": 1.0,
                    "noise": rng.normal(0, 1, n_cycles),
                }
            )
        )
    return pd.concat(rows, ignore_index=True)


class StatisticsTests(unittest.TestCase):
    def test_trend_signs_and_undefined_engines(self):
        df = _frame()
        trend = st.within_engine_trend(df, ["up", "down", "flat"])
        self.assertGreater(trend.loc["up", "mean_rho"], 0.99)
        self.assertLess(trend.loc["down", "mean_rho"], -0.99)
        self.assertEqual(trend.loc["up", "sign_consistency"], 1.0)
        self.assertEqual(trend.loc["up", "n_engines"], 5)
        self.assertEqual(trend.loc["flat", "n_engines"], 0)  # constant everywhere -> undefined
        self.assertTrue(np.isnan(trend.loc["flat", "mean_rho"]))

    def test_sensor_summary_flags_constants(self):
        summary = st.sensor_summary(_frame(), ["up", "flat"])
        self.assertTrue(summary.loc["flat", "is_constant"])
        self.assertFalse(summary.loc["up", "is_constant"])

    def test_spike_detection_finds_injected_outlier(self):
        df = _frame(n_units=3, n_cycles=100)
        df.loc[50, "noise"] += 100.0
        out = st.spike_fraction(df, ["noise"], window=11, z_threshold=5.0)
        self.assertGreaterEqual(out.loc["noise", "spike_count"], 1)
        self.assertFalse(out.loc["noise", "mad_zero"])

    def test_spike_detection_ignores_noisy_trend(self):
        rng = np.random.default_rng(3)
        c = np.arange(1, 201)
        df = pd.DataFrame({"unit": 1, "cycle": c, "ramp": c * 0.5 + rng.normal(0, 1.0, c.size)})
        out = st.spike_fraction(df, ["ramp"], window=11, z_threshold=5.0)
        self.assertEqual(out.loc["ramp", "spike_count"], 0)
        self.assertFalse(out.loc["ramp", "mad_zero"])

    def test_noiseless_monotone_signal_flags_zero_mad(self):
        # A rolling median reproduces a (nearly) noiseless monotone ramp exactly,
        # so residuals are 0 and z-scores are undefined; this must be flagged, not divided by zero.
        out = st.spike_fraction(_frame(n_units=3, n_cycles=100), ["up"], window=11, z_threshold=5.0)
        self.assertTrue(out.loc["up", "mad_zero"])
        self.assertTrue(np.isnan(out.loc["up", "spike_fraction"]))

    def test_initial_spread_detects_offsets(self):
        rng = np.random.default_rng(1)
        rows = []
        for u in range(1, 21):
            offset = rng.normal(0, 5)
            rows.append(pd.DataFrame({"unit": u, "cycle": np.arange(1, 31), "offset": offset + rng.normal(0, 0.1, 30), "pure_noise": rng.normal(0, 1, 30)}))
        spread = st.initial_unit_spread(pd.concat(rows), ["offset", "pure_noise"], n_cycles=20)
        self.assertGreater(spread.loc["offset", "offset_ratio"], 10)
        self.assertLess(spread.loc["pure_noise", "offset_ratio"], 3)

    def test_share_outside_range(self):
        train = pd.DataFrame({"a": [0.0, 1.0], "b": [0.0, 1.0]})
        test = pd.DataFrame({"a": [0.5, 2.0], "b": [0.5, 0.6]})
        share = st.share_outside_train_range(train, test, ["a", "b"])
        self.assertEqual(share["a"], 0.5)
        self.assertEqual(share["b"], 0.0)

    def test_high_correlation_pairs(self):
        corr = pd.DataFrame([[1, 0.95, 0.1], [0.95, 1, -0.2], [0.1, -0.2, 1]], columns=list("abc"), index=list("abc"))
        self.assertEqual(st.high_correlation_pairs(corr, 0.9), [("a", "b", 0.95)])


if __name__ == "__main__":
    unittest.main()
