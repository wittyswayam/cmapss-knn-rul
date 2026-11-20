import unittest

import numpy as np
import pandas as pd

from cmapss_knn.data.targets import add_train_rul, clip_rul, truncated_engine_table

from .helpers import make_subset


class TargetTests(unittest.TestCase):
    def test_train_rul_is_zero_at_last_cycle_and_counts_down(self):
        subset = make_subset()
        out = add_train_rul(subset.train)
        for _, g in out.groupby("unit"):
            self.assertEqual(g["rul"].iloc[-1], 0)
            self.assertEqual(g["rul"].iloc[0], len(g) - 1)
            self.assertTrue((g["rul"].diff().dropna() == -1).all())
        self.assertNotIn("rul", subset.train.columns)  # input is not mutated

    def test_clipping(self):
        out = add_train_rul(make_subset().train, cap=30)
        self.assertEqual(out["rul_clipped"].max(), 30)
        self.assertTrue((out["rul_clipped"] <= out["rul"]).all())
        np.testing.assert_array_equal(clip_rul(np.array([0, 10, 200]), 125), [0, 10, 125])
        with self.assertRaises(ValueError):
            clip_rul(pd.Series([1, 2]), 0)

    def test_truncated_engine_table(self):
        subset = make_subset()
        table = truncated_engine_table(subset.test, subset.rul)
        self.assertEqual(len(table), subset.test["unit"].nunique())
        expected_last = subset.test.groupby("unit")["cycle"].max()
        pd.testing.assert_series_equal(table["observed_cycles"], expected_last.rename("observed_cycles"))
        np.testing.assert_array_equal(table["implied_lifetime"], table["observed_cycles"] + subset.rul.to_numpy())
        self.assertTrue(((table["fraction_observed"] > 0) & (table["fraction_observed"] < 1)).all())


if __name__ == "__main__":
    unittest.main()
