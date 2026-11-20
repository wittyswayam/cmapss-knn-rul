import math
import unittest

import numpy as np
import pandas as pd

from cmapss_knn.evaluation import engine_folds, holdout_engines, mae, mean_nasa_score, nasa_score, rmse, sample_cutoffs


class MetricTests(unittest.TestCase):
    def test_rmse_mae(self):
        self.assertAlmostEqual(rmse([0, 0], [3, 4]), math.sqrt(12.5))
        self.assertAlmostEqual(mae([0, 0], [3, -4]), 3.5)

    def test_nasa_score_formula_and_sign_convention(self):
        # d = predicted - true. d < 0 (early): exp(-d/13) - 1 ; d >= 0 (late): exp(d/10) - 1
        self.assertAlmostEqual(nasa_score([50], [50]), 0.0)
        self.assertAlmostEqual(nasa_score([50], [37]), math.e - 1)  # d = -13
        self.assertAlmostEqual(nasa_score([50], [60]), math.e - 1)  # d = +10
        self.assertAlmostEqual(nasa_score([50, 50], [37, 60]), 2 * (math.e - 1))  # sum over engines
        self.assertAlmostEqual(mean_nasa_score([50, 50], [37, 60]), math.e - 1)

    def test_overestimating_rul_is_penalised_more(self):
        for d in (5, 10, 20, 40):
            self.assertGreater(nasa_score([100], [100 + d]), nasa_score([100], [100 - d]))

    def test_invalid_inputs(self):
        for args in (([1, 2], [1]), ([], []), ([[1]], [[1]])):
            with self.assertRaises(ValueError):
                rmse(*args)


class SplitTests(unittest.TestCase):
    def test_folds_partition_engines(self):
        units = np.arange(1, 81)
        folds = engine_folds(units, 5, seed=1)
        val = np.concatenate([v for _, v in folds])
        self.assertEqual(sorted(val), list(units))  # each engine validated exactly once
        for tr, va in folds:
            self.assertEqual(len(set(tr) & set(va)), 0)
            self.assertEqual(len(tr) + len(va), 80)
            self.assertEqual(len(va), 16)

    def test_reproducible_and_seed_dependent(self):
        units = np.arange(1, 51)
        a, b, c = engine_folds(units, 5, 7), engine_folds(units, 5, 7), engine_folds(units, 5, 8)
        for (t1, v1), (t2, v2) in zip(a, b):
            np.testing.assert_array_equal(v1, v2)
        self.assertFalse(all(np.array_equal(x[1], y[1]) for x, y in zip(a, c)))

    def test_holdout_disjoint(self):
        tuning, hold = holdout_engines(np.arange(1, 101), 20, seed=3)
        self.assertEqual((len(tuning), len(hold)), (80, 20))
        self.assertEqual(len(set(tuning) & set(hold)), 0)

    def test_split_errors(self):
        with self.assertRaises(ValueError):
            engine_folds(np.arange(3), 5, 0)
        with self.assertRaises(ValueError):
            holdout_engines(np.arange(10), 10, 0)

    def test_cutoffs(self):
        lifetimes = pd.Series({1: 150, 2: 200, 3: 40}, name="lifetime")
        cut = sample_cutoffs(lifetimes, n_per_engine=5, min_cutoff=30, seed=0)
        for unit, g in cut.groupby("unit"):
            life = lifetimes[unit]
            self.assertTrue(((g.cutoff >= min(30, life - 1)) & (g.cutoff <= life - 1)).all())
            self.assertEqual(g.cutoff.nunique(), len(g))  # distinct
        self.assertEqual((cut.unit == 1).sum(), 5)
        pd.testing.assert_frame_equal(cut, sample_cutoffs(lifetimes, 5, 30, 0))


if __name__ == "__main__":
    unittest.main()
