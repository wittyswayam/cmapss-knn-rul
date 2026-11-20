"""Independent verification of the metric implementations against literal, loop-based references."""

import math
import unittest

import numpy as np
import pandas as pd

from cmapss_knn.evaluation import mae, mean_nasa_score, nasa_score, regression_metrics, rmse
from cmapss_knn.evaluation.reference import mae_reference, mean_nasa_reference, nasa_score_reference, rmse_reference
from cmapss_knn.experiments import phase3 as p3


class HandComputedNasa(unittest.TestCase):
    def test_values_worked_out_by_hand(self):
        # d = predicted - true
        cases = {0: 0.0, 10: math.e - 1, -13: math.e - 1, 1: math.exp(0.1) - 1, -1: math.exp(1 / 13) - 1, 20: math.exp(2) - 1, -26: math.exp(2) - 1}
        for d, expected in cases.items():
            with self.subTest(d=d):
                self.assertAlmostEqual(nasa_score([50.0], [50.0 + d]), expected, places=12)
                self.assertAlmostEqual(nasa_score_reference([50.0], [50.0 + d]), expected, places=12)

    def test_error_sign_convention(self):
        true = 100.0
        self.assertGreater(nasa_score([true], [true + 15]), nasa_score([true], [true - 15]))  # overestimating costs more
        self.assertAlmostEqual(nasa_score([true], [true + 15]), math.exp(1.5) - 1, places=12)  # late: denominator 10
        self.assertAlmostEqual(nasa_score([true], [true - 15]), math.exp(15 / 13) - 1, places=12)  # early: denominator 13

    def test_aggregation_is_a_sum_and_the_mean_divides_by_the_number_of_predictions(self):
        t, p = [100, 80, 60, 40], [110, 70, 60, 55]
        per = [nasa_score([a], [b]) for a, b in zip(t, p)]
        self.assertAlmostEqual(nasa_score(t, p), sum(per), places=12)
        self.assertAlmostEqual(mean_nasa_score(t, p), sum(per) / 4, places=12)
        self.assertAlmostEqual(nasa_score(t + t, p + p), 2 * nasa_score(t, p), places=12)  # additive over predictions
        self.assertAlmostEqual(mean_nasa_score(t + t, p + p), mean_nasa_score(t, p), places=12)  # the mean does not grow with n


class VectorisedMatchesReference(unittest.TestCase):
    def test_random_inputs(self):
        rng = np.random.default_rng(0)
        for n in (1, 2, 7, 500):
            t = rng.uniform(0, 125, n)
            p = t + rng.normal(0, 25, n)
            with self.subTest(n=n):
                self.assertTrue(math.isclose(nasa_score(t, p), nasa_score_reference(t.tolist(), p.tolist()), rel_tol=1e-12, abs_tol=1e-12))
                self.assertTrue(math.isclose(mean_nasa_score(t, p), mean_nasa_reference(t.tolist(), p.tolist()), rel_tol=1e-12, abs_tol=1e-12))
                self.assertTrue(math.isclose(rmse(t, p), rmse_reference(t.tolist(), p.tolist()), rel_tol=1e-12))
                self.assertTrue(math.isclose(mae(t, p), mae_reference(t.tolist(), p.tolist()), rel_tol=1e-12))

    def test_extreme_errors_do_not_overflow(self):
        self.assertTrue(math.isfinite(nasa_score([0.0], [125.0])))  # d = 125 -> exp(12.5)
        self.assertTrue(math.isclose(nasa_score([0.0], [125.0]), nasa_score_reference([0.0], [125.0]), rel_tol=1e-12))

    def test_regression_metrics_dictionary_uses_the_same_definitions(self):
        t, p = np.array([100.0, 60.0, 30.0]), np.array([90.0, 70.0, 30.0])
        m = regression_metrics(t, p)
        self.assertTrue(math.isclose(m["nasa_mean"], mean_nasa_reference(t.tolist(), p.tolist()), rel_tol=1e-12))
        self.assertAlmostEqual(m["bias"], 0.0)  # (-10 + 10 + 0) / 3
        self.assertAlmostEqual(m["late_share"], 1 / 3)


class SweepScoringUsesTheSameMetric(unittest.TestCase):
    def test_score3_reproduces_reference_values(self):
        meta = pd.DataFrame({"rul": [90, 50, 120, 10], "rul_clipped": [90, 50, 120, 10], "is_cutoff": [True, True, True, True],
                             "is_win": [True, True, False, False], "is_stress": [False, True, True, False]})
        pred = np.array([100.0, 40.0, 120.0, 25.0])
        s = p3.score3(meta, pred)
        self.assertTrue(math.isclose(s["eng_nasa_clip"], mean_nasa_reference(meta.rul_clipped.tolist(), pred.tolist()), rel_tol=1e-12))
        self.assertTrue(math.isclose(s["eng_rmse_clip"], rmse_reference(meta.rul_clipped.tolist(), pred.tolist()), rel_tol=1e-12))

    def test_cv_summary_is_the_unweighted_mean_of_fold_means(self):
        rows = []
        for f, (nasa, n) in enumerate(((2.0, 10), (4.0, 30))):
            rows.append({"subset": "X", "fold": f, "normalization": "global", "baseline": "none", "window": 10, "history": "adaptive", "metric": "euclidean",
                         "weights": "uniform", "k": 5, "n_features": 3, "eng_nasa_clip": nasa, "eng_rmse_clip": 1.0, "n_eng": n, "n_win": 5, "n_stress": 5})
        s = p3.summarise3(pd.DataFrame(rows))
        self.assertAlmostEqual(s["eng_nasa_clip_mean"].iloc[0], 3.0)  # not the point-weighted 3.5


if __name__ == "__main__":
    unittest.main()


class PerEngineTerms(unittest.TestCase):
    def test_nasa_terms_match_the_reference_one_by_one_and_sum_to_the_score(self):
        from cmapss_knn.evaluation import metrics as mt

        rng = np.random.default_rng(3)
        t = rng.uniform(0, 150, 50)
        p = t + rng.normal(0, 20, 50)
        terms = mt.nasa_terms(t, p)
        for a, b, term in zip(t, p, terms):
            self.assertTrue(math.isclose(term, nasa_score_reference([a], [b]), rel_tol=1e-12, abs_tol=1e-12))
        self.assertTrue(math.isclose(terms.sum(), nasa_score(t, p), rel_tol=1e-12))
